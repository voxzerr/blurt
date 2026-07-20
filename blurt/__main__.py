"""Command line entry point for blurt: ``python3 -m blurt`` (or ``blurt``).

Subcommands::

    blurt            run the dictation daemon (the default)
    blurt doctor     diagnose this machine -- run this first when something is wrong
    blurt bench      measure real transcription latency HERE, not in a datasheet
    blurt config     print the resolved configuration and where it came from

``doctor`` is the important one. blurt's failure modes on macOS are almost all
permission or environment problems that produce silence rather than errors: a
hotkey that never fires, a microphone that returns zeros, a paste that macOS
discards. None of those raise an exception, so a confused user has nothing to
read. ``doctor`` exists to turn all of that into text.

``bench`` reports measured numbers from this specific machine. The spread is
enormous -- tiny.en takes ~2s on a 2017 Intel i7 and a fraction of that on an
M-series -- so quoting anyone else's figures would be dishonest.

What can go wrong on macOS:
  - ``doctor`` briefly opens the microphone (about half a second) to check
    whether permission is actually granted, because a denied app receives silence
    rather than an error. This lights the orange mic indicator; that is expected.
  - Probing the apple-speech engine can trigger the system speech-permission
    dialog when blurt runs from a bundled .app.
  - Nothing here needs the network except the one-time model download that
    ``run`` and ``bench`` trigger on first use.

Python 3.9 floor: lazy annotations, typing generics only, no PEP 604 unions.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import sys
import time
from typing import Any, List, Optional, Sequence, Tuple

from . import __version__
from . import engines as _engines
from . import hardware as _hardware
from .config import (
    VALID_CLEANUP_LEVELS,
    VALID_ENGINES,
    Config,
    default_config_path,
    load_config,
)
from .hotkey import SUPPORTED_HOTKEYS, UnsupportedHotkeyError, normalize_key_name

__all__ = ["main"]

# Third-party modules blurt needs, with what each one is for. Printed by doctor
# in this order; the first three are load-bearing, the rest are per-feature.
_DEPENDENCIES: Tuple[Tuple[str, str], ...] = (
    ("numpy", "audio buffers"),
    ("sounddevice", "microphone capture"),
    ("faster_whisper", "speech recognition (primary engine)"),
    ("pynput", "global hotkey"),
    ("AppKit", "clipboard (pyobjc-framework-Cocoa)"),
    ("Quartz", "synthetic paste (pyobjc-framework-Quartz)"),
    ("rumps", "menu bar item (optional)"),
)

# Modules whose absence does not stop blurt from dictating.
_OPTIONAL_DEPENDENCIES = frozenset({"rumps"})

_BENCH_DEFAULT_SECONDS = 5.0
_BENCH_COUNTDOWN = 3


def _out(message: str = "") -> None:
    try:
        print(message, flush=True)
    except Exception:  # pragma: no cover - stdout closed
        pass


def _err(message: str = "") -> None:
    try:
        print(message, file=sys.stderr, flush=True)
    except Exception:  # pragma: no cover - stderr closed
        pass


# -- argument parsing -------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the parser.

    The override flags are attached to the top-level parser AND to every
    subparser, so ``blurt --model base.en doctor`` and ``blurt doctor --model
    base.en`` both work. They default to ``argparse.SUPPRESS`` specifically so a
    subparser that did not see the flag leaves the top-level value alone --
    without that, subparser defaults overwrite it in the shared namespace.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--engine",
        default=argparse.SUPPRESS,
        metavar="NAME",
        help="override the engine for this run (%s)" % ", ".join(sorted(VALID_ENGINES)),
    )
    common.add_argument(
        "--model",
        default=argparse.SUPPRESS,
        metavar="SIZE",
        help="override the model for this run (auto, tiny.en, base.en, small.en, ...)",
    )
    common.add_argument(
        "--cleanup",
        default=argparse.SUPPRESS,
        metavar="LEVEL",
        help="override the cleanup level (%s)" % ", ".join(sorted(VALID_CLEANUP_LEVELS)),
    )
    common.add_argument(
        "--hotkey",
        default=argparse.SUPPRESS,
        metavar="KEY",
        help="override the push-to-talk key (%s)" % ", ".join(SUPPORTED_HOTKEYS),
    )

    parser = argparse.ArgumentParser(
        prog="blurt",
        parents=[common],
        description="Hold a key, talk, and have your words typed where the cursor is.",
        epilog="Run 'blurt doctor' first if anything is not working.",
    )
    parser.add_argument(
        "--version", action="version", version="blurt " + __version__
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    sub.add_parser(
        "run",
        parents=[common],
        help="run the dictation daemon (default)",
        description="Load the model and listen for the push-to-talk key.",
    )

    sub.add_parser(
        "doctor",
        parents=[common],
        help="diagnose hardware, engines, permissions and dependencies",
        description=(
            "Print everything blurt knows about this machine. Briefly opens the "
            "microphone to check whether permission is really granted."
        ),
    )

    bench = sub.add_parser(
        "bench",
        parents=[common],
        help="measure real transcription latency on this machine",
        description=(
            "Record (or synthesize) a sample and time transcription of it. "
            "Reports numbers measured here, on your hardware."
        ),
    )
    bench.add_argument(
        "--seconds",
        type=float,
        default=_BENCH_DEFAULT_SECONDS,
        metavar="N",
        help="length of the sample (default: %(default)s)",
    )
    bench.add_argument(
        "--repeat",
        type=int,
        default=3,
        metavar="N",
        help="transcription passes to time (default: %(default)s)",
    )
    bench.add_argument(
        "--synth",
        action="store_true",
        help="skip the microphone and use synthetic audio (measures compute only)",
    )

    sub.add_parser(
        "config",
        parents=[common],
        help="print the resolved configuration and its file path",
        description="Show the effective settings and where they were read from.",
    )

    return parser


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Fold the command-line overrides into a config, validating each one.

    Overrides are for one run only; nothing is written back to disk. An invalid
    value exits with status 2 rather than being quietly ignored -- a typo'd
    ``--model tinyen`` that silently ran something else would be worse than useless.
    """
    engine = getattr(args, "engine", None)
    if engine is not None:
        engine = engine.strip().lower()
        if engine not in VALID_ENGINES:
            _err(
                "blurt: --engine %r is not valid. Choose one of: %s"
                % (engine, ", ".join(sorted(VALID_ENGINES)))
            )
            raise SystemExit(2)
        cfg.engine = engine

    model = getattr(args, "model", None)
    if model is not None:
        model = model.strip()
        if not model:
            _err("blurt: --model must not be empty")
            raise SystemExit(2)
        # Model names are not validated against a list: faster-whisper accepts
        # local paths and Hugging Face ids as well as the well-known sizes, and
        # rejecting an unfamiliar string here would block a legitimate use.
        cfg.model = model

    cleanup = getattr(args, "cleanup", None)
    if cleanup is not None:
        cleanup = cleanup.strip().lower()
        if cleanup not in VALID_CLEANUP_LEVELS:
            _err(
                "blurt: --cleanup %r is not valid. Choose one of: %s"
                % (cleanup, ", ".join(sorted(VALID_CLEANUP_LEVELS)))
            )
            raise SystemExit(2)
        cfg.cleanup_level = cleanup

    hotkey = getattr(args, "hotkey", None)
    if hotkey is not None:
        try:
            cfg.hotkey = normalize_key_name(hotkey)
        except UnsupportedHotkeyError as exc:
            _err("blurt: --hotkey %r is not usable.\n  %s" % (hotkey, exc))
            _err("  Supported: %s" % ", ".join(SUPPORTED_HOTKEYS))
            raise SystemExit(2)

    return cfg


def _overrides_in_effect(args: argparse.Namespace) -> List[str]:
    """Names of the flags the user actually passed, for display."""
    return [
        name
        for name in ("engine", "model", "cleanup", "hotkey")
        if getattr(args, name, None) is not None
    ]


# -- shared helpers ---------------------------------------------------------


def _import_probe(module_name: str) -> Tuple[bool, str]:
    """Import a module and describe the result. Never raises.

    Catches ``BaseException`` on purpose: a wheel built for the wrong macOS fails
    in the dynamic loader as ``OSError``, not ``ImportError``. That is exactly how
    pywhispercpp failed on the floor machine, and an ``except ImportError`` would
    have let it take the whole command down.
    """
    try:
        module = __import__(module_name)
    except BaseException as exc:  # noqa: BLE001 - loader failures are not ImportError
        return False, "%s: %s" % (type(exc).__name__, exc)

    version = getattr(module, "__version__", None)
    if not version:
        try:
            from importlib import metadata  # Python 3.8+

            version = metadata.version(module_name)
        except Exception:  # noqa: BLE001 - distribution name may differ
            version = "version unknown"
    return True, str(version)


def _resolved_model(cfg: Config, hw: Any) -> Tuple[str, str]:
    """(model, why) for the model that would actually be loaded."""
    configured = (cfg.model or "auto").strip()
    if configured and configured != "auto":
        return configured, "set in config or overridden on the command line"
    return (
        _hardware.recommend_model(hw),
        "chosen automatically for a '%s' machine" % hw.tier,
    )


def _resolved_threads(cfg: Config, hw: Any) -> Tuple[int, str]:
    if isinstance(cfg.cpu_threads, int) and cfg.cpu_threads > 0:
        return cfg.cpu_threads, "set in config"
    return _hardware.recommend_threads(hw), "chosen automatically"


def _yes_no_unknown(value: Optional[bool]) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "NO"
    return "could not determine"


# -- doctor -----------------------------------------------------------------


def _cmd_doctor(cfg: Config, args: argparse.Namespace) -> int:
    """Print a full diagnostic. Always exits 0 unless something is truly broken."""
    _out("blurt %s -- diagnostics" % __version__)
    _out("=" * 60)

    problems: List[str] = []

    _doctor_python(problems)
    hw = _doctor_hardware()
    _doctor_dependencies(problems)
    _doctor_engines(cfg, hw, problems)
    _doctor_model(cfg, hw)
    _doctor_permissions(problems)
    _doctor_hotkey(cfg, problems)
    _doctor_config(cfg, args)

    _out("")
    _out("=" * 60)
    if problems:
        _out("Problems found (%d):" % len(problems))
        for item in problems:
            _out("  - " + item)
        _out("")
        _out("Fix these in order; the first one usually explains the rest.")
        return 1

    _out("No problems detected. blurt should work on this machine.")
    return 0


def _doctor_python(problems: List[str]) -> None:
    _out("")
    _out("PYTHON")
    version = sys.version_info
    _out("  version    : %d.%d.%d" % (version[0], version[1], version[2]))
    _out("  executable : %s" % sys.executable)
    _out("  platform   : %s %s" % (platform.system(), platform.release()))
    if version < (3, 9):
        problems.append(
            "Python %d.%d is too old; blurt needs 3.9 or newer."
            % (version[0], version[1])
        )
    if platform.system() != "Darwin":
        problems.append(
            "This is not macOS. blurt's hotkey, clipboard and paste paths are "
            "macOS-only and will not work here."
        )


def _doctor_hardware() -> Any:
    _out("")
    _out("HARDWARE")
    hw = _hardware.detect()
    _out("  cpu        : %s" % hw.cpu_brand)
    _out("  arch       : %s%s" % (hw.arch, " (running under Rosetta)" if hw.under_rosetta else ""))
    _out("  cores      : %d physical" % hw.physical_cores)
    _out("  memory     : %.1f GB" % hw.ram_gb)
    _out(
        "  macOS      : %d.%d.%d"
        % (hw.macos_version[0], hw.macos_version[1], hw.macos_version[2])
    )
    _out("  tier       : %s" % hw.tier)
    if hw.under_rosetta:
        _out("")
        _out("  Note: this Python is running under Rosetta on Apple Silicon.")
        _out("  A native arm64 Python would be considerably faster.")
    return hw


def _doctor_dependencies(problems: List[str]) -> None:
    _out("")
    _out("DEPENDENCIES")
    for module_name, purpose in _DEPENDENCIES:
        ok, detail = _import_probe(module_name)
        optional = module_name in _OPTIONAL_DEPENDENCIES
        if ok:
            _out("  [ok]   %-16s %-14s %s" % (module_name, detail, purpose))
            continue
        label = "warn" if optional else "FAIL"
        _out("  [%s] %-16s %s" % (label, module_name, purpose))
        _out("         %s" % detail)
        if not optional:
            problems.append(
                "%s does not import (%s). Install it: python3 -m pip install %s"
                % (module_name, purpose, _pip_name(module_name))
            )


def _pip_name(module_name: str) -> str:
    """Map an import name to the thing you actually pip install."""
    mapping = {
        "faster_whisper": "faster-whisper",
        "AppKit": "pyobjc-framework-Cocoa",
        "Quartz": "pyobjc-framework-Quartz",
    }
    return mapping.get(module_name, module_name)


def _doctor_engines(cfg: Config, hw: Any, problems: List[str]) -> None:
    """Report each engine's availability and, when unavailable, the reason.

    Uses the registry's own probe so the answers here match what ``run`` will do.
    Falls back to the public listing if that internal helper ever moves.
    """
    _out("")
    _out("ENGINES")

    probe = getattr(_engines, "_probe", None)
    names = getattr(_engines, "ENGINE_NAMES", ("faster-whisper", "apple-speech"))

    usable: List[str] = []
    if callable(probe):
        for name in names:
            try:
                engine, reason = probe(name, cfg, hw)
            except Exception as exc:  # noqa: BLE001 - diagnostics must not crash
                engine, reason = None, "probe raised: %s: %s" % (type(exc).__name__, exc)
            if engine is not None:
                usable.append(name)
                _out("  [ok]   %s" % name)
            else:
                _out("  [FAIL] %s" % name)
                _out("         %s" % (reason or "no reason given"))
    else:  # pragma: no cover - only if the registry internals change
        try:
            usable = list(_engines.available_engines())
        except Exception:  # noqa: BLE001
            usable = []
        for name in names:
            _out("  [%s] %s" % ("ok  " if name in usable else "FAIL", name))

    _out("")
    configured = (cfg.engine or "auto").strip().lower()
    if not usable:
        _out("  No engine can run here.")
        problems.append(
            "No speech engine is usable. Install the primary one: "
            "python3 -m pip install faster-whisper"
        )
    elif configured == "auto":
        _out("  engine=auto would select: %s" % usable[0])
    elif configured in usable:
        _out("  engine=%s (explicitly configured) is available." % configured)
    else:
        _out("  engine=%s is configured but NOT available." % configured)
        _out("  blurt will not substitute a different engine for an explicit choice.")
        problems.append(
            "Configured engine %r is unavailable. Either install it or change "
            '"engine" in your config (available: %s).' % (configured, ", ".join(usable))
        )


def _doctor_model(cfg: Config, hw: Any) -> None:
    _out("")
    _out("MODEL")
    model, why = _resolved_model(cfg, hw)
    threads, thread_why = _resolved_threads(cfg, hw)
    _out("  model      : %s (%s)" % (model, why))
    _out("  threads    : %d (%s)" % (threads, thread_why))
    _out("  compute    : int8 on the CPU")
    _out("")
    _out("  Two things worth knowing about these numbers:")
    _out("    - Whisper pads every input to a 30-second window, so a 2-second")
    _out("      phrase costs about the same as a 15-second one. Speaking briefly")
    _out("      does not make it faster.")
    _out("    - The thread count is physical cores, not logical. Oversubscribing")
    _out("      hyperthreads measurably hurts tail latency on Intel.")
    _out("  Run 'blurt bench' for real measured numbers from THIS machine --")
    _out("  the spread across supported hardware is far too wide to quote here.")


def _doctor_permissions(problems: List[str]) -> None:
    _out("")
    _out("PERMISSIONS")

    accessibility: Optional[bool] = None
    secure: Optional[bool] = None
    try:
        from .inject import accessibility_trusted, secure_input_active

        accessibility = accessibility_trusted()
        secure = secure_input_active()
    except Exception as exc:  # noqa: BLE001 - pyobjc missing or broken
        _out("  could not check: %s: %s" % (type(exc).__name__, exc))

    _out("  accessibility (paste + hotkey) : %s" % _yes_no_unknown(accessibility))
    if accessibility is False:
        problems.append(
            "Accessibility permission is missing. The hotkey will never fire and "
            "pasting will be silently discarded. Grant it in System Settings > "
            "Privacy & Security > Accessibility to the app that launches blurt "
            "(your terminal, not blurt), then relaunch that app."
        )

    _out("  secure input active right now  : %s" % _yes_no_unknown(secure))
    if secure is True:
        _out("    (Some app has Secure Event Input on -- a password field has focus,")
        _out("     or a terminal has Secure Keyboard Entry enabled. While that is")
        _out("     true, macOS blocks synthetic paste system-wide and blurt falls")
        _out("     back to leaving text on the clipboard.)")

    _doctor_microphone(problems)


def _doctor_microphone(problems: List[str]) -> None:
    """Actually open the mic for half a second. Nothing else answers this question.

    macOS gives a denied app a working stream full of zeros instead of an error,
    so the only real test is to record and look at the level.
    """
    _out("  microphone                     : testing (0.5s)...")
    try:
        from .audio import AudioUnavailable, Recorder
    except Exception as exc:  # noqa: BLE001
        _out("    could not load the audio module: %s: %s" % (type(exc).__name__, exc))
        problems.append("The audio module does not import; blurt cannot record.")
        return

    recorder = None
    try:
        recorder = Recorder(sample_rate=16000, preroll_ms=0)
        _out("    input device rate: %d Hz" % recorder.device_sample_rate)
        recorder.start()
        time.sleep(0.5)
        recorder.stop()
        level = recorder.last_capture_rms()
        if recorder.last_capture_was_silent():
            _out("    level: silent (rms %.6f)" % level)
            _out("    Either microphone permission is denied, the wrong input")
            _out("    device is selected, the mic is muted, or the room is silent.")
            problems.append(
                "The microphone produced no signal. Check System Settings > "
                "Privacy & Security > Microphone (enable the app that launches "
                "blurt), and System Settings > Sound > Input. If the room was "
                "genuinely quiet, re-run doctor while speaking."
            )
        else:
            _out("    level: signal present (rms %.4f) -- microphone works" % level)
    except AudioUnavailable as exc:
        _out("    unavailable: %s" % exc)
        problems.append("No usable microphone: %s" % str(exc).splitlines()[0])
    except Exception as exc:  # noqa: BLE001 - PortAudio raises broadly
        _out("    failed: %s: %s" % (type(exc).__name__, exc))
        problems.append("Microphone test failed: %s: %s" % (type(exc).__name__, exc))
    finally:
        if recorder is not None:
            try:
                recorder.close()
            except Exception:  # noqa: BLE001 - teardown is best effort
                pass


def _doctor_hotkey(cfg: Config, problems: List[str]) -> None:
    _out("")
    _out("HOTKEY")
    try:
        canonical = normalize_key_name(cfg.hotkey)
        _out("  configured : %s" % cfg.hotkey)
        if canonical != cfg.hotkey:
            _out("  canonical  : %s" % canonical)
        _out("  supported  : yes")
    except UnsupportedHotkeyError as exc:
        _out("  configured : %s" % cfg.hotkey)
        _out("  supported  : NO")
        _out("  %s" % exc)
        problems.append(
            "Hotkey %r cannot be used. Supported: %s"
            % (cfg.hotkey, ", ".join(SUPPORTED_HOTKEYS))
        )
    _out("  min hold   : %d ms" % cfg.min_hold_ms)
    _out("  cancel     : Esc while holding")


def _doctor_config(cfg: Config, args: argparse.Namespace) -> None:
    _out("")
    _out("CONFIG")
    path = default_config_path()
    _out("  path       : %s" % path)
    _out("  exists     : %s" % ("yes" if path.exists() else "no (using defaults)"))
    overrides = _overrides_in_effect(args)
    if overrides:
        _out("  overridden on the command line: %s" % ", ".join(overrides))
    _out("  cleanup    : %s" % cfg.cleanup_level)
    _out("  sample rate: %d Hz" % cfg.sample_rate)
    _out("  preroll    : %d ms" % cfg.preroll_ms)
    _out("  raw history: %s" % ("kept" if cfg.keep_raw_history else "not kept"))
    if cfg.dictionary:
        _out("  dictionary : %d replacement(s)" % len(cfg.dictionary))


# -- bench ------------------------------------------------------------------


def _cmd_bench(cfg: Config, args: argparse.Namespace) -> int:
    """Measure model load time and transcription latency on this machine."""
    seconds = max(0.5, float(getattr(args, "seconds", _BENCH_DEFAULT_SECONDS)))
    repeat = max(1, int(getattr(args, "repeat", 3)))
    synth = bool(getattr(args, "synth", False))

    try:
        import numpy as np
    except BaseException as exc:  # noqa: BLE001
        _err("blurt: numpy is required for bench (%s: %s)" % (type(exc).__name__, exc))
        return 1

    hw = _hardware.detect()
    _out("blurt %s -- benchmark" % __version__)
    _out("  machine : %s, %d cores, %s (tier '%s')"
         % (hw.cpu_brand, hw.physical_cores, hw.arch, hw.tier))

    try:
        engine = _engines.select_engine(cfg, hw)
    except _engines.NoEngineAvailable as exc:
        _err("")
        _err("blurt: %s" % exc)
        return 1

    model, _why = _resolved_model(cfg, hw)
    threads, _tw = _resolved_threads(cfg, hw)
    _out("  engine  : %s" % getattr(engine, "name", "unknown"))
    _out("  model   : %s, %d threads, int8" % (model, threads))
    _out("")

    pcm, actual_seconds, source = _bench_sample(np, cfg, seconds, synth)
    _out("  sample  : %.1fs of %s audio" % (actual_seconds, source))
    _out("")

    # Load is timed separately: it happens once at startup and the user never
    # waits for it again, so folding it into per-dictation latency would misreport
    # both numbers.
    try:
        print("  loading model... ", end="", flush=True)
    except Exception:  # pragma: no cover
        pass
    load_started = time.monotonic()
    try:
        engine.load()
    except BaseException as exc:  # noqa: BLE001
        _out("failed")
        _err("blurt: could not load the model: %s: %s" % (type(exc).__name__, exc))
        _err("  On the first run this needs network access to download the weights.")
        return 1
    load_seconds = time.monotonic() - load_started
    _out("done in %.1fs (once, at startup -- not per dictation)" % load_seconds)
    _out("")

    latencies: List[float] = []
    text = ""
    for index in range(repeat):
        started = time.monotonic()
        try:
            text = engine.transcribe(pcm, cfg.sample_rate)
        except Exception as exc:  # noqa: BLE001
            _err("blurt: transcription failed: %s: %s" % (type(exc).__name__, exc))
            return 1
        elapsed = time.monotonic() - started
        latencies.append(elapsed)
        _out("  pass %d: %.2fs" % (index + 1, elapsed))

    try:
        engine.unload()
    except Exception:  # noqa: BLE001 - teardown is best effort
        pass

    # A benchmark that measures nothing must never publish a number.
    #
    # If the recording captured silence -- microphone permission denied, wrong
    # input device, nobody actually spoke -- the VAD filter trims the audio to
    # nothing and transcription returns almost instantly. That produces
    # impressive-looking sub-100ms timings that mean absolutely nothing, and it
    # happens precisely in the situation where a user is running `bench` to
    # diagnose a problem. Reporting "best: 0.02s" there would be worse than
    # useless: it would tell them their setup is fast when it is broken.
    if not text.strip():
        _out("")
        _err("blurt: transcription produced no text -- these timings are not valid.")
        _err("")
        _err("  The model returned nothing, which almost always means it received")
        _err("  silence rather than speech. Timings measured on silence are")
        _err("  meaningless (the VAD filter trims the audio to nothing and the")
        _err("  model returns immediately), so they are not reported.")
        _err("")
        if source == "microphone":
            _err("  Most likely causes, in order:")
            _err("    1. Microphone permission is not granted to this terminal.")
            _err("       Run 'blurt doctor' -- it tests the mic and reports the level.")
            _err("    2. The wrong input device is selected in System Settings > Sound.")
            _err("    3. Nothing was said during the recording window.")
        else:
            _err("  The synthetic sample failed to generate usable audio. Check that")
            _err("  the 'say' command works: say -o /tmp/t.wav --data-format=LEI16@16000 hello")
        return 1

    ordered = sorted(latencies)
    median = ordered[len(ordered) // 2]
    _out("")
    _out("RESULT on this machine")
    _out("  audio       : %.1fs" % actual_seconds)
    _out("  best        : %.2fs" % ordered[0])
    _out("  median      : %.2fs" % median)
    _out("  worst       : %.2fs" % ordered[-1])
    _out("  model load  : %.1fs (once)" % load_seconds)
    _out("")
    _out("  This is the delay between releasing the key and seeing your text.")
    _out("  Whisper pads every input to a 30-second window, so a 2-second phrase")
    _out("  costs roughly the same as a 15-second one -- speaking briefly does not")
    _out("  make it faster. The first pass is often slowest as caches warm up.")

    if source == "synthetic":
        _out("")
        _out("  Synthetic audio measures compute only, not accuracy. Re-run without")
        _out("  --synth to time real speech.")
    elif text.strip():
        _out("")
        _out("  Transcript: %s" % text.strip())

    return 0


def _bench_sample(
    np: Any, cfg: Config, seconds: float, synth: bool
) -> Tuple[Any, float, str]:
    """Return (pcm, seconds, source). Records unless told not to; falls back to synthetic.

    A silent recording falls back rather than failing: the point of bench is the
    latency number, and that is valid either way. It says which one it used.
    """
    if synth:
        return _synth_sample(np, cfg.sample_rate, seconds), seconds, "synthetic"

    try:
        from .audio import AudioUnavailable, Recorder
    except Exception as exc:  # noqa: BLE001
        _out("  (audio module unavailable: %s -- using synthetic audio)" % exc)
        return _synth_sample(np, cfg.sample_rate, seconds), seconds, "synthetic"

    recorder = None
    try:
        recorder = Recorder(sample_rate=cfg.sample_rate, preroll_ms=0)
    except AudioUnavailable as exc:
        _out("  (no microphone: %s)" % str(exc).splitlines()[0])
        _out("  falling back to synthetic audio")
        return _synth_sample(np, cfg.sample_rate, seconds), seconds, "synthetic"
    except Exception as exc:  # noqa: BLE001
        _out("  (microphone unavailable: %s: %s)" % (type(exc).__name__, exc))
        return _synth_sample(np, cfg.sample_rate, seconds), seconds, "synthetic"

    try:
        _out("  Speak normally for %.0f seconds when recording starts." % seconds)
        for remaining in range(_BENCH_COUNTDOWN, 0, -1):
            _out("    %d..." % remaining)
            time.sleep(1.0)
        recorder.start()
        _out("  recording...")
        time.sleep(seconds)
        pcm = recorder.stop()
        silent = recorder.last_capture_was_silent()
    except Exception as exc:  # noqa: BLE001
        _out("  (recording failed: %s: %s -- using synthetic audio)" % (type(exc).__name__, exc))
        return _synth_sample(np, cfg.sample_rate, seconds), seconds, "synthetic"
    finally:
        if recorder is not None:
            try:
                recorder.close()
            except Exception:  # noqa: BLE001
                pass

    frames = int(pcm.shape[0]) if hasattr(pcm, "shape") else 0
    if frames == 0 or silent:
        _out("  (the recording was silent -- check microphone permission)")
        _out("  falling back to synthetic audio; latency is still measured correctly")
        return _synth_sample(np, cfg.sample_rate, seconds), seconds, "synthetic"

    return pcm, frames / float(cfg.sample_rate), "recorded"


# Deliberately conversational, with the disfluencies real dictation contains, so
# the benchmark exercises the same path a real utterance would.
_SYNTH_SCRIPT = (
    "Hey, so I was thinking we should probably refactor the authentication "
    "module before we ship this, because right now it's doing a database "
    "lookup on every single request."
)


def _synth_sample(np: Any, sample_rate: int, seconds: float) -> Any:
    """Generate REAL synthesized speech using the macOS ``say`` command.

    This must be actual speech, not a synthetic tone, and that is not a matter of
    taste. An earlier version of this function built a sum of sine waves under a
    syllable-rate envelope and asserted that the timing was still valid even
    though the transcript was meaningless. That was wrong: the engine runs with
    ``vad_filter=True``, so voice-activity detection recognises a tone as
    non-speech and discards it BEFORE the encoder ever runs. The result was a
    benchmark that reported 0.02s and measured nothing whatsoever.

    ``say`` ships with every macOS install, needs no dependency, and emits a
    16 kHz mono WAV directly -- which is exactly the format Whisper wants, so
    there is no resampling and no ffmpeg in the path.

    Falls back to a tone only if ``say`` is unavailable, and in that case the
    caller's empty-transcript guard will correctly refuse to publish numbers.
    """
    import subprocess
    import tempfile
    import wave

    frames_wanted = max(1, int(sample_rate * seconds))

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "synth.wav")
            subprocess.run(
                [
                    "say",
                    "-o",
                    wav_path,
                    "--data-format=LEI16@%d" % sample_rate,
                    _SYNTH_SCRIPT,
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            with wave.open(wav_path, "rb") as handle:
                raw = handle.readframes(handle.getnframes())
                channels = handle.getnchannels()

        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1)

        # Trim or pad to the requested length so --seconds stays meaningful.
        if pcm.shape[0] > frames_wanted:
            pcm = pcm[:frames_wanted]
        elif pcm.shape[0] < frames_wanted:
            pcm = np.pad(pcm, (0, frames_wanted - pcm.shape[0]))
        return pcm.astype(np.float32)

    except Exception as exc:  # noqa: BLE001
        _out("  (could not synthesize speech with 'say': %s)" % exc)
        _out("  falling back to a test tone -- VAD will discard it, so no")
        _out("  timing will be reported. Check that 'say' works.")
        t = np.arange(frames_wanted, dtype=np.float32) / float(sample_rate)
        signal = np.zeros(frames_wanted, dtype=np.float32)
        for frequency, amplitude in ((120.0, 0.30), (330.0, 0.18), (900.0, 0.10)):
            signal += amplitude * np.sin(2.0 * np.pi * frequency * t)
        envelope = 0.5 + 0.5 * np.sin(2.0 * np.pi * 4.0 * t)
        return (signal * envelope * 0.5).astype(np.float32)


# -- config -----------------------------------------------------------------


def _cmd_config(cfg: Config, args: argparse.Namespace) -> int:
    """Print the resolved config, its path, and any command-line overrides."""
    path = default_config_path()
    _out("path   : %s" % path)
    _out("exists : %s" % ("yes" if path.exists() else "no (showing defaults)"))

    overrides = _overrides_in_effect(args)
    if overrides:
        _out("overridden for this run: %s" % ", ".join(overrides))
        _out("(overrides are not saved to disk)")

    hw = _hardware.detect()
    model, why = _resolved_model(cfg, hw)
    threads, thread_why = _resolved_threads(cfg, hw)

    _out("")
    _out("resolved settings:")
    _out(json.dumps(dataclasses.asdict(cfg), indent=2, sort_keys=True))
    _out("")
    _out("what 'auto' resolves to on this machine:")
    _out("  model   : %s (%s)" % (model, why))
    _out("  threads : %d (%s)" % (threads, thread_why))
    if (cfg.engine or "auto").strip().lower() == "auto":
        try:
            usable = _engines.available_engines()
        except Exception:  # noqa: BLE001
            usable = []
        _out("  engine  : %s" % (usable[0] if usable else "none available"))
    return 0


# -- run --------------------------------------------------------------------


def _cmd_run(cfg: Config, args: argparse.Namespace) -> int:
    """Start the dictation daemon. Imported late so doctor still works if it fails.

    ``blurt doctor`` has to survive a broken install -- that is its entire job --
    so the module that pulls in numpy, pynput and pyobjc is imported here rather
    than at the top of the file.
    """
    try:
        from .app import run as run_app
    except BaseException as exc:  # noqa: BLE001 - a missing dependency lands here
        _err("blurt: could not start (%s: %s)" % (type(exc).__name__, exc))
        _err("  Run 'blurt doctor' to see which dependency is missing.")
        return 1
    return run_app(cfg)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments and dispatch. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = _apply_overrides(load_config(), args)

    command = getattr(args, "command", None) or "run"
    if command == "doctor":
        return _cmd_doctor(cfg, args)
    if command == "bench":
        return _cmd_bench(cfg, args)
    if command == "config":
        return _cmd_config(cfg, args)
    return _cmd_run(cfg, args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        # Ctrl+C before the app installs its own handler. Exit quietly: a
        # traceback here would suggest a crash where the user simply quit.
        _err("")
        raise SystemExit(130)
