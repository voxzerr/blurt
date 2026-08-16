"""Command line entry point for blurt: ``python3 -m blurt`` (or ``blurt``).

Subcommands::

    blurt                       run the dictation daemon (the default)
    blurt doctor                diagnose this machine -- run this first when
                                something is wrong
    blurt bench                 measure real transcription latency HERE, not in
                                a datasheet
    blurt config                print the resolved configuration and where it
                                came from
    blurt config get KEY        print one setting's value, raw, for scripting
    blurt config set KEY VALUE  change one setting and save it
    blurt learn                 review your transcript journal and personalize
                                from it

``doctor`` is the important one. blurt's failure modes on macOS are almost all
permission or environment problems that produce silence rather than errors: a
hotkey that never fires, a microphone that returns zeros, a paste that macOS
discards. None of those raise an exception, so a confused user has nothing to
read. ``doctor`` exists to turn all of that into text.

``bench`` reports measured numbers from this specific machine. The spread is
enormous -- tiny.en takes ~2s on a 2017 Intel i7 and a fraction of that on an
M-series -- so quoting anyone else's figures would be dishonest.

``config set`` exists because for a long time it did not, and the whole learning
loop was the casualty. ``history_enabled`` defaults to false, so the journal that
``learn`` reads only starts filling once the user turns it on -- and the only way
to turn it on was to hand-author JSON at a path that does not exist yet on a
fresh install. Asking someone to create ``~/.config/blurt/config.json`` from
memory, correctly, before they can try a feature is the same as not shipping the
feature. ``set`` is deliberately narrow: one scalar setting at a time, validated
against exactly the rules ``load_config`` enforces, and it never writes anything
it could not read back.

``learn`` reads the opt-in transcript journal (``history_enabled``) and proposes
``dictionary`` and ``initial_prompt`` entries from it. It proposes; it does not
decide. ``--apply`` walks the suggestions one at a time, ``--yes`` accepts the
high-confidence ones unattended, and ``--forget`` deletes the journal outright.

Two commands here write to disk -- ``learn --apply`` and ``config set`` -- and
neither of them persists the in-memory config that ``main`` built. That one has
the one-run override flags folded into it, and saving it would silently promote a
temporary ``--cleanup standard`` into a permanent setting.

Both go further than that and never build a ``Config`` for the user's file at
all. They read the config as the raw JSON object it is on disk, change only the
keys they were actually asked to change, and write that object back, so keys this
version of blurt does not recognise and values its loader would have replaced are
both left exactly where the user put them. ``set`` changes one key;
``learn --apply`` changes ``dictionary`` and ``initial_prompt`` and nothing else.
See ``_read_config_document`` for what the alternative destroys, and
``_merge_into_config_file`` for the write.

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
import pathlib
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import __version__
from . import engines as _engines
from . import hardware as _hardware
from . import history as _history
from . import learn as _learn
from .config import (
    VALID_CLEANUP_LEVELS,
    VALID_ENGINES,
    Config,
    default_config_path,
    load_config,
    save_config,
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
        epilog=(
            "Run 'blurt doctor' first if anything is not working.\n"
            "\n"
            "Settings live in a JSON file, but you never have to write it by hand:\n"
            "  blurt config                            show everything, and its path\n"
            "  blurt config get history_enabled        print one value, raw\n"
            "  blurt config set history_enabled true   turn on the journal 'learn' reads\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    # `config` gained subcommands rather than flags (`--get`, `--set`) so that the
    # bare, read-only invocation stays exactly what it always was. Someone typing
    # `blurt config` to look at their settings must never discover that they have
    # changed something.
    config = sub.add_parser(
        "config",
        parents=[common],
        help="show, read or change the configuration",
        # Wrapped by hand: RawDescriptionHelpFormatter is here to keep the
        # example block in the epilog aligned, and it does not re-wrap the
        # description either.
        description=(
            "With no action, print the effective settings and where they were\n"
            "read from -- read-only, as it has always been. 'get' prints a single\n"
            "value for scripting. 'set' is the supported way to change a setting\n"
            "without hand-editing JSON."
        ),
        epilog=(
            "Examples:\n"
            "  blurt config                            show everything\n"
            "  blurt config get history_enabled        print one value, raw\n"
            "  blurt config set history_enabled true   turn on the transcript journal\n"
            "  blurt config set cleanup_level standard\n"
            "\n"
            "The 'dictionary' setting holds many entries and cannot be set this\n"
            "way; 'blurt learn --apply' builds it from your own transcripts.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    config_actions = config.add_subparsers(dest="config_action", metavar="ACTION")

    config_get = config_actions.add_parser(
        "get",
        parents=[common],
        help="print one setting's value, raw, on a single line",
        description=(
            "Print the resolved value of one setting and nothing else -- no "
            "label, no quotes, no trailing commentary -- so it can be piped or "
            "captured in a shell variable. Booleans print as true/false, which "
            "is what 'set' accepts back."
        ),
    )
    config_get.add_argument(
        "key",
        metavar="KEY",
        help="setting name; 'blurt config' lists them all with their values",
    )

    config_set = config_actions.add_parser(
        "set",
        parents=[common],
        help="change one setting and write it to the config file",
        description=(
            "Validate a value, merge it into the config file, and say what "
            "changed. Only the named setting is touched -- everything else in "
            "the file is left exactly as written, including settings this "
            "version of blurt does not recognise. A rejected value writes "
            "nothing at all. Settings take effect the next time blurt starts."
        ),
    )
    config_set.add_argument(
        "key",
        metavar="KEY",
        help="setting name; 'blurt config' lists them all",
    )
    config_set.add_argument(
        "value",
        metavar="VALUE",
        help=(
            "true/false for a switch (yes/no/on/off/1/0 also work), a whole "
            "number for a count, otherwise the literal text"
        ),
    )

    learn = sub.add_parser(
        "learn",
        parents=[common],
        help="review your transcript journal and personalize blurt from it",
        description=(
            "Analyse the opt-in transcript journal and propose dictionary and "
            "initial_prompt entries. Prints suggestions by default and changes "
            "nothing without --apply."
        ),
    )
    learn.add_argument(
        "--apply",
        action="store_true",
        help="review each suggestion and write the accepted ones to your config",
    )
    learn.add_argument(
        "--yes",
        action="store_true",
        help="with --apply, accept every high-confidence suggestion without asking",
    )
    learn.add_argument(
        "--forget",
        action="store_true",
        help="delete the transcript journal and exit",
    )
    learn.add_argument(
        "--min",
        dest="min_occurrences",
        type=int,
        default=_learn.DEFAULT_MIN_OCCURRENCES,
        metavar="N",
        help="occurrences before a pattern is reported (default: %(default)s)",
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
    _doctor_journal(cfg)


def _doctor_journal(cfg: Config) -> None:
    """Report the on-disk journal, including when it is off.

    Printed unconditionally rather than only when enabled. A tool that writes your
    speech to disk should be visible in the diagnostic whether or not it is
    currently doing it -- "off" is the answer most users need to be able to
    confirm, and they should not have to infer it from an absent line.
    """
    path = _history.default_history_path()
    if not cfg.history_enabled:
        _out("  journal    : off (history_enabled = false)")
        if path.exists():
            _out(
                "               a journal file still exists at %s -- "
                "'blurt learn --forget' deletes it" % path
            )
        return

    _out("  journal    : ON -- transcripts are written to disk")
    _out("               %s" % path)
    _out(
        "               %d record(s), capped at %d"
        % (_history.history_size(path), cfg.history_limit)
    )
    _out("               'blurt learn' reads it; 'blurt learn --forget' deletes it")


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


#: Accepted spellings for a boolean setting. Generous on input and strict on
#: output: whatever the user types, the file always ends up holding a real JSON
#: ``true``/``false``, because a quoted "true" is the one thing ``load_config``
#: will refuse and fall back to the default for.
#:
#: Tuples rather than sets so the help text can print them in the order a person
#: would say them. Four entries each; membership testing is not the bottleneck.
_TRUE_WORDS: Tuple[str, ...] = ("true", "yes", "on", "1")
_FALSE_WORDS: Tuple[str, ...] = ("false", "no", "off", "0")

#: Returned by the parsing helpers instead of raising, and deliberately not
#: ``None``: ``False``, ``0`` and ``""`` are all legitimate settings, so any
#: falsy sentinel would make "rejected" indistinguishable from "set to off".
_REJECTED: Any = object()


def _config_fields() -> Dict[str, "dataclasses.Field"]:
    """Map setting name -> dataclass field, read straight off :class:`Config`.

    Derived rather than written out by hand, and that is not fastidiousness. A
    hardcoded list of settable keys is wrong the moment somebody adds a field to
    ``Config`` without knowing this file exists, and the failure is quiet: the new
    setting simply cannot be reached from the CLI, ``blurt config get`` calls it
    unknown, and nothing anywhere says why. Reading the dataclass means a new
    field is settable the day it lands.
    """
    return dict((f.name, f) for f in dataclasses.fields(Config))


def _declared_type(field: "dataclasses.Field") -> Any:
    """Resolve a field's declared annotation to a real type, or None.

    ``blurt.config`` uses ``from __future__ import annotations``, so ``field.type``
    is the *string* ``"bool"`` rather than the class -- comparing it against
    ``bool`` directly silently fails for every field, and the symptom would be
    "every setting is unsettable" with no error to read. The ``isinstance(type)``
    branch keeps this working if that import is ever dropped from config.py.

    Returns None for anything that is not a scalar (today: ``dictionary``, whose
    annotation is ``Dict[str, str]``). Callers treat None as "cannot come from a
    single command-line word" rather than guessing.
    """
    declared = field.type
    if isinstance(declared, type):
        return declared
    return {"bool": bool, "int": int, "str": str}.get(str(declared).strip())


def _format_value(value: Any) -> str:
    """Render a setting the way the config file spells it.

    ``get`` exists to be piped, and the obvious thing to pipe it into is ``set``,
    so the two have to agree: booleans print as ``true``/``false`` rather than
    Python's ``True``, and strings print bare so ``$(blurt config get hotkey)`` is
    a hotkey name and not a hotkey name wrapped in punctuation. The dictionary
    prints as one line of JSON -- it cannot be fed back to ``set``, but a value
    that is read-only should still be readable.
    """
    if isinstance(value, bool):  # before int: bool is a subclass of it
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _unknown_setting(key: str) -> int:
    """Complain about a key nobody has, and list the ones that exist. Exits 2."""
    _err("blurt: unknown setting %r." % (key,))
    _err("  Valid settings: %s" % ", ".join(sorted(_config_fields())))
    _err("  'blurt config' prints all of them with their current values.")
    return 2


def _parse_string_setting(key: str, raw: str) -> Any:
    """Validate a string setting using whatever rule already governs it.

    Every check here is borrowed rather than reinvented, because a second copy of
    a rule is a copy that drifts: ``engine`` and ``cleanup_level`` go against the
    same frozensets ``load_config`` uses, and both hotkey fields go through
    ``normalize_key_name`` so the CLI refuses exactly what the input layer
    refuses. That last one matters more than it looks -- ``fn`` is a perfectly
    reasonable thing to want as a push-to-talk key and pynput cannot see it on
    macOS at all, so accepting it here would save a config that produces a
    daemon which starts cleanly and then never responds to the key.

    Values are stripped because ``load_config`` strips them on the way back in.
    Storing ``"  right_cmd  "`` would only mean the file disagrees with the
    setting blurt actually uses.
    """
    value = raw.strip()

    if key == "engine":
        value = value.lower()
        if value not in VALID_ENGINES:
            _err(
                "blurt: engine %r is not valid. Choose one of: %s"
                % (raw, ", ".join(sorted(VALID_ENGINES)))
            )
            return _REJECTED
        return value

    if key == "cleanup_level":
        value = value.lower()
        if value not in VALID_CLEANUP_LEVELS:
            _err(
                "blurt: cleanup_level %r is not valid. Choose one of: %s"
                % (raw, ", ".join(sorted(VALID_CLEANUP_LEVELS)))
            )
            return _REJECTED
        return value

    if key in ("hotkey", "assistant_hotkey"):
        try:
            return normalize_key_name(value)
        except UnsupportedHotkeyError as exc:
            _err("blurt: %s %r is not usable.\n  %s" % (key, raw, exc))
            _err("  Supported: %s" % ", ".join(SUPPORTED_HOTKEYS))
            return _REJECTED

    if key == "model" and not value:
        # Same rule as --model: an empty model name is always a mistake, and
        # unlike the sizes it is one we can be certain about. Model names
        # otherwise go unvalidated because faster-whisper accepts local paths
        # and Hugging Face ids as well as tiny.en/base.en/small.en, and
        # rejecting an unfamiliar string would block a legitimate use.
        _err("blurt: model must not be empty")
        return _REJECTED

    return value


def _parse_setting(key: str, field: "dataclasses.Field", raw: str) -> Any:
    """Turn one command-line word into a value of the field's declared type.

    Returns :data:`_REJECTED` after printing the reason to stderr; it never
    raises, and it never guesses. Typed-but-unparseable is treated as a hard
    error rather than as "use the default", because a user who typed
    ``history_enabled banana`` has a belief about what is about to happen and
    quietly writing ``false`` would leave that belief intact and wrong.
    """
    kind = _declared_type(field)

    if kind is bool:
        text = raw.strip().lower()
        if text in _TRUE_WORDS:
            return True
        if text in _FALSE_WORDS:
            return False
        _err("blurt: %s is a true/false setting; %r is neither." % (key, raw))
        _err(
            "  On: %s.  Off: %s.  Case does not matter."
            % (" / ".join(_TRUE_WORDS), " / ".join(_FALSE_WORDS))
        )
        return _REJECTED

    if kind is int:
        try:
            return int(raw.strip(), 10)
        except ValueError:
            _err("blurt: %s is a whole number; %r is not one." % (key, raw))
            return _REJECTED

    if kind is str:
        return _parse_string_setting(key, raw)

    # Not a scalar. Today that is only `dictionary`, and the temptation is to
    # invent a syntax for it -- `set dictionary github=GitHub` or similar. That
    # would be a worse version of a feature that already exists and knows more
    # than the user does about what belongs in there.
    _err(
        "blurt: %s holds a %s, which cannot be set from a single value."
        % (key, field.type)
    )
    if key == "dictionary":
        _err("  'blurt learn --apply' builds it from your own transcripts, which")
        _err("  is the only way blurt can know what you actually say.")
    _err("  To edit it directly: %s" % default_config_path())
    return _REJECTED


def _loader_would_reject(cfg: Config, key: str, value: Any) -> bool:
    """Ask the loader itself whether this value survives a round trip.

    ``config.py`` owns the numeric ranges -- sample_rate 8000..48000, the
    millisecond caps, the journal cap -- and keeps them private. Copying them
    here would put a second statement of the rules in a file with no way to
    notice when the first one changes, so instead we serialize, read it back, and
    see what comes out. The bounds stay in exactly one place.

    Worth the round trip because the failure is otherwise invisible: an
    out-of-range ``sample_rate`` saves without complaint and is then silently
    replaced by the default on every single launch, behind a stderr warning the
    user has long since stopped reading. Refusing now costs a second of their
    time; accepting costs a setting that permanently does not do anything.

    Best effort in one direction only. If the probe cannot run at all -- no
    writable temp directory -- the answer is "not rejected", because refusing to
    save a legitimate setting because a scratch file could not be created would
    be the wrong trade in a helper whose entire job is catching a typo.

    ``cfg`` is a throwaway carrying defaults for everything except ``key`` (see
    :func:`_cmd_config_set`), not the user's own config. Nothing in
    ``_from_dict`` validates one field against another -- every setting is picked
    independently -- so the answer for ``key`` is the same either way, and a
    defaults-based probe keeps the warnings free of complaints about fields the
    user is not currently changing.

    THE STDERR CAPTURE THAT USED TO BE HERE IS GONE, and it is worth recording
    why it was here at all. This ran the probe load with stderr redirected into a
    buffer and then replayed only the captured lines containing ``key``, because
    a probe file built from the defaults was not quiet: ``save_config``
    serialises the whole dataclass, so it writes ``"initial_prompt": ""``, and
    ``config.py``'s ``_pick_str`` warned about an empty value even for a field
    whose default was empty too. A perfectly healthy config therefore announced
    "initial_prompt is empty" on the way in, and printing that above "Wrote your
    config" read as a failure that had not happened.

    That is now fixed where it belonged, in ``_pick_str``: an empty value only
    warns when the field has a real default to fall back to. With it fixed the
    workaround is not merely unnecessary but worse than nothing. A defaults file
    loads in total silence, so the only warning this probe can now produce is the
    one about ``key`` -- precisely the line we wanted the user to see -- and
    letting ``load_config`` print it itself also retires the ``key in line``
    substring match, which was a sieve waiting to mis-fire the day two settings
    shared a name fragment. Every path in ``_from_dict`` that warns also returns
    the default, so "the loader printed something" and "the value did not
    survive" are the same event; there is no case where suppressing output would
    still be doing work.
    """
    import tempfile

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            probe = pathlib.Path(tmpdir) / "config.json"
            save_config(cfg, probe)
            # Deliberately not captured. On a mismatch load_config states the
            # actual range, and no message written here could do it as
            # accurately; on a match it says nothing at all, so a successful
            # `set` still writes nothing to stderr.
            return getattr(load_config(probe), key) != value
    except OSError:
        return False


def _serialised_defaults() -> Dict[str, Any]:
    """The default config as the JSON object ``save_config`` would have written.

    ``set`` starts from this when there is no file yet. Writing only the one key
    the user named would load perfectly well -- every absent key falls back to
    its default -- but it would leave them with a file whose shape depends on
    which setting they happened to touch first, and nothing to read when they
    open it wondering what else is in there. A complete file is the friendlier
    artefact and costs nothing.
    """
    return dataclasses.asdict(Config())


def _read_config_document(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    """Read the config file as the plain JSON object it actually is on disk.

    NOT ``load_config``, and the difference is the whole point of this function.
    ``load_config`` returns a :class:`Config`, and a ``Config`` is a lossy view of
    the file in two directions that both destroy user data on the way back out:

      * It drops keys it does not recognise. That is deliberate and correct for
        *reading* -- ``config.py`` calls it forward compatibility, an older blurt
        quietly tolerating a newer blurt's settings -- but round-tripping through
        the dataclass turns "tolerated" into "deleted". Run an older blurt once,
        change one setting, and every setting the newer one wrote is gone, with
        nothing on screen to say so.
      * It substitutes the default for any value it rejects. So a typo'd
        ``sample_rate: 999999`` would be *overwritten with 16000* by the very act
        of changing an unrelated setting, erasing the evidence the user needs in
        order to find and fix their typo.

    Both are silent and neither is recoverable, which is why ``set`` merges into
    the raw dict instead: it changes the one key it was asked to change and
    leaves every other byte's worth of meaning alone.

    Returns the serialised defaults when there is no file (the normal fresh
    install), and ``None`` -- after printing why -- when a file exists but cannot
    be understood. ``None`` deliberately does not mean "start fresh": an
    unparseable file is still the only copy of whatever the user wrote in it, and
    silently replacing it with defaults would be the single most destructive
    thing this command could do. Note that this reads the file directly rather
    than going anywhere near ``load_config``, which renames a corrupt config to
    ``.bak`` as a side effect of being asked to read it; refusing means refusing,
    including refusing to move their file.
    """
    try:
        with open(str(path), "r", encoding="utf-8") as handle:
            text = handle.read()
    except FileNotFoundError:
        return _serialised_defaults()
    except (OSError, UnicodeDecodeError) as exc:
        _err("blurt: could not read %s (%s)" % (path, exc))
        _err("  Nothing was written. blurt will not replace a config file it was")
        _err("  unable to read first -- the bytes in there may still be yours.")
        return None

    try:
        data = json.loads(text)
    except ValueError as exc:  # JSONDecodeError on 3.9; keep it broad
        _err("blurt: %s is not valid JSON (%s)." % (path, exc))
        _explain_unusable_config(path)
        return None

    if not isinstance(data, dict):
        _err(
            "blurt: %s must contain a JSON object, got %s."
            % (path, type(data).__name__)
        )
        _explain_unusable_config(path)
        return None

    return data


def _explain_unusable_config(path: pathlib.Path) -> None:
    """Tell the user how to get out of an unreadable config, and write nothing.

    Split out only so the two shapes of "unusable" -- text that is not JSON at
    all, and valid JSON that is not an object -- give identical advice, because
    the way out of both of them is identical.
    """
    _err("  Nothing was written. Overwriting it would destroy whatever is in")
    _err("  there, and a config file can hold a replacement dictionary built from")
    _err("  months of your own speech that exists nowhere else.")
    _err("  Fix the JSON, or move the file out of the way and run this again:")
    _err("      mv %s %s.bak" % (path, path))


def _write_config_document(document: Dict[str, Any], path: pathlib.Path) -> None:
    """Write a raw config object with exactly ``save_config``'s durability.

    This duplicates ``save_config``'s body, and that is the deliberate choice
    rather than the lazy one. The alternatives were worse:

      * Round-trip the dict through a ``Config`` and call ``save_config``. That
        is precisely the data loss this whole change exists to remove.
      * Widen ``save_config`` to accept a raw dict. It lives in ``config.py``,
        which is owned elsewhere, and giving the module's one durable-write
        function a second calling convention to serve one CLI subcommand is a
        cost paid by every future reader of it.
      * Write the file with ``open(path, "w")``. Absolutely not. That truncates
        the user's real config *first* and then writes; a crash, a full disk or a
        SIGKILL in between leaves them with a half-written or empty config where
        a good one used to be. The whole reason ``save_config`` builds a temp file
        and calls ``os.replace`` is that ``os.replace`` is atomic -- the config is
        the old one or the new one, never a fragment of either.

    So: same temp-file-in-the-destination-directory (``os.replace`` is only
    atomic within one filesystem), same fsync before the rename, same 0600 before
    it is reachable by name, same best-effort directory fsync after. Same
    ``json.dumps`` arguments too, so a ``set`` that changes nothing produces a
    byte-identical file. Raises ``OSError``, like ``save_config``; the caller
    reports it.
    """
    import tempfile

    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"

    fd, tmp_name = tempfile.mkstemp(
        dir=str(directory), prefix="." + path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # chmod before the rename: the file must never be readable at its real
        # name with the wrong mode, not even for an instant.
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, str(path))
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

    try:
        dir_fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def _merge_into_config_file(
    document: Dict[str, Any], changes: Dict[str, Any], path: pathlib.Path
) -> bool:
    """Set exactly ``changes`` in an already-read document and save it. True on success.

    The back half of the read-modify-write that both writing commands share;
    :func:`_read_config_document` is the front half. Factored out rather than
    written twice because the halves only mean anything as a pair: reading the
    raw JSON is what preserves the keys and values a ``Config`` would have
    dropped, and that preservation is undone the instant somebody writes the file
    back from anything other than the object they read. ``learn --apply`` was the
    proof -- ``config set`` was carefully merging while ``learn --apply``, four
    hundred lines away, still did ``load_config()`` then ``save_config()`` and
    deleted the same unknown keys the other command was protecting.

    ``changes`` is the whole point of the signature. Callers pass only the keys
    they actually decided to change, so a key nobody touched is not rewritten
    even to an identical value -- which matters for a key whose current contents
    the loader would not accept, since re-deriving it is the step that would
    quietly replace it.

    Nothing is validated here. Both callers validate before they get this far,
    each against the rules that apply to what they are writing, and a second
    opinion at the write barrier would only be a third statement of rules that
    already live in two places.
    """
    document.update(changes)
    try:
        _write_config_document(document, path)
    except OSError as exc:
        _err("blurt: could not write %s (%s)" % (path, exc))
        _err("  Nothing was changed. Check that %s is writable, or point" % path.parent)
        _err("  XDG_CONFIG_HOME somewhere that is.")
        return False
    return True


#: The two settings that name a physical key. Held together because the only
#: interesting rule about either is the one that relates them to each other.
_HOTKEY_FIELDS: Tuple[str, str] = ("hotkey", "assistant_hotkey")


def _canonical_hotkey(name: str) -> str:
    """Fold a hotkey spelling for comparison, tolerating ones blurt cannot use.

    ``normalize_key_name`` is the authority, but it raises for anything
    unsupported and a *comparison* must not. A config that already says
    ``hotkey: "fn"`` is broken in its own separate way; that is not this check's
    business, and it still has to be able to answer "is this the same key as the
    one being set" without exploding. Falling back to the stripped, lowercased
    text answers that correctly: two identical unusable spellings compare equal,
    and an unusable one never compares equal to a canonical one.
    """
    try:
        return normalize_key_name(name)
    except UnsupportedHotkeyError:
        return name.strip().lower()


def _effective_hotkey(document: Dict[str, Any], key: str) -> str:
    """What blurt would really use for one hotkey field, given the file as it is.

    Mirrors ``_pick_str``: a present, non-empty string is taken as written and
    anything else (missing, null, a number, blank) falls back to the field's
    default. Asking the raw document rather than a loaded ``Config`` keeps this
    honest about a file that has not been through the loader yet -- which is the
    only kind of file ``set`` ever sees now.
    """
    raw = document.get(key)
    if not isinstance(raw, str) or not raw.strip():
        return getattr(Config(), key)
    return raw.strip()


def _hotkey_would_collide(document: Dict[str, Any], key: str, value: str) -> bool:
    """Refuse a hotkey that would leave both keys pointing at the same physical key.

    THE FAILURE THIS PREVENTS. Validated in isolation, ``assistant_hotkey
    right_option`` is a perfectly good value, and so is ``hotkey right_option``.
    Together they are not. ``BlurtApp._build_assistant`` resolves the tie by
    switching assistant mode off entirely -- one key cannot mean two things, and
    silently guessing which the user meant would be worse -- so a single
    valid-looking command turns off command mode without ever saying it did. The
    user then has no hotkey for the assistant AND, since command mode is where
    ``revert_last`` lives, no way to speak an undo either. Two features gone, no
    error, and the config file looks entirely reasonable.

    THE RULE IS THE RUNTIME'S RULE. ``_build_assistant`` compares
    ``assistant_hotkey`` against ``hotkey`` and disables the assistant when they
    are equal. This asks the same question, with one deliberate strengthening:
    both sides are folded through :func:`_canonical_hotkey` first. The runtime
    compares the strings as loaded, so ``right_alt`` and ``right_option`` -- the
    same key, two accepted spellings -- slip past it and produce something worse
    than the case it does catch: two hotkey listeners bound to one physical key,
    with no warning at all. Refusing here covers both, and cannot drift from the
    runtime in the direction that matters, because everything the runtime rejects
    this rejects too.

    Not conditioned on ``assistant_enabled``, though ``_build_assistant`` checks
    it first. A collision that is currently harmless because the assistant is off
    is a trap armed for whenever somebody runs ``config set assistant_enabled
    true``, and at that point the message would be attached to the wrong command
    entirely. Refusing now costs one retry with a different key; the alternative
    costs a feature that quietly does not exist.

    Prints and returns True on a conflict, in the style of
    :func:`_loader_would_reject`. Nothing is written by the caller either way.
    """
    other = "assistant_hotkey" if key == "hotkey" else "hotkey"
    other_value = _effective_hotkey(document, other)
    if _canonical_hotkey(value) != _canonical_hotkey(other_value):
        return False

    current = _effective_hotkey(document, key)
    _err(
        "blurt: %s %s would collide with %s; nothing was written."
        % (key, _format_value(value), other)
    )
    _err("  %-16s : %s" % (other, other_value))
    _err("  %-16s : %s  (you asked for %s)" % (key, current, _format_value(value)))
    _err("  One key cannot mean two things. With both set the same, blurt turns")
    _err("  assistant mode off at startup -- so this command would have disabled")
    _err("  command mode, and command mode is how you would speak the undo.")
    _err("  Give them different keys, or change %s first:" % other)
    _err("      blurt config set %s KEY" % other)
    _err("  Supported: %s" % ", ".join(SUPPORTED_HOTKEYS))
    return True


def _cmd_config(cfg: Config, args: argparse.Namespace) -> int:
    """Dispatch the three shapes of ``blurt config``.

    The bare form is unchanged and stays read-only. ``get`` and ``set`` are
    subcommands rather than flags so that adding them could not possibly alter
    what the bare form does.
    """
    action = getattr(args, "config_action", None)
    if action == "get":
        return _cmd_config_get(cfg, args)
    if action == "set":
        return _cmd_config_set(args)
    return _cmd_config_show(cfg, args)


def _cmd_config_get(cfg: Config, args: argparse.Namespace) -> int:
    """Print one setting and nothing else, for scripts.

    Reads the *resolved* config -- the one with any ``--model`` / ``--cleanup``
    overrides folded in -- so that ``blurt --model tiny.en config get model``
    answers the question actually asked: what would this run use. The file is
    untouched either way; only ``set`` reads it back off disk.
    """
    key = getattr(args, "key", "") or ""
    if key not in _config_fields():
        return _unknown_setting(key)
    _out(_format_value(getattr(cfg, key)))
    return 0


def _cmd_config_set(args: argparse.Namespace) -> int:
    """Validate one setting, merge it into the file on disk, and report the change.

    A MERGE, NOT A ROUND TRIP. This edits the raw JSON object in the config file
    and changes exactly one key in it. It used to load a ``Config`` and write the
    whole dataclass back, which quietly deleted every key the current version does
    not recognise and quietly overwrote every value the loader had rejected --
    see :func:`_read_config_document` for why both of those are unrecoverable.
    The user asked to change one setting; one setting is what changes.

    Deliberately ignores the config ``main`` resolved. That one has any
    ``--model`` / ``--cleanup`` / ``--engine`` / ``--hotkey`` overrides folded
    into it, and those are for one run by construction; writing them back would
    turn a flag someone passed to try something into a permanent setting they
    never chose and would have no reason to look for. ``_learn_apply`` avoids the
    same trap the same way -- it reads the file rather than the resolved config --
    and both are covered by tests that pass an override and assert it never lands.

    Validation still happens against the typed field, and against the loader's own
    range rules, before anything is written -- the file is only merged into once
    every check has passed. So a rejected value leaves it exactly as it was,
    including "not existing at all", which is the normal state on a fresh install.
    """
    key = getattr(args, "key", "") or ""
    raw = getattr(args, "value", None)
    if raw is None:
        raw = ""

    fields = _config_fields()
    if key not in fields:
        return _unknown_setting(key)

    value = _parse_setting(key, fields[key], raw)
    if value is _REJECTED:
        return 2

    path = default_config_path()
    document = _read_config_document(path)
    if document is None:
        # The file exists and could not be understood. _read_config_document has
        # already said so and said what to do about it; the only thing left that
        # could make this worse is writing.
        return 1

    if key in _HOTKEY_FIELDS and _hotkey_would_collide(document, key, value):
        return 2

    # The loader probe gets a throwaway config carrying this one field, rather
    # than the user's own settings, because the question is only ever "does this
    # value survive load_config" and every field is picked independently of the
    # rest. Building it from `document` would mean constructing a Config from the
    # file -- the exact lossy step this command now exists to avoid.
    probe = Config()
    setattr(probe, key, value)
    if _loader_would_reject(probe, key, value):
        _err(
            "blurt: %s=%s is not a value blurt can use; nothing was written."
            % (key, _format_value(value))
        )
        return 2

    # Reported from the file rather than from a loaded Config on purpose: if the
    # old value was one the loader rejects, the honest thing to show the user is
    # what their file said, not the default blurt was quietly substituting for it.
    before = document.get(key, getattr(Config(), key))
    if not _merge_into_config_file(document, {key: value}, path):
        return 1

    _out("%s: %s -> %s" % (key, _format_value(before), _format_value(value)))
    if before == value:
        _out("  (that was already the value; the file was rewritten anyway)")
    _out("Wrote %s" % path)

    if key == "history_enabled" and value is True:
        # Restate the cost at the moment it starts applying, not only in the
        # explainer someone read to get here. This is the one setting in blurt
        # that turns speech into a file.
        _out("")
        _out("blurt will now write your transcripts to disk (0600, in a 0700")
        _out("directory, never leaving this machine). 'blurt learn' reads them;")
        _out("'blurt learn --forget' deletes them.")

    _out("")
    _out("This takes effect the next time you start blurt.")
    return 0


def _cmd_config_show(cfg: Config, args: argparse.Namespace) -> int:
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

    _out("")
    _out("to change one of these:")
    _out("  blurt config set KEY VALUE   e.g. blurt config set cleanup_level standard")
    _out("  blurt config get KEY         prints one value alone, for scripts")
    if not cfg.history_enabled:
        _out("")
        _out("the transcript journal that 'blurt learn' needs is off. Turn it on with:")
        _out("  blurt config set history_enabled true")
    return 0


# -- learn ------------------------------------------------------------------


def _cmd_learn(cfg: Config, args: argparse.Namespace) -> int:
    """Report what the transcript journal suggests, and optionally apply it.

    Reads nothing but the journal and the config, and writes nothing unless
    ``--apply`` was passed. The split is deliberate: the default invocation is
    safe to run out of curiosity, which is the only way anyone will ever discover
    what this command is for.
    """
    if getattr(args, "forget", False):
        return _learn_forget()

    path = _history.default_history_path()

    if not cfg.history_enabled and not path.exists():
        return _learn_explain_disabled(path)

    records = _history.load_records(path)
    if not records:
        _out("blurt %s -- learn" % __version__)
        _out("")
        _out("  journal : %s" % path)
        _out("  records : 0")
        _out("")
        if cfg.history_enabled:
            _out("  Recording is on, but nothing has been journalled yet. Dictate for")
            _out("  a while and come back -- suggestions need a few days of your own")
            _out("  words before they mean anything.")
        else:
            _out("  Recording is off (history_enabled = false), so there is nothing")
            _out("  to learn from yet. Turn it on with:")
            _out("")
            _out("      blurt config set history_enabled true")
            _out("")
            _out("  That writes your dictations to disk; 'blurt learn --forget'")
            _out("  deletes them again.")
        return 0

    report = _learn.analyze(
        records,
        dictionary=cfg.dictionary,
        initial_prompt=cfg.initial_prompt,
        min_occurrences=getattr(args, "min_occurrences", _learn.DEFAULT_MIN_OCCURRENCES),
    )

    _print_learn_report(report, path, cfg)

    if not getattr(args, "apply", False):
        if report.suggestions:
            _out("")
            _out("Nothing was changed. To review and apply these:")
            _out("    blurt learn --apply")
        return 0

    return _learn_apply(report, args)


def _learn_explain_disabled(path: pathlib.Path) -> int:
    """Explain the feature to someone who has never turned it on.

    This is the only place the trade-off gets stated before the user opts in, so
    it states it plainly rather than selling the feature.

    It used to end by naming the setting and showing the JSON to add. That was a
    dead end in practice: on a fresh install the config file does not exist, so
    "add this to ~/.config/blurt/config.json" means "create a directory and a
    file, get the JSON right, and hope". Anyone unwilling to do that never saw
    what `learn` does, which made the whole feature effectively invisible. It now
    prints the one command that does it, which is why `config set` exists.
    """
    _out("blurt %s -- learn" % __version__)
    _out("")
    _out("  Transcript journalling is OFF, so there is nothing to learn from.")
    _out("")
    _out("  What it would do: keep a record of your dictations on disk, and use it")
    _out("  to suggest 'dictionary' and 'initial_prompt' entries -- the two settings")
    _out("  that make blurt recognise YOUR names, jargon and acronyms. blurt cannot")
    _out("  learn those without a record; there is nothing to compare against.")
    _out("")
    _out("  What it costs: this is the one thing blurt writes to disk. The file is")
    _out("  created 0600 in a 0700 directory, it never leaves your machine, and")
    _out("  'blurt learn --forget' deletes it. Everything else blurt does keeps your")
    _out("  speech transient, which is why this is off until you say otherwise.")
    _out("")
    _out("  Turn it on with:")
    _out("")
    _out("      blurt config set history_enabled true")
    _out("")
    _out("  That writes %s for you." % default_config_path())
    _out("  Then dictate normally for a few days and run 'blurt learn' again.")
    _out("")
    _out("  To stop it again:            blurt config set history_enabled false")
    _out("  To delete what it wrote:     blurt learn --forget")
    return 0


def _learn_forget() -> int:
    """Delete the journal, and be honest about what deleting does not do."""
    path = _history.default_history_path()
    removed = _history.purge_history(path)
    if removed:
        _out("Deleted %s" % path)
        _out("")
        _out("  Note: this unlinks the file. On a copy-on-write filesystem (APFS is")
        _out("  one) that does not reliably destroy the underlying blocks, so it is")
        _out("  not a secure erase and is not claimed to be. FileVault is what")
        _out("  actually solves that.")
    else:
        _out("No journal to delete (%s)" % path)
    return 0


def _print_learn_report(
    report: "_learn.Report", path: pathlib.Path, cfg: Config
) -> None:
    _out("blurt %s -- learn" % __version__)
    _out("=" * 60)

    _out("")
    _out("JOURNAL")
    _out("  path       : %s" % path)
    _out(
        "  records    : %d (%d dictation, %d command)"
        % (report.records, report.dictation_records, report.assistant_records)
    )
    if report.span_days >= 0.01:
        _out("  span       : %.1f days" % report.span_days)
    _out("  audio      : %.1f minutes total" % (report.total_audio_seconds / 60.0))
    _out("  latency    : %.2fs median" % report.median_latency_seconds)
    if report.raw_available:
        share = (100.0 * report.cleanup_changed / report.records) if report.records else 0.0
        _out(
            "  cleanup    : changed %d of %d dictations (%.0f%%) at level '%s'"
            % (report.cleanup_changed, report.records, share, cfg.cleanup_level)
        )
    else:
        _out("  cleanup    : not measurable -- keep_raw_history is off")
        _out("               (only cleaned text was journalled, so the raw output")
        _out("                the engine actually produced is not recoverable)")

    _print_learn_suggestions(report)
    _print_learn_dictionary_health(report)


def _print_learn_suggestions(report: "_learn.Report") -> None:
    _out("")
    if not report.suggestions:
        _out("SUGGESTIONS")
        _out("  None. Either blurt is already transcribing you consistently, or")
        _out("  there is not enough history yet. Try 'blurt learn --min 2' to lower")
        _out("  the evidence threshold.")
        return

    high = len(report.high_confidence())
    _out(
        "SUGGESTIONS (%d: %d high confidence, %d worth a look)"
        % (len(report.suggestions), high, len(report.suggestions) - high)
    )

    for section, kind, blurb in (
        ("dictionary", "dictionary", "literal replacements applied during cleanup"),
        ("prompt", "prompt", "vocabulary hints passed to Whisper before it listens"),
    ):
        items = report.by_kind(kind)
        if not items:
            continue
        _out("")
        _out("  %s -- %s" % (section, blurb))
        for item in items:
            if kind == "dictionary":
                headline = "%s -> %s" % (item.key, item.value)
            else:
                headline = item.value
            _out("    [%-6s] %s" % (item.confidence, headline))
            _out("             %s" % item.reason)


def _print_learn_dictionary_health(report: "_learn.Report") -> None:
    if not (report.stale_dictionary_keys or report.noop_dictionary_keys):
        return

    _out("")
    _out("DICTIONARY HEALTH")
    if report.noop_dictionary_keys:
        _out("  rewrites nothing (key and value are the same):")
        for key in report.noop_dictionary_keys:
            _out("    %r" % key)
    if report.stale_dictionary_keys:
        _out("  never matched anything in your journal:")
        for key in report.stale_dictionary_keys:
            _out("    %r" % key)
        if not report.raw_available:
            _out("")
            _out("  Treat that list as unreliable: with keep_raw_history off, only")
            _out("  cleaned text was journalled, and a dictionary entry that IS")
            _out("  working has already rewritten itself out of the cleaned text.")


def _learn_apply(report: "_learn.Report", args: argparse.Namespace) -> int:
    """Collect approvals and merge them into the config file.

    A MERGE, NOT A ROUND TRIP -- the same rule ``config set`` follows, arrived at
    the same way. This used to call ``load_config()``, mutate the two fields it
    cares about, and hand the whole dataclass to ``save_config()``. That is
    exactly the destructive round trip :func:`_read_config_document` was written
    to describe, and it destroyed real things: a key from a newer blurt was
    deleted outright, and a value the loader rejects -- a typo'd
    ``sample_rate: 999999`` -- was overwritten with the default, erasing the
    evidence the user needed in order to find their own typo. Neither says
    anything on screen. The command then signed off with "nothing here is
    irreversible", which was false for both.

    It writes ``dictionary`` and ``initial_prompt`` because those are the two
    settings the user just approved changes to, and it writes each of them only
    if it actually changed. Every other byte of meaning in the file is carried
    across untouched, including keys this version of ``Config`` has never heard
    of.

    Reads the file rather than the config ``main`` resolved, which also keeps the
    older guarantee intact: the resolved one has any ``--model`` / ``--cleanup``
    overrides folded into it, those are for one run by construction, and
    persisting them would turn a flag someone passed to try something into a
    permanent setting. Reading the document cannot express an override at all,
    which is a stronger version of the same promise than reloading was.
    """
    if not report.suggestions:
        return 0

    unattended = bool(getattr(args, "yes", False))
    accepted = _collect_approvals(report, unattended)
    if accepted is None:
        return 1
    if not accepted:
        _out("")
        _out("Nothing accepted; your config is unchanged.")
        return 0

    path = default_config_path()
    document = _read_config_document(path)
    if document is None:
        # A config that exists and cannot be understood. _read_config_document
        # has already said so and said what to do about it. Refusing is the whole
        # point: the file is the only copy of a dictionary that may represent
        # months of someone's speech, and this command's own suggestions are
        # reproducible from the journal whereas that file is not.
        return 1

    # Taken from the document rather than from a loaded Config so that entries
    # the loader would have dropped survive being written back. `_pick_dictionary`
    # discards individual malformed entries -- correct when reading, data loss
    # when the result is what gets saved. Both merge helpers already tolerate a
    # value of the wrong shape, so the raw JSON can be handed to them directly.
    raw_dictionary = document.get("dictionary")
    raw_prompt = document.get("initial_prompt")
    before_dictionary = dict(raw_dictionary) if isinstance(raw_dictionary, dict) else {}
    before_prompt = raw_prompt.strip() if isinstance(raw_prompt, str) else ""

    dictionary = _learn.merged_dictionary(raw_dictionary, accepted)
    prompt = _learn.merged_prompt(raw_prompt, accepted)

    added_entries = len(dictionary) - len(before_dictionary)
    prompt_changed = prompt != before_prompt

    if not added_entries and not prompt_changed:
        _out("")
        _out("Everything accepted was already covered; your config is unchanged.")
        return 0

    # Only the keys that moved. A `dictionary` that gained nothing is not
    # rewritten to an equal value, because "equal" is a claim about the loader's
    # view of it and this command does not have to make that claim.
    changes: Dict[str, Any] = {}
    if added_entries:
        changes["dictionary"] = dictionary
    if prompt_changed:
        changes["initial_prompt"] = prompt

    if not _merge_into_config_file(document, changes, path):
        return 1

    _out("")
    _out("Wrote %s" % path)
    if added_entries:
        _out("  dictionary     : %d new entr%s"
             % (added_entries, "y" if added_entries == 1 else "ies"))
    if prompt_changed:
        words = len(prompt.split())
        _out("  initial_prompt : now %d word%s" % (words, "" if words == 1 else "s"))
        if words >= _learn.MAX_PROMPT_WORDS:
            _out("                   (at the %d-word budget; further vocabulary will"
                 % _learn.MAX_PROMPT_WORDS)
            _out("                    be skipped until you prune it by hand)")
    _out("")
    _out("These take effect the next time you start blurt.")
    _out(
        "Only %s changed; everything else in %s is exactly as you left it."
        % (" and ".join(sorted(changes)), path)
    )
    _out("Undo by editing that file -- entries were added, none replaced.")
    return 0


def _collect_approvals(
    report: "_learn.Report", unattended: bool
) -> Optional[List["_learn.Suggestion"]]:
    """Return the accepted suggestions, or None if the user cannot be asked.

    ``--yes`` takes the high-confidence set and nothing else. The medium ones are
    exactly the findings that can be wrong in a way the user would not notice, so
    "accept everything without looking" is not offered for them at any flag.
    """
    if unattended:
        high = report.high_confidence()
        _out("")
        _out("--yes: accepting %d high-confidence suggestion(s)." % len(high))
        skipped = len(report.suggestions) - len(high)
        if skipped:
            _out(
                "       Skipping %d that need a human -- rerun without --yes to see them."
                % skipped
            )
        return high

    if not sys.stdin.isatty():
        _err("")
        _err("blurt: --apply needs a terminal to ask you about each suggestion.")
        _err("  Non-interactively, use --yes to accept the high-confidence ones:")
        _err("      blurt learn --apply --yes")
        return None

    _out("")
    _out("=" * 60)
    _out("Reviewing %d suggestion(s)." % len(report.suggestions))
    _out("  y = accept   n = skip (default)   a = accept all remaining   q = stop")
    _out("")

    accepted: List["_learn.Suggestion"] = []
    take_rest = False
    for index, item in enumerate(report.suggestions, 1):
        if item.kind == "dictionary":
            headline = "replace %r with %r everywhere" % (item.key, item.value)
        else:
            headline = "add %r to the Whisper vocabulary hint" % (item.value,)

        _out("[%d/%d] %s" % (index, len(report.suggestions), headline))
        _out("       %s -- %s confidence" % (item.reason, item.confidence))

        if take_rest:
            _out("       accepted")
            accepted.append(item)
            _out("")
            continue

        answer = _prompt_choice("       accept? [y/N/a/q] ")
        if answer == "q":
            _out("")
            _out("Stopped. Keeping the %d already accepted." % len(accepted))
            break
        if answer == "a":
            take_rest = True
            accepted.append(item)
        elif answer == "y":
            accepted.append(item)
        _out("")

    return accepted


def _prompt_choice(prompt: str) -> str:
    """Read one lowercase answer. EOF and Ctrl+C both mean "stop", not "yes"."""
    try:
        raw = input(prompt)
    except (EOFError, KeyboardInterrupt):
        _out("")
        return "q"
    answer = (raw or "").strip().lower()
    if answer in ("y", "yes"):
        return "y"
    if answer in ("a", "all"):
        return "a"
    if answer in ("q", "quit"):
        return "q"
    return "n"


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

    command = getattr(args, "command", None) or "run"

    # `config set` is the one command that must not read the config file through
    # load_config, for two reasons. It never uses the resolved config -- it merges
    # into the raw JSON itself -- so the load would be pure waste, and it is waste
    # with output: load_config warns about every field it fell back on, so a user
    # with one imperfect setting saw the same warnings printed twice for one
    # command, once here and once inside the subcommand. It also has a side
    # effect, since load_config renames an unparseable config to `.bak` on its way
    # past, and `set` refuses to touch a file it could not read.
    #
    # The overrides are still parsed and validated, just against the defaults, so
    # `blurt --hotkey fn config set ...` is still refused rather than ignored.
    setting_config = command == "config" and getattr(args, "config_action", None) == "set"

    # `learn --apply` merges into the same raw JSON document `set` does, and owes
    # the user the same refusal: a config file blurt could not parse must not be
    # replaced, only reported. It cannot get that for free the way `set` does,
    # because it genuinely has to load -- the report is built from
    # `history_enabled`, `dictionary` and `initial_prompt`.
    #
    # And that load is destructive to the evidence. `load_config` renames an
    # unparseable config to `.bak` on its way past, so by the time `_learn_apply`
    # reads the document the file is GONE, `_read_config_document` sees a missing
    # file, calls it a fresh install, and starts from the serialised defaults --
    # writing a brand-new config over a user whose only copy had just been moved
    # out from under them, and reporting success while doing it. Checking here,
    # before anything can move the file, is the only place the answer is still
    # true. `--forget` is excluded because it never touches the config at all.
    applying_learn = (
        command == "learn"
        and bool(getattr(args, "apply", False))
        and not bool(getattr(args, "forget", False))
    )
    if applying_learn and _read_config_document(default_config_path()) is None:
        return 1

    cfg = _apply_overrides(Config() if setting_config else load_config(), args)

    if command == "doctor":
        return _cmd_doctor(cfg, args)
    if command == "bench":
        return _cmd_bench(cfg, args)
    if command == "config":
        return _cmd_config(cfg, args)
    if command == "learn":
        return _cmd_learn(cfg, args)
    return _cmd_run(cfg, args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        # Ctrl+C before the app installs its own handler. Exit quietly: a
        # traceback here would suggest a crash where the user simply quit.
        _err("")
        raise SystemExit(130)
