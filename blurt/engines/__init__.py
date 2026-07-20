"""Engine registry: decide which ASR backend blurt will actually use.

Two backends exist, and they are not equals:

  * ``faster-whisper`` -- the real engine. Measured on the Intel floor machine,
    accurate, needs a one-time model download.
  * ``apple-speech``   -- EXPERIMENTAL. On-device SFSpeechRecognizer. No model
    download, but unproven for us, and unusable from a process without an
    Info.plist usage description (see that module's docstring).

``auto`` therefore means "faster-whisper if it can possibly run, apple-speech
only if it cannot".

The rule that shapes this module: **never silently substitute a backend the user
did not ask for.** A user who wrote ``"engine": "apple-speech"`` in their config
and gets Whisper instead has had a privacy and latency decision quietly reversed
on their behalf. So an explicit engine that is unavailable raises
:class:`NoEngineAvailable` with the actual reason and the actual fix, and only
``auto`` is permitted to choose.

Probing is total: :func:`available_engines` catches everything, because it runs
during startup and before the menu bar exists. An engine that explodes when
asked whether it works is simply an engine that does not work.

What can go wrong on macOS:

  - Importing a backend module can fail in the dynamic loader rather than in
    Python (a wheel built for a newer macOS -- exactly how pywhispercpp failed
    here), which surfaces as ``OSError``, not ``ImportError``. Every probe
    catches ``BaseException`` for that reason.
  - Probing ``apple-speech`` may trigger the macOS speech-permission dialog when
    blurt is running from a proper .app bundle. That is intended -- permission
    has to be requested sometime -- but it means :func:`available_engines` is not
    guaranteed to be side-effect-free in a bundled build.

Python 3.9 floor: lazy annotations, typing.List / typing.Optional / typing.Tuple
only, no PEP 604 unions, no builtin generics evaluated at runtime.
"""

from __future__ import annotations

import importlib
import logging
from typing import List, Optional, Tuple

from ..config import Config
from ..types import ASREngine, Hardware

__all__ = [
    "NoEngineAvailable",
    "ENGINE_NAMES",
    "available_engines",
    "select_engine",
]

_log = logging.getLogger(__name__)


class NoEngineAvailable(RuntimeError):
    """Raised when no usable ASR backend could be selected.

    The message is written to be shown to a human as-is: it names every engine
    that was tried and why each one was rejected. Subclasses ``RuntimeError``
    because callers that already guard startup against runtime failures should
    catch it without extra handling.
    """


# Preference order for ``engine="auto"``. faster-whisper first: it is the proven
# one. apple-speech is the fallback for machines that cannot run it at all.
ENGINE_NAMES: Tuple[str, ...] = ("faster-whisper", "apple-speech")

# Engine name -> (module suffix, class name). Resolved lazily through importlib
# so that one broken backend cannot prevent `import blurt.engines`.
_ENGINE_MODULES = {
    "faster-whisper": (".faster_whisper_engine", "FasterWhisperEngine"),
    "apple-speech": (".apple_speech_engine", "AppleSpeechEngine"),
}


def _construct(
    name: str,
    cfg: Optional[Config],
    hw: Optional[Hardware],
) -> Tuple[Optional[ASREngine], Optional[str]]:
    """Build an engine instance. Returns (engine, None) or (None, reason).

    Construction is required to be cheap -- no imports of heavy dependencies, no
    model loading -- so this is safe to call for engines we end up not using.
    """
    entry = _ENGINE_MODULES.get(name)
    if entry is None:
        return None, f"{name}: not a known engine"

    module_suffix, class_name = entry
    try:
        module = importlib.import_module(module_suffix, __name__)
        engine_class = getattr(module, class_name)
        return engine_class(cfg, hw), None
    except BaseException as exc:  # noqa: BLE001 - loader failures are not ImportError
        return None, f"{name}: backend module could not be loaded ({type(exc).__name__}: {exc})"


def _probe(
    name: str,
    cfg: Optional[Config],
    hw: Optional[Hardware],
) -> Tuple[Optional[ASREngine], Optional[str]]:
    """Ask one engine whether it can run here.

    Returns (engine, None) when usable and (None, reason) when not. Never
    raises: an engine whose ``is_available`` throws is treated as unavailable,
    since a backend that cannot answer that question cannot be trusted to
    transcribe either.
    """
    engine, reason = _construct(name, cfg, hw)
    if engine is None:
        return None, reason

    try:
        usable = bool(engine.is_available())
    except BaseException as exc:  # noqa: BLE001 - is_available must never escape
        return None, f"{name}: availability check raised ({type(exc).__name__}: {exc})"

    if usable:
        return engine, None

    detail = getattr(engine, "unavailable_reason", None) or "no reason given"
    return None, f"{name}: {detail}"


def available_engines() -> List[str]:
    """Names of the engines that can actually run on this machine, right now.

    Ordered by preference (the ``auto`` order), so ``available_engines()[0]`` is
    what ``auto`` would pick. Returns an empty list on a machine where nothing
    works; that is information, not an error, so this never raises.

    Uses default configuration for the probe. Availability does not depend on
    model size or thread count, so this answers the same question
    :func:`select_engine` will ask.
    """
    usable: List[str] = []
    for name in ENGINE_NAMES:
        engine, reason = _probe(name, None, None)
        if engine is not None:
            usable.append(name)
        else:
            _log.debug("engine unavailable -- %s", reason)
    return usable


def _auto_select(cfg: Config, hw: Hardware) -> ASREngine:
    """Pick the best available engine, or raise with every rejection listed."""
    reasons: List[str] = []

    for name in ENGINE_NAMES:
        engine, reason = _probe(name, cfg, hw)
        if engine is not None:
            _log.info("selected ASR engine: %s", name)
            return engine
        _log.debug("engine unavailable -- %s", reason)
        if reason:
            reasons.append(reason)

    raise NoEngineAvailable(
        "blurt could not find a usable speech recognition engine.\n"
        "Engines tried, and why each was rejected:\n"
        + "\n".join("  - " + reason for reason in reasons)
        + "\n\nMost likely fix: install faster-whisper, which is the engine blurt "
        "is built around:\n"
        "    python3 -m pip install faster-whisper"
    )


def _explicit_select(name: str, cfg: Config, hw: Hardware) -> ASREngine:
    """Honour an explicitly configured engine, or raise. Never substitutes.

    The user named this backend for a reason -- privacy, latency, or avoiding a
    download. Quietly handing them a different one would reverse that decision
    without telling them, so an unavailable choice is an error.
    """
    if name not in _ENGINE_MODULES:
        raise NoEngineAvailable(
            f"Unknown engine {name!r} in your blurt config.\n"
            f"  Valid values: auto, {', '.join(ENGINE_NAMES)}"
        )

    engine, reason = _probe(name, cfg, hw)
    if engine is not None:
        _log.info("selected ASR engine: %s (explicitly configured)", name)
        return engine

    alternatives = [other for other in available_engines() if other != name]
    if alternatives:
        suggestion = (
            "  Available instead: "
            + ", ".join(alternatives)
            + f".\n  blurt will NOT switch for you: you asked for {name!r}, and "
            "changing engines changes latency and privacy behaviour. Edit "
            '"engine" in your blurt config to switch deliberately.'
        )
    else:
        suggestion = "  No other engine is usable on this machine either."

    raise NoEngineAvailable(
        f"The configured engine {name!r} is not available.\n"
        f"  Reason: {reason}\n" + suggestion
    )


def select_engine(cfg: Config, hw: Hardware) -> ASREngine:
    """Choose the ASR backend for this run.

    ``cfg.engine == "auto"`` picks the first usable engine in preference order
    (faster-whisper, then the experimental apple-speech). Any other value is
    honoured exactly or raises :class:`NoEngineAvailable` -- there is no silent
    fallback away from an explicit choice.

    Returns an UNLOADED engine. The caller is responsible for calling ``load()``,
    which is where the slow work and the one-time model download happen; keeping
    them separate lets the caller show a "preparing..." state instead of
    freezing during selection.

    Raises :class:`NoEngineAvailable` if nothing usable was found.
    """
    name = (cfg.engine or "auto").strip().lower()

    if name == "auto":
        return _auto_select(cfg, hw)
    return _explicit_select(name, cfg, hw)
