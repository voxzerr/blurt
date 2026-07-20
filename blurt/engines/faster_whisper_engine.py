"""faster-whisper ASR backend: CTranslate2 Whisper, int8, on the CPU.

This is blurt's primary engine and the one measured on the Intel floor machine
(i7-7567U, 2 physical cores, macOS 13.7.8, Apple system Python 3.9.6). It runs
``device="cpu"`` with ``compute_type="int8"``, which is the configuration every
published timing in this project was produced with. Do not switch it to float16
or "auto" to chase quality: int8 is what fits in the thermal budget of a 2-core
Intel laptop, and "auto" silently picks something else on Apple Silicon.

The model is loaded ONCE, in :meth:`FasterWhisperEngine.load`, and kept resident
for the life of the process. :meth:`FasterWhisperEngine.transcribe` never loads.
This is not an optimisation, it is a correctness requirement for the product:
constructing a ``WhisperModel`` takes seconds, and paying that on the first
utterance makes the app feel broken exactly when the user is deciding whether to
trust it.

Latency, measured (int8, 4 threads, floor machine):

    tiny.en   2.06s for 13.8s audio  |  2.77s for  2.7s audio
    base.en   4.06s for 13.8s audio  |  5.24s for  2.7s audio
    small.en  8.27s for 13.8s audio  | 24.50s for  2.7s audio  <- unusable

Read those numbers carefully: the SHORT clip is not faster. Whisper pads every
input to a 30-second window, so latency is roughly constant regardless of how
briefly the user speaks. Anything in the UI that promises "quick for short
phrases" is lying.

What can go wrong on macOS:

  - The one-time model download. First use of a given model size fetches weights
    from HuggingFace. That is the ONLY network call blurt makes, and it is
    isolated in :func:`_load_model` -- we first try strictly offline
    (``local_files_only=True``) so that a cached model never touches the network
    at all, and only reach for the network when the cache genuinely misses. A
    failure there means offline, captive portal, or no disk space, and produces
    a message that says so.

  - Importing ``faster_whisper`` pulls in ``ctranslate2``, which loads a native
    shared library. A wheel built for a newer macOS fails here with ``OSError``
    (a broken ``@rpath`` reference), not ``ImportError``. This is exactly how
    pywhispercpp failed on the floor machine, so :func:`_import_faster_whisper`
    catches ``BaseException`` and reports the real error instead of pretending
    the package is merely missing.

  - Thread oversubscription. ctranslate2 and the PortAudio callback compete for
    2 physical cores. Threads come from ``hardware.recommend_threads`` (4 on the
    floor machine) unless the user pinned ``cpu_threads`` in config.

  - Sample rate. Whisper is a 16 kHz model. If audio arrives at another rate we
    resample rather than refuse, because refusing means the user cannot dictate
    at all. See :func:`_resample` for the caveat.

Python 3.9 floor: lazy annotations, typing.Optional / typing.List only, no PEP
604 unions, no builtin generics evaluated at runtime.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np

from .. import hardware as _hardware
from ..config import Config
from ..types import ASREngine, Hardware

if TYPE_CHECKING:  # pragma: no cover - resolved by type checkers only
    import numpy

__all__ = ["FasterWhisperEngine", "ENGINE_NAME"]

_log = logging.getLogger(__name__)

ENGINE_NAME = "faster-whisper"

# Whisper's native rate. Everything else has to be resampled to this.
_WHISPER_SAMPLE_RATE = 16000

# beam_size=1 (greedy). Every timing in this project was measured with it, and
# on a 2-core machine a wider beam costs more than the accuracy is worth.
_BEAM_SIZE = 1

# Cached result of importing faster_whisper, so that repeated is_available()
# calls cost nothing. None means "not attempted yet".
_import_lock = threading.Lock()
_import_result: Optional[Any] = None      # the module, on success
_import_error: Optional[str] = None       # human-readable reason, on failure


def _describe_import_failure(exc: BaseException) -> str:
    """Explain a failed ``import faster_whisper`` in terms the user can act on."""
    return (
        "The faster-whisper package could not be loaded.\n"
        f"  Underlying error: {type(exc).__name__}: {exc}\n"
        "  Fix: install it into the interpreter running blurt "
        f"({sys.executable}):\n"
        f"    {sys.executable} -m pip install faster-whisper\n"
        "  If it is already installed, the failure is most likely ctranslate2's "
        "native library being built for a newer macOS than this machine."
    )


def _import_faster_whisper() -> Any:
    """Import ``faster_whisper`` once, caching success and failure alike.

    Returns the module, or raises ``ImportError`` carrying a readable message.
    Catches ``BaseException`` on purpose: a native-library mismatch surfaces as
    ``OSError`` from the loader, not ``ImportError``, and a bad wheel has been
    known to raise almost anything.
    """
    global _import_result, _import_error

    with _import_lock:
        if _import_result is not None:
            return _import_result
        if _import_error is not None:
            raise ImportError(_import_error)

        try:
            import faster_whisper  # noqa: PLC0415 - deliberately deferred
        except BaseException as exc:  # noqa: BLE001 - imports fail many ways
            _import_error = _describe_import_failure(exc)
            raise ImportError(_import_error) from exc

        _import_result = faster_whisper
        return faster_whisper


def _resample(pcm: "numpy.ndarray", sample_rate: int) -> "numpy.ndarray":
    """Resample mono float32 audio to 16 kHz.

    Linear interpolation, no anti-aliasing filter. That is a real quality
    compromise when downsampling (48 kHz content above 8 kHz folds back as
    aliasing), accepted because the alternative is a scipy dependency that is
    not verified on the floor machine, and because blurt records at 16 kHz by
    default so this path is the exception rather than the rule.
    """
    if sample_rate == _WHISPER_SAMPLE_RATE or pcm.size == 0:
        return pcm

    duration = pcm.shape[0] / float(sample_rate)
    target_length = int(round(duration * _WHISPER_SAMPLE_RATE))
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)

    source_positions = np.linspace(0.0, pcm.shape[0] - 1, num=target_length)
    resampled = np.interp(source_positions, np.arange(pcm.shape[0]), pcm)
    return resampled.astype(np.float32, copy=False)


def _normalise_pcm(pcm: "numpy.ndarray", sample_rate: int) -> "numpy.ndarray":
    """Coerce whatever the recorder handed us into 1-D contiguous float32 @ 16 kHz.

    Defensive on purpose: this is the boundary between our audio code and a
    native library that will segfault rather than complain if it is handed a
    non-contiguous array or an unexpected dtype.
    """
    array = np.asarray(pcm)

    # Fold a (frames, channels) capture down to mono. sounddevice hands back 2-D
    # even for a single channel unless it is explicitly squeezed.
    if array.ndim > 1:
        array = array.mean(axis=1)
    array = np.reshape(array, (-1,))

    if array.dtype != np.float32:
        array = array.astype(np.float32, copy=False)

    # NaN/inf reach ctranslate2 as garbage rather than an error. Scrub them.
    if not np.all(np.isfinite(array)):
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    array = _resample(array, sample_rate)
    return np.ascontiguousarray(array, dtype=np.float32)


def _describe_download_failure(exc: BaseException, model_size: str) -> str:
    """Explain why fetching model weights failed, in terms the user can act on."""
    return (
        f"blurt could not download the {model_size!r} speech model.\n"
        f"  Underlying error: {type(exc).__name__}: {exc}\n"
        "  This is the one and only time blurt uses the network, and it needs "
        "to succeed once before dictation works offline forever after.\n"
        "  Likely causes: no internet connection, a captive portal or proxy "
        "intercepting HTTPS, a firewall blocking huggingface.co, or not enough "
        "free disk space in ~/.cache/huggingface."
    )


def _load_model(module: Any, model_size: str, cpu_threads: int) -> Any:
    """Construct the ``WhisperModel``, isolating the one permitted network call.

    Two attempts, deliberately separated so the network case is explicit and
    visible rather than an invisible side effect of construction:

      1. ``local_files_only=True`` -- pure cache hit, guaranteed no network. This
         is the steady-state path on every launch after the first.
      2. Only if that misses: log loudly that we are downloading, then allow the
         fetch from HuggingFace.

    Raises ``RuntimeError`` with a human-readable message if the download fails.
    """
    whisper_model = module.WhisperModel

    try:
        return whisper_model(
            model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=cpu_threads,
            # No network at all: succeed from cache or fail immediately.
            local_files_only=True,
        )
    except Exception as exc:  # noqa: BLE001 - a cache miss raises several types
        _log.info(
            "model %r is not in the local cache (%s); a one-time download is needed",
            model_size,
            exc,
        )

    _log.warning(
        "Downloading the %r speech model from HuggingFace. This happens once, "
        "needs an internet connection, and can take a few minutes. Every later "
        "launch is fully offline.",
        model_size,
    )

    try:
        return whisper_model(
            model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=cpu_threads,
            local_files_only=False,
        )
    except BaseException as exc:  # noqa: BLE001 - network/disk fail many ways
        raise RuntimeError(_describe_download_failure(exc, model_size)) from exc


class FasterWhisperEngine(ASREngine):
    """Whisper via CTranslate2, int8 on the CPU. blurt's primary engine.

    Lifecycle is the one in :class:`blurt.types.ASREngine`::

        engine = FasterWhisperEngine(cfg, hw)
        if engine.is_available():
            engine.load()          # slow, once, at startup
            text = engine.transcribe(pcm, 16000)   # fast path, never loads

    Not thread-safe for concurrent :meth:`transcribe` calls. blurt dictates one
    utterance at a time, so a lock would only hide a bug in the caller; the load
    path IS locked, because startup and a first hotkey press can race.
    """

    name = ENGINE_NAME

    def __init__(self, cfg: Optional[Config] = None, hw: Optional[Hardware] = None) -> None:
        """Store configuration. Does no work: no import, no probe, no model.

        ``hw`` is resolved lazily (see :meth:`_hardware`) so that constructing an
        engine merely to ask :meth:`is_available` never pays for a hardware probe.
        """
        self._cfg = cfg if cfg is not None else Config()
        self._hw = hw
        self._model: Optional[Any] = None
        self._loaded_model_size: Optional[str] = None
        self._load_lock = threading.Lock()
        self._warned_about_rate = False
        # Read by blurt.engines to explain why a backend was passed over.
        self.unavailable_reason: Optional[str] = None

    # -- configuration -----------------------------------------------------

    def _hardware(self) -> Hardware:
        """Detect hardware on first use and cache it. Never raises."""
        if self._hw is None:
            self._hw = _hardware.detect()
        return self._hw

    def resolve_model(self) -> str:
        """The model size this engine will actually load.

        ``model="auto"`` defers to ``hardware.recommend_model``, which refuses to
        hand small.en to a slow machine -- 24.5s for a 2.7s utterance is not a
        product.
        """
        configured = (self._cfg.model or "auto").strip()
        if configured and configured != "auto":
            return configured
        return _hardware.recommend_model(self._hardware())

    def resolve_threads(self) -> int:
        """The ctranslate2 thread count. ``cpu_threads=0`` in config means auto."""
        configured = self._cfg.cpu_threads
        if isinstance(configured, int) and configured > 0:
            return configured
        return _hardware.recommend_threads(self._hardware())

    # -- ASREngine ---------------------------------------------------------

    def is_available(self) -> bool:
        """True if ``faster_whisper`` imports here. Cheap, total, never raises.

        Deliberately performs the real import rather than a ``find_spec`` probe:
        the failure mode we most need to catch is a native ctranslate2 library
        that is present but unloadable, and only an actual import surfaces that.
        The result is cached, so this costs once per process.

        Says nothing about whether model weights are on disk -- that question
        belongs to :meth:`load`, because answering it here would mean either a
        network call or a false negative on a machine that just needs to fetch.
        """
        try:
            _import_faster_whisper()
        except ImportError as exc:
            self.unavailable_reason = str(exc)
            return False

        self.unavailable_reason = None
        return True

    def load(self) -> None:
        """Load the model and keep it resident. Slow, once, at startup.

        Idempotent: a second call is a no-op rather than a reload. Raises
        ``ImportError`` if faster-whisper cannot be imported and ``RuntimeError``
        if the weights cannot be obtained; both carry messages fit to show a
        human as-is.
        """
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return

            module = _import_faster_whisper()
            model_size = self.resolve_model()
            cpu_threads = self.resolve_threads()

            _log.info(
                "loading faster-whisper model=%s device=cpu compute_type=int8 threads=%d",
                model_size,
                cpu_threads,
            )
            model = _load_model(module, model_size, cpu_threads)

            self._model = model
            self._loaded_model_size = model_size
            _log.info("faster-whisper model %s ready", model_size)

    def transcribe(self, pcm: "numpy.ndarray", sample_rate: int) -> str:
        """Transcribe mono float32 PCM. Never loads the model.

        The array goes straight to faster-whisper, which accepts a numpy array
        natively -- no temp WAV, no ffmpeg, which is what makes this workable on
        a machine with no Homebrew.

        Returns "" for silence. Silence is an ordinary outcome (the user tapped
        the hotkey by accident), not an error.
        """
        model = self._model
        if model is None:
            raise RuntimeError(
                "FasterWhisperEngine.transcribe() called before load(). The model "
                "must be loaded once at startup; loading it here would add seconds "
                "to the first dictation."
            )

        if sample_rate != _WHISPER_SAMPLE_RATE and not self._warned_about_rate:
            self._warned_about_rate = True
            _log.warning(
                "audio arrived at %d Hz; resampling to %d Hz for Whisper. Recording "
                "at 16 kHz avoids this.",
                sample_rate,
                _WHISPER_SAMPLE_RATE,
            )

        audio = _normalise_pcm(pcm, sample_rate)
        if audio.size == 0:
            return ""

        segments, _info = model.transcribe(
            audio,
            beam_size=_BEAM_SIZE,
            # Whisper's context carryover makes short utterances degenerate into
            # runaway repetition loops ("thank you thank you thank you..."), and
            # short utterances are the entire use case here.
            condition_on_previous_text=False,
            # Trims leading/trailing silence so the model is not asked to
            # hallucinate words out of room tone.
            vad_filter=True,
            language=self._language_hint(),
        )

        # `segments` is a lazy generator: the actual work happens as it is
        # consumed, so this loop is where the seconds are spent.
        parts: List[str] = []
        for segment in segments:
            text = getattr(segment, "text", "") or ""
            if text.strip():
                parts.append(text.strip())

        return " ".join(parts).strip()

    def unload(self) -> None:
        """Drop the model. Safe if load() never ran or failed partway."""
        self._model = None
        self._loaded_model_size = None

    # -- internals ---------------------------------------------------------

    def _language_hint(self) -> Optional[str]:
        """"en" for English-only models, None (autodetect) otherwise.

        The ``.en`` checkpoints are English-only and cannot detect language;
        naming it explicitly skips the detection pass. A multilingual model gets
        None so the user can dictate in whatever they speak.
        """
        model_size = getattr(self, "_loaded_model_size", None) or self.resolve_model()
        return "en" if model_size.endswith(".en") else None
