"""Shared data types and the ASR engine interface for blurt.

This module is the contract every other module imports from. It is deliberately
dependency-free: importing it must never pull in numpy, faster-whisper, pyobjc,
or anything else that can fail to load. If this module cannot import, nothing
can, so it stays boring on purpose.

What can go wrong on macOS:
  - Nothing here touches the OS. That is the point. Hardware probing lives in
    blurt.hardware, audio in blurt.audio, and so on. Keep it that way: if you
    add a runtime import of a native extension to this file, a bad wheel on the
    user's machine becomes a total startup failure instead of one broken engine.
  - numpy is referenced only as a type annotation. It is imported under
    TYPE_CHECKING so that type checkers resolve `numpy.ndarray` while the
    runtime never imports numpy just to describe a signature.

Python 3.9 floor: `from __future__ import annotations` keeps every annotation
lazy, and we use typing.Tuple / typing.Optional rather than PEP 585/604 forms.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:  # pragma: no cover - resolved by type checkers only
    import numpy


@dataclass(frozen=True)
class Hardware:
    """A description of the machine we are actually running on.

    `arch` is the REAL silicon, corrected for Rosetta 2. A Python interpreter
    translated by Rosetta reports platform.machine() == "x86_64" even on an M-series
    Mac, so trusting platform.machine() alone would mislabel an Apple Silicon
    machine as Intel and hand it the wrong model and thread count.
    """

    arch: str  # "arm64" or "x86_64" (real hardware, Rosetta-corrected)
    is_apple_silicon: bool
    under_rosetta: bool
    cpu_brand: str
    physical_cores: int
    ram_gb: float
    macos_version: Tuple[int, int, int]
    tier: str  # "fast" | "medium" | "slow"


@dataclass(frozen=True)
class Transcript:
    """One completed dictation.

    `raw` is preserved verbatim so a bad cleanup pass can never destroy what the
    engine actually heard. `cleaned` is what we type into the focused app.

    Note on `latency_seconds` vs `audio_seconds`: Whisper pads every input to a
    30-second window, so latency is roughly constant and does NOT shrink for
    short utterances. Do not compute or display a "realtime factor" from these
    two fields as if brief phrases were cheap -- on the Intel floor machine a
    2.7s clip can take longer than a 13.8s one.
    """

    raw: str  # exactly what the ASR engine produced
    cleaned: str  # after cleanup pass
    audio_seconds: float
    engine: str
    latency_seconds: float


class ASREngine(abc.ABC):
    """Interface every speech backend implements.

    Lifecycle: is_available() -> load() -> transcribe() ... -> unload().

    Implementations must treat is_available() as cheap and total: it answers
    "could this engine work here?" without loading models, without hitting the
    network, and without raising. Callers use it to pick a backend at startup,
    so an exception there takes the whole app down.

    load() is where the expensive and failure-prone work belongs (reading model
    files, allocating native contexts, the one-time model download). It may
    raise; the engine selector is expected to catch and fall back.
    """

    name: str

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if this engine can run on this machine, right now.

        Must not raise, must not download anything, must not load a model.
        """

    @abc.abstractmethod
    def load(self) -> None:
        """Prepare the engine for transcription.

        May be slow and may raise. Safe to call more than once; implementations
        should make repeat calls a no-op rather than reloading.
        """

    @abc.abstractmethod
    def transcribe(self, pcm: "numpy.ndarray", sample_rate: int) -> str:
        """Turn mono float32 PCM into text.

        `pcm` is a 1-D float32 array in [-1.0, 1.0]; `sample_rate` is its rate in
        Hz. Returns the engine's raw output, uncleaned. Returns "" when nothing
        intelligible was heard -- silence is a normal outcome, not an error.
        """

    def unload(self) -> None:
        """Release models and native resources.

        Optional. The default is a no-op so engines with nothing to free do not
        need to implement it. Must be safe to call when load() was never called
        or when load() failed partway through.
        """
        return None
