"""Apple SFSpeechRecognizer -- INVESTIGATED AND REJECTED. Do not revive this.

This module deliberately contains no working implementation. It exists so that
the next person who thinks "why don't we just use Apple's built-in speech
recognition? It ships with macOS and needs no model download" finds the answer
here instead of spending a day rediscovering it.

It is a genuinely attractive idea. It is also a trap, for three independent
reasons, any one of which is disqualifying.

---------------------------------------------------------------------------
1. IT CAN SILENTLY SEND AUDIO TO APPLE ON INTEL MACS
---------------------------------------------------------------------------
This is the one that actually matters, and it makes this a hard rejection rather
than a "maybe later".

``SFSpeechRecognizer`` exposes ``requiresOnDeviceRecognition``, and on the Intel
floor machine ``supportsOnDeviceRecognition()`` cheerfully returns True. That
looks like a guarantee. It is not one.

Intel Macs have no Neural Engine, and the on-device speech assets are frequently
not installed at all -- verified absent on the 2017 floor machine. In that state
the framework commonly falls back to SERVER-BASED recognition: the user's audio
is uploaded to Apple.

blurt's entire promise is that nothing leaves your machine. A backend that can
quietly violate that promise, under a flag literally named on-device, cannot
ship -- not as a default, not as an option, not behind a warning. A privacy
guarantee with an exception is not a privacy guarantee.

---------------------------------------------------------------------------
2. IT CANNOT BE DELIVERED FROM A pip-INSTALLED TOOL
---------------------------------------------------------------------------
Even setting privacy aside, the delivery model is incompatible with ours.

Invoked from a plain CLI process, TCC kills the process outright:

    exit 134 (SIGABRT), TCC_CRASHING_DUE_TO_PRIVACY_VIOLATION

That is not a permission prompt that can be accepted -- the process is
terminated. Wrapping it in an ad-hoc-signed ``.app`` bundle carrying a valid
``NSSpeechRecognitionUsageDescription`` still aborted. Making it work requires a
properly signed bundle, LaunchServices invocation, and a one-time consent click:
shipping a signed Mac application, which is a fundamentally different product
than ``pip install blurt``.

That path is real, but it belongs to the native rewrite. See ``docs/SWIFT-V2.md``.

---------------------------------------------------------------------------
3. FIRST USE MAY REQUIRE A NETWORK DOWNLOAD ANYWAY
---------------------------------------------------------------------------
Where the on-device SpeechRecognitionCore assets are missing, macOS fetches them
on demand -- so the headline advantage, "no model download", does not reliably
hold. faster-whisper's one-time, explicit, inspectable download is the better
bargain: it happens once, we control when, and we can tell the user exactly what
is being fetched.

---------------------------------------------------------------------------
WHAT TO USE INSTEAD
---------------------------------------------------------------------------
``blurt.engines.faster_whisper_engine`` -- measured working on the Intel floor
machine, runs on both architectures from a single pip install, and never touches
the network after the initial model fetch.

If you want Apple-native speech recognition, the correct vehicle is a signed
``.app`` -- and at that point Parakeet-on-ANE is both faster and more accurate
than SFSpeechRecognizer anyway. Again: ``docs/SWIFT-V2.md``.

Python 3.9 floor: lazy annotations, typing only, no PEP 604 unions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ..types import ASREngine

if TYPE_CHECKING:  # pragma: no cover
    import numpy


#: Why this engine is permanently unavailable. Surfaced by ``blurt doctor`` so
#: the rejection is visible to users rather than buried in a source comment.
REJECTION_REASON = (
    "Apple SFSpeechRecognizer is not supported: on Intel Macs it can silently "
    "fall back to Apple's servers, which would break blurt's guarantee that "
    "audio never leaves your machine. It also cannot run from a pip-installed "
    "CLI (TCC terminates the process with SIGABRT). See "
    "blurt/engines/apple_speech_engine.py for the full findings."
)


class AppleSpeechEngine(ASREngine):
    """Permanently unavailable. Retained as documentation, not as a backend."""

    name = "apple-speech"

    def __init__(
        self, cfg: Optional[Any] = None, hw: Optional[Any] = None
    ) -> None:
        # Signature matches the registry's ``engine_class(cfg, hw)`` call so this
        # stub reports itself unavailable cleanly, rather than blowing up with a
        # TypeError that reads like a bug in the registry.
        self._cfg = cfg
        self._hw = hw
        self.unavailable_reason = REJECTION_REASON

    def is_available(self) -> bool:
        """Always False. See the module docstring for the three reasons."""
        return False

    def load(self) -> None:
        raise RuntimeError(REJECTION_REASON)

    def transcribe(self, pcm: "numpy.ndarray", sample_rate: int) -> str:
        raise RuntimeError(REJECTION_REASON)

    def unload(self) -> None:
        return None
