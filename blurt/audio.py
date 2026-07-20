"""Microphone capture for blurt: pre-roll ring buffer in, 16 kHz mono float32 out.

The Recorder keeps an InputStream open continuously and pushes every block into a
short ring buffer. start() snapshots that ring so the beginning of the utterance
is already captured before the user's key press finished registering; stop()
returns one float32 array at `sample_rate` (16 kHz by default), resampled from
whatever rate the device actually gave us.

THE SAMPLE RATE TRAP (the reason this module resamples itself)
--------------------------------------------------------------
Do not ask a CoreAudio device for 16000 Hz and believe the answer. Asking for a
rate the hardware does not run at can succeed anyway -- the OS or PortAudio may
hand back a stream at the device's own rate, and PortAudio's `samplerate`
attribute is what was negotiated, not proof that the samples are at that rate.
Most Mac built-in mics and every USB headset we have seen run at 44100 or 48000.

If we assume 16 kHz while receiving 48 kHz, every downstream consumer reads the
audio 3x too slow: a 4-second sentence becomes a 12-second one, pitch drops an
octave and a half, and Whisper does not error -- it transcribes the stretched
signal into fluent, confident, completely wrong text. That failure is invisible
in testing unless you listen to the captured audio, which is exactly how these
apps ship broken.

So: open the stream at the device's own default rate, record at that rate, and
resample to the target ourselves in stop(). See _resample_linear.

WHAT ELSE GOES WRONG ON macOS
-----------------------------
  - Microphone permission (TCC). When the user has not granted mic access, macOS
    does NOT raise. It opens the stream happily and delivers digital silence --
    an unbroken run of 0.0 samples. "You said nothing" and "we are not allowed to
    hear you" look identical at the API level, so we measure RMS and expose
    last_capture_was_silent() for the caller to turn into a real error message
    ("Grant blurt microphone access in System Settings > Privacy & Security"),
    not a shrug. A CLI launched from Terminal inherits Terminal's permission, so
    a build can work in dev and be silent as a bundled .app.
  - The mic indicator stays lit. Pre-roll requires a live stream at all times, so
    the orange dot in the menu bar is on for as long as the Recorder exists. That
    is inherent to never clipping the first word. close() releases it.
  - Devices vanish. Unplugging a headset mid-recording kills the stream. PortAudio
    surfaces that as a callback status flag or an error on a later call, not as a
    clean exception you can catch around your recording. We keep the audio already
    captured, mark the stream dead, and let the next start() reopen on whatever
    device is now the default.
  - Audio callbacks run on a realtime CoreAudio thread. Anything slow in there --
    allocation storms, I/O, a contended lock, a print -- causes dropouts and
    glitching. The callback in this file copies the block and appends it. Nothing
    else. All resampling and analysis happens in stop(), on the caller's thread.

Python 3.9 floor: `from __future__ import annotations`, typing.Optional / List,
no PEP 604 unions, no match statements. numpy only -- scipy is not a dependency.
"""

from __future__ import annotations

import collections
import math
import sys
import threading
from typing import Any, Deque, List, Optional

import numpy as np

__all__ = ["AudioUnavailable", "Recorder"]


# RMS below this counts as "effectively silent". Digital silence is exactly 0.0;
# a working mic in a quiet room still picks up self-noise well above 1e-3, so
# 1e-4 (-80 dBFS) sits in the dead zone between "muted/denied" and "quiet room"
# without flagging a genuinely soft speaker as a permission failure.
_SILENCE_RMS = 1e-4

# --- gain normalization ----------------------------------------------------
#
# Whisper is trained on roughly normalized audio, and it degrades badly when fed
# a very quiet signal. This is not a theoretical concern -- measured on this
# project's reference machine with base.en, using identical synthesized speech
# attenuated to different levels:
#
#     input rms 0.0014   raw ->  0.0% word error rate
#     input rms 0.0002   raw -> 26.5% word error rate
#     input rms 0.0002   peak-normalized ->  0.0% word error rate
#
# The failure mode is the nasty kind: at 0.0002 the model silently DROPPED the
# first seven words of the utterance rather than garbling them, so the output
# looks like a confident, fluent, wrong transcript. A user experiences that as
# "it isn't transcribing what I said".
#
# Normalizing costs nothing measurable (1.78s vs 1.79s on the same clip), so it
# is applied unconditionally to every capture.
#
# Two guards keep this from making things worse:
#   * Near-silent input is left alone. Amplifying a room-tone-only buffer to full
#     scale invites Whisper to hallucinate speech out of noise.
#   * Gain is capped. Without a ceiling, a buffer whose peak is 1e-6 would be
#     multiplied by ~950,000, turning the noise floor into a roar.
_TARGET_PEAK = 0.95
# The floor below, not this cap, is what stops us amplifying room tone -- so the
# cap only needs to bound the absurd case. 60x was the first value tried and it
# was too tight: a peak of 0.01 needs 95x to reach target and got stuck at 0.6.
# 200x reaches target for anything at or above a 0.005 peak, which covers a very
# quiet speaker, while still bounding a just-above-the-floor buffer to ~0.12
# rather than full scale.
_MAX_GAIN = 200.0
_NORMALIZE_FLOOR_PEAK = 5e-4
# RMS floor. This one is delicate: it has to sit ABOVE measured room tone
# (rms 0.000225-0.000535 on the reference machine) so ambient noise is not
# amplified, but BELOW real speech so quiet talkers are still rescued -- the
# whole point of normalization. 7e-4 threads that gap. Set it much higher and
# it re-breaks the quiet-speech fix; much lower and spiky room tone slips
# through. Speech with an rms below this has essentially no SNR to recover
# anyway, so leaving it for VAD is the right call.
_NORMALIZE_FLOOR_RMS = 7e-4


def _normalize_gain(pcm: "np.ndarray") -> "np.ndarray":
    """Scale quiet audio up toward _TARGET_PEAK, with silence and gain guards.

    Peak normalization rather than RMS: both measured identically (0.0% WER) on
    the reference clip, and peak has no clipping to defend against.
    """
    if pcm.size == 0:
        return pcm
    peak = float(np.max(np.abs(pcm)))
    # Two independent "too quiet to be speech" floors, and BOTH must be cleared
    # before we amplify. The peak floor alone is not enough: measured real room
    # tone had rms 0.000535 but a peak of 0.028 (a few transient clicks), which
    # sails over a peak-only guard and would get a ~33x boost -- turning silence
    # into a speech-level roar. Adding the RMS floor closes that path. (VAD would
    # currently catch the amplified noise downstream, but relying on a second
    # system to clean up after this one is how latent bugs are born.)
    rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
    if peak < _NORMALIZE_FLOOR_PEAK or rms < _NORMALIZE_FLOOR_RMS:
        return pcm
    # Already at a healthy level; don't touch it.
    if peak >= _TARGET_PEAK:
        return pcm
    gain = min(_TARGET_PEAK / peak, _MAX_GAIN)
    return (pcm * gain).astype(np.float32)


# Requested block size, in seconds. ~20 ms is small enough that pre-roll is
# accurate to within one block and large enough that the callback fires at a
# sane rate. PortAudio may ignore this and pick its own size; nothing here
# depends on the request being honoured.
_BLOCK_SECONDS = 0.02

# Smallest block size we will plan for when sizing the ring's hard cap. Purely a
# memory bound, see _Ring below.
_MIN_PLAUSIBLE_BLOCK = 32


class AudioUnavailable(RuntimeError):
    """Raised when microphone capture cannot work on this machine.

    Carries a message meant to be shown to a human as-is: it names the cause and
    the fix. Raised from Recorder.__init__, never at import time -- importing
    blurt.audio must stay safe on a machine with no sound hardware at all so the
    app can start and report the problem instead of dying in an import.
    """


def _describe_import_failure(exc: BaseException) -> str:
    return (
        "Could not load the sounddevice module, so blurt cannot use the "
        "microphone.\n"
        f"  Underlying error: {exc}\n"
        "  Fix: install it into the interpreter running blurt "
        f"({sys.executable}):\n"
        "    " + sys.executable + " -m pip install sounddevice\n"
        "  sounddevice ships PortAudio inside its wheel; Homebrew is not needed."
    )


def _import_sounddevice() -> Any:
    """Import sounddevice, converting any failure into AudioUnavailable.

    Imported lazily rather than at module scope. A broken or missing PortAudio
    binary raises OSError from the import itself, not ImportError, so both are
    caught -- and neither should be able to prevent `import blurt.audio`.
    """
    try:
        import sounddevice  # noqa: PLC0415 - deliberately deferred
    except BaseException as exc:  # noqa: BLE001 - import can fail many ways
        raise AudioUnavailable(_describe_import_failure(exc)) from exc
    return sounddevice


class _Ring:
    """Fixed-size ring of recent audio blocks.

    Bounded two ways, on purpose:

      - By duration, in frames: after each append we drop blocks from the front
        while the ring would still hold at least `max_frames` without them. This
        is the real policy, and it is correct no matter what block size PortAudio
        chose -- counting blocks alone would hold 200 ms or 2 s depending on the
        device.
      - By block count, via deque(maxlen=...): a hard ceiling so a device that
        delivers pathologically tiny blocks cannot grow this without limit. Sized
        so it never triggers before the frame policy does under any plausible
        block size, which keeps the frame counter honest.

    Not thread-safe. The Recorder owns the lock.
    """

    def __init__(self, max_frames: int) -> None:
        self._max_frames = max(0, int(max_frames))
        cap = self._max_frames // _MIN_PLAUSIBLE_BLOCK + 4
        self._blocks: Deque[np.ndarray] = collections.deque(maxlen=cap)
        self._frames = 0

    def append(self, block: np.ndarray) -> None:
        blocks = self._blocks
        blocks.append(block)
        self._frames += block.shape[0]
        # Bounded: each iteration removes one block, and we stop as soon as the
        # remainder no longer covers max_frames on its own.
        while blocks and self._frames - blocks[0].shape[0] >= self._max_frames:
            self._frames -= blocks.popleft().shape[0]

    def snapshot(self) -> List[np.ndarray]:
        """Copy out the current block list. Cheap: copies references, not audio."""
        return list(self._blocks)

    def clear(self) -> None:
        self._blocks.clear()
        self._frames = 0

    @property
    def max_frames(self) -> int:
        return self._max_frames


def _resample_linear(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample mono float32 audio with linear interpolation. numpy only.

    Linear interpolation is not the highest-quality resampler in existence. It is
    the right one here: it is a few lines, has no dependencies (scipy is not
    installable-by-default on the floor machine and is not a dependency), costs
    nothing on a 2-core i7, and its shortcomings are inaudible to a speech model
    that only cares about the spectrum below ~8 kHz.

    Downsampling needs care. Dropping from 48 kHz to 16 kHz cuts the Nyquist
    limit from 24 kHz to 8 kHz, and any energy above 8 kHz that survives folds
    back down as aliasing -- a 15 kHz hiss reappears as a 1 kHz tone sitting in
    the middle of the speech band, which is exactly where it does the most damage
    to a transcript. So we low-pass before decimating.

    The filter is two cascaded box (moving-average) filters of width ~= the
    decimation ratio, which is the same as convolving with a triangular kernel.
    One box alone only reaches about -22 dB in the stopband; cascading squares the
    response for roughly -44 dB, at the cost of one extra convolution and a gentle
    tilt in the passband (about -1.6 dB at 4 kHz, more approaching 8 kHz). That
    trade is right for speech: the rolloff sits where Whisper has little to lose,
    and a mild high-frequency tilt is far kinder to a transcript than aliased
    energy folded on top of the formants. A windowed-sinc would beat both, and is
    not worth hand-rolling and tuning here.

    Known and accepted: stopband suppression is not uniform. A box filter has
    nulls at multiples of src_rate/width (16 kHz and 32 kHz for the 48->16 case),
    and the shoulder between DC and the first null is its weakest point. Measured
    worst case for 48->16 is 12 kHz, which survives at -19 dB and folds down to
    4 kHz; 15 kHz gets -44 dB and 20 kHz gets -25 dB. That is fine here rather
    than lucky: speech spectra have already fallen off steeply by 12 kHz, so the
    folded remnant lands far below the formants it lands on. Adding a third
    cascade would buy -29 dB there but cost -2.4 dB at 3 kHz, which is real
    speech. Do not "fix" the 12 kHz shoulder without measuring what the extra
    passband droop does to transcripts.

    Returns a 1-D float32 array. Never raises on empty input.
    """
    if pcm.size == 0:
        return np.zeros(0, dtype=np.float32)

    flat = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if src_rate == dst_rate:
        return flat

    n_src = flat.shape[0]
    work = flat.astype(np.float64, copy=False)

    # Anti-alias only when going down in rate. Upsampling invents no new content
    # above the old Nyquist, so there is nothing to fold.
    if src_rate > dst_rate:
        width = int(round(float(src_rate) / float(dst_rate)))
        if width >= 2 and n_src >= width:
            box = np.full(width, 1.0 / width, dtype=np.float64)
            # Cascade of two boxes == convolution with a triangle. Built once so
            # the signal is walked a single time instead of twice.
            kernel = np.convolve(box, box)
            work = np.convolve(work, kernel, mode="same")

    n_dst = int(round(n_src * (float(dst_rate) / float(src_rate))))
    if n_dst < 1:
        # Sub-one-sample input. Nothing meaningful survives; say so honestly.
        return np.zeros(0, dtype=np.float32)

    # Map each output sample onto a fractional position in the source. Using
    # (n_src - 1) / (n_dst - 1) would stretch the last sample to the exact end of
    # the input; using the rate ratio keeps playback speed exactly right, which is
    # the entire point of this function. Positions past the end are clamped by
    # np.interp, which costs at most one sample of tail.
    step = float(src_rate) / float(dst_rate)
    positions = np.arange(n_dst, dtype=np.float64) * step
    source_index = np.arange(n_src, dtype=np.float64)
    out = np.interp(positions, source_index, work)
    return out.astype(np.float32, copy=False)


def _rms(pcm: np.ndarray) -> float:
    """Root-mean-square level of a float32 buffer. 0.0 for empty input."""
    if pcm.size == 0:
        return 0.0
    acc = np.asarray(pcm, dtype=np.float64)
    value = float(np.sqrt(np.mean(np.square(acc))))
    if math.isnan(value) or math.isinf(value):
        # A device fault can emit NaN/Inf samples. Treat as silence rather than
        # letting a NaN propagate into the model.
        return 0.0
    return value


class Recorder:
    """Hold-to-talk microphone capture with pre-roll.

    Typical use::

        rec = Recorder(sample_rate=16000, preroll_ms=500)   # may raise AudioUnavailable
        rec.start()
        ...
        pcm = rec.stop()               # float32 mono at 16 kHz, includes pre-roll
        if rec.last_capture_was_silent():
            ...tell the user about microphone permission...
        rec.close()

    The stream runs from construction to close(), which is what makes pre-roll
    possible and what keeps the macOS mic indicator lit. Construct one, keep it,
    close it on exit.

    Thread safety: start/stop/close are safe to call from any thread and are safe
    to call in the wrong order (stop without start returns an empty array; close
    twice is a no-op). The audio callback shares one lock with them and holds it
    only for an append.
    """

    def __init__(self, sample_rate: int = 16000, preroll_ms: int = 500) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if preroll_ms < 0:
            raise ValueError("preroll_ms must not be negative")

        self._target_rate = int(sample_rate)
        self._preroll_ms = int(preroll_ms)

        self._sd = _import_sounddevice()
        self._device_rate = self._query_default_input_rate()

        self._lock = threading.Lock()
        self._ring = _Ring(self._preroll_frames())
        self._chunks: List[np.ndarray] = []
        self._preroll_excess = 0
        self._recording = False
        self._closed = False

        self._stream: Optional[Any] = None
        self._stream_broken = False
        self._last_rms = 0.0
        self._last_silent = False
        self._overflowed = False

        self._open_stream()

    # -- setup ------------------------------------------------------------

    def _query_default_input_rate(self) -> int:
        """Ask the OS what rate the default input device actually runs at.

        Raises AudioUnavailable when there is no input device -- a Mac with no
        built-in mic and nothing plugged in, or a machine where CoreAudio is
        wedged. The message tells the user what to do about it.
        """
        sd = self._sd
        try:
            info = sd.query_devices(kind="input")
        except BaseException as exc:  # noqa: BLE001 - PortAudio raises broadly
            raise AudioUnavailable(
                "No microphone is available.\n"
                f"  Underlying error: {exc}\n"
                "  Fix: connect or enable an input device, then check System "
                "Settings > Sound > Input."
            ) from exc

        if not info:
            raise AudioUnavailable(
                "macOS reports no audio input device.\n"
                "  Fix: connect a microphone or headset, then check System "
                "Settings > Sound > Input."
            )

        rate = 0
        if isinstance(info, dict):
            try:
                rate = int(round(float(info.get("default_samplerate", 0) or 0)))
            except (TypeError, ValueError):
                rate = 0

        if rate <= 0:
            # Unknown rather than broken. 48000 is the near-universal CoreAudio
            # default; the stream open below is the real test, and if the device
            # disagrees PortAudio will tell us there.
            rate = 48000
        return rate

    def _preroll_frames(self) -> int:
        return int(self._device_rate * self._preroll_ms / 1000.0)

    def _open_stream(self) -> None:
        """Open and start the capture stream at the DEVICE's rate, not ours."""
        sd = self._sd
        blocksize = max(_MIN_PLAUSIBLE_BLOCK, int(self._device_rate * _BLOCK_SECONDS))
        try:
            stream = sd.InputStream(
                samplerate=self._device_rate,
                channels=1,
                dtype="float32",
                blocksize=blocksize,
                callback=self._callback,
            )
            stream.start()
        except BaseException as exc:  # noqa: BLE001 - PortAudio raises broadly
            raise AudioUnavailable(
                "Could not open the microphone.\n"
                f"  Underlying error: {exc}\n"
                "  Fix: check that another app is not holding the input device "
                "exclusively, and that blurt is allowed under System Settings > "
                "Privacy & Security > Microphone."
            ) from exc

        # PortAudio may have negotiated a different rate than we asked for. Trust
        # what the stream reports over what we requested -- this is the whole
        # sample-rate trap in three lines.
        actual = getattr(stream, "samplerate", None)
        if actual:
            try:
                resolved = int(round(float(actual)))
            except (TypeError, ValueError):
                resolved = self._device_rate
            if resolved > 0 and resolved != self._device_rate:
                self._device_rate = resolved

        self._stream = stream
        self._stream_broken = False
        with self._lock:
            self._ring = _Ring(self._preroll_frames())

    # -- realtime callback ------------------------------------------------

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,  # noqa: ARG002 - PortAudio signature
        time_info: Any,  # noqa: ARG002 - PortAudio signature
        status: Any,
    ) -> None:
        """Runs on the CoreAudio realtime thread. Copy, append, return.

        Must never raise: an exception here aborts the stream inside PortAudio and
        takes the recording with it. Must never block: no I/O, no printing, no
        allocation beyond the one required copy.

        The copy is not optional. PortAudio reuses `indata`'s buffer for the next
        block, so keeping a view of it means every stored block silently mutates
        into whatever arrives later.
        """
        try:
            if status:
                # Overflow means we were too slow and PortAudio dropped input.
                # Record the fact; report it later, never from this thread.
                self._overflowed = True

            block = np.array(indata[:, 0], dtype=np.float32, copy=True)

            with self._lock:
                if self._recording:
                    self._chunks.append(block)
                else:
                    self._ring.append(block)
        except BaseException:  # noqa: BLE001 - realtime thread must not raise
            self._stream_broken = True

    # -- public API -------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    @property
    def device_sample_rate(self) -> int:
        """The rate the hardware is actually running at. For diagnostics/logging."""
        return self._device_rate

    def start(self) -> None:
        """Begin recording, seeded with the pre-roll already in the ring.

        Idempotent: calling start() while recording keeps the existing take
        rather than truncating it. Attempts one stream reopen if the previous
        device went away (headset unplugged and replugged, output switched).
        """
        if self._closed:
            raise AudioUnavailable("Recorder is closed; construct a new one.")

        if not self._stream_is_healthy():
            # Reopen on whatever the default device is now. Failure raises
            # AudioUnavailable with an actionable message. Note the pre-roll ring
            # is empty right after a reopen, so this one take starts at the key
            # press instead of 500 ms before it -- better than recording silence
            # into a stream that no longer exists.
            self._reopen_after_failure()

        with self._lock:
            if self._recording:
                return
            # Snapshot, don't drain: if start() is somehow racing a stop() the
            # ring stays valid either way.
            self._chunks = self._ring.snapshot()
            # The ring is trimmed a whole block at a time, so its first block
            # usually straddles the pre-roll boundary and the snapshot carries a
            # little extra history. Record the overshoot NOW, while pre-roll and
            # live audio are still distinguishable -- after concatenation in
            # stop() there is no way to find the seam.
            captured = 0
            for block in self._chunks:
                captured += block.shape[0]
            self._preroll_excess = max(0, captured - self._preroll_frames())
            self._recording = True
            self._ring.clear()
        self._overflowed = False

    def stop(self) -> "np.ndarray":
        """Stop recording and return mono float32 PCM at the target sample rate.

        Includes the pre-roll. Resampled from the device rate. Returns an empty
        array when stop() is called without a matching start(), when the device
        produced nothing, or when the stream died mid-take -- callers should treat
        an empty result as "no audio", not as an error, and consult
        last_capture_was_silent() to tell silence from a permission denial.
        """
        with self._lock:
            if not self._recording:
                return np.zeros(0, dtype=np.float32)
            chunks = self._chunks
            self._chunks = []
            self._recording = False

        # Everything below is off the realtime path and outside the lock.
        raw = self._concatenate(chunks)
        raw = self._trim_leading_preroll(raw)

        self._last_rms = _rms(raw)
        self._last_silent = self._last_rms < _SILENCE_RMS

        resampled = _resample_linear(raw, self._device_rate, self._target_rate)
        return _normalize_gain(resampled)

    def last_capture_was_silent(self) -> bool:
        """True when the last stop() returned silence (or effectively silence).

        On macOS this is the signal for "microphone permission was denied": a
        denied app gets a working stream full of zeros, not an error. It is also
        true when the user genuinely said nothing, when the wrong input device is
        selected, or when the mic is hardware-muted. Any of those deserve a real
        message; none of them deserve being fed to Whisper, which will happily
        hallucinate a sentence out of silence.
        """
        return self._last_silent

    def last_capture_rms(self) -> float:
        """RMS level of the last capture, for logging and threshold tuning."""
        return self._last_rms

    def last_capture_overflowed(self) -> bool:
        """True when PortAudio reported dropped input during the last take."""
        return self._overflowed

    def close(self) -> None:
        """Stop and release the stream. Safe to call twice, safe after failure."""
        self._closed = True
        with self._lock:
            self._recording = False
            self._chunks = []
            self._ring.clear()

        stream = self._stream
        self._stream = None
        if stream is None:
            return

        # A dead device makes stop() and close() throw from PortAudio. There is
        # nothing to do about it and nothing to gain by propagating it out of a
        # cleanup path, so both are best-effort and independent.
        try:
            stream.stop()
        except BaseException:  # noqa: BLE001 - teardown is best-effort
            pass
        try:
            stream.close()
        except BaseException:  # noqa: BLE001 - teardown is best-effort
            pass

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # -- internals --------------------------------------------------------

    def _stream_is_healthy(self) -> bool:
        """Is the stream still alive and delivering audio?

        Losing a device is quieter than you would expect. When a headset is
        unplugged mid-session PortAudio usually does not raise anything into our
        code -- it simply stops invoking the callback, and the stream goes
        inactive. Checking only for an exception we caught in the callback
        (_stream_broken) misses that case entirely and leaves start() recording
        from a corpse, which returns zero frames and looks to the user like the
        app ignored them. So we ask the stream whether it is still active, and
        treat "cannot even answer" as dead.
        """
        stream = self._stream
        if stream is None or self._stream_broken:
            return False
        try:
            active = getattr(stream, "active", None)
        except BaseException:  # noqa: BLE001 - a dead stream raises here
            return False
        if active is None:
            return True  # cannot tell; assume alive rather than churn the device
        return bool(active)

    def _reopen_after_failure(self) -> None:
        """Tear down a dead stream and open a fresh one on the current default."""
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.close()
            except BaseException:  # noqa: BLE001 - it is already broken
                pass
        # Re-query: the default device may have changed, and the new one very
        # likely runs at a different rate than the old one.
        self._device_rate = self._query_default_input_rate()
        self._open_stream()

    @staticmethod
    def _concatenate(chunks: List[np.ndarray]) -> np.ndarray:
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        if len(chunks) == 1:
            return chunks[0]
        return np.concatenate(chunks).astype(np.float32, copy=False)

    def _trim_leading_preroll(self, pcm: np.ndarray) -> np.ndarray:
        """Drop the sub-one-block overshoot measured by start().

        Keeps the returned pre-roll at exactly preroll_ms instead of "preroll_ms
        plus however much of a block happened to be in flight", so the returned
        duration is predictable. Costs at most ~20 ms of leading room tone.
        """
        excess = self._preroll_excess
        if excess <= 0:
            return pcm
        if excess >= pcm.shape[0]:
            return np.zeros(0, dtype=np.float32)
        return pcm[excess:]
