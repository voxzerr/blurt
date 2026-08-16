"""Tests for :class:`blurt.audio.Recorder` -- the thing between a microphone and the model.

:mod:`tests.test_audio_gain` already covers the pure functions in this module.
Nothing here re-tests them. What is defended here is the part of the Recorder
that has state, a lock and a realtime thread, because that is the part that can
lose or misreport what the user said:

  * **An overflow verdict is never erased.** PortAudio reports dropped input by
    setting a status flag on a callback, and the app turns that into "text may be
    clipped". A callback that lands between ``start()``'s claim and its return
    belongs to the take that was just claimed, so a reset placed AFTER the lock
    wipes a real dropped-frame report and the user is told nothing about the
    words that went missing. The reset therefore happens inside the lock, before
    ``_recording`` goes true. That is one statement, it has no other observable
    effect, and the interleaving that exposes it is built deliberately below --
    a scheduler will not volunteer it.
  * **...but a stale overflow does not haunt the next take either.** The same
    statement carries the other half: last take's dropped frames must not warn
    about this take's clean audio.
  * **Pre-roll survives.** Audio that arrived before the key press is in the take.
    That is the entire reason this module keeps a stream open (and the reason the
    macOS mic indicator stays lit), and losing it clips the first word of every
    dictation -- which reads as "the app missed what I said", not as a bug.
  * **Nothing raises at the caller.** ``stop()`` without ``start()`` returns an
    empty array; a callback that blows up marks the stream broken and returns.
    That second one is not tidiness: the callback runs on a CoreAudio realtime
    thread, and an exception there aborts the stream inside PortAudio and takes
    the recording with it.
  * **Silence is detected.** macOS answers a denied microphone with a working
    stream full of zeros, so ``last_capture_was_silent()`` is the only permission
    signal there is. Whisper answers silence with confident invented sentences,
    which makes this the single most expensive flag in the module to get wrong.

NO MICROPHONE IS OPENED. ``Recorder.__init__`` reaches the outside world through
exactly one module-level function, ``_import_sounddevice``, so replacing that name
replaces the entire device. The fake it gets back exposes ``query_devices`` and an
``InputStream`` that stores its callback instead of spawning a thread -- so the
test decides which blocks arrive, with which status flags, on which thread, at
which instant. A real device could not be asked for any of that.

DETERMINISM. One test needs a callback to land inside ``start()``. It uses a real
thread, but nothing about the ordering is left to the scheduler: the thread is
released from a seam inside the locked section, and the main thread waits on the
recorder's own state (the overflow flag having been set) rather than on a clock.
Every other test drives the callback on the main thread, where "which happened
first" is not a question.

Python 3.9 floor: lazy annotations, typing generics only, no PEP 604 unions.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from blurt import audio as blurt_audio
from blurt.audio import _SILENCE_RMS, Recorder

# --------------------------------------------------------------------------- #
# A microphone made of nothing
# --------------------------------------------------------------------------- #

#: What the fake device claims to run at. 48000 rather than 16000 because that is
#: what real Mac hardware does, and because the tests below then ask for a
#: Recorder at the SAME rate: with source and target equal, ``_resample_linear``
#: is the identity, so a test can assert on the exact samples it delivered
#: instead of on a tolerance. A recorder built at 16 kHz over a 48 kHz device is
#: a resampling test, and resampling already has its own.
_DEVICE_RATE = 48000

#: Blocks are delivered at full scale. ``_normalize_gain`` leaves anything whose
#: peak is already at or above ``_TARGET_PEAK`` (0.95) completely alone, so a
#: 1.0-amplitude block comes back out of ``stop()`` byte-identical and the
#: assertions can be equalities rather than approximations.
_LOUD = 1.0

#: One block, ~6.7 ms at the fake device rate. Small on purpose: several fit
#: inside the pre-roll ring with room to spare, so no test depends on the ring's
#: trimming policy, which is not what any of them are about.
_BLOCK_FRAMES = 320


def _block(value: float, frames: int = _BLOCK_FRAMES) -> "np.ndarray":
    """A constant-valued mono block, the way a test can recognise it again."""
    return np.full((frames,), np.float32(value), dtype=np.float32)


class FakeStatus:
    """PortAudio's CallbackFlags: an object that is truthy when a flag is set.

    The callback tests ``if status:`` rather than comparing to anything, because
    the real object is a flag set whose only reliable property is its
    truthiness. A test that passed ``True`` would be testing a device that does
    not exist; this one at least has the same shape as the thing being stood in
    for.
    """

    def __init__(self, text: str = "input overflow") -> None:
        self.text = text

    def __bool__(self) -> bool:
        return True

    def __str__(self) -> str:
        return self.text


class HostileBlock:
    """Something that is not an array, arriving where an array was promised.

    Models the class of failure the callback's ``except BaseException`` exists
    for: a device coming apart mid-block. The specific exception does not matter
    -- what matters is that it happens on the realtime thread, where PortAudio
    is the caller and an escaping exception kills the stream.
    """

    def __getitem__(self, item):
        raise RuntimeError("device vanished mid-block")


class FakeStream:
    """An InputStream that delivers exactly the blocks a test hands it.

    The real one calls back from a CoreAudio realtime thread whenever it likes.
    That is not merely awkward to test, it is nondeterministic in precisely the
    dimension this file cares about -- WHEN a callback lands relative to
    ``start()`` -- so the callback is stored here and the test invokes it.
    """

    def __init__(self, samplerate, blocksize, callback) -> None:
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.callback = callback
        self.active = False
        self.starts = 0
        self.stops = 0
        self.closes = 0

    # -- the surface Recorder uses -----------------------------------------
    def start(self) -> None:
        self.starts += 1
        self.active = True

    def stop(self) -> None:
        self.stops += 1
        self.active = False

    def close(self) -> None:
        self.closes += 1
        self.active = False

    # -- the seam the tests drive ------------------------------------------
    def deliver(self, block, status=None) -> None:
        """Hand one block to the callback exactly as PortAudio would.

        Shaped ``(frames, 1)``, because the callback reads ``indata[:, 0]``: a
        mono stream from PortAudio is still two-dimensional, and a fake that
        delivered a flat array would let a real indexing bug through.
        """
        indata = np.asarray(block, dtype=np.float32).reshape(-1, 1)
        self.callback(indata, indata.shape[0], None, status)

    def deliver_raw(self, indata, frames: int = 0, status=None) -> None:
        """Hand the callback something arbitrary -- including something broken."""
        self.callback(indata, frames, None, status)


class FakeSounddevice:
    """The whole sounddevice module, in the two attributes Recorder touches."""

    def __init__(self, default_samplerate: int = _DEVICE_RATE) -> None:
        self.default_samplerate = float(default_samplerate)
        self.query_calls = 0
        self.streams = []
        self.open_error = None

    def query_devices(self, kind=None):
        self.query_calls += 1
        return {"name": "Fake Input", "default_samplerate": self.default_samplerate}

    def InputStream(  # noqa: N802 - PortAudio's spelling, not ours
        self, samplerate=None, channels=1, dtype="float32", blocksize=None, callback=None
    ):
        if self.open_error is not None:
            raise self.open_error
        stream = FakeStream(samplerate, blocksize, callback)
        self.streams.append(stream)
        return stream

    @property
    def stream(self) -> FakeStream:
        """The stream currently open. Not the only one ever opened -- see reopen."""
        return self.streams[-1]


@pytest.fixture
def sd(monkeypatch):
    """Replace the one function through which this module reaches the OS."""
    fake = FakeSounddevice()
    monkeypatch.setattr(blurt_audio, "_import_sounddevice", lambda: fake)
    return fake


@pytest.fixture
def recorder(sd):
    """A Recorder on the fake device, always closed again.

    Target rate equals the device rate deliberately; see ``_DEVICE_RATE``.
    """
    rec = Recorder(sample_rate=_DEVICE_RATE, preroll_ms=500)
    try:
        yield rec
    finally:
        rec.close()


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    """Block until ``predicate()`` is true. Waits on a CONDITION, never a duration.

    The timeout is a deadlock escape hatch, not a synchronisation mechanism: in a
    passing run this returns on the first or second poll. A test that slept for a
    fixed period instead would be asserting something about this machine's
    scheduler.
    """
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.001)


# --------------------------------------------------------------------------- #
# The overflow verdict: reported when it happened, and only then
# --------------------------------------------------------------------------- #
def test_a_clean_take_is_not_reported_as_having_dropped_frames(recorder, sd):
    """A false "text may be clipped" teaches the user to ignore the true one."""
    recorder.start()
    sd.stream.deliver(_block(_LOUD))
    recorder.stop()

    assert recorder.last_capture_overflowed() is False


def test_a_status_flagged_block_is_reported_as_an_overflow(recorder, sd):
    """PortAudio reports dropped input as a status flag and in no other way.

    The audio that DID arrive still comes back: dropping frames degrades a
    transcript, it does not invalidate the take, and throwing the rest away would
    turn a clipped sentence into no sentence at all.
    """
    recorder.start()
    sd.stream.deliver(_block(_LOUD), status=FakeStatus())
    pcm = recorder.stop()

    assert recorder.last_capture_overflowed() is True
    assert pcm.size == _BLOCK_FRAMES


def test_an_overflow_reported_after_the_claim_is_not_erased_by_start(recorder, sd):
    """THE ORDERING. A callback that joins this take is heard, not overwritten.

    ``start()`` resets the overflow verdict INSIDE its lock, before ``_recording``
    goes true. Move that one statement after the ``with`` block and this is what
    happens: a callback fires in the window between the claim and the return, its
    block is appended to the take that was just claimed -- so its dropped frames
    are this take's dropped frames -- and then ``start()`` finishes by clearing
    the flag it just set. The take is short a chunk of the user's sentence and
    nothing anywhere says so.

    The interleaving is constructed, not awaited. ``_Ring.clear()`` is the last
    call inside the locked section, after ``_recording = True``, which makes it a
    seam at exactly the right instant: from there the test releases a second
    thread to deliver a status-flagged block, and waits for the recorder's own
    ``_overflowed`` to go true before letting ``start()`` continue. The callback
    then blocks on the recorder's lock -- which is the real behaviour, and which
    is why its block lands in ``_chunks`` rather than the pre-roll ring.

    No sleep orders anything here. The main thread waits on the flag it is about
    to make an assertion about, and the callback thread is joined before the
    assertions run.
    """
    rec, stream = recorder, sd.stream
    admit = threading.Event()
    observed = {}
    real_clear = rec._ring.clear

    def clear_and_admit_the_callback():
        # Inside start(), under the lock, after _recording went true: from this
        # instant an arriving callback belongs to THIS take.
        real_clear()
        admit.set()
        observed["reported"] = _wait_until(lambda: rec._overflowed)

    rec._ring.clear = clear_and_admit_the_callback

    late = _block(_LOUD)

    def report_an_overflow():
        admit.wait(timeout=5.0)
        stream.deliver(late, status=FakeStatus())

    caller = threading.Thread(target=report_an_overflow, name="portaudio-callback")
    caller.start()
    try:
        rec.start()
    finally:
        rec._ring.clear = real_clear
        caller.join(timeout=5.0)

    assert observed.get("reported") is True, (
        "the callback never reported its overflow while start() was still inside "
        "its lock, so this test never built the interleaving it is about"
    )
    assert not caller.is_alive()

    assert rec.last_capture_overflowed() is True, (
        "start() erased an overflow that a callback had already reported against "
        "this take -- the user is never told their words were clipped"
    )

    pcm = rec.stop()
    assert np.array_equal(pcm, late), (
        "the block that reported the overflow did not join the take, so the "
        "overflow it reported was not this take's after all"
    )
    assert rec.last_capture_overflowed() is True, "stop() must not clear the verdict"


def test_an_overflow_in_one_take_is_not_reported_against_the_next(recorder, sd):
    """The other half of the same statement: the reset really does happen.

    "Never erase an overflow" and "never report a stale one" are one line of code
    apart, and a fix for either that loses the other is not a fix. A warning that
    fires on every subsequent dictation is a warning the user stops reading.
    """
    recorder.start()
    sd.stream.deliver(_block(_LOUD), status=FakeStatus())
    recorder.stop()
    assert recorder.last_capture_overflowed() is True

    recorder.start()
    sd.stream.deliver(_block(_LOUD))
    recorder.stop()

    assert recorder.last_capture_overflowed() is False, (
        "the previous take's dropped frames were reported against a clean one"
    )


# --------------------------------------------------------------------------- #
# Pre-roll: the reason the stream is never closed
# --------------------------------------------------------------------------- #
def test_audio_captured_before_start_is_returned_by_stop(recorder, sd):
    """The first word is spoken before the key press finishes registering.

    Exact equality is available here rather than a tolerance because the take is
    at the device's own rate (no resampling) and at full scale (no gain), and
    that is worth having: "the pre-roll is in there somewhere" would still pass
    if the two halves were concatenated in the wrong order.
    """
    before = _block(_LOUD)
    after = _block(-_LOUD)

    sd.stream.deliver(before)
    recorder.start()
    sd.stream.deliver(after)
    pcm = recorder.stop()

    assert pcm.size == 2 * _BLOCK_FRAMES, (
        "the take is {0} frames; the pre-roll block never made it in".format(pcm.size)
    )
    assert np.array_equal(pcm, np.concatenate([before, after])), (
        "the pre-roll is missing, truncated or spliced in after the live audio"
    )


def test_audio_arriving_while_nothing_is_recording_stays_out_of_the_next_take(recorder, sd):
    """Pre-roll is a short ring, not an accumulator.

    ``start()`` clears it, so the pre-roll offered to one take is never offered
    to the next as well. Without that, a long pause between dictations would
    prepend somebody's earlier sentence to this one -- and blurt.inject can paste
    but cannot delete.
    """
    sd.stream.deliver(_block(_LOUD))
    recorder.start()
    recorder.stop()

    recorder.start()
    sd.stream.deliver(_block(-_LOUD))
    pcm = recorder.stop()

    assert pcm.size == _BLOCK_FRAMES
    assert np.array_equal(pcm, _block(-_LOUD))


# --------------------------------------------------------------------------- #
# Nothing raises at the caller, and nothing raises on the realtime thread
# --------------------------------------------------------------------------- #
def test_stop_without_a_start_returns_an_empty_array_rather_than_raising(recorder, sd):
    """Callers treat an empty result as "no audio"; an exception is not that.

    ``stop()`` is called from a hotkey release, and the release can genuinely
    arrive without a matching press -- a key held from before startup, a claim
    the other hotkey won. Raising there turns a harmless mis-order into a warning
    printed at the user about a dictation that never existed.
    """
    sd.stream.deliver(_block(_LOUD))  # audio arriving with nobody recording

    pcm = recorder.stop()

    assert isinstance(pcm, np.ndarray)
    assert pcm.size == 0
    assert pcm.dtype == np.float32


def test_a_callback_that_raises_marks_the_stream_broken_instead_of_propagating(recorder, sd):
    """An exception on the CoreAudio thread aborts the stream inside PortAudio.

    So the callback swallows everything and records the fact for the next
    ``start()`` to act on. The take in progress is left alone: whatever was
    captured before the fault is still the user's, and tearing it down from the
    realtime thread is the one thing that must not happen here.
    """
    recorder.start()

    sd.stream.deliver_raw(HostileBlock(), frames=_BLOCK_FRAMES)

    assert recorder._stream_broken is True
    assert recorder.is_recording is True


def test_a_stream_marked_broken_is_reopened_by_the_next_start(recorder, sd):
    """What "broken" buys: the next press records on whatever device exists now.

    Losing a device is quiet -- unplug a headset and PortAudio simply stops
    calling back. Without the reopen, ``start()`` records from a corpse, returns
    zero frames, and the user sees an app that ignored them.
    """
    sd.stream.deliver_raw(HostileBlock(), frames=_BLOCK_FRAMES)
    assert recorder._stream_broken is True
    first = sd.streams[0]

    recorder.start()

    assert len(sd.streams) == 2, "the dead stream was recorded from a second time"
    assert first.closes == 1, "the dead stream was never released"
    assert sd.stream.starts == 1
    assert recorder._stream_broken is False

    sd.stream.deliver(_block(_LOUD))
    assert recorder.stop().size == _BLOCK_FRAMES, "the reopened stream captured nothing"


# --------------------------------------------------------------------------- #
# Silence: the only microphone-permission signal macOS gives us
# --------------------------------------------------------------------------- #
def test_an_all_zero_take_is_reported_as_silent(recorder, sd):
    """A denied microphone on macOS does not raise. It delivers zeros.

    "You said nothing" and "we are not allowed to hear you" are the same bytes,
    so this flag is what the app turns into "grant blurt microphone access"
    instead of feeding digital silence to a model that will hallucinate a
    sentence out of it.
    """
    recorder.start()
    sd.stream.deliver(np.zeros(_BLOCK_FRAMES, dtype=np.float32))
    pcm = recorder.stop()

    assert recorder.last_capture_was_silent() is True
    assert recorder.last_capture_rms() == 0.0
    # The audio is still handed back. The caller decides what silence means.
    assert pcm.size == _BLOCK_FRAMES


def test_a_take_with_real_signal_is_not_reported_as_silent(recorder, sd):
    """The flag sends the user to a permissions dialog, so a false one is costly.

    A dictation that worked, reported as a permission failure, sends them
    hunting through System Settings for a checkbox that was never unticked.
    """
    recorder.start()
    sd.stream.deliver(_block(0.2))
    recorder.stop()

    assert recorder.last_capture_was_silent() is False
    assert recorder.last_capture_rms() > _SILENCE_RMS
