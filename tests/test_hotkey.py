"""Tests for blurt.hotkey -- the hold-to-talk state machine.

blurt/hotkey.py is the only place in the product where a physical gesture turns
into a decision, and every one of its decisions is destructive if it goes the
wrong way: firing when it should not have starts a transcription the user never
asked for, and NOT firing when it should have silently drops something they
said. So this file is written the way the module is written -- around the
things that must NOT happen.

WHY THERE IS NO REAL KEYBOARD ANYWHERE IN HERE
----------------------------------------------
pynput is imported lazily, inside ``_import_keyboard()``, precisely so that
config validation still works on a machine with a broken input stack. That
lazy import is also the seam these tests use: every test that builds a
``HoldToTalk`` monkeypatches ``_import_keyboard`` to hand back a :class:`FakeKeyboard`
whose ``Key`` members are plain sentinel objects and whose ``Listener`` just
records the two callbacks it was given. The test then calls those callbacks
itself, on the test's own thread, which is what "driving the state machine"
means below. Nothing here opens an event tap, and nothing asks macOS for
Accessibility -- ``accessibility_trusted`` is stubbed out for the whole module
by an autouse fixture, so the suite behaves identically on Linux CI and on a
Mac and never touches pyobjc.

WHY THE ASSERTIONS LOOK PARANOID ABOUT THREADS
----------------------------------------------
Callbacks do not run inline. They go onto a queue and are run by one private
worker thread, which is the entire reason a slow transcription cannot stall the
release event that ends a recording. That means "no callback fired" is not
something you can assert by looking immediately -- the callback might merely be
in the queue. Every negative assertion here therefore goes through
:meth:`Rig.drain`, which pushes a probe job onto the SAME fifo queue and waits
for it: one worker plus one fifo means anything queued earlier has already run
by the time the probe runs. Positive assertions wait on the recorder's
condition variable with a 2 second cap. Nothing in this file asserts on elapsed
time, and nothing sleeps waiting for a callback.

WHY SOME TESTS FAKE THE TIMER AND OTHERS LET IT RUN
---------------------------------------------------
Arming is done by a ``threading.Timer``. Tests that need the timer to fire use a
real 20 ms one and block on an Event -- waiting longer than necessary cannot
make those flaky. Tests that need the timer to NOT fire (every tap test, which
is the most important group in the file) instead set an unreachable threshold
and call ``_on_hold_elapsed`` by hand, so the outcome does not depend on how
loaded the machine is. Each test says which technique it uses and why.

Python 3.9 floor: lazy annotations, typing generics, no PEP 585/604 syntax.
"""

from __future__ import annotations

import builtins
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from blurt import hotkey as hotkey_mod
from blurt.hotkey import (
    SUPPORTED_HOTKEYS,
    HoldToTalk,
    UnsupportedHotkeyError,
    normalize_key_name,
)

#: The real Accessibility probe, grabbed before the autouse fixture below stubs
#: it out, so the one test that wants to exercise it for real still can.
_REAL_ACCESSIBILITY_TRUSTED = hotkey_mod.accessibility_trusted

#: Cap on every wait for a worker callback. Long enough that a loaded CI box
#: cannot trip it, short enough that a genuine hang fails the run in seconds
#: instead of blocking it forever.
TIMEOUT_S = 2.0

#: A hold threshold the arming timer cannot reach inside a test. Tests that use
#: it and still want the hold armed call ``_on_hold_elapsed`` themselves, which
#: is what the real timer would do, minus the scheduling luck.
NEVER_MS = 60_000

#: A hold threshold a real timer clears promptly. Only ever used where the test
#: wants the timer to fire, never where it wants it not to.
SOON_MS = 20


# --------------------------------------------------------------------------- #
# The fake keyboard: a stand-in for pynput.keyboard, installed over the lazy
# import seam. Key members are identity-compared sentinels because that is all
# hotkey.py ever does with them (set membership and ==).
# --------------------------------------------------------------------------- #
_KEY_NAMES: Tuple[str, ...] = (
    "alt",
    "alt_l",
    "alt_r",
    "alt_gr",
    "cmd",
    "cmd_l",
    "cmd_r",
    "ctrl",
    "ctrl_l",
    "ctrl_r",
    "shift",
    "shift_l",
    "shift_r",
    "esc",
)


class FakeKey:
    """One pynput ``Key`` member. Hashable and compared by identity, like the real one."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return "<FakeKey.{0}>".format(self.name)


class FakeKeyNamespace:
    """``keyboard.Key``, with every member hotkey.py looks up.

    ``macos_shared_left`` reproduces a detail that is easy to forget and that
    hotkey.py's own comments call out: on macOS ``Key.alt_l`` IS ``Key.alt``
    (both are virtual key 0x3A), and likewise for cmd, ctrl and shift -- only
    the right-hand keys have distinct codes. Binding ``left_option`` there means
    binding an object that is also the bare-modifier member, and the sibling set
    collapses from four members to three. A fake that always kept them distinct
    would let a regression in that collapse go unnoticed on the one platform
    blurt actually ships to.
    """

    def __init__(self, macos_shared_left: bool = False) -> None:
        for name in _KEY_NAMES:
            setattr(self, name, FakeKey(name))
        if macos_shared_left:
            self.alt_l = self.alt
            self.cmd_l = self.cmd
            self.ctrl_l = self.ctrl
            self.shift_l = self.shift


class FakeListener:
    """Records the handlers pynput would have called and lets the test call them.

    ``stop_error`` exists because pynput's teardown is genuinely noisy -- it
    re-raises callback exceptions out of ``join()`` -- and hotkey.py promises
    that a shutdown swallows that rather than letting it escape a stop().
    """

    def __init__(self, on_press: Any, on_release: Any) -> None:
        self.on_press = on_press
        self.on_release = on_release
        self.daemon = False
        self.started = False
        self.stopped = False
        self.joined = False
        self.stop_error: Optional[BaseException] = None

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True
        if self.stop_error is not None:
            raise self.stop_error

    def join(self, timeout: Optional[float] = None) -> None:
        self.joined = True


class FakeKeyboard:
    """The module object ``_import_keyboard()`` is made to return."""

    def __init__(self, macos_shared_left: bool = False) -> None:
        self.Key = FakeKeyNamespace(macos_shared_left=macos_shared_left)
        self.listeners: List[FakeListener] = []
        #: Set to an exception to make constructing a Listener blow up, which is
        #: what an untrusted or wedged Quartz tap looks like from up here.
        self.listener_error: Optional[BaseException] = None

    # Capitalised because pynput's is a class and hotkey.py calls it as one.
    def Listener(  # noqa: N802
        self, on_press: Any = None, on_release: Any = None
    ) -> FakeListener:
        if self.listener_error is not None:
            raise self.listener_error
        listener = FakeListener(on_press, on_release)
        self.listeners.append(listener)
        return listener


# --------------------------------------------------------------------------- #
# Recorder: the three callbacks, plus the synchronisation that makes assertions
# about them honest rather than hopeful.
# --------------------------------------------------------------------------- #
class Recorder:
    """Fake on_start / on_stop / on_cancel that record their order of arrival.

    ``wait_for(n)`` is the Event-with-a-timeout pattern generalised to "the nth
    callback", so a test can wait for exactly the callbacks it expects and then
    assert on the whole list. ``gate(label)`` makes one callback block until the
    test releases it, which is how the "a slow callback must not stall the key
    events" property is tested without a sleep. ``raise_on`` makes a callback
    blow up after recording itself, for the swallowed-exception tests.
    """

    def __init__(self, raise_on: Sequence[str] = ()) -> None:
        self._cond = threading.Condition()
        self._calls: List[str] = []
        self._threads: List[str] = []
        self._raise_on = frozenset(raise_on)
        self._gates: Dict[str, threading.Event] = {}

        self.on_start = self._make("on_start")
        self.on_stop = self._make("on_stop")
        self.on_cancel = self._make("on_cancel")

    def _make(self, label: str) -> Any:
        def callback() -> None:
            with self._cond:
                self._calls.append(label)
                self._threads.append(threading.current_thread().name)
                self._cond.notify_all()
            gate = self._gates.get(label)
            if gate is not None:
                # Bounded, so a test that forgets to open the gate fails on its
                # own assertions instead of wedging the whole run.
                gate.wait(TIMEOUT_S)
            if label in self._raise_on:
                raise RuntimeError("{0} blew up, as this test asked it to".format(label))

        return callback

    @property
    def calls(self) -> List[str]:
        with self._cond:
            return list(self._calls)

    @property
    def threads(self) -> List[str]:
        with self._cond:
            return list(self._threads)

    def gate(self, label: str) -> threading.Event:
        gate = threading.Event()
        self._gates[label] = gate
        return gate

    def wait_for(self, count: int, timeout: float = TIMEOUT_S) -> None:
        with self._cond:
            arrived = self._cond.wait_for(lambda: len(self._calls) >= count, timeout)
        assert arrived, "waited {0}s for {1} callbacks, got {2!r}".format(
            timeout, count, self.calls
        )


class Rig:
    """A started HoldToTalk, the fake keyboard under it, and the recorder over it."""

    def __init__(
        self, hotkey: HoldToTalk, keyboard: FakeKeyboard, recorder: Recorder
    ) -> None:
        self.hotkey = hotkey
        self.keyboard = keyboard
        self.recorder = recorder
        # Captured rather than looked up from ``keyboard.listeners[-1]``: two
        # rigs can share one fake keyboard (app.py runs two hotkeys), and stop()
        # drops the module's own reference, which several tests inspect after.
        self._listener = hotkey._listener

    # -- driving the listener -------------------------------------------------
    @property
    def listener(self) -> FakeListener:
        if self._listener is None:
            self._listener = self.hotkey._listener
        assert self._listener is not None, "this rig was never started"
        return self._listener

    def press(self, key: Any) -> None:
        """Deliver a key-down through the handler pynput was actually given."""
        self.listener.on_press(key)

    def release(self, key: Any) -> None:
        self.listener.on_release(key)

    def elapse(self) -> None:
        """Run the arming timer's callback by hand.

        The pending real timer is cancelled first so it cannot also fire later
        in the run; ``_on_hold_elapsed`` clears the handle itself, which would
        otherwise leave a live daemon timer behind for a minute.
        """
        timer = self.hotkey._hold_timer
        if timer is not None:
            timer.cancel()
        self.hotkey._on_hold_elapsed()

    # -- synchronisation ------------------------------------------------------
    def drain(self, timeout: float = TIMEOUT_S) -> None:
        """Block until every callback queued before now has finished running.

        One worker thread consuming one fifo queue is the whole trick: a probe
        job queued here cannot run until everything queued ahead of it has run.
        Waiting for the probe converts "no callback has fired" from a race into
        a fact -- a stray on_start would already be in ``calls``.
        """
        done = threading.Event()
        self.hotkey._dispatch("test-probe", done.set)
        assert done.wait(timeout), "the hotkey worker never reached the probe job"

    def wait_for(self, count: int, timeout: float = TIMEOUT_S) -> None:
        self.recorder.wait_for(count, timeout)

    @property
    def calls(self) -> List[str]:
        return self.recorder.calls


def _live_workers() -> List[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "blurt-hotkey-worker"]


def _wait_until(predicate: Any, timeout: float = TIMEOUT_S) -> bool:
    """Poll for a thread-liveness fact that has no Event to wait on.

    Used only where the thing being waited for is a thread exiting, which
    exposes no handle we are allowed to join from here. 5 ms steps, bounded.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def never_ask_macos(monkeypatch):
    """Stub the Accessibility probe for every test in this module.

    start() calls it, and on a Mac the real one imports pyobjc and asks the OS a
    question whose answer differs between the developer's machine and CI. The
    tests that care about the answer patch it again with the value they want.
    """
    monkeypatch.setattr(hotkey_mod, "accessibility_trusted", lambda: None)


@pytest.fixture
def fake_keyboard(monkeypatch):
    """Install a fake pynput over the lazy-import seam."""
    keyboard = FakeKeyboard()
    monkeypatch.setattr(hotkey_mod, "_import_keyboard", lambda: keyboard)
    return keyboard


@pytest.fixture
def rig(fake_keyboard, monkeypatch):
    """Factory for started rigs. Every one is stopped at teardown, come what may."""
    built: List[Rig] = []

    def build(
        key_name: str = "right_option",
        min_hold_ms: int = NEVER_MS,
        raise_on: Sequence[str] = (),
        keyboard: Optional[FakeKeyboard] = None,
        start: bool = True,
    ) -> Rig:
        board = fake_keyboard if keyboard is None else keyboard
        monkeypatch.setattr(hotkey_mod, "_import_keyboard", lambda: board)
        recorder = Recorder(raise_on=raise_on)
        hotkey = HoldToTalk(
            key_name=key_name,
            on_start=recorder.on_start,
            on_stop=recorder.on_stop,
            on_cancel=recorder.on_cancel,
            min_hold_ms=min_hold_ms,
        )
        if start:
            hotkey.start()
        item = Rig(hotkey, board, recorder)
        built.append(item)
        return item

    yield build

    for item in reversed(built):
        # Open every gate first: a rig torn down mid-gate would otherwise make
        # stop() wait out the worker join timeout.
        for gate in item.recorder._gates.values():
            gate.set()
        item.hotkey.stop()


@pytest.fixture
def trigger(fake_keyboard):
    """The default trigger key object (right option)."""
    return fake_keyboard.Key.alt_r


@pytest.fixture
def esc(fake_keyboard):
    return fake_keyboard.Key.esc


# --------------------------------------------------------------------------- #
# normalize_key_name: config validation, and it must work with no pynput at all
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", SUPPORTED_HOTKEYS)
def test_every_supported_name_round_trips(name):
    assert normalize_key_name(name) == name


@pytest.mark.parametrize("name", SUPPORTED_HOTKEYS)
def test_every_supported_name_maps_to_a_real_pynput_attribute(name):
    """SUPPORTED_HOTKEYS is user-facing; a name in it with no binding is a lie."""
    assert name in hotkey_mod._KEY_ATTRS


@pytest.mark.parametrize(
    "alias,canonical", sorted(hotkey_mod._KEY_ALIASES.items())
)
def test_every_alias_folds_onto_a_supported_name(alias, canonical):
    assert normalize_key_name(alias) == canonical
    assert canonical in SUPPORTED_HOTKEYS


@pytest.mark.parametrize(
    "spelling",
    [
        "right_option",
        "RIGHT_OPTION",
        "  right_option  ",
        "right-option",
        "right option",
        "Right-Alt",
        "  RALT ",
    ],
)
def test_spelling_is_forgiven_so_a_config_file_cannot_brick_startup(spelling):
    assert normalize_key_name(spelling) == "right_option"


def test_fn_is_refused_and_names_a_working_alternative():
    with pytest.raises(UnsupportedHotkeyError) as caught:
        normalize_key_name("fn")
    message = str(caught.value)
    assert "fn" in message
    # Refusing without saying what to use instead just moves the dead end.
    assert "right_option" in message


def test_globe_is_refused_and_names_a_working_alternative():
    with pytest.raises(UnsupportedHotkeyError) as caught:
        normalize_key_name("globe")
    assert "right_option" in str(caught.value)


def test_caps_lock_is_refused_because_a_toggle_cannot_express_a_hold():
    with pytest.raises(UnsupportedHotkeyError) as caught:
        normalize_key_name("caps lock")
    message = str(caught.value)
    assert "toggle" in message
    assert "right_option" in message


def test_an_unknown_name_lists_every_supported_one():
    with pytest.raises(UnsupportedHotkeyError) as caught:
        normalize_key_name("f13")
    message = str(caught.value)
    assert "f13" in message
    for name in SUPPORTED_HOTKEYS:
        assert name in message


@pytest.mark.parametrize(
    "value,type_name", [(None, "NoneType"), (37, "int"), (["right_option"], "list")]
)
def test_a_non_string_is_refused_by_type_rather_than_crashing_on_lookup(
    value, type_name
):
    """A config file that deserialised to the wrong type must not raise AttributeError."""
    with pytest.raises(UnsupportedHotkeyError) as caught:
        normalize_key_name(value)
    assert type_name in str(caught.value)


def test_unsupported_hotkey_error_is_catchable_as_a_value_error():
    """Callers guarding only against bad config values must still catch it."""
    assert issubclass(UnsupportedHotkeyError, ValueError)
    with pytest.raises(ValueError):
        normalize_key_name("fn")


# --------------------------------------------------------------------------- #
# Construction: a bad argument is reported as a bad argument, never as a
# missing dependency, and never as a hotkey that silently never fires
# --------------------------------------------------------------------------- #
def _no_pynput(monkeypatch):
    """Make ``from pynput import keyboard`` fail the way a bare machine does."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pynput" or name.startswith("pynput."):
            raise ImportError("No module named 'pynput'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_a_bad_hotkey_name_is_reported_even_when_pynput_is_missing(monkeypatch):
    """The name is validated first, on purpose: bad config must read as bad config."""
    _no_pynput(monkeypatch)
    noop = lambda: None  # noqa: E731
    with pytest.raises(UnsupportedHotkeyError):
        HoldToTalk("fn", noop, noop, noop)


def test_a_missing_pynput_becomes_advice_rather_than_an_import_traceback(monkeypatch):
    _no_pynput(monkeypatch)
    noop = lambda: None  # noqa: E731
    with pytest.raises(RuntimeError) as caught:
        HoldToTalk("right_option", noop, noop, noop)
    message = str(caught.value)
    assert "pynput" in message
    assert "pyobjc" in message
    assert "Accessibility" in message


@pytest.mark.parametrize("bad_slot", [0, 1, 2])
def test_a_non_callable_callback_is_refused(fake_keyboard, bad_slot):
    callbacks: List[Any] = [lambda: None, lambda: None, lambda: None]
    callbacks[bad_slot] = "not a function"
    with pytest.raises(TypeError):
        HoldToTalk("right_option", callbacks[0], callbacks[1], callbacks[2])


@pytest.mark.parametrize("bad", ["soon", None, object(), float("nan")])
def test_a_min_hold_that_is_not_a_number_of_milliseconds_is_refused(
    fake_keyboard, bad
):
    noop = lambda: None  # noqa: E731
    with pytest.raises(ValueError):
        HoldToTalk("right_option", noop, noop, noop, min_hold_ms=bad)


def test_a_negative_min_hold_is_clamped_rather_than_rejected(fake_keyboard):
    """Negative is meaningless but harmless; clamping beats refusing to start."""
    noop = lambda: None  # noqa: E731
    hotkey = HoldToTalk("right_option", noop, noop, noop, min_hold_ms=-50)
    assert hotkey.min_hold_ms == 0


def test_an_alias_is_stored_in_its_canonical_form(fake_keyboard):
    noop = lambda: None  # noqa: E731
    hotkey = HoldToTalk("Right-Alt", noop, noop, noop)
    assert hotkey.key_name == "right_option"


def test_a_fresh_hotkey_is_neither_running_nor_armed(rig):
    item = rig(start=False)
    assert item.hotkey.is_running is False
    assert item.hotkey.is_armed is False
    assert item.keyboard.listeners == []


# --------------------------------------------------------------------------- #
# THE TAP. The single most important behaviour in the module: a press shorter
# than min_hold_ms must produce NO callbacks at all, not even on_start.
# All of these use the unreachable-threshold technique, so the result cannot
# depend on how quickly the machine got round to the timer.
# --------------------------------------------------------------------------- #
def test_a_tap_shorter_than_min_hold_fires_nothing_at_all(rig, trigger):
    item = rig(min_hold_ms=NEVER_MS)

    item.press(trigger)
    item.release(trigger)

    item.drain()
    assert item.calls == []


def test_a_tap_is_not_armed_by_a_timer_that_runs_after_the_release(rig, trigger):
    """The arming timer firing late must find the hold gone and do nothing."""
    item = rig(min_hold_ms=NEVER_MS)

    item.press(trigger)
    item.release(trigger)
    item.hotkey._on_hold_elapsed()  # the timer callback, arriving after the fact

    item.drain()
    assert item.calls == []


def test_a_tap_leaves_the_machine_idle_rather_than_half_armed(rig, trigger):
    item = rig(min_hold_ms=NEVER_MS)

    item.press(trigger)
    item.release(trigger)

    assert item.hotkey.is_armed is False
    assert item.hotkey._state == hotkey_mod._IDLE
    assert item.hotkey._key_down is False


def test_a_tap_cancels_its_arming_timer(rig, trigger):
    item = rig(min_hold_ms=NEVER_MS)

    item.press(trigger)
    timer = item.hotkey._hold_timer
    assert timer is not None
    item.release(trigger)

    assert item.hotkey._hold_timer is None
    assert timer.finished.is_set()


def test_a_tap_before_a_real_hold_contributes_no_callbacks(rig, trigger):
    """The brushed-modifier case: the stray tap must not show up in the record."""
    item = rig(min_hold_ms=NEVER_MS)

    item.press(trigger)
    item.release(trigger)

    item.press(trigger)
    item.elapse()
    item.release(trigger)

    item.wait_for(2)
    item.drain()
    assert item.calls == ["on_start", "on_stop"]


def test_ten_taps_in_a_row_fire_nothing(rig, trigger):
    item = rig(min_hold_ms=NEVER_MS)

    for _ in range(10):
        item.press(trigger)
        item.release(trigger)

    item.drain()
    assert item.calls == []


# --------------------------------------------------------------------------- #
# The hold. on_start comes from the timer while the user is still speaking,
# on_stop from the release, in that order, once each.
# --------------------------------------------------------------------------- #
def test_a_hold_past_the_threshold_arms_from_a_real_timer(rig, trigger):
    """Real 20 ms timer, waited on with an Event: waiting too long cannot hurt."""
    item = rig(min_hold_ms=SOON_MS)

    item.press(trigger)
    item.wait_for(1)

    # on_start arrives while the key is still down -- recording begins during
    # the utterance, not after it.
    assert item.calls == ["on_start"]
    assert item.hotkey.is_armed is True

    item.release(trigger)
    item.wait_for(2)
    item.drain()
    assert item.calls == ["on_start", "on_stop"]
    assert item.hotkey.is_armed is False


def test_a_hold_fires_start_then_stop_exactly_once_each(rig, trigger):
    """Same property, driven deterministically through _on_hold_elapsed."""
    item = rig(min_hold_ms=NEVER_MS)

    item.press(trigger)
    item.elapse()
    item.wait_for(1)
    assert item.calls == ["on_start"]

    item.release(trigger)
    item.wait_for(2)
    item.drain()
    assert item.calls == ["on_start", "on_stop"]


def test_the_arming_timer_is_a_named_daemon(rig, trigger):
    """A forgotten timer must never hold interpreter shutdown open."""
    item = rig(min_hold_ms=NEVER_MS)

    item.press(trigger)

    timer = item.hotkey._hold_timer
    assert timer.daemon is True
    assert timer.name == "blurt-hotkey-hold"


def test_back_to_back_holds_each_fire_their_own_pair(rig, trigger):
    item = rig(min_hold_ms=NEVER_MS)

    for _ in range(3):
        item.press(trigger)
        item.elapse()
        item.release(trigger)

    item.wait_for(6)
    item.drain()
    assert item.calls == ["on_start", "on_stop"] * 3


# --------------------------------------------------------------------------- #
# Auto-repeat. macOS repeats key-down while a key is held; each repeat must be
# swallowed, or one hold becomes a burst of recordings.
# --------------------------------------------------------------------------- #
def test_repeated_presses_while_down_produce_exactly_one_on_start(rig, trigger):
    item = rig(min_hold_ms=NEVER_MS)

    item.press(trigger)
    item.press(trigger)
    item.press(trigger)
    item.elapse()
    item.press(trigger)
    item.press(trigger)

    item.wait_for(1)
    item.drain()
    assert item.calls == ["on_start"]


def test_auto_repeat_does_not_restart_the_arming_timer(rig, trigger):
    """A repeat that reset the timer would push arming out indefinitely."""
    item = rig(min_hold_ms=NEVER_MS)

    item.press(trigger)
    first = item.hotkey._hold_timer
    item.press(trigger)

    assert item.hotkey._hold_timer is first


def test_a_repeat_press_after_arming_does_not_double_the_pair(rig, trigger):
    item = rig(min_hold_ms=NEVER_MS)

    item.press(trigger)
    item.elapse()
    item.press(trigger)
    item.release(trigger)

    item.wait_for(2)
    item.drain()
    assert item.calls == ["on_start", "on_stop"]


def test_a_release_with_no_press_fires_nothing(rig, trigger):
    """The listener starts mid-keystroke often enough for this to be real."""
    item = rig(min_hold_ms=NEVER_MS)

    item.release(trigger)

    item.drain()
    assert item.calls == []


def test_a_second_release_does_not_fire_a_second_on_stop(rig, trigger):
    item = rig(min_hold_ms=NEVER_MS)

    item.press(trigger)
    item.elapse()
    item.release(trigger)
    item.release(trigger)

    item.wait_for(2)
    item.drain()
    assert item.calls == ["on_start", "on_stop"]


# --------------------------------------------------------------------------- #
# Esc. A cancelled recording must never be transcribed anyway -- which means
# the release that follows the cancel has to fire nothing at all.
# --------------------------------------------------------------------------- #
def test_escape_while_armed_cancels_exactly_once(rig, trigger, esc):
    item = rig(min_hold_ms=NEVER_MS)

    item.press(trigger)
    item.elapse()
    item.wait_for(1)
    item.press(esc)

    item.wait_for(2)
    item.drain()
    assert item.calls == ["on_start", "on_cancel"]


def test_the_release_after_a_cancelled_hold_fires_nothing(rig, trigger, esc):
    """The _ABORTED state exists for exactly this: no on_stop after a cancel."""
    item = rig(min_hold_ms=NEVER_MS)

    item.press(trigger)
    item.elapse()
    item.press(esc)
    item.wait_for(2)

    item.release(trigger)

    item.drain()
    assert item.calls == ["on_start", "on_cancel"]
    assert "on_stop" not in item.calls


def test_a_cancelled_hold_sits_in_aborted_until_the_key_comes_up(rig, trigger, esc):
    """The awkward state the module was built around: cancelled, but still held.

    Pinned white-box because the four named states are the module's design, and
    because "cancelled" collapsing back into "idle" while the key is physically
    down is precisely the shape of mistake that would let a later change fire an
    on_stop for a recording the user threw away.
    """
    item = rig(min_hold_ms=NEVER_MS)

    item.press(trigger)
    item.elapse()
    item.press(esc)
    item.wait_for(2)

    assert item.hotkey._state == hotkey_mod._ABORTED
    assert item.hotkey._key_down is True
    assert item.hotkey.is_armed is False

    item.release(trigger)
    assert item.hotkey._state == hotkey_mod._IDLE
    assert item.hotkey._key_down is False


def test_escape_while_pending_leaves_the_hold_abandoned_not_pending(
    rig, trigger, esc
):
    """Still _PENDING would let the timer arm it; _IDLE would lose that the key is down."""
    item = rig(min_hold_ms=NEVER_MS)

    item.press(trigger)
    item.press(esc)

    assert item.hotkey._state == hotkey_mod._ABORTED
    assert item.hotkey._key_down is True


def test_a_second_escape_while_aborted_fires_nothing(rig, trigger, esc):
    item = rig(min_hold_ms=NEVER_MS)

    item.press(trigger)
    item.elapse()
    item.press(esc)
    item.wait_for(2)
    item.press(esc)
    item.press(esc)

    item.drain()
    assert item.calls == ["on_start", "on_cancel"]


def test_escape_while_pending_fires_nothing_at_all(rig, trigger, esc):
    """Nothing was started, so nothing is owed -- not even a cancel."""
    item = rig(min_hold_ms=NEVER_MS)

    item.press(trigger)
    item.press(esc)
    item.release(trigger)

    item.drain()
    assert item.calls == []


def test_a_timer_cannot_arm_a_hold_that_escape_already_abandoned(rig, trigger, esc):
    """Deterministic: the abandoned timer's callback is invoked by hand."""
    item = rig(min_hold_ms=NEVER_MS)

    item.press(trigger)
    item.press(esc)
    item.hotkey._on_hold_elapsed()

    item.drain()
    assert item.calls == []
    assert item.hotkey.is_armed is False


def test_escape_while_pending_cancels_the_arming_timer(rig, trigger, esc):
    item = rig(min_hold_ms=NEVER_MS)

    item.press(trigger)
    timer = item.hotkey._hold_timer
    item.press(esc)

    assert item.hotkey._hold_timer is None
    assert timer.finished.is_set()


def test_escape_while_idle_fires_nothing(rig, trigger, esc):
    item = rig(min_hold_ms=NEVER_MS)

    item.press(esc)
    item.release(esc)

    item.drain()
    assert item.calls == []


def test_a_hold_after_a_cancelled_one_works_normally(rig, trigger, esc):
    """Cancelling must not leave the machine stuck in _ABORTED for good."""
    item = rig(min_hold_ms=NEVER_MS)

    item.press(trigger)
    item.elapse()
    item.press(esc)
    item.wait_for(2)
    item.release(trigger)

    item.press(trigger)
    item.elapse()
    item.release(trigger)

    item.wait_for(4)
    item.drain()
    assert item.calls == ["on_start", "on_cancel", "on_start", "on_stop"]


# --------------------------------------------------------------------------- #
# The macOS shared-modifier-flag quirk. Both option keys set one Alternate bit,
# so pynput can report a sibling's release as OUR release -- and a sibling's
# release as a press. hotkey.py's answer is documented and deliberate: any
# sibling RELEASE ends the hold, and only the exact trigger PRESS arms.
# --------------------------------------------------------------------------- #
def test_pressing_a_sibling_modifier_does_not_arm(rig, fake_keyboard):
    """Left option must not start a recording when right option is the trigger."""
    item = rig(key_name="right_option", min_hold_ms=NEVER_MS)

    item.press(fake_keyboard.Key.alt_l)
    item.hotkey._on_hold_elapsed()

    item.drain()
    assert item.calls == []
    assert item.hotkey._key_down is False


@pytest.mark.parametrize("sibling", ["alt_l", "alt", "alt_gr"])
def test_releasing_a_sibling_modifier_ends_the_hold(rig, fake_keyboard, sibling):
    """Deliberate: the shared flag mask has cleared, so every option key is up."""
    item = rig(key_name="right_option", min_hold_ms=NEVER_MS)

    item.press(fake_keyboard.Key.alt_r)
    item.elapse()
    item.release(getattr(fake_keyboard.Key, sibling))

    item.wait_for(2)
    item.drain()
    assert item.calls == ["on_start", "on_stop"]


def test_a_sibling_release_ends_a_pending_hold_without_firing_anything(
    rig, fake_keyboard
):
    item = rig(key_name="right_option", min_hold_ms=NEVER_MS)

    item.press(fake_keyboard.Key.alt_r)
    item.release(fake_keyboard.Key.alt_l)
    item.hotkey._on_hold_elapsed()

    item.drain()
    assert item.calls == []


@pytest.mark.parametrize("stranger", ["cmd_r", "ctrl_l", "shift_r"])
def test_an_unrelated_modifier_release_does_not_end_the_hold(
    rig, fake_keyboard, stranger
):
    """Sharing no flag with the trigger means it must be ignored entirely."""
    item = rig(key_name="right_option", min_hold_ms=NEVER_MS)

    item.press(fake_keyboard.Key.alt_r)
    item.elapse()
    item.wait_for(1)
    item.release(getattr(fake_keyboard.Key, stranger))

    item.drain()
    assert item.calls == ["on_start"]
    assert item.hotkey.is_armed is True

    item.release(fake_keyboard.Key.alt_r)
    item.wait_for(2)
    assert item.calls == ["on_start", "on_stop"]


def test_the_sibling_group_is_scoped_to_the_triggers_own_family(rig, fake_keyboard):
    item = rig(key_name="right_cmd", min_hold_ms=NEVER_MS)

    assert fake_keyboard.Key.cmd_l in item.hotkey._release_keys
    assert fake_keyboard.Key.alt_l not in item.hotkey._release_keys
    assert fake_keyboard.Key.esc not in item.hotkey._release_keys


def test_left_option_binds_even_though_macos_aliases_it_to_bare_alt(rig):
    """On the real platform Key.alt_l IS Key.alt; binding it must still work."""
    keyboard = FakeKeyboard(macos_shared_left=True)
    item = rig(key_name="left_option", min_hold_ms=NEVER_MS, keyboard=keyboard)

    # The sibling set collapses because two of its four members are one object.
    assert len(item.hotkey._release_keys) == 3

    item.press(keyboard.Key.alt)
    item.elapse()
    item.release(keyboard.Key.alt)

    item.wait_for(2)
    item.drain()
    assert item.calls == ["on_start", "on_stop"]


def test_an_unrelated_key_press_is_ignored(rig, trigger):
    item = rig(min_hold_ms=NEVER_MS)

    item.press(FakeKey("char:a"))
    item.release(FakeKey("char:a"))

    item.drain()
    assert item.calls == []


def test_a_key_object_pynput_cannot_hash_does_not_kill_the_listener(rig):
    """Silence is this module's worst failure mode; a dead listener is silence."""
    item = rig(min_hold_ms=NEVER_MS)

    item.press(["not", "hashable"])  # would raise TypeError out of `key in set`
    item.release(["not", "hashable"])

    item.drain()
    assert item.calls == []


# --------------------------------------------------------------------------- #
# min_hold_ms == 0: opt out of the tap guard entirely, with no timer at all
# --------------------------------------------------------------------------- #
def test_zero_min_hold_arms_on_the_press_itself_with_no_timer(rig, trigger):
    item = rig(min_hold_ms=0)

    item.press(trigger)

    item.wait_for(1)
    assert item.calls == ["on_start"]
    assert item.hotkey._hold_timer is None
    assert item.hotkey.is_armed is True


def test_zero_min_hold_still_pairs_start_with_stop(rig, trigger):
    item = rig(min_hold_ms=0)

    item.press(trigger)
    item.release(trigger)

    item.wait_for(2)
    item.drain()
    assert item.calls == ["on_start", "on_stop"]


def test_zero_min_hold_still_honours_escape(rig, trigger, esc):
    item = rig(min_hold_ms=0)

    item.press(trigger)
    item.press(esc)
    item.release(trigger)

    item.wait_for(2)
    item.drain()
    assert item.calls == ["on_start", "on_cancel"]


def test_a_clamped_negative_min_hold_behaves_like_zero(rig, trigger):
    item = rig(min_hold_ms=-1)

    item.press(trigger)

    item.wait_for(1)
    assert item.hotkey._hold_timer is None
    assert item.calls == ["on_start"]


# --------------------------------------------------------------------------- #
# Callbacks that raise. A blown transcription must not take the hotkey with it.
# --------------------------------------------------------------------------- #
def test_a_raising_on_start_is_logged_and_swallowed(rig, trigger, caplog):
    caplog.set_level(logging.ERROR, logger="blurt.hotkey")
    item = rig(min_hold_ms=NEVER_MS, raise_on=["on_start"])

    item.press(trigger)
    item.elapse()
    item.wait_for(1)
    item.drain()

    assert "on_start" in caplog.text
    assert "raised" in caplog.text


def test_a_raising_on_start_does_not_prevent_its_on_stop(rig, trigger):
    """Otherwise a failed start would strand the recorder open forever."""
    item = rig(min_hold_ms=NEVER_MS, raise_on=["on_start"])

    item.press(trigger)
    item.elapse()
    item.release(trigger)

    item.wait_for(2)
    item.drain()
    assert item.calls == ["on_start", "on_stop"]


def test_a_raising_callback_does_not_wedge_the_worker(rig, trigger):
    item = rig(min_hold_ms=NEVER_MS, raise_on=["on_start", "on_stop", "on_cancel"])

    item.press(trigger)
    item.elapse()
    item.release(trigger)
    item.wait_for(2)

    item.press(trigger)
    item.elapse()
    item.release(trigger)

    item.wait_for(4)
    item.drain()
    assert item.calls == ["on_start", "on_stop"] * 2


def test_a_raising_on_cancel_does_not_wedge_the_worker(rig, trigger, esc):
    item = rig(min_hold_ms=NEVER_MS, raise_on=["on_cancel"])

    item.press(trigger)
    item.elapse()
    item.press(esc)
    item.wait_for(2)
    item.release(trigger)

    item.press(trigger)
    item.elapse()
    item.release(trigger)

    item.wait_for(4)
    item.drain()
    assert item.calls == ["on_start", "on_cancel", "on_start", "on_stop"]


def test_a_raising_callback_leaves_the_hotkey_running(rig, trigger):
    item = rig(min_hold_ms=NEVER_MS, raise_on=["on_start"])

    item.press(trigger)
    item.elapse()
    item.wait_for(1)
    item.drain()

    assert item.hotkey.is_running is True


# --------------------------------------------------------------------------- #
# Threading. Callbacks run on the private worker, never on pynput's listener
# thread, because a callback can block for seconds and the release event that
# ends the recording is queued behind it.
# --------------------------------------------------------------------------- #
def test_callbacks_run_on_the_worker_thread_not_the_caller(rig, trigger):
    item = rig(min_hold_ms=NEVER_MS)

    item.press(trigger)
    item.elapse()
    item.release(trigger)
    item.wait_for(2)

    assert item.recorder.threads == ["blurt-hotkey-worker"] * 2
    assert threading.current_thread().name not in item.recorder.threads


def test_a_slow_on_start_does_not_stall_the_release_event(rig, trigger):
    """The whole reason the queue exists. No sleeps: the callback holds an Event."""
    item = rig(min_hold_ms=NEVER_MS)
    gate = item.recorder.gate("on_start")

    item.press(trigger)
    item.elapse()
    item.wait_for(1)  # on_start has begun and is now blocked in the worker

    item.release(trigger)  # returns even though on_start has not finished
    assert item.calls == ["on_start"]

    gate.set()
    item.wait_for(2)
    assert item.calls == ["on_start", "on_stop"]


def test_the_worker_is_a_named_daemon_thread(rig):
    item = rig()
    worker = item.hotkey._worker
    assert worker.daemon is True
    assert worker.name == "blurt-hotkey-worker"


def test_two_hotkeys_do_not_share_a_worker_or_a_trigger(rig, fake_keyboard):
    """app.py runs dictation and the assistant side by side; they must not cross."""
    dictation = rig(key_name="right_option", min_hold_ms=NEVER_MS)
    assistant = rig(key_name="right_cmd", min_hold_ms=NEVER_MS)

    assert dictation.hotkey._worker is not assistant.hotkey._worker

    dictation.press(fake_keyboard.Key.alt_r)
    dictation.elapse()
    dictation.release(fake_keyboard.Key.alt_r)

    dictation.wait_for(2)
    dictation.drain()
    assistant.drain()
    assert dictation.calls == ["on_start", "on_stop"]
    assert assistant.calls == []


# --------------------------------------------------------------------------- #
# Lifecycle: start, stop, and the shutdown that must not emit anything
# --------------------------------------------------------------------------- #
def test_start_creates_one_listener_wired_to_the_state_machine(rig):
    item = rig()

    assert len(item.keyboard.listeners) == 1
    assert item.listener.on_press == item.hotkey._handle_press
    assert item.listener.on_release == item.hotkey._handle_release
    assert item.listener.started is True
    assert item.hotkey.is_running is True


def test_the_listener_is_started_as_a_daemon(rig):
    """A forgotten stop() must not wedge interpreter shutdown."""
    item = rig()
    assert item.listener.daemon is True


def test_start_twice_is_a_no_op(rig):
    item = rig()
    worker = item.hotkey._worker

    item.hotkey.start()

    assert len(item.keyboard.listeners) == 1
    assert item.hotkey._worker is worker
    assert len(_live_workers()) == 1


def test_start_after_stop_raises_rather_than_half_working(rig):
    item = rig()
    item.hotkey.stop()

    with pytest.raises(RuntimeError) as caught:
        item.hotkey.start()
    assert "construct a new one" in str(caught.value)


def test_stop_twice_is_safe(rig):
    item = rig()
    item.hotkey.stop()
    item.hotkey.stop()
    assert item.hotkey.is_running is False


def test_stop_before_start_is_safe(rig):
    item = rig(start=False)
    item.hotkey.stop()
    assert item.hotkey.is_running is False


def test_stop_stops_and_joins_the_listener(rig):
    item = rig()
    listener = item.listener

    item.hotkey.stop()

    assert listener.stopped is True
    assert listener.joined is True


def test_stop_joins_the_worker_thread(rig):
    item = rig()

    item.hotkey.stop()

    assert item.hotkey._worker is None
    assert _live_workers() == []


def test_stop_while_armed_emits_neither_on_stop_nor_on_cancel(rig, trigger):
    """A shutdown is not a dictation, and it is not a cancellation either."""
    item = rig(min_hold_ms=NEVER_MS)
    item.press(trigger)
    item.elapse()
    item.wait_for(1)

    item.hotkey.stop()  # joins the worker, so the record is final afterwards

    assert item.calls == ["on_start"]


def test_stop_while_pending_emits_nothing(rig, trigger):
    item = rig(min_hold_ms=NEVER_MS)
    item.press(trigger)

    item.hotkey.stop()

    assert item.calls == []


def test_stop_cancels_a_pending_arming_timer(rig, trigger):
    item = rig(min_hold_ms=NEVER_MS)
    item.press(trigger)
    timer = item.hotkey._hold_timer

    item.hotkey.stop()

    assert timer.finished.is_set()
    assert item.hotkey._hold_timer is None


def test_key_events_arriving_after_stop_are_ignored(rig, trigger, esc):
    """pynput can deliver a straggler after stop(); it must not queue work."""
    item = rig(min_hold_ms=NEVER_MS)
    item.hotkey.stop()

    item.press(trigger)
    item.hotkey._on_hold_elapsed()
    item.press(esc)
    item.release(trigger)

    assert item.calls == []
    # Nothing was queued behind the departed worker either.
    assert item.hotkey._jobs.empty()


def test_a_queued_callback_still_runs_during_stop(rig, trigger):
    """Shutdown drains what is already queued: a transcription is not truncated."""
    item = rig(min_hold_ms=NEVER_MS)
    gate = item.recorder.gate("on_start")

    item.press(trigger)
    item.elapse()
    item.wait_for(1)
    item.release(trigger)  # on_stop queued behind the blocked on_start
    gate.set()

    item.hotkey.stop()

    assert item.calls == ["on_start", "on_stop"]


def test_stop_from_inside_a_callback_does_not_deadlock(rig, trigger):
    """Otherwise stop() would join the very thread it is running on."""
    item = rig(min_hold_ms=NEVER_MS)
    finished = threading.Event()

    def stop_from_callback() -> None:
        item.hotkey.stop()
        finished.set()

    item.hotkey._dispatch("test-stop", stop_from_callback)

    assert finished.wait(TIMEOUT_S), "stop() from the worker thread never returned"
    assert item.hotkey.is_running is False


def test_a_listener_that_raises_on_stop_does_not_break_shutdown(rig):
    item = rig()
    item.listener.stop_error = RuntimeError("pynput teardown is noisy")

    item.hotkey.stop()

    assert item.hotkey.is_running is False


def test_a_failed_listener_leaves_no_worker_thread_behind(rig, fake_keyboard):
    """A failed start must not leak a live thread that nothing will ever join."""
    fake_keyboard.listener_error = RuntimeError("no event tap for you")

    with pytest.raises(RuntimeError):
        rig(start=False).hotkey.start()

    assert _wait_until(lambda: _live_workers() == []), "worker thread survived"


def test_a_failed_start_leaves_the_hotkey_not_running(rig, fake_keyboard):
    fake_keyboard.listener_error = RuntimeError("no event tap for you")
    item = rig(start=False)

    with pytest.raises(RuntimeError):
        item.hotkey.start()

    assert item.hotkey.is_running is False
    assert item.hotkey._listener is None
    assert item.hotkey._worker is None


def test_is_armed_tracks_the_hold_and_nothing_else(rig, trigger):
    item = rig(min_hold_ms=NEVER_MS)
    assert item.hotkey.is_armed is False

    item.press(trigger)
    assert item.hotkey.is_armed is False  # pending, not armed

    item.elapse()
    assert item.hotkey.is_armed is True

    item.release(trigger)
    assert item.hotkey.is_armed is False


# --------------------------------------------------------------------------- #
# The Accessibility warning. An untrusted process starts fine and then never
# sees a key, so start() has to say so out loud.
# --------------------------------------------------------------------------- #
def test_start_warns_when_the_process_is_not_trusted(rig, monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="blurt.hotkey")
    monkeypatch.setattr(hotkey_mod, "accessibility_trusted", lambda: False)

    item = rig(key_name="right_option")

    assert "Accessibility" in caplog.text
    assert item.hotkey.key_name in caplog.text
    # Untrusted is not fatal: the listener is up, it just will not hear anything.
    assert item.hotkey.is_running is True


def test_start_says_nothing_when_trust_cannot_be_determined(rig, caplog):
    """On Linux CI the answer is unknowable; a warning there would be noise."""
    caplog.set_level(logging.WARNING, logger="blurt.hotkey")

    rig()

    assert caplog.text == ""


def test_accessibility_trusted_never_raises_on_this_platform():
    """It is a diagnostic; turning it into a crash would defeat the point.

    Calls the REAL probe (captured at import, before the autouse stub replaces
    it) on whatever machine the suite is running on, so it asserts only that the
    answer is one of the three legal ones. Safe everywhere: AXIsProcessTrusted
    without options never shows the Accessibility prompt, and off macOS every
    candidate import simply fails and the function returns None.
    """
    assert _REAL_ACCESSIBILITY_TRUSTED() in (True, False, None)


# --------------------------------------------------------------------------- #
# Known wart, recorded rather than fixed (repo convention: see test_cleanup.py).
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    strict=False,
    reason=(
        "KNOWN WART: _on_hold_elapsed identifies the press it is arming by state "
        "alone, not by identity, so a timer belonging to press #1 can arm press "
        "#2. Reachable whenever a release lands microseconds before its own "
        "arming timer fires and the user presses again before that timer wins "
        "the lock: the stale callback finds _key_down True and _state _PENDING "
        "and arms immediately, so a genuine 0 ms tap fires a full "
        "on_start/on_stop pair. It also clears _hold_timer, orphaning the new "
        "press's timer. A per-press generation counter checked inside "
        "_on_hold_elapsed would close it."
    ),
)
def test_a_stale_arming_timer_does_not_arm_the_next_press(rig, trigger):
    item = rig(min_hold_ms=NEVER_MS)

    item.press(trigger)  # press #1, timer #1 pending
    item.release(trigger)  # a tap: fires nothing, cancels timer #1
    item.press(trigger)  # press #2, timer #2 pending
    stranded = item.hotkey._hold_timer

    # Timer #1's callback, arriving late -- it was already running when the
    # release cancelled it, and only now wins the lock.
    item.hotkey._on_hold_elapsed()

    item.drain()
    try:
        assert item.calls == []
    finally:
        stranded.cancel()  # _on_hold_elapsed dropped the handle; do not leak it
