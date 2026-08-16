"""Tests for :mod:`blurt.app` -- the module that wires the whole app together.

``BlurtApp`` is the only object that knows the order of operations: detect, load,
listen, capture, transcribe, deliver, tear down. Every component below it already
has its own tests, so nothing here re-tests a component. What is tested here is
the WIRING and the ORDERING, because that is where this module can lose something
the user said.

The properties defended, grouped the way the file below is:

  * **Startup order.** The model is resident BEFORE the hotkey is live. A
    regression here does not raise: it makes the first dictation of every session
    swallow its first few seconds, which reads to the user as "this app is
    broken".
  * **Startup failures are :class:`StartupError`, never a traceback.** Every
    failure a user can fix (no engine, model will not load, no microphone, hotkey
    refused) is converted into a message that names the fix. ``__main__`` prints
    it verbatim.
  * **The assistant is a bonus, dictation is the product.** Anything that fails
    while wiring command mode degrades to "assistant off" and startup continues.
  * **The capture claim is indivisible.** The two hotkeys call back on two
    DIFFERENT threads and share one recorder. The headline test hammers
    ``_begin_capture``/``_end_capture`` from both threads through a barrier and
    asserts that no two takes ever overlap and that a capture is delivered under
    the same mode that claimed it -- a dictation coming back tagged "assistant"
    means the user's words were handed to the command router and executed.
  * **Silence never reaches the engine.** macOS answers a denied microphone with
    a working stream of zeros, and Whisper answers silence with confident
    invented sentences. Feeding it is worse than failing.
  * **Delivery never drops text.** When ``insert_text`` fails the words go to the
    clipboard byte-for-byte and the user is told where they went.
  * **"Undo that" gives back the DICTATION, never the undo itself.** The revert
    is only reachable by speaking it, which makes the undo command a capture like
    any other -- so the buffer the revert reads from has to refuse it. That
    refusal lives in ``_handle_capture``, upstream of ``revert_last``, which is
    why calling ``revert_last`` in a test cannot see it break. The section "THE
    WIRED REVERT PATH" drives the whole thing instead: two captures in as PCM,
    the real router in the middle, and an assertion on the string that came back
    out of ``insert_text``.
  * **The revert marker names an EVENT, not a value.** ``revert_last`` asks
    ``transcript is self._reverted_marker``. ``Transcript`` is a frozen
    dataclass, so under ``==`` two dictations of the same short sentence compare
    equal and the second one arrives already reverted. Every other revert test
    in this file speaks two different sentences and cannot see that.
  * **A revert that could not paste is not a revert.** The marker is one-shot per
    dictation, so spending it on a paste macOS refused (Secure Event Input, no
    Accessibility) throws away the user's only route back to their raw text --
    silently, since the premise of command mode is that they are looking at
    another app.
  * **Nothing the user did not ask for is written to disk**, and a journal that
    cannot be written never costs a dictation that was already spoken.
  * **Teardown loses the least**: hotkey first (no new captures), then drain the
    worker so a transcription already in flight still lands.

Everything is faked at the seams :mod:`blurt.app` imports by name -- ``_hardware``,
``select_engine``, ``Recorder``, ``HoldToTalk``, ``insert_text``,
``copy_to_clipboard``, ``accessibility_trusted``, ``secure_input_active``,
``build_default_router``, ``append_record``. No microphone is opened, no model is
downloaded, no key is grabbed, no clipboard is touched and no macOS permission
dialog can appear, so this file runs unattended on Linux CI and on the Intel floor
machine alike.

ONE SEAM IS DELIBERATELY LEFT REAL. ``Wiring.use_real_router()`` lets
``build_default_router`` build the production router, and the revert integration
tests turn it on. Faking the router in those tests would fake away the thing
under test -- which handler claims the words "undo that", and which transcript is
on the tail of the history deque when its handler runs. The router is pure logic
with no macOS dependency, so this costs nothing in portability; its two backends
that do shell out, ``open_app`` and ``notify``, stay faked, so no application is
launched and no notification is posted.

Determinism with threads: the concurrent tests synchronise on a
:class:`threading.Barrier` and on :meth:`queue.Queue.join`, never on a sleep, and
they assert invariants that hold for every interleaving rather than asserting that
a particular interleaving happened. The one place a real thread has to be ordered
against the main thread (shutdown draining) is gated by an
:class:`threading.Event`, so the test cannot pass by winning a race.

Python 3.9 floor: lazy annotations, typing generics only, no PEP 604 unions.
"""

from __future__ import annotations

import queue
import signal
import sys
import threading
import time

import pytest

from blurt import app as blurt_app
from blurt.app import HISTORY_LIMIT, BlurtApp, StartupError
from blurt.assistant import build_default_router as build_real_router
from blurt.assistant import system_actions as real_system_actions
from blurt.assistant.types import Action, ActionResult
from blurt.audio import AudioUnavailable
from blurt.cleanup import clean
from blurt.config import Config
from blurt.engines import NoEngineAvailable
from blurt.hotkey import UnsupportedHotkeyError
from blurt.types import Hardware

# --------------------------------------------------------------------------- #
# Fakes
#
# Every fake records into one shared ``order`` list. Ordering is the thing this
# module gets right or wrong, so the tests need to see the sequence, not just the
# call counts.
# --------------------------------------------------------------------------- #

_UNSET = object()

#: Text with an accent, an em dash and a currency sign. Used wherever a test
#: claims a string arrived "byte-identical": ASCII would pass those assertions
#: even if something in the path re-encoded the text.
_SPOKEN = "meet Zoe at the cafe — 5 € — naïve"


class FakeCapture:
    """Stands in for the numpy PCM buffer a Recorder hands back.

    :meth:`BlurtApp._handle_capture` only ever asks a capture for ``shape[0]``,
    so a real array would buy nothing but a numpy import.

    ``owner`` and ``started_by`` are test-only provenance, stamped by the fake
    recorder when the take began: the mode that held the claim, and the name of
    the thread that pressed. They are what let the race test say something
    stronger than "no exception was raised" -- namely that the words in this
    buffer are delivered under the mode of the press that spoke them.
    """

    def __init__(self, frames: int = 16000, owner=None, started_by: str = "") -> None:
        self.shape = (frames,)
        self.owner = owner
        self.started_by = started_by


class FakeEngine:
    """An ASR engine that never loads anything and answers instantly.

    ``before_transcribe`` is the seam the shutdown test uses to hold a
    transcription open across a teardown without a sleep.
    """

    name = "fake-whisper"

    def __init__(self, order) -> None:
        self.order = order
        self.text = "hello world"
        self.model = "tiny.en"
        self.load_error = None
        self.transcribe_error = None
        self.unload_error = None
        self.before_transcribe = None
        self.loads = 0
        self.unloads = 0
        self.transcribe_calls = []  # list of (pcm, sample_rate)

    def is_available(self) -> bool:
        return True

    def resolve_model(self) -> str:
        return self.model

    def load(self) -> None:
        self.order.append("engine.load")
        self.loads += 1
        if self.load_error is not None:
            raise self.load_error

    def transcribe(self, pcm, sample_rate):
        if self.before_transcribe is not None:
            self.before_transcribe()
        self.order.append("engine.transcribe")
        self.transcribe_calls.append((pcm, sample_rate))
        if self.transcribe_error is not None:
            raise self.transcribe_error
        return self.text

    def unload(self) -> None:
        self.order.append("engine.unload")
        self.unloads += 1
        if self.unload_error is not None:
            raise self.unload_error


class FakeRecorder:
    """One microphone, shared by both hotkeys -- and it says so when it is abused.

    The real Recorder cannot record two takes at once; it would simply mix them
    into one buffer and hand the result to whichever release got there first.
    This fake makes that visible instead: it remembers who owns the current take
    (the mode read straight off ``BlurtApp._capture_mode``, which the app sets
    under its lock immediately before calling ``start()``, plus the name of the
    thread that pressed), records an entry in ``overlaps`` if a second ``start()``
    arrives while a take is already live, and stamps every buffer it returns with
    that provenance.

    A second ``start()`` deliberately does NOT take ownership away from the first.
    That models the thing being defended against: the audio in the buffer was
    spoken by whoever started the take, and if a different press ends it, that
    press walks away with words it never recorded.

    It also records, for each call, whether ``_capture_lock`` was actually held at
    the time. That is the documented invariant of :meth:`BlurtApp._begin_capture`
    ("there is no instant in which the recorder is running unclaimed") and it is
    checked deterministically, so removing the lock fails a test even on a machine
    whose scheduler refuses to lose the race.
    """

    def __init__(self, wiring) -> None:
        self.wiring = wiring
        self.order = wiring.order
        self.frames = 16000
        self.silent = False
        self.overflowed = False
        self.start_error = None
        self.stop_error = None
        self.close_error = None

        self._guard = threading.Lock()
        self._owner = None
        self._started_by = ""
        self.starts = []  # owning mode of every successful start
        self.stops = 0
        self.closed = 0
        self.overlaps = []  # (owner already recording, mode that barged in)
        self.orphan_stops = 0  # stop() while nothing was recording
        self.claim_held_on_start = []
        self.claim_held_on_stop = []
        self.claim_held_on_flag_read = []

    # -- probes into the app under test ------------------------------------
    def _claiming_mode(self):
        app = self.wiring.app
        return None if app is None else app._capture_mode

    def _claim_held(self):
        app = self.wiring.app
        return None if app is None else app._capture_lock.locked()

    # -- Recorder surface ---------------------------------------------------
    def start(self) -> None:
        self.order.append("recorder.start")
        self.claim_held_on_start.append(self._claim_held())
        if self.start_error is not None:
            raise self.start_error
        with self._guard:
            claimer = self._claiming_mode()
            self.starts.append(claimer)
            if self._owner is not None:
                self.overlaps.append((self._owner, claimer))
                return
            self._owner = claimer
            self._started_by = threading.current_thread().name

    def stop(self):
        self.order.append("recorder.stop")
        self.claim_held_on_stop.append(self._claim_held())
        with self._guard:
            owner, started_by = self._owner, self._started_by
            self._owner, self._started_by = None, ""
            self.stops += 1
            if owner is None:
                self.orphan_stops += 1
        if self.stop_error is not None:
            raise self.stop_error
        return FakeCapture(self.frames, owner, started_by)

    def last_capture_was_silent(self) -> bool:
        self.claim_held_on_flag_read.append(self._claim_held())
        return self.silent

    def last_capture_overflowed(self) -> bool:
        self.claim_held_on_flag_read.append(self._claim_held())
        return self.overflowed

    def close(self) -> None:
        self.order.append("recorder.close")
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error

    @property
    def is_recording(self) -> bool:
        return self._owner is not None


class FakeHotkey:
    """A HoldToTalk that grabs no keys. Its callbacks are exposed for the tests."""

    def __init__(self, wiring, role, key_name, on_start, on_stop, on_cancel, min_hold_ms):
        self.wiring = wiring
        self.role = role  # "hotkey" (dictation) or "assistant"
        self.key_name = key_name
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_cancel = on_cancel
        self.min_hold_ms = min_hold_ms
        self.start_error = None
        self.stop_error = None
        self.started = 0
        self.stopped = 0

    def start(self) -> None:
        self.wiring.order.append(self.role + ".start")
        if self.start_error is not None:
            raise self.start_error
        self.started += 1

    def stop(self) -> None:
        self.wiring.order.append(self.role + ".stop")
        self.wiring.hotkey_stopped.set()
        if self.stop_error is not None:
            raise self.stop_error
        self.stopped += 1

    def is_running(self) -> bool:
        return self.started > self.stopped


class FakeHardwareModule:
    """Stands in for the :mod:`blurt.hardware` module, which app.py imports whole.

    A module rather than a function because that is the seam: app.py calls
    ``_hardware.detect()``, so replacing the name has to hand back something with
    a ``detect``. Probing the real machine would work on CI too -- and would make
    every assertion about what was detected depend on the runner.
    """

    def __init__(self, wiring) -> None:
        self.wiring = wiring

    def detect(self):
        self.wiring.order.append("hardware.detect")
        self.wiring.detected += 1
        return self.wiring.hardware


class FakeRouter:
    """Records what it was asked to route and executes nothing real."""

    def __init__(self) -> None:
        self.routed = []
        self.executed = []
        self.route_error = None
        self.action = Action(kind="timer", summary="Set a timer for 5 minutes")
        self.result = ActionResult(ok=True, message="Timer set for 5 minutes")

    def route(self, text):
        self.routed.append(text)
        if self.route_error is not None:
            raise self.route_error
        return self.action

    def execute(self, action):
        self.executed.append(action)
        return self.result


class Wiring:
    """Every seam of :mod:`blurt.app`, replaced and observable.

    One object rather than a dozen fixtures because these parts only make sense
    together: a test that wants a paste to fail also wants to see where the text
    went, and a test about startup order wants one list with every step in it.
    """

    def __init__(self, monkeypatch) -> None:
        self.monkeypatch = monkeypatch
        self.order = []
        self.app = None
        self.apps = []

        self.hardware = Hardware(
            arch="x86_64",
            is_apple_silicon=False,
            under_rosetta=False,
            cpu_brand="Intel Core i5",
            physical_cores=2,
            ram_gb=8.0,
            macos_version=(13, 6, 1),
            tier="slow",
        )
        self.detected = 0

        self.engine = FakeEngine(self.order)
        self.select_error = None
        self.select_calls = []

        self.recorder = FakeRecorder(self)
        self.recorder_error = None
        self.recorder_kwargs = []

        self.hotkeys = []
        self.hotkey_construct_errors = {}  # key_name -> exception raised by HoldToTalk()
        self.hotkey_start_errors = {}  # key_name -> exception raised by .start()
        self.hotkey_stopped = threading.Event()

        self.router = FakeRouter()
        self.router_error = None
        self.router_kwargs = []
        # Flipped by use_real_router(). When set, _build_router hands back the
        # router build_default_router really builds instead of FakeRouter, and
        # the two lists below record what its macOS backends were asked to do
        # rather than letting them shell out.
        self.real_router = False
        self.opened_apps = []
        self.notifications = []

        self.insert_ok = True
        self.inserted = []
        self.insert_kwargs = []
        self.clipboard = []
        self.accessibility = None
        self.secure_input = None

        self.journal = []
        self.journal_error = None

        self._install()

    # -- installation -------------------------------------------------------
    def _install(self) -> None:
        set_ = self.monkeypatch.setattr

        set_(blurt_app, "_hardware", FakeHardwareModule(self))
        set_(blurt_app, "select_engine", self._select_engine)
        set_(blurt_app, "Recorder", self._open_recorder)
        set_(blurt_app, "HoldToTalk", self._build_hotkey)
        set_(blurt_app, "insert_text", self._insert_text)
        set_(blurt_app, "copy_to_clipboard", self._copy_to_clipboard)
        set_(blurt_app, "accessibility_trusted", lambda: self.accessibility)
        set_(blurt_app, "secure_input_active", lambda: self.secure_input)
        set_(blurt_app, "build_default_router", self._build_router)
        set_(blurt_app, "append_record", self._append_record)

    def use_real_router(self) -> "Wiring":
        """Stop faking the router. Must be called BEFORE ``startup()``.

        The fake router is the right choice for every test about wiring: those
        ask what the app handed to ``build_default_router`` and what it did with
        the result, and a real router would only add intent-matching noise to the
        answer.

        It is exactly the wrong choice for a test about the revert, because a
        fake router answers by fiat the two questions that path can get wrong --
        which handler claims the words "undo that", and which transcript is on
        the tail of the history deque by the time that handler runs. Both live in
        the seam between the capture pipeline and the router, and a fake router
        IS that seam, so a test built on one can only assert its own setup back
        to itself. So this swaps in the real :class:`IntentRouter`, built by the
        real ``build_default_router`` with the real handlers in the real order.

        Two backends stay faked. ``open_app`` runs ``open -a`` and ``notify``
        runs ``osascript``; on this machine both would just fail slowly, and on a
        developer's Mac the first would genuinely launch Safari. Each is imported
        by name INSIDE the function that uses it (``build_default_router`` and
        ``BlurtApp._handle_command`` respectively), never at module import, so
        replacing the attribute on the module is enough to catch both -- and it
        has to happen before startup, because that is when the handler captures
        its reference to ``open_app``.

        Returns self so it can be chained onto a ``wiring`` in one line.
        """
        self.real_router = True
        self.monkeypatch.setattr(real_system_actions, "open_app", self._open_app)
        self.monkeypatch.setattr(real_system_actions, "notify", self._notify)
        return self

    def _select_engine(self, cfg, hw):
        self.order.append("select_engine")
        self.select_calls.append((cfg, hw))
        if self.select_error is not None:
            raise self.select_error
        return self.engine

    def _open_recorder(self, sample_rate=16000, preroll_ms=500):
        self.order.append("recorder.open")
        self.recorder_kwargs.append({"sample_rate": sample_rate, "preroll_ms": preroll_ms})
        if self.recorder_error is not None:
            raise self.recorder_error
        return self.recorder

    def _build_hotkey(self, key_name, on_start, on_stop, on_cancel, min_hold_ms=200):
        error = self.hotkey_construct_errors.get(key_name)
        if error is not None:
            raise error
        role = "hotkey" if not self.hotkeys else "assistant"
        hotkey = FakeHotkey(self, role, key_name, on_start, on_stop, on_cancel, min_hold_ms)
        hotkey.start_error = self.hotkey_start_errors.get(key_name)
        self.hotkeys.append(hotkey)
        return hotkey

    def _insert_text(self, text, paste_delay_ms=120, restore_delay_ms=400):
        self.inserted.append(text)
        self.insert_kwargs.append(
            {"paste_delay_ms": paste_delay_ms, "restore_delay_ms": restore_delay_ms}
        )
        return self.insert_ok

    def _copy_to_clipboard(self, text):
        self.clipboard.append(text)

    def _build_router(self, dictate_fallback, now_fn=None, revert_fn=None):
        self.router_kwargs.append(
            {"dictate_fallback": dictate_fallback, "now_fn": now_fn, "revert_fn": revert_fn}
        )
        if self.router_error is not None:
            raise self.router_error
        if self.real_router:
            # Exactly the call the app would have made had this seam never been
            # faked: the kwargs are recorded above and then handed straight on,
            # untouched. Anything else here would let a test pass against a
            # router the app does not actually build.
            self.router = build_real_router(
                dictate_fallback=dictate_fallback, now_fn=now_fn, revert_fn=revert_fn
            )
        return self.router

    def _open_app(self, name):
        """Stands in for ``system_actions.open_app``: records, launches nothing."""
        self.opened_apps.append(name)
        return ActionResult(ok=True, message="Opened {0}".format(name))

    def _notify(self, title, message):
        """Stands in for ``system_actions.notify``: records, posts nothing."""
        self.notifications.append((title, message))

    def _append_record(self, record, limit=2000):
        self.journal.append(record)
        if self.journal_error is not None:
            raise self.journal_error
        return True

    # -- app construction ---------------------------------------------------
    def build(self, hw=_UNSET, **cfg_kwargs) -> BlurtApp:
        """A BlurtApp with hardware already injected (nothing probes the machine)."""
        settings = {"assistant_enabled": False}
        settings.update(cfg_kwargs)
        app = BlurtApp(Config(**settings), self.hardware if hw is _UNSET else hw)
        self.app = app
        self.apps.append(app)
        return app

    def started(self, hw=_UNSET, **cfg_kwargs) -> BlurtApp:
        app = self.build(hw=hw, **cfg_kwargs)
        app.startup()
        return app

    @property
    def dictation_hotkey(self) -> FakeHotkey:
        return self.hotkeys[0]

    @property
    def assistant_hotkey(self) -> FakeHotkey:
        return self.hotkeys[1]


@pytest.fixture
def wiring(monkeypatch):
    """Fake out every seam, and always tear the app down again.

    The teardown matters: :meth:`BlurtApp.startup` starts a real (daemon) worker
    thread, and a test file that leaks one per test would eventually be measuring
    the thread table rather than blurt.
    """
    bundle = Wiring(monkeypatch)
    yield bundle
    for app in bundle.apps:
        try:
            app.shutdown()
        except Exception:  # noqa: BLE001 - teardown must never mask a failure
            pass


def _drain(jobs):
    """Everything currently queued for the transcription worker, in order."""
    drained = []
    while True:
        try:
            drained.append(jobs.get_nowait())
        except queue.Empty:
            return drained


def _assert_actionable(message: str, cause: str = "") -> None:
    """A StartupError has to leave the user holding something they can act on.

    Two shapes count. Either the message names the remedy outright ("Fix: grant
    Input Monitoring..."), or -- where the fix depends on a failure blurt cannot
    interpret for the user -- it reproduces the underlying error verbatim, which
    is the string they will paste into a search box. What it may never be is a
    shrug: ``__main__`` prints this instead of a traceback, so an empty or
    contentless message leaves the user with strictly less than a stack trace
    would have given them.
    """
    assert message.strip(), "a StartupError with an empty message tells the user nothing"
    assert len(message.split()) >= 6, "too terse to act on: {0!r}".format(message)
    lowered = message.lower()
    assert any(
        phrase in lowered
        for phrase in ("could not", "cannot", "unavailable", "no usable", "no engine")
    ), "never says what failed: {0!r}".format(message)
    names_a_remedy = any(
        word in lowered
        for word in ("fix:", "grant", "install", "check", "enable", "network")
    )
    quotes_the_cause = bool(cause) and cause in message
    assert names_a_remedy or quotes_the_cause, (
        "neither a remedy nor the underlying cause: {0!r}".format(message)
    )


# --------------------------------------------------------------------------- #
# Startup: the order of operations IS the feature
# --------------------------------------------------------------------------- #
def test_the_model_is_loaded_before_the_hotkey_can_fire(wiring):
    """The whole reason startup blocks: no keypress may arrive before the model."""
    app = wiring.started()

    assert "engine.load" in wiring.order
    assert "hotkey.start" in wiring.order
    assert wiring.order.index("engine.load") < wiring.order.index("hotkey.start"), (
        "the hotkey went live before the model was resident; the first dictation "
        "of every session would be swallowed"
    )
    assert app.engine_label == "fake-whisper tiny.en"


def test_startup_detects_the_machine_before_choosing_an_engine_for_it(wiring):
    app = wiring.build(hw=None)
    app.startup()

    steps = [step for step in wiring.order if step in ("hardware.detect", "select_engine", "engine.load")]
    assert steps == ["hardware.detect", "select_engine", "engine.load"]
    # And the engine was chosen for the machine we actually detected, not a guess.
    _cfg, hw = wiring.select_calls[0]
    assert hw is wiring.hardware
    assert app.hw is wiring.hardware


def test_injected_hardware_is_never_re_probed(wiring):
    wiring.started()
    assert wiring.detected == 0


def test_the_recorder_and_hotkey_are_built_from_the_user_config(wiring):
    wiring.started(sample_rate=22050, preroll_ms=250, min_hold_ms=333, hotkey="right_ctrl")

    assert wiring.recorder_kwargs == [{"sample_rate": 22050, "preroll_ms": 250}]
    assert wiring.dictation_hotkey.key_name == "right_ctrl"
    assert wiring.dictation_hotkey.min_hold_ms == 333


def test_a_second_startup_does_nothing_at_all(wiring):
    app = wiring.started()
    before = list(wiring.order)

    app.startup()

    assert wiring.order == before
    assert wiring.engine.loads == 1
    assert wiring.dictation_hotkey.started == 1


def test_an_untrusted_process_is_warned_about_but_still_starts(wiring, capsys):
    """Accessibility can be undeterminable, and may just need a relaunch.

    Aborting here would be wrong more often than right, so the rule is: say it
    loudly, keep running.
    """
    wiring.accessibility = False
    app = wiring.started()

    assert app._started is True
    err = capsys.readouterr().err
    assert "Accessibility" in err
    assert "hotkey will never fire" in err


# --------------------------------------------------------------------------- #
# Startup failures: a StartupError with a remedy, never a bare exception
# --------------------------------------------------------------------------- #
def test_no_usable_engine_becomes_a_startup_error(wiring):
    wiring.select_error = NoEngineAvailable(
        "No usable speech engine.\n  Fix: python3 -m pip install faster-whisper"
    )
    app = wiring.build()

    with pytest.raises(StartupError) as excinfo:
        app.startup()

    message = str(excinfo.value)
    _assert_actionable(message)
    # The registry already wrote the user-facing explanation; it must survive
    # the trip rather than being replaced by something vaguer.
    assert "python3 -m pip install faster-whisper" in message


def test_a_model_that_will_not_load_becomes_a_startup_error(wiring):
    wiring.engine.load_error = OSError("model file is truncated")
    app = wiring.build()

    with pytest.raises(StartupError) as excinfo:
        app.startup()

    message = str(excinfo.value)
    _assert_actionable(message)
    assert "fake-whisper tiny.en" in message
    assert "model file is truncated" in message
    # First run downloads weights; a long pause or a network failure has to be
    # explained here or it looks like a hang.
    assert "network" in message


def test_a_model_loader_that_raises_a_baseexception_is_still_a_startup_error(wiring):
    """Loader failures are not politely limited to Exception subclasses."""

    class Fatal(BaseException):
        pass

    wiring.engine.load_error = Fatal("ctranslate2 aborted")
    app = wiring.build()

    with pytest.raises(StartupError):
        app.startup()


def test_a_missing_microphone_becomes_a_startup_error(wiring):
    wiring.recorder_error = AudioUnavailable(
        "no input device found; check System Settings > Sound"
    )
    app = wiring.build()

    with pytest.raises(StartupError) as excinfo:
        app.startup()

    message = str(excinfo.value)
    _assert_actionable(message, cause="no input device found")
    assert "Microphone unavailable" in message


def test_any_other_microphone_failure_becomes_a_startup_error(wiring):
    """PortAudio raises broadly; none of it may reach the user as a traceback."""
    wiring.recorder_error = RuntimeError("PortAudio: Invalid device")
    app = wiring.build()

    with pytest.raises(StartupError) as excinfo:
        app.startup()

    message = str(excinfo.value)
    _assert_actionable(message, cause="PortAudio: Invalid device")
    assert "RuntimeError" in message


def test_a_hotkey_that_cannot_be_expressed_becomes_a_startup_error(wiring):
    wiring.hotkey_construct_errors["right_option"] = UnsupportedHotkeyError(
        "pynput cannot distinguish left from right on this platform"
    )
    app = wiring.build()

    with pytest.raises(StartupError) as excinfo:
        app.startup()

    message = str(excinfo.value)
    _assert_actionable(message, cause="pynput cannot distinguish left from right")
    assert "right_option" in message


def test_a_missing_pynput_becomes_a_startup_error_naming_the_install(wiring):
    wiring.hotkey_construct_errors["right_option"] = ImportError("No module named 'pynput'")
    app = wiring.build()

    with pytest.raises(StartupError) as excinfo:
        app.startup()

    message = str(excinfo.value)
    _assert_actionable(message)
    assert "pip install pynput" in message


def test_a_hotkey_listener_that_will_not_start_becomes_a_startup_error(wiring):
    app = wiring.build()
    wiring.hotkey_start_errors["right_option"] = OSError(
        "this process is not trusted to monitor input"
    )

    with pytest.raises(StartupError) as excinfo:
        app.startup()

    message = str(excinfo.value)
    _assert_actionable(message)
    assert "Input Monitoring" in message
    assert "Accessibility" in message


def test_a_failed_startup_never_reports_itself_as_started(wiring):
    wiring.engine.load_error = OSError("nope")
    app = wiring.build()

    with pytest.raises(StartupError):
        app.startup()

    assert app._started is False


def test_a_failed_startup_can_still_be_shut_down_cleanly(wiring):
    """The worker thread is already running when the hotkey fails to start."""
    app = wiring.build()
    wiring.hotkey_start_errors["right_option"] = OSError("untrusted")

    with pytest.raises(StartupError):
        app.startup()
    worker = app._worker

    app.shutdown()

    assert app._worker is None
    assert worker is not None
    worker.join(timeout=5.0)
    assert not worker.is_alive()


# --------------------------------------------------------------------------- #
# The assistant is an addition to dictation, never a precondition for it
# --------------------------------------------------------------------------- #
def test_an_assistant_hotkey_that_will_not_start_leaves_dictation_running(wiring, capsys):
    app = wiring.build(assistant_enabled=True, assistant_hotkey="right_cmd")
    wiring.hotkey_start_errors["right_cmd"] = OSError("cmd is taken by something else")

    app.startup()

    assert app._started is True
    assert app._assistant_hotkey is None
    assert wiring.dictation_hotkey.started == 1
    assert "assistant hotkey unavailable" in capsys.readouterr().err


def test_a_router_that_will_not_build_leaves_dictation_running(wiring, capsys):
    wiring.router_error = RuntimeError("EventKit is missing")
    app = wiring.build(assistant_enabled=True, assistant_hotkey="right_cmd")

    app.startup()

    assert app._started is True
    assert app._router is None
    assert app._assistant_hotkey is None
    assert len(wiring.hotkeys) == 1, "no assistant key should be bound without a router"
    assert wiring.dictation_hotkey.started == 1
    assert "assistant unavailable" in capsys.readouterr().err


def test_an_unsupported_assistant_hotkey_leaves_dictation_running(wiring, capsys):
    wiring.hotkey_construct_errors["right_cmd"] = UnsupportedHotkeyError("unknown key")
    app = wiring.build(assistant_enabled=True, assistant_hotkey="right_cmd")

    app.startup()

    assert app._started is True
    assert app._assistant_hotkey is None
    assert app._router is None
    assert wiring.dictation_hotkey.started == 1
    assert "unsupported" in capsys.readouterr().err


def test_an_assistant_hotkey_equal_to_the_dictation_hotkey_is_refused(wiring, capsys):
    """Binding both modes to one key would make every dictation a coin flip."""
    app = wiring.build(
        assistant_enabled=True, hotkey="right_option", assistant_hotkey="right_option"
    )

    app.startup()

    assert len(wiring.hotkeys) == 1
    assert app._assistant_hotkey is None
    assert app._router is None
    err = capsys.readouterr().err
    assert "equals the dictation hotkey" in err
    assert "disabled" in err


def test_an_unset_assistant_hotkey_disables_the_assistant_without_scolding(wiring, capsys):
    app = wiring.build(assistant_enabled=True, assistant_hotkey="")

    app.startup()

    assert app._assistant_hotkey is None
    assert "conflict" not in capsys.readouterr().err


def test_a_disabled_assistant_builds_no_router_at_all(wiring):
    wiring.started(assistant_enabled=False)
    assert wiring.router_kwargs == []
    assert len(wiring.hotkeys) == 1


def test_the_assistant_is_given_a_way_to_dictate_and_to_revert(wiring):
    """The router's fallback pastes, and revert is reachable only through it."""
    app = wiring.started(assistant_enabled=True, assistant_hotkey="right_cmd")

    kwargs = wiring.router_kwargs[0]
    assert kwargs["revert_fn"] == app.revert_last

    result = kwargs["dictate_fallback"]("just some words")
    assert result.ok is True
    assert wiring.inserted == ["just some words"]


# --------------------------------------------------------------------------- #
# THE CAPTURE CLAIM
#
# Two hotkeys, two dispatch threads, one microphone. The claim on _capture_mode
# is what stops a dictation from being delivered to the command router.
# --------------------------------------------------------------------------- #
def _widen_the_claim_window(monkeypatch) -> None:
    """Give the scheduler somewhere to preempt inside the claim, deliberately.

    Without this, the test below is theatre on CPython. The unsynchronised
    version of :meth:`BlurtApp._begin_capture` reads ``_capture_mode`` and writes
    it back with no call and no backward jump in between, and CPython only
    considers handing the GIL to another thread at points like those -- so on
    this interpreter the window is not merely narrow, it is effectively closed.
    Measured: with the lock removed and nothing else changed, 1500 colliding
    rounds produced exactly zero corruptions. The race is still real -- another
    interpreter, a GC pass, a signal, or one more line of code landing in that
    gap reopens it, and the payload is a dictation executed as a spoken command
    -- so a test that waits for the scheduler to volunteer would pass today,
    pass tomorrow, and be worthless on the day it mattered.

    So for the duration of one test the attribute is stored behind a property
    that yields -- ``time.sleep(0)`` drops the GIL and offers the interpreter to
    whoever else wants it -- on every read and every write. Nothing about the
    app's behaviour changes; the same accesses happen in the same order. All that
    changes is that the gap between "is anything recording?" and "it is mine now"
    becomes wide enough for the other hotkey to walk through it.

    With the claim taken under ``_capture_lock`` this is a non-event: both
    accesses are inside the critical section, so a yield there costs a context
    switch and nothing else. With the lock removed, the collision was observed on
    299 of 300 rounds -- which is what "fails loudly" has to mean for a test
    guarding a race.
    """

    def _get(self):
        time.sleep(0)
        return self.__dict__.get("_probed_capture_mode")

    def _set(self, value):
        time.sleep(0)
        self.__dict__["_probed_capture_mode"] = value

    monkeypatch.setattr(BlurtApp, "_capture_mode", property(_get, _set), raising=False)


def test_two_hotkeys_racing_never_overlap_and_never_swap_modes(wiring, monkeypatch):
    """The headline: hammer both capture paths from two threads and hold the line.

    Both hotkeys press and release through a :class:`threading.Barrier` so the two
    threads are released at the same instant instead of politely taking turns, and
    the round is closed by a second barrier so every iteration starts from a clean
    slate (nothing claimed, nothing recording). What is asserted is what must NOT
    happen:

      * no take ever overlaps another -- one recorder cannot serve two takes, and
        the buffer it would hand back is a dictation and a command mixed into one;
      * nothing is ever stopped that was not recording, and nothing records
        without a claim;
      * every capture handed to the worker carries the SAME mode that claimed the
        recorder for it. A dictation arriving as "assistant" is executed as a
        command instead of typed -- silent, unrecoverable, and precisely the bug
        the claim exists to prevent.

    How many of the 2N presses win a claim is up to the scheduler -- a round in
    which one thread finishes before the other starts yields two perfectly good
    takes -- so the count is asserted as a range rather than a number. Everything
    that is asserted as an equality holds for every possible interleaving, which
    is what keeps this from being a flaky test about the scheduler.

    See :func:`_widen_the_claim_window` for why the collision is manufactured
    rather than hoped for.
    """
    _widen_the_claim_window(monkeypatch)
    app = wiring.started(assistant_enabled=True, assistant_hotkey="right_cmd")

    # (thread that recorded the audio, mode it was finally delivered under)
    delivered = []

    def record(pcm, was_silent, overflowed, mode="dictate"):
        delivered.append((pcm.started_by, "press-" + mode))

    monkeypatch.setattr(app, "_handle_capture", record)

    rounds = 300
    barrier = threading.Barrier(2, timeout=10.0)
    failures = []

    def press(mode):
        try:
            for _ in range(rounds):
                barrier.wait()
                app._begin_capture(mode)
                app._end_capture(mode)
                barrier.wait()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            failures.append(exc)
            barrier.abort()

    threads = [
        threading.Thread(target=press, args=(mode,), name="press-" + mode)
        for mode in ("dictate", "assistant")
    ]
    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60.0)
    finally:
        sys.setswitchinterval(previous_interval)

    assert failures == []
    assert [thread.is_alive() for thread in threads] == [False, False]

    # At most one capture is ever claimed at a time.
    assert wiring.recorder.overlaps == [], (
        "two presses held the one microphone at the same time: {0!r}".format(
            wiring.recorder.overlaps[:5]
        )
    )
    assert wiring.recorder.orphan_stops == 0, "a take was stopped that nobody had started"
    assert wiring.recorder.starts.count(None) == 0, "the microphone ran unclaimed"
    assert set(wiring.recorder.claim_held_on_start) == {True}
    assert set(wiring.recorder.claim_held_on_stop) == {True}

    app._jobs.join()
    # ...and the words each press recorded come back under that press's own mode.
    swapped = [pair for pair in delivered if pair[0] != pair[1]]
    assert swapped == [], (
        "audio recorded by one hotkey was delivered under the other's mode "
        "(recorded_by, delivered_as): {0!r}".format(swapped[:5])
    )

    # At least one press per round wins the claim -- both cannot lose, since a
    # loser only ever loses to the other one -- and at most both do.
    assert rounds <= len(delivered) <= 2 * rounds
    # Every take that started was stopped exactly once and delivered exactly once.
    assert len(wiring.recorder.starts) == len(delivered)
    assert wiring.recorder.stops == len(delivered)
    assert app._capture_mode is None


def test_the_recorder_is_only_ever_started_with_the_claim_held(wiring):
    """No instant exists in which the microphone is running unclaimed."""
    app = wiring.started()

    app._begin_capture("dictate")

    assert wiring.recorder.claim_held_on_start == [True]
    assert app._capture_mode == "dictate"


def test_the_silence_verdict_is_read_before_the_claim_is_released(wiring):
    """Those flags are per-recorder state that the next start() overwrites.

    Reading them after releasing the claim would let a press landing in that
    window hand this capture somebody else's verdict -- and a wrong "nothing was
    heard" sends the user to a permissions dialog for a dictation that was fine.
    """
    app = wiring.started()
    app._begin_capture("dictate")

    app._end_capture("dictate")

    assert wiring.recorder.claim_held_on_stop == [True]
    assert wiring.recorder.claim_held_on_flag_read == [True, True]


def test_a_second_press_while_recording_is_ignored(wiring):
    app = wiring.started()

    app._begin_capture("dictate")
    app._begin_capture("assistant")

    assert len(wiring.recorder.starts) == 1
    assert app._capture_mode == "dictate"


def test_the_other_hotkeys_release_cannot_end_this_capture(wiring):
    app = wiring.started()
    app._begin_capture("dictate")

    app._end_capture("assistant")

    assert wiring.recorder.stops == 0
    assert wiring.recorder.is_recording is True
    assert app._capture_mode == "dictate"
    assert _drain(app._jobs) == []


def test_the_other_hotkeys_cancel_cannot_kill_this_capture(wiring):
    """Esc during a command must never throw away a dictation the other key owns."""
    app = wiring.started()
    app._begin_capture("dictate")

    app._cancel_capture("assistant")

    assert wiring.recorder.stops == 0
    assert app._capture_mode == "dictate"


def test_a_cancelled_capture_is_never_transcribed(wiring, capsys):
    app = wiring.started()
    app._begin_capture("dictate")

    app._cancel_capture("dictate")

    assert wiring.recorder.stops == 1
    assert app._capture_mode is None
    assert _drain(app._jobs) == []
    assert "cancelled" in capsys.readouterr().out


def test_a_microphone_that_will_not_start_does_not_wedge_the_app(wiring, capsys):
    """One failed press must not lock out every press for the life of the process."""
    app = wiring.started()
    wiring.recorder.start_error = AudioUnavailable("device disappeared")

    app._begin_capture("dictate")

    assert app._capture_mode is None
    assert "device disappeared" in capsys.readouterr().err

    wiring.recorder.start_error = None
    app._begin_capture("dictate")
    assert app._capture_mode == "dictate"


def test_a_capture_that_fails_to_stop_queues_nothing_and_frees_the_claim(wiring, capsys):
    app = wiring.started()
    app._begin_capture("dictate")
    wiring.recorder.stop_error = RuntimeError("stream died")

    app._end_capture("dictate")

    assert app._capture_mode is None
    assert _drain(app._jobs) == []
    assert "capture failed" in capsys.readouterr().err


def test_capture_callbacks_before_startup_do_nothing(wiring):
    """Callbacks can only arrive after startup, but they must be harmless if not."""
    app = wiring.build()

    app._begin_capture("dictate")
    app._end_capture("dictate")
    app._cancel_capture("dictate")

    assert app._capture_mode is None
    assert _drain(app._jobs) == []


def test_the_hotkey_callbacks_are_wired_to_the_right_modes(wiring):
    app = wiring.started(assistant_enabled=True, assistant_hotkey="right_cmd")

    wiring.assistant_hotkey.on_start()
    assert app._capture_mode == "assistant"
    wiring.assistant_hotkey.on_cancel()
    assert app._capture_mode is None

    wiring.dictation_hotkey.on_start()
    assert app._capture_mode == "dictate"
    wiring.dictation_hotkey.on_stop()
    assert app._capture_mode is None
    assert [job[3] for job in _drain(app._jobs)] == ["dictate"]


# --------------------------------------------------------------------------- #
# What must NOT reach the engine
# --------------------------------------------------------------------------- #
def test_an_empty_capture_never_reaches_the_engine(wiring, capsys):
    app = wiring.started()

    app._handle_capture(FakeCapture(frames=0), False, False, "dictate")

    assert wiring.engine.transcribe_calls == []
    assert wiring.inserted == []
    assert "no audio captured" in capsys.readouterr().err


def test_silence_never_reaches_the_engine(wiring):
    """Whisper answers silence with confident invented sentences.

    A denied microphone on macOS does not raise; it hands back a working stream
    of zeros. Transcribing that would type a fabricated sentence into whatever
    the user had focused, which is the worst outcome this app has.
    """
    app = wiring.started()

    app._handle_capture(FakeCapture(), True, False, "dictate")

    assert wiring.engine.transcribe_calls == []
    assert wiring.inserted == []
    assert app.history == []


def test_silence_is_reported_as_a_permission_problem(wiring, capsys):
    app = wiring.started()

    app._handle_capture(FakeCapture(), True, False, "dictate")

    err = capsys.readouterr().err
    assert "Nothing was heard" in err
    assert "microphone permission" in err.lower()
    assert "Privacy & Security" in err


def test_a_transcript_that_cleans_away_to_nothing_is_not_delivered(wiring, capsys):
    app = wiring.started()
    wiring.engine.text = "   "

    app._handle_capture(FakeCapture(), False, False, "dictate")

    assert wiring.inserted == []
    assert "no speech was recognised" in capsys.readouterr().err


def test_an_engine_that_raises_delivers_nothing_and_says_so(wiring, capsys):
    app = wiring.started()
    wiring.engine.transcribe_error = RuntimeError("ctranslate2 blew up")

    app._handle_capture(FakeCapture(), False, False, "dictate")

    assert wiring.inserted == []
    assert wiring.clipboard == []
    assert app.history == []
    assert "transcription failed" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #
def test_a_dictation_is_transcribed_cleaned_recorded_and_typed(wiring):
    app = wiring.started()
    wiring.engine.text = "hello world"
    expected = clean("hello world", app.cfg.cleanup_level, app.cfg.dictionary)

    app._handle_capture(FakeCapture(frames=32000), False, False, "dictate")

    pcm, sample_rate = wiring.engine.transcribe_calls[0]
    assert pcm.shape == (32000,)
    assert sample_rate == app.cfg.sample_rate

    assert wiring.inserted == [expected]
    assert len(app.history) == 1
    transcript = app.last_transcript
    assert transcript.raw == "hello world"
    assert transcript.cleaned == expected
    assert transcript.engine == "fake-whisper tiny.en"
    assert transcript.audio_seconds == pytest.approx(2.0)


def test_the_paste_timings_come_from_the_config(wiring):
    app = wiring.started(paste_delay_ms=45, clipboard_restore_ms=90)

    app._handle_capture(FakeCapture(), False, False, "dictate")

    assert wiring.insert_kwargs == [{"paste_delay_ms": 45, "restore_delay_ms": 90}]


def test_raw_text_is_dropped_from_history_when_the_user_asked_for_that(wiring):
    app = wiring.started(keep_raw_history=False)

    app._handle_capture(FakeCapture(), False, False, "dictate")

    assert app.last_transcript.raw == ""
    assert app.last_transcript.cleaned != ""


def test_a_dropped_frame_warning_does_not_stop_delivery(wiring, capsys):
    app = wiring.started()

    app._handle_capture(FakeCapture(), False, True, "dictate")

    assert wiring.inserted != []
    assert "dropped frames" in capsys.readouterr().err


def test_history_stays_a_small_buffer_not_an_archive(wiring):
    """Dictated speech is exactly the thing that must not accumulate in RAM."""
    app = wiring.started()

    for index in range(HISTORY_LIMIT + 5):
        wiring.engine.text = "utterance number {0}".format(index)
        app._handle_capture(FakeCapture(), False, False, "dictate")

    history = app.history
    assert len(history) == HISTORY_LIMIT
    assert "utterance number 4" not in history[0].raw
    assert history[-1].raw == "utterance number {0}".format(HISTORY_LIMIT + 4)


# --------------------------------------------------------------------------- #
# Delivery: the words belong to the user, we owe them the words
# --------------------------------------------------------------------------- #
def test_text_that_cannot_be_pasted_reaches_the_clipboard_unchanged(wiring):
    app = wiring.started()
    wiring.engine.text = _SPOKEN
    wiring.insert_ok = False
    expected = clean(_SPOKEN, app.cfg.cleanup_level, app.cfg.dictionary)

    app._handle_capture(FakeCapture(), False, False, "dictate")

    assert wiring.inserted == [expected]
    assert wiring.clipboard == [expected]
    # Byte-for-byte: whatever would have been pasted is what was copied.
    assert wiring.clipboard[0] == wiring.inserted[0]


def test_a_failed_paste_tells_the_user_where_their_words_went(wiring, capsys):
    app = wiring.started()
    wiring.engine.text = _SPOKEN
    wiring.insert_ok = False
    expected = clean(_SPOKEN, app.cfg.cleanup_level, app.cfg.dictionary)

    app._handle_capture(FakeCapture(), False, False, "dictate")

    err = capsys.readouterr().err
    assert "clipboard" in err
    assert expected in err, "the text was never shown, so it can only be recovered blind"


def test_secure_event_input_is_named_when_it_blocked_the_paste(wiring, capsys):
    app = wiring.started()
    wiring.insert_ok = False
    wiring.secure_input = True

    app._handle_capture(FakeCapture(), False, False, "dictate")

    err = capsys.readouterr().err
    assert "Secure Event Input" in err
    assert wiring.clipboard != []


def test_a_missing_accessibility_permission_is_named_when_it_blocked_the_paste(wiring, capsys):
    app = wiring.started()
    wiring.insert_ok = False
    wiring.secure_input = False
    wiring.accessibility = False
    capsys.readouterr()  # discard the startup-time Accessibility warning

    app._handle_capture(FakeCapture(), False, False, "dictate")

    err = capsys.readouterr().err
    assert "no Accessibility permission" in err
    assert wiring.clipboard != []


def test_a_paste_failure_with_no_known_cause_still_hands_over_the_text(wiring, capsys):
    app = wiring.started()
    wiring.insert_ok = False
    wiring.secure_input = None
    wiring.accessibility = None

    app._handle_capture(FakeCapture(), False, False, "dictate")

    err = capsys.readouterr().err
    assert "Could not paste into the focused app" in err
    assert wiring.clipboard != []


# --------------------------------------------------------------------------- #
# Command mode
# --------------------------------------------------------------------------- #
def test_a_spoken_command_is_routed_and_not_pasted(wiring, monkeypatch):
    from blurt.assistant import system_actions

    notifications = []
    monkeypatch.setattr(
        system_actions, "notify", lambda title, message: notifications.append((title, message))
    )
    app = wiring.started(assistant_enabled=True, assistant_hotkey="right_cmd")
    wiring.engine.text = "set a timer for five minutes"
    expected = clean(wiring.engine.text, app.cfg.cleanup_level, app.cfg.dictionary)

    app._handle_capture(FakeCapture(), False, False, "assistant")

    assert wiring.router.routed == [expected]
    assert wiring.router.executed == [wiring.router.action]
    assert wiring.inserted == []
    assert notifications == [("blurt", "Timer set for 5 minutes")]


def test_a_command_capture_with_no_router_is_still_typed_out(wiring):
    """Speech is never dropped for being in the wrong mode."""
    app = wiring.started(assistant_enabled=False)
    expected = clean(wiring.engine.text, app.cfg.cleanup_level, app.cfg.dictionary)

    app._handle_capture(FakeCapture(), False, False, "assistant")

    assert wiring.inserted == [expected]


def test_a_router_that_raises_does_not_take_the_app_down(wiring, capsys):
    app = wiring.started(assistant_enabled=True, assistant_hotkey="right_cmd")
    wiring.router.route_error = ValueError("unparseable")

    app._handle_capture(FakeCapture(), False, False, "assistant")

    assert "command failed" in capsys.readouterr().err
    assert app._started is True


# --------------------------------------------------------------------------- #
# Journalling: opt-in, and never worth losing a dictation over
# --------------------------------------------------------------------------- #
def test_nothing_is_journalled_unless_the_user_switched_it_on(wiring):
    app = wiring.started(history_enabled=False)

    for _ in range(3):
        app._handle_capture(FakeCapture(), False, False, "dictate")

    assert wiring.journal == [], "speech was written to disk without being asked for"


def test_an_enabled_journal_records_one_entry_per_dictation(wiring):
    app = wiring.started(history_enabled=True)

    app._handle_capture(FakeCapture(), False, False, "dictate")
    app._handle_capture(FakeCapture(), False, False, "dictate")

    assert len(wiring.journal) == 2
    assert wiring.journal[0].raw == "hello world"
    assert wiring.journal[0].mode == "dictate"


def test_the_journal_records_which_mode_the_capture_was_made_in(wiring):
    app = wiring.started(
        history_enabled=True, assistant_enabled=True, assistant_hotkey="right_cmd"
    )

    app._handle_capture(FakeCapture(), False, False, "assistant")

    assert [record.mode for record in wiring.journal] == ["assistant"]


def test_a_journal_that_cannot_be_written_never_costs_the_user_their_words(wiring):
    app = wiring.started(history_enabled=True)
    wiring.journal_error = OSError("read-only file system")
    expected = clean(wiring.engine.text, app.cfg.cleanup_level, app.cfg.dictionary)

    app._handle_capture(FakeCapture(), False, False, "dictate")

    assert wiring.inserted == [expected]
    assert len(app.history) == 1


def test_a_silent_capture_is_never_journalled(wiring):
    app = wiring.started(history_enabled=True)

    app._handle_capture(FakeCapture(), True, False, "dictate")

    assert wiring.journal == []


# --------------------------------------------------------------------------- #
# revert_last() IN ISOLATION: the promise that makes cleanup safe to switch on
#
# Called directly, with no assistant and no router in the picture. That is what
# keeps each refusal reason -- nothing said yet, raw history switched off,
# cleanup was a no-op, this one was already reverted -- readable as one small
# test that fails for one reason.
#
# What this section does NOT cover, and must not be read as covering, is the
# path a user actually takes to reach revert_last: speaking it. That path runs
# through a capture, the cleanup pass and the intent router before it arrives
# here, and it is where the interesting failure lives. See "THE WIRED REVERT
# PATH" at the bottom of this file.
# --------------------------------------------------------------------------- #
def test_revert_with_nothing_said_yet_is_refused(wiring, capsys):
    app = wiring.started()

    assert app.revert_last() is False
    assert wiring.inserted == []
    assert "nothing to revert" in capsys.readouterr().err


def test_revert_is_refused_when_raw_history_is_switched_off(wiring, capsys):
    app = wiring.started(keep_raw_history=False)
    app._handle_capture(FakeCapture(), False, False, "dictate")
    delivered = list(wiring.inserted)

    assert app.revert_last() is False
    assert wiring.inserted == delivered
    assert "keep_raw_history" in capsys.readouterr().err


def test_revert_is_refused_when_no_raw_text_was_kept(wiring, capsys):
    app = wiring.started(keep_raw_history=True)
    wiring.engine.text = "  "
    app._handle_capture(FakeCapture(), False, False, "dictate")
    assert len(app.history) == 1, "this test needs a transcript in history to be about anything"

    assert app.revert_last() is False
    assert wiring.inserted == []
    assert "no raw text" in capsys.readouterr().err


def test_revert_is_refused_when_cleanup_changed_nothing(wiring, capsys):
    app = wiring.started()
    wiring.engine.text = "Hello world."
    assert clean("Hello world.", app.cfg.cleanup_level, app.cfg.dictionary) == "Hello world.", (
        "this test proves nothing unless cleanup really is a no-op for this text"
    )
    app._handle_capture(FakeCapture(), False, False, "dictate")
    delivered = list(wiring.inserted)

    assert app.revert_last() is False
    assert wiring.inserted == delivered
    assert "cleanup did not change" in capsys.readouterr().err


# The next two used to be named as though they covered "saying undo that gives
# back the dictation". They never did and could not: they call revert_last()
# themselves, so the capture a spoken undo arrives on never happens, and the
# guard in _handle_capture that keeps that capture out of the revert buffer is
# never touched. Both passed, green and unchanged, while that guard was missing
# and a spoken undo pasted the words "undo that" into the user's document.
#
# They are kept rather than deleted -- pinning "raw, not cleaned" and the
# already-reverted marker with nothing else in the frame is still worth a test --
# and renamed to advertise only what they do. The end-to-end coverage the old
# names implied now exists, in "THE WIRED REVERT PATH" at the bottom of the file.
def test_revert_last_called_directly_delivers_the_raw_text_and_not_the_cleaned_one(wiring):
    app = wiring.started()
    wiring.engine.text = "hello world"
    cleaned = clean("hello world", app.cfg.cleanup_level, app.cfg.dictionary)
    assert cleaned != "hello world", "cleanup must actually change this text"

    app._handle_capture(FakeCapture(), False, False, "dictate")
    assert wiring.inserted == [cleaned]

    assert app.revert_last() is True
    assert wiring.inserted == [cleaned, "hello world"]


def test_revert_last_called_directly_twice_on_one_dictation_is_refused(wiring, capsys):
    app = wiring.started()
    wiring.engine.text = "hello world"
    app._handle_capture(FakeCapture(), False, False, "dictate")

    assert app.revert_last() is True
    delivered = list(wiring.inserted)

    assert app.revert_last() is False
    assert wiring.inserted == delivered
    assert "already reverted" in capsys.readouterr().err


def test_a_new_dictation_becomes_revertible_again(wiring):
    app = wiring.started()
    wiring.engine.text = "hello world"
    app._handle_capture(FakeCapture(), False, False, "dictate")
    app.revert_last()

    wiring.engine.text = "second thing said"
    app._handle_capture(FakeCapture(), False, False, "dictate")

    assert app.revert_last() is True
    assert wiring.inserted[-1] == "second thing said"


# --------------------------------------------------------------------------- #
# THE ONE-SHOT MARKER IDENTIFIES AN EVENT, NOT A VALUE
#
# The test above is the closest thing this file had to covering that, and it
# does not: it speaks a SECOND, DIFFERENT sentence, so the two transcripts differ
# in their fields and an ``==`` comparison answers the same as ``is``. Say the
# same short sentence twice -- which costs a user no effort at all -- and the two
# answers diverge.
# --------------------------------------------------------------------------- #


class _FrozenClock:
    """``time``, as :mod:`blurt.app` uses it, stopped dead.

    ``monotonic`` is the only member app.py touches: it times the model load and
    each transcription, and nothing else. Freezing it makes two dictations of the
    same sentence produce Transcripts that agree on every field, which is the
    collision the test below is about.

    Why fake it rather than let the clock run: with a live clock the two
    transcripts differ by a few microseconds in ``latency_seconds``, so ``==``
    and ``is`` agree and a broken comparison passes. That difference is an
    accident of measurement, not a property anything may rely on -- a coarse
    clock, a cached transcription, or a shorter utterance closes it -- and the
    field is not part of what makes a dictation the dictation it is anyway.
    """

    @staticmethod
    def monotonic() -> float:
        return 1000.0


def test_reverting_one_dictation_never_blocks_an_identical_one_said_again(
    wiring, monkeypatch, capsys
):
    """Two utterances that read the same are still two utterances.

    ``revert_last`` asks ``transcript is self._reverted_marker``, and the ``is``
    is the whole of it. :class:`Transcript` is a frozen dataclass, so ``==``
    compares five fields -- and a user who repeats a short sentence produces two
    rows that agree on all five. Under ``==`` the second dictation arrives
    already reverted: refused with "that dictation was already reverted", its raw
    text reachable only from a clipboard the next dictation overwrites, and the
    refusal delivered as a notification to someone who is looking at another
    application. The feature is gone for that utterance, permanently, and the
    user's only clue is that saying it again does nothing.

    The equality is asserted here before it is relied on, so this cannot quietly
    become a test about two objects that were never equal in the first place.
    """
    app = wiring.started()
    monkeypatch.setattr(blurt_app, "time", _FrozenClock())
    wiring.engine.text = "hello world"
    assert clean("hello world", app.cfg.cleanup_level, app.cfg.dictionary) != "hello world", (
        "cleanup must change this text, or there is nothing to revert and this "
        "test is about a refusal it did not mean to test"
    )

    app._handle_capture(FakeCapture(), False, False, "dictate")
    first = app.last_transcript
    assert app.revert_last() is True

    app._handle_capture(FakeCapture(), False, False, "dictate")
    second = app.last_transcript

    assert second == first, (
        "the two dictations are not field-for-field equal, so this test cannot "
        "tell an identity check from an equality check: {0!r} vs {1!r}".format(first, second)
    )
    assert second is not first, "they must still be two distinct objects"
    capsys.readouterr()

    assert app.revert_last() is True, (
        "the second dictation was refused because an EARLIER one happened to say "
        "the same words -- the marker is comparing values, not events"
    )
    assert wiring.inserted[-1] == "hello world"
    assert "already reverted" not in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# A REVERT WHOSE PASTE WAS BLOCKED: REFUSED, AND NOT SPENT
#
# ``_deliver`` returns False when macOS refuses the paste outright -- Secure
# Event Input, no Accessibility, no pasteboard -- and that is the one failure the
# OS actually reports. The marker is one-shot per dictation, so setting it on
# that outcome spends the user's only remaining route back to their raw text on
# nothing at all.
# --------------------------------------------------------------------------- #
def test_a_revert_whose_paste_was_blocked_is_refused_without_spending_the_marker(
    wiring, capsys
):
    app = wiring.started()
    wiring.engine.text = "hello world"
    app._handle_capture(FakeCapture(), False, False, "dictate")
    assert wiring.clipboard == [], "the dictation itself was supposed to paste cleanly"
    wiring.insert_ok = False
    wiring.secure_input = True

    assert app.revert_last() is False, "a paste macOS refused was reported as a revert"
    assert app._reverted_marker is None, (
        "the one-shot marker was spent on a paste that never landed; this "
        "dictation can never be reverted again"
    )
    assert wiring.clipboard == ["hello world"], (
        "the raw text is not even on the clipboard, so it exists nowhere the user "
        "can reach: {0!r}".format(wiring.clipboard)
    )
    err = capsys.readouterr().err
    assert "did NOT go through" in err, (
        "the last word on screen was the optimistic 'reverting to raw transcript'"
    )
    assert "still revertible" in err


def test_a_dictation_whose_revert_was_blocked_can_be_reverted_once_pasting_works(wiring):
    """Secure Event Input clears when the password field loses focus.

    That is the whole point of refusing to spend the marker: the user says it
    again a moment later and gets their raw text. Then -- and only then -- the
    dictation is spent.
    """
    app = wiring.started()
    wiring.engine.text = "hello world"
    app._handle_capture(FakeCapture(), False, False, "dictate")
    wiring.insert_ok = False
    assert app.revert_last() is False

    wiring.insert_ok = True

    assert app.revert_last() is True, (
        "the blocked attempt burned the marker: the raw text is unreachable now"
    )
    assert wiring.inserted[-1] == "hello world"
    assert app.revert_last() is False, (
        "the attempt that DID land must still spend the marker exactly once"
    )


# --------------------------------------------------------------------------- #
# The transcription worker: one bad dictation must not end the loop
# --------------------------------------------------------------------------- #
def test_a_failed_dictation_does_not_end_the_worker(wiring):
    app = wiring.started()
    wiring.engine.transcribe_error = RuntimeError("boom")

    app._jobs.put((FakeCapture(), False, False, "dictate"))
    app._jobs.join()

    wiring.engine.transcribe_error = None
    wiring.engine.text = "still alive"
    app._jobs.put((FakeCapture(), False, False, "dictate"))
    app._jobs.join()

    assert wiring.inserted == [clean("still alive", app.cfg.cleanup_level, app.cfg.dictionary)]


def test_a_malformed_job_does_not_end_the_worker(wiring):
    """Nothing enqueues garbage today; the worker must survive it if anything ever does."""
    app = wiring.started()

    app._jobs.put("this is not a capture")
    app._jobs.join()

    app._jobs.put((FakeCapture(), False, False, "dictate"))
    app._jobs.join()

    assert wiring.inserted == [clean("hello world", app.cfg.cleanup_level, app.cfg.dictionary)]


# --------------------------------------------------------------------------- #
# run(): blocks until asked to stop, and leaves the process as it found it
# --------------------------------------------------------------------------- #
def test_run_starts_the_app_and_returns_when_stop_is_requested(wiring):
    app = wiring.build()

    app.request_stop()
    app.run()

    assert wiring.dictation_hotkey.started == 1


def test_run_restores_the_signal_handlers_it_borrowed(wiring):
    app = wiring.started()
    before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))

    app.request_stop()
    app.run()

    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == before


def test_a_startup_failure_exits_nonzero_with_a_message_not_a_traceback(wiring, capsys):
    wiring.select_error = NoEngineAvailable(
        "No usable speech engine.\n  Fix: python3 -m pip install faster-whisper"
    )

    code = blurt_app.run(Config(assistant_enabled=False), wiring.hardware)

    assert code == 1
    assert "python3 -m pip install faster-whisper" in capsys.readouterr().err


def test_ctrl_c_during_startup_is_not_an_error_report(wiring):
    wiring.select_error = KeyboardInterrupt()

    code = blurt_app.run(Config(assistant_enabled=False), wiring.hardware)

    assert code == 130


# --------------------------------------------------------------------------- #
# Shutdown: tear down in the order that loses the least
# --------------------------------------------------------------------------- #
def test_shutdown_stops_the_hotkey_before_draining_the_worker(wiring):
    """No new capture may be accepted while the in-flight one is still landing.

    The ordering is forced, not observed by luck: the fake engine parks inside
    ``transcribe`` until the hotkey has actually been stopped. If shutdown drained
    the worker first it would block on a transcription that is waiting for a stop
    that has not happened, and the recorded order would come out reversed.
    """
    app = wiring.started()
    entered = threading.Event()

    def park():
        entered.set()
        wiring.hotkey_stopped.wait(timeout=2.0)

    wiring.engine.before_transcribe = park
    app._jobs.put((FakeCapture(), False, False, "dictate"))
    assert entered.wait(timeout=5.0), "the worker never picked the job up"

    del wiring.order[:]  # measure the teardown only
    app.shutdown()

    assert wiring.order == [
        "hotkey.stop",
        "engine.transcribe",
        "recorder.close",
        "engine.unload",
    ]


def test_shutdown_releases_the_microphone_and_the_model(wiring):
    app = wiring.started()

    app.shutdown()

    assert wiring.recorder.closed == 1
    assert wiring.engine.unloads == 1
    assert wiring.dictation_hotkey.stopped == 1
    assert app._recorder is None
    assert app._engine is None


def test_shutdown_stops_both_hotkeys(wiring):
    app = wiring.started(assistant_enabled=True, assistant_hotkey="right_cmd")

    app.shutdown()

    assert wiring.dictation_hotkey.stopped == 1
    assert wiring.assistant_hotkey.stopped == 1


def test_shutdown_is_safe_to_call_twice(wiring):
    app = wiring.started()

    app.shutdown()
    app.shutdown()

    assert wiring.dictation_hotkey.stopped == 1
    assert wiring.recorder.closed == 1
    assert wiring.engine.unloads == 1


def test_shutdown_without_startup_is_safe(wiring):
    app = wiring.build()

    app.shutdown()

    assert wiring.recorder.closed == 0
    assert wiring.engine.unloads == 0


def test_a_component_that_raises_on_teardown_does_not_strand_the_others(wiring):
    """A hotkey that will not let go must not leave PortAudio running."""
    app = wiring.started(assistant_enabled=True, assistant_hotkey="right_cmd")
    wiring.dictation_hotkey.stop_error = RuntimeError("listener wedged")
    wiring.recorder.close_error = RuntimeError("stream wedged")
    wiring.engine.unload_error = RuntimeError("model wedged")

    app.shutdown()

    assert wiring.assistant_hotkey.stopped == 1
    assert wiring.recorder.closed == 1
    assert wiring.engine.unloads == 1


def test_the_worker_thread_is_gone_after_shutdown(wiring):
    app = wiring.started()
    worker = app._worker

    app.shutdown()

    worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert app._worker is None


# --------------------------------------------------------------------------- #
# THE WIRED REVERT PATH
#
# Every revert test above this line calls revert_last() itself, and every one of
# them stays green with the feature completely broken. The break they cannot see
# is one step upstream, in _handle_capture: the guard that keeps command-mode
# captures out of the revert buffer. Delete it and revert_last is unchanged, the
# history property is unchanged, the router is unchanged -- and yet saying "undo
# that" pastes the words "undo that", because the undo command is itself a
# capture and by the time its handler runs it is the newest entry in the deque.
# Worse, it is self-perpetuating: each retry appends another command and pushes
# the dictation one slot further out of reach.
#
# So these tests run the thing whole. A capture goes in as PCM and the assertion
# is on the string that came back out of insert_text, with the real router in
# between (Wiring.use_real_router) so the handler that claims "undo that" is the
# handler that would claim it in production. The seam is crossed rather than
# assumed.
#
# Why it is worth the extra machinery: blurt.inject can paste but cannot delete.
# A wrong revert puts text into a document the user was not thinking about and
# blurt has no way to take it back. Failing to revert costs one repeat; reverting
# the wrong thing is silent and permanent.
# --------------------------------------------------------------------------- #

#: Chosen because cleanup demonstrably changes it -- the filler "um" is dropped
#: and the sentence is capitalised -- so raw and cleaned differ and there is
#: genuinely something for a revert to give back. Every test below asserts that
#: difference before relying on it.
_DICTATED = "um so the deploy is finished"

#: The two words this whole section exists for. Not a constant for tidiness: it
#: is asserted against as the string that must NEVER be pasted.
_UNDO = "undo that"


def _assistant_app(wiring, **cfg_kwargs) -> BlurtApp:
    """A started app whose command mode is driven by the REAL intent router.

    Command mode has to be genuinely enabled -- ``_handle_capture`` only routes
    an assistant capture when ``self._router`` is not None, so an app built with
    the assistant off would send "undo that" down the dictation path and paste
    it, which is the very bug these tests are about and would look like a pass.
    """
    wiring.use_real_router()
    settings = {"assistant_enabled": True, "assistant_hotkey": "right_cmd"}
    settings.update(cfg_kwargs)
    app = wiring.started(**settings)
    assert app._router is wiring.router, (
        "the app is not holding the real router; these tests would be asserting "
        "against a fake that answers by fiat"
    )
    return app


def _speak(wiring, app, text: str, mode: str) -> None:
    """Put one utterance through the whole pipeline under the given mode.

    Everything a real capture goes through except PortAudio and Whisper: the
    engine hands back ``text``, cleanup runs, the history and journal decisions
    are taken, and the result is either delivered or routed. ``mode`` is what the
    hotkey claimed, and it is the ONLY thing distinguishing "the user dictated
    this" from "the user said this as a command" -- which is exactly the
    distinction under test, so it is passed explicitly at every call site rather
    than defaulted.
    """
    wiring.engine.text = text
    app._handle_capture(FakeCapture(), False, False, mode)


def test_saying_undo_that_pastes_the_previous_dictation_and_never_the_undo_itself(wiring):
    """The headline. Two captures in, and the second thing pasted is the first thing said."""
    app = _assistant_app(wiring)
    cleaned = clean(_DICTATED, app.cfg.cleanup_level, app.cfg.dictionary)
    assert cleaned != _DICTATED, (
        "this test proves nothing unless cleanup really changed the dictation"
    )

    _speak(wiring, app, _DICTATED, "dictate")
    _speak(wiring, app, _UNDO, "assistant")

    assert wiring.inserted[:1] == [cleaned], "the dictation itself was not delivered"
    assert len(wiring.inserted) == 2, (
        "the spoken undo never reached insert_text; inserted={0!r}".format(wiring.inserted)
    )
    assert wiring.inserted[1] == _DICTATED, (
        "the revert pasted {0!r}, but the raw text of the dictation before it was "
        "{1!r}".format(wiring.inserted[1], _DICTATED)
    )
    assert wiring.inserted[1] != _UNDO, (
        "the undo command was pasted into the user's document instead of undone -- "
        "and blurt.inject cannot delete it again"
    )


def test_a_spoken_command_is_kept_out_of_the_buffer_a_dictation_goes_into(wiring):
    """History is the revert buffer, so what it holds is the whole question."""
    app = _assistant_app(wiring)

    _speak(wiring, app, _DICTATED, "dictate")
    assert [t.raw for t in app.history] == [_DICTATED], "the dictation is missing"

    _speak(wiring, app, _UNDO, "assistant")

    assert [t.raw for t in app.history] == [_DICTATED], (
        "the undo command entered the revert buffer; it is now the thing the next "
        "revert would paste"
    )
    assert app.last_transcript is not None
    assert app.last_transcript.raw == _DICTATED


def test_the_journal_still_records_both_modes_now_that_history_records_one(wiring):
    """Keeping commands out of the revert buffer is not a licence to forget them.

    The buffer and the journal answer different questions -- "what may a revert
    target" and "what did the user say" -- and the fix for the first must not
    quietly narrow the second. A user who switched journalling on asked for a
    record of what they said, not a record of half of it.
    """
    app = _assistant_app(wiring, history_enabled=True)

    _speak(wiring, app, _DICTATED, "dictate")
    _speak(wiring, app, _UNDO, "assistant")

    assert [record.mode for record in wiring.journal] == ["dictate", "assistant"], (
        "the journal lost a mode: {0!r}".format([r.mode for r in wiring.journal])
    )
    assert [record.raw for record in wiring.journal] == [_DICTATED, _UNDO]
    assert len(app.history) == 1, "and the revert buffer still holds only the dictation"


def test_saying_undo_that_twice_refuses_the_second_instead_of_reverting_again(wiring, capsys):
    """The second undo must decline, not go looking for something else to undo."""
    app = _assistant_app(wiring)

    _speak(wiring, app, _DICTATED, "dictate")
    _speak(wiring, app, _UNDO, "assistant")
    after_first_undo = list(wiring.inserted)
    assert after_first_undo[-1] == _DICTATED, "the first undo did not do its job"
    capsys.readouterr()

    _speak(wiring, app, _UNDO, "assistant")

    assert wiring.inserted == after_first_undo, (
        "the second undo pasted {0!r}; it should have refused and pasted "
        "nothing".format(wiring.inserted[len(after_first_undo):])
    )
    assert "already reverted" in capsys.readouterr().err


def test_saying_undo_that_with_nothing_dictated_yet_refuses_and_pastes_nothing(wiring, capsys):
    """The empty case reaches the user as a refusal, never as a stray paste."""
    app = _assistant_app(wiring)

    _speak(wiring, app, _UNDO, "assistant")

    assert wiring.inserted == [], (
        "an undo with nothing to undo put {0!r} into the focused app".format(wiring.inserted)
    )
    assert app.history == []
    assert "nothing to revert" in capsys.readouterr().err


def test_an_unrelated_command_in_between_does_not_displace_the_dictation(wiring):
    """An intervening "open Safari" is a capture too, and must not become the target."""
    app = _assistant_app(wiring)

    _speak(wiring, app, _DICTATED, "dictate")
    _speak(wiring, app, "open Safari", "assistant")
    assert wiring.opened_apps == ["Safari"], (
        "the intervening command did not actually route anywhere, so this test is "
        "not about anything"
    )
    assert wiring.inserted == [
        clean(_DICTATED, app.cfg.cleanup_level, app.cfg.dictionary)
    ], "a command that ran an action must not also paste its own words"

    _speak(wiring, app, _UNDO, "assistant")

    assert [t.raw for t in app.history] == [_DICTATED]
    assert len(wiring.inserted) == 2
    assert wiring.inserted[-1] == _DICTATED, (
        "the revert reached back to {0!r} instead of the dictation the user "
        "wanted".format(wiring.inserted[-1])
    )


def test_a_spoken_undo_whose_paste_was_blocked_can_simply_be_spoken_again(wiring, capsys):
    """The blocked revert, along the path a user actually takes to reach it.

    Everything about this failure is invisible from the terminal: the user is in
    another application (that is what command mode is FOR), so the notification
    is the only thing they see, and a refused paste that reported success said
    "Reverted to the raw transcript." while nothing had been inserted. Saying it
    again was then answered with "already reverted", and the raw text sat on a
    clipboard the next dictation overwrites.

    So the assertions are on what reaches the user: the console verdict, the
    notification, the clipboard, and the fact that the second attempt works.
    """
    app = _assistant_app(wiring)
    _speak(wiring, app, _DICTATED, "dictate")
    wiring.insert_ok = False
    capsys.readouterr()

    _speak(wiring, app, _UNDO, "assistant")

    assert app._reverted_marker is None, "a refused paste spent the one-shot marker"
    assert wiring.clipboard == [_DICTATED], (
        "the raw text is not on the clipboard either: {0!r}".format(wiring.clipboard)
    )
    assert "  x " in capsys.readouterr().out, (
        "the console reported the blocked revert as a success"
    )
    assert wiring.notifications[-1] == ("blurt", "Nothing to revert."), (
        "the only message the user sees claimed the revert happened: {0!r}".format(
            wiring.notifications[-1]
        )
    )

    wiring.insert_ok = True
    _speak(wiring, app, _UNDO, "assistant")

    assert len(wiring.inserted) == 3
    assert wiring.inserted[-1] == _DICTATED, (
        "the second undo did not deliver the raw text; the dictation was lost to "
        "a paste that never happened"
    )
    assert wiring.notifications[-1] == ("blurt", "Reverted to the raw transcript.")
