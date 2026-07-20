"""Global hold-to-talk key handling for blurt, built on pynput.

The user holds one key, speaks, and releases it. This module turns that gesture
into three callbacks -- ``on_start``, ``on_stop``, ``on_cancel`` -- and nothing
more. It records nothing, transcribes nothing, and has never heard of a
Transcript.

WHY THE DEFAULT TRIGGER IS A BARE MODIFIER (right option)
---------------------------------------------------------
A bare modifier inserts no character into the focused application. Holding
right option in a text editor produces exactly nothing on screen. That single
property dissolves an entire class of bugs that dogs letter- and chord-based
dictation hotkeys: the trigger leaking a stray character into the document, or
a half-typed chord leaving a modifier stuck down. There is nothing to swallow,
so there is nothing to leak.

That matters more than usual here because **pynput cannot suppress events on
macOS in any way we are willing to depend on**. Its Quartz tap is created
listen-only when suppression is off, so the keystroke reaches the focused app
whether we like it or not. With a bare modifier that is fine: the app sees a
modifier state change it will ignore. With a printable key it would not be
fine. Do not "improve" the default to a letter.

macOS PERMISSIONS -- READ THIS BEFORE DEBUGGING "MY HOTKEY DOES NOTHING"
------------------------------------------------------------------------
pynput taps global Quartz events, which macOS gates behind **Accessibility**
(System Settings > Privacy & Security > Accessibility). Some macOS versions
also want **Input Monitoring**. Grant both if events never arrive.

The confusing part, and the number one support question for tools like this:

  * When blurt is launched from a shell, macOS attributes the grant to the
    **terminal application** -- Terminal.app, iTerm2, Ghostty, VS Code's
    integrated terminal -- and *not* to blurt or to Python. The user therefore
    has to enable *Terminal* in the Accessibility list, which reads like a
    mistake but is not. Running the same command from a different terminal
    means granting it again for that other terminal.
  * The grant is remembered per signed binary, so a macOS update, a terminal
    update, or moving the app can silently invalidate it. The symptom is
    always identical: the process starts fine and simply never sees a key.
  * pynput does **not** raise when the process is untrusted. It logs a warning
    on its own logger, fails to create the event tap, and delivers no events
    ever. Silence is the failure mode. Call :func:`accessibility_trusted` at
    startup and tell the user plainly instead of letting them stare at a dead
    hotkey.
  * The proper long-term fix is shipping blurt as a bundled ``.app`` with a
    stable code signature, so the permission attaches to blurt itself and
    survives updates. Until then this arrangement is genuinely fragile and the
    user deserves to be told why.

OTHER THINGS THAT GO WRONG ON macOS
-----------------------------------
  * **Secure input.** While a password field holds focus (login window, a
    terminal ``sudo`` prompt, a password manager), macOS enables secure event
    input and global taps go dark. Nothing we can do; the hotkey just will not
    fire there.
  * **Left/right modifiers share one flag.** pynput's Quartz backend decides
    whether a modifier event is a press or a release by testing the *global*
    flag mask, not that individual key's state, and both option keys set the
    same Alternate bit. So if the user holds left option and then releases
    right option, the mask is still set and pynput reports the right-option
    release as another **press**. To avoid getting wedged in the armed state
    we treat a release of any key sharing the trigger's flag as a release of
    the trigger (see ``_release_keys``); a sibling release is only reported
    once the shared mask has actually cleared, which means every key in the
    group, ours included, is physically up. The visible consequence: if the
    user happens to be holding the *other* option key, the dictation ends when
    that key comes up rather than when the trigger does. Rare, harmless,
    documented.
  * **``fn`` is not supported.** The globe/fn key is not in pynput's macOS key
    table at all and does not reach the event tap as an observable key.
    Requesting it raises :class:`UnsupportedHotkeyError` at construction rather
    than handing back a hotkey that never fires.
  * **Caps lock is a toggle**, not a hold, so it cannot express hold-to-talk.
    Also refused at construction.

THREADING
---------
pynput delivers key events on its own listener thread. Our callbacks kick off
recording and transcription and can block for seconds -- Whisper pads every
clip to a 30 second window, so even a two second utterance costs real time on
the Intel floor machine. Running a callback inline on the listener thread would
stall every later key event, including the release that ends the recording. So
callbacks go to a single dedicated worker thread through a queue. One worker,
not a pool: it keeps ``on_start`` strictly ordered before its ``on_stop``.

Python 3.9 floor: ``from __future__ import annotations``, ``typing`` generics,
no PEP 604 unions, no ``match``.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

__all__ = [
    "HoldToTalk",
    "UnsupportedHotkeyError",
    "SUPPORTED_HOTKEYS",
    "accessibility_trusted",
    "normalize_key_name",
]

_log = logging.getLogger(__name__)


class UnsupportedHotkeyError(ValueError):
    """Raised when a hotkey name cannot be honoured on this platform.

    Subclasses :class:`ValueError` so callers guarding only against bad config
    values still catch it.
    """


# Canonical blurt hotkey name -> pynput ``Key`` attribute name.
#
# Looked up by attribute name rather than by importing the enum members, so
# this table stays readable without pynput installed and so pynput's own
# aliasing stays pynput's problem. (On macOS ``Key.alt_l`` and ``Key.alt`` are
# the same object -- both are virtual key 0x3A -- and likewise for cmd, ctrl
# and shift. Only the *right* hand variants have distinct key codes.)
_KEY_ATTRS: Dict[str, str] = {
    "right_option": "alt_r",
    "left_option": "alt_l",
    "right_cmd": "cmd_r",
    "right_ctrl": "ctrl_r",
    "right_shift": "shift_r",
    # Left-hand variants cost nothing to support and some users want them.
    "left_cmd": "cmd_l",
    "left_ctrl": "ctrl_l",
    "left_shift": "shift_l",
}

# Spellings we quietly fold into the canonical names above, so a config file
# saying "right_alt" or "right_command" does not brick startup.
_KEY_ALIASES: Dict[str, str] = {
    "right_alt": "right_option",
    "right_opt": "right_option",
    "ralt": "right_option",
    "left_alt": "left_option",
    "left_opt": "left_option",
    "lalt": "left_option",
    "right_command": "right_cmd",
    "right_meta": "right_cmd",
    "left_command": "left_cmd",
    "left_meta": "left_cmd",
    "right_control": "right_ctrl",
    "left_control": "left_ctrl",
}

# Keys we refuse, with the reason. Failing loudly at construction beats
# shipping a hotkey that silently never fires.
_UNSUPPORTED: Dict[str, str] = {
    "fn": (
        "the fn / globe key is not observable through pynput on macOS -- it is "
        "handled below the Quartz event tap and is not in pynput's macOS key "
        "table at all, so blurt would never see it. Use 'right_option' "
        "instead: it is the default, and like fn it types no character."
    ),
    "globe": (
        "the fn / globe key is not observable through pynput on macOS. Use "
        "'right_option' instead."
    ),
    "caps_lock": (
        "macOS reports caps lock as a toggle rather than as a press followed "
        "by a release, so it cannot express hold-to-talk. Use 'right_option'."
    ),
}

#: Hotkey names this module accepts, in a stable order for help text and UI.
SUPPORTED_HOTKEYS: Tuple[str, ...] = (
    "right_option",
    "left_option",
    "right_cmd",
    "right_ctrl",
    "right_shift",
    "left_cmd",
    "left_ctrl",
    "left_shift",
)

# Keys sharing one macOS modifier flag mask. See the module docstring: pynput
# infers press-vs-release for modifiers from that shared mask, so a *release*
# reported for any member of a group means the whole group is up.
_SIBLING_ATTRS: Dict[str, Tuple[str, ...]] = {
    "alt_l": ("alt", "alt_l", "alt_r", "alt_gr"),
    "alt_r": ("alt", "alt_l", "alt_r", "alt_gr"),
    "cmd_l": ("cmd", "cmd_l", "cmd_r"),
    "cmd_r": ("cmd", "cmd_l", "cmd_r"),
    "ctrl_l": ("ctrl", "ctrl_l", "ctrl_r"),
    "ctrl_r": ("ctrl", "ctrl_l", "ctrl_r"),
    "shift_l": ("shift", "shift_l", "shift_r"),
    "shift_r": ("shift", "shift_l", "shift_r"),
}

# Press lifecycle. Named states beat a pile of booleans; the awkward case is
# "cancelled but the user is still holding the key down", which must be
# distinguishable from "idle" so that the eventual release fires nothing.
_IDLE = "idle"  # trigger up, nothing in flight
_PENDING = "pending"  # trigger down, has not passed min_hold yet
_ARMED = "armed"  # on_start has fired; on_stop is owed on release
_ABORTED = "aborted"  # cancelled via Esc; trigger still physically down


def normalize_key_name(key_name: str) -> str:
    """Fold an accepted spelling of a hotkey name into its canonical form.

    Raises :class:`UnsupportedHotkeyError` for names blurt cannot honour, with
    a message aimed at the user rather than at whoever reads the traceback.
    Imports nothing, so config validation still works on a machine where the
    input stack is broken.
    """
    if not isinstance(key_name, str):
        raise UnsupportedHotkeyError(
            "hotkey must be a string, got {0!r}".format(type(key_name).__name__)
        )

    name = key_name.strip().lower().replace("-", "_").replace(" ", "_")
    name = _KEY_ALIASES.get(name, name)

    if name in _UNSUPPORTED:
        raise UnsupportedHotkeyError(
            "hotkey {0!r} is not supported: {1}".format(key_name, _UNSUPPORTED[name])
        )
    if name not in _KEY_ATTRS:
        raise UnsupportedHotkeyError(
            "unknown hotkey {0!r}. Supported values: {1}".format(
                key_name, ", ".join(SUPPORTED_HOTKEYS)
            )
        )
    return name


def accessibility_trusted() -> Optional[bool]:
    """Report whether this process may observe global input events.

    Returns ``True`` when macOS trusts the host process for Accessibility,
    ``False`` when it does not, and ``None`` when the answer cannot be
    determined (not macOS, or pyobjc missing). ``False`` is the important one:
    the hotkey will start without error and then never fire, so surface it.

    The trust belongs to the *host* application -- the terminal, when blurt is
    run from a shell -- not to blurt itself. Asking does not prompt; macOS only
    shows the Accessibility prompt for APIs we deliberately avoid calling here.
    """
    # pyobjc exposes AXIsProcessTrusted from more than one framework module and
    # which ones are installed varies with the pyobjc split. Try each; give up
    # quietly rather than turning a diagnostic into a crash.
    for module_name in ("HIServices", "ApplicationServices", "Quartz"):
        try:
            module = __import__(module_name)
        except Exception:  # pragma: no cover - non-macOS or pyobjc missing
            continue
        probe = getattr(module, "AXIsProcessTrusted", None)
        if probe is None:
            continue
        try:
            return bool(probe())
        except Exception:  # pragma: no cover - API shape changed
            return None
    return None


def _import_keyboard() -> Any:
    """Import ``pynput.keyboard``, turning failure into actionable advice."""
    try:
        from pynput import keyboard
    except Exception as exc:  # ImportError, or pyobjc exploding underneath
        raise RuntimeError(
            "could not load pynput's keyboard backend ({0}: {1}). blurt needs "
            "pynput and pyobjc installed, and on macOS it needs Accessibility "
            "permission for the application that launched it.".format(
                type(exc).__name__, exc
            )
        ) from exc
    return keyboard


class HoldToTalk:
    """Hold a key to dictate, release to finish, press Esc to throw it away.

    Behaviour:
      * A press shorter than ``min_hold_ms`` is a stray tap and produces **no
        callbacks at all** -- not even ``on_start``. This is the difference
        between a usable tool and one that fires a transcription every time the
        user brushes option while reaching for a bracket.
      * ``on_start`` fires from a timer once the key has been held past the
        threshold, not from the release, so recording begins while the user is
        still speaking rather than after they finish.
      * ``on_stop`` fires on release, but only for a press that reached
        ``on_start``.
      * ``on_cancel`` fires if Esc is pressed while armed. The recording is to
        be discarded, and the eventual release of the trigger fires nothing.
      * Key auto-repeat is ignored: repeat key-down events arriving while the
        trigger is already down do nothing.

    All three callbacks run on a private worker thread, never on pynput's
    listener thread, and are serialised so ``on_start`` always completes before
    its ``on_stop`` begins. Exceptions raised inside a callback are logged and
    swallowed -- a failed transcription must not take the hotkey down with it.

    :param key_name: one of :data:`SUPPORTED_HOTKEYS`, or an accepted alias.
    :param on_start: called when a hold passes the threshold.
    :param on_stop: called when an armed hold is released.
    :param on_cancel: called when an armed hold is abandoned via Esc.
    :param min_hold_ms: minimum hold in milliseconds before arming.

    :raises UnsupportedHotkeyError: for ``fn`` and other unusable keys.
    :raises RuntimeError: if pynput cannot be loaded.
    """

    def __init__(
        self,
        key_name: str,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        on_cancel: Callable[[], None],
        min_hold_ms: int = 200,
    ) -> None:
        # Validate the name first so a bad config is reported the same way
        # whether or not pynput imports -- as a bad config, never as a missing
        # dependency.
        self.key_name: str = normalize_key_name(key_name)

        if not callable(on_start) or not callable(on_stop) or not callable(on_cancel):
            raise TypeError("on_start, on_stop and on_cancel must all be callable")

        try:
            hold_ms = int(min_hold_ms)
        except (TypeError, ValueError):
            raise ValueError(
                "min_hold_ms must be an integer number of milliseconds, got "
                "{0!r}".format(min_hold_ms)
            )
        self.min_hold_ms: int = max(0, hold_ms)

        self._on_start = on_start
        self._on_stop = on_stop
        self._on_cancel = on_cancel

        keyboard = _import_keyboard()
        self._keyboard = keyboard
        self._trigger, self._release_keys = self._resolve_keys(keyboard, self.key_name)
        self._esc = keyboard.Key.esc

        self._lock = threading.RLock()
        self._state: str = _IDLE
        self._key_down: bool = False
        self._press_monotonic: float = 0.0
        self._hold_timer: Optional[threading.Timer] = None

        self._listener: Optional[Any] = None
        self._worker: Optional[threading.Thread] = None
        self._jobs: "queue.Queue[Optional[Tuple[str, Callable[[], None]]]]" = (
            queue.Queue()
        )
        self._started = False
        self._stopped = False

    # -- construction helpers -----------------------------------------------

    @staticmethod
    def _resolve_keys(keyboard: Any, name: str) -> Tuple[Any, FrozenSet[Any]]:
        """Map a canonical blurt name to its pynput key plus its flag siblings."""
        attr = _KEY_ATTRS[name]
        trigger = getattr(keyboard.Key, attr, None)
        if trigger is None:  # pragma: no cover - would mean a pynput API change
            raise UnsupportedHotkeyError(
                "this build of pynput has no key {0!r}, so blurt cannot bind "
                "{1!r} on this platform.".format(attr, name)
            )

        siblings: List[Any] = [trigger]
        for sibling_attr in _SIBLING_ATTRS.get(attr, ()):
            sibling = getattr(keyboard.Key, sibling_attr, None)
            if sibling is not None:
                siblings.append(sibling)
        return trigger, frozenset(siblings)

    # -- public API ----------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """True once :meth:`start` has run and :meth:`stop` has not."""
        with self._lock:
            return self._started and not self._stopped

    @property
    def is_armed(self) -> bool:
        """True while a hold has passed the threshold and not yet ended."""
        with self._lock:
            return self._state == _ARMED

    def start(self) -> None:
        """Begin listening. Returns immediately; the work happens on threads.

        Starting cannot verify permission in any useful sense -- there is
        nothing to ask macOS for. If the host app is untrusted the listener
        starts happily and no key ever arrives, so check
        :func:`accessibility_trusted` and tell the user before you get here.
        Calling start() twice is a no-op; calling it after stop() raises.
        """
        with self._lock:
            if self._stopped:
                raise RuntimeError(
                    "this HoldToTalk has been stopped; construct a new one to "
                    "listen again"
                )
            if self._started:
                return
            self._started = True

            worker = threading.Thread(
                target=self._worker_loop,
                name="blurt-hotkey-worker",
                daemon=True,
            )
            self._worker = worker
            worker.start()

            try:
                listener = self._keyboard.Listener(
                    on_press=self._handle_press,
                    on_release=self._handle_release,
                )
                # Daemon so a forgotten stop() cannot wedge interpreter
                # shutdown. pynput sets this itself; we do not rely on that.
                listener.daemon = True
                listener.start()
            except Exception:
                # Do not leave a live worker thread behind a failed start.
                self._listener = None
                self._started = False
                self._jobs.put(None)
                self._worker = None
                raise
            self._listener = listener

        trusted = accessibility_trusted()
        if trusted is False:
            _log.warning(
                "This process is not trusted for macOS Accessibility, so the "
                "%s hotkey will never fire. Grant Accessibility (and Input "
                "Monitoring) to the application that launched blurt -- usually "
                "your terminal, not blurt itself -- in System Settings > "
                "Privacy & Security.",
                self.key_name,
            )

    def stop(self) -> None:
        """Stop listening and join the helper threads. Safe to call twice.

        A hold in flight is abandoned silently: no ``on_stop`` and no
        ``on_cancel`` are emitted for it, because a shutdown is not a
        dictation. Callbacks already queued are allowed to finish so a
        transcription in progress is not truncated.
        """
        with self._lock:
            if self._stopped:
                return
            self._stopped = True

            listener = self._listener
            worker = self._worker
            timer = self._hold_timer

            self._listener = None
            self._hold_timer = None
            self._state = _IDLE
            self._key_down = False

        if timer is not None:
            timer.cancel()
            if timer is not threading.current_thread():
                timer.join(timeout=1.0)

        if listener is not None:
            try:
                listener.stop()
            except Exception:  # pragma: no cover - pynput teardown is noisy
                _log.debug("keyboard listener stop() raised", exc_info=True)
            # Joining from inside a listener callback would deadlock on the
            # listener's own thread. pynput also re-raises callback exceptions
            # out of join(), which must not escape a shutdown path.
            if listener is not threading.current_thread():
                try:
                    listener.join(timeout=2.0)
                except Exception:  # pragma: no cover
                    _log.debug("keyboard listener join() raised", exc_info=True)

        # The sentinel unblocks the worker's get(); jobs queued ahead of it
        # still run first.
        self._jobs.put(None)
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=10.0)
            if worker.is_alive():  # pragma: no cover - a callback is wedged
                _log.warning(
                    "hotkey worker thread did not exit within 10s; a callback "
                    "is probably blocked."
                )
        with self._lock:
            self._worker = None

    # -- pynput callbacks (listener thread: stay fast, never raise) ----------

    def _handle_press(self, key: Any) -> None:
        try:
            if key in self._release_keys:
                # Only the exact trigger arms. Siblings matter on release only.
                if key == self._trigger:
                    self._on_trigger_down()
                return
            if key == self._esc:
                self._on_escape()
        except Exception:  # pragma: no cover - never kill the listener
            _log.exception("error handling key press")

    def _handle_release(self, key: Any) -> None:
        try:
            if key in self._release_keys:
                self._on_trigger_up()
        except Exception:  # pragma: no cover
            _log.exception("error handling key release")

    # -- state machine (holds the lock, does no user work) -------------------

    def _on_trigger_down(self) -> None:
        with self._lock:
            if self._stopped or self._key_down:
                # Already down: this is auto-repeat, or the macOS flag-mask
                # quirk reporting a sibling release as a press. Ignore it.
                return
            self._key_down = True
            self._state = _PENDING
            self._press_monotonic = time.monotonic()

            if self.min_hold_ms <= 0:
                self._arm_locked()
                return

            timer = threading.Timer(self.min_hold_ms / 1000.0, self._on_hold_elapsed)
            timer.name = "blurt-hotkey-hold"
            timer.daemon = True
            self._hold_timer = timer
            timer.start()

    def _on_hold_elapsed(self) -> None:
        with self._lock:
            self._hold_timer = None
            if self._stopped or not self._key_down or self._state != _PENDING:
                # Released, cancelled or shut down while the timer ran: a tap,
                # which by design produces no callbacks whatsoever.
                return
            self._arm_locked()

    def _arm_locked(self) -> None:
        """Caller must hold ``self._lock``."""
        self._state = _ARMED
        self._dispatch("on_start", self._on_start)

    def _on_trigger_up(self) -> None:
        with self._lock:
            if not self._key_down:
                return
            self._key_down = False
            self._cancel_timer_locked()

            previous = self._state
            self._state = _IDLE
            if previous == _ARMED and not self._stopped:
                self._dispatch("on_stop", self._on_stop)
            elif previous == _PENDING:
                held_ms = (time.monotonic() - self._press_monotonic) * 1000.0
                _log.debug(
                    "ignoring %.0f ms tap of %s (below the %d ms threshold)",
                    held_ms,
                    self.key_name,
                    self.min_hold_ms,
                )

    def _on_escape(self) -> None:
        with self._lock:
            if self._stopped:
                return
            if self._state == _ARMED:
                # Throw the recording away. The trigger is still physically
                # down, so sit in _ABORTED until it comes up; that release must
                # fire nothing.
                self._state = _ABORTED
                self._dispatch("on_cancel", self._on_cancel)
            elif self._state == _PENDING:
                # Never armed, so nothing is owed and no callback fires. Just
                # make sure the timer cannot arm it after the fact.
                self._cancel_timer_locked()
                self._state = _ABORTED

    def _cancel_timer_locked(self) -> None:
        """Caller must hold ``self._lock``."""
        timer = self._hold_timer
        self._hold_timer = None
        if timer is not None:
            timer.cancel()

    # -- callback dispatch ---------------------------------------------------

    def _dispatch(self, label: str, callback: Callable[[], None]) -> None:
        """Hand a callback to the worker thread. Never blocks the caller."""
        self._jobs.put((label, callback))

    def _worker_loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            label, callback = job
            try:
                callback()
            except Exception:
                # A blown transcription must not take the hotkey with it.
                _log.exception("hotkey callback %s raised", label)
