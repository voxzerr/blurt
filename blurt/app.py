"""The blurt application: wires hardware, config, engine, hotkey, mic and injection together.

This is the only module that knows the whole story. Everything below it is a
component with one job; this is where the order of operations lives.

Startup, in this order and for these reasons::

    detect hardware -> load config -> select engine -> LOAD engine -> listen

The model is loaded during startup, before the hotkey is live. That is the whole
point: a user who has to wait 3-10 seconds on their first dictation concludes the
app is broken. Better to be honest at launch, where waiting is expected, than
mysterious on first use.

Runtime, per dictation::

    key held past min_hold_ms -> Recorder.start()
    key released              -> Recorder.stop() -> queue -> worker transcribes
                                 -> cleanup -> insert_text (clipboard on failure)
    Esc while armed           -> discard, nothing is transcribed

Transcription runs on a private worker thread. It must never run on the hotkey
listener's thread: on the Intel floor machine a transcription takes 2-8 seconds,
and a listener blocked that long would drop the next key press entirely.

What can go wrong on macOS:

  - No Accessibility permission. The hotkey listener starts, reports success, and
    then no key event ever arrives. There is no error to catch, so we check
    ``accessibility_trusted()`` at startup and say so loudly. The permission
    belongs to the host app -- your terminal -- not to blurt.
  - Microphone permission denied. macOS does NOT raise; it hands us a working
    stream full of zeros. A silent capture is therefore reported as a probable
    permission problem, never as an empty transcript, because feeding silence to
    Whisper produces confident hallucinated sentences.
  - Secure Event Input. While a password field or a terminal with Secure Keyboard
    Entry has focus, macOS discards synthetic Cmd+V system-wide. ``insert_text``
    returns False and we fall back to leaving the text on the clipboard. A
    transcript is never silently dropped -- the user spoke it, we owe them the words.
  - Ctrl+C arrives on the main thread only. Worker and listener threads are
    daemons, and shutdown is explicit and ordered so the interpreter never exits
    with a live PortAudio stream.

Python 3.9 floor: lazy annotations, typing generics only, no PEP 604 unions.
"""

from __future__ import annotations

import logging
import queue
import signal
import sys
import threading
import time
from typing import Any, Deque, Dict, List, Optional

from collections import deque

from . import hardware as _hardware
from .assistant import build_default_router
from .assistant.types import ActionResult
from .audio import AudioUnavailable, Recorder
from .cleanup import clean
from .config import Config, load_config
from .engines import NoEngineAvailable, select_engine
from .hotkey import HoldToTalk, UnsupportedHotkeyError, accessibility_trusted
from .inject import copy_to_clipboard, insert_text, secure_input_active
from .types import ASREngine, Hardware, Transcript

__all__ = ["BlurtApp", "StartupError", "HISTORY_LIMIT", "run"]

_log = logging.getLogger(__name__)

#: How many finished dictations to keep in memory. Small on purpose: this is a
#: convenience buffer for revert_last(), not a transcript archive, and dictated
#: text is exactly the kind of thing that should not accumulate in RAM forever.
HISTORY_LIMIT = 20

#: Longest we wait for an in-flight transcription during shutdown. Generous
#: relative to the floor machine's worst measured case (~8s) so Ctrl+C finishes
#: the dictation you just spoke instead of throwing it away.
_WORKER_DRAIN_TIMEOUT_S = 30.0

# Sentinel used to wake the worker for shutdown.
_STOP = None


class StartupError(RuntimeError):
    """Startup failed for a reason the user has to fix.

    The message is written to be printed as-is: it names what failed and what to
    do about it. ``__main__`` prints it and exits non-zero rather than showing a
    traceback, because a traceback tells the user nothing they can act on.
    """


def _say(message: str = "") -> None:
    """Print to stdout, flushed. Never raises, even with stdout detached."""
    try:
        print(message, flush=True)
    except Exception:  # pragma: no cover - stdout closed (launchd, py2app)
        pass


def _warn(message: str) -> None:
    """Print to stderr, flushed. Never raises."""
    try:
        print(message, file=sys.stderr, flush=True)
    except Exception:  # pragma: no cover - stderr closed
        pass


class BlurtApp:
    """One dictation session: owns the engine, the mic, the hotkey and the history.

    Usage::

        app = BlurtApp(load_config())
        app.startup()      # raises StartupError with an actionable message
        try:
            app.run()      # blocks until SIGINT
        finally:
            app.shutdown()

    Thread model: :meth:`startup`, :meth:`run` and :meth:`shutdown` belong to the
    main thread. Hotkey callbacks arrive on pynput's dispatch thread and do only
    cheap work. All transcription happens on one private worker thread, so
    dictations are serialised and can never overlap -- which also means the engine
    never sees a concurrent :meth:`transcribe`, something the engines do not support.
    """

    def __init__(self, cfg: Optional[Config] = None, hw: Optional[Hardware] = None) -> None:
        self.cfg: Config = cfg if cfg is not None else Config()
        self.hw: Optional[Hardware] = hw

        self._engine: Optional[ASREngine] = None
        self._engine_label: str = "unknown"
        self._recorder: Optional[Recorder] = None
        self._hotkey: Optional[HoldToTalk] = None
        # Assistant (command) mode: a second hotkey routes speech to actions
        # instead of pasting it. None when disabled or unavailable.
        self._assistant_hotkey: Optional[HoldToTalk] = None
        self._router: Optional[Any] = None
        # Which mode the CURRENT capture is for. Set on hold-start, read when the
        # capture is handed to the worker. Guards against both keys at once by
        # ignoring a second start while one is already armed.
        self._capture_mode: Optional[str] = None

        self._jobs: "queue.Queue[Any]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None

        self._history: Deque[Transcript] = deque(maxlen=HISTORY_LIMIT)
        self._history_lock = threading.Lock()
        self._reverted_marker: Optional[Transcript] = None

        self._stop_event = threading.Event()
        self._previous_handlers: Dict[int, Any] = {}
        self._started = False
        self._shut_down = False

    # -- introspection ------------------------------------------------------

    @property
    def history(self) -> List[Transcript]:
        """Finished dictations, oldest first. A copy; safe to iterate."""
        with self._history_lock:
            return list(self._history)

    @property
    def last_transcript(self) -> Optional[Transcript]:
        """The most recent dictation, or None if nothing has been said yet."""
        with self._history_lock:
            return self._history[-1] if self._history else None

    @property
    def engine_label(self) -> str:
        """Engine plus model, e.g. ``faster-whisper tiny.en``. For display only."""
        return self._engine_label

    # -- startup ------------------------------------------------------------

    def startup(self) -> None:
        """Detect, configure, load the model, and start listening.

        Blocks for as long as the model takes to load -- that is deliberate; see
        the module docstring. Raises :class:`StartupError` with a message meant
        for the user on any failure the user can fix.
        """
        if self._started:
            return

        _say("blurt: starting up")

        self.hw = self.hw if self.hw is not None else _hardware.detect()
        self._report_hardware(self.hw)

        self._engine = self._select_engine(self.hw)
        self._engine_label = self._describe_engine(self._engine)
        self._load_engine(self._engine)

        self._recorder = self._open_recorder()
        self._hotkey = self._build_hotkey()
        self._build_assistant()

        self._check_accessibility()

        self._worker = threading.Thread(
            target=self._worker_loop, name="blurt-transcribe", daemon=True
        )
        self._worker.start()

        try:
            self._hotkey.start()
        except Exception as exc:  # noqa: BLE001 - pynput failures are varied
            raise StartupError(
                "Could not start the hotkey listener.\n"
                f"  Underlying error: {type(exc).__name__}: {exc}\n"
                "  Fix: grant Input Monitoring and Accessibility to the app that "
                "launched blurt (your terminal) in System Settings > Privacy & "
                "Security, then try again."
            ) from exc

        if self._assistant_hotkey is not None:
            try:
                self._assistant_hotkey.start()
            except Exception as exc:  # noqa: BLE001 - degrade, don't die
                # The assistant is a bonus; a failure to bind its key must not
                # take down dictation, which is the core function.
                _warn(f"  assistant hotkey unavailable: {type(exc).__name__}: {exc}")
                self._assistant_hotkey = None

        self._started = True
        _say("")
        _say(f"Ready. Hold {self._hotkey.key_name} to dictate, Esc while holding to cancel.")
        if self._assistant_hotkey is not None:
            _say(
                f"Hold {self._assistant_hotkey.key_name} to issue a command "
                '("schedule lunch tomorrow at noon", "set a timer for 5 minutes", '
                '"open Safari").'
            )
        _say("Ctrl+C to quit.")
        _say("")

    def _report_hardware(self, hw: Hardware) -> None:
        rosetta = " (under Rosetta)" if hw.under_rosetta else ""
        _say(
            f"  hardware: {hw.cpu_brand}, {hw.physical_cores} cores, "
            f"{hw.ram_gb:.0f} GB, {hw.arch}{rosetta} -> tier '{hw.tier}'"
        )

    def _select_engine(self, hw: Hardware) -> ASREngine:
        try:
            return select_engine(self.cfg, hw)
        except NoEngineAvailable as exc:
            # The registry already writes a complete, user-facing explanation.
            raise StartupError(str(exc)) from exc

    @staticmethod
    def _describe_engine(engine: ASREngine) -> str:
        """Engine name plus the model it will load, when it has one to report."""
        name = getattr(engine, "name", "unknown")
        resolve = getattr(engine, "resolve_model", None)
        if callable(resolve):
            try:
                model = resolve()
            except Exception:  # noqa: BLE001 - display only, never fatal
                return str(name)
            if model:
                return f"{name} {model}"
        return str(name)

    def _load_engine(self, engine: ASREngine) -> None:
        """Load the model now, with visible progress and an honest duration.

        The first run also downloads weights, which is the one network call blurt
        makes; the message says so, so a long pause is explained rather than
        alarming.
        """
        label = self._engine_label
        try:
            print(f"  loading {label}... ", end="", flush=True)
        except Exception:  # pragma: no cover - stdout closed
            pass

        started = time.monotonic()
        try:
            engine.load()
        except BaseException as exc:  # noqa: BLE001 - loader errors are not just ImportError
            _say("failed")
            raise StartupError(
                f"Could not load the {label} model.\n"
                f"  Reason: {type(exc).__name__}: {exc}\n"
                "  If this is the first run, blurt downloads the model once and "
                "needs network access for that. Afterwards it works entirely offline."
            ) from exc

        _say(f"ready ({time.monotonic() - started:.1f}s)")

    def _open_recorder(self) -> Recorder:
        try:
            return Recorder(
                sample_rate=self.cfg.sample_rate, preroll_ms=self.cfg.preroll_ms
            )
        except AudioUnavailable as exc:
            raise StartupError(f"Microphone unavailable.\n  {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - PortAudio raises broadly
            raise StartupError(
                "Could not open the microphone.\n"
                f"  Underlying error: {type(exc).__name__}: {exc}"
            ) from exc

    def _build_hotkey(self) -> HoldToTalk:
        try:
            return HoldToTalk(
                key_name=self.cfg.hotkey,
                on_start=self._on_hold_start,
                on_stop=self._on_hold_stop,
                on_cancel=self._on_hold_cancel,
                min_hold_ms=self.cfg.min_hold_ms,
            )
        except UnsupportedHotkeyError as exc:
            raise StartupError(
                f"Cannot use hotkey {self.cfg.hotkey!r}.\n  {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - missing pynput surfaces here
            raise StartupError(
                "Could not set up the hotkey listener.\n"
                f"  Underlying error: {type(exc).__name__}: {exc}\n"
                "  Fix: python3 -m pip install pynput"
            ) from exc

    def _build_assistant(self) -> None:
        """Wire the command-mode router and its hotkey. Never fatal.

        The assistant is an addition to dictation, not a requirement for it, so
        every failure here degrades to "assistant off" with a warning rather than
        aborting startup. The router's dictate fallback is _deliver, so an
        unrecognised command is simply typed out.
        """
        if not getattr(self.cfg, "assistant_enabled", False):
            return

        assistant_key = getattr(self.cfg, "assistant_hotkey", "") or ""
        if not assistant_key or assistant_key == self.cfg.hotkey:
            if assistant_key == self.cfg.hotkey:
                _warn(
                    "  assistant hotkey equals the dictation hotkey; "
                    "assistant mode disabled to avoid a conflict."
                )
            return

        try:
            self._router = build_default_router(
                dictate_fallback=self._deliver_as_result
            )
        except Exception as exc:  # noqa: BLE001 - assistant is optional
            _warn(f"  assistant unavailable: {type(exc).__name__}: {exc}")
            self._router = None
            return

        try:
            self._assistant_hotkey = HoldToTalk(
                key_name=assistant_key,
                on_start=self._on_assistant_start,
                on_stop=self._on_assistant_stop,
                on_cancel=self._on_assistant_cancel,
                min_hold_ms=self.cfg.min_hold_ms,
            )
        except UnsupportedHotkeyError as exc:
            _warn(f"  assistant hotkey {assistant_key!r} unsupported: {exc}")
            self._assistant_hotkey = None
            self._router = None
        except Exception as exc:  # noqa: BLE001 - degrade, don't die
            _warn(f"  assistant hotkey unavailable: {type(exc).__name__}: {exc}")
            self._assistant_hotkey = None
            self._router = None

    def _deliver_as_result(self, text: str) -> "ActionResult":
        """Dictate fallback for the router: paste the text, report it as a result."""
        self._deliver(text)
        return ActionResult(ok=True, message=f"Dictated: {text}")

    def _check_accessibility(self) -> None:
        """Warn -- do not fail -- when the host app is untrusted.

        We warn rather than abort because the answer can be None (undeterminable)
        and because a user who has just granted permission may only need to
        restart their terminal. Failing here would be wrong more often than right.
        """
        if accessibility_trusted() is False:
            _warn("")
            _warn("  WARNING: this process is not trusted for Accessibility.")
            _warn("  The hotkey will never fire and pasting will be silently dropped.")
            _warn("  Fix: System Settings > Privacy & Security > Accessibility,")
            _warn("  and enable the app that launched blurt (your terminal, not blurt).")
            _warn("  Then quit that app completely and relaunch it.")

    # -- hotkey callbacks ---------------------------------------------------
    #
    # These run on pynput's dispatch thread. They must stay cheap: anything slow
    # here delays the next key press.

    def _begin_capture(self, mode: str) -> None:
        """Start recording for ``mode`` ("dictate" or "assistant").

        If a capture is already in progress -- e.g. both hotkeys held at once --
        the second start is ignored, because one recorder cannot serve two takes.
        """
        recorder = self._recorder
        if recorder is None:
            return
        if self._capture_mode is not None:
            return  # already recording in some mode; do not clobber it
        try:
            recorder.start()
        except AudioUnavailable as exc:
            _warn(f"blurt: cannot record: {exc}")
            return
        except Exception as exc:  # noqa: BLE001 - never kill the listener thread
            _warn(f"blurt: cannot record: {type(exc).__name__}: {exc}")
            return
        self._capture_mode = mode
        _say("  listening for a command..." if mode == "assistant" else "  recording...")

    def _end_capture(self, mode: str) -> None:
        """Stop recording and queue the capture, tagged with its mode."""
        recorder = self._recorder
        if recorder is None or self._capture_mode != mode:
            return  # not our capture (or none in progress)
        try:
            pcm = recorder.stop()
            # Read the silence verdict NOW. It is per-recorder state that the next
            # capture overwrites, and the worker may not look at it for seconds.
            was_silent = recorder.last_capture_was_silent()
            overflowed = recorder.last_capture_overflowed()
        except Exception as exc:  # noqa: BLE001 - never kill the listener thread
            _warn(f"blurt: capture failed: {type(exc).__name__}: {exc}")
            self._capture_mode = None
            return
        finally:
            self._capture_mode = None

        # Hand off immediately. Transcription is seconds of work and does not
        # belong on the thread that has to notice the next key press.
        self._jobs.put((pcm, was_silent, overflowed, mode))

    def _cancel_capture(self, mode: str) -> None:
        recorder = self._recorder
        if recorder is None or self._capture_mode != mode:
            return
        try:
            recorder.stop()  # discard the audio; nothing is queued
        except Exception:  # noqa: BLE001 - cancelling must never raise
            pass
        self._capture_mode = None
        _say("  cancelled")

    # Dictation hotkey callbacks.
    def _on_hold_start(self) -> None:
        self._begin_capture("dictate")

    def _on_hold_stop(self) -> None:
        self._end_capture("dictate")

    def _on_hold_cancel(self) -> None:
        self._cancel_capture("dictate")

    # Assistant (command) hotkey callbacks.
    def _on_assistant_start(self) -> None:
        self._begin_capture("assistant")

    def _on_assistant_stop(self) -> None:
        self._end_capture("assistant")

    def _on_assistant_cancel(self) -> None:
        self._cancel_capture("assistant")

    # -- worker -------------------------------------------------------------

    def _worker_loop(self) -> None:
        """Serialise every transcription onto one thread until told to stop."""
        while True:
            job = self._jobs.get()
            try:
                if job is _STOP:
                    return
                pcm, was_silent, overflowed, mode = job
                self._handle_capture(pcm, was_silent, overflowed, mode)
            except Exception:  # noqa: BLE001 - one bad dictation must not end the loop
                _log.exception("transcription job failed")
                _warn("blurt: that dictation failed; the app is still running.")
            finally:
                self._jobs.task_done()

    def _handle_capture(
        self, pcm: Any, was_silent: bool, overflowed: bool, mode: str = "dictate"
    ) -> None:
        """Transcribe one capture, clean it, and get it in front of the user."""
        engine = self._engine
        if engine is None:
            return

        sample_rate = max(1, int(self.cfg.sample_rate))
        frames = int(getattr(pcm, "shape", (0,))[0]) if hasattr(pcm, "shape") else 0
        audio_seconds = frames / float(sample_rate)

        if frames == 0:
            _warn("  no audio captured -- hold the key a little longer.")
            return

        if was_silent:
            # macOS gives a denied app a working stream of zeros rather than an
            # error, so this is the closest thing to a permission signal we get.
            # Report it as such: an "empty transcript" would send the user
            # looking for a bug in the model instead of at a checkbox.
            _warn("")
            _warn("  Nothing was heard on the microphone.")
            _warn("  Most likely: microphone permission is not granted.")
            _warn("  Check System Settings > Privacy & Security > Microphone and")
            _warn("  enable the app that launched blurt (your terminal).")
            _warn("  Also worth checking: the right input device in Sound settings,")
            _warn("  and that the mic is not hardware-muted.")
            return

        if overflowed:
            _warn("  (audio dropped frames -- the machine was busy; text may be clipped)")

        started = time.monotonic()
        try:
            raw = engine.transcribe(pcm, sample_rate)
        except Exception as exc:  # noqa: BLE001 - engines raise many things
            _log.exception("engine.transcribe failed")
            _warn(f"  transcription failed: {type(exc).__name__}: {exc}")
            return
        latency = time.monotonic() - started

        raw = raw or ""
        cleaned = clean(raw, self.cfg.cleanup_level, self.cfg.dictionary)

        transcript = Transcript(
            raw=raw if self.cfg.keep_raw_history else "",
            cleaned=cleaned,
            audio_seconds=audio_seconds,
            engine=self._engine_label,
            latency_seconds=latency,
        )
        with self._history_lock:
            self._history.append(transcript)

        self._report_timing(transcript)

        if not cleaned.strip():
            # Audio had signal but the engine found no words. Genuinely empty,
            # unlike the silent case above.
            _warn("  heard audio but no speech was recognised.")
            return

        if mode == "assistant" and self._router is not None:
            self._handle_command(cleaned)
        else:
            self._deliver(cleaned)

    def _handle_command(self, text: str) -> None:
        """Route a spoken command to an action and run it, announcing the result.

        The router's dictate fallback (wired in startup) pastes the text when no
        command matches, so nothing a user says is ever silently dropped.
        """
        from .assistant.system_actions import notify

        router = self._router
        try:
            action = router.route(text)
            result = router.execute(action)
        except Exception as exc:  # noqa: BLE001 - a bad command must not crash the app
            _log.exception("assistant command failed")
            _warn(f"  command failed: {type(exc).__name__}: {exc}")
            return

        message = getattr(result, "message", "") or ""
        ok = bool(getattr(result, "ok", False))
        _say(f"  {'OK' if ok else 'x'} {message}")
        # A desktop notification too, since the user is usually in another app
        # when issuing a command and will not be looking at this terminal.
        if getattr(action, "kind", "") != "dictate":
            try:
                notify("blurt", message)
            except Exception:  # noqa: BLE001 - notification is best-effort
                pass

    def _report_timing(self, transcript: Transcript) -> None:
        """Print real numbers for this machine.

        No realtime factor: Whisper pads every input to a 30-second window, so
        latency barely tracks utterance length and a ratio would imply a speedup
        for short phrases that does not exist.
        """
        _say(
            f"  {transcript.audio_seconds:.1f}s audio -> "
            f"{transcript.latency_seconds:.2f}s  ({transcript.engine})"
        )

    def _deliver(self, text: str) -> None:
        """Type the text, or leave it on the clipboard and say so. Never drop it."""
        ok = insert_text(
            text,
            paste_delay_ms=self.cfg.paste_delay_ms,
            restore_delay_ms=self.cfg.clipboard_restore_ms,
        )
        if ok:
            _say(f"  {text}")
            return

        # Pasting was blocked. The words still belong to the user, so put them
        # somewhere they can retrieve them and explain what happened.
        copy_to_clipboard(text)
        _warn("")
        if secure_input_active() is True:
            _warn("  Could not paste: macOS Secure Event Input is active.")
            _warn("  (A password field has focus, or a terminal has Secure Keyboard")
            _warn("   Entry enabled -- macOS blocks synthetic keystrokes system-wide.)")
        elif accessibility_trusted() is False:
            _warn("  Could not paste: no Accessibility permission.")
            _warn("  Grant it to the app that launched blurt in System Settings >")
            _warn("  Privacy & Security > Accessibility, then relaunch that app.")
        else:
            _warn("  Could not paste into the focused app.")
        _warn("  Your text is on the clipboard -- press Cmd+V to insert it:")
        _warn(f"  {text}")

    # -- the trust mechanism ------------------------------------------------

    def revert_last(self) -> bool:
        """Re-insert the RAW text of the last dictation, replacing the cleaned one.

        This is the promise that makes cleanup safe to enable: if the cleanup pass
        mangles something, one gesture (Opt+Z, once it is wired) gets back exactly
        what the engine heard. Returns True if raw text was delivered.

        v1 limitation, stated plainly: blurt inserts the raw text at the cursor
        but cannot delete the cleaned text it typed earlier -- :mod:`blurt.inject`
        exposes pasting only, with no way to send backspaces, and guessing at the
        cursor's position in someone else's app is how you destroy their document.
        So the user removes the cleaned copy. Reverting the same transcript twice
        is a no-op, and a transcript whose cleanup changed nothing is not worth
        reverting at all.
        """
        with self._history_lock:
            transcript = self._history[-1] if self._history else None

        if transcript is None:
            _warn("  nothing to revert yet.")
            return False

        if transcript is self._reverted_marker:
            _warn("  that dictation was already reverted.")
            return False

        if not self.cfg.keep_raw_history:
            _warn("  cannot revert: raw history is disabled (keep_raw_history=false).")
            return False

        raw = (transcript.raw or "").strip()
        if not raw:
            _warn("  cannot revert: no raw text was kept for that dictation.")
            return False

        if raw == (transcript.cleaned or "").strip():
            _warn("  nothing to revert: cleanup did not change that dictation.")
            return False

        _say("  reverting to raw transcript (delete the cleaned text above it):")
        self._deliver(raw)
        self._reverted_marker = transcript
        return True

    # -- run / shutdown -----------------------------------------------------

    def run(self) -> None:
        """Block until SIGINT/SIGTERM. Calls :meth:`startup` if needed.

        Signal handlers can only be installed from the main thread; when called
        from anywhere else we simply wait, and the caller owns the shutdown.
        """
        if not self._started:
            self.startup()

        installed = self._install_signal_handlers()

        try:
            # Wake periodically rather than blocking forever: a timed wait keeps
            # the main thread responsive to signals on every platform and Python
            # build, without relying on interruptible lock acquisition.
            while not self._stop_event.is_set():
                self._stop_event.wait(0.5)
        except KeyboardInterrupt:
            # Reachable when handlers could not be installed (non-main thread).
            pass
        finally:
            if installed:
                self._restore_signal_handlers()

        _say("")
        _say("blurt: shutting down")

    def _install_signal_handlers(self) -> bool:
        self._previous_handlers.clear()
        ok = False
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous_handlers[signum] = signal.signal(signum, self._on_signal)
                ok = True
            except (ValueError, OSError, RuntimeError):
                # ValueError: not the main thread. Not fatal -- run() still waits.
                continue
        return ok

    def _restore_signal_handlers(self) -> None:
        for signum, handler in getattr(self, "_previous_handlers", {}).items():
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError, RuntimeError, TypeError):
                pass

    def _on_signal(self, signum: int, frame: Any) -> None:
        """Signal handler: set a flag and return. Nothing else is async-safe."""
        self._stop_event.set()

    def request_stop(self) -> None:
        """Ask :meth:`run` to return. Safe from any thread."""
        self._stop_event.set()

    def shutdown(self) -> None:
        """Tear down in the order that loses the least. Safe to call twice.

        Hotkey first (no new captures), then drain the worker so a transcription
        you already spoke still lands, then the mic, then the model.
        """
        if self._shut_down:
            return
        self._shut_down = True
        self._stop_event.set()

        for attr in ("_hotkey", "_assistant_hotkey"):
            hotkey = getattr(self, attr, None)
            setattr(self, attr, None)
            if hotkey is not None:
                try:
                    hotkey.stop()
                except Exception:  # noqa: BLE001 - teardown is best effort
                    _log.debug("%s stop failed", attr, exc_info=True)

        worker = self._worker
        self._worker = None
        if worker is not None and worker.is_alive():
            self._jobs.put(_STOP)
            worker.join(timeout=_WORKER_DRAIN_TIMEOUT_S)
            if worker.is_alive():  # pragma: no cover - pathological engine hang
                _warn("blurt: a transcription is still running; exiting anyway.")

        recorder = self._recorder
        self._recorder = None
        if recorder is not None:
            try:
                recorder.close()
            except Exception:  # noqa: BLE001 - teardown is best effort
                _log.debug("recorder close failed", exc_info=True)

        engine = self._engine
        self._engine = None
        if engine is not None:
            try:
                engine.unload()
            except Exception:  # noqa: BLE001 - teardown is best effort
                _log.debug("engine unload failed", exc_info=True)

        self._started = False


def run(cfg: Optional[Config] = None, hw: Optional[Hardware] = None) -> int:
    """Run the dictation daemon until interrupted. Returns a process exit code.

    Turns every expected failure into a printed message plus a non-zero exit,
    never a traceback: a traceback at startup tells the user nothing they can act on.
    """
    app = BlurtApp(cfg if cfg is not None else load_config(), hw)
    try:
        app.startup()
    except StartupError as exc:
        _warn("")
        _warn(f"blurt: {exc}")
        return 1
    except KeyboardInterrupt:
        _warn("")
        return 130

    try:
        app.run()
    finally:
        app.shutdown()
    return 0
