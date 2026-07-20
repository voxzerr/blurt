"""System-level side effects for blurt's assistant: open apps, set timers, notify.

This module holds the three actions that talk to the operating system without
going through EventKit: launching a macOS application, scheduling a one-shot
timer, and posting a Desktop notification. Everything here is driven from a
background worker thread (the same thread that finishes a dictation), so every
call must be thread-safe and return quickly instead of blocking the worker.

Design choices that matter:

  * No pyobjc / AppKit imports. These actions only need the standard library
    (:mod:`subprocess`, :mod:`threading`), so this module imports cleanly on any
    platform and can be unit-tested off a Mac -- the macOS-specific parts are the
    ``open`` and ``osascript`` binaries, which only exist at *run* time on macOS.
  * No shell. Every command is passed as an argument *list* to
    :func:`subprocess.run`, never a shell string, so a spoken app name like
    ``foo"; rm -rf ~`` is handled as a literal (missing) app name, not executed.
  * Only create / open. Nothing here deletes or overwrites anything: opening an
    app, arming a timer and showing a notification are all harmless and
    reversible, in keeping with blurt's v1 "create-only" safety rule.

What can go wrong, and how it is contained:

  * The app name is unknown -> ``open -a`` exits non-zero; :func:`open_app`
    reports "Couldn't find an app called '<name>'" instead of raising.
  * ``open`` / ``osascript`` is missing or hangs (non-macOS, or a stuck
    LaunchServices) -> each call uses a short timeout and turns any
    :class:`OSError` / :class:`subprocess.TimeoutExpired` into a plain-language
    :class:`~blurt.assistant.types.ActionResult`.
  * A garbage duration ("set a timer for -5 minutes", or a parse that yielded a
    huge number) -> :meth:`TimerService.schedule` clamps to a sane range and
    refuses absurd values rather than arming a runaway timer.
  * A notification fails -> :func:`notify` swallows *every* ordinary exception
    and returns ``None``. Notifying is the least important thing blurt does; a
    failed notification must never crash the dictation that triggered it.
"""
from __future__ import annotations

import math
import subprocess
import threading
from typing import Dict, Optional

from .types import ActionResult

# Short timeouts: ``open -a`` and ``osascript`` both hand off almost immediately
# on a healthy Mac. If they block longer than this, something is wrong and we
# would rather return a message than wedge the worker thread.
_OPEN_TIMEOUT_SECONDS = 10.0
_OSASCRIPT_TIMEOUT_SECONDS = 10.0

# A timer longer than a day is almost certainly a misparse ("at 2" -> 200000
# minutes), not a real request. Refuse it rather than arm it.
_MAX_TIMER_MINUTES = 24 * 60


def _sanitize_app_name(name: str) -> str:
    """Normalize a spoken app name: strip, collapse whitespace, tidy casing.

    Speech-to-text usually hands us lowercase words with stray spacing, e.g.
    ``"  google   chrome "``. We collapse the whitespace and title-case only the
    words that are *entirely* lowercase letters, so ``"google chrome"`` becomes
    ``"Google Chrome"`` while already-cased names like ``"iTerm"`` or ``"VLC"``
    are left untouched. ``open -a`` is case-insensitive anyway, so this is purely
    to make the spoken-back message read nicely. Returns "" if nothing usable is
    left (caller treats that as "no app name given").
    """
    if not isinstance(name, str):
        return ""
    cleaned = " ".join(name.split())
    if not cleaned:
        return ""
    words = []
    for word in cleaned.split(" "):
        if word.isalpha() and word.islower():
            words.append(word.capitalize())
        else:
            words.append(word)
    return " ".join(words)


def open_app(name: str) -> ActionResult:
    """Launch a macOS application by (spoken) name via ``open -a <name>``.

    Returns ``ActionResult(ok=True, "Opened Safari")`` when LaunchServices could
    open the app, and ``ActionResult(ok=False, "Couldn't find an app called
    'Safri'")`` when it could not. Never raises: a missing ``open`` binary
    (non-macOS) or a timeout is turned into an ``ok=False`` result. The name is
    passed as a list argument, never through a shell, so it cannot inject.
    """
    app = _sanitize_app_name(name)
    if not app:
        return ActionResult(ok=False, message="I didn't catch which app to open.")

    try:
        proc = subprocess.run(
            ["open", "-a", app],
            capture_output=True,
            text=True,
            timeout=_OPEN_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        # No ``open`` binary -- e.g. running the tests on a non-macOS box.
        return ActionResult(ok=False, message="I can't open apps on this system.")
    except subprocess.TimeoutExpired:
        return ActionResult(ok=False, message="Opening '{0}' took too long.".format(app))
    except OSError:
        return ActionResult(ok=False, message="I couldn't open '{0}'.".format(app))

    if proc.returncode == 0:
        return ActionResult(ok=True, message="Opened {0}".format(app))
    return ActionResult(ok=False, message="Couldn't find an app called '{0}'".format(app))


def _format_minutes(minutes: float) -> str:
    """Render a positive, finite minute count for a human, e.g. "5 minutes"."""
    if abs(minutes - round(minutes)) < 1e-9:
        whole = int(round(minutes))
        unit = "minute" if whole == 1 else "minutes"
        return "{0} {1}".format(whole, unit)
    return "{0:g} minutes".format(minutes)


class TimerService:
    """Schedule one-shot timers that fire a Desktop notification when they elapse.

    Backed by :class:`threading.Timer`, one daemon thread per live timer. Each
    scheduled timer is stashed in ``self._timers`` and kept there until it fires
    (or is cancelled) so the Timer object is not garbage-collected mid-wait --
    the classic threading.Timer footgun. All bookkeeping is done under a lock,
    so :meth:`schedule` is safe to call from the background worker thread.

    What can go wrong: an absurd duration (negative, zero, NaN, or longer than a
    day) is rejected up front with an ``ok=False`` result, so no runaway timer is
    ever armed. When a timer fires it calls :func:`notify`, which never raises,
    so the timer thread always exits cleanly.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._timers = {}  # type: Dict[int, threading.Timer]
        self._counter = 0

    def schedule(self, minutes: float, label: str) -> ActionResult:
        """Arm a timer to fire in ``minutes`` and announce ``label`` when it does.

        Returns ``ActionResult(ok=True, "Timer set for 5 minutes")`` on success
        (with the label appended when one was given). Guards against garbage
        durations -- negatives, zero, non-numbers, NaN/inf and anything over 24
        hours -- returning an ``ok=False`` explanation instead of scheduling.
        """
        try:
            mins = float(minutes)
        except (TypeError, ValueError):
            return ActionResult(ok=False, message="I couldn't tell how long to set the timer for.")

        if not math.isfinite(mins):
            return ActionResult(ok=False, message="That timer length doesn't make sense.")
        if mins <= 0:
            return ActionResult(ok=False, message="I can't set a timer for zero or negative time.")
        if mins > _MAX_TIMER_MINUTES:
            return ActionResult(
                ok=False,
                message="That's longer than a day; I can only set timers up to 24 hours.",
            )

        clean_label = (label or "").strip()
        delay_seconds = mins * 60.0

        with self._lock:
            self._counter += 1
            timer_id = self._counter
            timer = threading.Timer(delay_seconds, self._fire, args=(timer_id, clean_label))
            timer.daemon = True
            self._timers[timer_id] = timer

        timer.start()

        pretty = _format_minutes(mins)
        message = "Timer set for {0}".format(pretty)
        if clean_label:
            message = "{0}: {1}".format(message, clean_label)
        return ActionResult(ok=True, message=message)

    def _fire(self, timer_id: int, label: str) -> None:
        """Callback run on the timer thread: notify, then drop the reference."""
        try:
            body = "{0} - time's up".format(label) if label else "Time's up"
            notify("Timer", body)
        finally:
            with self._lock:
                self._timers.pop(timer_id, None)

    def cancel_all(self) -> None:
        """Cancel every pending timer. Safe to call on shutdown; never raises."""
        with self._lock:
            timers = list(self._timers.values())
            self._timers.clear()
        for timer in timers:
            try:
                timer.cancel()
            except Exception:
                # Cancelling is best-effort; a timer that already fired is fine.
                pass


def _applescript_escape(text: str) -> str:
    """Escape a Python string for safe embedding in an AppleScript "..." literal.

    Backslashes and double-quotes are the only characters AppleScript treats
    specially inside a quoted string, so we escape those. Real newlines can't
    live in a one-line string literal, so we flatten them to spaces. The result
    is dropped straight into the ``osascript -e`` program text.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    return text.replace("\\", "\\\\").replace('"', '\\"')


def notify(title: str, message: str) -> None:
    """Post a macOS Desktop notification via ``osascript``. NEVER raises.

    Runs ``osascript -e 'display notification "<message>" with title "<title>"'``
    with both strings AppleScript-escaped and passed as a single list argument
    (no shell). Any failure -- missing ``osascript``, a timeout, a bad string --
    is swallowed and the function returns ``None``. Notifying is the lowest-stakes
    thing blurt does; a failed notification must never crash the dictation that
    asked for it.
    """
    try:
        safe_title = _applescript_escape(title if title is not None else "")
        safe_message = _applescript_escape(message if message is not None else "")
        script = 'display notification "{0}" with title "{1}"'.format(safe_message, safe_title)
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=_OSASCRIPT_TIMEOUT_SECONDS,
        )
    except Exception:
        # Deliberately broad: no notification failure may propagate to the
        # dictation worker. (KeyboardInterrupt / SystemExit still propagate.)
        return None
    return None


__all__ = ["open_app", "TimerService", "notify"]
