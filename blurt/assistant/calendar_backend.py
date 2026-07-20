"""Create calendar events and reminders through EventKit, for blurt's assistant.

This is the one assistant backend that talks to macOS system frameworks. It wraps
``EKEventStore`` so the rest of blurt can say "add lunch tomorrow at noon" and get
back a plain-language :class:`~blurt.assistant.types.ActionResult` describing
exactly what happened -- never a traceback, never a silent write.

WHAT THIS MODULE WILL AND WILL NOT DO
-------------------------------------
It only ever CREATES -- events and reminders. It never edits and never deletes:
those are irreversible and out of scope for v1 (see the safety notes in the repo).
It also creates NOTHING at import or construction time. Importing this module and
constructing :class:`CalendarBackend` touch no system state at all; the EventKit
framework is imported lazily *inside* methods and the ``EKEventStore`` is built on
first use. A calendar entry appears only when :meth:`CalendarBackend.create_event`
or :meth:`CalendarBackend.create_reminder` is called, i.e. only when the user
deliberately asked for it out loud.

IMPORT SAFETY
-------------
``import EventKit`` happens inside methods, never at module top, so this file
imports cleanly on any machine (including CI runners with no pyobjc, and Intel or
Apple Silicon alike) and in unit tests. If EventKit cannot be imported,
:meth:`~CalendarBackend.available` returns ``False`` and records why in
``last_reason`` instead of raising. Every method converts every failure into
``ActionResult(ok=False, ...)``; none of them raise.

THE TCC REALITY (READ THIS BEFORE DEBUGGING "IT WON'T PROMPT")
--------------------------------------------------------------
Calendar and Reminders are protected by TCC, exactly like the microphone. The
privacy grant is attached to the *host application*, not to Python and not to
blurt. When blurt is run from a shell, the host is the **terminal application**
(Terminal, iTerm2, VS Code's integrated terminal, ...), so it is that app the user
must tick in System Settings > Privacy & Security > Calendars (and > Reminders).
The very first ``create_*`` call (or :meth:`~CalendarBackend.request_access`) is
what triggers the system permission prompt -- there is no separate "ask" step the
user sees before then. Until they respond, the authorization status sits at
``notDetermined`` and the first create will drive the prompt.

macOS 13 vs 14 -- TWO DIFFERENT ASK APIS
----------------------------------------
The access-request selector changed in macOS 14:

  * macOS 13:  ``requestAccessToEntityType:completion:`` (one call, per entity type)
  * macOS 14+: ``requestFullAccessToEventsWithCompletion:`` and, separately,
               ``requestFullAccessToRemindersWithCompletion:``

We detect which exists with ``hasattr`` on the store and call the right one, so
this works on both. All three take a completion block of shape
``(BOOL granted, NSError *error)``.

WHAT ELSE CAN GO WRONG
----------------------
  * **No default calendar / list.** ``defaultCalendarForNewEvents`` (or
    ``...ForNewReminders``) can be ``nil`` -- e.g. every calendar is read-only, or
    the account that owns the default was removed. We surface that as a clear
    ``ok=False`` result telling the user to pick a default, rather than crashing.
  * **The access callback and run loops.** The request API is asynchronous: the
    completion block fires on an EventKit-managed queue, not necessarily after the
    caller's run loop spins. We bridge it to a synchronous answer with a
    ``threading.Event`` and a timeout, so a user who ignores the system dialog (or
    an environment where the callback never arrives) makes us time out and return
    ``False`` -- we never hang forever. On timeout we also re-read the
    authorization status, so a grant that landed without our callback still counts.
  * **Naive datetimes.** Python datetimes without a tzinfo are treated as *local*
    time. ``datetime.timestamp()`` already interprets a naive datetime in the
    machine's local zone, which is exactly what a spoken "noon tomorrow" means.

Python 3.9 floor: ``from __future__ import annotations`` and ``typing.Optional`` /
``Tuple`` forms -- no PEP 604 unions, no ``list[...]`` at runtime.
"""
from __future__ import annotations

import datetime
import logging
import threading
from typing import Any, Optional, Tuple

from .types import ActionResult

__all__ = ["CalendarBackend"]

_log = logging.getLogger(__name__)

# How long request_access waits for the async completion before giving up. Long
# enough that a user has time to read and answer the system dialog on the first
# call; bounded so we never block the caller indefinitely.
_ACCESS_TIMEOUT_S = 60.0

# EKAuthorizationStatus values, verified on the floor machine (macOS 13.7.8,
# pyobjc-framework-EventKit 12.0). We read them from the framework by name at
# runtime and only fall back to these literals if a name is somehow absent.
_STATUS_NOT_DETERMINED = 0
_STATUS_RESTRICTED = 1
_STATUS_DENIED = 2
_STATUS_AUTHORIZED = 3   # == EKAuthorizationStatusFullAccess on macOS 14+
_STATUS_WRITE_ONLY = 4   # macOS 14+, enough to create events


def _friendly_datetime(dt: datetime.datetime) -> str:
    """Format a datetime like ``Tue Jul 22, 12:00 PM``.

    Built by hand rather than with ``%-I``/``%-d`` strftime codes, which are not
    portable across platforms; this keeps the message identical everywhere and
    makes it trivial to assert on in tests.
    """
    weekday = dt.strftime("%a")
    month = dt.strftime("%b")
    hour12 = dt.hour % 12
    if hour12 == 0:
        hour12 = 12
    meridiem = "AM" if dt.hour < 12 else "PM"
    return "{wd} {mon} {day}, {h}:{m:02d} {ap}".format(
        wd=weekday, mon=month, day=dt.day, h=hour12, m=dt.minute, ap=meridiem
    )


class CalendarBackend:
    """EventKit-backed creator of calendar events and reminders.

    Construction is inert: it imports nothing macOS-specific and creates no
    ``EKEventStore``. The store is built lazily on first real use and cached for
    reuse. Every public method is failure-tolerant -- it returns ``False`` or an
    ``ActionResult(ok=False, ...)`` on any error and records a human-readable
    explanation in :attr:`last_reason`; none of them raise.
    """

    def __init__(self) -> None:
        self._store: Optional[Any] = None
        self._last_reason: Optional[str] = None

    # -- introspection ----------------------------------------------------

    @property
    def last_reason(self) -> Optional[str]:
        """Why the last ``available``/request/create decision went the way it did.

        Set whenever EventKit is missing, access is denied, or a create fails.
        ``None`` before anything interesting has happened. Intended for logs and
        diagnostics, not for control flow.
        """
        return self._last_reason

    def available(self) -> bool:
        """Return True if EventKit imports and calendar access is not blocked.

        "Not blocked" means the calendar (event) authorization status is neither
        ``denied`` nor ``restricted``. ``notDetermined`` counts as available on
        purpose: the user simply has not been asked yet, and the first
        :meth:`create_event` will trigger the system prompt. Reminders carry their
        own status, checked separately when a reminder is actually created.

        Never prompts and never raises. Records the blocking reason in
        :attr:`last_reason` when it returns ``False``.
        """
        eventkit = self._import("EventKit")
        if eventkit is None:
            return False
        status = self._status(eventkit, self._entity_event(eventkit))
        denied = getattr(eventkit, "EKAuthorizationStatusDenied", _STATUS_DENIED)
        restricted = getattr(
            eventkit, "EKAuthorizationStatusRestricted", _STATUS_RESTRICTED
        )
        if status in (denied, restricted):
            self._last_reason = (
                "Calendar access is turned off. Grant it in System Settings > "
                "Privacy & Security > Calendars (to the terminal running blurt)."
            )
            return False
        return True

    # -- access -----------------------------------------------------------

    def request_access(self, timeout: float = _ACCESS_TIMEOUT_S) -> bool:
        """Ask macOS for calendar and reminders access; block briefly for the answer.

        Requests events and reminders separately (they are distinct TCC grants),
        using ``requestFullAccess...`` on macOS 14+ and
        ``requestAccessToEntityType:completion:`` on macOS 13. The API is
        asynchronous; we bridge each call to a synchronous result with a
        ``threading.Event`` and ``timeout`` so we never hang.

        Returns ``True`` when calendar (event) access ends up granted -- the
        headline capability :meth:`available` keys on. Reminders access is
        requested too; its outcome governs :meth:`create_reminder` on its own and
        does not, by itself, make this return ``False``. Never raises.

        ``timeout`` is the per-request ceiling in seconds. The default is generous
        because the first call shows a dialog the user must answer.
        """
        eventkit = self._import("EventKit")
        if eventkit is None:
            return False
        store = self._get_store(eventkit)
        if store is None:
            return False

        events_ok = self._request_one(
            eventkit, store, self._entity_event(eventkit), timeout
        )
        # Always ask for reminders too, so a single voice session can do both
        # without a second surprise prompt later. Its result is recorded but does
        # not gate the return value.
        self._request_one(
            eventkit, store, self._entity_reminder(eventkit), timeout
        )
        return events_ok

    def _request_one(
        self, eventkit: Any, store: Any, entity_type: int, timeout: float
    ) -> bool:
        """Drive one access request to a synchronous granted/not-granted answer.

        Picks the macOS-14 ``requestFullAccess...`` selector when present, else the
        macOS-13 ``requestAccessToEntityType:completion:``. Waits up to ``timeout``
        for the completion block. On timeout (or any error) it re-reads the
        authorization status, so a grant that arrived without our callback still
        counts as success. Never raises.
        """
        done = threading.Event()
        box = {"granted": False}

        def _completion(granted: bool, error: Any) -> None:
            box["granted"] = bool(granted)
            if error is not None:
                try:
                    _log.debug("access request error: %s", error.localizedDescription())
                except Exception:  # pragma: no cover - defensive
                    pass
            done.set()

        try:
            is_event = entity_type == self._entity_event(eventkit)
            if is_event:
                full = getattr(store, "requestFullAccessToEventsWithCompletion_", None)
            else:
                full = getattr(
                    store, "requestFullAccessToRemindersWithCompletion_", None
                )

            if full is not None:
                # macOS 14+ full-access selector.
                full(_completion)
            else:
                # macOS 13 selector, per entity type.
                store.requestAccessToEntityType_completion_(entity_type, _completion)
        except Exception:
            _log.debug("could not start access request", exc_info=True)
            # Fall through to a status re-check below rather than declaring failure
            # outright -- access may already have been granted in a prior session.
            done.set()

        got_callback = done.wait(max(0.0, timeout))
        if got_callback and box["granted"]:
            return True

        # Either the callback said no, or it never came. Trust the recorded TCC
        # status as the source of truth in both cases.
        status = self._status(eventkit, entity_type)
        return self._status_is_granted(eventkit, status)

    # -- creation ---------------------------------------------------------

    def create_event(
        self,
        title: str,
        start: datetime.datetime,
        end: datetime.datetime,
        notes: Optional[str] = None,
    ) -> ActionResult:
        """Create a timed calendar event and report what happened.

        Builds an ``EKEvent`` on the default calendar for new events, sets its
        title / start / end / notes, and saves it. ``start`` and ``end`` are
        Python datetimes; naive ones are treated as local time. Returns
        ``ActionResult(ok=True, ...)`` on success with a message like
        ``Added 'Lunch with Sam' to your calendar for Tue Jul 22, 12:00 PM``.

        On any failure -- EventKit missing, access denied, no default calendar, or
        a save error -- returns ``ok=False`` with the real reason. Never raises.
        """
        clean_title = (title or "").strip()
        if not clean_title:
            return ActionResult(
                ok=False, message="I didn't catch what to put on your calendar."
            )

        eventkit = self._import("EventKit")
        foundation = self._import("Foundation")
        if eventkit is None or foundation is None:
            return ActionResult(
                ok=False,
                message=(
                    "Couldn't add '{t}': the calendar system (EventKit) isn't "
                    "available on this machine.".format(t=clean_title)
                ),
            )

        granted, reason = self._ensure_access(
            eventkit, self._entity_event(eventkit), "Calendar", "Calendars"
        )
        if not granted:
            return ActionResult(
                ok=False,
                message="Couldn't add '{t}' to your calendar: {r}".format(
                    t=clean_title, r=reason
                ),
            )

        store = self._get_store(eventkit)
        if store is None:
            return ActionResult(
                ok=False,
                message="Couldn't add '{t}': the calendar store wouldn't open.".format(
                    t=clean_title
                ),
            )

        try:
            calendar = store.defaultCalendarForNewEvents()
        except Exception:
            calendar = None
        if calendar is None:
            return ActionResult(
                ok=False,
                message=(
                    "Couldn't add '{t}': no default calendar is set. Open Calendar "
                    "and choose a default calendar for new events.".format(t=clean_title)
                ),
            )

        try:
            event = eventkit.EKEvent.eventWithEventStore_(store)
            event.setTitle_(clean_title)
            event.setCalendar_(calendar)
            event.setStartDate_(self._to_nsdate(foundation, start))
            event.setEndDate_(self._to_nsdate(foundation, end))
            if notes:
                event.setNotes_(notes)

            span = getattr(eventkit, "EKSpanThisEvent", 0)
            success, error = store.saveEvent_span_error_(event, span, None)
        except Exception as exc:  # pragma: no cover - defensive
            _log.debug("saving event failed", exc_info=True)
            self._last_reason = str(exc)
            return ActionResult(
                ok=False,
                message="Couldn't add '{t}' to your calendar: {e}".format(
                    t=clean_title, e=exc
                ),
            )

        if not success:
            detail = self._error_text(error) or "the save was rejected"
            self._last_reason = detail
            return ActionResult(
                ok=False,
                message="Couldn't add '{t}' to your calendar: {e}.".format(
                    t=clean_title, e=detail
                ),
            )

        return ActionResult(
            ok=True,
            message="Added '{t}' to your calendar for {when}".format(
                t=clean_title, when=_friendly_datetime(start)
            ),
        )

    def create_reminder(
        self, title: str, due: Optional[datetime.datetime] = None
    ) -> ActionResult:
        """Create a reminder on the default Reminders list; report what happened.

        Builds an ``EKReminder`` on ``defaultCalendarForNewReminders`` with the
        given title. If ``due`` is provided it is set as due date components (local
        wall-clock time), and a best-effort alarm is attached so the reminder
        actually alerts. Returns ``ActionResult(ok=True, ...)`` on success, e.g.
        ``Added reminder 'Buy milk' for Tue Jul 22, 12:00 PM`` (or without the
        time clause when no due date was given).

        On any failure -- EventKit missing, access denied, no default list, or a
        save error -- returns ``ok=False`` with the real reason. Never raises.
        """
        clean_title = (title or "").strip()
        if not clean_title:
            return ActionResult(ok=False, message="I didn't catch what to remind you about.")

        eventkit = self._import("EventKit")
        foundation = self._import("Foundation")
        if eventkit is None or foundation is None:
            return ActionResult(
                ok=False,
                message=(
                    "Couldn't add reminder '{t}': the Reminders system (EventKit) "
                    "isn't available on this machine.".format(t=clean_title)
                ),
            )

        granted, reason = self._ensure_access(
            eventkit, self._entity_reminder(eventkit), "Reminders", "Reminders"
        )
        if not granted:
            return ActionResult(
                ok=False,
                message="Couldn't add reminder '{t}': {r}".format(
                    t=clean_title, r=reason
                ),
            )

        store = self._get_store(eventkit)
        if store is None:
            return ActionResult(
                ok=False,
                message="Couldn't add reminder '{t}': the reminders store wouldn't "
                "open.".format(t=clean_title),
            )

        try:
            rlist = store.defaultCalendarForNewReminders()
        except Exception:
            rlist = None
        if rlist is None:
            return ActionResult(
                ok=False,
                message=(
                    "Couldn't add reminder '{t}': no default Reminders list is set. "
                    "Open Reminders and choose a default list.".format(t=clean_title)
                ),
            )

        try:
            reminder = eventkit.EKReminder.reminderWithEventStore_(store)
            reminder.setTitle_(clean_title)
            reminder.setCalendar_(rlist)
            if due is not None:
                components = self._due_components(foundation, due)
                if components is not None:
                    reminder.setDueDateComponents_(components)
                # A reminder with only due-date components does not alert; attach an
                # absolute-date alarm so it actually fires. Best effort: a failure
                # here must not sink the reminder itself.
                try:
                    alarm = eventkit.EKAlarm.alarmWithAbsoluteDate_(
                        self._to_nsdate(foundation, due)
                    )
                    reminder.addAlarm_(alarm)
                except Exception:  # pragma: no cover - defensive
                    _log.debug("could not attach reminder alarm", exc_info=True)

            success, error = store.saveReminder_commit_error_(reminder, True, None)
        except Exception as exc:  # pragma: no cover - defensive
            _log.debug("saving reminder failed", exc_info=True)
            self._last_reason = str(exc)
            return ActionResult(
                ok=False,
                message="Couldn't add reminder '{t}': {e}".format(
                    t=clean_title, e=exc
                ),
            )

        if not success:
            detail = self._error_text(error) or "the save was rejected"
            self._last_reason = detail
            return ActionResult(
                ok=False,
                message="Couldn't add reminder '{t}': {e}.".format(
                    t=clean_title, e=detail
                ),
            )

        if due is not None:
            message = "Added reminder '{t}' for {when}".format(
                t=clean_title, when=_friendly_datetime(due)
            )
        else:
            message = "Added reminder '{t}'".format(t=clean_title)
        return ActionResult(ok=True, message=message)

    # -- internals --------------------------------------------------------

    def _import(self, module_name: str) -> Optional[Any]:
        """Import a pyobjc framework by name, or return None and record why.

        Deliberately generic so EventKit and Foundation share one guarded path.
        Never raises; a missing or broken pyobjc install just yields ``None``.
        """
        try:
            return __import__(module_name)
        except Exception as exc:  # pragma: no cover - non-macOS or broken pyobjc
            self._last_reason = "{m} could not be imported: {e}".format(
                m=module_name, e=exc
            )
            _log.debug("could not import %s", module_name, exc_info=True)
            return None

    def _get_store(self, eventkit: Any) -> Optional[Any]:
        """Return the cached ``EKEventStore``, building it on first use.

        Constructing the store creates no user-visible data; it is just the handle
        every EventKit call needs. Cached so repeated actions reuse one store.
        Returns ``None`` (recording the reason) if construction fails.
        """
        if self._store is not None:
            return self._store
        try:
            self._store = eventkit.EKEventStore.alloc().init()
        except Exception as exc:  # pragma: no cover - defensive
            self._last_reason = "EKEventStore could not be created: {e}".format(e=exc)
            _log.debug("EKEventStore construction failed", exc_info=True)
            return None
        return self._store

    def _entity_event(self, eventkit: Any) -> int:
        return getattr(eventkit, "EKEntityTypeEvent", 0)

    def _entity_reminder(self, eventkit: Any) -> int:
        return getattr(eventkit, "EKEntityTypeReminder", 1)

    def _status(self, eventkit: Any, entity_type: int) -> Optional[int]:
        """Read the TCC authorization status for an entity type, or None on error.

        This is a class method on ``EKEventStore`` and needs no instance, so it is
        safe to call from :meth:`available` without constructing a store.
        """
        try:
            return int(
                eventkit.EKEventStore.authorizationStatusForEntityType_(entity_type)
            )
        except Exception:  # pragma: no cover - defensive
            _log.debug("could not read authorization status", exc_info=True)
            return None

    def _status_is_granted(self, eventkit: Any, status: Optional[int]) -> bool:
        """True if ``status`` is an access level that permits creating entries."""
        if status is None:
            return False
        authorized = getattr(
            eventkit, "EKAuthorizationStatusAuthorized", _STATUS_AUTHORIZED
        )
        full = getattr(eventkit, "EKAuthorizationStatusFullAccess", _STATUS_AUTHORIZED)
        write_only = getattr(
            eventkit, "EKAuthorizationStatusWriteOnly", _STATUS_WRITE_ONLY
        )
        return status in (authorized, full, write_only)

    def _ensure_access(
        self, eventkit: Any, entity_type: int, feature: str, pane: str
    ) -> Tuple[bool, str]:
        """Make sure we may write ``entity_type``, prompting once if undetermined.

        Returns ``(True, "")`` when access is granted. Returns ``(False, reason)``
        when it is denied/restricted, or still not granted after prompting. This is
        where the very first create triggers the system dialog. ``feature`` and
        ``pane`` only shape the human-readable reason (e.g. "Calendar" and the
        "Calendars" Settings pane). Never raises.
        """
        denied = getattr(eventkit, "EKAuthorizationStatusDenied", _STATUS_DENIED)
        restricted = getattr(
            eventkit, "EKAuthorizationStatusRestricted", _STATUS_RESTRICTED
        )
        not_determined = getattr(
            eventkit, "EKAuthorizationStatusNotDetermined", _STATUS_NOT_DETERMINED
        )

        status = self._status(eventkit, entity_type)
        if status in (denied, restricted):
            reason = (
                "{f} access was denied. Grant it in System Settings > Privacy & "
                "Security > {p} (to the terminal running blurt).".format(
                    f=feature, p=pane
                )
            )
            self._last_reason = reason
            return False, reason

        if status == not_determined:
            # First use: this is what raises the system permission dialog.
            store = self._get_store(eventkit)
            if store is not None:
                self._request_one(eventkit, store, entity_type, _ACCESS_TIMEOUT_S)
            status = self._status(eventkit, entity_type)

        if self._status_is_granted(eventkit, status):
            return True, ""

        reason = (
            "{f} access wasn't granted. Approve it for the terminal running blurt "
            "in System Settings > Privacy & Security > {p}.".format(f=feature, p=pane)
        )
        self._last_reason = reason
        return False, reason

    def _to_nsdate(self, foundation: Any, dt: datetime.datetime) -> Any:
        """Convert a Python datetime to an ``NSDate``.

        Uses ``dateWithTimeIntervalSince1970_`` on the POSIX timestamp.
        ``datetime.timestamp()`` interprets a naive datetime in the machine's local
        zone, so "noon tomorrow" with no tzinfo lands at local noon -- which is what
        the speaker meant.
        """
        return foundation.NSDate.dateWithTimeIntervalSince1970_(dt.timestamp())

    def _due_components(
        self, foundation: Any, dt: datetime.datetime
    ) -> Optional[Any]:
        """Build ``NSDateComponents`` for a reminder due date from ``dt``'s fields.

        Set straight from the datetime's own year/month/day/hour/minute rather than
        by converting through NSDate, so the reminder is due at the local wall-clock
        time the user spoke, with no timezone round-trip to second-guess. Returns
        ``None`` on failure (the reminder is still created, just without a due date).
        """
        try:
            components = foundation.NSDateComponents.alloc().init()
            components.setYear_(dt.year)
            components.setMonth_(dt.month)
            components.setDay_(dt.day)
            components.setHour_(dt.hour)
            components.setMinute_(dt.minute)
            components.setSecond_(dt.second)
            return components
        except Exception:  # pragma: no cover - defensive
            _log.debug("could not build due-date components", exc_info=True)
            return None

    def _error_text(self, error: Any) -> Optional[str]:
        """Pull a human-readable string out of an NSError, if there is one."""
        if error is None:
            return None
        try:
            return str(error.localizedDescription())
        except Exception:  # pragma: no cover - defensive
            return str(error)
