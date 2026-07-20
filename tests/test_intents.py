"""Unit tests for the four blurt intent handlers (pure matching + execute).

These tests run with ZERO macOS access. They never construct the real
EventKit / AppKit backends; instead each handler is handed a tiny in-memory
FAKE that only records the call it received and returns a canned
``ActionResult``. That keeps the suite runnable unattended on the Intel floor
machine under the Apple system Python (3.9.6) -- no Calendar/Reminders
permission prompt, no LaunchServices, no threads left running.

Determinism: every handler that reads a clock is given the same injected
``now_fn`` (fixed at 2026-07-20 10:30, a Monday), so relative phrases like
"tomorrow" and "friday" resolve to fixed calendar dates and the assertions can
compare exact datetimes.

What these tests pin down:
  * the KEY regression -- a loose shopping thought ("add milk") must NOT become
    a calendar event (weak verb + no time -> no match, falls through to
    dictation);
  * correct title extraction and time resolution for the happy paths;
  * that execute() drives the injected backend exactly once with the parsed
    payload;
  * the handler boundaries (timer ignores "remind me...", open-app declines
    "open the door").

Python 3.9 floor: lazy annotations, stdlib + pytest only, no PEP 604/585 syntax.
"""

from __future__ import annotations

import datetime

import pytest

from blurt.assistant.intents import (
    CalendarHandler,
    OpenAppHandler,
    ReminderHandler,
    TimerHandler,
)
from blurt.assistant.types import ActionResult

# Fixed clock injected into every time-aware handler. 2026-07-20 is a Monday, so
# "tomorrow" -> Tue 2026-07-21 and "friday" -> Fri 2026-07-24. Task-mandated shape.
now_fn = lambda: datetime.datetime(2026, 7, 20, 10, 30)  # noqa: E731


# --------------------------------------------------------------------------- #
# Fakes -- record the call, return a canned ok result, touch nothing on macOS.
# --------------------------------------------------------------------------- #
class FakeCalendar:
    """Stand-in for CalendarBackend: records create_event / create_reminder.

    Signatures mirror the real backend exactly (see calendar_backend.py) so the
    handlers call it the same way they call the EventKit-backed one:
    ``create_event(title, start, end, notes=None)`` and
    ``create_reminder(title, due=None)``.
    """

    def __init__(self) -> None:
        self.events = []      # list of dicts: title/start/end/notes
        self.reminders = []   # list of dicts: title/due

    def create_event(self, title, start, end, notes=None):
        self.events.append(
            {"title": title, "start": start, "end": end, "notes": notes}
        )
        return ActionResult(ok=True, message="Added '{0}' to your calendar".format(title))

    def create_reminder(self, title, due=None):
        self.reminders.append({"title": title, "due": due})
        return ActionResult(ok=True, message="Added reminder '{0}'".format(title))


class FakeTimer:
    """Stand-in for TimerService: records schedule(minutes, label), arms nothing."""

    def __init__(self) -> None:
        self.calls = []  # list of (minutes, label)

    def schedule(self, minutes, label):
        self.calls.append((minutes, label))
        return ActionResult(ok=True, message="Timer set for {0} minutes".format(minutes))


class FakeOpener:
    """Callable stand-in for system_actions.open_app: records names, launches nothing.

    OpenAppHandler expects a ``Callable[[str], ActionResult]``, so this is a
    callable object -- ``fake_open(name)`` records the name and returns
    ``ActionResult(ok=True, "Opened <name>")``, exactly like the real opener does
    on success, but without shelling out to ``open -a``.
    """

    def __init__(self) -> None:
        self.opened = []  # list of names passed to the opener

    def __call__(self, name):
        self.opened.append(name)
        return ActionResult(ok=True, message="Opened {0}".format(name))


@pytest.fixture
def cal():
    return FakeCalendar()


@pytest.fixture
def timer_service():
    return FakeTimer()


@pytest.fixture
def opener():
    return FakeOpener()


# --------------------------------------------------------------------------- #
# CalendarHandler
# --------------------------------------------------------------------------- #
def test_calendar_schedule_lunch_matches_and_executes(cal):
    handler = CalendarHandler(cal, now_fn=now_fn)

    action = handler.match("schedule lunch with Sam tomorrow at noon")

    assert action is not None
    assert action.kind == "calendar_event"
    # Casing is cosmetic (the handler title-cases the first char); accept either
    # "lunch with Sam" or "Lunch with Sam". The proper noun "Sam" must survive.
    assert action.payload["title"].lower() == "lunch with sam"
    assert "Sam" in action.payload["title"]
    assert action.payload["start"] == datetime.datetime(2026, 7, 21, 12, 0)
    assert action.payload["all_day"] is False
    assert action.confidence == pytest.approx(0.9)

    # execute() must drive the backend exactly once with the parsed payload.
    result = handler.execute(action)
    assert result.ok is True
    assert len(cal.events) == 1
    written = cal.events[0]
    assert written["title"] == action.payload["title"]
    assert written["start"] == datetime.datetime(2026, 7, 21, 12, 0)
    assert written["end"] == datetime.datetime(2026, 7, 21, 13, 0)


def test_calendar_add_milk_does_not_match(cal):
    """KEY REGRESSION: a shopping thought must not become a calendar event.

    "add" is a weak/ambiguous verb and there is no time in the phrase, so the
    handler must decline (return None) and let the phrase fall through to plain
    dictation. If this ever starts matching, "add milk" would silently create a
    calendar event -- exactly the bug this test guards against.
    """
    handler = CalendarHandler(cal, now_fn=now_fn)

    assert handler.match("add milk") is None
    # And nothing was written as a side effect of matching.
    assert cal.events == []


def test_calendar_put_dentist_on_calendar_friday(cal):
    handler = CalendarHandler(cal, now_fn=now_fn)

    action = handler.match("put dentist on my calendar friday at 3")

    assert action is not None
    assert action.kind == "calendar_event"
    assert action.payload["title"].lower() == "dentist"
    # 2026-07-20 is a Monday, so the coming Friday is 2026-07-24; "at 3" -> 3 PM.
    assert action.payload["start"] == datetime.datetime(2026, 7, 24, 15, 0)
    assert action.payload["all_day"] is False


# --------------------------------------------------------------------------- #
# ReminderHandler
# --------------------------------------------------------------------------- #
def test_reminder_call_mom_matches_without_due(cal):
    handler = ReminderHandler(cal, now_fn=now_fn)

    action = handler.match("remind me to call mom")

    assert action is not None
    assert action.kind == "reminder"
    assert action.payload["title"].lower() == "call mom"
    assert action.payload["due"] is None

    result = handler.execute(action)
    assert result.ok is True
    assert len(cal.reminders) == 1
    assert cal.reminders[0]["title"] == action.payload["title"]
    assert cal.reminders[0]["due"] is None


def test_reminder_submit_report_matches_with_due(cal):
    handler = ReminderHandler(cal, now_fn=now_fn)

    action = handler.match("remind me to submit the report tomorrow at 9")

    assert action is not None
    assert action.kind == "reminder"
    # The middle article "the" is preserved; only edge filler is trimmed.
    assert action.payload["title"].lower() == "submit the report"
    assert action.payload["due"] == datetime.datetime(2026, 7, 21, 9, 0)
    assert action.confidence == pytest.approx(0.9)

    result = handler.execute(action)
    assert result.ok is True
    assert cal.reminders == [
        {"title": action.payload["title"], "due": datetime.datetime(2026, 7, 21, 9, 0)}
    ]


# --------------------------------------------------------------------------- #
# TimerHandler
# --------------------------------------------------------------------------- #
def test_timer_five_minutes_matches_and_executes(timer_service):
    handler = TimerHandler(timer_service, now_fn=now_fn)

    action = handler.match("set a timer for 5 minutes")

    assert action is not None
    assert action.kind == "timer"
    assert action.payload["minutes"] == 5
    assert action.confidence == pytest.approx(0.9)

    result = handler.execute(action)
    assert result.ok is True
    assert len(timer_service.calls) == 1
    minutes, _label = timer_service.calls[0]
    assert minutes == 5


def test_timer_half_an_hour_is_thirty_minutes(timer_service):
    handler = TimerHandler(timer_service, now_fn=now_fn)

    action = handler.match("timer for half an hour")

    assert action is not None
    assert action.kind == "timer"
    assert action.payload["minutes"] == 30


def test_timer_ignores_remind_me(timer_service):
    """A "remind me..." phrase belongs to ReminderHandler, not TimerHandler."""
    handler = TimerHandler(timer_service, now_fn=now_fn)

    assert handler.match("remind me to call mom") is None
    assert timer_service.calls == []


# --------------------------------------------------------------------------- #
# OpenAppHandler
# --------------------------------------------------------------------------- #
def test_open_safari_matches_and_calls_opener(opener):
    handler = OpenAppHandler(opener)

    action = handler.match("open Safari")

    assert action is not None
    assert action.kind == "open_app"
    assert action.payload["name"] == "Safari"

    result = handler.execute(action)
    assert result.ok is True
    assert result.message == "Opened Safari"
    assert opener.opened == ["Safari"]


def test_open_launch_the_notes_app(opener):
    handler = OpenAppHandler(opener)

    action = handler.match("launch the notes app")

    assert action is not None
    assert action.kind == "open_app"
    # The human-facing summary title-cases the recognized app to "Notes".
    assert action.summary == "Open Notes"
    # The payload keeps the RAW spoken name ("notes"); the real open_app does a
    # case-insensitive lookup, so the lowercase name still launches Notes.app.
    # We assert case-insensitively rather than pinning the exact casing.
    assert action.payload["name"].lower() == "notes"

    handler.execute(action)
    assert len(opener.opened) == 1
    assert opener.opened[0].lower() == "notes"


def test_open_the_door_is_not_an_app(opener):
    """"open the door" must NOT fire -- "door" is a physical object, not an app.

    The handler tells app from non-app via a small denylist of concrete objects
    (door, window, ticket, ...). That heuristic is intentionally simple: a novel
    non-app noun that is NOT on the list ("open the pod bay") would still be
    attempted, and the real opener would harmlessly report it found no such app.
    Here "door" IS on the list, so the handler declines and the phrase dictates.
    """
    handler = OpenAppHandler(opener)

    assert handler.match("open the door") is None
    assert opener.opened == []


# --------------------------------------------------------------------------- #
# Cross-handler regression: a loose thought matches nothing at all.
# --------------------------------------------------------------------------- #
def test_add_milk_matches_no_handler(cal, timer_service, opener):
    """Stronger form of the key regression: "add milk" is inert across ALL four
    handlers, so a full router can only send it to dictation."""
    handlers = [
        CalendarHandler(cal, now_fn=now_fn),
        ReminderHandler(cal, now_fn=now_fn),
        TimerHandler(timer_service, now_fn=now_fn),
        OpenAppHandler(opener),
    ]
    for handler in handlers:
        assert handler.match("add milk") is None
    assert cal.events == []
    assert cal.reminders == []
    assert timer_service.calls == []
    assert opener.opened == []
