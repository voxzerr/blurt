"""Unit tests for blurt's IntentRouter (routing + execute dispatch).

Like test_intents, these run with ZERO macOS access: the router is wired up
with the four real handlers, but every handler is backed by an in-memory FAKE
(records the call, returns a canned ``ActionResult``) and the dictate fallback
is a recording stub. Nothing here opens Calendar, arms a threading.Timer, or
shells out to ``open``.

What these tests pin down:
  * a scheduling phrase routes to the calendar handler (it wins on confidence);
  * plain narration with no command routes to ``kind == "dictate"`` and its
    execute() calls the injected fallback with the raw text;
  * a timer command routes to the timer handler, not the calendar handler;
  * execute() dispatches each Action kind back to the one handler that owns it
    (and only that one), and the dictate fallback for the dictation kind.

Determinism: the same fixed ``now_fn`` (2026-07-20 10:30, a Monday) is injected
into every time-aware handler, so relative dates are stable.

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
from blurt.assistant.router import IntentRouter
from blurt.assistant.types import ActionResult

now_fn = lambda: datetime.datetime(2026, 7, 20, 10, 30)  # noqa: E731


# --------------------------------------------------------------------------- #
# Fakes (same shapes as test_intents; duplicated so each test file stands alone).
# --------------------------------------------------------------------------- #
class FakeCalendar:
    def __init__(self) -> None:
        self.events = []
        self.reminders = []

    def create_event(self, title, start, end, notes=None):
        self.events.append(
            {"title": title, "start": start, "end": end, "notes": notes}
        )
        return ActionResult(ok=True, message="Added '{0}' to your calendar".format(title))

    def create_reminder(self, title, due=None):
        self.reminders.append({"title": title, "due": due})
        return ActionResult(ok=True, message="Added reminder '{0}'".format(title))


class FakeTimer:
    def __init__(self) -> None:
        self.calls = []

    def schedule(self, minutes, label):
        self.calls.append((minutes, label))
        return ActionResult(ok=True, message="Timer set for {0} minutes".format(minutes))


class FakeOpener:
    def __init__(self) -> None:
        self.opened = []

    def __call__(self, name):
        self.opened.append(name)
        return ActionResult(ok=True, message="Opened {0}".format(name))


class FakeDictate:
    """Recording stand-in for the dictate fallback: records the text it typed."""

    def __init__(self) -> None:
        self.calls = []  # list of dictated strings

    def __call__(self, text):
        self.calls.append(text)
        return ActionResult(ok=True, message="Dictated: {0}".format(text))


def build_router():
    """Fresh router + its four fakes + the dictate stub, all isolated per call."""
    cal = FakeCalendar()
    timer_service = FakeTimer()
    opener = FakeOpener()
    dictate = FakeDictate()
    handlers = [
        CalendarHandler(cal, now_fn=now_fn),
        ReminderHandler(cal, now_fn=now_fn),
        TimerHandler(timer_service, now_fn=now_fn),
        OpenAppHandler(opener),
    ]
    router = IntentRouter(handlers, dictate)
    return router, cal, timer_service, opener, dictate


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def test_schedule_routes_to_calendar():
    router, _cal, _timer, _opener, _dictate = build_router()

    action = router.route("schedule lunch tomorrow at noon")

    assert action.kind == "calendar_event"
    # It wins on confidence (a verb + explicit clock time scores highest).
    assert action.confidence == pytest.approx(0.9)


def test_plain_text_routes_to_dictate_and_execute_calls_fallback():
    router, _cal, _timer, _opener, dictate = build_router()
    text = "just some regular text I'm dictating"

    action = router.route(text)
    assert action.kind == "dictate"
    assert action.confidence == 0.0

    result = router.execute(action)
    assert result.ok is True
    # execute() on a dictate action must call the injected fallback with the
    # raw text, and nothing else.
    assert dictate.calls == [text]


def test_timer_command_routes_to_timer_not_calendar():
    router, cal, _timer, _opener, _dictate = build_router()

    action = router.route("set a timer for 10 minutes")

    assert action.kind == "timer"
    assert action.kind != "calendar_event"
    # Pure routing has no side effects: nothing written until execute() runs.
    assert cal.events == []


# --------------------------------------------------------------------------- #
# Execute dispatch: each kind reaches exactly the one handler that owns it.
# --------------------------------------------------------------------------- #
def test_execute_calendar_dispatches_only_to_calendar_backend():
    router, cal, timer_service, opener, dictate = build_router()

    router.execute(router.route("schedule lunch with Sam tomorrow at noon"))

    assert len(cal.events) == 1
    assert cal.reminders == []
    assert timer_service.calls == []
    assert opener.opened == []
    assert dictate.calls == []


def test_execute_reminder_dispatches_only_to_reminder_backend():
    router, cal, timer_service, opener, dictate = build_router()

    router.execute(router.route("remind me to call mom"))

    assert len(cal.reminders) == 1
    assert cal.events == []
    assert timer_service.calls == []
    assert opener.opened == []
    assert dictate.calls == []


def test_execute_timer_dispatches_only_to_timer_service():
    router, cal, timer_service, opener, dictate = build_router()

    router.execute(router.route("set a timer for 5 minutes"))

    assert len(timer_service.calls) == 1
    minutes, _label = timer_service.calls[0]
    assert minutes == 5
    assert cal.events == []
    assert cal.reminders == []
    assert opener.opened == []
    assert dictate.calls == []


def test_execute_open_app_dispatches_only_to_opener():
    router, cal, timer_service, opener, dictate = build_router()

    router.execute(router.route("open Safari"))

    assert opener.opened == ["Safari"]
    assert cal.events == []
    assert cal.reminders == []
    assert timer_service.calls == []
    assert dictate.calls == []


def test_execute_dictate_dispatches_only_to_fallback():
    router, cal, timer_service, opener, dictate = build_router()
    text = "just narrating some thoughts here"

    router.execute(router.route(text))

    assert dictate.calls == [text]
    assert cal.events == []
    assert cal.reminders == []
    assert timer_service.calls == []
    assert opener.opened == []
