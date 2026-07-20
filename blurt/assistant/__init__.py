"""blurt voice-assistant subpackage: parse spoken text into actions and run them.

The public surface is intentionally tiny. :class:`Action` and :class:`ActionResult`
are the immutable data types passed between components; :class:`IntentHandler` is the
base class each intent (calendar, reminder, timer, open-app) implements; and
:class:`IntentRouter` picks the best-matching handler for a phrase and dispatches it.

Only pure-logic names are re-exported here. Backends that touch macOS APIs
(EventKit, AppKit, osascript) live in their own modules and import those APIs lazily,
so importing this package stays safe on any platform.

What can go wrong: importing the package pulls in :mod:`.router`, which in turn
imports :mod:`.types`. Both are pure Python with no macOS-only dependencies, so this
should import cleanly everywhere. If :mod:`.router` is missing, the package will fail
to import -- that is by design, since the router is core to the public interface.
"""
from __future__ import annotations

from typing import Callable, Optional

from .router import IntentRouter
from .types import Action, ActionResult, IntentHandler

__all__ = [
    "Action",
    "ActionResult",
    "IntentHandler",
    "IntentRouter",
    "build_default_router",
]


def build_default_router(
    dictate_fallback: "Callable[[str], ActionResult]",
    now_fn: "Optional[Callable[[], object]]" = None,
) -> IntentRouter:
    """Wire the real local backends into a router ready for the app to use.

    ``dictate_fallback`` is what runs when no command matches -- the app passes
    its normal "paste this text" path here, so an unrecognised phrase is simply
    dictated instead of lost.

    ``now_fn`` returns the current time; defaults to ``datetime.datetime.now``.
    It is injectable so tests can pin the clock (the parsers never call a clock
    themselves).

    Backends are imported lazily, inside this function, so merely importing the
    package never touches EventKit/AppKit and stays safe on any platform and in
    tests. The handler order sets tie-breaking: reminder before timer so
    "remind me..." is a reminder, not a timer.
    """
    import datetime as _dt

    from .calendar_backend import CalendarBackend
    from .intents import (
        CalendarHandler,
        OpenAppHandler,
        ReminderHandler,
        TimerHandler,
    )
    from .system_actions import TimerService, open_app

    if now_fn is None:
        now_fn = _dt.datetime.now

    calendar = CalendarBackend()
    timer = TimerService()

    handlers = [
        CalendarHandler(calendar, now_fn),
        ReminderHandler(calendar, now_fn),
        TimerHandler(timer, now_fn),
        OpenAppHandler(open_app),
    ]
    return IntentRouter(handlers, dictate_fallback)
