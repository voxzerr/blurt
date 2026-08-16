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

from typing import Callable, List, Optional

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
    revert_fn: "Optional[Callable[[], bool]]" = None,
) -> IntentRouter:
    """Wire the real local backends into a router ready for the app to use.

    ``dictate_fallback`` is what runs when no command matches -- the app passes
    its normal "paste this text" path here, so an unrecognised phrase is simply
    dictated instead of lost.

    ``now_fn`` returns the current time; defaults to ``datetime.datetime.now``.
    It is injectable so tests can pin the clock (the parsers never call a clock
    themselves).

    ``revert_fn`` is :meth:`blurt.app.BlurtApp.revert_last` -- re-insert the raw
    text of the last dictation. It is OPTIONAL and defaults to None, in which
    case no :class:`~blurt.assistant.intents.RevertHandler` is registered at all
    and the router behaves exactly as it did before revert existed. That matters
    for callers with no dictation history to revert to (tests, tools that only
    want the calendar/timer intents): registering a handler whose backend is a
    stub would put a live "undo that" in front of a function that cannot do it.

    Backends are imported lazily, inside this function, so merely importing the
    package never touches EventKit/AppKit and stays safe on any platform and in
    tests.

    HANDLER ORDER IS THE TIE-BREAK RULE. :meth:`IntentRouter.route` keeps the
    highest confidence and, on a tie, the handler listed first. Two orderings
    here are load-bearing:

      * Revert goes FIRST when it is registered. A bare "undo" is one word with
        no object, and a future handler that gets clever about short imperatives
        must not be able to claim it -- the undo is the one command whose failure
        the user cannot work around, because the raw text it restores exists
        nowhere else they can reach.
      * Reminder before timer, so "remind me in five minutes" is a reminder
        rather than a timer.
    """
    import datetime as _dt

    from .calendar_backend import CalendarBackend
    from .intents import (
        CalendarHandler,
        OpenAppHandler,
        ReminderHandler,
        RevertHandler,
        TimerHandler,
    )
    from .system_actions import TimerService, open_app

    if now_fn is None:
        now_fn = _dt.datetime.now

    calendar = CalendarBackend()
    timer = TimerService()

    # Annotated, not a ``# type:`` comment (the older style still used in
    # router.py): pyflakes no longer parses type comments, so a name referenced
    # only from one looks like an unused import and gets tidied away by the next
    # person to run a linter -- taking the annotation's meaning with it. The
    # annotation is never evaluated at runtime (``from __future__ import
    # annotations``), so this stays a plain list on the 3.9 floor.
    handlers: List[IntentHandler] = []
    if revert_fn is not None:
        handlers.append(RevertHandler(revert_fn))
    handlers.extend(
        [
            CalendarHandler(calendar, now_fn),
            ReminderHandler(calendar, now_fn),
            TimerHandler(timer, now_fn),
            OpenAppHandler(open_app),
        ]
    )
    return IntentRouter(handlers, dictate_fallback)
