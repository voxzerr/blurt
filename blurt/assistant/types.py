"""Shared vocabulary for the blurt voice-assistant subpackage.

This module defines the small, immutable data types that every other assistant
module speaks in: :class:`Action` (a parsed intent), :class:`ActionResult` (what
happened when an action ran), and :class:`IntentHandler` (the abstract base every
handler implements). It is deliberately pure -- only the standard library, no
macOS / pyobjc / EventKit imports -- so it stays import-clean on any platform and
can be unit-tested anywhere (Intel or Apple Silicon, any macOS).

What can go wrong: very little. These are frozen dataclasses and one ABC, so the
main failure mode would be constructing an ``Action`` with a ``kind`` that no
handler recognizes -- routing simply treats that as "no match" rather than raising.
Keep this module dependency-free; anything that needs macOS APIs belongs elsewhere.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Action:
    kind: str                       # "calendar_event"|"reminder"|"timer"|"open_app"|"dictate"|"none"
    summary: str                    # human-readable intent, e.g. "Add 'Lunch with Sam' tomorrow 12:00 PM"
    payload: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0         # 0..1
    needs_confirmation: bool = False


@dataclass(frozen=True)
class ActionResult:
    ok: bool
    message: str                    # what happened, for notification/console


class IntentHandler(abc.ABC):
    name: str

    @abc.abstractmethod
    def match(self, text: str) -> Optional[Action]:
        """Return an Action if this handler applies to ``text``, else None."""
        ...

    @abc.abstractmethod
    def execute(self, action: Action) -> ActionResult:
        """Perform ``action`` and return a human-readable result."""
        ...
