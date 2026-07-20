"""Route a spoken phrase to the best intent handler and dispatch its action.

:class:`IntentRouter` is the glue between :func:`blurt.assistant.nldate.parse_when`
/ the handlers in :mod:`.intents` and the rest of the app. It does two things:

  * :meth:`route` asks every handler to :meth:`~blurt.assistant.types.IntentHandler.match`
    the text, collects the non-None :class:`~blurt.assistant.types.Action` results,
    and returns the highest-confidence one. Ties are broken by handler order (the
    order handlers were passed to the constructor -- the earlier handler wins). If
    nothing matches, it returns a fallback ``Action(kind="dictate", summary=text,
    confidence=0.0)`` so the caller simply types out what was said.

  * :meth:`execute` runs a chosen Action by handing it back to the handler that
    produced it (looked up via ``payload["_handler"]``, with a kind->handler map
    as a backstop), or, for the dictate fallback, by calling ``dictate_fallback``.

Design notes:

  * PURE LOGIC. Only the standard library and the pure :mod:`.types` module are
    imported here -- no macOS / pyobjc. Every side effect is reached through a
    handler or the injected ``dictate_fallback``, so the router is unit-testable
    anywhere with fake handlers.
  * NEVER RAISE ON BAD SPEECH OR A MISBEHAVING HANDLER. A handler whose match()
    or execute() throws is treated as "no match" / a clean failure result rather
    than being allowed to crash the dictation worker. The router's own message is
    always human-readable.
  * NO SIDE EFFECTS IN route(). Matching is pure inspection; only :meth:`execute`
    performs the real action, and only when the caller asks for it.

What can go wrong: a handler could return something that is not an Action, or an
Action whose ``payload["_handler"]`` names no known handler. Both are handled
defensively -- a non-Action match is ignored, and an unowned Action falls back to
dictation rather than raising.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .types import Action, ActionResult, IntentHandler

# Backstop mapping from an Action's kind to the owning handler's ``name``, used
# only when ``payload["_handler"]`` is missing or unrecognized.
_KIND_TO_NAME = {
    "calendar_event": "calendar",
    "reminder": "reminder",
    "timer": "timer",
    "open_app": "open_app",
}


class IntentRouter:
    """Pick the best-matching handler for a phrase and dispatch its Action."""

    def __init__(
        self,
        handlers: List[IntentHandler],
        dictate_fallback: Callable[[str], ActionResult],
    ) -> None:
        self._handlers = list(handlers) if handlers else []
        self._dictate_fallback = dictate_fallback
        # name -> handler, for dispatch in execute().
        self._by_name = {}  # type: Dict[str, IntentHandler]
        for handler in self._handlers:
            name = getattr(handler, "name", None)
            if name:
                self._by_name[name] = handler

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #
    def route(self, text: str) -> Action:
        """Return the best Action for ``text``, or a dictate fallback.

        Tries every handler in order, keeps the highest confidence, and breaks
        ties in favor of the earlier handler. Never raises: a handler that
        throws is skipped. Empty / non-string input routes straight to dictation.
        """
        if not isinstance(text, str) or not text.strip():
            return self._dictate_action(text if isinstance(text, str) else "")

        best = None  # type: Optional[Action]
        for handler in self._handlers:
            try:
                action = handler.match(text)
            except Exception:
                # A misbehaving handler must not break routing.
                action = None
            if action is None or not isinstance(action, Action):
                continue
            # Strictly-greater keeps the FIRST handler on a tie (handler order).
            if best is None or action.confidence > best.confidence:
                best = action

        if best is None:
            return self._dictate_action(text)
        return best

    @staticmethod
    def _dictate_action(text: str) -> Action:
        """The no-match fallback: dictate the raw text verbatim."""
        return Action(kind="dictate", summary=text, confidence=0.0)

    # ------------------------------------------------------------------ #
    # Dispatch
    # ------------------------------------------------------------------ #
    def execute(self, action: Action) -> ActionResult:
        """Run ``action`` via its owning handler (or the dictate fallback).

        Never raises: a handler whose execute() throws, or one that returns a
        non-result, is turned into a plain-language ``ActionResult(ok=False,...)``
        so the app always has something to announce.
        """
        if action is None or not isinstance(action, Action):
            return ActionResult(ok=False, message="There was nothing to do.")

        if action.kind == "dictate":
            return self._run_dictate(action.summary)

        handler = self._owner(action)
        if handler is None:
            # Unknown owner -> safest thing is to just dictate the text.
            return self._run_dictate(action.summary)

        try:
            result = handler.execute(action)
        except Exception:
            return ActionResult(ok=False, message="Sorry, I couldn't complete that.")
        if not isinstance(result, ActionResult):
            return ActionResult(
                ok=False, message="Sorry, that action didn't report a result."
            )
        return result

    def _owner(self, action: Action) -> Optional[IntentHandler]:
        """Find the handler that produced ``action``, or None."""
        name = None
        try:
            name = action.payload.get("_handler")
        except Exception:
            name = None
        if name and name in self._by_name:
            return self._by_name[name]
        # Backstop: map the kind to a handler name.
        fallback_name = _KIND_TO_NAME.get(action.kind)
        if fallback_name and fallback_name in self._by_name:
            return self._by_name[fallback_name]
        return None

    def _run_dictate(self, text: str) -> ActionResult:
        """Call the injected dictate fallback, defending against misbehavior."""
        try:
            result = self._dictate_fallback(text if isinstance(text, str) else "")
        except Exception:
            return ActionResult(ok=False, message="I couldn't dictate that.")
        if not isinstance(result, ActionResult):
            return ActionResult(ok=False, message="Dictation didn't report a result.")
        return result


__all__ = ["IntentRouter"]
