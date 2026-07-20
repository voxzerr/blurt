"""LLM question-answering backend for blurt's assistant -- deliberately OFF in v1.

blurt's promise is that nothing you say leaves your machine: dictation, calendar
writes, timers and app-opening all run locally. Free-form question answering
("what's the capital of Peru?", "summarize this") is the one feature that cannot
be done well with the small on-device machinery blurt ships today, so in v1 this
backend is a stub. Its whole job is to:

  1. Document the interface a real question-answering backend will implement
     (:class:`LLMBackend`), and
  2. Answer :meth:`~LLMBackend.available` with ``False`` so the router cleanly
     falls through to dictation instead of pretending it can answer.

Two real backends are meant to slot in here later. Neither is implemented yet;
this module only sketches the shape so the wiring is obvious when the time comes:

  * ``CloudLLM`` -- calls the Anthropic Claude API. Requires an API key and sends
    the user's text off-device for processing. This is the ONLY path in all of
    blurt that breaks the "nothing leaves your machine" promise, so it must be
    strictly opt-in: enabled by an explicit config flag, never the default, and
    surfaced with a visible indicator whenever it is active so the user always
    knows a question was sent to the cloud. Scope it narrowly (just the question
    text, nothing ambient) and announce it every time.

  * ``LocalLLM`` -- runs a small language model entirely on-device. Practical on
    Apple Silicon (M-series has the memory bandwidth and Metal acceleration to
    make a few-billion-parameter model usable); too slow to be worth it on the
    2-core 2017 Intel floor machine, so on that hardware it would stay disabled
    and answering would remain off. Keeps the privacy promise intact because no
    text ever leaves the machine.

What can go wrong: essentially nothing in v1. This module is pure standard
library with no macOS / pyobjc imports, so it is import-clean and unit-testable
on any platform. :class:`DisabledLLM` performs no I/O and cannot raise; it just
returns a polite "not enabled yet" result. The real risk lives in the future
backends -- an unguarded ``CloudLLM`` that leaks text off-device -- which is
exactly why the opt-in requirement is documented here before either exists.
"""
from __future__ import annotations

import abc
from typing import Any

from .types import ActionResult


class LLMBackend(abc.ABC):
    """Interface a question-answering backend must implement.

    A backend takes a natural-language question and returns an
    :class:`~blurt.assistant.types.ActionResult` whose ``message`` is the answer
    (or an explanation of why it could not answer). The router only calls
    :meth:`answer` when :meth:`available` is ``True``, so an unavailable backend
    is free to be a no-op.

    ``name`` is a short, stable identifier used for logging and for the visible
    "which backend answered this" indicator -- important because a cloud backend
    sending text off-device must always be distinguishable from a local one.
    """

    name: str = "llm"

    @abc.abstractmethod
    def available(self) -> bool:
        """Return True only if this backend can actually answer questions now.

        Implementations should check everything a call would need (model loaded,
        API key present, user opt-in flag set, hardware sufficient) and return
        False rather than raising when anything is missing. The router uses this
        to decide whether to route a question here or fall through to dictation.
        """
        ...

    @abc.abstractmethod
    def answer(self, question: str) -> ActionResult:
        """Answer ``question`` and return the result for the app to announce.

        Must never raise on ordinary failure (no network, model busy, empty
        question): return ``ActionResult(ok=False, message=...)`` explaining the
        problem in plain language instead. Should only be called when
        :meth:`available` is True.
        """
        ...


class DisabledLLM(LLMBackend):
    """The v1 backend: question answering is switched off.

    :meth:`available` always returns ``False`` so the router never routes a
    question here, and :meth:`answer` returns a plain-language explanation of why
    the feature is off and what enabling it would cost -- either a local model or
    the off-device Claude API. Performs no I/O and cannot raise.
    """

    name = "disabled"

    def available(self) -> bool:
        return False

    def answer(self, question: str) -> ActionResult:
        return ActionResult(
            ok=False,
            message=(
                "Question answering isn't enabled yet. It will need either a "
                "local model (practical on Apple Silicon) or the Claude API "
                "(which would send your question off-device -- off by default)."
            ),
        )


def make_llm(config: Any) -> LLMBackend:
    """Return the LLM backend to use, given the app's config.

    In v1 this always returns :class:`DisabledLLM`, so question answering is off
    no matter what ``config`` says. ``config`` is accepted (and ignored) now so
    that callers already pass it and the signature is stable when the real
    backends land.

    TODO(v2): pick a real backend from ``config`` behind an explicit flag, e.g.::

        backend = getattr(config, "llm_backend", "off")
        if backend == "cloud" and getattr(config, "anthropic_api_key", None):
            # Off-device: sends question text to the Claude API. Must be opt-in
            # and shown with a visible indicator -- see the module docstring.
            return CloudLLM(api_key=config.anthropic_api_key,
                            model=getattr(config, "llm_model", None))
        if backend == "local":
            # On-device: practical on Apple Silicon, too slow on 2-core Intel.
            return LocalLLM(model_path=getattr(config, "llm_model_path", None))

    Until CloudLLM / LocalLLM exist, keep returning the disabled stub.
    """
    return DisabledLLM()


__all__ = ["LLMBackend", "DisabledLLM", "make_llm"]
