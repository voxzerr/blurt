"""Unit tests for RevertHandler -- the "undo that" escape hatch.

Cleanup runs by default, so something has to be able to take it back. Saying
"undo that" re-inserts the RAW transcript of the last dictation; that is the
promise which makes cleanup-by-default defensible at all. These tests run with
ZERO hardware: no microphone, no model, no network, no Calendar prompt. The
revert backend is a recording callable that returns whatever the test tells it
to, and the dictate fallback is a recording stub, so nothing here pastes text
anywhere.

RevertHandler is the one handler in blurt where the asymmetry runs BACKWARDS.
Everywhere else, declining to act costs the user a second and a repeat. Here,
acting wrongly pastes a stale raw transcript into whatever buffer the user
happened to be typing in -- an email, a commit message, someone else's
document -- and blurt cannot take it back, because :mod:`blurt.inject` can paste
but not delete. So the groups below are weighted accordingly:

  * THE FALSE-POSITIVE DIRECTION (the expensive one). Ordinary dictation that
    merely contains "undo", "revert" or "scratch" must return None from match().
    These are the tests that matter most; if they ever go green-to-red, someone
    has widened the funnel and a sentence about a git commit will start pasting
    old text. Two sentences in that group even OPEN with a revert verb -- the
    only thing separating them from a real command is that they keep going.
  * EVERY NEGATIVE NAMES THE GATE THAT REJECTS IT. A ``match() is None`` proves
    the utterance was refused; it says nothing about WHICH of the handler's two
    gates -- the whole-utterance anchor or the word ceiling -- did the refusing.
    That gap is how a test goes vacuous: it credits one gate for a rejection the
    other gate was already making, keeps passing after the credited gate is
    deleted, and advertises a defence that is not there. So the negatives here
    are sorted by the gate they exercise and each group is pinned from both
    sides -- anchor cases must stay rejected with the ceiling lifted out of the
    way, ceiling cases must START matching once it is lifted. See the "THE TWO
    GATES, KEPT HONEST" banner below.
  * THE POSITIVE DIRECTION. Every phrase the handler is meant to accept -- with
    a trailing "please", trailing punctuation, and whatever casing Whisper felt
    like -- yields an Action of kind "revert" at the module's top confidence.
    A revert that does not fire is a bad day; a revert that fires wrongly is a
    lost paragraph. Both directions are pinned, in that order of importance.
  * MATCHING IS INERT. route()/match() must never actually revert. Recognition
    and execution are separate steps and only the caller decides to cross that
    line.
  * EXECUTE NEVER RAISES. execute() runs on the dictation worker thread. An
    exception escaping it kills the loop and every subsequent dictation is lost
    silently, which is exactly the unrecoverable failure this whole module is
    organised against. A revert_fn that raises must come back as a clean
    ok=False, forever.
  * DISPATCH WIRING. The Action produced by match() has to find its way back to
    the handler that made it, both via ``payload["_handler"]`` and, with that key
    stripped, via the router's kind->name backstop. A revert Action that cannot
    find its owner would be dictated as its own summary text -- the failure mode
    is pasting the words "Revert to the raw transcript" into the user's buffer.
  * OPT-IN, AND FIRST. ``build_default_router`` without a ``revert_fn`` must not
    register the handler at all (an existing caller keeps the exact behaviour it
    had before revert existed, and "undo that" dictates). With one, the handler
    must be registered FIRST, because the router breaks confidence ties by
    handler order and a bare "undo" is the one command the user has no way to
    work around.

Python 3.9 floor: lazy annotations, stdlib + pytest only, no PEP 604/585 syntax.
"""

from __future__ import annotations

import dataclasses
import datetime

import pytest

from blurt import assistant as assistant_pkg
from blurt.assistant import build_default_router
from blurt.assistant import intents as intents_module
from blurt.assistant.intents import RevertHandler
from blurt.assistant.router import IntentRouter
from blurt.assistant.types import Action, ActionResult, IntentHandler

# Fixed clock for the handlers that read one, so nothing in a full router
# depends on the day the suite happens to run. 2026-07-20 is a Monday.
now_fn = lambda: datetime.datetime(2026, 7, 20, 10, 30)  # noqa: E731


# --------------------------------------------------------------------------- #
# Fakes -- record the call, return what the test asked for, touch nothing real.
# --------------------------------------------------------------------------- #
class FakeRevert:
    """Stand-in for ``BlurtApp.revert_last``: counts calls, returns a canned bool.

    The real revert_last re-inserts the raw transcript through the injector and
    returns True/False depending on whether there was anything to put back. This
    records every call so the tests can assert the far more interesting fact:
    that it was NOT called.
    """

    def __init__(self, result=True, raises=None) -> None:
        self.result = result
        self.raises = raises
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return self.result


class FakeDictate:
    """Recording stand-in for the dictate fallback: remembers what it typed."""

    def __init__(self) -> None:
        self.calls = []  # list of dictated strings

    def __call__(self, text):
        self.calls.append(text)
        return ActionResult(ok=True, message="Dictated: {0}".format(text))


class GreedyHandler(IntentHandler):
    """A handler that claims EVERYTHING at revert's own confidence.

    Exists to prove the tie-break, not to model anything real. If a future
    handler ever gets clever about short imperatives, this is the shape it would
    take, and the ordering test says such a handler must still lose "undo" to
    the revert handler registered ahead of it.
    """

    name = "greedy"

    def __init__(self) -> None:
        self.executed = 0

    def match(self, text):
        return Action(
            kind="greedy",
            summary=text,
            payload={"_handler": self.name},
            confidence=0.95,
        )

    def execute(self, action):
        self.executed += 1
        return ActionResult(ok=True, message="greedy ran")


def build_revert_router(revert=None):
    """A router holding only the revert handler, plus its two recording fakes."""
    reverter = revert if revert is not None else FakeRevert(result=True)
    dictate = FakeDictate()
    router = IntentRouter([RevertHandler(reverter)], dictate)
    return router, reverter, dictate


# --------------------------------------------------------------------------- #
# THE FALSE-POSITIVE DIRECTION -- the expensive one, tested hardest.
# --------------------------------------------------------------------------- #
#: Ordinary dictation that happens to contain a revert-ish word. Every one of
#: these is something a person plausibly says into a text field, and every one
#: of them must be typed out verbatim rather than triggering a paste of stale
#: raw text. The first two are the sharp cases: they OPEN with the command verb
#: and are saved only by the full-utterance anchor.
_DICTATION_THAT_MENTIONS_UNDOING = [
    "I need to undo the migration before the deploy, can you note that",
    "undo the last commit in git and then force push to origin",
    "the revert button is greyed out in the admin panel",
    "scratch that itch",
    "we should revert to the previous vendor if the pricing does not improve",
    "tell Priya the undo feature shipped",
]

#: Second wave: shorter, closer to the command phrases, and therefore the ones a
#: loosened regex would claim first. "undo the migration" is two tokens away
#: from "undo the last one"; "never mind" is what people say when they change
#: their mind about a sentence, not when they want the previous one rewritten.
#:
#: Every entry here is refused by the ANCHOR -- the phrase is not on the closed
#: list, or it is but keeps going afterwards. None of them lean on the word
#: ceiling, which is enforced by
#: :func:`test_anchor_rejected_dictation_stays_rejected_with_the_ceiling_lifted`.
#: "please give me the raw text please" used to live here and does not any more:
#: it is the one negative in this file that the ANCHOR happily accepts, so filing
#: it under anchoring credited the wrong gate. It now sits in
#: :data:`_CEILING_ONLY_DICTATION` where its failure points at the right rule.
_NEAR_MISS_DICTATION = [
    "undo the last commit",
    "undo the migration",
    "can you undo the deploy",
    "revert to the previous vendor",
    "let's revert that decision",
    "scratch the plan",
    "scratch that, I meant tuesday",
    "never mind",
    "never mind, I'll do it later",
    "the undo button",
    "use the raw data instead",
    "I want the raw text from the sensor",
    "undo that change in the config",
]

#: THE RAW-TEXT FAMILY, AFTER THE TIGHTENING.
#:
#: The handler used to accept a GRAMMAR here -- one of six verbs, an optional
#: "the", "raw", and an optional noun -- which meant "use raw", "insert raw" and
#: "i want raw" were revert commands. They are not: they are a camera setting, a
#: file format, and a sentence about sushi. Firing on any of them pastes a stale
#: transcript into whatever the user was mid-way through writing.
#:
#: The grammar is gone, replaced by five literal phrases (see
#: ``_REVERT_RAW_PHRASE`` in :mod:`blurt.assistant.intents`). The first five
#: entries below are exactly the ones that regressed and must never come back.
#: The rest are the neighbourhood around them, because a closed list is only
#: worth having if something notices when it silently reopens: a verb with no
#: object, an accepted phrase with one more word on the end, and the same phrase
#: with the noun swapped. All are well under the word ceiling, so the anchor --
#: which is to say the vocabulary itself -- is the only thing refusing them.
_RAW_FAMILY_NEAR_MISSES = [
    # The five that regressed. These are the reason the grammar was deleted.
    "paste the raw text",
    "use the raw version",
    "insert raw",
    "i want raw",
    "use raw",
    # A verb plus "raw" with no complete noun phrase: never a command.
    "give me the raw",
    "show me the raw",
    "use the raw",
    "paste the raw",
    "raw text",
    "the raw text",
    # An accepted phrase plus one more word -- a request about a FILE, not an
    # undo. The anchor is the only thing between these and a paste.
    "show me the raw text file",
    "use the raw transcript from yesterday",
    "insert the raw text here",
    "export the raw text",
    "the raw text is fine",
    # The accepted five with the noun swapped. The list is enumerated, not
    # generated, so these near neighbours are outside it on purpose.
    "show me the raw transcript",
    "give me the raw version",
    "paste the raw transcript",
]

#: Plausible dictated English that contains a trigger word and is obviously not a
#: command. These are the sentences a person actually says into an email or a
#: ticket: talking ABOUT an undo, asking someone else to revert, using "scratch"
#: in its ordinary sense, or naming raw data that has nothing to do with a
#: transcript. Each is short enough that the word ceiling never comes into play,
#: which is the point -- if one of them ever matches, the vocabulary grew and no
#: length cap will save it.
_ADVERSARIAL_DICTATION = [
    # Talking about a revert rather than asking for one.
    "can we revert that",
    "i think we should undo that",
    "did you undo it",
    "he said to undo it",
    "we need to revert",
    "the undo stack is broken",
    # A revert of something that is not a dictation.
    "undo my last commit",
    "revert this pull request",
    "revert it back to draft",
    "undo it in photoshop",
    # "scratch" in its ordinary English sense.
    "scratch that off the list",
    "scratch that entry from the minutes",
    "scratch that idea",
    # "never mind" without the undo verb the pattern requires.
    "never mind undo the migration",
    "never mind that",
    "okay never mind",
    "nvm",
    # "raw" as an adjective about data, not about a transcript.
    "show me the raw data",
    "i want the raw file",
    "use the raw footage",
    "give me the raw numbers",
]

#: Every negative whose rejection is the ANCHOR's doing, gathered so one test can
#: assert that of all of them at once. Used only by the ceiling-lifted audit
#: below; the per-group tests above still carry the explanations.
_ANCHOR_REJECTED_DICTATION = (
    _DICTATION_THAT_MENTIONS_UNDOING
    + _NEAR_MISS_DICTATION
    + _RAW_FAMILY_NEAR_MISSES
    + _ADVERSARIAL_DICTATION
)


@pytest.mark.parametrize("text", _DICTATION_THAT_MENTIONS_UNDOING)
def test_ordinary_dictation_mentioning_undo_does_not_match(text):
    """THE KEY REGRESSION: a sentence that merely contains "undo" is dictation.

    If any of these starts matching, the handler will paste the previous raw
    transcript into whatever the user was actually writing, and nothing in blurt
    can retrieve what was overwritten. A missed revert costs one repeat; this
    costs a paragraph the user was not thinking about.
    """
    reverter = FakeRevert(result=True)
    handler = RevertHandler(reverter)

    assert handler.match(text) is None
    # And declining cost nothing: matching never reaches the backend.
    assert reverter.calls == 0


@pytest.mark.parametrize("text", _NEAR_MISS_DICTATION)
def test_near_miss_phrasing_does_not_match(text):
    """Phrases one word away from a command still fall through to dictation."""
    reverter = FakeRevert(result=True)
    handler = RevertHandler(reverter)

    assert handler.match(text) is None
    assert reverter.calls == 0


@pytest.mark.parametrize("text", _RAW_FAMILY_NEAR_MISSES)
def test_a_verb_plus_raw_is_dictation_not_a_revert_command(text):
    """The raw-text vocabulary is a closed list of five, and these are outside it.

    "use raw", "insert raw" and "i want raw" matched until the grammar behind
    this family was deleted; they are the regression this group exists to hold
    down. The rest map the border around the five accepted spellings, so a future
    edit that reintroduces a rule -- an optional article, an optional noun, a
    "raw" wildcard -- fails here rather than in someone's half-written email.
    """
    reverter = FakeRevert(result=True)
    handler = RevertHandler(reverter)

    assert handler.match(text) is None
    assert reverter.calls == 0


@pytest.mark.parametrize("text", _ADVERSARIAL_DICTATION)
def test_plausible_speech_containing_a_trigger_word_does_not_match(text):
    """Sentences a person really dictates, chosen to be as close as English gets.

    The handler cannot know what the user meant; all it has is the string. These
    are the strings where a human reader is certain -- "can we revert that" in a
    message to a colleague, "scratch that off the list", "show me the raw data"
    -- and the handler must reach the same conclusion from the words alone.
    """
    reverter = FakeRevert(result=True)
    handler = RevertHandler(reverter)

    assert handler.match(text) is None
    assert reverter.calls == 0


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_empty_or_blank_speech_does_not_match(text):
    handler = RevertHandler(FakeRevert(result=True))

    assert handler.match(text) is None


@pytest.mark.parametrize("text", [None, 123, object()])
def test_non_string_input_returns_none_rather_than_raising(text):
    """Garbage in must not become a traceback on the dictation worker thread."""
    handler = RevertHandler(FakeRevert(result=True))

    assert handler.match(text) is None


# --------------------------------------------------------------------------- #
# THE TWO GATES, KEPT HONEST -- a negative has to name the rule that rejects it.
# --------------------------------------------------------------------------- #
# A rejection is not evidence about WHICH gate rejected. That is exactly how the
# first version of the word-ceiling test below went vacuous. It read:
#
#     text = "undo that " + "and also this thing over here too"
#     assert len(text.split()) > 6
#     assert handler.match(text) is None
#
# and it claimed to pin the ceiling. It did not. The anchor refuses ANY utterance
# with a tail after the command, so that string was already dead on arrival; the
# ceiling never got a vote. The test passed with _MAX_REVERT_WORDS at 6 and
# passed just as green with it at a million -- it defended nothing while
# advertising that it did, which is worse than not existing, because the next
# person to touch the ceiling reads it and believes they are covered.
#
# The repair is to test each gate from BOTH sides, and it is cheap:
#
#   * a negative that credits the ANCHOR must stay rejected when the ceiling is
#     lifted out of the way. If lifting the ceiling makes it match, the anchor
#     was never the thing refusing it and the test was mis-filed.
#   * a negative that credits the CEILING must START matching when the ceiling is
#     lifted. If it stays rejected, something else is doing the work and the test
#     is the vacuous one all over again.
#
# Together those two make it impossible for this file to go quiet: whichever gate
# stops working, exactly the tests that name it turn red.

#: Big enough that no utterance a human produces could reach it, so the ceiling
#: is genuinely out of the way rather than merely relaxed.
_LIFTED_CEILING = 10 ** 6


def lift_the_word_ceiling(monkeypatch):
    """Take the word ceiling out of play for the duration of one test.

    Patched on the intents MODULE, not on a copy of anything: the ceiling is read
    out of the module global inside ``_is_revert_command`` on every call, so
    rebinding the name is enough and no handler has to be rebuilt. monkeypatch
    restores it at teardown, so a test that lifts the ceiling cannot leak a
    wide-open matcher into the next one -- which, given what these tests are
    about, would be a particularly ironic way to lose coverage.
    """
    monkeypatch.setattr(intents_module, "_MAX_REVERT_WORDS", _LIFTED_CEILING)


#: Utterances the anchor ACCEPTS in full and the word count alone turns down.
#: Each is a real command phrase wearing enough filler to go over six words --
#: stacked leads ("hey please just ..."), the politeness sandwich ("please give
#: me the raw text please", the example the source comment itself cites), and the
#: longest spelling of the mid-sentence correction ("it's not what i said undo
#: it", seven words). These are the only strings in this file that can prove the
#: ceiling exists, because they are the only ones where nothing else objects.
_CEILING_ONLY_DICTATION = [
    "please give me the raw text please",
    "it's not what i said undo it",
    "hey please just undo the last dictation",
    "okay so please show me the raw text",
    "um please never mind undo that one",
]


@pytest.mark.parametrize("text", _CEILING_ONLY_DICTATION)
def test_an_over_long_utterance_is_refused_by_the_word_ceiling_alone(text, monkeypatch):
    """The ceiling, pinned from both sides so it can never go vacuous again.

    Half of this test is the ordinary assertion: the utterance does not match.
    The other half is the half that has teeth -- lift ``_MAX_REVERT_WORDS`` out
    of the way and the SAME string matches. That second assertion is what proves
    the first one was about the ceiling: if the anchor or the vocabulary were
    also refusing this text, raising the ceiling would change nothing and the
    test would fail here instead of quietly passing forever.

    Which direction a failure points is worth spelling out. First assertion red
    means the ceiling stopped applying and over-long speech is now matching --
    the dangerous direction. Second assertion red means the ceiling is no longer
    the thing refusing these strings, so whatever is refusing them may not be a
    length rule at all and this test has stopped guarding the constant.
    """
    reverter = FakeRevert(result=True)
    handler = RevertHandler(reverter)

    assert len(text.split()) > intents_module._MAX_REVERT_WORDS
    assert handler.match(text) is None

    lift_the_word_ceiling(monkeypatch)

    action = handler.match(text)
    assert action is not None, (
        "with the word ceiling lifted this text still does not match, so the "
        "ceiling was not what rejected it -- this test is no longer pinning the "
        "constant it names"
    )
    assert action.kind == "revert"
    # Recognising it is still not doing it, even in the lifted world.
    assert reverter.calls == 0


def test_a_trailing_clause_is_refused_by_the_anchor_not_the_word_count():
    """The old ceiling test's string, filed under the gate that actually rejects.

    "undo that and also this thing over here too" opens with a real command and
    keeps going. The anchor is what stops it: ``^...$`` with no wildcard anywhere
    means a command phrase is never a prefix of a longer sentence. That is a
    property worth a test -- it is the difference between "scratch that" and
    "scratch that itch" -- it just is not a property of the word ceiling.
    """
    reverter = FakeRevert(result=True)
    handler = RevertHandler(reverter)
    text = "undo that " + "and also this thing over here too"

    assert handler.match(text) is None
    assert reverter.calls == 0


def test_a_trailing_clause_stays_refused_with_the_word_ceiling_lifted(monkeypatch):
    """And the anchor holds that string up on its own, with no help from length.

    This is the assertion the original test needed and did not have. Lift the
    ceiling and the sentence is still not a command, which is what makes the
    previous test a statement about anchoring rather than an accident of word
    count.
    """
    handler = RevertHandler(FakeRevert(result=True))
    text = "undo that " + "and also this thing over here too"

    lift_the_word_ceiling(monkeypatch)

    assert handler.match(text) is None


@pytest.mark.parametrize("text", _ANCHOR_REJECTED_DICTATION)
def test_anchor_rejected_dictation_stays_rejected_with_the_ceiling_lifted(
    text, monkeypatch
):
    """Audit, made executable: every other negative in this file is the anchor's.

    Run over all four negative lists at once. It says something narrow and
    useful: none of those rejections depend on the length cap. Two things follow.
    The anchor tests are honest -- they fail if anchoring breaks, instead of
    being propped up by a word count nobody was testing. And a new negative that
    is secretly a ceiling case cannot be dropped into the wrong list, because it
    matches here and this test names the mistake.

    The one entry that DID belong to the ceiling ("please give me the raw text
    please") is not in these lists any more; it moved to
    :data:`_CEILING_ONLY_DICTATION`. This test is why the move was not optional.
    """
    reverter = FakeRevert(result=True)
    handler = RevertHandler(reverter)

    lift_the_word_ceiling(monkeypatch)

    assert handler.match(text) is None, (
        "this text only fails to match because of the word ceiling, so it belongs "
        "in _CEILING_ONLY_DICTATION -- as written it credits the anchor for a "
        "rejection the anchor is not making"
    )
    assert reverter.calls == 0


def test_the_ceiling_sits_exactly_between_six_words_and_seven():
    """Where the cut falls, pinned by a pair that differs by one filler word.

    "give me the raw text please" is six words and a command. Put one more
    "please" on the front and it is seven words, still a perfect match for the
    pattern, and refused. Nothing else separates them -- same vocabulary, same
    anchor, same politeness -- so the pair pins the VALUE of the constant the way
    the tests above pin its existence, without hard-coding the number and without
    a lifted-ceiling monkeypatch.

    It also doubles as insurance that the lifts above were undone: this runs
    after them, and a leaked ceiling would let the seven-word half through.
    """
    handler = RevertHandler(FakeRevert(result=True))
    six_words = "give me the raw text please"
    seven_words = "please give me the raw text please"

    assert len(six_words.split()) == 6
    assert len(seven_words.split()) == 7

    assert handler.match(six_words) is not None
    assert handler.match(seven_words) is None


# --------------------------------------------------------------------------- #
# THE POSITIVE DIRECTION -- everything the handler is meant to accept.
# --------------------------------------------------------------------------- #
#: Every phrase the handler is documented to take, plus the shapes speech
#: actually arrives in: a trailing "please", the full stop Whisper adds to short
#: utterances, and casing nobody controls.
_REVERT_COMMANDS = [
    "undo that",
    "undo",
    "revert that",
    "revert",
    "undo last",
    "undo the last one",
    "revert the last dictation",
    "scratch that",
    # Trailing politeness.
    "undo that please",
    "revert that please",
    "scratch that please",
    "give me the raw text please",
    # Leading filler.
    "please undo that",
    "just undo that",
    "okay revert that",
    # Punctuation Whisper sprinkles on, and casing it picks at random.
    "Undo that.",
    "UNDO THAT!",
    "  Revert, please  ",
    "Undo That?",
    "Scratch that.",
    "\n undo that \n",
    # Other objects of the same command.
    "undo it",
    "undo this",
    "revert it",
    "undo the last thing",
    "undo the last transcript",
    "undo the cleanup",
    "scratch this",
    # The mid-thought correction, and the raw-text phrasings.
    "that's wrong undo it",
    "never mind undo",
    # The five raw-text spellings that SURVIVED the tightening, in every shape
    # they arrive in. The family used to be generated from a verb list and is now
    # an enumerated set of exactly these five; the negatives above cover what was
    # removed, and these cover what was kept, so a future edit cannot quietly
    # narrow the list either.
    "use the raw text",
    "use the raw transcript",
    "give me the raw text",
    "show me the raw text",
    "paste the raw version",
    "use the raw text please",
    "show me the raw text please",
    "paste the raw version please",
    "please use the raw text",
    "just show me the raw text",
]


@pytest.mark.parametrize("text", _REVERT_COMMANDS)
def test_a_whole_utterance_command_matches_with_high_confidence(text):
    """Each accepted phrase yields a revert Action at the module's top score.

    0.95 is deliberately above every other handler's ceiling (calendar and timer
    top out at 0.9). Nothing here matches unless the ENTIRE utterance is a known
    command, so when it does match there is no competing reading left to weigh --
    and a bare "undo" outranks anything else that might get creative about the
    word.
    """
    handler = RevertHandler(FakeRevert(result=True))

    action = handler.match(text)

    assert action is not None
    assert action.kind == "revert"
    assert action.confidence == pytest.approx(0.95)
    assert action.confidence > 0.9
    # The user said it out loud on purpose; nothing to confirm.
    assert action.needs_confirmation is False
    assert action.summary


#: The verbs and objects the deleted grammar used to multiply together. Six verbs
#: times five objects is thirty phrases, of which exactly five are commands; the
#: other twenty-five are what the grammar was handing out for free.
_RAW_GRAMMAR_VERBS = ("use", "give me", "show me", "paste", "insert", "i want")
_RAW_GRAMMAR_OBJECTS = (
    "the raw text",
    "the raw version",
    "the raw transcript",
    "the raw",
    "raw",
)

#: The whole accepted raw-text vocabulary, written out. Duplicated from the
#: source on purpose: a closed list is only closed if something outside it says
#: what it contains, otherwise "the list is whatever the regex currently says"
#: and reopening it costs nothing.
_ACCEPTED_RAW_PHRASES = frozenset(
    [
        "use the raw text",
        "use the raw transcript",
        "give me the raw text",
        "show me the raw text",
        "paste the raw version",
    ]
)


def test_the_raw_text_family_is_a_closed_list_and_not_a_grammar():
    """Five spellings match; the twenty-five near neighbours do not.

    This walks the exact cross-product the handler used to generate and asserts
    the accepted set is precisely the enumerated five. It is the difference
    between a rule and a list: a rule that admits "use the raw text" also admits
    "insert raw" for free, and "insert raw" is a sentence about a photograph.

    Every phrase here is at or under the word ceiling, asserted below, so the
    ceiling cannot be what refuses the twenty-five. The vocabulary is doing all
    of the work, which is what this test is for.

    If this fails because the accepted set GREW, that is the audit question
    arriving on schedule: is there a plausible English sentence where somebody
    says exactly the new phrase, as their whole utterance, and does not mean
    "undo my last dictation"? If the answer is no, add the spelling to
    :data:`_ACCEPTED_RAW_PHRASES` deliberately. It must not arrive as a side
    effect of loosening something else.
    """
    handler = RevertHandler(FakeRevert(result=True))

    phrases = [
        "{0} {1}".format(verb, obj)
        for verb in _RAW_GRAMMAR_VERBS
        for obj in _RAW_GRAMMAR_OBJECTS
    ]
    for phrase in phrases:
        assert len(phrase.split()) <= intents_module._MAX_REVERT_WORDS, (
            "this phrase is over the word ceiling, so its rejection would prove "
            "nothing about the raw-text vocabulary"
        )

    accepted = set(phrase for phrase in phrases if handler.match(phrase) is not None)

    assert accepted == set(_ACCEPTED_RAW_PHRASES)


def test_the_action_names_its_owner_in_the_payload():
    handler = RevertHandler(FakeRevert(result=True))

    action = handler.match("undo that")

    assert action.payload["_handler"] == "revert"
    assert RevertHandler.name == "revert"


def test_the_summary_says_it_is_about_the_raw_transcript():
    """The summary is spoken/logged back, so it has to describe what happened."""
    handler = RevertHandler(FakeRevert(result=True))

    action = handler.match("undo that")

    lowered = action.summary.lower()
    assert "revert" in lowered
    assert "raw" in lowered


# --------------------------------------------------------------------------- #
# MATCHING IS INERT -- recognition must never be execution.
# --------------------------------------------------------------------------- #
def test_matching_a_command_does_not_revert_anything():
    """match() inspects; only execute() may touch the user's buffer.

    Worth its own test even though it looks obvious: a handler that reverted
    during match() would fire on every routed phrase the moment any caller asked
    "is this a command?" -- including a caller that then decided not to run it.
    """
    reverter = FakeRevert(result=True)
    handler = RevertHandler(reverter)

    for text in _REVERT_COMMANDS:
        handler.match(text)

    assert reverter.calls == 0


def test_routing_a_command_does_not_revert_anything():
    router, reverter, dictate = build_revert_router()

    action = router.route("undo that")

    assert action.kind == "revert"
    assert reverter.calls == 0
    assert dictate.calls == []


# --------------------------------------------------------------------------- #
# EXECUTE -- the three outcomes, and the one that must never escape.
# --------------------------------------------------------------------------- #
def test_execute_reports_success_when_the_revert_happened():
    reverter = FakeRevert(result=True)
    handler = RevertHandler(reverter)

    result = handler.execute(handler.match("undo that"))

    assert isinstance(result, ActionResult)
    assert result.ok is True
    assert reverter.calls == 1
    assert "revert" in result.message.lower()


def test_execute_does_not_claim_success_when_the_revert_declined():
    """A False from revert_fn must not be announced as if it worked.

    revert_last returns False for several ordinary reasons -- nothing dictated
    yet, already reverted, raw history switched off, cleanup changed nothing --
    and it has already printed the specific one to the console. What matters
    here is only that the spoken-back message does not contradict it by claiming
    the text was put back when it was not; a user who believes a revert happened
    stops checking.
    """
    succeeded = RevertHandler(FakeRevert(result=True))
    success_message = succeeded.execute(succeeded.match("undo that")).message

    reverter = FakeRevert(result=False)
    handler = RevertHandler(reverter)

    result = handler.execute(handler.match("undo that"))

    assert result.ok is False
    assert reverter.calls == 1
    assert result.message
    assert result.message != success_message
    lowered = result.message.lower()
    for claim in ("reverted to", "restored", "put back", "undone", "here's the raw"):
        assert claim not in lowered, "the decline message claims the revert happened"


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("injector is wedged"),
        ValueError("no raw text"),
        OSError("clipboard unavailable"),
        AttributeError("app went away"),
    ],
)
def test_execute_swallows_a_raising_revert_fn(exc):
    """An exception out of execute() would kill the dictation worker thread.

    This runs on the same thread that finishes a dictation. If it escapes, the
    loop dies and every later utterance is lost with no error the user will see
    -- silent and unrecoverable, the exact failure this codebase is built to
    avoid. A backend that blows up must look identical to one that declined.
    """
    reverter = FakeRevert(raises=exc)
    handler = RevertHandler(reverter)

    result = handler.execute(handler.match("undo that"))

    assert isinstance(result, ActionResult)
    assert result.ok is False
    assert result.message
    assert reverter.calls == 1


def test_a_raising_revert_fn_does_not_poison_the_handler():
    """The handler stays usable after a backend blew up on the previous try."""
    reverter = FakeRevert(raises=RuntimeError("transient"))
    handler = RevertHandler(reverter)

    assert handler.execute(handler.match("undo that")).ok is False

    reverter.raises = None
    reverter.result = True
    assert handler.execute(handler.match("undo that")).ok is True
    assert reverter.calls == 2


def test_execute_needs_nothing_from_the_payload():
    """execute() must not depend on payload contents it did not put there.

    The router's backstop path can hand back an Action whose payload has been
    rebuilt (see the dispatch tests below), so reading anything but the handler
    name out of it would be a latent KeyError on the worker thread.
    """
    reverter = FakeRevert(result=True)
    handler = RevertHandler(reverter)

    result = handler.execute(Action(kind="revert", summary="", payload={}))

    assert result.ok is True
    assert reverter.calls == 1


# --------------------------------------------------------------------------- #
# DISPATCH WIRING -- the Action has to find its way home through the router.
# --------------------------------------------------------------------------- #
def test_router_routes_a_command_to_revert_and_executes_it_there():
    router, reverter, dictate = build_revert_router()

    action = router.route("undo that")
    result = router.execute(action)

    assert action.kind == "revert"
    assert result.ok is True
    assert reverter.calls == 1
    # The revert took the place of dictation entirely: nothing was typed out.
    assert dictate.calls == []


def test_router_still_finds_the_owner_with_the_handler_key_removed():
    """The kind->name backstop has to work on its own.

    ``payload["_handler"]`` is the fast path, and the ``_KIND_TO_NAME`` map in
    the router is the belt to its braces. If only the fast path worked, an Action
    that lost its payload anywhere in the pipeline would be routed to
    ``_run_dictate`` and its SUMMARY would be typed into the user's document --
    the literal words "Revert to the raw transcript". So the backstop is tested
    with the key genuinely gone, not merely present-and-correct.
    """
    router, reverter, dictate = build_revert_router()

    action = router.route("undo that")
    stripped = dataclasses.replace(action, payload={})
    assert "_handler" not in stripped.payload

    result = router.execute(stripped)

    assert result.ok is True
    assert reverter.calls == 1
    assert dictate.calls == []


def test_router_falls_back_to_the_kind_when_the_handler_name_is_unknown():
    """An unrecognised owner name is the same case as a missing one."""
    router, reverter, dictate = build_revert_router()

    action = router.route("undo that")
    mislabelled = dataclasses.replace(action, payload={"_handler": "no_such_handler"})

    result = router.execute(mislabelled)

    assert result.ok is True
    assert reverter.calls == 1
    assert dictate.calls == []


def test_a_revert_action_on_a_router_without_the_handler_reverts_nothing():
    """No owner anywhere -> dictate the summary; never guess at another handler."""
    reverter = FakeRevert(result=True)
    dictate = FakeDictate()
    router = IntentRouter([], dictate)

    result = router.execute(
        Action(kind="revert", summary="Revert to the raw transcript", payload={})
    )

    assert reverter.calls == 0
    assert result.ok is True
    assert dictate.calls == ["Revert to the raw transcript"]


def test_revert_wins_a_confidence_tie_against_a_later_handler():
    """Handler order is the tie-break, and revert is registered ahead of others.

    A bare "undo" is one word with no object. Any handler that gets creative
    about short imperatives could claim it at the same score, and the user would
    have no way to reach the raw text -- it exists nowhere else they can get to.
    """
    reverter = FakeRevert(result=True)
    greedy = GreedyHandler()
    dictate = FakeDictate()
    router = IntentRouter([RevertHandler(reverter), greedy], dictate)

    action = router.route("undo")
    router.execute(action)

    assert action.kind == "revert"
    assert reverter.calls == 1
    assert greedy.executed == 0


# --------------------------------------------------------------------------- #
# build_default_router -- opt-in, and registered first when opted into.
# --------------------------------------------------------------------------- #
class RecordingRouter(IntentRouter):
    """A real IntentRouter that also remembers the handler list it was built with.

    Substituted for :class:`IntentRouter` inside :mod:`blurt.assistant` so the
    registration ORDER can be asserted without reaching into the router's
    private attributes. It is a subclass rather than a stub because the router
    it returns is still used for real routing in the same tests.
    """

    def __init__(self, handlers, dictate_fallback) -> None:
        self.built_with = list(handlers)
        IntentRouter.__init__(self, handlers, dictate_fallback)


@pytest.mark.parametrize("text", ["undo that", "undo", "scratch that", "revert that"])
def test_default_router_without_revert_fn_dictates_undo(text):
    """No revert_fn -> no handler -> existing callers behave exactly as before.

    Registering the handler with a stub backend would put a live "undo that" in
    front of a function that cannot do it. Opting out has to mean the phrase is
    just words, and words get typed.
    """
    dictate = FakeDictate()
    router = build_default_router(dictate, now_fn=now_fn)

    action = router.route(text)
    result = router.execute(action)

    assert action.kind == "dictate"
    assert result.ok is True
    assert dictate.calls == [text]


def test_default_router_without_revert_fn_registers_no_revert_handler(monkeypatch):
    monkeypatch.setattr(assistant_pkg, "IntentRouter", RecordingRouter)

    router = build_default_router(FakeDictate(), now_fn=now_fn)

    assert not any(
        isinstance(handler, RevertHandler) for handler in router.built_with
    )


@pytest.mark.parametrize("text", ["undo that", "undo", "scratch that", "revert that"])
def test_default_router_with_revert_fn_routes_undo_to_revert(text):
    dictate = FakeDictate()
    reverter = FakeRevert(result=True)
    router = build_default_router(dictate, now_fn=now_fn, revert_fn=reverter)

    action = router.route(text)
    result = router.execute(action)

    assert action.kind == "revert"
    assert result.ok is True
    assert reverter.calls == 1
    assert dictate.calls == []


def test_default_router_registers_revert_first(monkeypatch):
    """Position, not just presence: the tie-break rule is the whole point.

    :meth:`IntentRouter.route` keeps the first handler on a confidence tie, so
    "registered" and "registered first" are different guarantees and only the
    second one protects a bare "undo".
    """
    monkeypatch.setattr(assistant_pkg, "IntentRouter", RecordingRouter)

    router = build_default_router(
        FakeDictate(), now_fn=now_fn, revert_fn=FakeRevert(result=True)
    )

    assert router.built_with, "the default router registered no handlers at all"
    assert isinstance(router.built_with[0], RevertHandler)
    # And it was added in FRONT of the others, not swapped in for one of them.
    names = [getattr(handler, "name", None) for handler in router.built_with]
    for name in ("calendar", "reminder", "timer", "open_app"):
        assert name in names


@pytest.mark.parametrize("text", _DICTATION_THAT_MENTIONS_UNDOING)
def test_default_router_with_revert_fn_still_dictates_ordinary_speech(text):
    """The false-positive guard end to end, with every real handler in play.

    The unit tests above prove RevertHandler declines these. This proves the
    assembled router does too -- that no other handler picks them up and, above
    all, that the user's raw text is not pasted into a sentence about a git
    commit.
    """
    dictate = FakeDictate()
    reverter = FakeRevert(result=True)
    router = build_default_router(dictate, now_fn=now_fn, revert_fn=reverter)

    action = router.route(text)

    assert action.kind != "revert"
    # Checked BEFORE execute(): this router holds the real calendar/timer/app
    # backends, and executing an Action that some other handler unexpectedly
    # claimed would reach out to the machine. The kind assertion above is the
    # property under test; this one keeps a future false positive from turning
    # a unit test into a side effect.
    assert action.kind == "dictate"
    router.execute(action)

    assert reverter.calls == 0
    assert dictate.calls == [text]


@pytest.mark.parametrize(
    "text", _RAW_FAMILY_NEAR_MISSES + _ADVERSARIAL_DICTATION
)
def test_default_router_dictates_the_tightened_near_misses_verbatim(text):
    """The tightened vocabulary, checked where the user actually meets it.

    RevertHandler declining in isolation is necessary and not sufficient: what
    reaches the buffer is whatever the assembled router decides. This asserts the
    full path -- no handler claims these, the fallback dictates them, and the
    text that comes out the far end is character-for-character what went in. A
    phrase that silently lost a word on its way through would be its own bug,
    quieter than a false revert but the same kind.
    """
    dictate = FakeDictate()
    reverter = FakeRevert(result=True)
    router = build_default_router(dictate, now_fn=now_fn, revert_fn=reverter)

    action = router.route(text)

    assert action.kind != "revert"
    assert action.kind == "dictate"
    router.execute(action)

    assert reverter.calls == 0
    assert dictate.calls == [text]


def test_default_router_with_revert_fn_leaves_the_other_intents_alone():
    """Adding revert in front must not shadow the handlers behind it."""
    reverter = FakeRevert(result=True)
    router = build_default_router(
        FakeDictate(), now_fn=now_fn, revert_fn=reverter
    )

    assert router.route("schedule lunch tomorrow at noon").kind == "calendar_event"
    assert router.route("remind me to call mom").kind == "reminder"
    assert router.route("set a timer for 5 minutes").kind == "timer"
    assert router.route("open Safari").kind == "open_app"
    assert reverter.calls == 0
