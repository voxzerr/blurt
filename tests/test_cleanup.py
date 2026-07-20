"""Tests for blurt.cleanup -- the pass that edits the user's words.

This is the most important test file in the project, and the reason is narrow:
cleanup is the only module that DELETES things the user actually said. Every
other failure in blurt is loud (no text appears, the hotkey does nothing, the
model fails to load). A cleanup failure is silent -- the user gets text that
looks fine and means something different.

The centre of gravity of this file is the PROTECTED DISCOURSE WORDS section.
Those tests exist to fail loudly the day somebody decides "like" and "so" are
filler words and adds them to FILLERS. They are written as many small explicit
assertions rather than a parametrized sweep so that a failure names the exact
sentence that broke.

Everything here is pure logic: no microphone, no model, no network, no macOS
permissions. It runs on the Intel floor machine under Python 3.9.6.

Python 3.9 floor: lazy annotations, typing.List rather than list[...].
"""

from __future__ import annotations

import re
from typing import List

import pytest

from blurt.cleanup import (
    FILLERS,
    MAX_CORRECTION_TOKENS,
    PROTECTED_DISCOURSE_WORDS,
    REPETITION_ALLOWLIST,
    clean,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _words(text: str) -> List[str]:
    """Lowercased word tokens of a string, for 'is this word still present?'."""
    return re.findall(r"[a-z0-9']+", text.lower())


def _survives(sentence: str, word: str, level: str = "light") -> bool:
    """True if every word of `word` is still present after cleaning `sentence`."""
    got = _words(clean(sentence, level))
    return all(part in got for part in _words(word))


# ---------------------------------------------------------------------------
# Level "none": strip only
# ---------------------------------------------------------------------------


def test_none_level_strips_leading_and_trailing_whitespace_only():
    assert clean("  hello world  ", "none") == "hello world"


def test_none_level_does_not_remove_fillers():
    assert clean("um I think so", "none") == "um I think so"


def test_none_level_does_not_change_casing():
    assert clean("hello there. how are you?", "none") == "hello there. how are you?"


def test_none_level_does_not_collapse_internal_whitespace():
    assert clean("hello    world", "none") == "hello    world"


def test_none_level_does_not_collapse_repeated_words():
    assert clean("the the cat", "none") == "the the cat"


def test_none_level_ignores_the_dictionary():
    assert clean("github", "none", {"github": "GitHub"}) == "github"


# ---------------------------------------------------------------------------
# Filler removal (light level) -- start, middle, end
# ---------------------------------------------------------------------------


def test_filler_um_removed_at_start():
    assert clean("um I think so") == "I think so"


def test_filler_uh_removed_in_middle():
    assert clean("I think uh we should go") == "I think we should go"


def test_filler_um_removed_at_end():
    assert clean("we should go um") == "We should go"


def test_filler_er_removed():
    assert clean("er I know") == "I know"


def test_filler_ah_removed():
    assert clean("I ah know") == "I know"


def test_filler_hmm_removed_at_start():
    assert clean("hmm let me think") == "Let me think"


def test_filler_mm_removed():
    assert clean("mm yes") == "Yes"


def test_filler_uhm_removed():
    assert clean("uhm okay") == "Okay"


def test_filler_erm_removed():
    assert clean("erm okay") == "Okay"


def test_multiple_fillers_in_one_sentence_all_removed():
    assert clean("um I uh think er we should ah go") == "I think we should go"


def test_filler_removal_is_case_insensitive():
    assert clean("Um I think so") == "I think so"
    assert clean("UM I think so") == "I think so"


def test_filler_removal_repairs_the_comma_it_leaves_behind():
    assert clean("Um, yes") == "Yes"


def test_filler_removal_repairs_a_doubled_comma():
    assert clean("yes, um, please") == "Yes, please"


def test_filler_removal_before_terminal_punctuation_keeps_the_period():
    assert clean("we should go, um.") == "We should go."


def test_repeated_filler_run_collapses_to_nothing():
    assert clean("um, um, hello") == "Hello"


# ---------------------------------------------------------------------------
# Fillers must match WHOLE WORDS ONLY
# ---------------------------------------------------------------------------


def test_um_inside_umbrella_is_not_touched():
    assert clean("the umbrella is red") == "The umbrella is red"


def test_um_inside_drum_is_not_touched():
    assert clean("he played the drum") == "He played the drum"


def test_um_inside_hummed_is_not_touched():
    assert clean("I hummed") == "I hummed"


def test_uh_inside_uhuru_is_not_touched():
    assert "uhuru" in _words(clean("the uhuru monument"))


def test_er_inside_error_is_not_touched():
    assert clean("an error occurred") == "An error occurred"


def test_ah_inside_ahead_is_not_touched():
    assert clean("go ahead") == "Go ahead"


def test_mm_inside_hammer_is_not_touched():
    assert clean("pass the hammer") == "Pass the hammer"


# ===========================================================================
# ***  REGRESSION: MEANING-BEARING DISCOURSE WORDS ARE NEVER REMOVED.    ***
# ===========================================================================
#
# READ THIS BEFORE "IMPROVING" blurt.cleanup.FILLERS.
#
# Every word below shows up on internet lists of "filler words to remove".
# Every one of them is also load-bearing English. There is no reliable way to
# tell the two uses apart without a parser we do not have and latency we cannot
# afford, so blurt does not try: it never deletes any of them.
#
# Deleting these does not make the transcript tidier, it makes it WRONG:
#
#   "I like it"            -> "I it"            (verb destroyed)
#   "so we should ship"    -> "we should ship"  (the causal link is gone)
#   "the right answer"     -> "the answer"      (adjective destroyed)
#   "it works well"        -> "it works"        (adverb destroyed)
#   "you know the answer"  -> "the answer"      (subject and verb destroyed)
#   "it's actually cheaper"-> "it's cheaper"    (the contrast was the point)
#   "basically it works"   -> "it works"        (a hedge became a promise)
#
# The last two are the dangerous ones: the sentence still reads as fluent
# English, so the user has no signal that we changed what they said. That is
# the failure mode this entire file exists to prevent.
#
# If one of these tests fails, do not update the test. Revert the change.
# ===========================================================================


def test_REGRESSION_like_is_never_removed_as_a_filler():
    # "like" as a verb. Removing it leaves "I it".
    assert clean("I like it") == "I like it"


def test_REGRESSION_like_survives_as_a_preposition():
    assert clean("do it like this") == "Do it like this"


def test_REGRESSION_like_survives_as_a_quotative():
    assert _survives("she was like no way", "like")


def test_REGRESSION_like_survives_even_next_to_a_real_filler():
    # The filler goes, "like" stays. This is the exact discrimination that
    # matters: we remove sounds, not words.
    assert clean("um I like it") == "I like it"


def test_REGRESSION_so_is_never_removed_as_a_filler():
    # "so" as a conjunction. This is the canonical case from the spec.
    assert clean("so we should ship") == "So we should ship"


def test_REGRESSION_so_survives_as_an_intensifier():
    assert clean("I am so tired") == "I am so tired"


def test_REGRESSION_so_survives_mid_sentence():
    assert clean("it broke so I left") == "It broke so I left"


def test_REGRESSION_right_is_never_removed_as_a_filler():
    assert clean("the right answer") == "The right answer"


def test_REGRESSION_right_survives_as_a_direction():
    assert clean("turn right at the corner") == "Turn right at the corner"


def test_REGRESSION_right_survives_twice_in_one_sentence():
    got = clean("the right answer is right there")
    assert got == "The right answer is right there"
    assert _words(got).count("right") == 2


def test_REGRESSION_well_is_never_removed_as_a_filler():
    assert clean("it works well") == "It works well"


def test_REGRESSION_well_survives_as_an_adjective():
    assert clean("I am not well") == "I am not well"


def test_REGRESSION_you_know_is_never_removed_as_a_filler():
    # A literal claim about what the listener knows.
    assert clean("you know the answer") == "You know the answer"


def test_REGRESSION_you_know_survives_mid_sentence():
    got = clean("I think you know why")
    assert got == "I think you know why"
    assert "you" in _words(got)
    assert "know" in _words(got)


def test_REGRESSION_actually_is_never_removed_as_a_filler():
    # The contrast IS the sentence. "it is cheaper" is a different claim.
    assert clean("it is actually cheaper") == "It is actually cheaper"


def test_REGRESSION_actually_survives_at_the_start_of_a_sentence():
    assert clean("actually I agree") == "Actually I agree"


def test_REGRESSION_basically_is_never_removed_as_a_filler():
    # A hedge. Removing it upgrades a qualified statement into a flat one.
    assert clean("basically it works") == "Basically it works"


def test_REGRESSION_basically_survives_mid_sentence():
    assert clean("it is basically done") == "It is basically done"


def test_REGRESSION_i_mean_is_never_removed_as_a_filler():
    assert clean("I mean it") == "I mean it"


def test_REGRESSION_every_protected_word_survives_a_sentence_containing_it():
    # Belt and braces: whatever else changes, the word is still in the output.
    sentences = {
        "like": "I like it",
        "so": "so we should ship",
        "right": "the right answer",
        "well": "it works well",
        "you know": "you know the answer",
        "actually": "it is actually cheaper",
        "basically": "basically it works",
        "i mean": "I mean it",
    }
    for word, sentence in sentences.items():
        assert _survives(sentence, word), (
            "cleanup deleted the protected word %r from %r -- it produced %r. "
            "This destroys meaning. See PROTECTED_DISCOURSE_WORDS in "
            "blurt/cleanup.py." % (word, sentence, clean(sentence))
        )


def test_REGRESSION_protected_words_survive_at_every_level():
    # "actually" and "i mean" are excluded here: at "standard" they are
    # correction MARKERS, which is a different, explicitly bounded operation
    # covered by its own test below. They are still never removed as fillers.
    for word in ["like", "so", "right", "well", "you know", "basically"]:
        sentence = "we think %s it is fine" % (word,)
        for level in ("none", "light", "standard"):
            assert _survives(sentence, word, level), (
                "level %r deleted protected word %r" % (level, word)
            )


def test_REGRESSION_fillers_and_protected_words_never_overlap():
    # The import-time tripwire in cleanup.py guards this too; this test makes
    # the failure legible instead of an ImportError during collection.
    overlap = FILLERS & PROTECTED_DISCOURSE_WORDS
    assert overlap == frozenset(), (
        "These meaning-bearing words were added to FILLERS: %s" % (sorted(overlap),)
    )


def test_REGRESSION_the_filler_list_stays_closed():
    # Pin the exact contents. Adding a word here must be a deliberate edit to
    # this test, reviewed on its own, not a drive-by change to cleanup.py.
    assert FILLERS == frozenset(
        {"um", "uh", "er", "ah", "mm", "hmm", "uhm", "erm"}
    )


# ---------------------------------------------------------------------------
# Adjacent repetition collapse
# ---------------------------------------------------------------------------


def test_adjacent_repetition_collapses():
    assert clean("the the cat sat") == "The cat sat"


def test_triple_repetition_collapses_to_one():
    assert clean("the the the cat") == "The cat"


def test_repetition_collapse_is_case_insensitive_and_keeps_the_first_casing():
    assert clean("THE THE CAT") == "THE CAT"


def test_repetition_collapse_keeps_the_first_token_casing_at_sentence_start():
    assert clean("The the cat") == "The cat"


def test_had_had_is_preserved_because_past_perfect_is_real():
    assert clean("he had had enough") == "He had had enough"


def test_that_that_is_preserved():
    assert clean("the fact that that man left") == "The fact that that man left"


def test_very_very_is_preserved():
    assert clean("very very good") == "Very very good"


def test_so_so_is_preserved():
    assert clean("so, so tired") == "So, so tired"


def test_spoken_digit_runs_are_preserved_because_they_are_data():
    assert clean("five five five") == "Five five five"


def test_is_is_collapses_because_it_is_a_stutter_not_the_copula():
    assert clean("is is broken") == "Is broken"


def test_repetition_separated_by_a_comma_is_left_alone():
    # A repetition with punctuation between it is likelier intentional.
    assert clean("the, the cat") == "The, the cat"


def test_repetition_separated_by_another_word_is_left_alone():
    assert clean("the cat the cat") == "The cat the cat"


def test_repetition_allowlist_contains_the_load_bearing_entries():
    for word in ("had", "that", "very", "really", "so", "no"):
        assert word in REPETITION_ALLOWLIST


# ---------------------------------------------------------------------------
# Sentence casing
# ---------------------------------------------------------------------------


def test_first_word_is_capitalized():
    assert clean("hello world") == "Hello world"


def test_capitalizes_after_a_period():
    assert clean("hello. world") == "Hello. World"


def test_capitalizes_after_a_question_mark():
    assert clean("how are you? fine") == "How are you? Fine"


def test_capitalizes_after_an_exclamation_mark():
    assert clean("stop! now") == "Stop! Now"


def test_capitalizes_across_several_sentences():
    assert (
        clean("hello there. how are you? fine! good")
        == "Hello there. How are you? Fine! Good"
    )


def test_does_not_capitalize_after_a_comma():
    assert clean("hello, world") == "Hello, world"


def test_does_not_recase_a_word_that_already_has_an_inner_capital():
    assert clean("iPhone stays") == "iPhone stays"


def test_does_not_recase_macos():
    assert clean("macOS stays") == "macOS stays"


def test_decimal_number_is_not_treated_as_a_sentence_end():
    assert clean("3.5 is a number") == "3.5 is a number"


def test_abbreviation_is_not_treated_as_a_sentence_end():
    assert clean("e.g. this works") == "E.g. this works"


def test_initials_are_not_treated_as_sentence_ends():
    assert clean("J. R. R. Tolkien wrote") == "J. R. R. Tolkien wrote"


def test_honorific_does_not_end_a_sentence_but_a_real_period_does():
    assert clean("Dr. Smith left. He went home") == "Dr. Smith left. He went home"


def test_casing_happens_after_filler_removal_so_the_surviving_word_is_capitalized():
    # "um" was the first word; "let" must end up capitalized, not "Um" removed
    # and "let" left lowercase.
    assert clean("hmm let me think") == "Let me think"


# ---------------------------------------------------------------------------
# Whitespace normalization and spacing around punctuation
# ---------------------------------------------------------------------------


def test_runs_of_spaces_collapse_to_one():
    assert clean("a  b   c") == "A b c"


def test_leading_and_trailing_whitespace_is_stripped():
    assert clean("   hello   ") == "Hello"


def test_tabs_collapse_to_a_single_space():
    assert clean("tabs\tand more") == "Tabs and more"


def test_space_before_a_comma_is_removed():
    assert clean("hello , world") == "Hello, world"


def test_space_before_a_period_is_removed():
    assert clean("hello world .") == "Hello world."


def test_space_before_a_question_mark_is_removed():
    assert clean("what ? ok") == "What? Ok"


def test_spaces_inside_parentheses_are_tightened():
    assert clean("(  hi  )") == "(Hi)"


def test_messy_spacing_and_punctuation_together():
    assert clean("hello    there  ,  world .") == "Hello there, world."


def test_a_single_newline_is_preserved_as_a_line_break():
    assert clean("first line\nsecond line") == "First line\nSecond line"


# ---------------------------------------------------------------------------
# Custom dictionary
# ---------------------------------------------------------------------------


def test_dictionary_replaces_a_word():
    assert clean("i pushed to github", "light", {"github": "GitHub"}) == (
        "I pushed to GitHub"
    )


def test_dictionary_match_is_case_insensitive_on_the_key():
    assert clean("GITHUB is down", "light", {"github": "GitHub"}) == "GitHub is down"
    assert clean("GitHub is down", "light", {"github": "GitHub"}) == "GitHub is down"
    assert clean("gItHuB is down", "light", {"github": "GitHub"}) == "GitHub is down"


def test_dictionary_value_is_emitted_verbatim_not_recased():
    # The whole point of the dictionary is to fix casing the ASR got wrong.
    # Sentence casing must not undo it into "Github".
    assert clean("github is down", "light", {"github": "GitHub"}) == "GitHub is down"


def test_dictionary_value_keeps_a_lowercase_first_letter_at_sentence_start():
    # "iPhone" must not become "IPhone" just because it starts the sentence.
    assert clean("iphone is nice", "light", {"iphone": "iPhone"}) == "iPhone is nice"


def test_dictionary_value_keeps_its_casing_after_a_period():
    assert clean("hello. github rocks", "light", {"github": "GitHub"}) == (
        "Hello. GitHub rocks"
    )


def test_dictionary_value_that_is_all_caps_survives():
    assert clean("the api call", "light", {"api": "API"}) == "The API call"


def test_dictionary_handles_multi_word_keys():
    assert clean("use vs code today", "light", {"vs code": "VS Code"}) == (
        "Use VS Code today"
    )


def test_dictionary_applies_to_several_entries_in_one_sentence():
    got = clean("check the api and github", "light", {"api": "API", "github": "GitHub"})
    assert got == "Check the API and GitHub"


def test_dictionary_matches_whole_words_only():
    # "api" must not fire inside "rapid".
    assert clean("a rapid change", "light", {"api": "API"}) == "A rapid change"


def test_dictionary_of_none_is_fine():
    assert clean("hello", "light", None) == "Hello"


def test_empty_dictionary_is_fine():
    assert clean("hello", "light", {}) == "Hello"


def test_malformed_dictionary_entries_are_skipped_not_raised():
    # A typo in the user's config must never break dictation.
    bad = {"x": 1, 2: "y", "": "z", "test": "TEST"}
    assert clean("test x", "light", bad) == "TEST x"


# ---------------------------------------------------------------------------
# Standard level: spoken punctuation commands
# ---------------------------------------------------------------------------


def test_standard_turns_spoken_comma_and_period_into_punctuation():
    assert clean("hello comma world period", "standard") == "Hello, world."


def test_standard_handles_question_mark():
    assert clean("wait question mark", "standard") == "Wait?"


def test_standard_handles_exclamation_point():
    assert clean("stop exclamation point", "standard") == "Stop!"


def test_standard_handles_exclamation_mark_as_a_synonym():
    assert clean("stop exclamation mark", "standard") == "Stop!"


def test_standard_new_line_produces_one_newline():
    assert clean("line one new line line two", "standard") == "Line one\nLine two"


def test_standard_new_paragraph_produces_a_blank_line():
    assert clean("para one new paragraph para two", "standard") == (
        "Para one\n\nPara two"
    )


def test_standard_capitalizes_after_a_spoken_period():
    assert clean("first period second", "standard") == "First. Second"


def test_light_level_does_not_apply_spoken_commands():
    # This is why spoken punctuation lives at "standard": at the default level
    # the literal words must survive.
    assert clean("hello comma world period", "light") == "Hello comma world period"


def test_light_level_leaves_new_line_as_words():
    assert clean("first new line second", "light") == "First new line second"


def test_light_level_still_collapses_a_stutter_next_to_the_words_new_line():
    # "new line line two" contains an adjacent duplicate, so rule 3 collapses
    # it even at "light". The spoken command itself is NOT applied: the words
    # "new line" survive as words rather than becoming a newline.
    assert clean("line one new line line two", "light") == "Line one new line two"


def test_spoken_command_matches_whole_words_only():
    # "period" must not fire inside "periodic".
    assert "periodic" in _words(clean("a periodic review", "standard"))


def test_KNOWN_LIMITATION_spoken_period_fires_on_the_noun_period():
    # Documented, accepted cost of supporting spoken punctuation at all: the
    # noun "period" is indistinguishable from the command. This is exactly why
    # the feature is gated behind "standard" and is not the default level.
    assert clean("the period costs money", "standard") == "The. Costs money"
    # At the default level the sentence is safe.
    assert clean("the period costs money", "light") == "The period costs money"


# ---------------------------------------------------------------------------
# Standard level: bounded self-correction
# ---------------------------------------------------------------------------


def test_scratch_that_deletes_the_abandoned_fragment():
    assert clean("send it to bob scratch that send it to alice", "standard") == (
        "Send it to alice"
    )


def test_no_wait_deletes_the_abandoned_fragment():
    assert clean("meet at three no wait meet at four", "standard") == "Meet at four"


def test_i_mean_acts_as_a_correction_marker_at_standard():
    assert clean("the red car i mean the blue car", "standard") == "The blue car"


def test_sorry_acts_as_a_correction_marker_at_standard():
    assert clean("lets go to the park sorry the beach", "standard") == "The beach"


def test_actually_acts_as_a_correction_marker_at_standard():
    # NOTE: this is a different operation from filler removal. "actually" is
    # never DELETED as a noise word; at "standard" it explicitly signals that
    # the user is retracting what they just said.
    assert clean("it is cheap actually it is expensive", "standard") == (
        "It is expensive"
    )


def test_light_level_does_not_self_correct():
    # The default level never deletes on a marker. The user's literal words
    # survive, which is the safe direction to be wrong in.
    assert clean("send it to bob scratch that send it to alice", "light") == (
        "Send it to bob scratch that send it to alice"
    )


def test_correction_stops_at_a_comma_boundary():
    assert clean("first part, drop this scratch that new", "standard") == (
        "First part, new"
    )


def test_correction_stops_at_a_line_break():
    assert clean("line one\nsecond line scratch that ok", "standard") == (
        "Line one\nOk"
    )


def test_correction_is_capped_at_max_correction_tokens():
    # Twelve words before the marker, cap is 8, so the first four survive.
    got = clean(
        "a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 scratch that end", "standard"
    )
    assert got == "A1 a2 a3 a4 end"
    assert MAX_CORRECTION_TOKENS == 8


def test_a_bare_marker_with_nothing_to_correct_is_left_alone():
    # "Actually I think we should go" is discourse, not a retraction. Dropping
    # the marker alone would lose a word and gain nothing.
    assert clean("Actually I think we should go", "standard") == (
        "Actually I think we should go"
    )


def test_i_mean_with_nothing_before_it_is_left_alone():
    assert clean("I mean it", "standard") == "I mean it"


# --- self-correction must never reach across a sentence boundary -----------
#
# This is the hard safety property of rule 7. A correction that eats the
# previous sentence is unrecoverable data loss; one that under-deletes leaves a
# visible fragment the user fixes in one keystroke.


def test_correction_never_deletes_across_a_period():
    got = clean(
        "I finished the report. send it to bob scratch that send it to alice",
        "standard",
    )
    assert got == "I finished the report. Send it to alice"
    assert "I finished the report." in got


def test_correction_never_deletes_across_an_exclamation_mark():
    got = clean("Keep this! drop that scratch that kept", "standard")
    assert got == "Keep this! Kept"
    assert "Keep this!" in got


def test_correction_never_deletes_across_a_question_mark():
    got = clean("Keep this? drop that scratch that kept", "standard")
    assert got == "Keep this? Kept"
    assert "Keep this?" in got


def test_correction_stops_at_the_period_even_with_a_short_second_sentence():
    got = clean("First sentence stays. one two three scratch that done", "standard")
    assert got == "First sentence stays. Done"
    assert "First sentence stays." in got


def test_correction_at_the_start_of_a_later_sentence_deletes_nothing():
    got = clean("I finished the report. Actually lets ship it", "standard")
    assert got == "I finished the report. Actually lets ship it"


def test_correction_never_eats_a_whole_multi_sentence_paragraph():
    got = clean(
        "One is done. Two is done. Three is done. bad bit scratch that good bit",
        "standard",
    )
    assert got == "One is done. Two is done. Three is done. Good bit"
    assert "One is done." in got
    assert "Two is done." in got
    assert "Three is done." in got
    assert "good bit" in got.lower()
    assert "bad bit" not in got.lower()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_string():
    assert clean("") == ""


def test_empty_string_at_every_level():
    assert clean("", "none") == ""
    assert clean("", "light") == ""
    assert clean("", "standard") == ""


def test_whitespace_only_string():
    assert clean("   ") == ""


def test_whitespace_only_string_of_tabs_and_newlines():
    assert clean("\t\n  \n") == ""


def test_single_word():
    assert clean("hello") == "Hello"


def test_single_word_that_is_a_filler_becomes_empty():
    assert clean("um") == ""


def test_only_fillers_becomes_empty():
    assert clean("um uh er") == ""


def test_repeated_single_filler_becomes_empty():
    assert clean("uh uh") == ""


def test_no_punctuation_at_all():
    assert clean("this has no punctuation at all") == (
        "This has no punctuation at all"
    )


def test_punctuation_only_input_does_not_raise():
    # Whatever it returns, it must not blow up.
    assert isinstance(clean("..."), str)
    assert isinstance(clean("!?!?"), str)
    assert isinstance(clean(","), str)


def test_unknown_level_falls_back_to_light_rather_than_raising():
    assert clean("um hello", "bogus") == "Hello"


def test_level_is_case_and_whitespace_insensitive():
    assert clean("um hello", " LIGHT ") == "Hello"
    assert clean("um hello", "None") == "um hello"


def test_non_string_input_returns_empty_rather_than_raising():
    # clean() is on the dictation hot path; it must never take the app down.
    assert clean(None) == ""  # type: ignore[arg-type]
    assert clean(123) == ""  # type: ignore[arg-type]


def test_unicode_survives():
    assert clean("café is naïve") == "Café is naïve"


def test_apostrophes_are_kept_inside_words():
    assert clean("it's the user's text") == "It's the user's text"


def test_hyphenated_words_are_kept_whole():
    assert clean("a well-known problem") == "A well-known problem"


def test_a_long_input_does_not_raise():
    text = "the cat sat on the mat. " * 200
    got = clean(text, "standard")
    assert isinstance(got, str)
    assert got.startswith("The cat sat on the mat.")


@pytest.mark.xfail(
    strict=False,
    reason=(
        "KNOWN WART: removing a filler that stood as its own sentence leaves an "
        "orphaned terminal period -- clean('um. yes') returns '. Yes'. "
        "_tidy_punctuation repairs orphaned commas but not orphaned '.', '!' or "
        "'?'. Cosmetic only (no words are lost), so it is recorded rather than "
        "asserted green."
    ),
)
def test_orphaned_terminal_punctuation_after_filler_removal_is_repaired():
    assert clean("um. yes") == "Yes"
