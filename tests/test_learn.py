"""Tests for the transcript analysis that drives ``blurt learn``.

:mod:`blurt.learn` is a pure function, so these need no fixtures and no macOS --
the same property that makes :mod:`blurt.cleanup` the best-tested module in the
project.

The bar here is set by what a bad suggestion costs. A suggestion the user declines
costs one keystroke. A wrong ``dictionary`` entry silently rewrites a word in every
dictation from then on, and they may never work out which setting did it. So the
false-positive tests below matter more than the true-positive ones, and most of
this file is about what must NOT be suggested.
"""

from __future__ import annotations

import pytest

from blurt.learn import (
    COMMON_WORDS,
    DEFAULT_MIN_OCCURRENCES,
    MAX_PROMPT_WORDS,
    analyze,
    iter_words,
    merged_dictionary,
    merged_prompt,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _Rec:
    """A minimal record. learn.py reads attributes, so it needs nothing more."""

    def __init__(self, raw, cleaned=None, mode="dictate", timestamp=0.0,
                 audio_seconds=1.0, latency_seconds=0.5):
        self.raw = raw
        self.cleaned = cleaned if cleaned is not None else raw
        self.mode = mode
        self.timestamp = timestamp
        self.audio_seconds = audio_seconds
        self.latency_seconds = latency_seconds

    def text(self):
        return self.raw or self.cleaned


def _records(*texts):
    return [_Rec(text, timestamp=float(index)) for index, text in enumerate(texts)]


def _keys(report, kind):
    return [item.key for item in report.by_kind(kind)]


def _suggestion(report, kind, key):
    for item in report.by_kind(kind):
        if item.key == key:
            return item
    return None


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #


def test_first_word_is_sentence_initial():
    assert list(iter_words("Hello world")) == [("Hello", True), ("world", False)]


def test_word_after_a_period_is_sentence_initial():
    words = list(iter_words("one thing. Redis again"))
    assert ("Redis", True) in words


def test_word_after_a_newline_is_sentence_initial():
    words = list(iter_words("first line\nSecond line"))
    assert ("Second", True) in words


def test_mid_sentence_word_is_not_sentence_initial():
    words = list(iter_words("we deployed Redis today"))
    assert ("Redis", False) in words


def test_hyphens_and_apostrophes_stay_inside_a_word():
    assert [word for word, _ in iter_words("don't re-run it")] == ["don't", "re-run", "it"]


def test_iter_words_on_empty_input():
    assert list(iter_words("")) == []
    assert list(iter_words(None)) == []


# --------------------------------------------------------------------------- #
# Empty and degenerate input
# --------------------------------------------------------------------------- #


def test_no_records_gives_an_empty_report():
    report = analyze([])
    assert report.records == 0
    assert report.suggestions == ()


def test_records_with_no_text_are_dropped():
    report = analyze([_Rec("", ""), _Rec("", "")])
    assert report.records == 0


def test_analyze_never_raises_on_junk_arguments():
    report = analyze(_records("hello there"), dictionary="not a dict", initial_prompt=None,
                     min_occurrences="not a number")
    assert report.records == 1


def test_min_occurrences_has_a_floor_of_two():
    """A threshold of one would nominate every word said once. Clamped, not honoured."""
    report = analyze(_records("Grafana dashboards"), min_occurrences=1)
    assert report.by_kind("prompt") == []


# --------------------------------------------------------------------------- #
# Rule 1: spelling variance -> dictionary
# --------------------------------------------------------------------------- #


def test_spelling_variance_becomes_a_dictionary_suggestion():
    report = analyze(
        _records(
            "we shipped GitHub actions",
            "the github token expired",
            "check Github again",
        )
    )
    item = _suggestion(report, "dictionary", "github")
    assert item is not None
    assert item.value == "GitHub"
    assert item.confidence == "high"


def test_variance_evidence_is_in_the_reason():
    report = analyze(
        _records("use GitHub now", "use github now", "use GitHub again")
    )
    item = _suggestion(report, "dictionary", "github")
    assert "GitHub (2)" in item.reason
    assert "github (1)" in item.reason


def test_internal_capital_wins_over_raw_frequency():
    """'macOS' is a spelling the engine would not invent; frequency does not beat it."""
    report = analyze(
        _records(
            "the macos build",
            "another macos build",
            "another macos thing",
            "shipped macOS today",
        )
    )
    item = _suggestion(report, "dictionary", "macos")
    assert item.value == "macOS"


def test_a_consistently_spelled_term_is_not_suggested():
    report = analyze(_records("Redis is up", "Redis is down", "Redis again"))
    assert "redis" not in _keys(report, "dictionary")


def test_sentence_initial_capitals_are_not_variance():
    """Every sentence starts with a capital. That is grammar, not a misrecognition."""
    report = analyze(
        _records(
            "Deploy the thing. deploy it again",
            "Deploy once more",
            "we should deploy now",
        )
    )
    assert "deploy" not in _keys(report, "dictionary")


def test_common_words_are_never_rewritten():
    """{'us': 'US'} would shout at the user in every sentence containing 'us'."""
    report = analyze(
        _records(
            "give it to us now",
            "the US team agreed",
            "between us it is fine",
            "the US again",
        )
    )
    assert "us" not in _keys(report, "dictionary")


def test_every_common_word_is_excluded_from_rewriting():
    texts = []
    for word in list(COMMON_WORDS)[:40]:
        texts.extend(["x %s y" % word, "x %s y" % word.upper(), "x %s y" % word.title()])
    report = analyze(_records(*texts))
    for key in _keys(report, "dictionary"):
        assert key not in COMMON_WORDS


def test_a_term_already_in_the_dictionary_is_not_re_suggested():
    report = analyze(
        _records("GitHub is fine", "github is fine", "Github is fine"),
        dictionary={"github": "GitHub"},
    )
    assert "github" not in _keys(report, "dictionary")


def test_dictionary_key_matching_is_case_insensitive():
    report = analyze(
        _records("GitHub is fine", "github is fine", "Github is fine"),
        dictionary={"GitHub": "GitHub"},
    )
    assert "github" not in _keys(report, "dictionary")


def test_variance_below_the_threshold_is_ignored():
    report = analyze(_records("GitHub here", "github there"), min_occurrences=3)
    assert "github" not in _keys(report, "dictionary")


def test_short_tokens_are_ignored():
    report = analyze(_records("go GO Go", "go GO Go", "go GO Go"))
    assert "go" not in _keys(report, "dictionary")


def test_pure_digits_are_never_terms():
    report = analyze(_records("call 555 now", "call 555 now", "call 555 now"))
    assert _keys(report, "dictionary") == []


def test_suggestion_is_deterministic_across_runs():
    texts = ["Kafka broker", "kafka broker", "KAFKA broker", "Kafka again"]
    first = analyze(_records(*texts)).suggestions
    second = analyze(_records(*texts)).suggestions
    assert first == second


# --------------------------------------------------------------------------- #
# Rule 2: near-misses
# --------------------------------------------------------------------------- #


def test_a_rare_near_miss_of_a_frequent_term_is_flagged():
    report = analyze(
        _records(
            "deploy to kubernetes now",
            "kubernetes is healthy",
            "restart kubernetes please",
            "kubernetes again",
            "check the kubernetis pod",
        )
    )
    item = _suggestion(report, "dictionary", "kubernetis")
    assert item is not None
    assert item.value == "kubernetes"
    assert item.confidence == "medium", "a guess must never be applied unattended"


def test_near_miss_never_reaches_high_confidence():
    report = analyze(
        _records(
            *(["run kubernetes now"] * 6 + ["run kubernetis now"])
        )
    )
    for item in report.by_kind("dictionary"):
        if item.key == "kubernetis":
            assert item.confidence == "medium"


def test_plurals_are_not_collapsed():
    report = analyze(
        _records(
            "the meeting ran long",
            "another meeting today",
            "a third meeting",
            "one more meeting",
            "back to back meetings",
        )
    )
    assert "meetings" not in _keys(report, "dictionary")


def test_past_tense_is_not_collapsed():
    report = analyze(
        _records(
            *(["we deploy on friday"] * 5 + ["we deployed on friday"])
        )
    )
    assert "deployed" not in _keys(report, "dictionary")


def test_short_similar_words_are_not_collapsed():
    """'form' and 'from' are one edit apart and both real."""
    report = analyze(
        _records(*(["read from the file"] * 8 + ["read form the file"]))
    )
    assert "form" not in _keys(report, "dictionary")


def test_different_first_letters_are_not_near_misses():
    report = analyze(
        _records(*(["the postgres cluster"] * 8 + ["the costgres cluster"]))
    )
    assert "costgres" not in _keys(report, "dictionary")


def test_two_frequent_terms_are_not_collapsed_into_each_other():
    report = analyze(
        _records(
            *(["the staging cluster"] * 6 + ["the stading cluster"] * 6)
        )
    )
    assert "stading" not in _keys(report, "dictionary")


def test_a_term_cannot_get_two_conflicting_dictionary_entries():
    report = analyze(
        _records(
            "GitHub actions",
            "github actions",
            "Github actions",
            "gitbub actions",
            "GitHub again",
        )
    )
    keys = _keys(report, "dictionary")
    assert len(keys) == len(set(keys))


# --------------------------------------------------------------------------- #
# Rule 3: vocabulary -> initial_prompt
# --------------------------------------------------------------------------- #


def test_mid_sentence_proper_nouns_become_prompt_suggestions():
    report = analyze(
        _records(
            "ask Priya about the rollout",
            "we should tell Priya",
            "send it to Priya please",
        )
    )
    item = _suggestion(report, "prompt", "priya")
    assert item is not None
    assert item.value == "Priya"
    assert item.confidence == "high"


def test_a_single_mid_sentence_capital_is_only_medium_confidence():
    """One capital is real evidence, but not enough to apply without a human."""
    report = analyze(
        _records(
            "ask Priya about the rollout",
            "Priya is reviewing it",
            "Priya said it is fine",
        )
    )
    item = _suggestion(report, "prompt", "priya")
    assert item is not None
    assert item.confidence == "medium"


def test_a_word_seen_lowercase_mid_sentence_is_not_a_proper_noun():
    """The engine disagreeing with itself about casing is evidence of nothing."""
    report = analyze(
        _records(
            "we ate an Apple today",
            "another apple please",
            "one more apple here",
            "the Apple store",
        )
    )
    assert "apple" not in _keys(report, "prompt")


def test_weekdays_and_months_do_not_consume_prompt_budget():
    report = analyze(
        _records(
            "see you Monday about it",
            "moved to Monday again",
            "and again on Monday",
        )
    )
    assert "monday" not in _keys(report, "prompt")


def test_a_proper_noun_needs_more_than_one_dictation():
    report = analyze([_Rec("we met Priya and Priya and Priya", timestamp=0.0)])
    assert "priya" not in _keys(report, "prompt")


def test_sentence_initial_capitals_do_not_make_a_proper_noun():
    report = analyze(
        _records(
            "Deploy the service",
            "Deploy it again",
            "Deploy once more",
            "Deploy finally",
        )
    )
    assert "deploy" not in _keys(report, "prompt")


def test_jargon_becomes_a_medium_confidence_prompt_suggestion():
    report = analyze(
        _records(
            "the idempotent handler is fine",
            "make it idempotent please",
            "an idempotent retry",
            "unrelated sentence about lunch",
            "another unrelated sentence",
            "a third unrelated one",
        )
    )
    item = _suggestion(report, "prompt", "idempotent")
    assert item is not None
    assert item.confidence == "medium"


def test_a_word_in_almost_every_dictation_is_not_jargon():
    """Whisper already handles your ordinary speech; prompt budget is finite."""
    report = analyze(_records(*(["the release looks fine"] * 10)))
    assert "release" not in _keys(report, "prompt")


def test_common_words_are_never_prompt_suggestions():
    report = analyze(_records(*(["something about the thing"] * 4)))
    for key in _keys(report, "prompt"):
        assert key not in COMMON_WORDS


def test_terms_already_in_the_prompt_are_not_re_suggested():
    report = analyze(
        _records(
            "ask Priya about it",
            "Priya is reviewing",
            "tell Priya please",
        ),
        initial_prompt="Priya, Kubernetes",
    )
    assert "priya" not in _keys(report, "prompt")


def test_prompt_terms_match_case_insensitively():
    report = analyze(
        _records("ask Priya", "Priya again", "and Priya"),
        initial_prompt="priya",
    )
    assert "priya" not in _keys(report, "prompt")


def test_partially_capitalized_terms_are_too_ambiguous_for_the_prompt():
    report = analyze(
        _records(
            "the Widget broke",
            "a widget again",
            "some widget thing",
            "unrelated one",
            "unrelated two",
            "unrelated three",
        )
    )
    assert "widget" not in _keys(report, "prompt")


def test_prompt_suggestions_are_capped():
    texts = []
    for index in range(120):
        name = "Zeta%dton" % index
        texts.extend(["we asked %s today" % name, "and %s replied" % name,
                      "then %s again" % name])
    report = analyze(_records(*texts))
    assert len(report.by_kind("prompt")) <= 40


def test_high_confidence_prompt_suggestions_sort_first():
    report = analyze(
        _records(
            "ask Priya about the idempotent handler",
            "Priya says it is idempotent",
            "Priya and the idempotent retry",
            "some unrelated sentence",
            "another unrelated sentence",
            "a third unrelated sentence",
        )
    )
    confidences = [item.confidence for item in report.by_kind("prompt")]
    assert confidences == sorted(confidences, key=lambda c: 0 if c == "high" else 1)


# --------------------------------------------------------------------------- #
# Rule 4: dictionary health
# --------------------------------------------------------------------------- #


def test_a_key_that_never_appears_is_reported_as_stale():
    report = analyze(_records("hello there", "goodbye now"),
                     dictionary={"grafana": "Grafana"})
    assert "grafana" in report.stale_dictionary_keys


def test_a_key_that_does_appear_is_not_stale():
    report = analyze(_records("open grafana now"), dictionary={"grafana": "Grafana"})
    assert "grafana" not in report.stale_dictionary_keys


def test_a_multi_word_key_is_matched_as_a_sequence():
    report = analyze(_records("the machine learning model"),
                     dictionary={"machine learning": "ML"})
    assert "machine learning" not in report.stale_dictionary_keys


def test_a_multi_word_key_in_the_wrong_order_is_stale():
    report = analyze(_records("learning machine things"),
                     dictionary={"machine learning": "ML"})
    assert "machine learning" in report.stale_dictionary_keys


def test_an_entry_that_rewrites_nothing_is_reported():
    report = analyze(_records("open grafana"), dictionary={"grafana": "grafana"})
    assert "grafana" in report.noop_dictionary_keys
    assert "grafana" not in report.stale_dictionary_keys


# --------------------------------------------------------------------------- #
# Raw availability
# --------------------------------------------------------------------------- #


def test_raw_available_is_true_when_raw_was_kept():
    assert analyze(_records("hello")).raw_available is True


def test_raw_available_is_false_when_only_cleaned_was_kept():
    report = analyze([_Rec("", "Hello there.", timestamp=0.0)])
    assert report.raw_available is False
    assert report.records == 1, "cleaned text is still worth analysing"


def test_cleanup_changed_counts_only_real_differences():
    report = analyze(
        [
            _Rec("um hello there", "Hello there.", timestamp=0.0),
            _Rec("already clean", "already clean", timestamp=1.0),
        ]
    )
    assert report.cleanup_changed == 1


def test_cleanup_changed_is_zero_without_raw():
    report = analyze([_Rec("", "Hello there.", timestamp=0.0)])
    assert report.cleanup_changed == 0


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def test_mode_counts_are_split():
    report = analyze(
        [
            _Rec("hello", mode="dictate", timestamp=0.0),
            _Rec("open safari", mode="assistant", timestamp=1.0),
            _Rec("goodbye", mode="dictate", timestamp=2.0),
        ]
    )
    assert report.dictation_records == 2
    assert report.assistant_records == 1


def test_span_days_is_computed_from_timestamps():
    report = analyze([_Rec("a", timestamp=0.0), _Rec("b", timestamp=86400.0)])
    assert report.span_days == pytest.approx(1.0)


def test_span_days_is_zero_for_a_single_record():
    assert analyze([_Rec("a", timestamp=5.0)]).span_days == 0.0


def test_median_latency_of_an_even_count_averages_the_middle():
    records = [
        _Rec("a", timestamp=0.0, latency_seconds=1.0),
        _Rec("b", timestamp=1.0, latency_seconds=2.0),
        _Rec("c", timestamp=2.0, latency_seconds=3.0),
        _Rec("d", timestamp=3.0, latency_seconds=4.0),
    ]
    assert analyze(records).median_latency_seconds == pytest.approx(2.5)


def test_total_audio_is_summed():
    records = [
        _Rec("a", timestamp=0.0, audio_seconds=2.0),
        _Rec("b", timestamp=1.0, audio_seconds=3.5),
    ]
    assert analyze(records).total_audio_seconds == pytest.approx(5.5)


def test_high_confidence_filter():
    report = analyze(
        _records(
            "ask Priya about GitHub",
            "Priya opened a github issue",
            "Priya and Github again",
        )
    )
    assert all(item.confidence == "high" for item in report.high_confidence())


# --------------------------------------------------------------------------- #
# Merging into config
# --------------------------------------------------------------------------- #


def _dict_suggestion(key, value):
    from blurt.learn import Suggestion

    return Suggestion("dictionary", key, value, "because", 3, "high")


def _prompt_suggestion(key, value):
    from blurt.learn import Suggestion

    return Suggestion("prompt", key, value, "because", 3, "high")


def test_merged_dictionary_adds_new_entries():
    merged = merged_dictionary({}, [_dict_suggestion("github", "GitHub")])
    assert merged == {"github": "GitHub"}


def test_merged_dictionary_never_overwrites_the_user():
    """An entry the user wrote is a stronger statement than anything we inferred."""
    merged = merged_dictionary(
        {"github": "GITHUB"}, [_dict_suggestion("github", "GitHub")]
    )
    assert merged["github"] == "GITHUB"


def test_merged_dictionary_respects_existing_key_casing():
    merged = merged_dictionary(
        {"GitHub": "GITHUB"}, [_dict_suggestion("github", "GitHub")]
    )
    assert merged == {"GitHub": "GITHUB"}


def test_merged_dictionary_ignores_prompt_suggestions():
    merged = merged_dictionary({}, [_prompt_suggestion("priya", "Priya")])
    assert merged == {}


def test_merged_dictionary_does_not_mutate_the_input():
    original = {"a": "A"}
    merged_dictionary(original, [_dict_suggestion("github", "GitHub")])
    assert original == {"a": "A"}


def test_merged_dictionary_tolerates_a_missing_dictionary():
    assert merged_dictionary(None, [_dict_suggestion("x", "X")]) == {"x": "X"}


def test_merged_prompt_from_empty():
    result = merged_prompt("", [_prompt_suggestion("priya", "Priya"),
                                _prompt_suggestion("redis", "Redis")])
    assert result == "Priya, Redis"


def test_merged_prompt_preserves_existing_text():
    result = merged_prompt("We discuss infrastructure", [_prompt_suggestion("redis", "Redis")])
    assert result.startswith("We discuss infrastructure")
    assert "Redis" in result


def test_merged_prompt_does_not_double_punctuate():
    result = merged_prompt("Terms:", [_prompt_suggestion("redis", "Redis")])
    assert result == "Terms: Redis"


def test_merged_prompt_skips_terms_already_present():
    result = merged_prompt("Redis", [_prompt_suggestion("redis", "Redis")])
    assert result == "Redis"


def test_merged_prompt_enforces_the_word_budget():
    suggestions = [_prompt_suggestion("term%d" % i, "Term%d" % i) for i in range(200)]
    result = merged_prompt("", suggestions)
    assert len(result.split()) <= MAX_PROMPT_WORDS


def test_merged_prompt_never_drops_the_users_own_words():
    """An over-budget prompt is left exactly as the user wrote it."""
    existing = " ".join("word%d" % i for i in range(MAX_PROMPT_WORDS + 20))
    result = merged_prompt(existing, [_prompt_suggestion("redis", "Redis")])
    assert result == existing


def test_merged_prompt_ignores_dictionary_suggestions():
    assert merged_prompt("", [_dict_suggestion("github", "GitHub")]) == ""


def test_merged_prompt_with_no_suggestions_is_a_no_op():
    assert merged_prompt("existing text", []) == "existing text"


# --------------------------------------------------------------------------- #
# End to end over a realistic journal
# --------------------------------------------------------------------------- #


def test_a_realistic_journal_produces_useful_and_safe_suggestions():
    journal = _records(
        "let's ask Priya whether the kubernetes rollout is done",
        "the github action failed again on the kubernetes cluster",
        "Priya said the GitHub token expired",
        "I think we should make the handler idempotent",
        "restart kubernetes and tell Priya",
        "the Github issue is still open",
        "an idempotent retry would fix this",
        "so I was thinking we could just wait and see",
        "that is basically what I said in the meeting",
        "the idempotent path is the safe one",
    )
    report = analyze(journal)

    # Found what it should.
    assert _suggestion(report, "dictionary", "github").value == "GitHub"
    assert "priya" in _keys(report, "prompt")

    # Did not touch what it must not.
    dictionary_keys = _keys(report, "dictionary")
    for protected in ("so", "just", "basically", "that", "the", "i"):
        assert protected not in dictionary_keys

    # Every suggestion carries evidence a human can evaluate.
    for item in report.suggestions:
        assert item.reason
        assert item.confidence in ("high", "medium")
        assert item.value


def test_the_realistic_journal_survives_being_applied():
    journal = _records(
        "ask Priya about the kubernetes rollout",
        "the github action failed on kubernetes",
        "Priya said the GitHub token expired",
        "the Github issue is open, ask Priya",
    )
    report = analyze(journal)
    accepted = report.high_confidence()

    dictionary = merged_dictionary({}, accepted)
    prompt = merged_prompt("", accepted)

    # Nothing protected leaked into the dictionary...
    for key in dictionary:
        assert key not in COMMON_WORDS
    # ...and the prompt stayed within budget.
    assert len(prompt.split()) <= MAX_PROMPT_WORDS


def test_applying_twice_is_idempotent():
    """A second learn pass must not re-suggest what the first one already wrote."""
    journal = _records(
        "the github action failed",
        "check GitHub again",
        "Github is down",
    )
    first = analyze(journal)
    dictionary = merged_dictionary({}, first.high_confidence())
    prompt = merged_prompt("", first.high_confidence())

    second = analyze(journal, dictionary=dictionary, initial_prompt=prompt)
    assert _keys(second, "dictionary") == []


def test_default_threshold_is_what_the_cli_advertises():
    assert DEFAULT_MIN_OCCURRENCES == 3
