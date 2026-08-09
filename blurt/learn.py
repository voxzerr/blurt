"""Turn a transcript journal into concrete, reviewable personalization suggestions.

This is the analysis half of blurt's learning loop. :mod:`blurt.history` records
what you said; this module reads those records and proposes ``dictionary`` and
``initial_prompt`` entries. It is a pure function of its inputs -- no I/O, no
clock, no model, no randomness -- for the same reason :mod:`blurt.cleanup` is:
the output is a change to how blurt will transcribe you from now on, and that has
to be auditable.

WHAT CAN AND CANNOT BE LEARNED
------------------------------
Be honest about the limits up front, because the tempting version of this feature
does not exist.

blurt has no ground truth. It knows what the engine heard; it does not know what
you meant. The obvious way to close that gap -- watch what you edit after the text
lands -- would mean reading your keystrokes in other applications, which is
precisely the thing blurt refuses to be. So no rule in this file infers your
intent, and none of them edit anything: every finding is a suggestion a human
approves.

What that leaves is still worth having, because these patterns are visible in the
transcripts alone:

  * **Spelling variance.** The engine wrote ``GitHub`` five times, ``github``
    three times and ``Github`` once. Exactly one of those is what you wanted, and
    a one-line ``dictionary`` entry pins it forever. This needs no guess about
    intent -- the inconsistency is the evidence.
  * **Your proper nouns.** A word the engine capitalizes in the *middle* of a
    sentence is one it believes is a name. Yours -- colleagues, projects, products
    -- are exactly the vocabulary ``initial_prompt`` exists to bias toward.
  * **Your jargon.** Words you use often that most people do not. Same
    destination, weaker evidence, so they are marked as such.
  * **Near-misses.** A rare token two edits from a frequent one is often the
    frequent one, misheard. This is the only genuinely speculative rule here, and
    it is fenced accordingly (§ ``_nearmiss_suggestions``).
  * **Dead dictionary entries.** A key that has never once appeared in your raw
    transcripts is not doing anything, and telling you so is more useful than
    letting it accumulate.

CONFIDENCE IS PART OF THE OUTPUT, NOT A DECORATION
--------------------------------------------------
Every suggestion carries ``high`` or ``medium``. ``blurt learn --apply --yes``
applies only ``high``; ``medium`` requires a human to look at it. That split is
the whole safety model of this module, so a rule's confidence is chosen by how
much it could be wrong rather than by how useful it would be if right.

The governing principle is the one from :mod:`blurt.cleanup`, applied one level
up: a suggestion you decline costs a keystroke; a wrong entry silently rewrites a
word in every dictation you make from then on, and you may not notice which.

WHAT CAN GO WRONG
-----------------
Nothing here touches the OS, the filesystem or the network -- deliberately, so
this module is exhaustively testable with no fixtures on any platform. The real
hazards are analytical, and each is handled where it arises:

  * **Sentence-initial capitals are not evidence.** Every sentence starts with a
    capital, so counting those as proper-noun signal would nominate every common
    word you ever used to start a sentence. Occurrences at a sentence boundary are
    tracked but excluded from the casing rules (:func:`iter_words`).
  * **The dictionary matches case-insensitively.** ``blurt.cleanup`` lowercases
    keys with ``str.lower``, so suggested keys are built with ``str.lower`` too --
    not ``casefold``, which would disagree on non-ASCII and produce an entry that
    silently never fires.
  * **Common words must never become dictionary entries.** ``{"us": "US"}`` would
    shout at the user in every sentence containing "us". Any term whose lowercase
    form is common English is excluded from the rewriting rules outright.
  * **Morphology looks like a typo.** ``meeting``/``meetings`` are two edits
    apart and both correct. The near-miss rule filters known inflections rather
    than proposing to collapse them.
  * **``initial_prompt`` is not free.** Whisper's prompt window is bounded and a
    *mismatched* prompt measurably hurts accuracy. Suggestions are capped and the
    merge helper enforces a word budget instead of letting the prompt grow
    without limit.

Python 3.9 floor: lazy annotations, typing generics only, no PEP 604 unions.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import (
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

__all__ = [
    "Suggestion",
    "Report",
    "analyze",
    "iter_words",
    "merged_dictionary",
    "merged_prompt",
    "COMMON_WORDS",
    "MAX_PROMPT_WORDS",
]


# --------------------------------------------------------------------------- #
# Tuning constants
# --------------------------------------------------------------------------- #

#: Default number of occurrences before a pattern is worth mentioning. Two is too
#: low -- the engine varies on a word once and we would nominate it -- and much
#: higher than three means a new project name takes a week of use to surface.
DEFAULT_MIN_OCCURRENCES = 3

#: A term appearing in more than this share of all dictations is part of your
#: ordinary speech, not specialist vocabulary, whatever the stoplist thinks.
#: Prompting Whisper toward words it already gets right wastes prompt budget.
_JARGON_MAX_DOCUMENT_SHARE = 0.6

#: Shortest term the casing and rewriting rules will consider. Two-letter tokens
#: are overwhelmingly initialisms and pronouns where casing is genuinely variable.
_MIN_TERM_LENGTH = 3

#: Jargon needs more letters than that: short lowercase words are almost all
#: ordinary English that simply missed the stoplist.
_MIN_JARGON_LENGTH = 5

#: Near-miss pairs must be at least this long. Below it, two words an edit apart
#: are usually two different words ("form"/"from", "then"/"than").
_MIN_NEARMISS_LENGTH = 6

#: A distance-2 pair needs to be longer still, because two edits in a short word
#: leaves very little shared evidence.
_MIN_NEARMISS_LENGTH_FOR_TWO = 8

#: A near-miss needs a clearly dominant form: this many occurrences of the
#: frequent spelling...
_NEARMISS_FREQUENT_MIN = 4
#: ...at most this many of the rare one...
_NEARMISS_RARE_MAX = 2
#: ...and this ratio between them. Together these describe "you say it often and
#: it came out wrong once or twice", which is the only shape worth flagging.
_NEARMISS_RATIO = 3

#: Ceiling on how many vocabulary terms a single report will propose. The prompt
#: is a hint, not a lexicon.
MAX_PROMPT_SUGGESTIONS = 40

#: Word budget for the merged ``initial_prompt``. Whisper's prompt window is
#: bounded (224 tokens) and a long prompt crowds out the audio's own context, so
#: this sits well below the hard limit.
MAX_PROMPT_WORDS = 60

#: Suffixes that make two spellings inflections of one word rather than a
#: misrecognition. Checked in both directions.
_INFLECTIONS = ("s", "es", "ed", "d", "ing", "'s", "’s", "ly", "er", "est")


# Ordinary English. Not a lexicon and not trying to be -- it is a guard rail for
# three specific rules that must never fire on a common word, and every entry
# earns its place by being a word whose casing or spelling varies legitimately in
# real writing. Adding to it makes blurt more conservative, which is the safe
# direction; removing from it does not.
COMMON_WORDS: FrozenSet[str] = frozenset(
    """
    a about above after again against all almost also am an and another any anyone
    anything are around as ask at away back be because been before being below
    best better between big both but by call came can cannot come could did do
    does doing done down during each either else end enough even ever every
    everyone everything exactly far few find first for from get give go going
    good got great had half has have having he her here hers herself him himself
    his how however i if in into is it its itself just keep kind know last later
    least left less let like little long look lot made make many may maybe me
    mean might mine more most much must my myself near need never new next nice
    no nobody none nor not nothing now number of off often ok okay old on once one
    only or other others our ours out over own part people perhaps place please
    point probably put quite rather real really right said same saw say see seem
    seen set several shall she should show side since so some someone something
    sometimes soon still such sure take tell than that the their theirs them
    themselves then there therefore these they thing things think this those
    though thought three through time to today together too took two under until
    up upon us use used using usually very want was way we well went were what
    when where whether which while who whole whom whose why will with within
    without won work would yes yet you your yours yourself

    monday tuesday wednesday thursday friday saturday sunday
    january february march april may june july august september october november
    december morning afternoon evening night tomorrow tonight week weekend month
    year day hour minute second

    add answer ask begin break bring build buy call carry change check choose
    close cover create cut decide drop expect explain fall feel fill finish fix
    follow forget grow happen hear help hold hope join keep learn leave listen
    live lose love mean meet miss move open pass pay pick play pull push read
    reach remember remove return run save send share sit sleep sound speak spend
    stand start stay stop suggest talk teach thank throw touch travel try turn
    understand wait walk watch wear win wonder worry write
    """.split()
)


def _is_common(term: str) -> bool:
    """True if ``term`` is ordinary English, allowing for routine inflection.

    ``COMMON_WORDS`` lists base forms, so a bare membership test lets every
    inflection through -- "finished", "looking", "wanted" are all as ordinary as
    the words they come from, and every one of them would otherwise be nominated
    as specialist vocabulary. Stripping the common suffixes before the lookup
    fixes that without turning the list into a lexicon it is explicitly not
    trying to be.

    Errs toward "common", which is the conservative direction everywhere this is
    used: a term wrongly judged ordinary is one we decline to suggest, and a
    declined suggestion costs the user nothing.
    """
    if term in COMMON_WORDS:
        return True
    for suffix in ("s", "es", "ed", "d", "ing", "ly", "er", "est"):
        if len(term) > len(suffix) + 2 and term.endswith(suffix):
            stem = term[: -len(suffix)]
            if stem in COMMON_WORDS:
                return True
            # "running" -> "runn" -> "run": undo a doubled final consonant.
            if len(stem) > 2 and stem[-1] == stem[-2] and stem[:-1] in COMMON_WORDS:
                return True
            # "carries" -> "carri" -> "carry": undo the y -> i shift.
            if stem.endswith("i") and (stem[:-1] + "y") in COMMON_WORDS:
                return True
    return False


# A word: a run of letters/digits, optionally joined by an internal apostrophe or
# hyphen. Matches how ``blurt.cleanup`` splits text, so a suggested dictionary key
# tokenizes the same way there as it was counted here -- if these two disagreed,
# suggestions would be generated that could never match.
_WORD_RE = re.compile(r"[^\W_]+(?:[-'’´][^\W_]+)*")

#: Characters that end a sentence, for deciding whether a capital is evidence.
_TERMINATORS = frozenset(".!?\n\r")


# --------------------------------------------------------------------------- #
# Public data types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Suggestion:
    """One proposed change, with the evidence that produced it.

    ``kind`` is ``"dictionary"`` (add ``key`` -> ``value``) or ``"prompt"`` (add
    ``value`` to the vocabulary hint; ``key`` is its lowercase form). ``reason`` is
    written to be shown to the user verbatim -- it is the whole justification, and
    a suggestion nobody can evaluate is one nobody should accept.
    """

    kind: str            # "dictionary" | "prompt"
    key: str             # lowercase term (dictionary key, or the term's identity)
    value: str           # canonical spelling to write
    reason: str          # human-readable evidence
    occurrences: int     # how many times the term was seen
    confidence: str      # "high" | "medium"


@dataclass(frozen=True)
class Report:
    """Everything :func:`analyze` found, ready to print or apply."""

    records: int = 0
    dictation_records: int = 0
    assistant_records: int = 0
    first_timestamp: float = 0.0
    last_timestamp: float = 0.0
    total_audio_seconds: float = 0.0
    median_latency_seconds: float = 0.0
    #: Records whose cleanup pass changed something. Only meaningful when raw
    #: transcripts were kept; see ``raw_available``.
    cleanup_changed: int = 0
    #: False when every record has an empty ``raw`` (``keep_raw_history=false``).
    #: Several findings are weaker or unavailable in that state and the caller is
    #: expected to say so rather than quietly reporting less.
    raw_available: bool = False
    suggestions: Tuple[Suggestion, ...] = ()
    #: Dictionary keys that never appeared in any transcript.
    stale_dictionary_keys: Tuple[str, ...] = ()
    #: Dictionary entries whose value equals the key -- they rewrite nothing.
    noop_dictionary_keys: Tuple[str, ...] = ()

    @property
    def span_days(self) -> float:
        """Days between the oldest and newest record. 0.0 if fewer than two."""
        if self.records < 2 or self.last_timestamp <= self.first_timestamp:
            return 0.0
        return (self.last_timestamp - self.first_timestamp) / 86400.0

    def by_kind(self, kind: str) -> List[Suggestion]:
        """Suggestions of one kind, in the report's existing order."""
        return [item for item in self.suggestions if item.kind == kind]

    def high_confidence(self) -> List[Suggestion]:
        """The suggestions ``--yes`` is allowed to apply without asking."""
        return [item for item in self.suggestions if item.confidence == "high"]


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #


def iter_words(text: str) -> Iterator[Tuple[str, bool]]:
    """Yield ``(word, sentence_initial)`` for each word in ``text``.

    ``sentence_initial`` is True for the first word and for any word whose
    preceding gap contains a terminator or a line break. Callers use it to discard
    capitals that carry no information: every sentence begins with one, so
    counting them as evidence of a proper noun would nominate the whole language.

    An abbreviation ("e.g. Redis") makes the following word look sentence-initial
    and its capital is therefore ignored. That is the conservative failure: we
    miss evidence rather than invent it.
    """
    if not isinstance(text, str) or not text:
        return

    previous_end = 0
    first = True
    for match in _WORD_RE.finditer(text):
        gap = text[previous_end : match.start()]
        boundary = first or any(char in _TERMINATORS for char in gap)
        yield match.group(), boundary
        previous_end = match.end()
        first = False


def _key(word: str) -> str:
    """The dictionary-matching identity of a word.

    ``str.lower`` rather than ``str.casefold`` on purpose: ``blurt.cleanup``
    lowercases dictionary keys with ``lower``, and a key produced by ``casefold``
    would differ for some non-ASCII input and silently never match.
    """
    return word.lower()


def _is_wordlike(word: str) -> bool:
    """False for pure digits and anything too short to reason about."""
    if len(word) < _MIN_TERM_LENGTH:
        return False
    return not word.isdigit()


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #


@dataclass
class _Index:
    """Counts derived from the journal, keyed by lowercase term."""

    #: term -> Counter of surface spellings, EXCLUDING sentence-initial uses.
    surfaces: Dict[str, "Counter[str]"] = field(default_factory=dict)
    #: term -> occurrences that were sentence-initial (tracked so totals are honest).
    initial: "Counter[str]" = field(default_factory=Counter)
    #: term -> number of records it appeared in at all.
    documents: "Counter[str]" = field(default_factory=Counter)
    #: Every token sequence seen, for checking whether a dictionary key ever fired.
    seen_sequences: Dict[Tuple[str, ...], int] = field(default_factory=dict)
    records: int = 0

    def total(self, term: str) -> int:
        """All occurrences of ``term``, sentence-initial ones included."""
        counter = self.surfaces.get(term)
        return (sum(counter.values()) if counter else 0) + self.initial[term]


def _build_index(records: Sequence["object"], max_key_words: int) -> _Index:
    """Count terms across every record. One pass, no text retained.

    ``max_key_words`` bounds the n-gram width recorded in ``seen_sequences``; it
    is the longest dictionary key we will be asked about, so there is no reason to
    index wider than that.
    """
    index = _Index()

    for record in records:
        text = _record_text(record)
        if not text:
            continue
        index.records += 1

        words = list(iter_words(text))
        present: List[str] = []
        for word, boundary in words:
            if not _is_wordlike(word):
                continue
            term = _key(word)
            present.append(term)
            if boundary:
                # Counted toward totals, but never toward casing evidence.
                index.initial[term] += 1
            else:
                index.surfaces.setdefault(term, Counter())[word] += 1

        for term in set(present):
            index.documents[term] += 1

        if max_key_words > 0:
            _index_sequences(index, present, max_key_words)

    return index


def _index_sequences(index: _Index, terms: Sequence[str], width: int) -> None:
    """Record every n-gram up to ``width`` so key lookups are a dict hit."""
    count = len(terms)
    for start in range(count):
        for size in range(1, min(width, count - start) + 1):
            gram = tuple(terms[start : start + size])
            index.seen_sequences[gram] = index.seen_sequences.get(gram, 0) + 1


def _record_text(record: "object") -> str:
    """Best analysable text for a record, tolerating anything record-shaped."""
    getter = getattr(record, "text", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:  # pragma: no cover - defensive
            value = ""
        if isinstance(value, str):
            return value
    raw = getattr(record, "raw", "") or ""
    cleaned = getattr(record, "cleaned", "") or ""
    return str(raw or cleaned)


def _canonical_surface(counter: "Counter[str]") -> str:
    """Pick the spelling to standardise on.

    Preference order, and each step is doing real work:

    1. A spelling with an INTERNAL capital ("GitHub", "macOS", "PyTorch"). The
       engine does not invent those; if it produced one, it recognised a name.
    2. Otherwise the most frequent spelling.
    3. Ties broken toward the capitalized form, then lexicographically -- so the
       same input always yields the same suggestion, which matters because these
       get written into a config file and diffed.
    """
    if not counter:
        return ""

    def rank(item: Tuple[str, int]) -> Tuple[int, int, int, str]:
        surface, count = item
        inner_capital = any(char.isupper() for char in surface[1:])
        leading_capital = surface[:1].isupper()
        # Negated counts so a plain ascending sort puts the winner first.
        return (0 if inner_capital else 1, -count, 0 if leading_capital else 1, surface)

    return sorted(counter.items(), key=rank)[0][0]


def _describe_spellings(counter: "Counter[str]") -> str:
    """Render the evidence: ``GitHub (5), github (3), Github (1)``."""
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join("%s (%d)" % (surface, count) for surface, count in ordered)


# --------------------------------------------------------------------------- #
# Rule 1: spelling variance -> dictionary (high confidence)
# --------------------------------------------------------------------------- #


def _variant_suggestions(
    index: _Index,
    dictionary_keys: FrozenSet[str],
    min_occurrences: int,
) -> List[Suggestion]:
    """Terms the engine spelled more than one way.

    The strongest rule here, and the only one confident enough to apply
    unattended, because the evidence is an inconsistency rather than an
    inference: whatever the right spelling is, the engine is not producing it
    reliably, and pinning it can only reduce variance.

    Excluded outright: anything whose lowercase form is ordinary English. Casing
    on a common word varies for legitimate reasons we cannot see from here
    (emphasis, a title, a quoted fragment), and ``{"us": "US"}`` would corrupt
    every later dictation containing the word.
    """
    out: List[Suggestion] = []
    for term, counter in index.surfaces.items():
        if term in dictionary_keys or _is_common(term):
            continue
        if len(counter) < 2:
            continue
        total = sum(counter.values())
        if total < min_occurrences:
            continue

        canonical = _canonical_surface(counter)
        if not canonical:
            continue

        out.append(
            Suggestion(
                kind="dictionary",
                key=term,
                value=canonical,
                reason="spelled %d ways: %s"
                % (len(counter), _describe_spellings(counter)),
                occurrences=index.total(term),
                confidence="high",
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Rule 2: near-misses -> dictionary (medium confidence)
# --------------------------------------------------------------------------- #


def _edit_distance(left: str, right: str, maximum: int = 2) -> int:
    """Levenshtein distance, abandoning early once it exceeds ``maximum``.

    Returns ``maximum + 1`` to mean "further than we care about". Bounding it
    matters: this runs over every candidate pair, and the answer is only ever
    compared against a small threshold.
    """
    if left == right:
        return 0
    if abs(len(left) - len(right)) > maximum:
        return maximum + 1

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        best = i
        for j, right_char in enumerate(right, 1):
            cost = 0 if left_char == right_char else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            if value < best:
                best = value
        if best > maximum:
            return maximum + 1
        previous = current
    return previous[-1]


def _is_inflection(left: str, right: str) -> bool:
    """True if one term is a routine grammatical form of the other.

    ``meeting``/``meetings`` and ``deploy``/``deployed`` are one and two edits
    apart respectively and both perfectly correct. Without this check the
    near-miss rule would propose collapsing plurals into singulars, which would
    be wrong in every dictation containing either.
    """
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    if not longer.startswith(shorter):
        # "carries"/"carry" and friends: allow one substituted final character.
        if len(shorter) > 2 and longer.startswith(shorter[:-1]):
            tail = longer[len(shorter) - 1 :]
            return any(tail == suffix or tail[1:] == suffix for suffix in _INFLECTIONS)
        return False
    return longer[len(shorter) :] in _INFLECTIONS


def _nearmiss_suggestions(
    index: _Index,
    dictionary_keys: FrozenSet[str],
) -> List[Suggestion]:
    """A rare spelling within two edits of a frequent one.

    THE SPECULATIVE RULE. Everything else here reports something that is visibly
    true of the transcripts; this one guesses that two similar strings are the
    same word. It is fenced on five axes at once -- length, edit distance,
    absolute counts, the ratio between them, and known inflections -- and it is
    never higher than ``medium``, so ``--yes`` will not apply it.

    The shape it is looking for is narrow and specific: a word you use constantly
    that came out wrong once or twice. A word you have said twice total is not
    evidence of anything, which is why the rare side is capped rather than merely
    outnumbered.
    """
    frequent: Dict[str, List[str]] = {}
    rare: Dict[str, List[str]] = {}
    for term in index.surfaces:
        if _is_common(term) or len(term) < _MIN_NEARMISS_LENGTH:
            continue
        total = index.total(term)
        bucket = term[0]
        if total >= _NEARMISS_FREQUENT_MIN:
            frequent.setdefault(bucket, []).append(term)
        elif total <= _NEARMISS_RARE_MAX:
            rare.setdefault(bucket, []).append(term)

    out: List[Suggestion] = []
    for bucket, rare_terms in rare.items():
        # Same first letter only. A misrecognition that changes the opening sound
        # is not something two edits of distance are evidence for.
        for rare_term in sorted(rare_terms):
            if rare_term in dictionary_keys:
                continue
            rare_count = index.total(rare_term)
            best: Optional[Tuple[int, int, str]] = None
            for frequent_term in frequent.get(bucket, ()):
                if _is_inflection(rare_term, frequent_term):
                    continue
                frequent_count = index.total(frequent_term)
                if frequent_count < rare_count * _NEARMISS_RATIO:
                    continue
                distance = _edit_distance(rare_term, frequent_term)
                if distance == 0 or distance > 2:
                    continue
                if distance == 2 and len(frequent_term) < _MIN_NEARMISS_LENGTH_FOR_TWO:
                    continue
                candidate = (distance, -frequent_count, frequent_term)
                if best is None or candidate < best:
                    best = candidate

            if best is None:
                continue
            distance, negative_count, frequent_term = best
            canonical = _canonical_surface(index.surfaces[frequent_term])
            out.append(
                Suggestion(
                    kind="dictionary",
                    key=rare_term,
                    value=canonical,
                    reason="seen %d time(s); %d edit(s) from %r, which you said %d times"
                    % (rare_count, distance, canonical, -negative_count),
                    occurrences=rare_count,
                    confidence="medium",
                )
            )
    return out


# --------------------------------------------------------------------------- #
# Rule 3: vocabulary -> initial_prompt
# --------------------------------------------------------------------------- #


def _prompt_terms(prompt: str) -> FrozenSet[str]:
    """Lowercase words already present in the prompt, so we never re-suggest them."""
    if not isinstance(prompt, str) or not prompt:
        return frozenset()
    return frozenset(_key(word) for word, _initial in iter_words(prompt))


def _prompt_suggestions(
    index: _Index,
    existing_prompt: str,
    min_occurrences: int,
) -> List[Suggestion]:
    """Vocabulary worth biasing Whisper toward.

    Two rules with genuinely different evidence, and they are scored differently
    for that reason:

    * **Proper nouns (high).** Capitalized away from a sentence boundary, and
      never seen lowercase there. Whisper emits capitalization, so a mid-sentence
      capital is the engine telling us it thinks the word is a name; a mid-sentence
      lowercase of the same word is it disagreeing with itself, which is evidence
      of nothing. Requiring the term across several dictations filters the person
      you mentioned once. A single mid-sentence capital drops the finding to
      ``medium`` -- it is real evidence, but not enough of it to apply unattended.
    * **Jargon (medium).** Lowercase, long enough to be specialist, used across
      several dictations but not most of them. The upper bound is the load-bearing
      part: a word in nearly every transcript is your ordinary speech, and Whisper
      already handles it. Spending prompt budget there displaces the terms that
      actually need help.
    """
    already = _prompt_terms(existing_prompt)
    out: List[Suggestion] = []

    document_ceiling = max(1, int(index.records * _JARGON_MAX_DOCUMENT_SHARE))

    for term, counter in index.surfaces.items():
        if term in already or _is_common(term):
            continue

        # Sentence-initial occurrences are excluded from ``counter`` entirely, so
        # everything counted here is a position where casing carried information.
        mid_capital = sum(
            count for surface, count in counter.items() if surface[:1].isupper()
        )
        mid_lower = sum(counter.values()) - mid_capital
        documents = index.documents[term]

        if mid_capital and not mid_lower and documents >= min_occurrences:
            out.append(
                Suggestion(
                    kind="prompt",
                    key=term,
                    value=_canonical_surface(counter),
                    reason="always capitalized mid-sentence (%d time%s across %d dictations)"
                    % (mid_capital, "" if mid_capital == 1 else "s", documents),
                    occurrences=index.total(term),
                    confidence="high" if mid_capital >= 2 else "medium",
                )
            )
            continue

        if len(term) < _MIN_JARGON_LENGTH:
            continue
        if mid_capital:
            # Some capitals but not enough to call it a name. Ambiguous, and the
            # prompt is too small a budget to spend on ambiguous entries.
            continue
        if documents < min_occurrences or documents > document_ceiling:
            continue

        out.append(
            Suggestion(
                kind="prompt",
                key=term,
                value=_canonical_surface(counter),
                reason="used in %d of %d dictations, and it is not ordinary English"
                % (documents, index.records),
                occurrences=index.total(term),
                confidence="medium",
            )
        )

    out.sort(key=lambda item: (0 if item.confidence == "high" else 1, -item.occurrences, item.key))
    return out[:MAX_PROMPT_SUGGESTIONS]


# --------------------------------------------------------------------------- #
# Rule 4: dictionary hygiene
# --------------------------------------------------------------------------- #


def _dictionary_health(
    index: _Index,
    dictionary: Dict[str, str],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return (keys that never appeared, keys that rewrite nothing).

    "Never appeared" is judged against the transcripts as indexed. When raw
    history is kept that is exactly right -- the dictionary runs *after* the
    engine, so a key that fires still shows up in raw. When it is not kept we are
    reading cleaned text, where a working entry has already been rewritten away
    and would look stale; :attr:`Report.raw_available` exists so the caller can
    say so instead of reporting a falsehood confidently.
    """
    stale: List[str] = []
    noop: List[str] = []

    for key, value in sorted(dictionary.items()):
        tokens = tuple(_key(word) for word, _initial in iter_words(key))
        if not tokens:
            continue
        # Exact comparison, NOT case-insensitive. ``{"grafana": "Grafana"}`` is the
        # single most common legitimate entry there is -- it exists precisely to
        # fix casing -- and folding case here would report every one of them as
        # dead weight and invite the user to delete the entries that work.
        if key.strip() == str(value).strip():
            noop.append(key)
            continue
        if tokens not in index.seen_sequences:
            stale.append(key)

    return tuple(stale), tuple(noop)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def analyze(
    records: Sequence["object"],
    dictionary: Optional[Dict[str, str]] = None,
    initial_prompt: str = "",
    min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
) -> Report:
    """Analyse journal records and return everything worth telling the user.

    Args:
        records: :class:`blurt.history.HistoryRecord` objects, or anything with
            ``raw``/``cleaned`` attributes. Order does not matter.
        dictionary: the user's current replacements, so we never re-suggest an
            entry they already have and can report the ones doing nothing.
        initial_prompt: their current vocabulary hint, for the same reason.
        min_occurrences: evidence threshold. Raising it makes every rule quieter.

    Returns:
        A :class:`Report`. Never raises: an empty or nonsensical journal produces
        an empty report, because this runs from a CLI command whose entire job is
        to explain the situation rather than to fail in it.
    """
    dictionary = dictionary if isinstance(dictionary, dict) else {}
    initial_prompt = initial_prompt if isinstance(initial_prompt, str) else ""
    try:
        min_occurrences = max(2, int(min_occurrences))
    except (TypeError, ValueError):
        min_occurrences = DEFAULT_MIN_OCCURRENCES

    usable = [record for record in records if _record_text(record)]
    if not usable:
        return Report()

    key_width = 1
    for key in dictionary:
        width = len(list(iter_words(key)))
        if width > key_width:
            key_width = width

    index = _build_index(usable, key_width)
    dictionary_keys = frozenset(_key(key) for key in dictionary)

    suggestions: List[Suggestion] = []
    suggestions.extend(
        _variant_suggestions(index, dictionary_keys, min_occurrences)
    )
    # Anything already proposed as a dictionary rewrite is off the table for the
    # near-miss rule, so one term never produces two conflicting entries.
    proposed = dictionary_keys | frozenset(item.key for item in suggestions)
    suggestions.extend(_nearmiss_suggestions(index, proposed))
    suggestions.extend(_prompt_suggestions(index, initial_prompt, min_occurrences))

    suggestions.sort(
        key=lambda item: (
            0 if item.confidence == "high" else 1,
            0 if item.kind == "dictionary" else 1,
            -item.occurrences,
            item.key,
        )
    )

    stale, noop = _dictionary_health(index, dictionary)

    return Report(
        records=len(usable),
        dictation_records=sum(
            1 for record in usable if getattr(record, "mode", "dictate") != "assistant"
        ),
        assistant_records=sum(
            1 for record in usable if getattr(record, "mode", "") == "assistant"
        ),
        first_timestamp=min(float(getattr(r, "timestamp", 0.0) or 0.0) for r in usable),
        last_timestamp=max(float(getattr(r, "timestamp", 0.0) or 0.0) for r in usable),
        total_audio_seconds=sum(
            float(getattr(r, "audio_seconds", 0.0) or 0.0) for r in usable
        ),
        median_latency_seconds=_median(
            [float(getattr(r, "latency_seconds", 0.0) or 0.0) for r in usable]
        ),
        cleanup_changed=sum(
            1
            for r in usable
            if getattr(r, "raw", "")
            and (getattr(r, "raw", "") or "").strip()
            != (getattr(r, "cleaned", "") or "").strip()
        ),
        raw_available=any(getattr(r, "raw", "") for r in usable),
        suggestions=tuple(suggestions),
        stale_dictionary_keys=stale,
        noop_dictionary_keys=noop,
    )


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


# --------------------------------------------------------------------------- #
# Applying suggestions
# --------------------------------------------------------------------------- #


def merged_dictionary(
    existing: Optional[Dict[str, str]],
    accepted: Sequence[Suggestion],
) -> Dict[str, str]:
    """Fold accepted dictionary suggestions into a copy of ``existing``.

    Existing entries always win. A user who has already decided how a word should
    be spelled has made a stronger statement than any pattern we found, and a
    learning pass silently overwriting it would be exactly the kind of surprise
    this whole module is arranged to avoid.
    """
    merged: Dict[str, str] = dict(existing) if isinstance(existing, dict) else {}
    present = frozenset(_key(key) for key in merged)
    for suggestion in accepted:
        if suggestion.kind != "dictionary":
            continue
        if suggestion.key in present or not suggestion.value:
            continue
        merged[suggestion.key] = suggestion.value
    return merged


def merged_prompt(
    existing: str,
    accepted: Sequence[Suggestion],
    max_words: int = MAX_PROMPT_WORDS,
) -> str:
    """Append accepted vocabulary to ``existing``, within a word budget.

    Terms are comma-separated, which is how a bare vocabulary list is usually fed
    to Whisper's ``initial_prompt``, and the user's existing text is preserved
    verbatim ahead of them -- it may be a hand-written sentence whose phrasing is
    doing work.

    The budget is enforced by dropping terms from the end rather than truncating
    mid-word, and existing content is never dropped: if their prompt is already
    over budget, nothing is added and it is left exactly as they wrote it.
    """
    existing = existing.strip() if isinstance(existing, str) else ""
    try:
        budget = max(0, int(max_words))
    except (TypeError, ValueError):
        budget = MAX_PROMPT_WORDS

    used = len([word for word, _initial in iter_words(existing)])
    seen = set(_prompt_terms(existing))

    additions: List[str] = []
    for suggestion in accepted:
        if suggestion.kind != "prompt" or not suggestion.value:
            continue
        if suggestion.key in seen:
            continue
        width = len([word for word, _initial in iter_words(suggestion.value)]) or 1
        if used + width > budget:
            continue
        seen.add(suggestion.key)
        additions.append(suggestion.value)
        used += width

    if not additions:
        return existing
    if not existing:
        return ", ".join(additions)
    separator = " " if existing.endswith((".", "!", "?", ",", ";", ":")) else ". "
    return existing + separator + ", ".join(additions)
