"""Deterministic text cleanup for blurt transcripts.

This is the pass that turns raw ASR output into the text we actually type into
the user's focused application. It is a pure function: no I/O, no globals, no
model, no randomness. The same input always produces the same output, which is
the only reason it can be trusted to run without the user reviewing it first.

The governing principle here is ASYMMETRIC RISK. Failing to clean something is
a small annoyance the user fixes in a second. Deleting a word the user actually
said is a silent corruption they may not notice until it matters. Every rule in
this file is therefore biased toward doing nothing when it is unsure. If you are
tempted to make a rule smarter, make it narrower instead.

Levels:
  "none"     -- strip leading/trailing whitespace, nothing else.
  "light"    -- (default) whitespace, sentence casing, stutter collapse,
                non-lexical filler removal, custom dictionary.
  "standard" -- light, plus spoken punctuation commands and bounded,
                marker-triggered self-correction.

What can go wrong on macOS:
  Nothing here touches the OS, the filesystem, or the network. That is
  deliberate -- this module is the one piece of the pipeline that must be
  trivially unit-testable with no fixtures. The only macOS-adjacent concern is
  downstream: blurt.inject pastes the returned string, and a literal "\\n" in
  the result becomes a Return keypress in some apps (chat clients send the
  message). That is why "new paragraph" only ever produces newlines at
  "standard", never at the default level.

Python 3.9 floor: `from __future__ import annotations` keeps annotations lazy,
and typing.Dict / typing.List / typing.Optional are used rather than PEP 585/604
forms, which are not safe at runtime on 3.9.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

__all__ = [
    "clean",
    "FILLERS",
    "PROTECTED_DISCOURSE_WORDS",
    "REPETITION_ALLOWLIST",
    "SPOKEN_COMMANDS",
    "CORRECTION_MARKERS",
    "MAX_CORRECTION_TOKENS",
]


# ---------------------------------------------------------------------------
# Rule 4 word lists -- READ THIS BEFORE EDITING
# ---------------------------------------------------------------------------

# Non-lexical fillers. These are sounds, not words: they carry no grammatical
# role and no propositional content, so deleting them cannot change what the
# user meant. This list is CLOSED. It is not a starting point.
FILLERS: FrozenSet[str] = frozenset(
    {"um", "uh", "er", "ah", "mm", "hmm", "uhm", "erm"}
)

# =========================================================================
# ***  DO NOT ADD ANY OF THESE TO `FILLERS`.  NOT EVEN "like".  EVER.   ***
# =========================================================================
# Every one of these is grammatically load-bearing in ordinary English, and
# there is no reliable way to tell the "filler" use from the meaning-bearing
# use without a parser we do not have and latency we cannot afford:
#
#   "like"      -- verb ("I like it"), preposition ("like this"), quotative
#                  ("she was like, no"). Deleting it produces nonsense.
#   "so"        -- conjunction ("so I left"), intensifier ("so tired").
#   "right"     -- adjective ("the right answer"), direction, confirmation.
#   "well"      -- adverb ("it works well"), adjective ("I'm not well").
#   "you know"  -- a literal claim about the listener's knowledge.
#   "actually"  -- contrastive; it is often the entire point of the sentence
#                  ("it's actually cheaper"). See the note in rule 7: at
#                  "standard" it can act as a CORRECTION MARKER, which is a
#                  different, explicitly bounded operation -- it is still
#                  never deleted as a filler.
#   "basically" -- hedge that changes the strength of a claim.
#   "I mean"    -- same as "actually"; a marker at "standard", never a filler.
#
# A user who dictates "I mean, I like it" and receives ", it" will stop
# trusting the app, and they will be right to. The whole product rests on the
# user believing we did not touch their words. Guard this list.
PROTECTED_DISCOURSE_WORDS: FrozenSet[str] = frozenset(
    {
        "like",
        "so",
        "right",
        "well",
        "you know",
        "actually",
        "basically",
        "i mean",
    }
)

# Import-time integrity check. This is not defensive programming against user
# input (which must never raise); it is a tripwire against a future edit to
# FILLERS. It fires deterministically at import, so the test suite catches it
# long before a user does.
_CONFLICTS = FILLERS & PROTECTED_DISCOURSE_WORDS
if _CONFLICTS:
    raise RuntimeError(
        "blurt.cleanup: FILLERS contains protected discourse word(s) "
        + repr(sorted(_CONFLICTS))
        + ". These words carry meaning and must never be deleted. "
        "See the comment above PROTECTED_DISCOURSE_WORDS."
    )


# ---------------------------------------------------------------------------
# Rule 3: words that legitimately double
# ---------------------------------------------------------------------------

# Adjacent identical words are usually a disfluency ("the the"), but not
# always. When the doubling is grammatical or idiomatic, collapsing it is a
# silent corruption, so these are left alone. Note the deliberate absence of
# "is": "is is" is a stutter, not the copula doubled.
REPETITION_ALLOWLIST: FrozenSet[str] = frozenset(
    {
        "had",  # past perfect: "he had had enough"
        "that",  # "the fact that that man left"
        "very",  # intensifier reduplication, extremely common in speech
        "really",
        "so",  # "so so" -- mediocre
        "no",  # "a no no"
        "ha",  # laughter
        "bye",
        "night",
        "blah",
        # Spoken digit strings ("five five five") are data, not stutters.
        "zero",
        "oh",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    }
)


# ---------------------------------------------------------------------------
# Rule 6 / 7 tables (standard level only)
# ---------------------------------------------------------------------------

# Spoken punctuation. Each entry is (phrase tokens, replacement, spacing).
# Spacing is "left" (hug the preceding word), "right" (hug the following word),
# or "break" (a hard newline, which absorbs surrounding whitespace).
SPOKEN_COMMANDS: Tuple[Tuple[Tuple[str, ...], str, str], ...] = (
    (("new", "paragraph"), "\n\n", "break"),
    (("new", "line"), "\n", "break"),
    (("question", "mark"), "?", "left"),
    (("exclamation", "point"), "!", "left"),
    (("exclamation", "mark"), "!", "left"),
    (("open", "quote"), '"', "right"),
    (("close", "quote"), '"', "left"),
    (("period",), ".", "left"),
    (("comma",), ",", "left"),
)

# Explicit self-correction markers. Nothing else triggers rule 7: we never try
# to infer that the user changed their mind from the content of the sentence.
CORRECTION_MARKERS: Tuple[Tuple[str, ...], ...] = (
    ("scratch", "that"),
    ("no", "wait"),
    ("i", "mean"),
    ("actually",),
    ("sorry",),
)

# Hard ceiling on how much a single marker may delete. Whichever is SHORTER --
# this or the distance back to the nearest clause boundary -- wins. A wrong
# guess that deletes four words is recoverable; one that eats a whole paragraph
# is not.
MAX_CORRECTION_TOKENS = 8

# When a marker has nothing to delete before it in the current clause, it is
# discourse ("Actually, I think we should go"), not a correction. Dropping the
# marker alone in that case removes information and adds none, so we leave it.
# Set to False to follow the literal "always drop the marker" reading.
_REQUIRE_CONTENT_BEFORE_MARKER = True


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Order matters: breaks before words before punctuation. Every non-whitespace
# character matches exactly one alternative, so the gaps between matches are
# guaranteed to be pure whitespace -- which is what lets us normalize spacing
# by rewriting gaps instead of rebuilding the string from scratch.
_TOKEN_RE = re.compile(
    r"""
      (?P<para>\r?\n[ \t]*\r?\n(?:[ \t\r\n]*))   # blank line -> paragraph break
    | (?P<line>\r?\n)                            # single newline -> line break
    | (?P<word>[^\W_]+(?:[-'‘’´][^\W_]+)*)
    | (?P<punct>[^\s\w]|_)
    """,
    re.VERBOSE,
)

# Punctuation that hugs the word before it. Quotes and apostrophes are
# deliberately absent: a bare ' or " is ambiguous between opening and closing,
# and guessing wrong turns `he said 'hi'` into `he said'hi'`. For those we keep
# whatever spacing the source had.
_TIGHT_LEFT_CHARS = frozenset(".,!?;:)]}%…")

# Punctuation that hugs the word after it.
_TIGHT_RIGHT_CHARS = frozenset("([{")

_TERMINAL_CHARS = frozenset(".!?")
_COMMA_LIKE = frozenset(",;:")
_CLAUSE_BOUNDARY_CHARS = _TERMINAL_CHARS | _COMMA_LIKE

# A period after one of these is an abbreviation, not the end of a sentence, so
# the next word is not capitalized. Kept short on purpose -- each entry costs us
# a missed capital when the word genuinely ends a sentence. "no." is omitted for
# exactly that reason ("I said no. Then I left" is far commoner than "No. 5").
_ABBREVIATIONS: FrozenSet[str] = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "mx",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "mt",
        "vs",
        "etc",
        "inc",
        "ltd",
        "co",
        "corp",
        "dept",
        "univ",
        "approx",
        "fig",
        "eg",
        "ie",
        "al",
    }
)


@dataclass
class _Tok:
    """One token plus the whitespace that followed it in the source.

    Carrying the trailing gap (rather than rebuilding spacing from rules alone)
    means we only ever *normalize* the user's spacing -- we never invent it.
    """

    text: str
    kind: str  # "word" | "punct" | "break"
    gap: str = ""  # raw whitespace that followed this token
    tight_left: bool = False  # render with no space before
    tight_right: bool = False  # render with no space after


def _tokenize(text: str) -> List[_Tok]:
    """Split text into tokens, recording the whitespace between them."""
    toks: List[_Tok] = []
    prev_end = -1
    for match in _TOKEN_RE.finditer(text):
        if toks and prev_end >= 0:
            toks[-1].gap = text[prev_end : match.start()]
        prev_end = match.end()

        if match.lastgroup == "para":
            toks.append(_Tok("\n\n", "break"))
        elif match.lastgroup == "line":
            toks.append(_Tok("\n", "break"))
        elif match.lastgroup == "word":
            toks.append(_Tok(match.group(), "word"))
        else:
            char = match.group()
            toks.append(
                _Tok(
                    char,
                    "punct",
                    tight_left=char in _TIGHT_LEFT_CHARS,
                    tight_right=char in _TIGHT_RIGHT_CHARS,
                )
            )
    if toks:
        toks[-1].gap = ""
    return toks


def _render(toks: Sequence[_Tok]) -> str:
    """Rule 1: emit tokens with collapsed whitespace and no space before punctuation.

    We collapse existing whitespace but never insert whitespace that was not
    there, because inserting it breaks things the tokenizer had to split, like
    "3.5" and "e.g". The one exception is two adjacent words, which can only
    end up touching as a result of a deletion in this module.
    """
    if not toks:
        return ""
    parts: List[str] = []
    last = len(toks) - 1
    for i, tok in enumerate(toks):
        parts.append(tok.text)
        if i == last:
            break
        nxt = toks[i + 1]
        if tok.kind == "break" or nxt.kind == "break":
            sep = ""
        elif nxt.tight_left or tok.tight_right:
            sep = ""
        elif tok.gap:
            sep = " "
        elif tok.kind == "word" and nxt.kind == "word":
            sep = " "  # safety net: never fuse two words together
        else:
            sep = ""
        parts.append(sep)
    return "".join(parts).strip()


# ---------------------------------------------------------------------------
# Small token helpers
# ---------------------------------------------------------------------------


def _word_key(tok: _Tok) -> Optional[str]:
    """Lowercased text if this token is a word, else None."""
    if tok.kind != "word":
        return None
    return tok.text.lower()


def _match_phrase(toks: Sequence[_Tok], i: int, phrase: Sequence[str]) -> bool:
    """True if `phrase` matches whole word tokens starting at index i.

    Matching on tokens rather than on the raw string is what gives us free word
    boundaries: "period" never matches inside "periodic", and "new paragraph"
    only matches when both words are present and adjacent with nothing between.
    """
    if i + len(phrase) > len(toks):
        return False
    for offset, want in enumerate(phrase):
        if _word_key(toks[i + offset]) != want:
            return False
    return True


def _delete_span(toks: List[_Tok], start: int, stop: int) -> None:
    """Remove toks[start:stop], handing the trailing gap back to the previous token.

    Without the gap transfer, deleting "um" from "I um think" would leave "I"
    holding a gap that belongs to a token that no longer exists.
    """
    if start >= stop or start < 0 or stop > len(toks):
        return
    trailing_gap = toks[stop - 1].gap
    del toks[start:stop]
    if start > 0:
        toks[start - 1].gap = trailing_gap


# ---------------------------------------------------------------------------
# Rule 3: adjacent repetition collapse
# ---------------------------------------------------------------------------


def _collapse_repeats(toks: List[_Tok]) -> List[_Tok]:
    """Collapse immediately-adjacent identical words, keeping the first's casing.

    Only true neighbours count. "the, the" and "the cat the" are left alone,
    because a repetition separated by punctuation or another word is far more
    likely to be intentional than a stutter.
    """
    i = 0
    while i + 1 < len(toks):
        first = _word_key(toks[i])
        second = _word_key(toks[i + 1])
        if (
            first is not None
            and first == second
            and first not in REPETITION_ALLOWLIST
            and not first.isdigit()  # "5 5 5" is a phone number, not a stutter
        ):
            _delete_span(toks, i + 1, i + 2)
            continue  # re-check the same index: "the the the" -> "the"
        i += 1
    return toks


# ---------------------------------------------------------------------------
# Rule 4: non-lexical filler removal
# ---------------------------------------------------------------------------


def _strip_fillers(toks: List[_Tok]) -> List[_Tok]:
    """Delete whole-word fillers from the CLOSED list in FILLERS.

    Token-level matching means "um" never fires inside "umbrella" or "hum".
    """
    i = 0
    while i < len(toks):
        key = _word_key(toks[i])
        if key is not None and key in FILLERS:
            _delete_span(toks, i, i + 1)
            continue
        i += 1
    return toks


# ---------------------------------------------------------------------------
# Punctuation tidy-up after deletions
# ---------------------------------------------------------------------------


def _tidy_punctuation(toks: List[_Tok]) -> List[_Tok]:
    """Repair punctuation orphaned by a deletion.

    Removing "um" from "Um, yes" leaves a leading comma; removing it from
    "yes, um, please" leaves a doubled one. Only punctuation adjacent to a
    boundary is touched, so "1,000" survives untouched.
    """
    out: List[_Tok] = []
    for tok in toks:
        if tok.kind == "punct" and tok.text in _COMMA_LIKE:
            if not out:
                continue  # leading comma
            prev = out[-1]
            if prev.kind == "break":
                continue
            if prev.kind == "punct" and prev.text in _CLAUSE_BOUNDARY_CHARS:
                prev.gap = tok.gap  # ",," / ".," -> keep the first
                continue
        if (
            tok.kind == "punct"
            and tok.text in _TERMINAL_CHARS
            and out
            and out[-1].kind == "punct"
            and out[-1].text in _COMMA_LIKE
        ):
            out.pop()  # "hello ,." -> "hello."
        out.append(tok)

    while out and out[-1].kind == "punct" and out[-1].text in _COMMA_LIKE:
        out.pop()
    while out and out[0].kind == "break":
        out.pop(0)
    if out:
        out[-1].gap = ""
    return out


# ---------------------------------------------------------------------------
# Rule 2: sentence casing
# ---------------------------------------------------------------------------


def _is_terminal(toks: Sequence[_Tok], i: int) -> bool:
    """True if toks[i] ends a sentence.

    A period only counts when whitespace (or the end of the text) follows it,
    which is what keeps "3.5" and "e.g" from being read as sentence ends, and
    what makes the first two dots of an ellipsis inert.
    """
    tok = toks[i]
    if tok.kind != "punct" or tok.text not in _TERMINAL_CHARS:
        return False
    if tok.text != ".":
        return True
    if i + 1 < len(toks) and not tok.gap and toks[i + 1].kind != "break":
        return False
    if i > 0 and toks[i - 1].kind == "word":
        prev = toks[i - 1].text.lower()
        if prev in _ABBREVIATIONS:
            return False
        if len(prev) == 1 and prev.isalpha():
            return False  # initials: "J. R. R. Tolkien"
    return True


def _capitalize_first_alpha(word: str) -> str:
    """Uppercase the first alphabetic character, unless the word is already cased.

    Any existing inner capital means the word's casing is intentional, so we
    leave it: "iPhone" must not become "IPhone", nor "macOS" become "MacOS".
    """
    for idx, char in enumerate(word):
        if not char.isalpha():
            continue
        if char.isupper():
            return word
        if any(c.isupper() for c in word[idx + 1 :]):
            return word
        return word[:idx] + char.upper() + word[idx + 1 :]
    return word


def _apply_sentence_case(toks: List[_Tok]) -> List[_Tok]:
    """Capitalize the first word of the text and of every following sentence."""
    at_start = True
    for i, tok in enumerate(toks):
        if tok.kind == "word":
            if at_start:
                tok.text = _capitalize_first_alpha(tok.text)
                at_start = False
        elif tok.kind == "break":
            at_start = True
        elif _is_terminal(toks, i):
            at_start = True
    return toks


# ---------------------------------------------------------------------------
# Rule 5: custom dictionary
# ---------------------------------------------------------------------------


def _prepare_dictionary(
    dictionary: Optional[Dict[str, str]]
) -> List[Tuple[Tuple[str, ...], str]]:
    """Normalize the user dictionary into (key tokens, value) pairs, longest first.

    Entries that are not str -> str, or whose key contains no word characters,
    are skipped rather than raised on: a typo in the config must not break
    dictation.
    """
    if not isinstance(dictionary, dict):
        return []
    prepared: List[Tuple[Tuple[str, ...], str]] = []
    for key, value in dictionary.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        key_tokens = tuple(
            t.text.lower() for t in _tokenize(key.strip()) if t.kind == "word"
        )
        if key_tokens:
            prepared.append((key_tokens, value))
    prepared.sort(key=lambda pair: len(pair[0]), reverse=True)
    return prepared


def _apply_dictionary(
    toks: List[_Tok], entries: Sequence[Tuple[Tuple[str, ...], str]]
) -> List[_Tok]:
    """Case-insensitive whole-word replacement, emitting the value verbatim.

    Runs last so the user's dictionary always wins: an entry of
    {"github": "GitHub"} survives sentence casing rather than being re-cased
    into "Github".
    """
    if not entries:
        return toks
    i = 0
    while i < len(toks):
        if toks[i].kind != "word":
            i += 1
            continue
        for key_tokens, value in entries:
            if not _match_phrase(toks, i, key_tokens):
                continue
            span = len(key_tokens)
            toks[i].text = value
            toks[i].gap = toks[i + span - 1].gap
            if span > 1:
                del toks[i + 1 : i + span]
            break
        i += 1
    return toks


# ---------------------------------------------------------------------------
# Rule 6: spoken punctuation commands (standard only)
# ---------------------------------------------------------------------------


def _apply_spoken_commands(toks: List[_Tok]) -> List[_Tok]:
    """Turn spoken punctuation into real punctuation.

    Whole-token matching is the only guard we have here, and it is not a
    complete one: "the Victorian period is over" contains the token "period"
    and will be rewritten. That is the accepted cost of supporting spoken
    punctuation at all, and it is why this rule lives at "standard" rather than
    in the default level.
    """
    i = 0
    while i < len(toks):
        for phrase, replacement, spacing in SPOKEN_COMMANDS:
            if not _match_phrase(toks, i, phrase):
                continue
            span = len(phrase)
            tail_gap = toks[i + span - 1].gap
            if span > 1:
                del toks[i + 1 : i + span]
            tok = toks[i]
            tok.text = replacement
            tok.gap = tail_gap
            if spacing == "break":
                tok.kind = "break"
                tok.tight_left = False
                tok.tight_right = False
            else:
                tok.kind = "punct"
                tok.tight_left = spacing == "left"
                tok.tight_right = spacing == "right"
            break
        # Advance past the (possibly replaced) token either way; the
        # replacement itself must never be re-scanned.
        i += 1
    return toks


# ---------------------------------------------------------------------------
# Rule 7: bounded, marker-triggered self-correction (standard only)
# ---------------------------------------------------------------------------


def _marker_length(toks: Sequence[_Tok], i: int) -> int:
    """Length in tokens of the correction marker starting at i, or 0."""
    for phrase in CORRECTION_MARKERS:
        if _match_phrase(toks, i, phrase):
            return len(phrase)
    return 0


def _apply_self_correction(toks: List[_Tok]) -> List[_Tok]:
    """Delete the abandoned fragment before an explicit correction marker.

    The extent is deliberately crippled: we walk backwards only to the nearest
    clause boundary (comma, semicolon, colon, terminal punctuation, or a line
    break) and never more than MAX_CORRECTION_TOKENS tokens, whichever comes
    first. Sentence boundaries are hard stops, so a correction can never reach
    into the previous sentence.

    Under-deleting leaves the user a visible fragment they can fix in one
    keystroke; over-deleting silently destroys text they will never get back.
    """
    i = 0
    while i < len(toks):
        span = _marker_length(toks, i)
        if not span:
            i += 1
            continue

        start = i
        removed = 0
        j = i - 1
        while j >= 0 and removed < MAX_CORRECTION_TOKENS:
            prev = toks[j]
            if prev.kind == "break":
                break
            if prev.kind == "punct" and prev.text in _CLAUSE_BOUNDARY_CHARS:
                break
            start = j
            removed += 1
            j -= 1

        if removed == 0 and _REQUIRE_CONTENT_BEFORE_MARKER:
            # Nothing to correct: this is discourse ("Actually, I think..."),
            # not a retraction. Dropping the marker alone would only lose a
            # word the user said, so leave the whole thing intact.
            i += span
            continue

        _delete_span(toks, start, i + span)
        i = start
    return toks


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def clean(
    text: str, level: str = "light", dictionary: Optional[Dict[str, str]] = None
) -> str:
    """Clean up a raw transcript.

    Args:
        text: raw ASR output. Empty, whitespace-only, and unpunctuated input
            are all normal and handled.
        level: "none", "light" (default), or "standard". An unrecognized level
            falls back to "light" rather than raising; blurt.config already
            validates this value, so an unknown one means a caller bug, not
            user input.
        dictionary: optional case-insensitive replacements, e.g.
            {"github": "GitHub"}. Values are emitted verbatim. Malformed
            entries are skipped.

    Returns:
        The cleaned text. Never raises: any unexpected failure degrades to the
        stripped input, because returning the user's words unimproved is always
        better than losing the dictation.
    """
    if not isinstance(text, str):
        return ""
    try:
        return _clean_impl(text, level, dictionary)
    except Exception:  # pragma: no cover - last-resort safety net
        try:
            return text.strip()
        except Exception:
            return ""


def _clean_impl(
    text: str, level: str, dictionary: Optional[Dict[str, str]]
) -> str:
    stripped = text.strip()
    if not stripped:
        return ""

    normalized_level = level.strip().lower() if isinstance(level, str) else "light"
    if normalized_level == "none":
        return stripped
    if normalized_level not in ("light", "standard"):
        normalized_level = "light"
    standard = normalized_level == "standard"

    toks = _tokenize(stripped)
    if not toks:
        return stripped

    # Order is load-bearing. Commands run first so the punctuation they create
    # is visible to self-correction (as clause boundaries) and to sentence
    # casing. Deletions run before the tidy-up that repairs their fallout.
    # Casing runs after every deletion so the surviving first word is the one
    # that gets capitalized. The dictionary runs last so it always wins.
    if standard:
        toks = _apply_spoken_commands(toks)
        toks = _apply_self_correction(toks)
    toks = _collapse_repeats(toks)
    toks = _strip_fillers(toks)
    toks = _tidy_punctuation(toks)
    toks = _apply_sentence_case(toks)
    toks = _apply_dictionary(toks, _prepare_dictionary(dictionary))
    return _render(toks)
