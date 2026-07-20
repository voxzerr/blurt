"""Natural-language date/time parsing for spoken blurt commands.

This is the hardest pure-logic module in the assistant, because it has to bridge
two messy worlds: how people actually *say* times out loud, and how Whisper
*transcribes* those times. The same spoken "two thirty" can land in the text as
``2:30``, ``2.30``, ``230`` or ``two thirty``; "three p.m." can arrive as
``3pm``, ``3 p.m.`` or ``3 PM``. :func:`parse_when` is the single entry point and
tries to be forgiving of all of that.

It is deliberately pure: standard-library :mod:`datetime` / :mod:`re` only, no
pyobjc, EventKit or AppKit imports anywhere, so it can be unit-tested on any
machine (Intel or Apple Silicon, any macOS, offline). ``now`` is always injected
by the caller and this module NEVER calls :func:`datetime.datetime.now` -- that
keeps tests deterministic and satisfies the workflow rule against ambient clocks.

------------------------------------------------------------------------------
JUDGMENT CALLS (a reviewer should read these -- they are choices, not facts)
------------------------------------------------------------------------------

AM/PM inference for a bare hour ("at 3", no am/pm, no separator):
  There is no correct answer, only a sensible default. The rule is:

    * A "daypart" hint taken from the sentence wins when present:
        - if the sentence mentions "morning"      -> bare hours stay AM
        - if it mentions "afternoon"/"evening"/
          "tonight"/"night"                        -> bare hours become PM
      So "tonight at 8" -> 20:00, "tomorrow morning at 8" -> 08:00.
    * With no hint, small numbers are read as afternoon appointments:
        - 1..6   -> PM  (1 -> 13:00 ... 6 -> 18:00)
        - 7..11  -> AM  (7 -> 07:00 ... 11 -> 11:00)
        - 12     -> 12:00 noon (unless an AM hint makes it 00:00)
      This matches how people schedule: "meet at 3" almost always means 3 PM,
      "call me at 9" almost always means 9 AM.
    * Explicit am/pm ("3pm", "at 3 a.m.") always overrides the heuristic.

Rolling a bare time into the future:
  If a clock time is given with NO date ("at 3", "at 2:30 pm") and that time has
  already passed today relative to ``now``, it rolls forward to the same time
  TOMORROW (a whole-day roll -- we do not flip AM/PM to stay today). Reminders
  and appointments are about the future, so "at 9" said at 10 AM means tomorrow
  9 AM, not five minutes ago. Times that come attached to an explicit date
  ("today", "tonight", "monday", "tomorrow") are NOT rolled -- an explicit date
  is taken at face value even if it lands in the past.

Weekday names ("monday", "next friday", "this saturday"):
  Let W be the named weekday and T = today's weekday.
    * "monday" / "this monday" / "on monday" / "coming monday":
        the nearest occurrence -- days_ahead = (W - T) % 7.
        If today already IS that weekday, that means TODAY (0 days).
    * "next monday":
        the same nearest occurrence, EXCEPT when today already IS that weekday,
        in which case it means a week out (7 days), never today.
  So on a Tuesday, "friday" and "next friday" both mean this coming Friday; the
  two only diverge when today is the named day. This follows the spec's stated
  rule ("next = the coming one; if today is that weekday, next X = a week out").

"tonight" / morning / afternoon / evening:
  "tonight"           -> today at 19:00 (has_time=True; anchored to today, no roll)
  "morning"           -> 09:00
  "afternoon"         -> 14:00
  "evening"/"night"   -> 19:00
  These carry a time-of-day intent, so has_time is True (a specific hour is
  scheduled) rather than an all-day event.

"next week" / "next month" / "next year":
  Weeks start on MONDAY. "next week" -> the coming Monday, all-day (09:00,
  has_time=False). "next month" -> the 1st of next month, all-day. "next year"
  -> January 1st of next year, all-day.

Date without a clock time ("tomorrow", "monday", "next week"):
  Returns ``has_time=False`` at a default hour of 09:00 so the caller can create
  an all-day event if it wants. The 09:00 is a placeholder, not a claimed time.

Duration ("for 30 minutes", "for an hour"):
  Fills ``duration_minutes``. Duration only rides along with a resolved start --
  a bare duration with no "when" ("for 30 minutes" on its own) returns ``None``
  from :func:`parse_when`, because a duration is not a point in time. (Handlers
  that want a bare duration, e.g. a timer, parse it themselves.)

------------------------------------------------------------------------------
What can go wrong
------------------------------------------------------------------------------
:func:`parse_when` is wrapped so it NEVER raises: any malformed input, impossible
date (Feb 30), or internal slip returns ``None`` -- the app then says "I didn't
understand" rather than showing a traceback. If ``now`` is timezone-aware the
returned ``start`` carries the same tzinfo; if ``now`` is naive the result is
naive. The parser is intentionally boring and rule-based, not clever: it favours
predictable, documented behaviour over guessing.
"""
from __future__ import annotations

import calendar
import datetime
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ParsedWhen:
    """A resolved date/time extracted from spoken text.

    * ``start`` -- the absolute datetime the expression resolves to. Naive or
      timezone-aware to match the injected ``now``.
    * ``has_time`` -- ``True`` when an actual clock time was stated (or clearly
      implied, e.g. "tonight", "at 3", "in 20 minutes"). ``False`` when only a
      date was given, so the caller may create an all-day event; in that case
      ``start`` uses a default hour (09:00) purely as a placeholder.
    * ``duration_minutes`` -- ``None`` unless a duration ("for 30 minutes") was
      stated.
    """

    start: datetime.datetime
    has_time: bool
    duration_minutes: Optional[int]


# --------------------------------------------------------------------------- #
# Word-number vocabulary (written out, as the spec asks)
# --------------------------------------------------------------------------- #
# Values 0..19 plus the tens, used for spoken clock minutes/hours and for
# spelled-out relative amounts. "quarter" and "half" map to their clock-minute
# meaning (15 and 30); their *fractional* meaning (0.25 / 0.5) for durations is
# handled separately in _RELATIVE_AMOUNT.
_ONES: Dict[str, int] = {
    "zero": 0, "oh": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
_TEENS: Dict[str, int] = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS: Dict[str, int] = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
}
# Single-token numbers usable as a minute/hour fragment (0..19 + tens + clock
# words). Combined tens+ones (e.g. "twenty three") are handled in
# _small_number_from_words.
_SIMPLE_NUM: Dict[str, int] = {}
_SIMPLE_NUM.update(_ONES)
_SIMPLE_NUM.update(_TEENS)
_SIMPLE_NUM.update(_TENS)
_SIMPLE_NUM["quarter"] = 15
_SIMPLE_NUM["half"] = 30

# Hours that can be spoken as words ("at eleven", "two thirty").
_HOUR_WORDS: Dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

# Amounts for relative/duration phrases ("in two hours", "for half an hour").
# Here "half"/"quarter" mean fractions of the following unit, not clock minutes.
_RELATIVE_AMOUNT: Dict[str, float] = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "forty-five": 45, "fifty": 50, "sixty": 60, "ninety": 90,
    "half": 0.5, "quarter": 0.25,
}

_WEEKDAYS: Dict[str, int] = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

# Default hours for the various dayparts and for the date-only placeholder.
_HOUR_MORNING = 9
_HOUR_AFTERNOON = 14
_HOUR_EVENING = 19
_HOUR_NOON = 12
_HOUR_MIDNIGHT = 0
_DEFAULT_DATE_HOUR = 9


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def parse_when(text: str, now: datetime.datetime) -> Optional[ParsedWhen]:
    """Parse a date/time out of spoken ``text``, resolved against ``now``.

    Returns a :class:`ParsedWhen`, or ``None`` when there is genuinely no
    date/time expression (or the input is unusable). Never raises: any error is
    swallowed and reported as ``None``. ``now`` is injected by the caller so the
    result is deterministic; this function does not read the system clock.
    """
    try:
        if not text or not isinstance(text, str):
            return None

        t = _normalize(text)
        if not t:
            return None

        hint = _daypart_hint(t)

        # Duration ("for 30 minutes") is stripped first so it can't be mistaken
        # for a clock time later; it rides along with whatever start we resolve.
        duration, t = _extract_duration(t)

        # Relative offsets ("in 20 minutes") fully determine the start on their
        # own -- return immediately, carrying any duration we found.
        rel = _parse_relative(t, now)
        if rel is not None:
            start = _match_tz(rel, now)
            return ParsedWhen(start=start, has_time=True,
                              duration_minutes=duration)

        # Otherwise: pull out a date anchor, then a clock time from the rest.
        date_res, t = _extract_date(t, now)
        time_res, t = _extract_time(t, hint)

        if date_res is None and time_res is None:
            # Nothing temporal (a lone duration is not a "when").
            return None

        # Resolve the date component.
        if date_res is not None:
            base_date, forced_time = date_res
            has_date = True
        else:
            base_date, forced_time = now.date(), None
            has_date = False

        # Resolve the time component: explicit clock time > forced daypart time
        # (e.g. "tonight") > date-only placeholder.
        if time_res is not None:
            hour, minute = time_res
            has_time = True
        elif forced_time is not None:
            hour, minute = forced_time
            has_time = True
        else:
            hour, minute = _DEFAULT_DATE_HOUR, 0
            has_time = False

        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None

        start = datetime.datetime.combine(
            base_date, datetime.time(hour=hour, minute=minute)
        )
        start = _match_tz(start, now)

        # Roll a bare time (no explicit date) forward if it already passed.
        if has_time and not has_date and start <= now:
            start = start + datetime.timedelta(days=1)

        return ParsedWhen(start=start, has_time=has_time,
                          duration_minutes=duration)
    except Exception:
        # Contract: never raise on bad user speech.
        return None


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def _normalize(text: str) -> str:
    """Lower-case ``text`` and smooth over Whisper/punctuation quirks.

    Keeps ``:`` and ``.`` between digits (they are time separators) but strips
    sentence punctuation, folds "p.m."/"a. m." to "pm"/"am", normalizes a few
    multi-word amount phrases ("half an hour" -> "30 minutes"), and collapses
    whitespace. Never touches digit groups themselves.
    """
    t = text.lower().strip()
    t = t.replace("’", "'").replace("‘", "'")  # curly quotes -> '

    # "p.m." / "a. m." / "p m" (with a dot somewhere) -> "pm" / "am".
    t = re.sub(r"\b([ap])\.\s*m\.?", r"\1m", t)
    # "o'clock" / "oclock" -> a stable marker word.
    t = re.sub(r"\bo['\s]?clock\b", " oclock ", t)

    # Multi-word amount phrases -> plain "<n> <unit>" so the numeric relative and
    # duration parsers can handle them uniformly. Order matters (longest first).
    _phrase_subs = [
        (r"\ban hour and a half\b", "90 minutes"),
        (r"\ba couple of\b", "2"),
        (r"\ba couple\b", "2"),
        (r"\bcouple of\b", "2"),
        (r"\ba few\b", "3"),
        (r"\bhalf an hour\b", "30 minutes"),
        (r"\ba half hour\b", "30 minutes"),
        (r"\bhalf hour\b", "30 minutes"),
        (r"\bquarter of an hour\b", "15 minutes"),
        (r"\ba quarter hour\b", "15 minutes"),
        (r"\bquarter hour\b", "15 minutes"),
    ]
    for pat, repl in _phrase_subs:
        t = re.sub(pat, repl, t)

    # Drop sentence punctuation but preserve '.'/':' (time separators) and "'".
    t = re.sub(r"[,!?;\"()]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _daypart_hint(t: str) -> Optional[str]:
    """Return 'am', 'pm', or None to bias bare-hour AM/PM inference.

    "morning" leans AM; "afternoon"/"evening"/"tonight"/"night" lean PM. If both
    a morning and an evening word appear, morning wins (checked first) -- an
    arbitrary but stable tie-break.
    """
    if re.search(r"\bmorning\b", t):
        return "am"
    if re.search(r"\b(afternoon|evening|tonight|night)\b", t):
        return "pm"
    return None


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _match_tz(dt: datetime.datetime, now: datetime.datetime) -> datetime.datetime:
    """Give ``dt`` the same tzinfo as ``now`` (keeps comparisons legal)."""
    if now.tzinfo is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=now.tzinfo)
    return dt


def _small_number_from_words(words: List[str]) -> Optional[int]:
    """Turn 1-2 number words into an int 0..59, or None if they are not numbers.

    Handles single tokens ("thirty" -> 30, "quarter" -> 15) and tens+ones
    ("twenty three" -> 23, "forty five" -> 45).
    """
    if not words:
        return None
    if len(words) == 1:
        return _SIMPLE_NUM.get(words[0])
    if len(words) == 2:
        tens = _TENS.get(words[0])
        ones = _ONES.get(words[1])
        if tens is not None and ones is not None and 1 <= ones <= 9:
            return tens + ones
    return None


def _hour_token(tok: str) -> Optional[int]:
    """Return a 1..12 hour from a spelled ("three") or digit ("3") token."""
    val = _HOUR_WORDS.get(tok)
    if val is not None:
        return val
    if tok.isdigit():
        num = int(tok)
        if 1 <= num <= 12:
            return num
    return None


def _infer_hour(hour: int, hint: Optional[str]) -> int:
    """Resolve a bare (no am/pm) hour to 24h form. See module docstring."""
    if hour == 12:
        return _HOUR_MIDNIGHT if hint == "am" else _HOUR_NOON
    if hint == "pm":
        return hour + 12 if 1 <= hour <= 11 else hour
    if hint == "am":
        return hour  # 0..11 stay as-is
    # No hint: small numbers read as afternoon appointments.
    if 1 <= hour <= 6:
        return hour + 12
    return hour  # 0, 7..11, and already-24h values pass through


def _apply_ampm(hour: int, ampm: Optional[str], hint: Optional[str]) -> int:
    """Combine an explicit am/pm marker with a 1..12 hour, else infer."""
    if ampm == "pm":
        return hour + 12 if hour < 12 else 12
    if ampm == "am":
        return 0 if hour == 12 else hour
    return _infer_hour(hour, hint)


def _cut(t: str, span: Tuple[int, int]) -> str:
    """Remove a matched span from ``t`` and tidy whitespace."""
    out = t[: span[0]] + " " + t[span[1] :]
    return re.sub(r"\s+", " ", out).strip()


def _add_months(dt: datetime.datetime, n: int) -> datetime.datetime:
    """Add ``n`` calendar months, clamping the day to the target month length."""
    month_index = dt.month - 1 + n
    year = dt.year + month_index // 12
    month = month_index % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    return dt.replace(year=year, month=month, day=min(dt.day, last_day))


# --------------------------------------------------------------------------- #
# Duration:  "for 30 minutes" / "for an hour"
# --------------------------------------------------------------------------- #
_DURATION_RE = re.compile(
    r"\bfor\s+(?:(\d+(?:\.\d+)?)|([a-z]+))\s+(?:of\s+)?(?:an?\s+)?"
    r"(minutes?|mins?|hours?|hrs?)\b"
)


def _extract_duration(t: str) -> Tuple[Optional[int], str]:
    """Pull a "for <amount> <unit>" duration out of ``t``.

    Returns ``(duration_minutes or None, text_with_duration_removed)``. Only
    minutes/hours count as a duration; anything else is left in place.
    """
    m = _DURATION_RE.search(t)
    if not m:
        return None, t
    amount = _amount_value(m.group(1), m.group(2))
    if amount is None or amount <= 0:
        return None, t
    unit = m.group(3)
    minutes = amount * 60 if unit.startswith(("hour", "hr")) else amount
    minutes_int = int(round(minutes))
    if minutes_int <= 0:
        return None, t
    return minutes_int, _cut(t, m.span())


def _amount_value(digit: Optional[str], word: Optional[str]) -> Optional[float]:
    """Resolve a captured amount to a float, from a digit group or a word."""
    if digit is not None:
        try:
            return float(digit)
        except ValueError:
            return None
    if word is not None:
        return _RELATIVE_AMOUNT.get(word)
    return None


# --------------------------------------------------------------------------- #
# Relative offsets:  "in 20 minutes" / "in 2 hours" / "in 3 days"
# --------------------------------------------------------------------------- #
_RELATIVE_RE = re.compile(
    r"\bin\s+(?:(\d+(?:\.\d+)?)|([a-z]+))\s+(?:of\s+)?(?:an?\s+)?"
    r"(minutes?|mins?|hours?|hrs?|days?|weeks?|months?)\b"
)


def _parse_relative(t: str, now: datetime.datetime) -> Optional[datetime.datetime]:
    """Resolve an "in <amount> <unit>" offset from ``now``, or None."""
    m = _RELATIVE_RE.search(t)
    if not m:
        return None
    amount = _amount_value(m.group(1), m.group(2))
    if amount is None or amount <= 0:
        return None
    unit = m.group(3)
    if unit.startswith("month"):
        return _add_months(now, int(round(amount)))
    if unit.startswith(("minute", "min")):
        delta = datetime.timedelta(minutes=amount)
    elif unit.startswith(("hour", "hr")):
        delta = datetime.timedelta(hours=amount)
    elif unit.startswith("day"):
        delta = datetime.timedelta(days=amount)
    elif unit.startswith("week"):
        delta = datetime.timedelta(weeks=amount)
    else:
        return None
    return now + delta


# --------------------------------------------------------------------------- #
# Date anchors:  today / tomorrow / weekdays / next week ...
# --------------------------------------------------------------------------- #
_DateResult = Tuple[datetime.date, Optional[Tuple[int, int]]]
# (resolved date, optional forced (hour, minute) e.g. "tonight" -> (19, 0))


def _extract_date(
    t: str, now: datetime.datetime
) -> Tuple[Optional[_DateResult], str]:
    """Find a date anchor in ``t`` and return it plus the leftover text.

    The returned tuple's first element is ``(date, forced_time)`` or ``None``.
    ``forced_time`` is set only for "tonight" (today at 19:00) so the time stage
    can still override it with an explicit clock time.
    """
    today = now.date()

    # "the day after tomorrow" (checked before "tomorrow").
    m = re.search(r"\b(?:the\s+)?day\s+after\s+tomorrow\b", t)
    if m:
        return (today + datetime.timedelta(days=2), None), _cut(t, m.span())

    m = re.search(r"\btomorrow\b", t)
    if m:
        return (today + datetime.timedelta(days=1), None), _cut(t, m.span())

    m = re.search(r"\btonight\b", t)
    if m:
        return (today, (_HOUR_EVENING, 0)), _cut(t, m.span())

    m = re.search(r"\btoday\b", t)
    if m:
        return (today, None), _cut(t, m.span())

    m = re.search(r"\bnext\s+week\b", t)
    if m:
        # Start of next week = the coming Monday.
        days = 7 - today.weekday()
        return (today + datetime.timedelta(days=days), None), _cut(t, m.span())

    m = re.search(r"\bnext\s+month\b", t)
    if m:
        nxt = _add_months(
            datetime.datetime.combine(today.replace(day=1), datetime.time()), 1
        )
        return (nxt.date(), None), _cut(t, m.span())

    m = re.search(r"\bnext\s+year\b", t)
    if m:
        return (datetime.date(today.year + 1, 1, 1), None), _cut(t, m.span())

    # Weekday, with optional leading qualifier. Full names only (abbreviations
    # like "mon"/"sun"/"wed" collide with ordinary words).
    m = re.search(
        r"\b(next|this|coming|on)?\s*"
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        t,
    )
    if m:
        qualifier = m.group(1)
        target = _WEEKDAYS[m.group(2)]
        days_ahead = (target - today.weekday()) % 7
        if qualifier == "next" and days_ahead == 0:
            days_ahead = 7  # "next monday" on a Monday -> a week out
        return (today + datetime.timedelta(days=days_ahead), None), _cut(t, m.span())

    return None, t


# --------------------------------------------------------------------------- #
# Clock times
# --------------------------------------------------------------------------- #
_TimeResult = Tuple[int, int]  # (hour24, minute)


def _extract_time(
    t: str, hint: Optional[str]
) -> Tuple[Optional[_TimeResult], str]:
    """Find a clock time in ``t`` and return ``((hour24, minute), leftover)``.

    Tries, in order: noon/midnight, "<n> in the morning/afternoon/evening",
    H:MM / H.MM, H am/pm, HHMM digit runs, "at <bare hour>", spoken word times
    ("two thirty", "quarter past three"), "at <hour word>", and finally the
    daypart words on their own.
    """
    # noon / midday / midnight
    m = re.search(r"\b(noon|midday)\b", t)
    if m:
        return (_HOUR_NOON, 0), _cut(t, m.span())
    m = re.search(r"\bmidnight\b", t)
    if m:
        return (_HOUR_MIDNIGHT, 0), _cut(t, m.span())

    # "<n>[:mm] in the morning/afternoon/evening" -> daypart pins am/pm.
    m = re.search(
        r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s+in\s+the\s+"
        r"(morning|afternoon|evening)\b",
        t,
    )
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2)) if m.group(2) else 0
        if 1 <= hour <= 12 and 0 <= minute <= 59:
            local_hint = "am" if m.group(3) == "morning" else "pm"
            return (_infer_hour(hour, local_hint), minute), _cut(t, m.span())

    # H:MM or H.MM, optional am/pm. Guarded so it is not part of a longer number.
    m = re.search(
        r"(?<!\d)(?:at\s+)?(\d{1,2})[:.](\d{2})\s*(am|pm)?(?!\d)", t
    )
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
        ampm = m.group(3)
        if 0 <= minute <= 59 and (
            (ampm and 1 <= hour <= 12) or (not ampm and 0 <= hour <= 23)
        ):
            return (_apply_ampm(hour, ampm, hint), minute), _cut(t, m.span())

    # H am/pm  ("3pm", "3 p.m." -> normalized to "3 pm").
    m = re.search(r"(?<![\d:.])(\d{1,2})\s*(am|pm)\b", t)
    if m:
        hour = int(m.group(1))
        if 0 <= hour <= 12:
            return (_apply_ampm(hour, m.group(2), hint), 0), _cut(t, m.span())

    # Numeric "<n> o'clock" ("3 oclock" -> 3:00). Spelled hours are handled by
    # the word-time parser further down.
    m = re.search(r"\b(\d{1,2})\s*oclock\b", t)
    if m:
        hour = int(m.group(1))
        if 0 <= hour <= 23:
            resolved = hour if hour >= 13 else _infer_hour(hour, hint)
            return (resolved, 0), _cut(t, m.span())

    # HHMM digit run after "at" ("at 230" -> 2:30, "at 1430" -> 14:30).
    m = re.search(r"\bat\s+(\d{3,4})\b", t)
    if m:
        digits = m.group(1)
        hour = int(digits[:-2])
        minute = int(digits[-2:])
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            resolved = hour if hour >= 13 else _infer_hour(hour, hint)
            return (resolved, minute), _cut(t, m.span())

    # Bare numeric hour after "at" ("at 3" -> heuristic).
    m = re.search(r"\bat\s+(\d{1,2})\b", t)
    if m:
        hour = int(m.group(1))
        if 0 <= hour <= 23:
            resolved = hour if hour >= 13 else _infer_hour(hour, hint)
            return (resolved, 0), _cut(t, m.span())

    # A lone HHMM digit run that is ALL that is left ("230", "tomorrow 230").
    # Requiring it to be the whole remaining text keeps stray numbers (prices,
    # counts, years buried in a sentence) from being read as clock times.
    stripped = t.strip()
    m = re.fullmatch(r"(\d{3,4})", stripped)
    if m:
        digits = m.group(1)
        hour = int(digits[:-2])
        minute = int(digits[-2:])
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            resolved = hour if hour >= 13 else _infer_hour(hour, hint)
            return (resolved, minute), ""

    # Spoken word times ("two thirty", "half past three", "eleven oclock").
    word = _extract_word_time(t, hint)
    if word is not None:
        return word

    # Daypart words on their own -> their default hour.
    m = re.search(r"\bmorning\b", t)
    if m:
        return (_HOUR_MORNING, 0), _cut(t, m.span())
    m = re.search(r"\bafternoon\b", t)
    if m:
        return (_HOUR_AFTERNOON, 0), _cut(t, m.span())
    m = re.search(r"\b(evening|night)\b", t)
    if m:
        return (_HOUR_EVENING, 0), _cut(t, m.span())

    return None, t


def _extract_word_time(
    t: str, hint: Optional[str]
) -> Optional[Tuple[_TimeResult, str]]:
    """Parse spoken clock times built from number words.

    Handles: "quarter/half past <hour>", "<mins> past <hour>",
    "quarter to <hour>", "<mins> to <hour>", "<hour> oclock", and
    "<hour> <minutes>" ("two thirty", "eleven fifteen"). Also a bare spoken hour
    when preceded by "at" ("at eleven"). Returns ``((hour24, minute), leftover)``
    or ``None``.
    """
    toks = t.split()
    n = len(toks)

    # "<mins> past|to <hour>"  (hour and mins may be spelled or digits)
    for i, w in enumerate(toks):
        if w in ("past", "to") and 1 <= i and i + 1 < n:
            hour = _hour_token(toks[i + 1])
            if hour is None:
                continue
            mins = _small_number_from_words(toks[max(0, i - 2):i])
            if mins is None:
                mins = _SIMPLE_NUM.get(toks[i - 1])
            if mins is None and toks[i - 1].isdigit():
                mins = int(toks[i - 1])
            if mins is None or not (0 <= mins < 60):
                continue
            if w == "to":
                hour = (hour - 1) % 12 or 12
                minute = (60 - mins) % 60
            else:
                minute = mins
            resolved = _infer_hour(hour, hint)
            return (resolved, minute), _remove_words(t, toks)

    # "<hour> oclock"
    for i, w in enumerate(toks):
        if w == "oclock" and i >= 1:
            hour = _HOUR_WORDS.get(toks[i - 1])
            if hour is not None:
                return (_infer_hour(hour, hint), 0), _remove_words(t, toks)

    # "<hour> <minutes>"  ("two thirty")
    for i, w in enumerate(toks):
        hour = _HOUR_WORDS.get(w)
        if hour is None:
            continue
        for take in (2, 1):
            if i + take < n + 1:
                frag = toks[i + 1:i + 1 + take]
                mins = _small_number_from_words(frag)
                if mins is not None and 0 <= mins < 60 and len(frag) == take:
                    return (_infer_hour(hour, hint), mins), _remove_words(t, toks)

    # Bare spoken hour, only right after "at" ("at eleven").
    for i, w in enumerate(toks):
        if w == "at" and i + 1 < n:
            hour = _HOUR_WORDS.get(toks[i + 1])
            if hour is not None:
                return (_infer_hour(hour, hint), 0), _remove_words(t, toks)

    return None


def _remove_words(t: str, _toks: List[str]) -> str:
    """Leftover text after a word-time match.

    Word times are woven through the sentence, so rather than surgically excise
    tokens we just hand back the text unchanged -- the time has been captured and
    nothing downstream re-reads it.
    """
    return t


__all__ = ["ParsedWhen", "parse_when"]
