"""Intent handlers: turn a spoken phrase into a concrete, runnable Action.

Each handler recognizes ONE family of spoken commands and, when it applies,
returns an :class:`~blurt.assistant.types.Action` carrying everything needed to
run it plus a confidence score; when it does not apply it returns ``None``. The
router (see :mod:`.router`) asks every handler and runs the most confident match.

Five handlers live here:

  * :class:`CalendarHandler`  -- "add / schedule / put ... on my calendar"
  * :class:`ReminderHandler`  -- "remind me to ...", "reminder to ..."
  * :class:`TimerHandler`     -- "set a timer for N minutes", "timer 10 minutes"
  * :class:`OpenAppHandler`   -- "open X", "launch X", "open the X app"
  * :class:`RevertHandler`    -- "undo that", "revert", "scratch that"

Design rules that matter here:

  * PURE LOGIC. This module imports only the standard library plus the pure
    :mod:`.types` and :mod:`.nldate` modules -- NO pyobjc / EventKit / AppKit.
    The real side effects live behind injected backends (``cal_backend``,
    ``timer_service``, ``app_opener``), so the matching logic is unit-testable on
    any machine and the handlers can be handed fakes in tests.
  * TOLERANT MATCHING. People speak loosely and Whisper mis-transcribes, so
    matching is done on lower-cased text with forgiving regexes. Detection is
    case-insensitive; the human-facing event/reminder TITLE is extracted from the
    original text so proper nouns keep their capitals.
  * NEVER RAISE ON BAD SPEECH. Every :meth:`match` and :meth:`execute` is wrapped
    so a weird phrase or a backend hiccup yields a plain-language result, never a
    traceback. A phrase nothing understands simply produces no match and the
    router falls through to dictation.
  * CREATE-ONLY / REVERSIBLE. These handlers only create events, reminders and
    timers, open apps, or paste text -- all reversible and harmless. Nothing here
    deletes or overwrites anything (blurt's v1 safety rule); even
    :class:`RevertHandler`, whose whole job is undoing, does it by *inserting*
    the raw transcript rather than by deleting what was typed before.
    ``needs_confirmation`` is False on every Action because the user triggered
    each one deliberately by voice and every executed action returns a
    spoken-back message describing what happened.

Confidence scoring (kept simple and explicit, per the contract):

  * an exact, whole-utterance command phrase -> certain (0.95)
  * command verb + a parsed time            -> high   (0.9)
  * unambiguous command verb but no time    -> medium (0.6)
  * weaker / plainer phrasing               -> low-ish (0.7 / 0.3)

What can go wrong: title extraction is heuristic -- it strips the command verb,
filler and the time phrase, and can occasionally clip a title word (e.g. a movie
called "Friday"). That is a cosmetic imperfection on a reversible, spoken-back
action, never a crash. AM/PM and date resolution are delegated entirely to
:func:`blurt.assistant.nldate.parse_when`, which also never raises.
"""
from __future__ import annotations

import datetime
import re
from typing import Callable, Optional

from .nldate import parse_when
from .types import Action, ActionResult, IntentHandler


# --------------------------------------------------------------------------- #
# Shared time/date/duration stripping (for isolating a title)
# --------------------------------------------------------------------------- #
# These patterns remove the parts of an utterance that describe *when* something
# happens, so what is left is the *what* (the title). They are deliberately
# conservative: bare numbers and bare day-parts ("night") are left alone so we do
# not butcher titles like "movie night" or "table 3". The actual date/time used
# for scheduling comes from parse_when, NOT from these patterns.
_WHEN_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # Durations ("for 30 minutes", "for an hour and a half").
        r"\bfor\s+an?\s+hour\s+and\s+a\s+half\b",
        r"\bfor\s+(?:about\s+|around\s+)?\d+(?:\.\d+)?\s*"
        r"(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)\b",
        r"\bfor\s+(?:half\s+an\s+hour|a\s+half\s+hour|half\s+hour|"
        r"a\s+quarter\s+hour|quarter\s+hour|an\s+hour|a\s+minute)\b",
        r"\bfor\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|"
        r"fifteen|twenty|thirty|forty|fifty|sixty|ninety)\s+"
        r"(?:hours?|minutes?|mins?)\b",
        # Relative offsets ("in 20 minutes", "in three days").
        r"\bin\s+(?:about\s+|around\s+)?\d+(?:\.\d+)?\s*"
        r"(?:hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|weeks?|months?)\b",
        r"\bin\s+(?:half\s+an\s+hour|an\s+hour|a\s+minute)\b",
        r"\bin\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|"
        r"fifteen|twenty|thirty)\s+"
        r"(?:hours?|minutes?|mins?|days?|weeks?|months?)\b",
        # Named dates.
        r"\b(?:the\s+)?day\s+after\s+tomorrow\b",
        r"\btomorrow\b",
        r"\btonight\b",
        r"\btoday\b",
        r"\bnext\s+(?:week|month|year|weekend)\b",
        r"\bthis\s+(?:week|month|year|weekend)\b",
        r"\b(?:next|this|coming|on)\s+"
        r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        # Clock times.
        r"\b(?:at|by|around|about|from)\s+\d{1,2}[:.]\d{2}\s*"
        r"(?:a\.?m\.?|p\.?m\.?)?\b",
        r"\b\d{1,2}[:.]\d{2}\s*(?:a\.?m\.?|p\.?m\.?)?\b",
        r"\b(?:at|by|around|about)\s+\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)?\b",
        r"\b\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)\b",
        r"\b\d{1,2}\s*o'?clock\b",
        r"\bat\s+\d{3,4}\b",
        r"\b(?:at\s+)?(?:noon|midday|midnight)\b",
        # Spoken clock times.
        r"\b(?:at\s+)?(?:quarter|half|five|ten|twenty|twenty[- ]five)\s+"
        r"(?:past|to)\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|\d{1,2})\b",
        r"\bat\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
        r"twelve)(?:\s+(?:o'?clock|thirty|fifteen|forty[- ]?five|oh\s+\w+))?\b",
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
        r"\s+o'?clock\b",
        # Day-parts, only with disambiguating context (keeps "movie night").
        r"\b(?:this|tomorrow)\s+(?:morning|afternoon|evening|night)\b",
        r"\bin\s+the\s+(?:morning|afternoon|evening)\b",
    ]
]

# Words that may dangle at the very start or end of a title after stripping and
# should be trimmed (prepositions, articles, filler). Only trimmed at the edges,
# never in the middle -- "lunch with sam" keeps its "with".
_EDGE_WORDS = frozenset(
    [
        "at", "on", "in", "for", "by", "to", "of", "and", "with", "from",
        "this", "next", "coming", "a", "an", "the", "my", "some",
        "please", "um", "uh", "so", "just", "then",
    ]
)


def _strip_when_phrases(text: str) -> str:
    """Remove date/time/duration expressions from ``text`` (for title isolation)."""
    out = text
    for pat in _WHEN_PATTERNS:
        out = pat.sub(" ", out)
    return out


def _tidy_title(text: str) -> str:
    """Collapse whitespace and trim dangling edge words / punctuation."""
    s = re.sub(r"\s+", " ", text).strip()
    toks = s.split(" ") if s else []
    while toks and toks[0].strip(".,:;!?-").lower() in _EDGE_WORDS:
        toks.pop(0)
    while toks and toks[-1].strip(".,:;!?-").lower() in _EDGE_WORDS:
        toks.pop()
    return " ".join(toks).strip(" .,:;-")


def _titlecase_first(text: str) -> str:
    """Capitalize only the first character, leaving proper-noun casing intact."""
    s = text.strip()
    if not s:
        return s
    return s[0].upper() + s[1:]


# --------------------------------------------------------------------------- #
# Datetime formatting for human-readable summaries
# --------------------------------------------------------------------------- #
def _fmt_date(dt: datetime.datetime) -> str:
    """"Tue Jul 22" -- weekday, month, un-padded day."""
    return "{0} {1} {2}".format(dt.strftime("%a"), dt.strftime("%b"), dt.day)


def _fmt_time(dt: datetime.datetime) -> str:
    """"12:00 PM" -- 12-hour clock, un-padded hour, padded minute."""
    hour12 = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    return "{0}:{1:02d} {2}".format(hour12, dt.minute, ampm)


def _fmt_when(dt: datetime.datetime, has_time: bool) -> str:
    """Format a resolved datetime, with or without a clock time."""
    if has_time:
        return "{0}, {1}".format(_fmt_date(dt), _fmt_time(dt))
    return _fmt_date(dt)


def _at(now: datetime.datetime, hour: int, minute: int) -> datetime.datetime:
    """Today's ``hour:minute``, matching ``now``'s tz-awareness."""
    dt = datetime.datetime.combine(now.date(), datetime.time(hour, minute))
    if now.tzinfo is not None:
        dt = dt.replace(tzinfo=now.tzinfo)
    return dt


_DEFAULT_EVENT_HOUR = 9
_DEFAULT_EVENT_MINUTES = 60


# --------------------------------------------------------------------------- #
# CalendarHandler
# --------------------------------------------------------------------------- #
# Leading command verb (optionally behind a polite prefix). Matched at the START
# of the utterance so a noun "schedule" mid-sentence ("what's my schedule") is
# NOT treated as a command.
_CAL_VERB_RE = re.compile(
    r"^\s*(?:please\s+|hey\s+|ok(?:ay)?\s+|so\s+|just\s+)*"
    r"(?:can\s+you\s+|could\s+you\s+|would\s+you\s+|"
    r"i(?:'d|\s+would)\s+like\s+to\s+|i\s+want\s+to\s+|i\s+need\s+to\s+|"
    r"let'?s\s+)?"
    r"(schedule|arrange|plan|add|create|put|set\s*up|setup|set|make|new|book|"
    r"organi[sz]e)\b",
    re.IGNORECASE,
)

# Same head, but consuming a trailing article too, so title extraction can drop
# "schedule a " / "add an " / "put the " in one shot.
_CAL_LEAD_STRIP_RE = re.compile(
    r"^\s*(?:please\s+|hey\s+|ok(?:ay)?\s+|so\s+|just\s+)*"
    r"(?:can\s+you\s+|could\s+you\s+|would\s+you\s+|"
    r"i(?:'d|\s+would)\s+like\s+to\s+|i\s+want\s+to\s+|i\s+need\s+to\s+|"
    r"let'?s\s+)?"
    r"(?:schedule|arrange|plan|add|create|put|set\s*up|setup|set|make|new|book|"
    r"organi[sz]e)\s+"
    r"(?:a\s+|an\s+|the\s+|my\s+|some\s+)?(?:new\s+)?",
    re.IGNORECASE,
)

_CAL_PHRASE_RE = re.compile(
    r"\b(?:on|to|in|onto)\s+(?:my|the)\s+calendar\b", re.IGNORECASE
)
_CALLED_RE = re.compile(r"\b(?:called|titled|named|entitled)\s+(.+)$", re.IGNORECASE)
_STRONG_CAL_VERBS = frozenset(["schedule", "arrange", "plan"])


def _extract_event_title(text: str) -> str:
    """Best-effort event title: drop verb + filler + calendar phrase + time."""
    s = _CAL_LEAD_STRIP_RE.sub("", text.strip())
    m = _CALLED_RE.search(s)
    if m:
        s = m.group(1)
    s = _CAL_PHRASE_RE.sub(" ", s)
    s = _strip_when_phrases(s)
    return _titlecase_first(_tidy_title(s))


class CalendarHandler(IntentHandler):
    """Recognize "add/schedule/put ... on my calendar" and create an event.

    Fires when the utterance opens with a calendar command verb AND either a
    time/date was parsed OR the verb is unambiguously calendar ("schedule", or a
    "... on my calendar" phrase). A bare "add milk" -- weak verb, no time -- does
    NOT become an event; it falls through to dictation. Reminder/timer phrases
    are deferred to their own handlers.

    Confidence: verb + explicit clock time -> 0.9; verb + a date only -> 0.7;
    unambiguous verb with no time at all -> 0.6 (an all-day event today).

    execute() calls ``cal_backend.create_event(title, start, end, notes)``.
    NOTE: that signature has no all-day flag, so a date-only or timeless request
    is realized as a 09:00 placeholder event (recorded as ``all_day`` in the
    payload); the backend's returned message reports the actual time written.
    """

    name = "calendar"

    def __init__(self, cal_backend: object, now_fn: Optional[Callable[[], datetime.datetime]] = None) -> None:
        self._cal_backend = cal_backend
        self._now_fn = now_fn or datetime.datetime.now

    def _now(self) -> datetime.datetime:
        try:
            return self._now_fn()
        except Exception:
            return datetime.datetime.now()

    def match(self, text: str) -> Optional[Action]:
        try:
            if not text or not text.strip():
                return None
            low = text.lower()

            # Defer reminder / timer phrasings to their dedicated handlers.
            if re.search(r"\bremind(?:er)?\b", low):
                return None
            if re.search(r"\btimers?\b", low):
                return None

            m = _CAL_VERB_RE.match(text)
            if not m:
                return None
            verb = m.group(1).lower().replace(" ", "")
            strong = verb in _STRONG_CAL_VERBS or bool(_CAL_PHRASE_RE.search(low))

            parsed = parse_when(text, self._now())
            if parsed is None and not strong:
                # Weak verb with no time -> not a calendar event ("add milk").
                return None

            if parsed is not None:
                start = parsed.start
                has_time = parsed.has_time
                dur = parsed.duration_minutes or _DEFAULT_EVENT_MINUTES
                if has_time:
                    confidence = 0.9
                else:
                    confidence = 0.7
            else:
                # Unambiguous verb, no time at all -> all-day today.
                start = _at(self._now(), _DEFAULT_EVENT_HOUR, 0)
                has_time = False
                dur = _DEFAULT_EVENT_MINUTES
                confidence = 0.6

            end = start + datetime.timedelta(minutes=dur)
            all_day = not has_time

            title = _extract_event_title(text) or "Event"

            when = _fmt_when(start, has_time)
            if all_day:
                when = "{0} (all day)".format(when)
            summary = "Add '{0}' -- {1}".format(title, when)

            return Action(
                kind="calendar_event",
                summary=summary,
                payload={
                    "title": title,
                    "start": start,
                    "end": end,
                    "all_day": all_day,
                    "_handler": self.name,
                },
                confidence=confidence,
                needs_confirmation=False,
            )
        except Exception:
            return None

    def execute(self, action: Action) -> ActionResult:
        try:
            p = action.payload
            return self._cal_backend.create_event(
                p["title"], p["start"], p["end"], notes=None
            )
        except Exception:
            return ActionResult(ok=False, message="I couldn't add that to your calendar.")


# --------------------------------------------------------------------------- #
# ReminderHandler
# --------------------------------------------------------------------------- #
# Ordered content extractors: first that matches wins; group(1) is the task text
# (taken from the ORIGINAL string so casing survives).
_REMIND_CONTENT_RES = [
    re.compile(r"\bremind\s+me\s+(?:to|that|about)\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bremind\s+me\s+(.+)$", re.IGNORECASE),
    re.compile(
        r"\b(?:set|create|add|make)\s+(?:up\s+)?(?:a|an)?\s*reminders?\s+"
        r"(?:to|that|about|for|:)?\s*(.+)$",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*reminders?\s+(?:to|that|about|for|:)?\s*(.+)$", re.IGNORECASE),
]


class ReminderHandler(IntentHandler):
    """Recognize "remind me to X" / "reminder to X" and create a reminder.

    The task text X is everything after the trigger, with any trailing time
    phrase stripped. An optional due date/time is pulled from the whole utterance
    via parse_when. The bare, taskless "remind me in 5 minutes" still matches
    (the spec prefers this handler for "remind me"); its title falls back to
    "Reminder".

    Confidence: trigger + a parsed due -> 0.9; trigger without a due -> 0.6.

    execute() calls ``cal_backend.create_reminder(title, due)``.
    """

    name = "reminder"

    def __init__(self, cal_backend: object, now_fn: Optional[Callable[[], datetime.datetime]] = None) -> None:
        self._cal_backend = cal_backend
        self._now_fn = now_fn or datetime.datetime.now

    def _now(self) -> datetime.datetime:
        try:
            return self._now_fn()
        except Exception:
            return datetime.datetime.now()

    def match(self, text: str) -> Optional[Action]:
        try:
            if not text or not text.strip():
                return None
            low = text.lower()

            has_remind_me = bool(re.search(r"\bremind\s+me\b", low))
            has_reminder_cmd = (
                bool(re.search(r"\b(?:set|create|add|make|new)\s+(?:up\s+)?(?:a|an)?\s*reminders?\b", low))
                or bool(re.match(r"\s*reminders?\b", low))
                or bool(re.search(r"\breminders?\s+(?:to|that|about|for)\b", low))
            )
            if not (has_remind_me or has_reminder_cmd):
                return None

            content = ""
            for rx in _REMIND_CONTENT_RES:
                m = rx.search(text)
                if m:
                    content = m.group(1)
                    break

            title = _titlecase_first(_tidy_title(_strip_when_phrases(content)))
            if not title:
                title = "Reminder"

            parsed = parse_when(text, self._now())
            if parsed is not None:
                due = parsed.start
                has_time = parsed.has_time
                confidence = 0.9
            else:
                due = None
                has_time = False
                confidence = 0.6

            if due is not None:
                when = _fmt_when(due, has_time)
                if title == "Reminder":
                    summary = "Reminder -- {0}".format(when)
                else:
                    summary = "Remind you to {0} -- {1}".format(title, when)
            else:
                if title == "Reminder":
                    summary = "Set a reminder"
                else:
                    summary = "Remind you to {0}".format(title)

            return Action(
                kind="reminder",
                summary=summary,
                payload={"title": title, "due": due, "_handler": self.name},
                confidence=confidence,
                needs_confirmation=False,
            )
        except Exception:
            return None

    def execute(self, action: Action) -> ActionResult:
        try:
            p = action.payload
            return self._cal_backend.create_reminder(p["title"], due=p.get("due"))
        except Exception:
            return ActionResult(ok=False, message="I couldn't create that reminder.")


# --------------------------------------------------------------------------- #
# TimerHandler
# --------------------------------------------------------------------------- #
# Spelled-out counts usable as a timer amount.
_DURATION_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45,
    "fifty": 50, "sixty": 60, "ninety": 90, "a": 1, "an": 1,
}
_DUR_DIGIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b",
    re.IGNORECASE,
)
_DUR_WORD_RE = re.compile(
    r"\b(" + "|".join(sorted(_DURATION_WORDS.keys(), key=len, reverse=True)) + r")\s+"
    r"(hours?|hrs?|minutes?|mins?|seconds?|secs?)\b",
    re.IGNORECASE,
)


def _parse_duration_minutes(text: str) -> Optional[float]:
    """Parse a timer length in minutes from spoken text, or None.

    Handles digit forms ("5 minutes", "2 hours", "90 seconds", "1 hour 30
    minutes" summed) and common word forms ("half an hour", "ten minutes"). A
    negative or nonsensical value is left for :meth:`TimerService.schedule` to
    reject; here we only fail to a clean None when nothing time-like is found.
    """
    t = text.lower()

    if re.search(r"\ban?\s+hour\s+and\s+a\s+half\b", t):
        return 90.0

    total = 0.0
    found = False
    for m in _DUR_DIGIT_RE.finditer(t):
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        unit = m.group(2)
        if unit.startswith(("hour", "hr")) or unit == "h":
            total += val * 60.0
        elif unit.startswith("sec") or unit == "s":
            total += val / 60.0
        else:  # minutes / min / m
            total += val
        found = True
    if found:
        return total

    if re.search(r"\b(?:half\s+an\s+hour|a\s+half\s+hour|half\s+hour)\b", t):
        return 30.0
    if re.search(r"\b(?:quarter\s+of\s+an\s+hour|a\s+quarter\s+hour|quarter\s+hour)\b", t):
        return 15.0
    if re.search(r"\ban\s+hour\b", t):
        return 60.0
    if re.search(r"\ba\s+minute\b", t):
        return 1.0

    m = _DUR_WORD_RE.search(t)
    if m:
        val = _DURATION_WORDS.get(m.group(1).lower())
        if val is None:
            return None
        unit = m.group(2)
        if unit.startswith(("hour", "hr")):
            return float(val) * 60.0
        if unit.startswith("sec"):
            return float(val) / 60.0
        return float(val)

    return None


_TIMER_LABEL_STRIP_RE = re.compile(
    r"\b(?:please|set|start|create|make|a|an|the|new|up|timer|timers)\b",
    re.IGNORECASE,
)


def _extract_timer_label(text: str) -> str:
    """Pull a short label out of a timer command ("pasta", "check the oven")."""
    s = _TIMER_LABEL_STRIP_RE.sub(" ", text)
    s = _strip_when_phrases(s)  # also removes "for 5 minutes" etc.
    # Drop numeric/word durations that survived (bare "5 minutes", "ten minutes").
    s = _DUR_DIGIT_RE.sub(" ", s)
    s = _DUR_WORD_RE.sub(" ", s)
    s = re.sub(r"^\s*(?:for|to|called|named|labell?ed)\s+", " ", s, flags=re.IGNORECASE)
    return _tidy_title(s)


class TimerHandler(IntentHandler):
    """Recognize "set a timer for N", "timer N minutes" and arm a countdown.

    Requires the word "timer" AND a timer-shaped phrasing (starts with a command
    verb + "timer", or "timer" with a parsed duration), so an incidental "the
    timer on the stove" does not fire. "remind me in 5 minutes" is intentionally
    left to :class:`ReminderHandler`; this handler owns "timer".

    Confidence: "timer" + a parsed duration -> 0.9; a timer command with no
    catchable duration -> 0.6 (execute then asks how long, rather than guessing).

    execute() calls ``timer_service.schedule(minutes, label)``.
    """

    name = "timer"

    def __init__(self, timer_service: object, now_fn: Optional[Callable[[], datetime.datetime]] = None) -> None:
        self._timer_service = timer_service
        self._now_fn = now_fn or datetime.datetime.now

    def match(self, text: str) -> Optional[Action]:
        try:
            if not text or not text.strip():
                return None
            low = text.lower()

            if not re.search(r"\btimers?\b", low):
                return None

            duration = _parse_duration_minutes(text)
            looks_like_timer = (
                bool(re.match(r"\s*(?:please\s+)?(?:set|start|create|make)\b.*\btimer", low))
                or bool(re.match(r"\s*(?:please\s+)?timers?\b", low))
                or duration is not None
            )
            if not looks_like_timer:
                return None

            label = _extract_timer_label(text)

            if duration is not None:
                confidence = 0.9
                if abs(duration - round(duration)) < 1e-9:
                    pretty = "{0:d} minute{1}".format(
                        int(round(duration)), "" if int(round(duration)) == 1 else "s"
                    )
                else:
                    pretty = "{0:g} minutes".format(duration)
                summary = "Set a timer for {0}".format(pretty)
            else:
                confidence = 0.6
                summary = "Set a timer"
            if label:
                summary = "{0}: {1}".format(summary, label)

            return Action(
                kind="timer",
                summary=summary,
                payload={"minutes": duration, "label": label, "_handler": self.name},
                confidence=confidence,
                needs_confirmation=False,
            )
        except Exception:
            return None

    def execute(self, action: Action) -> ActionResult:
        try:
            p = action.payload
            minutes = p.get("minutes")
            if minutes is None:
                return ActionResult(
                    ok=False, message="I didn't catch how long to set the timer for."
                )
            return self._timer_service.schedule(minutes, p.get("label", ""))
        except Exception:
            return ActionResult(ok=False, message="I couldn't set that timer.")


# --------------------------------------------------------------------------- #
# OpenAppHandler
# --------------------------------------------------------------------------- #
_OPEN_RE = re.compile(
    r"^\s*(?:please\s+|hey\s+|ok(?:ay)?\s+|just\s+|can\s+you\s+|could\s+you\s+|"
    r"would\s+you\s+)*"
    r"(open\s+up|open|launch|start\s+up|start|fire\s+up|boot\s+up|bring\s+up|"
    r"pull\s+up|switch\s+to)\s+"
    r"(?:the\s+|my\s+)?(.+?)\s*$",
    re.IGNORECASE,
)
_STRONG_OPEN_VERBS = frozenset(
    ["launch", "fireup", "bootup", "startup", "openup", "bringup", "pullup"]
)
# Objects one "opens" that are plainly not applications. Kept small and concrete;
# anything not listed is attempted and, if it isn't a real app, the opener simply
# reports it couldn't find one (a harmless, spoken-back miss).
_NON_APP_OBJECTS = frozenset(
    [
        "door", "doors", "window", "windows", "gate", "gates", "box", "boxes",
        "jar", "jars", "bottle", "bottles", "can", "cans", "fridge",
        "refrigerator", "drawer", "drawers", "cabinet", "cupboard", "curtain",
        "curtains", "blinds", "garage", "trunk", "hood", "lid", "envelope",
        "package", "packages", "present", "presents", "gift", "gifts", "bag",
        "bags", "umbrella", "wine", "beer", "mouth", "eyes", "car", "engine",
        "laundry", "dishwasher", "oven", "stove", "shower", "account",
        "accounts", "ticket", "tickets", "issue", "issues", "case", "bug",
        "bugs", "discussion", "conversation",
    ]
)
_APP_SUFFIX_RE = re.compile(
    r"\s+(?:app|application|program|please)\s*$", re.IGNORECASE
)


def _display_app(name: str) -> str:
    """Title-case only all-lowercase words, leaving "iTerm"/"VLC" untouched."""
    words = []
    for word in name.split():
        if word.isalpha() and word.islower():
            words.append(word.capitalize())
        else:
            words.append(word)
    return " ".join(words)


class OpenAppHandler(IntentHandler):
    """Recognize "open X" / "launch X" / "open the X app" and launch it.

    Guards against non-launch senses of "open": the target must look like an app
    name -- short (<= 4 words) and not an obvious physical/abstract object ("open
    the door", "open a ticket") -- otherwise this returns None and the phrase is
    dictated instead. ``app_opener`` (inject ``blurt.assistant.system_actions.
    open_app``) does its own case-insensitive lookup, so the raw spoken name is
    passed straight through.

    Confidence: an explicit launch verb or a "... app" suffix -> 0.9; a plain
    "open X" -> 0.7.
    """

    name = "open_app"

    def __init__(self, app_opener: Callable[[str], ActionResult]) -> None:
        self._app_opener = app_opener

    def match(self, text: str) -> Optional[Action]:
        try:
            if not text or not text.strip():
                return None

            m = _OPEN_RE.match(text)
            if not m:
                return None
            verb = m.group(1).lower().replace(" ", "")
            target = m.group(2).strip()

            had_app_suffix = bool(_APP_SUFFIX_RE.search(target))
            target = _APP_SUFFIX_RE.sub("", target).strip(" .,!?\"'")
            if not target:
                return None

            tokens = target.lower().split()
            # Too long to be an app name -> almost certainly not a launch.
            if len(tokens) > 4:
                return None
            # Obvious non-app object anywhere in the target -> don't fire.
            if any(tok in _NON_APP_OBJECTS for tok in tokens):
                return None

            strong = verb in _STRONG_OPEN_VERBS or had_app_suffix
            confidence = 0.9 if strong else 0.7

            summary = "Open {0}".format(_display_app(target))

            return Action(
                kind="open_app",
                summary=summary,
                payload={"name": target, "_handler": self.name},
                confidence=confidence,
                needs_confirmation=False,
            )
        except Exception:
            return None

    def execute(self, action: Action) -> ActionResult:
        try:
            return self._app_opener(action.payload["name"])
        except Exception:
            return ActionResult(ok=False, message="I couldn't open that app.")


# --------------------------------------------------------------------------- #
# RevertHandler
# --------------------------------------------------------------------------- #
# THE MATCHING RULE, STATED ONCE SO TESTS CAN CITE IT:
#
#   An utterance is a revert command only if, after normalization, it passes BOTH
#   gates in :func:`_is_revert_command`:
#
#     1. it is at most _MAX_REVERT_WORDS words long, AND
#     2. the WHOLE string is one of the fixed phrases below -- the pattern is
#        anchored ``^...$``, so a match is never a substring of a longer sentence
#        and no alternative is allowed to end in a wildcard tail.
#
# Both gates live in ONE function on purpose. They used to be two consecutive
# checks inside match(), which is fine right up until someone adds a second entry
# point -- a "would this match?" preview, a router fast-path, a test helper -- and
# copies across only the regex. The ceiling then silently stops applying to that
# path, and nothing fails: the funnel just gets wider. One gate, one call site,
# no way to consult half of it.
#
# The ceiling is load-bearing, not decoration. "please give me the raw text
# please" is seven words and satisfies the pattern; the ceiling is the only thing
# that turns it down. THE CHECK TO RE-RUN AFTER EDITING ANYTHING HERE: set
# _MAX_REVERT_WORDS to a huge number and confirm behaviour actually changes. If
# it does not, one of the two gates has stopped doing anything and this comment
# is describing a defence that is no longer there.
_MAX_REVERT_WORDS = 6

# Things you can say you want undone. Longest alternatives first so a full match
# is found without leaning on backtracking to fix the ordering.
_REVERT_OBJECT = (
    r"(?:the\s+last\s+dictation|the\s+last\s+transcript|the\s+last\s+thing|"
    r"the\s+last\s+one|that\s+one|this\s+one|the\s+last|the\s+cleanup|"
    r"that|this|it|last)"
)

# Only fillers people actually put in front of a two-word command. Kept short on
# purpose: every prefix admitted here widens the mouth of the funnel.
_REVERT_LEAD = r"(?:please\s+|hey\s+|ok(?:ay)?\s+|so\s+|just\s+|um\s+|uh\s+)*"

# THE RAW-TEXT FAMILY IS A CLOSED LIST, NOT A GRAMMAR.
#
# This was written as (use|give me|paste|insert|i want|show me) + an optional
# "the" + "raw" + an OPTIONAL noun. Six verbs times an optional article times an
# optional noun is about sixty accepted phrases, and the optional noun is what
# does the damage: it admits bare "use raw", "insert raw" and "i want raw".
# "I want raw" is a sentence about sushi, or a file format, or a camera setting.
# It is dictation, and firing on it pastes a stale transcript into whatever the
# user was in the middle of writing.
#
# So the generated set is gone, replaced by the literal phrases below. There is
# no rule that produces them, and that is the point: this list can only grow by
# someone typing a new line here, which forces the audit question to be asked
# once per spelling -- is there a plausible English sentence where a person says
# exactly this, as their entire utterance, and does not mean "undo my last
# dictation"? Near neighbours of the accepted five ("paste the raw text", "use
# the raw version") are deliberately NOT admitted by pattern; each spelling gets
# in one at a time or not at all.
#
# None of these five is load-bearing. "undo that", "revert that" and "scratch
# that" are what people actually say. The raw phrasings are a convenience, and if
# one of them ever turns out to fire on real dictation the correct fix is to
# delete the line rather than qualify it -- a smaller vocabulary that is never
# wrong beats a larger one that sometimes is.
_REVERT_RAW_PHRASE = (
    r"(?:use\s+the\s+raw\s+text"
    r"|use\s+the\s+raw\s+transcript"
    r"|give\s+me\s+the\s+raw\s+text"
    r"|show\s+me\s+the\s+raw\s+text"
    r"|paste\s+the\s+raw\s+version)"
)

# Every alternative below has been asked the audit question. Each either names an
# undo verb outright ("undo", "revert", "scratch") or ends in one ("that's wrong
# undo it", "never mind undo"), and each is closed at both ends by the anchors --
# there is no ".*" anywhere in this pattern and there must never be one.
_REVERT_PHRASE_RE = re.compile(
    r"^"
    + _REVERT_LEAD
    + r"(?:"
    # "undo", "revert", "undo that", "revert the last dictation", "undo last".
    # The object is a closed list too: "undo the migration" and "revert to the
    # previous vendor" name objects that are not on it, so they do not match.
    r"(?:undo|revert)(?:\s+" + _REVERT_OBJECT + r")?"
    # "that's wrong, undo it" -- the correction people blurt out mid-sentence.
    # The ceiling clips the longest spellings of this one ("it's not what i said
    # undo it" is seven words). That is the ceiling winning, which is the
    # intended precedence: a missed revert costs a repeat.
    r"|(?:that'?s|it'?s)\s+(?:wrong|not\s+right|not\s+what\s+i\s+said)\s+"
    r"(?:undo|revert)(?:\s+(?:it|that))?"
    # "scratch that" (but NOT "scratch that itch" -- the anchor forbids a tail,
    # and "scratch the plan" names an object that is not on the list).
    r"|scratch\s+(?:that|this|it)(?:\s+one)?"
    # "never mind undo". Bare "never mind" is NOT a revert: it is ordinary speech,
    # so the explicit undo/revert verb is required.
    r"|(?:never\s*mind|nvm)\s+(?:undo|revert)(?:\s+(?:it|that|that\s+one))?"
    # The five enumerated raw-text phrases; see the note above.
    r"|" + _REVERT_RAW_PHRASE + r")"
    r"(?:\s+please)?"
    r"$",
    re.IGNORECASE,
)

# Punctuation that Whisper sprinkles into short commands ("Undo that."). Dropped
# before matching so it cannot defeat the anchors. The apostrophe survives, so
# "that's" stays "that's"; the regex above accepts both spellings anyway because
# transcription drops it about as often as it keeps it.
_REVERT_PUNCT_RE = re.compile(r"[.,;:!?\"()\[\]–—…“”]+")


def _normalize_revert_text(text: str) -> str:
    """Lower-case, de-punctuate and collapse whitespace for full-phrase matching.

    Speech arrives as "Undo that." or " undo that, " or "Undo -- that"; none of
    those should be a different command from "undo that". Curly apostrophes are
    folded to straight ones because Whisper emits both.
    """
    s = text.replace("’", "'")
    s = _REVERT_PUNCT_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def _is_revert_command(normalized: str) -> bool:
    """Both gates, in one place, so no caller can consult one without the other.

    Takes ALREADY-normalized text (see :func:`_normalize_revert_text`) and applies
    the word ceiling and the whole-utterance anchor together. The ceiling is
    checked first only because it is the cheaper of the two; the order carries no
    meaning, and a phrase must clear both.

    Keeping this as a function rather than two lines in :meth:`RevertHandler.match`
    is the whole safety story: a future caller that wants to ask "is this a revert
    command?" gets both defences or neither. The alternative -- exporting the
    compiled pattern and trusting each caller to remember the ceiling -- is how the
    ceiling quietly stops applying.
    """
    if not normalized:
        return False
    if len(normalized.split(" ")) > _MAX_REVERT_WORDS:
        return False
    return _REVERT_PHRASE_RE.match(normalized) is not None


class RevertHandler(IntentHandler):
    """Recognize "undo that" and re-insert the RAW text of the last dictation.

    This is the voice half of :meth:`blurt.app.BlurtApp.revert_last` -- the escape
    hatch that makes running the cleanup pass by default defensible. If cleanup
    mangles a sentence, saying "undo that" hands back exactly what the engine
    heard. Until this handler existed, ``revert_last`` was unreachable code and
    the promise was theoretical.

    ASYMMETRIC RISK, POINTING THE OTHER WAY
    ---------------------------------------
    Everywhere else in blurt the cheap mistake is failing to act: you lose a
    second and say it again. Here it is inverted. A false positive pastes a stale
    raw transcript into whatever the user happened to be typing -- an email, a
    commit message, someone else's document -- and blurt cannot take it back
    (:mod:`blurt.inject` can paste but not delete; see ``revert_last``'s v1
    limitation). A missed revert costs one repeat; a spurious one corrupts a
    buffer the user was not thinking about.

    That asymmetry is the justification for every conservative choice below. It
    is also why the vocabulary is allowed to be small: the cost of an unrecognised
    phrasing is that the user says "undo that" instead, and "undo that" is what
    they were going to say anyway.

    THE RULE
    --------
    An utterance is a revert command only if it clears BOTH gates in
    :func:`_is_revert_command`:

      * ANCHORING. The normalized utterance must match a fixed phrase as a
        COMPLETE utterance -- the pattern is anchored ``^...$`` and contains no
        wildcards, so matching is never substring matching and no alternative may
        end in an open tail. "scratch that" is a command; "scratch that itch" is
        not, and the only difference is the anchor. The accepted objects ("that",
        "the last one", "the cleanup", ...) and the five raw-text phrasings are
        both closed lists, enumerated literally rather than generated, so the
        vocabulary cannot grow by accident.
      * CEILING. At most ``_MAX_REVERT_WORDS`` (6) words, counted after
        normalization. This is a second, cruder net that keeps holding if someone
        later loosens the pattern; it is genuinely load-bearing today, because
        "please give me the raw text please" satisfies the pattern and is turned
        down by word count alone.

    Adding a phrase means asking one question first: is there a plausible English
    sentence where a person says exactly this, as their whole utterance, and does
    not mean "undo my last dictation"? If yes, it does not go in. A smaller
    vocabulary that is never wrong beats a larger one that sometimes is.

    Concretely, these do NOT match, and must not:

      * "I need to undo the migration before the deploy, can you note that"
      * "undo the last commit in git and force push"
      * "the revert button is greyed out in the admin panel"
      * "scratch that itch"
      * "i want raw", "use raw", "insert raw", "paste the raw text"

    The first group is dictation that runs on past a revert-ish verb; the anchor
    is the only thing separating the first two from a real command. The last line
    is dictation that is SHORTER than a real command -- a sentence about sushi, a
    file format, a camera setting -- and it is the reason the raw-text family is
    an enumerated list of complete phrases instead of a verb-plus-"raw" grammar.

    Confidence is 0.95 -- higher than any other handler. That is not swagger,
    it is arithmetic: nothing here matches unless the entire utterance is a known
    command phrase, so when it does match there is no competing reading left for
    the router to weigh. It also guarantees a bare "undo" outranks any other
    handler that might get creative about the word.

    execute() calls the injected ``revert_fn`` (the app passes
    ``BlurtApp.revert_last``) and translates its boolean into something sayable.
    """

    name = "revert"

    def __init__(self, revert_fn: Callable[[], bool]) -> None:
        self._revert_fn = revert_fn

    def match(self, text: str) -> Optional[Action]:
        try:
            if not text or not text.strip():
                return None

            # One gate, both defences. Deliberately not inlined here: see
            # _is_revert_command for why the ceiling and the anchor travel
            # together.
            if not _is_revert_command(_normalize_revert_text(text)):
                return None

            return Action(
                kind="revert",
                summary="Revert to the raw transcript",
                payload={"_handler": self.name},
                confidence=0.95,
                needs_confirmation=False,
            )
        except Exception:
            return None

    def execute(self, action: Action) -> ActionResult:
        """Run the revert and report it without contradicting the console.

        ``revert_last`` has already printed the SPECIFIC reason it declined
        (nothing dictated yet, already reverted, raw history off, cleanup changed
        nothing). Restating a guess here would only risk disagreeing with it, so
        the failure message says the one thing that is true in every case and
        stops. A ``revert_fn`` that raises is treated the same as a decline: the
        router would swallow the traceback anyway, but a handler that lets an
        exception out of execute() is a handler that can lose the user's words
        somewhere less forgiving.
        """
        try:
            reverted = bool(self._revert_fn())
        except Exception:
            return ActionResult(ok=False, message="I couldn't revert that.")
        if reverted:
            return ActionResult(ok=True, message="Reverted to the raw transcript.")
        return ActionResult(ok=False, message="Nothing to revert.")


__all__ = [
    "CalendarHandler",
    "ReminderHandler",
    "TimerHandler",
    "OpenAppHandler",
    "RevertHandler",
]
