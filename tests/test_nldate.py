"""Tests for :func:`blurt.assistant.nldate.parse_when`.

parse_when turns spoken text ("tomorrow at 2:30", "in 20 minutes") into a
resolved :class:`ParsedWhen`. It is a pure-logic module -- standard-library
``datetime``/``re`` only, no pyobjc -- so these tests run on any machine, Intel
or Apple Silicon, offline, under the 3.9.6 system Python on the floor machine.

Determinism: ``now`` is always injected. The module NEVER reads the system
clock, and the workflow rule forbids real clocks in tests, so every call here
passes the same fixed ``NOW`` -- a Monday, 2026-07-20 10:30am. Every asserted
result was derived from that anchor.

These tests assert the rules the module *documents* in its own docstring (the
AM/PM heuristic, the future-roll, the weekday "next" rule, the duration-needs-a-
start rule). Where a choice is a documented judgement call rather than a fact,
the test comment says so, so a future reader does not mistake a deliberate
decision for an accident.

Python 3.9 floor: lazy annotations, no PEP 585/604 syntax.
"""

from __future__ import annotations

import datetime

import pytest

from blurt.assistant.nldate import ParsedWhen, parse_when


# A Monday, 10:30am. Chosen so weekday math is easy to reason about (Monday is
# weekday 0) and so that "morning" times like 9am have already passed while
# afternoon times like 3pm have not -- that split is what exercises the roll.
NOW = datetime.datetime(2026, 7, 20, 10, 30)


def _p(text):
    """Parse ``text`` against the fixed NOW. Every test goes through here."""
    return parse_when(text, NOW)


def test_now_is_the_monday_we_think_it_is():
    # Guards the whole file: if this anchor is ever edited, the weekday-based
    # expectations below would silently drift. weekday()==0 means Monday.
    assert NOW.weekday() == 0


# --------------------------------------------------------------------------- #
# Plain dates with no clock time -> has_time False, placeholder 09:00
# --------------------------------------------------------------------------- #
def test_tomorrow_is_next_calendar_day_all_day():
    r = _p("tomorrow")
    assert isinstance(r, ParsedWhen)
    assert r.start == datetime.datetime(2026, 7, 21, 9, 0)
    # No clock time was spoken, so the caller may treat this as all-day; the
    # 09:00 is a documented placeholder, not a claimed time.
    assert r.has_time is False
    assert r.duration_minutes is None


def test_the_day_after_tomorrow_is_two_days_out():
    r = _p("the day after tomorrow")
    assert r.start == datetime.datetime(2026, 7, 22, 9, 0)
    assert r.has_time is False


# --------------------------------------------------------------------------- #
# "tomorrow at 2:30" across the three ways Whisper writes the same spoken time.
# Documented AM/PM rule for a bare hour with NO am/pm and NO daypart hint:
# hours 1..6 read as afternoon (PM), so 2 -> 14:00. All three spellings must
# agree, and none roll (an explicit date is taken at face value).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "tomorrow at 2:30",   # colon separator
    "tomorrow at 2.30",   # Whisper often emits a dot
    "tomorrow at 230",    # ...or drops the separator entirely
])
def test_tomorrow_at_two_thirty_is_pm_by_the_small_hour_heuristic(text):
    r = _p(text)
    assert r.start == datetime.datetime(2026, 7, 21, 14, 30)
    assert r.has_time is True
    assert r.duration_minutes is None


def test_the_three_spellings_of_two_thirty_agree():
    a = _p("tomorrow at 2:30").start
    b = _p("tomorrow at 2.30").start
    c = _p("tomorrow at 230").start
    assert a == b == c == datetime.datetime(2026, 7, 21, 14, 30)


# --------------------------------------------------------------------------- #
# Bare clock time with no date: resolve, then roll forward only if it already
# passed today. At 10:30, 3pm is still ahead -> stays today.
# --------------------------------------------------------------------------- #
def test_at_3pm_resolves_to_this_afternoon_and_does_not_roll():
    r = _p("at 3pm")
    # 15:00 today is still in the future relative to 10:30, so no roll.
    assert r.start == datetime.datetime(2026, 7, 20, 15, 0)
    assert r.has_time is True


def test_at_9_rolls_to_tomorrow_because_9am_already_passed():
    # "at 9" with no hint: 7..11 read as AM -> 09:00. 09:00 today is before
    # 10:30 now, so the documented whole-day roll pushes it to tomorrow 09:00.
    r = _p("at 9")
    assert r.start == datetime.datetime(2026, 7, 21, 9, 0)
    assert r.has_time is True


def test_at_3_bare_hour_reads_as_afternoon():
    # 1..6 -> PM heuristic, and 15:00 has not passed, so it stays today.
    r = _p("at 3")
    assert r.start == datetime.datetime(2026, 7, 20, 15, 0)
    assert r.has_time is True


# --------------------------------------------------------------------------- #
# noon / midnight
# --------------------------------------------------------------------------- #
def test_at_noon_is_twelve_hundred_today():
    r = _p("at noon")
    assert r.start == datetime.datetime(2026, 7, 20, 12, 0)
    assert r.has_time is True


def test_at_midnight_is_hour_zero_and_rolls_to_the_coming_midnight():
    r = _p("at midnight")
    # Midnight resolves to hour 0. 00:00 *today* is already behind us at 10:30,
    # so the documented roll advances it a whole day -- which is exactly the
    # upcoming midnight a speaker means. Assert both the hour and the roll.
    assert r.start.hour == 0
    assert r.start.minute == 0
    assert r.start == datetime.datetime(2026, 7, 21, 0, 0)
    assert r.has_time is True


# --------------------------------------------------------------------------- #
# Relative offsets from NOW -> always has_time True.
# --------------------------------------------------------------------------- #
def test_in_20_minutes_is_now_plus_twenty():
    r = _p("in 20 minutes")
    assert r.start == NOW + datetime.timedelta(minutes=20)
    assert r.start == datetime.datetime(2026, 7, 20, 10, 50)
    assert r.has_time is True
    assert r.duration_minutes is None


def test_in_an_hour_is_now_plus_sixty_minutes():
    r = _p("in an hour")
    assert r.start == NOW + datetime.timedelta(hours=1)
    assert r.start == datetime.datetime(2026, 7, 20, 11, 30)
    assert r.has_time is True


def test_in_half_an_hour_is_now_plus_thirty_minutes():
    # "half an hour" is normalized to "30 minutes" before parsing.
    r = _p("in half an hour")
    assert r.start == NOW + datetime.timedelta(minutes=30)
    assert r.start == datetime.datetime(2026, 7, 20, 11, 0)
    assert r.has_time is True


# --------------------------------------------------------------------------- #
# Weekday names. Documented rule:
#   "friday"/"this friday"  -> nearest occurrence, days = (W - T) % 7
#   "next friday"           -> same, but a week out if today IS that weekday
# From a Monday, "friday" and "next friday" therefore both mean this coming
# Friday (2026-07-24); they only diverge when today is the named weekday.
# --------------------------------------------------------------------------- #
def test_next_friday_from_monday_is_this_coming_friday():
    # DOCUMENTED CHOICE, not a bug: the module defines "next X" as the coming
    # occurrence (it only jumps a week when today already is that weekday). From
    # Monday the coming Friday is 2026-07-24. This asserts the module's stated
    # rule; it is deliberately NOT the "Friday of next week" reading.
    r = _p("next friday")
    assert r.start == datetime.datetime(2026, 7, 24, 9, 0)
    assert r.has_time is False


def test_bare_friday_equals_next_friday_from_a_non_friday():
    # The two only differ when today is the named day; today is Monday, so they
    # must agree -- both the coming Friday.
    assert _p("friday").start == _p("next friday").start
    assert _p("friday").start == datetime.datetime(2026, 7, 24, 9, 0)


def test_bare_monday_means_today_when_today_is_monday():
    # Nearest occurrence with days = (0 - 0) % 7 = 0, i.e. today.
    r = _p("monday")
    assert r.start == datetime.datetime(2026, 7, 20, 9, 0)
    assert r.has_time is False


def test_next_monday_means_a_week_out_when_today_is_monday():
    # The one case where "next" diverges from the bare weekday: today already
    # is Monday, so "next monday" is 7 days out, never today.
    r = _p("next monday")
    assert r.start == datetime.datetime(2026, 7, 27, 9, 0)
    assert r.has_time is False


# --------------------------------------------------------------------------- #
# Duration. Documented rule: a duration only rides along with a resolved start.
# A bare "for 30 minutes" is NOT a point in time and returns None.
# --------------------------------------------------------------------------- #
def test_bare_duration_alone_is_not_a_when():
    assert _p("for 30 minutes") is None
    assert _p("for an hour") is None


def test_duration_rides_along_with_a_date():
    r = _p("tomorrow for 30 minutes")
    assert r.start == datetime.datetime(2026, 7, 21, 9, 0)
    assert r.has_time is False
    assert r.duration_minutes == 30


def test_for_an_hour_yields_sixty_minutes_when_attached_to_a_date():
    r = _p("tomorrow for an hour")
    assert r.duration_minutes == 60
    assert r.start == datetime.datetime(2026, 7, 21, 9, 0)


def test_duration_rides_along_with_an_explicit_clock_time():
    r = _p("tomorrow at 3pm for 30 minutes")
    assert r.start == datetime.datetime(2026, 7, 21, 15, 0)
    assert r.has_time is True
    assert r.duration_minutes == 30


# --------------------------------------------------------------------------- #
# Spelled-out clock time.
# --------------------------------------------------------------------------- #
def test_two_thirty_spelled_out_uses_the_pm_heuristic():
    # "two thirty" -> hour 2 (1..6 -> PM) + 30 -> 14:30. 14:30 has not passed,
    # so it stays today; no date was given.
    r = _p("two thirty")
    assert r.start == datetime.datetime(2026, 7, 20, 14, 30)
    assert r.has_time is True
    assert r.duration_minutes is None


# --------------------------------------------------------------------------- #
# Daypart hints override the bare-hour heuristic (documented).
# --------------------------------------------------------------------------- #
def test_tonight_is_seven_pm_today():
    r = _p("tonight")
    assert r.start == datetime.datetime(2026, 7, 20, 19, 0)
    assert r.has_time is True


def test_tonight_at_8_is_pushed_to_pm_by_the_evening_hint():
    # "tonight" makes bare hours PM -> 8 becomes 20:00.
    r = _p("tonight at 8")
    assert r.start == datetime.datetime(2026, 7, 20, 20, 0)
    assert r.has_time is True


def test_morning_hint_keeps_a_bare_hour_in_the_am():
    # "morning" biases AM, so "at 8" stays 08:00 rather than the 20:00 a bare
    # small hour might otherwise get; the explicit "tomorrow" date suppresses
    # the roll even though 08:00 is behind us.
    r = _p("tomorrow morning at 8")
    assert r.start == datetime.datetime(2026, 7, 21, 8, 0)
    assert r.has_time is True


# --------------------------------------------------------------------------- #
# Text with no temporal expression -> None (not an error, just nothing to do).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "buy groceries",
    "call the dentist",
    "remind me to call bob",
    "the meeting is important",
])
def test_text_with_no_time_returns_none(text):
    assert _p(text) is None


# --------------------------------------------------------------------------- #
# Robustness: garbage, empty, and non-string input must return None, never
# raise. This is the module's core safety contract -- a misparse becomes
# "I didn't understand", never a traceback.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [
    "",
    "   ",
    "!!!",
    "?!.,;",
    "xyzzy 12345 blah",
    "asdfghjkl",
    "\U0001f389\U0001f389\U0001f389",   # emoji only
    None,
    12345,
    3.14,
    [],
    {},
    object(),
])
def test_garbage_input_returns_none_and_never_raises(bad):
    # Must not raise for any of these, and must resolve to None (nothing found).
    assert _p(bad) is None


def test_never_raises_on_a_wide_spray_of_odd_inputs():
    # A broader fuzz: whatever comes back, it is either None or a ParsedWhen,
    # and nothing raises. Guards the "never crash on bad speech" contract.
    weird = [
        "at", "in", "for", "at at at", "in for at",
        "at :30", "at 2:99", "at 99:99", "at 25", "at 0",
        "in minutes", "for hours", "next", "next next friday",
        "tomorrow tomorrow", "monday tuesday wednesday",
        "meet at 3 or maybe 4 or 5", "12345678901234567890",
        "the 30th of never", "half past", "quarter to",
        "  tomorrow   at    3   pm  ",
    ]
    for w in weird:
        r = _p(w)
        assert r is None or isinstance(r, ParsedWhen)


# --------------------------------------------------------------------------- #
# Timezone handling: a tz-aware NOW yields a tz-aware start with the same
# tzinfo; a naive NOW yields a naive start. (Documented invariant.)
# --------------------------------------------------------------------------- #
def test_naive_now_gives_naive_start():
    r = _p("tomorrow")
    assert r.start.tzinfo is None


def test_aware_now_propagates_tzinfo_to_the_result():
    tz = datetime.timezone(datetime.timedelta(hours=-5))
    now_aware = datetime.datetime(2026, 7, 20, 10, 30, tzinfo=tz)
    r = parse_when("in 20 minutes", now_aware)
    assert r is not None
    assert r.start.tzinfo == tz
    assert r.start == now_aware + datetime.timedelta(minutes=20)
