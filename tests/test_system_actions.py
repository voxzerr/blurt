"""Unit tests for blurt's system actions (open an app, arm a timer, notify).

``blurt/assistant/system_actions.py`` is the only module in the assistant that
hands a string the user *spoke* to the operating system. Two of those strings end
up next to an interpreter: the app name goes to ``open``, and the notification
text goes to ``osascript``, which compiles AppleScript source we build by string
formatting. That makes this file the assistant's injection surface, and the
module's docstring makes two promises about it that nothing was checking:

  * "No shell. Every command is passed as an argument *list*", so a spoken name
    like ``foo"; rm -rf ~`` is a (missing) app name and not a command; and
  * every string embedded in AppleScript is escaped so it cannot terminate the
    quoted literal it lives inside and start a new statement.

Both promises are one careless refactor away from being false -- swapping the
argv list for an f-string is a two-line change that no other test would notice --
so the tests below assert the destructive case rather than the happy path: what
must NOT reach the OS, not merely what usually does.

The rest of the file defends the same asymmetry from the other direction. Failing
to open an app costs the user a second and a second attempt; arming a runaway
timer, leaking a Timer thread that can never be cancelled, or letting a failed
notification kill the worker thread mid-dictation costs them something they can't
get back. So:

  * :meth:`TimerService.schedule` must refuse garbage durations *without arming
    anything* -- the tests patch ``threading.Timer`` and assert it was never
    even constructed, because "returned ok=False but armed it anyway" is exactly
    the bug that would otherwise pass a message-only assertion;
  * a scheduled timer must be kept referenced until it fires, or the GC eats it
    mid-wait (the classic threading.Timer footgun) and the timer silently never
    goes off;
  * :func:`notify` must swallow *everything* -- missing binary, timeout, OSError,
    and any surprise exception -- and return ``None``.

Hardware-free by construction: an autouse fixture replaces ``subprocess.run`` on
the module's own ``subprocess`` reference with a detonator that fails the test if
it is ever called for real, so no test here can spawn ``open`` or ``osascript``
even by accident, on a Mac or on Linux CI. ``threading.Timer`` is likewise faked
wherever a timer is armed, so no test starts a thread or waits on wall-clock time.

Python 3.9 floor: lazy annotations, stdlib + pytest only, no PEP 604/585 syntax.
"""

from __future__ import annotations

import subprocess

import pytest

from blurt.assistant import system_actions
from blurt.assistant.system_actions import (
    TimerService,
    _applescript_escape,
    _MAX_TIMER_MINUTES,
    notify,
    open_app,
)


# --------------------------------------------------------------------------- #
# Fakes and fixtures
# --------------------------------------------------------------------------- #
class RecordingRun:
    """Stand-in for :func:`subprocess.run` that records argv and spawns nothing.

    Keeps the *whole* call -- positional argv and every keyword -- because two of
    the properties under test are about the shape of the call rather than its
    result: that the command is a list, and that ``shell=`` is never handed in.
    Returns a real :class:`subprocess.CompletedProcess` so the caller's
    ``proc.returncode`` branch is exercised against the genuine type.
    """

    def __init__(self, returncode: int = 0) -> None:
        self.calls = []  # list of (argv, kwargs)
        self.returncode = returncode

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            args=argv, returncode=self.returncode, stdout="", stderr=""
        )

    @property
    def argv(self):
        assert self.calls, "subprocess.run was never called"
        return self.calls[-1][0]

    @property
    def kwargs(self):
        assert self.calls, "subprocess.run was never called"
        return self.calls[-1][1]


class RecordedTimer:
    """Stand-in for :class:`threading.Timer`: records how it was armed, no thread.

    A real Timer would start a thread that sleeps for minutes, which is both slow
    and non-deterministic; every firing test here drives ``_fire`` directly
    instead. The attributes mirror the ones :meth:`TimerService.schedule` touches
    (``daemon``, ``start``, ``cancel``) so a regression that stops setting
    ``daemon = True`` -- and thereby wedges interpreter shutdown behind a pending
    timer -- shows up as a failed assertion rather than a hung test run.
    """

    def __init__(self, interval, function, args=None, kwargs=None) -> None:
        self.interval = interval
        self.function = function
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.daemon = None
        self.started = False
        self.cancelled = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True


class ExplodingCancelTimer(RecordedTimer):
    """A Timer whose ``cancel()`` raises, standing in for one that already fired."""

    def cancel(self):
        raise RuntimeError("cancel() on a timer that already ran")


@pytest.fixture(autouse=True)
def forbid_real_subprocess(monkeypatch):
    """Make it impossible for this file to spawn a process, even by accident.

    ``system_actions.subprocess`` *is* the stdlib module, so this replaces
    ``subprocess.run`` for the duration of every test in this file. Tests that
    want to observe a call install their own recorder over the top; anything that
    forgets to gets a loud AssertionError instead of a real ``open -a`` on a
    developer's Mac.
    """

    def detonate(*args, **kwargs):
        raise AssertionError(
            "a test spawned a real subprocess: {0!r} {1!r}".format(args, kwargs)
        )

    monkeypatch.setattr(system_actions.subprocess, "run", detonate)


def patch_run(monkeypatch, returncode=0):
    """Install a :class:`RecordingRun` and hand it back for assertions."""
    recorder = RecordingRun(returncode=returncode)
    monkeypatch.setattr(system_actions.subprocess, "run", recorder)
    return recorder


def patch_run_raising(monkeypatch, exc):
    """Make ``subprocess.run`` raise ``exc``, and record that it was reached."""
    calls = []

    def boom(*args, **kwargs):
        calls.append((args, kwargs))
        raise exc

    monkeypatch.setattr(system_actions.subprocess, "run", boom)
    return calls


def patch_timer(monkeypatch, timer_class=RecordedTimer):
    """Replace ``threading.Timer`` with a recorder; return the created-timer list.

    The list is the load-bearing part: the rejection tests assert it stays EMPTY,
    which is the difference between "refused" and "refused but armed it anyway".
    """
    created = []

    def make(interval, function, args=None, kwargs=None):
        timer = timer_class(interval, function, args=args, kwargs=kwargs)
        created.append(timer)
        return timer

    monkeypatch.setattr(system_actions.threading, "Timer", make)
    return created


def assert_no_shell(kwargs):
    """The whole injection defence in one line: no ``shell=True``, ever."""
    assert kwargs.get("shell", False) is False, (
        "the command was handed to a shell; a spoken app name would become code"
    )


def unescaped_quote_count(script):
    """Count the quotes AppleScript would read as string delimiters.

    Strips escaped backslashes first (so ``\\\\`` does not look like an escape
    for the quote that follows it), then escaped quotes; whatever ``"`` remains
    is a character that terminates a literal. A correctly escaped
    ``display notification "..." with title "..."`` has exactly four.
    """
    stripped = script.replace("\\\\", "").replace('\\"', "")
    return stripped.count('"')


# --------------------------------------------------------------------------- #
# open_app: the command is a list, never a shell string
# --------------------------------------------------------------------------- #
def test_open_app_passes_the_name_as_a_list_argument(monkeypatch):
    run = patch_run(monkeypatch)

    open_app("Safari")

    assert run.argv == ["open", "-a", "Safari"]
    assert isinstance(run.argv, list), "argv must be a list, not a shell string"
    assert_no_shell(run.kwargs)


def test_open_app_never_hands_the_command_to_a_shell(monkeypatch):
    run = patch_run(monkeypatch)

    open_app("Safari")

    assert run.kwargs.get("shell", False) is False
    assert not isinstance(run.argv, str), (
        "a string command implies a shell; the app name would be re-parsed as code"
    )


def test_open_app_uses_a_timeout_so_a_stuck_launchservices_cannot_wedge_the_worker(monkeypatch):
    run = patch_run(monkeypatch)

    open_app("Safari")

    assert run.kwargs.get("timeout") is not None
    assert 0 < run.kwargs["timeout"] <= 30


def test_a_malicious_spoken_app_name_stays_one_argv_element(monkeypatch):
    """The docstring's promise: ``foo"; rm -rf ~`` is a name, never a command."""
    run = patch_run(monkeypatch, returncode=1)

    open_app('foo"; rm -rf ~')

    # Exactly three elements: the shell metacharacters are payload inside the
    # third one, not separators. (The sanitizer title-cases the all-lowercase
    # word "rm" on its way through -- cosmetic, and irrelevant to the guarantee.)
    assert len(run.argv) == 3
    assert run.argv[0] == "open"
    assert run.argv[1] == "-a"
    assert run.argv[2] == 'foo"; Rm -rf ~'
    assert_no_shell(run.kwargs)


def test_a_malicious_spoken_app_name_never_becomes_a_command_string(monkeypatch):
    run = patch_run(monkeypatch, returncode=1)

    open_app('foo"; rm -rf ~')

    joined = " ".join(str(part) for part in run.argv)
    assert not isinstance(run.argv, str), "argv collapsed into a shell string"
    # No element may itself be the whole command line -- that is what a regression
    # to `subprocess.run("open -a " + name, shell=True)` would look like.
    for part in run.argv:
        assert part != joined
        assert not part.startswith("open -a")


def test_a_backticked_app_name_is_not_evaluated(monkeypatch):
    run = patch_run(monkeypatch, returncode=1)

    open_app("$(whoami)")

    assert run.argv == ["open", "-a", "$(whoami)"]
    assert_no_shell(run.kwargs)


# --------------------------------------------------------------------------- #
# open_app: spoken-name sanitising (_sanitize_app_name, observed through argv)
# --------------------------------------------------------------------------- #
def test_whitespace_around_and_inside_a_spoken_name_is_collapsed(monkeypatch):
    run = patch_run(monkeypatch)

    open_app("   google    chrome  ")

    assert run.argv == ["open", "-a", "Google Chrome"]


def test_an_all_lowercase_name_is_title_cased_for_the_spoken_reply(monkeypatch):
    run = patch_run(monkeypatch)

    result = open_app("google chrome")

    assert run.argv[2] == "Google Chrome"
    assert result.message == "Opened Google Chrome"


@pytest.mark.parametrize("name", ["iTerm", "VLC", "iTerm2", "GarageBand"])
def test_an_already_cased_name_is_left_exactly_alone(monkeypatch, name):
    """Mangling "iTerm" into "Iterm" would read as a typo in the spoken reply."""
    run = patch_run(monkeypatch)

    open_app(name)

    assert run.argv[2] == name


def test_a_tab_or_newline_in_the_name_is_treated_as_whitespace(monkeypatch):
    run = patch_run(monkeypatch)

    open_app("google\tchrome\n")

    assert run.argv == ["open", "-a", "Google Chrome"]


# --------------------------------------------------------------------------- #
# open_app: an unusable name must not reach the OS at all
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name",
    ["", "   ", "\t\n", None, 42, 3.5, ["Safari"], {"app": "Safari"}],
    ids=["empty", "spaces", "tab-newline", "none", "int", "float", "list", "dict"],
)
def test_an_unusable_app_name_is_refused_without_running_anything(monkeypatch, name):
    run = patch_run(monkeypatch)

    result = open_app(name)

    assert result.ok is False
    assert result.message
    assert run.calls == [], "an empty or non-string name still reached the OS"


def test_an_empty_name_says_what_went_wrong(monkeypatch):
    patch_run(monkeypatch)

    result = open_app("   ")

    assert result.ok is False
    assert "app" in result.message.lower()


# --------------------------------------------------------------------------- #
# open_app: exit codes
# --------------------------------------------------------------------------- #
def test_a_zero_exit_is_reported_as_success(monkeypatch):
    patch_run(monkeypatch, returncode=0)

    result = open_app("Safari")

    assert result.ok is True
    assert "Safari" in result.message


def test_a_nonzero_exit_is_reported_as_failure_naming_the_app(monkeypatch):
    patch_run(monkeypatch, returncode=1)

    result = open_app("Safri")

    assert result.ok is False
    assert "Safri" in result.message, (
        "the user needs the misheard name back to know what to say differently"
    )


def test_a_nonzero_exit_does_not_claim_the_app_was_opened(monkeypatch):
    patch_run(monkeypatch, returncode=255)

    result = open_app("Safari")

    assert result.ok is False
    assert not result.message.startswith("Opened")


# --------------------------------------------------------------------------- #
# open_app: every OS failure becomes a message, never an exception
# --------------------------------------------------------------------------- #
def test_a_missing_open_binary_is_a_message_not_a_crash(monkeypatch):
    """Running off a Mac: there is no ``open``, and blurt must not die for it."""
    patch_run_raising(monkeypatch, FileNotFoundError("open"))

    result = open_app("Safari")

    assert result.ok is False
    assert result.message


def test_a_timeout_opening_an_app_is_a_message_not_a_crash(monkeypatch):
    patch_run_raising(
        monkeypatch, subprocess.TimeoutExpired(cmd=["open", "-a", "Safari"], timeout=10.0)
    )

    result = open_app("Safari")

    assert result.ok is False
    assert "Safari" in result.message


def test_an_oserror_opening_an_app_is_a_message_not_a_crash(monkeypatch):
    patch_run_raising(monkeypatch, OSError("resource temporarily unavailable"))

    result = open_app("Safari")

    assert result.ok is False
    assert "Safari" in result.message


def test_open_app_returns_an_action_result_for_every_failure_mode(monkeypatch):
    """One sweep: no failure path may leak an exception into the worker thread."""
    for exc in (
        FileNotFoundError("open"),
        subprocess.TimeoutExpired(cmd="open", timeout=10.0),
        OSError(13, "Permission denied"),
        PermissionError("no exec"),
    ):
        patch_run_raising(monkeypatch, exc)
        result = open_app("Safari")
        assert result.ok is False, exc


# --------------------------------------------------------------------------- #
# TimerService: a garbage duration must be refused WITHOUT arming anything
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "minutes",
    [
        0,
        0.0,
        -1,
        -0.5,
        -1440,
        float("nan"),
        float("inf"),
        float("-inf"),
        "soon",
        "",
        None,
        object(),
        _MAX_TIMER_MINUTES + 1,
        200000,
        1e18,
    ],
    ids=[
        "zero-int", "zero-float", "negative", "negative-fraction", "negative-day",
        "nan", "inf", "neg-inf", "words", "empty-string", "none", "object",
        "over-max", "misparsed-clock-time", "absurd",
    ],
)
def test_a_garbage_duration_arms_no_timer_at_all(monkeypatch, minutes):
    """The failure being defended against is a runaway timer, not a bad message.

    Asserting only ``ok is False`` would still pass if ``schedule`` armed the
    timer and *then* reported failure, leaving a daemon thread that nothing holds
    a reference to and nothing can cancel. So the real assertion is that
    ``threading.Timer`` was never constructed.
    """
    created = patch_timer(monkeypatch)
    service = TimerService()

    result = service.schedule(minutes, "tea")

    assert result.ok is False
    assert result.message
    assert created == [], "a rejected duration still armed a timer"
    assert service._timers == {}, "a rejected duration left an entry in the registry"


def test_a_rejected_duration_does_not_consume_a_timer_id(monkeypatch):
    """Bookkeeping must not drift on the rejection path."""
    patch_timer(monkeypatch)
    service = TimerService()

    service.schedule(-5, "tea")
    service.schedule(float("nan"), "tea")

    assert service._counter == 0


def test_the_maximum_accepted_duration_is_exactly_a_day(monkeypatch):
    created = patch_timer(monkeypatch)
    service = TimerService()

    ok_at_limit = service.schedule(_MAX_TIMER_MINUTES, "wake up")
    over_limit = service.schedule(_MAX_TIMER_MINUTES + 0.001, "wake up")

    assert ok_at_limit.ok is True
    assert over_limit.ok is False
    assert len(created) == 1, "the over-limit request armed a second timer"


# --------------------------------------------------------------------------- #
# TimerService: a valid duration arms one daemon timer and keeps it alive
# --------------------------------------------------------------------------- #
def test_a_valid_duration_arms_exactly_one_daemon_timer(monkeypatch):
    created = patch_timer(monkeypatch)
    service = TimerService()

    result = service.schedule(5, "tea")

    assert result.ok is True
    assert len(created) == 1
    timer = created[0]
    assert timer.interval == pytest.approx(300.0)
    assert timer.daemon is True, (
        "a non-daemon timer thread would hold interpreter shutdown open for the "
        "whole delay"
    )
    assert timer.started is True


def test_the_armed_timer_is_wired_to_fire_with_its_own_id_and_label(monkeypatch):
    created = patch_timer(monkeypatch)
    service = TimerService()

    service.schedule(5, "  tea  ")

    timer = created[0]
    assert timer.function == service._fire
    timer_id, label = timer.args
    assert timer_id in service._timers
    assert label == "tea", "the spoken label should be stripped before it is stored"


def test_a_scheduled_timer_is_kept_referenced_so_the_gc_cannot_eat_it(monkeypatch):
    """The classic threading.Timer footgun: drop the reference, lose the timer.

    A dropped Timer is the worst kind of failure here -- silent. The user is told
    "Timer set for 5 minutes" and then nothing ever happens.
    """
    created = patch_timer(monkeypatch)
    service = TimerService()

    service.schedule(5, "tea")

    assert service._timers, "nothing holds the armed timer; it can be collected"
    assert list(service._timers.values()) == created


def test_two_timers_get_two_registry_slots(monkeypatch):
    created = patch_timer(monkeypatch)
    service = TimerService()

    service.schedule(5, "tea")
    service.schedule(10, "pasta")

    assert len(created) == 2
    assert len(service._timers) == 2, "a second timer overwrote the first"


@pytest.mark.parametrize(
    "minutes,expected",
    [
        (1, "1 minute"),
        (1.0, "1 minute"),
        (2, "2 minutes"),
        (5, "5 minutes"),
        (2.5, "2.5 minutes"),
        (0.5, "0.5 minutes"),
        (90, "90 minutes"),
    ],
)
def test_the_confirmation_reads_back_the_duration_with_the_right_plural(
    monkeypatch, minutes, expected
):
    """"1 minutes" is the sort of thing a user notices and stops trusting."""
    patch_timer(monkeypatch)
    service = TimerService()

    result = service.schedule(minutes, "")

    assert result.ok is True
    assert result.message == "Timer set for {0}".format(expected)


def test_the_confirmation_includes_the_label_when_one_was_spoken(monkeypatch):
    patch_timer(monkeypatch)
    service = TimerService()

    result = service.schedule(5, "tea")

    assert "5 minutes" in result.message
    assert "tea" in result.message


# --------------------------------------------------------------------------- #
# TimerService: firing (driven directly -- no sleeping, no thread scheduling)
# --------------------------------------------------------------------------- #
def test_a_firing_timer_notifies_and_then_forgets_itself(monkeypatch):
    notes = []
    monkeypatch.setattr(
        system_actions, "notify", lambda title, message: notes.append((title, message))
    )
    patch_timer(monkeypatch)
    service = TimerService()
    service.schedule(5, "tea")
    timer_id = list(service._timers)[0]

    service._fire(timer_id, "tea")

    assert notes == [("Timer", "tea - time's up")]
    assert service._timers == {}, "a fired timer stayed in the registry forever"


def test_a_firing_timer_with_no_label_still_says_something(monkeypatch):
    notes = []
    monkeypatch.setattr(
        system_actions, "notify", lambda title, message: notes.append((title, message))
    )
    patch_timer(monkeypatch)
    service = TimerService()
    service.schedule(5, "")
    timer_id = list(service._timers)[0]

    service._fire(timer_id, "")

    assert len(notes) == 1
    assert notes[0][1]


def test_one_timer_firing_leaves_the_other_armed(monkeypatch):
    monkeypatch.setattr(system_actions, "notify", lambda title, message: None)
    patch_timer(monkeypatch)
    service = TimerService()
    service.schedule(5, "tea")
    service.schedule(10, "pasta")
    first_id = sorted(service._timers)[0]

    service._fire(first_id, "tea")

    assert len(service._timers) == 1
    assert first_id not in service._timers


def test_a_notify_that_blows_up_still_clears_the_registry(monkeypatch):
    """``notify`` promises never to raise, but ``_fire``'s finally must not rely
    on that promise -- a leaked registry entry is an object that never dies."""

    def boom(title, message):
        raise RuntimeError("osascript exploded in a way notify did not catch")

    monkeypatch.setattr(system_actions, "notify", boom)
    patch_timer(monkeypatch)
    service = TimerService()
    service.schedule(5, "tea")
    timer_id = list(service._timers)[0]

    with pytest.raises(RuntimeError):
        service._fire(timer_id, "tea")

    assert service._timers == {}


def test_firing_an_unknown_timer_id_is_harmless(monkeypatch):
    monkeypatch.setattr(system_actions, "notify", lambda title, message: None)
    service = TimerService()

    service._fire(9999, "ghost")

    assert service._timers == {}


# --------------------------------------------------------------------------- #
# TimerService.cancel_all
# --------------------------------------------------------------------------- #
def test_cancel_all_cancels_every_pending_timer_and_empties_the_registry(monkeypatch):
    created = patch_timer(monkeypatch)
    service = TimerService()
    service.schedule(5, "tea")
    service.schedule(10, "pasta")
    service.schedule(15, "laundry")

    service.cancel_all()

    assert all(timer.cancelled for timer in created)
    assert service._timers == {}


def test_cancel_all_survives_a_timer_that_already_fired(monkeypatch):
    """Cancelling is best-effort; a Timer that already ran must not crash shutdown."""
    patch_timer(monkeypatch, timer_class=ExplodingCancelTimer)
    service = TimerService()
    service.schedule(5, "tea")
    service.schedule(10, "pasta")

    service.cancel_all()  # must not raise

    assert service._timers == {}


def test_cancel_all_on_an_idle_service_is_a_no_op():
    service = TimerService()

    service.cancel_all()
    service.cancel_all()

    assert service._timers == {}


def test_cancel_all_after_a_timer_fired_still_clears_the_rest(monkeypatch):
    monkeypatch.setattr(system_actions, "notify", lambda title, message: None)
    created = patch_timer(monkeypatch)
    service = TimerService()
    service.schedule(5, "tea")
    service.schedule(10, "pasta")
    fired_id = sorted(service._timers)[0]
    service._fire(fired_id, "tea")

    service.cancel_all()

    assert service._timers == {}
    assert created[1].cancelled is True


# --------------------------------------------------------------------------- #
# _applescript_escape: the second injection surface
# --------------------------------------------------------------------------- #
def test_a_double_quote_is_escaped():
    assert _applescript_escape('say "hi"') == 'say \\"hi\\"'


def test_a_backslash_is_escaped_before_the_quotes_are():
    # A lone backslash must double, or it would escape whatever follows it in the
    # generated source -- including our own closing quote.
    assert _applescript_escape("C:\\path") == "C:\\\\path"
    assert _applescript_escape('a\\b"c') == 'a\\\\b\\"c'


def test_a_trailing_backslash_cannot_swallow_the_closing_quote():
    escaped = _applescript_escape("ends with a backslash \\")
    assert escaped.endswith("\\\\")


def test_a_newline_is_flattened_to_a_space():
    assert _applescript_escape("line one\nline two") == "line one line two"


def test_a_carriage_return_is_flattened_to_a_space():
    assert _applescript_escape("line one\rline two") == "line one line two"
    assert _applescript_escape("a\r\nb") == "a  b"


def test_no_raw_newline_survives_escaping():
    """A raw newline inside a one-line ``-e`` literal is a syntax error at best."""
    escaped = _applescript_escape("a\nb\r\nc\rd")
    assert "\n" not in escaped
    assert "\r" not in escaped


def test_a_non_string_is_coerced_rather_than_raising():
    assert _applescript_escape(42) == "42"
    assert _applescript_escape(None) == "None"


def test_ordinary_text_is_left_alone():
    assert _applescript_escape("Timer - time's up") == "Timer - time's up"


# --------------------------------------------------------------------------- #
# notify: the escaped message cannot break out of the AppleScript literal
# --------------------------------------------------------------------------- #
def test_a_notification_payload_cannot_close_the_applescript_string(monkeypatch):
    """The AppleScript equivalent of the shell-injection test above.

    ``" & do shell script "whoami`` is the canonical break-out: close the string,
    concatenate, and run a command. Escaped, both quotes are inert, and the only
    delimiters left in the generated source are the four we wrote ourselves.
    """
    run = patch_run(monkeypatch)

    notify("Timer", 'done " & do shell script "whoami')

    script = run.argv[2]
    assert '\\" & do shell script \\"whoami' in script
    assert '" & do shell script "' not in script
    assert unescaped_quote_count(script) == 4, (
        "the payload introduced an unescaped quote; it can terminate the literal "
        "and start a new AppleScript statement"
    )


def test_a_malicious_title_cannot_close_the_applescript_string(monkeypatch):
    run = patch_run(monkeypatch)

    notify('x" & do shell script "id', "harmless body")

    script = run.argv[2]
    assert unescaped_quote_count(script) == 4


@pytest.mark.parametrize(
    "payload",
    [
        'back\\slash and a " quote',
        "ends with a backslash \\",
        '\\" & do shell script "id',
        "\\\\\\",
    ],
    ids=["mixed", "trailing", "pre-escaped-breakout", "only-backslashes"],
)
def test_a_backslash_heavy_payload_still_leaves_four_delimiters(monkeypatch, payload):
    """Backslashes are the subtle half of the escape.

    A payload ending in ``\\`` is the interesting one: unescaped, it would eat
    our own closing quote and the literal would run on into the rest of the
    program. Doubling it first is what keeps the delimiter count at four.
    """
    run = patch_run(monkeypatch)

    notify("Timer", payload)

    assert unescaped_quote_count(run.argv[2]) == 4


def test_notify_passes_argv_as_a_list_and_never_through_a_shell(monkeypatch):
    run = patch_run(monkeypatch)

    notify("Timer", "time's up")

    assert isinstance(run.argv, list)
    assert run.argv[0] == "osascript"
    assert run.argv[1] == "-e"
    assert len(run.argv) == 3
    assert_no_shell(run.kwargs)


def test_notify_builds_the_expected_applescript(monkeypatch):
    run = patch_run(monkeypatch)

    notify("Timer", "tea - time's up")

    assert run.argv[2] == (
        'display notification "tea - time\'s up" with title "Timer"'
    )


def test_notify_uses_a_timeout_so_a_stuck_osascript_cannot_wedge_the_worker(monkeypatch):
    run = patch_run(monkeypatch)

    notify("Timer", "time's up")

    assert run.kwargs.get("timeout") is not None
    assert 0 < run.kwargs["timeout"] <= 30


# --------------------------------------------------------------------------- #
# notify: never raises, whatever the OS does
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "exc",
    [
        FileNotFoundError("osascript"),
        subprocess.TimeoutExpired(cmd="osascript", timeout=10.0),
        OSError(24, "Too many open files"),
        PermissionError("not permitted"),
        RuntimeError("something nobody anticipated"),
        ValueError("embedded null byte"),
        UnicodeEncodeError("utf-8", "x", 0, 1, "boom"),
    ],
    ids=[
        "missing-binary", "timeout", "oserror", "permission",
        "arbitrary", "valueerror", "unicode",
    ],
)
def test_notify_swallows_every_ordinary_failure_and_returns_none(monkeypatch, exc):
    """A failed notification must never take the dictation down with it."""
    calls = patch_run_raising(monkeypatch, exc)

    assert notify("Timer", "time's up") is None
    assert len(calls) == 1, "the failure happened before osascript was even reached"


def test_notify_returns_none_on_success_too(monkeypatch):
    patch_run(monkeypatch)

    assert notify("Timer", "time's up") is None


def test_notify_returns_none_when_osascript_exits_nonzero(monkeypatch):
    patch_run(monkeypatch, returncode=1)

    assert notify("Timer", "time's up") is None


@pytest.mark.parametrize(
    "title,message",
    [
        (None, None),
        (None, "body"),
        ("Timer", None),
        (42, 3.5),
        ("", ""),
    ],
    ids=["both-none", "title-none", "message-none", "numbers", "both-empty"],
)
def test_notify_never_raises_on_odd_arguments(monkeypatch, title, message):
    run = patch_run(monkeypatch)

    assert notify(title, message) is None
    assert unescaped_quote_count(run.argv[2]) == 4
