"""Tests for the ``blurt learn`` command.

``blurt/__main__.py`` had no test coverage before the learning loop was added, and
this command is the one subcommand that WRITES -- it can rewrite the user's
config. So the properties defended here are mostly about restraint:

  * The default invocation changes nothing.
  * ``--yes`` applies high-confidence findings and nothing else, ever.
  * An override passed for one run (``--cleanup standard``) is never persisted by
    a later ``--apply``. That would silently promote a temporary flag into a
    permanent setting.
  * A user who declines everything gets their config left alone entirely.

The interactive loop is tested by driving :func:`_collect_approvals` with a fake
``input``, rather than through a pty: the branch that matters is which answers map
to which decision, and a pty would test the terminal rather than the decision.
"""

from __future__ import annotations

import json

import pytest

from blurt.__main__ import main
from blurt.config import Config, load_config, save_config
from blurt.history import HistoryRecord, append_record
from blurt.learn import Suggestion, analyze


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


#: Deliberately shaped to produce findings of BOTH confidences, so the tests that
#: assert "--yes touches only the high ones" are not vacuously true:
#:
#:   high   -- "github"/"GitHub"/"Github" variance, and "Priya" capitalized
#:             mid-sentence and never lowercase there.
#:   medium -- "kubernetis" sitting one edit from a "kubernetes" said six times,
#:             which is the near-miss rule's exact shape.
#:
#: Each line is journalled twice by the fixture, so occurrence counts are double
#: the line counts here.
_JOURNAL = [
    "the github action failed on the staging cluster",
    "Priya said the GitHub token expired",
    "the Github issue is still open",
    "ask Priya about the deploy, and tell Priya it is done",
    "we should ask Priya once github is back",
    "the kubernetes rollout finished cleanly",
    "restart kubernetes and check the pods",
    "kubernetes scheduling is fine, the disk is slow",
    "the kubernetis node is flapping again",
]


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An isolated XDG home, so no test can touch the developer's real config."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def seeded(home):
    """A journal with findings in it, and journalling switched on."""
    from blurt.history import default_history_path

    for index, line in enumerate(_JOURNAL * 2):
        append_record(
            HistoryRecord(
                timestamp=1_770_000_000.0 + index * 60.0,
                mode="dictate",
                raw=line,
                cleaned=line.capitalize() + ".",
                engine="faster-whisper base.en",
                audio_seconds=3.0,
                latency_seconds=1.0,
            ),
            limit=0,
        )
    save_config(Config(history_enabled=True))
    return default_history_path()


def _config_on_disk():
    from blurt.config import default_config_path

    return json.loads(default_config_path().read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# The default invocation is read-only
# --------------------------------------------------------------------------- #


def test_learn_with_no_journal_explains_the_feature(home, capsys):
    assert main(["learn"]) == 0
    out = capsys.readouterr().out
    assert "OFF" in out
    assert "history_enabled" in out
    # The cost must be stated in the same breath as the benefit.
    assert "disk" in out
    assert "--forget" in out


def test_learn_with_no_journal_writes_nothing(home):
    from blurt.config import default_config_path

    main(["learn"])
    assert not default_config_path().exists()


def test_learn_reports_findings(seeded, capsys):
    assert main(["learn"]) == 0
    out = capsys.readouterr().out
    assert "github -> GitHub" in out
    assert "Priya" in out
    assert "blurt learn --apply" in out


def test_learn_without_apply_never_writes(seeded):
    before = _config_on_disk()
    main(["learn"])
    assert _config_on_disk() == before


def test_learn_reports_dictionary_health(home, capsys):
    save_config(Config(history_enabled=True, dictionary={"grafana": "Grafana"}))
    append_record(
        HistoryRecord(1.0, "dictate", "hello there", "Hello there.", "e", 1.0, 1.0),
        limit=0,
    )
    main(["learn"])
    out = capsys.readouterr().out
    assert "never matched" in out
    assert "grafana" in out


def test_learn_min_threshold_is_honoured(seeded, capsys):
    main(["learn", "--min", "99"])
    assert "SUGGESTIONS" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# --apply --yes
# --------------------------------------------------------------------------- #


def test_apply_yes_writes_high_confidence_findings(seeded, capsys):
    assert main(["learn", "--apply", "--yes"]) == 0
    config = _config_on_disk()
    assert config["dictionary"]["github"] == "GitHub"
    assert "Priya" in config["initial_prompt"]


def test_apply_yes_never_writes_a_medium_finding(seeded):
    from blurt.history import load_records

    report = analyze(load_records())
    medium = {item.key for item in report.suggestions if item.confidence == "medium"}
    assert medium, "this test proves nothing if the journal produced no medium findings"

    main(["learn", "--apply", "--yes"])
    config = _config_on_disk()

    for key in medium:
        assert key not in config["dictionary"], (
            "a medium-confidence guess reached the config unattended"
        )


def test_apply_yes_says_what_it_skipped(seeded, capsys):
    main(["learn", "--apply", "--yes"])
    out = capsys.readouterr().out
    assert "high-confidence" in out
    assert "next time you start blurt" in out


def test_apply_is_idempotent(seeded):
    main(["learn", "--apply", "--yes"])
    first = _config_on_disk()
    main(["learn", "--apply", "--yes"])
    assert _config_on_disk() == first


def test_apply_preserves_unrelated_settings(home, seeded):
    save_config(Config(history_enabled=True, hotkey="right_ctrl", preroll_ms=250))
    main(["learn", "--apply", "--yes"])
    config = _config_on_disk()
    assert config["hotkey"] == "right_ctrl"
    assert config["preroll_ms"] == 250


def test_apply_never_overwrites_an_existing_dictionary_entry(home, seeded):
    save_config(Config(history_enabled=True, dictionary={"github": "GITHUB"}))
    main(["learn", "--apply", "--yes"])
    assert _config_on_disk()["dictionary"]["github"] == "GITHUB"


def test_a_one_run_override_is_never_persisted(seeded):
    """--cleanup applies to this run only; --apply must not write it to disk."""
    assert main(["--cleanup", "standard", "learn", "--apply", "--yes"]) == 0
    assert _config_on_disk()["cleanup_level"] == "light"


def test_a_model_override_is_never_persisted(seeded):
    main(["learn", "--apply", "--yes", "--model", "tiny.en"])
    assert _config_on_disk()["model"] == "auto"


def test_apply_with_nothing_to_apply_is_clean(home, capsys):
    save_config(Config(history_enabled=True))
    append_record(
        HistoryRecord(1.0, "dictate", "hello there", "Hello there.", "e", 1.0, 1.0),
        limit=0,
    )
    assert main(["learn", "--apply", "--yes"]) == 0


# --------------------------------------------------------------------------- #
# --apply without a terminal
# --------------------------------------------------------------------------- #


def test_apply_without_a_tty_refuses(seeded, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert main(["learn", "--apply"]) == 1
    assert "--yes" in capsys.readouterr().err


def test_apply_without_a_tty_writes_nothing(seeded, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    before = _config_on_disk()
    main(["learn", "--apply"])
    assert _config_on_disk() == before


# --------------------------------------------------------------------------- #
# The interactive approval loop
# --------------------------------------------------------------------------- #


def _suggestions(count=3):
    return [
        Suggestion("dictionary", "key%d" % i, "Value%d" % i, "because", 3, "high")
        for i in range(count)
    ]


def _drive(monkeypatch, answers, suggestions):
    """Run _collect_approvals against a scripted user."""
    from blurt import __main__ as cli
    from blurt.learn import Report

    replies = iter(answers)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(cli, "input", lambda _prompt="": next(replies), raising=False)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(replies))

    report = Report(records=10, suggestions=tuple(suggestions))
    return cli._collect_approvals(report, unattended=False)


def test_y_accepts_one(monkeypatch):
    accepted = _drive(monkeypatch, ["y", "n", "n"], _suggestions(3))
    assert [item.key for item in accepted] == ["key0"]


def test_n_is_the_default_for_a_bare_return(monkeypatch):
    accepted = _drive(monkeypatch, ["", "", ""], _suggestions(3))
    assert accepted == []


def test_anything_unrecognized_is_a_decline(monkeypatch):
    accepted = _drive(monkeypatch, ["maybe", "sure?", "!"], _suggestions(3))
    assert accepted == []


def test_a_accepts_the_rest(monkeypatch):
    accepted = _drive(monkeypatch, ["n", "a"], _suggestions(4))
    assert [item.key for item in accepted] == ["key1", "key2", "key3"]


def test_q_stops_and_keeps_what_came_before(monkeypatch):
    accepted = _drive(monkeypatch, ["y", "q"], _suggestions(4))
    assert [item.key for item in accepted] == ["key0"]


def test_eof_is_treated_as_stop_not_as_yes(monkeypatch):
    """A closed stdin must never be read as blanket approval."""
    from blurt import __main__ as cli

    def raise_eof(_prompt=""):
        raise EOFError

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", raise_eof)

    from blurt.learn import Report

    accepted = cli._collect_approvals(
        Report(records=10, suggestions=tuple(_suggestions(3))), unattended=False
    )
    assert accepted == []


def test_ctrl_c_is_treated_as_stop(monkeypatch):
    from blurt import __main__ as cli
    from blurt.learn import Report

    def raise_interrupt(_prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", raise_interrupt)

    accepted = cli._collect_approvals(
        Report(records=10, suggestions=tuple(_suggestions(2))), unattended=False
    )
    assert accepted == []


def test_unattended_takes_only_high_confidence(capsys):
    from blurt import __main__ as cli
    from blurt.learn import Report

    mixed = [
        Suggestion("dictionary", "a", "A", "r", 3, "high"),
        Suggestion("dictionary", "b", "B", "r", 3, "medium"),
        Suggestion("prompt", "c", "C", "r", 3, "high"),
    ]
    accepted = cli._collect_approvals(
        Report(records=10, suggestions=tuple(mixed)), unattended=True
    )
    assert [item.key for item in accepted] == ["a", "c"]


# --------------------------------------------------------------------------- #
# --forget
# --------------------------------------------------------------------------- #


def test_forget_deletes_the_journal(seeded, capsys):
    assert seeded.exists()
    assert main(["learn", "--forget"]) == 0
    assert not seeded.exists()


def test_forget_is_honest_about_not_being_a_secure_erase(seeded, capsys):
    main(["learn", "--forget"])
    out = capsys.readouterr().out
    assert "not a secure erase" in out


def test_forget_with_no_journal_is_not_an_error(home, capsys):
    assert main(["learn", "--forget"]) == 0
    assert "No journal" in capsys.readouterr().out


def test_forget_leaves_the_config_alone(seeded):
    before = _config_on_disk()
    main(["learn", "--forget"])
    assert _config_on_disk() == before


def test_forget_wins_over_apply(seeded):
    """--forget is destructive and unambiguous; it must not also apply anything."""
    from blurt.config import default_config_path

    main(["learn", "--forget", "--apply", "--yes"])
    assert not seeded.exists()
    assert _config_on_disk()["dictionary"] == {}
