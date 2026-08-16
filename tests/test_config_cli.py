"""Tests for ``blurt config get`` and ``blurt config set``.

``config set`` is the second command in blurt that writes the user's config file,
and it is the one people will reach for casually -- it exists so that turning on
the transcript journal does not require hand-authoring JSON at a path that does
not exist yet. Casual is exactly what makes it dangerous. Someone typing
``blurt config set history_enabled ture`` at 11pm has a config file that already
holds a dictionary built from months of their own speech, and the only acceptable
outcome of that typo is a message and an untouched file.

So the properties defended here are almost all negative:

  * The bare ``blurt config`` is still the read-only thing it always was.
    ``get``/``set`` were added as subcommands rather than as ``--get``/``--set``
    flags precisely so that someone typing ``blurt config`` to look at their
    settings can never discover they have changed something, and that has to keep
    being true.
  * A value that cannot be parsed, cannot be validated, or would not survive
    ``load_config`` writes NOTHING. Not a partial file, not a rewritten file with
    the other fields reformatted -- nothing. The tests assert the file is
    byte-identical across the failed command, and that a config file which did
    not exist beforehand still does not exist afterwards, because "no file yet"
    is the normal state on a fresh install and creating one full of defaults
    would be a silent change of its own.
  * A one-run override (``--cleanup standard``, ``--model tiny.en``) is never
    persisted by a later write. ``test_learn_cli.py`` defends the same guarantee
    for ``learn --apply``; both commands reload the config from disk rather than
    reusing the one ``main`` resolved, and both need a test that fails loudly if
    someone ever "simplifies" that away. Promoting a flag someone passed to try
    something into a permanent setting is unrecoverable in the way that matters:
    the user has no reason to ever look for it.
  * Setting one field leaves every other field exactly as it was -- and "exactly"
    now means the raw bytes' worth of meaning, not "every field a ``Config`` can
    hold". ``set`` MERGES the one changed key into the JSON object on disk
    instead of round-tripping the file through the dataclass, and the section
    below pins the three things that round trip used to destroy in silence:
    keys this version of blurt has never heard of (``load_config`` ignores them
    on purpose, for forward compatibility -- ignoring is not the same as
    deleting), values ``load_config`` rejects (overwriting a typo'd
    ``sample_rate`` with the default erases the evidence the user needs to find
    it), and a file that does not parse at all (the one and only copy of
    whatever they wrote in it).
  * A ``set`` that would leave both hotkeys on the same physical key is refused.
    A collision does not fail loudly at runtime; ``BlurtApp._build_assistant``
    resolves it by switching assistant mode off, which also removes the spoken
    undo, so the one command that could put a user there must not.
  * ``get`` writes one line to stdout and NOTHING to stderr -- and every
    assertion about that is made against a file blurt itself wrote, never
    against a hand-authored one. See the section banner above
    ``test_get_with_no_config_file_says_nothing_on_stderr`` for why that
    distinction is the whole test rather than a stylistic preference.

Everything runs against an isolated ``XDG_CONFIG_HOME`` (the ``home`` fixture,
shared with ``test_learn_cli.py`` from ``conftest.py``) so no test can reach the
developer's real config, and the CLI is driven through :func:`blurt.__main__.main`
with an argv list rather than through a subprocess: the exit status and the
bytes on disk are the whole contract, and a subprocess would only add a Python
startup per assertion.

Python 3.9 floor: lazy annotations, typing generics, no PEP 604 unions.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import stat
from typing import Any, Dict, List, Optional, Tuple

import pytest

from blurt.__main__ import main
from blurt.config import (
    VALID_CLEANUP_LEVELS,
    VALID_ENGINES,
    Config,
    default_config_path,
    load_config,
    save_config,
)
from blurt.hotkey import SUPPORTED_HOTKEYS

# Shared with tests/test_learn_cli.py, which asks the same question of the same
# file. The `home` fixture is shared through conftest.py as well, and needs no
# import here: pytest resolves a conftest fixture by name.
from conftest import config_on_disk


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _snapshot() -> Optional[bytes]:
    """The exact bytes of the config file, or None when there is no file.

    Bytes rather than parsed JSON, and ``None`` rather than ``{}`` for the
    missing case, because the thing being defended is stronger than "the setting
    did not change". A refused command must not reformat the file, must not
    reorder its keys, must not add fields the user never set, and above all must
    not bring the file into existence -- on a fresh install there is no config
    file at all, and writing one full of defaults would be a change the user did
    not ask for and would never think to look for.
    """
    path = default_config_path()
    if not path.exists():
        return None
    return path.read_bytes()


def _refused(argv: List[str]) -> int:
    """Run a ``config set`` that must fail, and prove the file did not move.

    Returns the exit status so callers still assert on it -- the point is that
    both halves hold at once. A command that returns 2 and writes anyway is the
    exact failure this file exists to catch, and it is invisible from the exit
    status alone.
    """
    before = _snapshot()
    status = main(argv)
    assert _snapshot() == before, "a refused value changed the config file"
    return status


def _write_raw(document: Dict[str, Any]) -> bytes:
    """Put a config file on disk exactly as given, bypassing ``save_config``.

    ``save_config`` can only express a :class:`Config`, and every file that
    matters in the merge tests below is one a ``Config`` cannot represent: a key
    from a newer blurt, or a value this version's loader refuses. Neither is
    hypothetical -- the first is what forward compatibility produces by design,
    and the second is what a typo produces -- but both are unreachable through
    the dataclass, so they have to be written by hand.

    Returns the bytes written, so a caller can assert against them directly.
    """
    path = default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return payload


def _write_bytes(raw: bytes) -> bytes:
    """Put arbitrary bytes at the config path. For files that are not JSON."""
    path = default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _plant_config(kind: str, capsys) -> None:
    """Put a config file on disk, either by hand or by asking blurt to write one.

    The two are not interchangeable, and the difference is what the stderr
    section below is really testing. A hand-planted file contains the values
    whoever wrote the test chose to name; the file ``config set`` leaves behind
    contains every field, at whatever value the user has not touched -- which
    for ``initial_prompt`` means ``""``, the value that used to make the next
    ``get`` print a warning. A test author picking field values by hand has no
    reason to think about the field they are not testing, so the planted file
    reliably dodges exactly the problem nobody was thinking about.

    Drains capsys, so the caller's assertions see only the command under test.
    """
    if kind == "hand-planted":
        save_config(_SEEDED)
    elif kind == "written by config set":
        assert main(["config", "set", "history_enabled", "true"]) == 0
    else:  # pragma: no cover - a typo in a parametrize list, not a code path
        raise AssertionError("unknown seed kind %r" % (kind,))
    capsys.readouterr()


#: Both ways of getting a config file onto disk, for the assertions that must
#: hold against the product's own output and not only against a fixture.
_SEED_KINDS: List[str] = ["hand-planted", "written by config set"]


def _config_dir_entries() -> List[str]:
    """Every name in the config directory, sorted.

    Used to assert the absence of things nobody should find there: a ``.bak``
    the user never asked for, or a leftover ``.config.json.XXXX.tmp`` from a
    write that failed halfway.
    """
    return sorted(entry.name for entry in default_config_path().parent.iterdir())


#: One command-line word per scalar setting, chosen so that every value differs
#: from that field's default -- otherwise the round-trip test would pass just as
#: happily against a `set` that did nothing at all.
#:
#: Driven from ``dataclasses.fields(Config)`` rather than written out again as a
#: literal list of names (see the coverage test below) so that adding a field to
#: Config fails this file loudly instead of leaving the new setting untested.
_ROUND_TRIP: Dict[str, Tuple[str, Any]] = {
    "engine": ("faster-whisper", "faster-whisper"),
    "model": ("tiny.en", "tiny.en"),
    "hotkey": ("right_ctrl", "right_ctrl"),
    "cleanup_level": ("standard", "standard"),
    "sample_rate": ("22050", 22050),
    "preroll_ms": ("250", 250),
    "min_hold_ms": ("150", 150),
    "paste_delay_ms": ("90", 90),
    "clipboard_restore_ms": ("300", 300),
    "cpu_threads": ("4", 4),
    "keep_raw_history": ("false", False),
    "initial_prompt": ("Priya Kubernetes Grafana", "Priya Kubernetes Grafana"),
    "assistant_enabled": ("false", False),
    "assistant_hotkey": ("left_ctrl", "left_ctrl"),
    "history_enabled": ("true", True),
    "history_limit": ("500", 500),
}

#: Fields that cannot come from a single command-line word. Today just the
#: replacement dictionary, which has its own command (`blurt learn --apply`) and
#: its own test below.
_NOT_SETTABLE = frozenset({"dictionary"})


# --------------------------------------------------------------------------- #
# The bare form is still read-only
# --------------------------------------------------------------------------- #


def test_bare_config_still_prints_everything_and_exits_zero(home, capsys):
    assert main(["config"]) == 0
    out = capsys.readouterr().out
    assert "path" in out
    assert "resolved settings" in out
    # The whole resolved config is printed as JSON, so every field is in there.
    assert "history_enabled" in out
    assert "cleanup_level" in out


def test_bare_config_creates_no_file(home):
    """Looking at your settings must never be the thing that writes them."""
    main(["config"])
    assert not default_config_path().exists()


def test_bare_config_leaves_an_existing_file_byte_identical(home):
    save_config(Config(history_enabled=True, hotkey="right_ctrl"))
    before = _snapshot()
    assert main(["config"]) == 0
    assert _snapshot() == before


# --------------------------------------------------------------------------- #
# config get
# --------------------------------------------------------------------------- #


def test_get_prints_a_bool_as_the_word_the_config_file_uses(home, capsys):
    assert main(["config", "get", "history_enabled"]) == 0
    # Exactly one line, no label, no quotes: `get` output is meant to be piped
    # straight back into `set`, and "history_enabled: false" is not.
    assert capsys.readouterr().out == "false\n"


def test_get_prints_an_int_bare(home, capsys):
    assert main(["config", "get", "sample_rate"]) == 0
    assert capsys.readouterr().out == "16000\n"


def test_get_prints_a_str_unquoted(home, capsys):
    assert main(["config", "get", "hotkey"]) == 0
    assert capsys.readouterr().out == "right_option\n"


def test_get_reads_back_what_set_wrote(home, capsys):
    """The real round trip, and both streams of it.

    Asserting only ``.out`` here is how the empty-``initial_prompt`` warning
    lived through a file full of tests named for stderr: this is the one test
    that ran against the bytes ``config set`` actually writes, and it was
    looking the other way. The write's own output is checked too -- a ``set``
    that succeeds has nothing to say on stderr, and if it did, ``get`` would be
    inheriting a file blurt had already complained about.
    """
    assert main(["config", "set", "cleanup_level", "standard"]) == 0
    assert capsys.readouterr().err == ""
    assert main(["config", "get", "cleanup_level"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "standard\n"
    assert captured.err == ""


def test_get_on_an_unknown_key_exits_two_and_lists_the_real_ones(home, capsys):
    assert main(["config", "get", "hisory_enabled"]) == 2
    err = capsys.readouterr().err
    assert "unknown setting" in err
    # Naming the valid keys is the whole point: a typo'd key is almost always a
    # near miss, and the answer is already on screen.
    assert "history_enabled" in err
    assert "cleanup_level" in err


def test_get_never_creates_the_config_file(home):
    main(["config", "get", "history_enabled"])
    main(["config", "get", "nonsense"])
    assert not default_config_path().exists()


# --------------------------------------------------------------------------- #
# config get writes one line to stdout and leaves stderr alone
#
# `get` exists to be captured: `HOTKEY=$(blurt config get hotkey)`. That makes
# both streams part of its contract rather than decoration. A second line on
# stdout silently changes the value the shell assigns, and anything on stderr
# lands on the user's terminal in the middle of a script that looked like it
# worked -- the two failures a `get` is least able to explain for itself.
#
# EVERY ASSERTION HERE IS MADE AGAINST A FILE BLURT WROTE. That rule is not
# fastidiousness; it is the difference between these tests working and these
# tests having already failed once. The version of
# `test_get_with_a_config_file_present_says_nothing_on_stderr` that shipped
# planted `initial_prompt="Priya, Grafana"` -- a value chosen to look like a
# realistic user's, and the only kind of value that could not produce the
# warning the test was named for. config.py's `_pick_str` warned "initial_prompt
# is empty" for a field whose default is ITSELF empty, and every file
# `save_config` or `config set` writes carries `"initial_prompt": ""`, so the
# real product printed that warning on the next `get` while a green test said
# stderr was clean. The bug and the test lived side by side for as long as the
# test picked its own inputs.
#
# So: seed with `save_config(Config(...))` or by running `config set`, never by
# naming values that happen to be safe. Where the choice could matter the test
# is parametrised over both (`_SEED_KINDS`), because "the file the product
# writes" is the only file any user will ever have.
# --------------------------------------------------------------------------- #


def test_get_with_no_config_file_says_nothing_on_stderr(home, capsys):
    """The fresh-install case: nothing to read, so nothing to complain about.

    Worth keeping and worth not trusting too far. No file means no field can be
    misread, so this test cannot fail for any of the reasons the ones below
    exist to catch -- it defends the state a user is in before their first
    `config set`, and only that one.
    """
    assert main(["config", "get", "hotkey"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "right_option\n"
    assert captured.err == ""


def test_get_with_a_config_file_present_says_nothing_on_stderr(home, capsys):
    """A file on disk must not turn a scriptable command into a chatty one.

    Seeded with a bare ``Config`` carrying the one setting under test, so every
    other field lands on disk at the value a real user's file holds. Naming more
    fields here would only narrow what the test can catch: each hand-picked
    value is a field this test stops asking about.
    """
    save_config(Config(history_enabled=True))
    assert main(["config", "get", "history_enabled"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "true\n"
    assert captured.err == ""


def test_get_is_silent_on_stderr_after_an_ordinary_config_set(home, capsys):
    """The whole loop, on the file the product itself wrote.

    Recorded as a strict=False xfail for a while, because the fault was
    config.py's rather than this command's: ``_pick_str`` warned about an empty
    value even for ``initial_prompt``, whose default is empty, so the most
    ordinary thing a user can do -- change one setting -- made every subsequent
    ``get`` print a warning about a value blurt had written itself and was not
    substituting anything for. ``set`` never noticed because it does not call
    ``load_config`` at all; ``get`` has to.

    ``_pick_str`` now warns only when there is a real default to fall back to,
    so the xfail is gone rather than merely satisfied. Kept as a plain test, and
    kept written as a round trip: the property is about the file this program
    leaves behind, and the only way to be sure of that file is to let the
    program write it.
    """
    assert main(["config", "set", "history_enabled", "true"]) == 0
    capsys.readouterr()
    assert main(["config", "get", "history_enabled"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "true\n"
    assert captured.err == ""


def test_an_ordinary_config_set_is_itself_silent_on_stderr(home, capsys):
    """The other half of the same root cause, on the writing side.

    ``set`` validates by serialising the value and reading it back through
    ``load_config`` (``_loader_would_reject``), and that probe's warnings go
    straight to the user's terminal on purpose -- an out-of-range value should
    be explained by the loader that knows the range. The cost of that honesty is
    that any warning the loader emits for a *healthy* defaults file is printed
    directly underneath "Wrote /path/to/config.json", which reads as a failure
    that did not happen. ``_cmd_config_set`` used to filter the probe's stderr
    by substring to hide exactly that; the filter is gone now that the warning
    is, and this is the test that notices if it comes back.
    """
    assert main(["config", "set", "history_enabled", "true"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Wrote" in captured.out


def test_the_file_config_set_writes_really_does_hold_an_empty_initial_prompt(home):
    """The premise of every stderr test above, asserted rather than assumed.

    Those tests only mean anything while the file underneath them can actually
    trigger the warning. ``set`` writes a COMPLETE document -- see
    ``test_a_first_set_on_a_fresh_install_writes_a_complete_file`` -- so
    ``"initial_prompt": ""`` is in there, which is the state every user who has
    ever run ``config set`` and never touched the prompt is in. If ``set`` ever
    started writing only the key it was given, or started omitting empty
    strings, the silence assertions would keep passing and would have stopped
    defending anything, which is precisely how this went wrong the first time.
    """
    assert main(["config", "set", "history_enabled", "true"]) == 0
    assert config_on_disk()["initial_prompt"] == ""


#: A config with nothing left at its default, so "unrelated settings survived"
#: cannot pass by accident.
#:
#: NOT a stand-in for a real user's config file, and nothing about stderr may
#: rest on it alone. It used to carry the note "every value is one load_config
#: accepts, so reading it back is silent" -- true, and beside the point: the
#: reason it read back silently was ``initial_prompt="Priya, Grafana,
#: Kubernetes"``, and the value in every real file is ``""``. A constant whose
#: fields were all chosen by hand can only ever be silent about the fields
#: somebody thought to choose. Assertions on stderr are parametrised over
#: ``_SEED_KINDS`` so they meet the product's own output too.
_SEEDED = Config(
    engine="faster-whisper",
    model="small.en",
    hotkey="right_ctrl",
    cleanup_level="standard",
    sample_rate=24000,
    preroll_ms=250,
    min_hold_ms=150,
    paste_delay_ms=90,
    clipboard_restore_ms=300,
    cpu_threads=4,
    keep_raw_history=False,
    dictionary={"github": "GitHub", "kubernetes": "Kubernetes"},
    history_enabled=False,
    history_limit=99,
    initial_prompt="Priya, Grafana, Kubernetes",
    assistant_enabled=False,
    assistant_hotkey="left_cmd",
)


@pytest.mark.parametrize("seed", _SEED_KINDS)
@pytest.mark.parametrize("key", sorted(list(_ROUND_TRIP) + ["dictionary"]))
def test_get_prints_exactly_one_line_for_every_setting(home, capsys, key, seed):
    """Including ``dictionary``, which is the one value with room to sprawl.

    It cannot be fed back to ``set``, but it is still printed as a single line
    of JSON rather than as an indented block: a reader that consumes one line
    per setting should not have to special-case one of them.

    Run against both kinds of file because the two answer different questions.
    ``_SEEDED`` has every field at a non-default value, which is what makes the
    one-line assertion mean something -- an empty dictionary would fit on one
    line whatever the formatting did. The file ``config set`` writes has the
    fields a user has not touched sitting at their defaults, empty strings
    included, which is what makes the stderr assertion mean something. Either
    seed alone leaves half of this test decorative.
    """
    _plant_config(seed, capsys)
    assert main(["config", "get", key]) == 0
    captured = capsys.readouterr()
    assert captured.out.endswith("\n")
    assert captured.out.count("\n") == 1
    assert captured.err == ""


# --------------------------------------------------------------------------- #
# config set persists
# --------------------------------------------------------------------------- #


def test_set_history_enabled_actually_reaches_disk(home):
    assert main(["config", "set", "history_enabled", "true"]) == 0
    assert load_config().history_enabled is True


def test_set_writes_valid_json(home):
    main(["config", "set", "history_enabled", "true"])
    # Parses, is an object, and holds a real JSON true rather than the string
    # "true" -- load_config refuses a quoted boolean and falls back to the
    # default, which would make this command look like it worked and do nothing.
    on_disk = config_on_disk()
    assert isinstance(on_disk, dict)
    assert on_disk["history_enabled"] is True


def test_set_says_what_changed(home, capsys):
    assert main(["config", "set", "cleanup_level", "standard"]) == 0
    out = capsys.readouterr().out
    assert "cleanup_level: light -> standard" in out
    assert str(default_config_path()) in out


def test_turning_the_journal_on_restates_what_it_costs(home, capsys):
    """The one setting that turns speech into a file says so as it is turned on."""
    main(["config", "set", "history_enabled", "true"])
    out = capsys.readouterr().out
    assert "disk" in out
    assert "learn --forget" in out


def test_set_is_idempotent(home):
    assert main(["config", "set", "history_limit", "500"]) == 0
    first = _snapshot()
    assert main(["config", "set", "history_limit", "500"]) == 0
    assert _snapshot() == first


def test_a_first_set_on_a_fresh_install_writes_a_complete_file(home):
    """No file yet is the normal first-run state, and the file it leaves is whole.

    Writing only the one key the user named would load perfectly well -- every
    absent key falls back to its default -- but it would leave them with a file
    whose shape depends on which setting they happened to touch first, and
    nothing to read when they open it wondering what else is in there.
    """
    assert not default_config_path().exists()
    assert main(["config", "set", "history_enabled", "true"]) == 0
    assert set(config_on_disk()) == set(f.name for f in dataclasses.fields(Config))


# --------------------------------------------------------------------------- #
# set MERGES into the file on disk; it does not rebuild it
#
# This is the section with real data loss behind it. `set` used to load a
# Config, change one attribute and write the whole dataclass back, and a Config
# is a lossy view of the file in two directions:
#
#   * it drops keys it does not recognise -- deliberate forward compatibility
#     when READING, and silent deletion when the result is written back out;
#   * it substitutes the default for any value it rejects -- so an unrelated
#     `set` would overwrite the user's typo'd sample_rate with 16000 and erase
#     the evidence they need in order to find it.
#
# Neither leaves a trace, and neither is recoverable. Every test here would have
# passed against the old round trip only by accident.
# --------------------------------------------------------------------------- #


def test_the_loader_really_does_ignore_a_key_it_has_never_heard_of(home):
    """The premise of the test below, asserted rather than assumed.

    If ``load_config`` ever started *keeping* unknown keys, the survival test
    would keep passing for a reason that has nothing to do with the merge, and
    would stop defending anything. Pin the premise so it cannot rot quietly.
    """
    _write_raw({"future_setting": 42})
    assert not hasattr(load_config(), "future_setting")


def test_a_setting_this_version_has_never_heard_of_survives_a_set(home):
    """The headline data-loss guarantee of the whole command.

    ``config.py`` ignores unknown keys on purpose, so that an older blurt can
    read a newer blurt's config file without complaining. Ignoring is not
    deleting. Round-tripping the file through ``Config`` turned the one into the
    other: run the older blurt once, change any single setting, and every
    setting the newer one wrote is gone -- from a command that printed success,
    with nothing on screen to say what it took away and no copy anywhere to
    restore from.
    """
    _write_raw({"history_enabled": False, "future_setting": 42})
    assert main(["config", "set", "history_enabled", "true"]) == 0
    on_disk = config_on_disk()
    assert on_disk["future_setting"] == 42
    assert on_disk["history_enabled"] is True


def test_a_whole_unknown_subtree_survives_a_set_unchanged(home):
    """Tomorrow's setting need not be a scalar, so nothing here may assume it is."""
    future = {"voice": "en_GB", "aliases": ["priya", "raj"], "confirm": {"timers": False}}
    _write_raw({"history_enabled": False, "assistant_profile": future})
    assert main(["config", "set", "history_enabled", "true"]) == 0
    assert config_on_disk()["assistant_profile"] == future


def test_set_adds_no_key_the_file_did_not_already_have(home):
    """A merge changes one key. It does not take the opportunity to normalise.

    The sharpest available proof that the write does not go through the
    dataclass: a rebuild would hand back all seventeen fields whatever the file
    said, so a sparse file staying sparse can only happen if the object on disk
    was edited in place.
    """
    _write_raw({"hotkey": "right_ctrl", "future_setting": 42})
    assert main(["config", "set", "history_enabled", "true"]) == 0
    assert set(config_on_disk()) == {"hotkey", "future_setting", "history_enabled"}


#: On-disk values ``load_config`` refuses, one per shape of refusal: out of
#: range, not in the allowed set, below the floor, and right type in JSON but
#: wrong type for the field. Each is what a plausible typo actually looks like.
_LOADER_REJECTS: Dict[str, Any] = {
    "sample_rate": 999999,          # outside 8000..48000
    "cleanup_level": "aggressive",  # not one of none/light/standard
    "engine": "vosk",               # not an engine blurt has
    "history_limit": -5,            # below the floor of 1
    "keep_raw_history": "true",     # a quoted bool is not a bool
}


@pytest.mark.parametrize("key", sorted(_LOADER_REJECTS))
def test_a_value_the_loader_rejects_is_not_rewritten_by_an_unrelated_set(home, key):
    """Changing setting A must not quietly "fix" the user's typo in setting B.

    The typo is already costing them: the setting looks set and does nothing,
    behind a stderr warning at every launch that nobody reads. That is bad and
    it is at least discoverable -- the wrong value is still sitting in the file
    where they typed it. Overwriting it with the default during an unrelated
    ``set`` removes the last piece of evidence that anything was ever wrong,
    and the user is left with a file that looks correct and a memory of having
    configured something that is no longer there.
    """
    bad = _LOADER_REJECTS[key]
    _write_raw({key: bad, "history_enabled": False})

    # Premise: the loader really does refuse this value and substitute the
    # default. Without this the survival assertion below could pass vacuously.
    loaded = getattr(load_config(), key)
    assert loaded != bad
    assert loaded == getattr(Config(), key)

    assert main(["config", "set", "history_enabled", "true"]) == 0
    on_disk = config_on_disk()
    assert on_disk[key] == bad, "an unrelated set overwrote a value the user typed"
    assert on_disk["history_enabled"] is True


def test_the_key_being_set_is_still_validated_even_though_the_rest_is_not(home):
    """Tolerating what is already in the file is not the same as accepting more.

    The merge leaves a bad ``sample_rate`` alone; it must still refuse to WRITE
    one. Otherwise the fix for the round trip would have quietly removed the
    range checking that made ``set`` worth having.
    """
    _write_raw({"sample_rate": 999999})
    assert _refused(["config", "set", "sample_rate", "999998"]) == 2


# --------------------------------------------------------------------------- #
# A config file `set` cannot understand is never written over
#
# An unparseable config is still the only copy of whatever the user put in it,
# and a config file can hold a replacement dictionary built from months of their
# own speech that exists nowhere else. Replacing it with defaults would be the
# single most destructive thing this command could do, and it would do it while
# reporting success.
# --------------------------------------------------------------------------- #


#: Shapes of "unusable", each one something a real editing accident produces.
_CORRUPT_FILES: Dict[str, bytes] = {
    "a truncated object": b'{"history_enabled": true,\n',
    "python-style quotes": b"{'history_enabled': True}\n",
    "a trailing comma": b'{"history_enabled": true,}\n',
    "a JSON array": b"[1, 2, 3]\n",
    "a bare JSON string": b'"history_enabled"\n',
    "an empty file": b"",
    "shell syntax": b"history_enabled=true\n",
}


@pytest.mark.parametrize("shape", sorted(_CORRUPT_FILES))
def test_a_config_file_set_cannot_parse_is_left_byte_identical(home, shape):
    before = _write_bytes(_CORRUPT_FILES[shape])
    assert main(["config", "set", "history_enabled", "true"]) != 0
    assert default_config_path().read_bytes() == before


def test_a_config_file_set_cannot_parse_is_not_even_moved_aside(home):
    """Refusing means refusing, including refusing to rename their file.

    ``load_config`` renames an unparseable config to ``config.json.bak`` as a
    side effect of being asked to read it -- reasonable there, since it warns
    once instead of at every launch, and it is about to run on defaults anyway.
    ``set`` deliberately reads the file itself rather than going near
    ``load_config``, because a user who runs ``config set`` and is told "nothing
    was written" should find their file where they left it, under the name they
    know, ready for the text editor they are about to open it in.
    """
    _write_bytes(b'{"history_enabled": true,\n')
    assert main(["config", "set", "history_enabled", "true"]) != 0
    assert _config_dir_entries() == ["config.json"]


def test_a_config_file_set_cannot_parse_explains_the_way_out(home, capsys):
    _write_bytes(b'{"history_enabled": true,\n')
    assert main(["config", "set", "history_enabled", "true"]) != 0
    err = capsys.readouterr().err
    assert "not valid JSON" in err
    assert "Nothing was written" in err
    # The escape hatch is spelled out as a command, because "move the file out
    # of the way" is exactly the instruction a stuck user cannot act on.
    assert "mv " in err
    assert str(default_config_path()) in err


def test_a_json_document_that_is_not_an_object_is_refused_the_same_way(home, capsys):
    _write_bytes(b"[1, 2, 3]\n")
    assert main(["config", "set", "history_enabled", "true"]) != 0
    err = capsys.readouterr().err
    assert "must contain a JSON object" in err
    assert "Nothing was written" in err


def test_a_corrupt_config_does_not_stop_get_from_answering(home, capsys):
    """The read path may fall back to defaults; only the write path must refuse.

    Different commands, different stakes. ``get`` cannot destroy anything, so
    answering from defaults is the useful behaviour; ``set`` can, so it stops.
    """
    _write_bytes(b'{"history_enabled": true,\n')
    assert main(["config", "get", "hotkey"]) == 0
    assert capsys.readouterr().out == "right_option\n"


# --------------------------------------------------------------------------- #
# The write itself: atomic, and readable only by its owner
# --------------------------------------------------------------------------- #


def test_the_file_set_writes_is_readable_only_by_its_owner(home):
    """0600, because a config file can hold a personal replacement dictionary.

    That dictionary is a list of the names, jargon and acronyms a person says
    out loud all day. It is not a secret in the password sense and it is nobody
    else's business either, and on a shared or managed Mac the default umask is
    not a thing to trust with it.
    """
    assert main(["config", "set", "history_enabled", "true"]) == 0
    assert stat.S_IMODE(default_config_path().stat().st_mode) == 0o600


def test_set_tightens_a_config_file_that_was_left_wide_open(home):
    """The mode comes from the write, not from whatever the old file had.

    A config file hand-created with a permissive umask, or restored out of a
    backup or a dotfiles repo, arrives 0644. Because the new file is created
    fresh and renamed over the old one, the mode is set on every write rather
    than inherited -- so the first ``config set`` quietly repairs it.
    """
    _write_raw({"history_enabled": False})
    os.chmod(str(default_config_path()), 0o644)
    assert main(["config", "set", "history_enabled", "true"]) == 0
    assert stat.S_IMODE(default_config_path().stat().st_mode) == 0o600


def test_the_write_is_a_rename_from_a_temp_file_in_the_same_directory(home, monkeypatch):
    """Atomicity is ``os.replace``, and ``os.replace`` is only atomic within one
    filesystem.

    Writing in place with ``open(path, "w")`` truncates the user's real config
    first and writes second; a crash, a full disk or a SIGKILL in between leaves
    an empty or half-written config where a good one used to be. The temp file
    also has to be created in the destination directory rather than in ``/tmp``,
    because on modern macOS ``/tmp`` is a different volume from the data volume
    and a cross-device ``os.replace`` is not atomic (it is not even permitted).

    Spying on ``os.replace`` rather than trying to interrupt a real write: the
    property is which two paths get handed to it, and that is exactly what a
    spy can state without any timing assumption at all.
    """
    real_replace = os.replace
    calls: List[Tuple[str, str]] = []

    def spy(src, dst):
        calls.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    assert main(["config", "set", "history_enabled", "true"]) == 0

    path = default_config_path()
    landed = [call for call in calls if call[1] == str(path)]
    assert len(landed) == 1, "the config file arrived by some route other than a rename"
    source = pathlib.Path(landed[0][0])
    assert source.parent == path.parent


def test_a_write_that_fails_at_the_rename_leaves_the_old_config_and_no_debris(
    home, monkeypatch, capsys
):
    """The half-second where a truncating write would have destroyed the file.

    Failure is injected at ``os.replace`` because that is the last possible
    moment: everything before it has already been written and fsynced to the
    temp file. An in-place write reaching the same point would have left the
    user's config truncated to nothing. Here the old bytes are still the bytes,
    the temp file is cleaned up rather than left lying next to the real config
    with a name nobody recognises, and the command says so and exits non-zero.
    """
    before = _write_raw({"history_enabled": False, "future_setting": 42})
    real_replace = os.replace
    target = str(default_config_path())

    def failing_replace(src, dst):
        if str(dst) == target:
            raise OSError(28, "No space left on device")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing_replace)
    assert main(["config", "set", "history_enabled", "true"]) != 0

    assert default_config_path().read_bytes() == before
    assert _config_dir_entries() == ["config.json"], "a temp file was left behind"
    err = capsys.readouterr().err
    assert "could not write" in err
    assert "Nothing was changed" in err


def test_a_successful_set_leaves_no_temp_file_behind(home):
    _write_raw({"history_enabled": False})
    assert main(["config", "set", "history_enabled", "true"]) == 0
    assert _config_dir_entries() == ["config.json"]


# --------------------------------------------------------------------------- #
# Two hotkeys, one physical key
#
# Validated on its own, `assistant_hotkey right_option` is a perfectly good
# value, and so is `hotkey right_option`. Together they are not.
# BlurtApp._build_assistant resolves the tie by switching assistant mode off
# entirely -- one key cannot mean two things -- so a single valid-looking
# command turns off command mode without ever saying it did. The user is then
# left with no hotkey for the assistant AND, because `revert_last` lives in
# command mode, no way to speak an undo either: two features gone, no error, and
# a config file that reads as entirely reasonable.
# --------------------------------------------------------------------------- #


def test_setting_the_assistant_key_to_the_dictation_key_is_refused(home):
    # Fresh install: hotkey is right_option, assistant_hotkey is right_cmd.
    assert _refused(["config", "set", "assistant_hotkey", "right_option"]) == 2


def test_setting_the_dictation_key_to_the_assistant_key_is_refused(home):
    """The same collision arrived at from the other side. Both directions, because
    a check written on one field is a check that half works."""
    save_config(Config(hotkey="right_ctrl", assistant_hotkey="right_cmd"))
    assert _refused(["config", "set", "hotkey", "right_cmd"]) == 2


@pytest.mark.parametrize(
    "spelling", ["right_alt", "right_opt", "ralt", "RIGHT_ALT", " right-alt "]
)
def test_an_alias_of_the_other_key_collides_just_as_hard(home, spelling):
    """right_alt IS right_option: same physical key, two accepted spellings.

    Comparing the strings as written would let this one through, and the result
    would be worse than the collision the runtime does catch. ``_build_assistant``
    compares as loaded, so ``right_alt`` != ``right_option`` and the assistant is
    NOT disabled -- blurt binds two listeners to one physical key instead, and
    every press starts both a dictation and a command with no warning anywhere.
    """
    assert _refused(["config", "set", "assistant_hotkey", spelling]) == 2


def test_an_alias_already_in_the_file_is_normalised_before_comparison(home):
    """The other side of the comparison comes off disk, where it may be an alias.

    A config written by hand -- or by an older blurt, or by a user who prefers
    the name their keyboard has printed on it -- can say ``right_alt``. Folding
    only the incoming value would catch half the cases and look like it worked.
    """
    _write_raw({"hotkey": "right_ctrl", "assistant_hotkey": "right_alt"})
    assert _refused(["config", "set", "hotkey", "right_option"]) == 2


@pytest.mark.parametrize(
    "hotkey_entry",
    [
        {},                    # the file never mentions the dictation key
        {"hotkey": ""},        # present and empty
        {"hotkey": "   "},     # present and blank
        {"hotkey": None},      # present and null
        {"hotkey": 42},        # present and the wrong type entirely
    ],
    ids=["missing", "empty", "blank", "null", "wrong-type"],
)
def test_a_collision_with_a_defaulted_dictation_key_is_still_a_collision(
    home, hotkey_entry
):
    """What matters is the key blurt will really use, not the key the file names.

    Every one of these files loads as ``hotkey = right_option``, so setting the
    assistant key to right_option collides at runtime exactly as if the file had
    said so. Asking the raw document instead of a loaded ``Config`` is what makes
    this an easy thing to get wrong.
    """
    document: Dict[str, Any] = {"assistant_hotkey": "right_cmd"}
    document.update(hotkey_entry)
    _write_raw(document)
    assert _refused(["config", "set", "assistant_hotkey", "right_option"]) == 2


def test_a_collision_is_refused_even_while_assistant_mode_is_off(home):
    """An armed trap is still a trap.

    ``_build_assistant`` checks ``assistant_enabled`` before it compares the
    keys, so a collision written while the assistant is off does nothing today.
    It goes off later, when somebody runs ``config set assistant_enabled true``
    and command mode silently fails to appear -- and at that point the
    explanation would have to come from a command that did nothing wrong.
    """
    save_config(Config(assistant_enabled=False, hotkey="right_ctrl"))
    assert _refused(["config", "set", "assistant_hotkey", "right_ctrl"]) == 2


def test_the_collision_message_names_both_keys_and_a_way_out(home, capsys):
    assert main(["config", "set", "assistant_hotkey", "right_option"]) == 2
    err = capsys.readouterr().err
    assert "assistant_hotkey" in err
    assert "hotkey" in err
    assert "nothing was written" in err
    # The way out is the other half of the pair, spelled as a command.
    assert "blurt config set hotkey" in err
    assert SUPPORTED_HOTKEYS[0] in err


def test_a_hotkey_that_does_not_collide_still_goes_straight_through(home):
    """The check must refuse collisions, not hotkeys."""
    save_config(Config(hotkey="right_ctrl", assistant_hotkey="right_cmd"))
    assert main(["config", "set", "assistant_hotkey", "left_cmd"]) == 0
    assert config_on_disk()["assistant_hotkey"] == "left_cmd"


def test_setting_a_hotkey_to_the_value_it_already_has_is_not_a_collision(home):
    """A key only collides with the OTHER field. Comparing it against itself would
    make every hotkey unsettable a second time."""
    save_config(Config(hotkey="right_ctrl", assistant_hotkey="right_cmd"))
    assert main(["config", "set", "hotkey", "right_ctrl"]) == 0
    assert config_on_disk()["hotkey"] == "right_ctrl"


def test_moving_a_key_off_an_existing_collision_is_allowed(home):
    """The escape route, and the reason the check is on the value rather than on
    the file.

    A config that already has both keys the same is exactly the config a user
    needs to fix, and ``config set hotkey`` is how they fix it. Refusing to touch
    a colliding file would leave them hand-editing JSON -- the situation this
    command exists to spare them.
    """
    save_config(Config(hotkey="right_ctrl", assistant_hotkey="right_ctrl"))
    assert main(["config", "set", "hotkey", "right_shift"]) == 0
    on_disk = config_on_disk()
    assert on_disk["hotkey"] == "right_shift"
    assert on_disk["assistant_hotkey"] == "right_ctrl"


def test_an_unrelated_set_is_not_blocked_by_a_collision_already_on_disk(home):
    """Turning the journal on is not the moment to litigate the hotkeys.

    The collision check guards the value being written. A file that already
    collides is a real problem, but refusing every other setting until it is
    fixed would punish the user for a state they may not have created and
    certainly cannot see from here.
    """
    save_config(Config(hotkey="right_ctrl", assistant_hotkey="right_ctrl"))
    assert main(["config", "set", "history_enabled", "true"]) == 0
    on_disk = config_on_disk()
    assert on_disk["history_enabled"] is True
    assert on_disk["hotkey"] == on_disk["assistant_hotkey"] == "right_ctrl"


def test_a_colliding_hotkey_is_refused_before_the_file_is_touched(home):
    """Belt and braces on the ordering: nothing is written, not even a temp file."""
    _write_raw({"hotkey": "right_ctrl", "assistant_hotkey": "right_cmd"})
    assert _refused(["config", "set", "assistant_hotkey", "right_ctrl"]) == 2
    assert _config_dir_entries() == ["config.json"]


# --------------------------------------------------------------------------- #
# Boolean parsing: generous on input, strict about refusing the rest
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "spelling", ["true", "TRUE", "True", "yes", "YES", "Yes", "on", "ON", "1"]
)
def test_every_spelling_of_yes_turns_a_switch_on(home, spelling):
    assert main(["config", "set", "history_enabled", spelling]) == 0
    assert load_config().history_enabled is True


@pytest.mark.parametrize(
    "spelling", ["false", "FALSE", "False", "no", "NO", "No", "off", "OfF", "0"]
)
def test_every_spelling_of_no_turns_a_switch_off(home, spelling):
    # keep_raw_history rather than history_enabled: it defaults to True, so
    # "off" landing on disk is a change and not a coincidence.
    assert main(["config", "set", "keep_raw_history", spelling]) == 0
    assert load_config().keep_raw_history is False


@pytest.mark.parametrize("rubbish", ["banana", "2", "", "   ", "-1", "y", "n", "truthy"])
def test_a_value_that_is_neither_true_nor_false_is_refused(home, rubbish):
    """Unparseable must not degrade to "off".

    Writing ``false`` for ``history_enabled banana`` would leave the user with a
    belief about what just happened that is both intact and wrong, and the only
    evidence against it is a setting they cannot see. ``y`` and ``n`` are in the
    list on purpose: they are what ``learn --apply`` asks for, so someone in the
    habit of typing them deserves a refusal rather than a guess.
    """
    assert _refused(["config", "set", "history_enabled", rubbish]) == 2


def test_a_refused_bool_does_not_bring_the_config_file_into_existence(home):
    assert not default_config_path().exists()
    assert main(["config", "set", "history_enabled", "banana"]) == 2
    assert not default_config_path().exists()


def test_a_refused_bool_leaves_an_existing_file_byte_identical(home):
    save_config(Config(history_enabled=True, hotkey="right_ctrl", preroll_ms=250))
    before = _snapshot()
    assert main(["config", "set", "history_enabled", "banana"]) == 2
    assert _snapshot() == before


def test_a_refused_bool_names_the_spellings_that_would_have_worked(home, capsys):
    main(["config", "set", "history_enabled", "banana"])
    err = capsys.readouterr().err
    assert "true" in err
    assert "false" in err


# --------------------------------------------------------------------------- #
# Int parsing
# --------------------------------------------------------------------------- #


def test_a_valid_int_persists(home):
    assert main(["config", "set", "preroll_ms", "250"]) == 0
    assert load_config().preroll_ms == 250


@pytest.mark.parametrize("rubbish", ["many", "2.5", "", "1e3", "0x10", "twelve", "12ms"])
def test_a_non_integer_is_refused_and_writes_nothing(home, rubbish):
    assert _refused(["config", "set", "history_limit", rubbish]) == 2


def test_a_refused_int_leaves_an_existing_file_byte_identical(home):
    save_config(Config(history_limit=99))
    before = _snapshot()
    assert main(["config", "set", "history_limit", "many"]) == 2
    assert _snapshot() == before
    assert load_config().history_limit == 99


def test_an_int_the_loader_would_reject_is_refused_before_it_is_written(home, capsys):
    """Out-of-range is refused now rather than silently ignored forever.

    ``sample_rate`` accepts 8000..48000. A saved 999999 would load as 16000 on
    every launch behind a warning nobody reads, so the setting would be there,
    look set, and do nothing -- which is exactly the shape of failure that is
    never noticed. The range itself lives in ``config.py``; this only asserts
    that the command asks it.
    """
    assert _refused(["config", "set", "sample_rate", "999999"]) == 2
    assert "sample_rate" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Validated string settings
# --------------------------------------------------------------------------- #


def test_an_engine_outside_the_valid_set_is_refused(home, capsys):
    assert "vosk" not in VALID_ENGINES
    assert _refused(["config", "set", "engine", "vosk"]) == 2
    err = capsys.readouterr().err
    assert "faster-whisper" in err


def test_a_valid_engine_persists(home):
    assert main(["config", "set", "engine", "apple-speech"]) == 0
    assert load_config().engine == "apple-speech"


def test_a_cleanup_level_outside_the_valid_set_is_refused(home, capsys):
    assert "aggressive" not in VALID_CLEANUP_LEVELS
    assert _refused(["config", "set", "cleanup_level", "aggressive"]) == 2
    assert "standard" in capsys.readouterr().err


def test_a_valid_cleanup_level_persists(home):
    assert main(["config", "set", "cleanup_level", "none"]) == 0
    assert load_config().cleanup_level == "none"


def test_the_fn_key_is_refused_because_macos_never_reports_it(home, capsys):
    """A hotkey blurt cannot observe would save fine and then never fire.

    fn is a perfectly reasonable thing to want as a push-to-talk key and pynput
    cannot see it on macOS at all. Accepting it here would produce a daemon that
    starts cleanly, prints nothing wrong, and ignores the key forever -- the
    failure mode ``doctor`` exists to fight, arriving via the settings command.
    """
    assert _refused(["config", "set", "hotkey", "fn"]) == 2
    err = capsys.readouterr().err
    assert "right_option" in err


def test_the_assistant_hotkey_is_validated_the_same_way(home):
    assert _refused(["config", "set", "assistant_hotkey", "fn"]) == 2


def test_an_unknown_hotkey_name_is_refused(home):
    assert _refused(["config", "set", "hotkey", "right_windows"]) == 2


def test_a_hotkey_alias_is_normalised_before_it_is_written(home):
    """right_alt is a legitimate spelling; the file should still say right_option.

    Stored unnormalised, the config would disagree with the key blurt actually
    listens on, and every later reader -- doctor, `config get`, the user -- would
    have to re-derive the mapping to find that out.
    """
    assert main(["config", "set", "hotkey", "right_ctrl"]) == 0
    assert main(["config", "set", "hotkey", "right_alt"]) == 0
    assert config_on_disk()["hotkey"] == "right_option"


def test_an_empty_model_is_refused(home):
    assert _refused(["config", "set", "model", ""]) == 2


def test_an_unfamiliar_model_name_is_still_accepted(home):
    """faster-whisper takes local paths and Hugging Face ids, not just the sizes."""
    assert main(["config", "set", "model", "distil-whisper/distil-small.en"]) == 0
    assert load_config().model == "distil-whisper/distil-small.en"


# --------------------------------------------------------------------------- #
# Keys that cannot be set this way
# --------------------------------------------------------------------------- #


def test_setting_the_dictionary_is_refused_and_points_at_learn(home, capsys):
    """The dictionary has a command that knows more than the user does.

    Inventing a `set dictionary github=GitHub` syntax would be a worse version of
    `blurt learn --apply`, which builds it from the user's own transcripts.
    """
    assert _refused(["config", "set", "dictionary", "github=GitHub"]) == 2
    err = capsys.readouterr().err
    assert "blurt learn --apply" in err


def test_setting_the_dictionary_does_not_disturb_an_existing_one(home):
    save_config(Config(dictionary={"github": "GitHub"}))
    before = _snapshot()
    assert main(["config", "set", "dictionary", "kubernetes=Kubernetes"]) == 2
    assert _snapshot() == before
    assert load_config().dictionary == {"github": "GitHub"}


def test_setting_an_unknown_key_exits_two_and_writes_nothing(home, capsys):
    assert _refused(["config", "set", "colour_scheme", "dark"]) == 2
    err = capsys.readouterr().err
    assert "unknown setting" in err
    assert "history_enabled" in err


def test_setting_an_unknown_key_leaves_an_existing_file_byte_identical(home):
    save_config(Config(history_enabled=True))
    before = _snapshot()
    assert main(["config", "set", "colour_scheme", "dark"]) == 2
    assert _snapshot() == before


def test_a_key_the_file_already_holds_is_still_not_settable(home):
    """Tolerating a stranger's key on disk does not make it a setting.

    The merge keeps ``future_setting`` alive across writes, which could be read
    as blurt half-supporting it. It does not: the command refuses to set a key
    this version has no meaning for, because "accepted" and "does something" are
    the same thing from the user's side of the terminal.
    """
    _write_raw({"future_setting": 42})
    assert _refused(["config", "set", "future_setting", "43"]) == 2


# --------------------------------------------------------------------------- #
# One-run overrides are never persisted
# --------------------------------------------------------------------------- #


def test_a_one_run_cleanup_override_is_never_persisted_by_a_set(home):
    """--cleanup is explicitly for one run; `config set` must not bank it.

    The config ``main`` resolved has the override folded into it. ``_cmd_config_set``
    deliberately throws that away and merges into the file on disk, and this is
    the test that fails if anyone ever "simplifies" it into reusing the resolved
    one. Persisting it would turn a flag someone passed to try something into a
    permanent setting they never chose and would have no reason to go looking
    for. The same guarantee is defended for ``learn --apply`` in
    test_learn_cli.py.
    """
    assert main(["--cleanup", "standard", "config", "set", "history_enabled", "true"]) == 0
    on_disk = config_on_disk()
    assert on_disk["history_enabled"] is True
    assert on_disk["cleanup_level"] == "light"


def test_a_one_run_model_override_is_never_persisted_by_a_set(home):
    """--model is for one run too, and lands on the subcommand as readily."""
    assert main(["config", "set", "history_enabled", "true", "--model", "tiny.en"]) == 0
    on_disk = config_on_disk()
    assert on_disk["history_enabled"] is True
    assert on_disk["model"] == "auto"


def test_a_one_run_engine_override_is_never_persisted_by_a_set(home):
    assert main(["--engine", "apple-speech", "config", "set", "history_enabled", "true"]) == 0
    assert config_on_disk()["engine"] == "auto"


def test_a_one_run_hotkey_override_is_never_persisted_by_a_set(home):
    assert main(["--hotkey", "right_shift", "config", "set", "history_enabled", "true"]) == 0
    assert config_on_disk()["hotkey"] == "right_option"


def test_a_one_run_hotkey_override_does_not_invent_a_collision(home):
    """The override is not the file, and the collision check reads the file.

    ``blurt --hotkey right_cmd config set assistant_hotkey right_cmd`` is two
    different keys' worth of confusion, but the one that counts is what ends up
    on disk: the override evaporates when the process exits, so refusing here
    would refuse a setting that is going to be perfectly valid. (The reverse --
    an override that HIDES a collision -- cannot happen either, for the same
    reason: the check never looks at the resolved config.)
    """
    save_config(Config(hotkey="right_ctrl", assistant_hotkey="right_option"))
    assert main(["--hotkey", "right_cmd", "config", "set", "assistant_hotkey", "right_cmd"]) == 0
    on_disk = config_on_disk()
    assert on_disk["hotkey"] == "right_ctrl"
    assert on_disk["assistant_hotkey"] == "right_cmd"


def test_an_override_does_not_survive_into_a_config_written_over_a_seeded_one(home):
    save_config(Config(cleanup_level="none", model="small.en"))
    assert main(["--cleanup", "standard", "--model", "tiny.en",
                 "config", "set", "history_enabled", "true"]) == 0
    on_disk = config_on_disk()
    assert on_disk["cleanup_level"] == "none"
    assert on_disk["model"] == "small.en"


def test_get_reports_the_override_without_writing_it(home, capsys):
    """`get` answers "what would this run use"; only `set` touches the file."""
    assert main(["--cleanup", "standard", "config", "get", "cleanup_level"]) == 0
    assert capsys.readouterr().out == "standard\n"
    assert not default_config_path().exists()


# --------------------------------------------------------------------------- #
# Everything the command was not asked to change
# --------------------------------------------------------------------------- #


def test_set_preserves_every_unrelated_setting(home):
    save_config(_SEEDED)
    assert main(["config", "set", "history_enabled", "true"]) == 0

    expected = dataclasses.asdict(_SEEDED)
    expected["history_enabled"] = True
    assert config_on_disk() == expected


def test_set_preserves_the_dictionary_it_did_not_touch(home):
    """The dictionary is the irreplaceable part: it is months of the user's own
    speech, and nothing can reconstruct it if a write drops it."""
    save_config(_SEEDED)
    main(["config", "set", "cleanup_level", "none"])
    assert load_config().dictionary == {
        "github": "GitHub",
        "kubernetes": "Kubernetes",
    }


def test_a_refused_set_preserves_the_dictionary_too(home):
    save_config(_SEEDED)
    before = _snapshot()
    assert main(["config", "set", "cleanup_level", "aggressive"]) == 2
    assert _snapshot() == before


def test_a_dictionary_entry_no_loader_would_keep_still_survives_a_set(home):
    """Even the parts of the dictionary blurt itself would discard.

    ``_pick_dictionary`` drops non-string entries one by one, so a round trip
    would quietly delete this line. It is malformed, it does nothing, and it is
    still the user's -- probably the wreckage of a hand-edit they are halfway
    through, and certainly not something an unrelated ``config set`` should tidy
    away on their behalf.
    """
    _write_raw({"dictionary": {"github": "GitHub", "kubernetes": 42}})
    assert main(["config", "set", "history_enabled", "true"]) == 0
    assert config_on_disk()["dictionary"] == {"github": "GitHub", "kubernetes": 42}


# --------------------------------------------------------------------------- #
# Every scalar setting round-trips
# --------------------------------------------------------------------------- #


def test_the_round_trip_table_covers_every_field_of_config():
    """Fails the day someone adds a field to Config without noticing this file.

    ``_config_fields()`` in the CLI reads the dataclass, so a new field becomes
    settable the moment it lands -- settable and, without this assertion,
    completely untested.
    """
    declared = set(f.name for f in dataclasses.fields(Config))
    assert declared == set(_ROUND_TRIP) | _NOT_SETTABLE, (
        "Config gained or lost a field; add it to _ROUND_TRIP (or to "
        "_NOT_SETTABLE if it cannot come from a single command-line word)"
    )


def test_every_round_trip_value_differs_from_its_default():
    """Otherwise the round-trip test would pass against a `set` that did nothing."""
    defaults = Config()
    for key, (word, expected) in _ROUND_TRIP.items():
        assert getattr(defaults, key) != expected, (
            "%s is being 'changed' to its own default (%r)" % (key, word)
        )


@pytest.mark.parametrize("key", sorted(_ROUND_TRIP))
def test_every_scalar_setting_round_trips_through_the_config_file(home, key):
    word, expected = _ROUND_TRIP[key]
    assert main(["config", "set", key, word]) == 0
    assert getattr(load_config(), key) == expected
    assert config_on_disk()[key] == expected


@pytest.mark.parametrize("key", sorted(_ROUND_TRIP))
def test_set_then_get_agree_for_every_scalar_setting(home, capsys, key):
    """`get` prints what `set` accepts, so the two can be piped together."""
    word, _expected = _ROUND_TRIP[key]
    assert main(["config", "set", key, word]) == 0
    capsys.readouterr()
    assert main(["config", "get", key]) == 0
    printed = capsys.readouterr().out.rstrip("\n")
    assert main(["config", "set", key, printed]) == 0
    assert getattr(load_config(), key) == _ROUND_TRIP[key][1]
