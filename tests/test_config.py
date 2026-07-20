"""Tests for blurt.config -- JSON settings that must never block dictation.

The contract under test is that config loading is TOTAL: every failure mode
degrades to a default plus a warning on stderr, and none of them raise. A user
who hand-edits config.json and makes a typo must still be able to dictate.

Two properties get special attention:

  * load_config() must NOT create the file. First run is the common case, and a
    config file that appears by itself is a file the user never chose to have
    and will not think to look at when something goes wrong.
  * A corrupt file is moved aside rather than deleted or overwritten in place,
    so the user's dictionary is recoverable and we warn once instead of on
    every launch.

Every test writes into pytest's tmp_path. Nothing touches the real
~/.config/blurt, and nothing needs the network or any macOS permission.

Python 3.9 floor: lazy annotations, typing.Dict / typing.Any.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
from typing import Any, Dict

from blurt import config
from blurt.config import Config, load_config, save_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: pathlib.Path, payload: Any) -> None:
    """Write JSON (or raw text, if a str is given) to path."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_config_defaults_match_the_agreed_contract():
    cfg = Config()
    assert cfg.engine == "auto"
    assert cfg.model == "auto"
    assert cfg.hotkey == "right_option"
    assert cfg.cleanup_level == "light"
    assert cfg.sample_rate == 16000
    assert cfg.preroll_ms == 500
    assert cfg.min_hold_ms == 200
    assert cfg.paste_delay_ms == 120
    assert cfg.clipboard_restore_ms == 400
    assert cfg.cpu_threads == 0
    assert cfg.keep_raw_history is True
    assert cfg.dictionary == {}


def test_each_config_instance_gets_its_own_dictionary():
    # A shared mutable default would leak one user's replacements into every
    # other Config in the process.
    first = Config()
    second = Config()
    first.dictionary["github"] = "GitHub"
    assert second.dictionary == {}


def test_defaults_load_when_no_file_exists(tmp_path):
    missing = tmp_path / "config.json"
    assert not missing.exists()

    cfg = load_config(missing)

    assert cfg == Config()


def test_loading_a_missing_file_creates_NO_file(tmp_path):
    # THE POINT: first run must leave the filesystem untouched.
    missing = tmp_path / "config.json"

    load_config(missing)

    assert not missing.exists(), "load_config() created the config file"


def test_loading_a_missing_file_creates_no_directory_either(tmp_path):
    missing = tmp_path / "nested" / "deeper" / "config.json"

    cfg = load_config(missing)

    assert cfg == Config()
    assert not (tmp_path / "nested").exists()


def test_loading_a_missing_file_leaves_the_directory_empty(tmp_path):
    load_config(tmp_path / "config.json")
    assert os.listdir(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# default_config_path
# ---------------------------------------------------------------------------


def test_default_config_path_ends_with_blurt_config_json(monkeypatch):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    path = config.default_config_path()
    assert path.name == "config.json"
    assert path.parent.name == "blurt"
    assert path.parent.parent.name == ".config"


def test_default_config_path_is_absolute(monkeypatch):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert config.default_config_path().is_absolute()


def test_default_config_path_honours_xdg_config_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert config.default_config_path() == tmp_path / "blurt" / "config.json"


def test_default_config_path_ignores_a_relative_xdg_config_home(monkeypatch):
    # Per the XDG spec a relative value is invalid and must be ignored.
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative/path")
    path = config.default_config_path()
    assert path.is_absolute()
    assert path.parent.parent.name == ".config"


def test_default_config_path_does_not_create_anything(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config.default_config_path()
    assert os.listdir(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_round_trip_of_the_default_config(tmp_path):
    path = tmp_path / "config.json"
    original = Config()

    save_config(original, path)

    assert load_config(path) == original


def test_round_trip_of_a_fully_customized_config(tmp_path):
    path = tmp_path / "config.json"
    original = Config(
        engine="faster-whisper",
        model="base.en",
        hotkey="right_command",
        cleanup_level="standard",
        sample_rate=48000,
        preroll_ms=250,
        min_hold_ms=300,
        paste_delay_ms=90,
        clipboard_restore_ms=600,
        cpu_threads=4,
        keep_raw_history=False,
        dictionary={"github": "GitHub", "api": "API"},
    )

    save_config(original, path)
    loaded = load_config(path)

    assert loaded == original
    assert loaded.dictionary == {"github": "GitHub", "api": "API"}
    assert loaded.keep_raw_history is False
    assert loaded.cpu_threads == 4


def test_round_trip_preserves_every_field_individually(tmp_path):
    path = tmp_path / "config.json"
    original = Config(
        engine="apple-speech",
        model="tiny.en",
        hotkey="fn",
        cleanup_level="none",
        sample_rate=22050,
        preroll_ms=0,
        min_hold_ms=1,
        paste_delay_ms=2,
        clipboard_restore_ms=3,
        cpu_threads=8,
        keep_raw_history=True,
        dictionary={"k": "V"},
    )

    save_config(original, path)
    loaded = load_config(path)

    for field in dataclasses.fields(Config):
        assert getattr(loaded, field.name) == getattr(original, field.name), (
            "field %r did not survive the round trip" % (field.name,)
        )


def test_save_creates_parent_directories(tmp_path):
    path = tmp_path / "a" / "b" / "c" / "config.json"

    save_config(Config(), path)

    assert path.exists()
    assert load_config(path) == Config()


def test_save_writes_valid_readable_json(tmp_path):
    path = tmp_path / "config.json"
    save_config(Config(engine="apple-speech"), path)

    data = json.loads(path.read_text(encoding="utf-8"))

    assert isinstance(data, dict)
    assert data["engine"] == "apple-speech"


def test_save_overwrites_an_existing_config(tmp_path):
    path = tmp_path / "config.json"
    save_config(Config(cleanup_level="none"), path)
    save_config(Config(cleanup_level="standard"), path)

    assert load_config(path).cleanup_level == "standard"


def test_save_leaves_no_temp_files_behind(tmp_path):
    # The atomic write uses a temp file in the destination directory; it must
    # not survive a successful save.
    path = tmp_path / "config.json"
    save_config(Config(), path)

    assert os.listdir(str(tmp_path)) == ["config.json"]


def test_saved_config_is_not_world_readable(tmp_path):
    # It can hold a personal replacement dictionary.
    path = tmp_path / "config.json"
    save_config(Config(), path)

    mode = path.stat().st_mode & 0o077
    assert mode == 0, "config.json is readable by group/other"


# ---------------------------------------------------------------------------
# Corrupt JSON
# ---------------------------------------------------------------------------


def test_corrupt_json_does_not_raise(tmp_path):
    path = tmp_path / "config.json"
    _write(path, "{not json at all")

    cfg = load_config(path)  # must not raise

    assert cfg == Config()


def test_corrupt_json_backs_up_the_bad_file(tmp_path):
    path = tmp_path / "config.json"
    _write(path, "{not json at all")

    load_config(path)

    backup = tmp_path / "config.json.bak"
    assert backup.exists(), "the unreadable config was not backed up"
    assert backup.read_text(encoding="utf-8") == "{not json at all", (
        "the backup must preserve the user's bytes verbatim so they can "
        "recover a hand-written dictionary"
    )


def test_corrupt_json_moves_the_bad_file_rather_than_copying_it(tmp_path):
    # Renaming means we warn once, not on every launch.
    path = tmp_path / "config.json"
    _write(path, "{not json at all")

    load_config(path)

    assert not path.exists()


def test_corrupt_json_warns_on_stderr(tmp_path, capsys):
    path = tmp_path / "config.json"
    _write(path, "{not json at all")

    load_config(path)

    err = capsys.readouterr().err
    assert "blurt: config:" in err
    assert "not valid JSON" in err


def test_a_second_load_after_a_corrupt_file_is_silent_and_clean(tmp_path, capsys):
    path = tmp_path / "config.json"
    _write(path, "{not json")
    load_config(path)
    capsys.readouterr()  # discard the first warning

    cfg = load_config(path)

    assert cfg == Config()
    assert capsys.readouterr().err == ""


def test_truncated_json_is_handled(tmp_path):
    path = tmp_path / "config.json"
    _write(path, '{"engine": "faster-whis')

    assert load_config(path) == Config()


def test_empty_file_is_handled(tmp_path):
    path = tmp_path / "config.json"
    _write(path, "")

    assert load_config(path) == Config()


def test_json_that_is_a_list_is_rejected_and_backed_up(tmp_path):
    path = tmp_path / "config.json"
    _write(path, [1, 2, 3])

    assert load_config(path) == Config()
    assert (tmp_path / "config.json.bak").exists()


def test_json_that_is_a_bare_string_is_rejected(tmp_path):
    path = tmp_path / "config.json"
    _write(path, '"just a string"')

    assert load_config(path) == Config()


def test_json_null_is_rejected(tmp_path):
    path = tmp_path / "config.json"
    _write(path, "null")

    assert load_config(path) == Config()


def test_a_good_config_can_be_saved_over_a_corrupt_one(tmp_path):
    # The recovery path end to end: corrupt file -> defaults -> save -> good.
    path = tmp_path / "config.json"
    _write(path, "{not json")

    cfg = load_config(path)
    cfg.cleanup_level = "standard"
    save_config(cfg, path)

    assert load_config(path).cleanup_level == "standard"


# ---------------------------------------------------------------------------
# Unknown keys
# ---------------------------------------------------------------------------


def test_unknown_keys_are_ignored_not_fatal(tmp_path):
    path = tmp_path / "config.json"
    _write(path, {"engine": "apple-speech", "totally_unknown": 1})

    cfg = load_config(path)

    assert cfg.engine == "apple-speech"
    assert not hasattr(cfg, "totally_unknown")


def test_many_unknown_keys_do_not_prevent_loading_the_known_ones(tmp_path):
    path = tmp_path / "config.json"
    _write(
        path,
        {
            "cleanup_level": "standard",
            "from_a_newer_blurt": {"nested": True},
            "removed_feature": [1, 2, 3],
            "another": None,
        },
    )

    cfg = load_config(path)

    assert cfg.cleanup_level == "standard"
    assert cfg.engine == "auto"


def test_unknown_keys_warn_but_still_return_a_usable_config(tmp_path, capsys):
    path = tmp_path / "config.json"
    _write(path, {"whats_this": 1})

    cfg = load_config(path)

    assert cfg == Config()
    assert "unknown key" in capsys.readouterr().err


def test_an_unknown_key_does_not_back_up_the_file(tmp_path):
    # Forward compatibility: an older blurt reading a newer config must leave
    # that config alone, not move it aside.
    path = tmp_path / "config.json"
    _write(path, {"engine": "auto", "future_setting": True})

    load_config(path)

    assert path.exists()
    assert not (tmp_path / "config.json.bak").exists()


# ---------------------------------------------------------------------------
# Invalid values fall back per-key
# ---------------------------------------------------------------------------


def test_invalid_cleanup_level_falls_back_to_the_default(tmp_path):
    path = tmp_path / "config.json"
    _write(path, {"cleanup_level": "wild"})

    assert load_config(path).cleanup_level == "light"


def test_invalid_cleanup_level_warns(tmp_path, capsys):
    path = tmp_path / "config.json"
    _write(path, {"cleanup_level": "wild"})

    load_config(path)

    err = capsys.readouterr().err
    assert "cleanup_level" in err
    assert "wild" in err


def test_every_valid_cleanup_level_is_accepted(tmp_path):
    path = tmp_path / "config.json"
    for level in ("none", "light", "standard"):
        _write(path, {"cleanup_level": level})
        assert load_config(path).cleanup_level == level


def test_invalid_engine_falls_back_to_auto(tmp_path):
    path = tmp_path / "config.json"
    _write(path, {"engine": "nope"})

    assert load_config(path).engine == "auto"


def test_every_valid_engine_is_accepted(tmp_path):
    path = tmp_path / "config.json"
    for engine in ("auto", "faster-whisper", "apple-speech"):
        _write(path, {"engine": engine})
        assert load_config(path).engine == engine


def test_one_bad_value_does_not_discard_the_other_keys(tmp_path):
    # Per-key fallback, not whole-file rejection.
    path = tmp_path / "config.json"
    _write(path, {"cleanup_level": "wild", "hotkey": "fn", "cpu_threads": 6})

    cfg = load_config(path)

    assert cfg.cleanup_level == "light"  # fell back
    assert cfg.hotkey == "fn"  # kept
    assert cfg.cpu_threads == 6  # kept


def test_a_bad_value_does_not_back_up_the_file(tmp_path):
    path = tmp_path / "config.json"
    _write(path, {"cleanup_level": "wild"})

    load_config(path)

    assert path.exists()
    assert not (tmp_path / "config.json.bak").exists()


def test_wrong_type_for_a_string_field_falls_back(tmp_path):
    path = tmp_path / "config.json"
    _write(path, {"hotkey": 42})

    assert load_config(path).hotkey == "right_option"


def test_empty_string_for_a_string_field_falls_back(tmp_path):
    path = tmp_path / "config.json"
    _write(path, {"hotkey": "   "})

    assert load_config(path).hotkey == "right_option"


def test_wrong_type_for_an_int_field_falls_back(tmp_path):
    path = tmp_path / "config.json"
    _write(path, {"sample_rate": "16000"})

    assert load_config(path).sample_rate == 16000


def test_a_boolean_is_not_accepted_as_an_integer(tmp_path):
    # bool is a subclass of int; treating True as 1 would hide a real typo.
    path = tmp_path / "config.json"
    _write(path, {"cpu_threads": True})

    assert load_config(path).cpu_threads == 0


def test_out_of_range_sample_rate_falls_back(tmp_path):
    path = tmp_path / "config.json"
    _write(path, {"sample_rate": 1})
    assert load_config(path).sample_rate == 16000

    _write(path, {"sample_rate": 999999})
    assert load_config(path).sample_rate == 16000


def test_negative_millisecond_values_fall_back(tmp_path):
    path = tmp_path / "config.json"
    _write(path, {"preroll_ms": -100})

    assert load_config(path).preroll_ms == 500


def test_absurd_millisecond_values_fall_back(tmp_path):
    # A minute of preroll is certainly a unit mixup.
    path = tmp_path / "config.json"
    _write(path, {"paste_delay_ms": 10 ** 9})

    assert load_config(path).paste_delay_ms == 120


def test_wrong_type_for_a_boolean_field_falls_back(tmp_path):
    path = tmp_path / "config.json"
    _write(path, {"keep_raw_history": "yes"})

    assert load_config(path).keep_raw_history is True


def test_dictionary_that_is_not_an_object_is_ignored(tmp_path):
    path = tmp_path / "config.json"
    _write(path, {"dictionary": ["github", "GitHub"]})

    assert load_config(path).dictionary == {}


def test_bad_dictionary_entries_are_dropped_individually(tmp_path):
    path = tmp_path / "config.json"
    _write(
        path,
        {"dictionary": {"github": "GitHub", "bad": 7, "": "empty key", "api": "API"}},
    )

    cfg = load_config(path)

    assert cfg.dictionary == {"github": "GitHub", "api": "API"}


def test_a_config_full_of_garbage_still_yields_working_defaults(tmp_path):
    # The worst realistic hand-edit. Nothing raises, everything falls back, and
    # the user can still dictate.
    path = tmp_path / "config.json"
    _write(
        path,
        {
            "engine": 1,
            "model": None,
            "hotkey": [],
            "cleanup_level": "wild",
            "sample_rate": "fast",
            "preroll_ms": -1,
            "min_hold_ms": {},
            "paste_delay_ms": 10 ** 9,
            "clipboard_restore_ms": False,
            "cpu_threads": "many",
            "keep_raw_history": "yes",
            "dictionary": 3,
            "unknown_thing": "x",
        },
    )

    cfg = load_config(path)

    assert cfg == Config()


def test_load_config_never_raises_on_any_of_these_payloads(tmp_path):
    path = tmp_path / "config.json"
    payloads = [
        "",
        "   ",
        "{",
        "}",
        "[]",
        "null",
        "true",
        "3.14",
        '"string"',
        "{'single': 'quotes'}",
        '{"engine": }',
        '{"dictionary": {"a": {"b": "c"}}}',
        '{"nested": {"deeply": {"very": true}}}',
    ]
    for payload in payloads:
        _write(path, payload)
        cfg = load_config(path)
        assert isinstance(cfg, Config), "payload %r did not yield a Config" % (payload,)
        if path.exists():
            path.unlink()


# ---------------------------------------------------------------------------
# Interaction with the rest of the app
# ---------------------------------------------------------------------------


def test_a_loaded_cleanup_level_is_always_one_cleanup_accepts(tmp_path):
    # config is the only validator of cleanup_level; cleanup itself falls back
    # silently. This test is the seam between the two.
    from blurt.cleanup import clean

    path = tmp_path / "config.json"
    for candidate in ("none", "light", "standard", "wild", "", "LIGHT"):
        _write(path, {"cleanup_level": candidate})
        cfg = load_config(path)
        assert cfg.cleanup_level in ("none", "light", "standard")
        assert isinstance(clean("um hello", cfg.cleanup_level), str)


def test_a_loaded_dictionary_is_usable_by_cleanup(tmp_path):
    from blurt.cleanup import clean

    path = tmp_path / "config.json"
    _write(path, {"dictionary": {"github": "GitHub"}})

    cfg = load_config(path)

    assert clean("github is down", cfg.cleanup_level, cfg.dictionary) == (
        "GitHub is down"
    )


def test_config_is_json_serializable_via_dataclasses_asdict():
    payload: Dict[str, Any] = dataclasses.asdict(Config())
    assert json.loads(json.dumps(payload)) == payload


def test_EVERY_field_round_trips_through_save_and_load(tmp_path):
    # Regression guard for a real bug: _from_dict lists fields explicitly, so a
    # newly added Config field silently fails to load until it is wired in there.
    # This test fails the moment any field does not survive save -> load, without
    # anyone having to remember to test the new field by hand.
    path = tmp_path / "config.json"

    # A non-default value for every field, so a dropped field shows as a mismatch.
    custom = Config(
        engine="faster-whisper",
        model="base.en",
        hotkey="right_ctrl",
        cleanup_level="standard",
        sample_rate=24000,
        preroll_ms=300,
        min_hold_ms=150,
        paste_delay_ms=90,
        clipboard_restore_ms=350,
        cpu_threads=3,
        keep_raw_history=False,
        dictionary={"oauth": "OAuth"},
        initial_prompt="OAuth, Kubernetes, voxzerr",
        assistant_enabled=False,
        assistant_hotkey="right_ctrl",
    )
    # Guard against the test itself going stale: if a field is added to Config
    # but not given a non-default value above, this catches it.
    for f in dataclasses.fields(Config):
        assert getattr(custom, f.name) != getattr(Config(), f.name), (
            "field %r has no non-default value in this test; add one" % f.name
        )

    save_config(custom, path)
    loaded = load_config(path)

    for f in dataclasses.fields(Config):
        assert getattr(loaded, f.name) == getattr(custom, f.name), (
            "field %r did not survive save->load -- wire it into _from_dict" % f.name
        )
