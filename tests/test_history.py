"""Tests for the opt-in on-disk transcript journal.

Everything here runs against a temp directory, so the suite stays runnable on any
platform -- :mod:`blurt.history` is standard library only for exactly that reason.

The properties worth defending, in rough order of how much damage getting them
wrong would do:

  * A journalling failure NEVER propagates. It is called from the dictation worker
    right after the user spoke, and losing their words to a full disk would be a
    far worse bug than losing the record of them.
  * The file is 0600 inside a 0700 directory, always. It can contain anything the
    user has ever said out loud.
  * A torn or malformed line costs that line and nothing else.
  * Trimming keeps the NEWEST records. Keeping the oldest would mean a journal
    that stops learning the moment it fills up.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from blurt.history import (
    HistoryRecord,
    append_record,
    default_history_path,
    history_size,
    load_records,
    purge_history,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _FakeTranscript:
    """Stands in for blurt.types.Transcript without importing numpy's neighbours."""

    def __init__(self, raw="hello there", cleaned="Hello there.", engine="fw base.en"):
        self.raw = raw
        self.cleaned = cleaned
        self.engine = engine
        self.audio_seconds = 2.5
        self.latency_seconds = 0.91


def _record(text="hello", timestamp=1000.0, mode="dictate"):
    return HistoryRecord(
        timestamp=timestamp,
        mode=mode,
        raw=text,
        cleaned=text.capitalize(),
        engine="fw base.en",
        audio_seconds=1.0,
        latency_seconds=0.5,
    )


@pytest.fixture
def journal(tmp_path):
    return tmp_path / "blurt" / "history.jsonl"


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #


def test_default_path_honours_xdg_data_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert default_history_path() == tmp_path / "blurt" / "history.jsonl"


def test_default_path_ignores_relative_xdg_data_home(monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", "relative/path")
    path = default_history_path()
    # A relative XDG value is invalid per spec; we must not create a directory
    # relative to whatever the working directory happens to be.
    assert path.is_absolute()
    assert path.parts[-3:] == (".local", "share", "blurt") or path.name == "history.jsonl"


def test_default_path_falls_back_to_local_share(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_history_path() == tmp_path / ".local" / "share" / "blurt" / "history.jsonl"


def test_journal_is_not_beside_the_config(monkeypatch, tmp_path):
    """Data, not configuration -- a synced dotfiles directory must not pick it up."""
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    from blurt.config import default_config_path

    assert default_history_path().parent != default_config_path().parent


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #


def test_append_then_load_round_trips(journal):
    original = _record("kubernetes is fine", timestamp=1234.5)
    assert append_record(original, journal) is True

    loaded = load_records(journal)
    assert len(loaded) == 1
    assert loaded[0] == original


def test_records_load_oldest_first(journal):
    for index in range(5):
        append_record(_record("utterance %d" % index, timestamp=1000.0 + index), journal)

    loaded = load_records(journal)
    assert [record.timestamp for record in loaded] == [1000.0, 1001.0, 1002.0, 1003.0, 1004.0]


def test_load_limit_keeps_the_newest(journal):
    for index in range(10):
        append_record(_record("u%d" % index, timestamp=float(index)), journal)

    loaded = load_records(journal, limit=3)
    assert [record.timestamp for record in loaded] == [7.0, 8.0, 9.0]


def test_missing_journal_loads_as_empty(tmp_path):
    assert load_records(tmp_path / "nope.jsonl") == []
    assert history_size(tmp_path / "nope.jsonl") == 0


def test_unicode_survives_the_round_trip(journal):
    append_record(_record("café naïve 東京"), journal)
    assert load_records(journal)[0].raw == "café naïve 東京"


# --------------------------------------------------------------------------- #
# from_transcript
# --------------------------------------------------------------------------- #


def test_from_transcript_copies_the_fields():
    record = HistoryRecord.from_transcript(_FakeTranscript(), "dictate", timestamp=7.0)
    assert record.raw == "hello there"
    assert record.cleaned == "Hello there."
    assert record.engine == "fw base.en"
    assert record.audio_seconds == 2.5
    assert record.latency_seconds == 0.91
    assert record.timestamp == 7.0
    assert record.mode == "dictate"


def test_from_transcript_rejects_an_unknown_mode():
    record = HistoryRecord.from_transcript(_FakeTranscript(), "nonsense", timestamp=1.0)
    assert record.mode == "dictate"


def test_from_transcript_keeps_assistant_mode():
    record = HistoryRecord.from_transcript(_FakeTranscript(), "assistant", timestamp=1.0)
    assert record.mode == "assistant"


def test_from_transcript_with_raw_history_disabled():
    """keep_raw_history=false empties Transcript.raw upstream; that must compose."""
    transcript = _FakeTranscript(raw="")
    record = HistoryRecord.from_transcript(transcript, "dictate", timestamp=1.0)
    assert record.raw == ""
    assert record.cleaned == "Hello there."
    # text() still has something to analyse.
    assert record.text() == "Hello there."


def test_text_prefers_raw():
    record = HistoryRecord(1.0, "dictate", "raw form", "cleaned form", "e", 1.0, 1.0)
    assert record.text() == "raw form"


# --------------------------------------------------------------------------- #
# Permissions
# --------------------------------------------------------------------------- #


def test_file_is_owner_only(journal):
    append_record(_record(), journal)
    mode = stat.S_IMODE(os.stat(str(journal)).st_mode)
    assert mode == 0o600, "journal can contain anything the user has ever said"


def test_directory_is_owner_only(journal):
    append_record(_record(), journal)
    mode = stat.S_IMODE(os.stat(str(journal.parent)).st_mode)
    assert mode == 0o700


def test_existing_loose_directory_is_tightened(tmp_path):
    directory = tmp_path / "blurt"
    directory.mkdir(mode=0o755)
    append_record(_record(), directory / "history.jsonl")
    assert stat.S_IMODE(os.stat(str(directory)).st_mode) == 0o700


# --------------------------------------------------------------------------- #
# Malformed input
# --------------------------------------------------------------------------- #


def test_one_bad_line_does_not_lose_the_others(journal):
    append_record(_record("first", timestamp=1.0), journal)
    with open(str(journal), "a", encoding="utf-8") as handle:
        handle.write("{not json at all\n")
    append_record(_record("third", timestamp=3.0), journal)

    loaded = load_records(journal)
    assert [record.raw for record in loaded] == ["first", "third"]


def test_blank_lines_are_ignored(journal):
    append_record(_record("only"), journal)
    with open(str(journal), "a", encoding="utf-8") as handle:
        handle.write("\n\n   \n")
    assert len(load_records(journal)) == 1


def test_json_that_is_not_an_object_is_skipped(journal):
    journal.parent.mkdir(parents=True)
    journal.write_text('["a list"]\n42\n"a string"\n', encoding="utf-8")
    assert load_records(journal) == []


def test_record_with_no_text_is_skipped(journal):
    journal.parent.mkdir(parents=True)
    journal.write_text(json.dumps({"raw": "", "cleaned": ""}) + "\n", encoding="utf-8")
    assert load_records(journal) == []


def test_unknown_fields_are_tolerated(journal):
    """A journal written by a newer blurt must still load in an older one."""
    journal.parent.mkdir(parents=True)
    journal.write_text(
        json.dumps(
            {
                "timestamp": 5.0,
                "mode": "dictate",
                "raw": "hello",
                "cleaned": "Hello",
                "engine": "e",
                "audio_seconds": 1.0,
                "latency_seconds": 0.5,
                "some_future_field": {"nested": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_records(journal)
    assert len(loaded) == 1
    assert loaded[0].raw == "hello"


def test_numeric_fields_arriving_as_strings_are_coerced(journal):
    journal.parent.mkdir(parents=True)
    journal.write_text(
        json.dumps({"raw": "x", "cleaned": "X", "audio_seconds": "2.5"}) + "\n",
        encoding="utf-8",
    )
    assert load_records(journal)[0].audio_seconds == 2.5


def test_garbage_numeric_fields_degrade_to_zero(journal):
    journal.parent.mkdir(parents=True)
    journal.write_text(
        json.dumps({"raw": "x", "cleaned": "X", "latency_seconds": "not a number"}) + "\n",
        encoding="utf-8",
    )
    assert load_records(journal)[0].latency_seconds == 0.0


def test_unknown_mode_in_a_stored_record_degrades(journal):
    journal.parent.mkdir(parents=True)
    journal.write_text(
        json.dumps({"raw": "x", "cleaned": "X", "mode": "telepathy"}) + "\n",
        encoding="utf-8",
    )
    assert load_records(journal)[0].mode == "dictate"


# --------------------------------------------------------------------------- #
# Failure never propagates
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root ignores directory permissions, so there is nothing to block here",
)
def test_append_to_an_unwritable_directory_returns_false(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.mkdir(mode=0o500)
    try:
        assert append_record(_record(), blocked / "sub" / "history.jsonl") is False
    finally:
        blocked.chmod(0o700)


def test_append_where_a_parent_is_a_file_returns_false(tmp_path):
    not_a_directory = tmp_path / "regular_file"
    not_a_directory.write_text("", encoding="utf-8")
    assert append_record(_record(), not_a_directory / "history.jsonl") is False


def test_load_from_a_directory_returns_empty(tmp_path):
    directory = tmp_path / "a_directory"
    directory.mkdir()
    assert load_records(directory) == []


def test_append_survives_a_root_owned_path(tmp_path, monkeypatch):
    """Any OSError from the write path degrades to False, not an exception."""

    def explode(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "open", explode)
    assert append_record(_record(), tmp_path / "h.jsonl") is False


# --------------------------------------------------------------------------- #
# Trimming
# --------------------------------------------------------------------------- #


def test_trim_keeps_the_newest_records(journal):
    for index in range(400):
        append_record(_record("u%d" % index, timestamp=float(index)), journal, limit=10)

    loaded = load_records(journal)
    # Amortized trimming means the file may sit somewhat above the limit, but it
    # must never run away and must always be keeping the newest end.
    assert len(loaded) <= 40
    assert loaded[-1].timestamp == 399.0
    assert loaded[-1].raw == "u399"


def test_trim_is_bounded_over_a_long_run(journal):
    for index in range(2000):
        append_record(_record("word%d" % index, timestamp=float(index)), journal, limit=50)
    assert len(load_records(journal)) <= 200


def test_limit_of_zero_disables_trimming(journal):
    for index in range(50):
        append_record(_record("u%d" % index, timestamp=float(index)), journal, limit=0)
    assert len(load_records(journal)) == 50


def test_negative_limit_is_ignored(journal):
    for index in range(20):
        append_record(_record("u%d" % index, timestamp=float(index)), journal, limit=-5)
    assert len(load_records(journal)) == 20


def test_trim_preserves_file_permissions(journal):
    for index in range(400):
        append_record(_record("u%d" % index, timestamp=float(index)), journal, limit=10)
    assert stat.S_IMODE(os.stat(str(journal)).st_mode) == 0o600


def test_trim_leaves_no_temp_files_behind(journal):
    for index in range(400):
        append_record(_record("u%d" % index, timestamp=float(index)), journal, limit=10)
    leftovers = [p.name for p in journal.parent.iterdir() if p.name != journal.name]
    assert leftovers == []


# --------------------------------------------------------------------------- #
# Size and purge
# --------------------------------------------------------------------------- #


def test_history_size_counts_records(journal):
    for index in range(7):
        append_record(_record("u%d" % index), journal, limit=0)
    assert history_size(journal) == 7


def test_purge_removes_the_file(journal):
    append_record(_record(), journal)
    assert purge_history(journal) is True
    assert not journal.exists()
    assert load_records(journal) == []


def test_purge_of_a_missing_file_is_false_not_an_error(tmp_path):
    assert purge_history(tmp_path / "never_existed.jsonl") is False


def test_purge_is_idempotent(journal):
    append_record(_record(), journal)
    assert purge_history(journal) is True
    assert purge_history(journal) is False
