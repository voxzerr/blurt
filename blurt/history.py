"""Opt-in, on-disk journal of finished dictations.

blurt's in-memory history (``BlurtApp._history``, 20 entries, RAM only) exists so
:meth:`~blurt.app.BlurtApp.revert_last` has something to revert to. This module is
a different thing with a different risk profile: a durable record of what you said,
written to disk, kept across restarts.

That record is the only way blurt can ever learn your vocabulary -- you cannot spot
that it mishears "Kubernetes" the same way every time from a buffer that is erased
when you quit. :mod:`blurt.learn` reads this file and turns it into concrete
``dictionary`` and ``initial_prompt`` suggestions.

IT IS OFF BY DEFAULT, AND THAT IS NOT A DEFAULT WE WILL FLIP
------------------------------------------------------------
Everything blurt does is designed so that dictated speech is transient. This file
is the single exception, and it is the kind of exception that has to be chosen
rather than discovered:

  * ``history_enabled`` defaults to ``False``. Nothing is written until the user
    sets it.
  * The file is created ``0600`` inside a ``0700`` directory. It can contain
    anything the user has ever said out loud.
  * ``blurt learn --forget`` deletes it outright, and that is advertised in the
    same breath as the feature itself.
  * It never leaves the machine. blurt has no network path to send it down, which
    is a property of the whole program rather than a promise made here.
  * ``keep_raw_history=False`` composes as you would expect: the app hands us a
    :class:`~blurt.types.Transcript` whose ``raw`` is already empty, so the journal
    stores only cleaned text. The user does not have to reason about two flags
    interacting.

WHERE IT LIVES
--------------
``$XDG_DATA_HOME/blurt/history.jsonl``, falling back to
``~/.local/share/blurt/history.jsonl``. Data, not configuration, so it does not
belong next to ``config.json`` -- a user syncing their dotfiles should not
accidentally sync a transcript of everything they have said.

One JSON object per line. The format is deliberately boring and greppable: a user
who wants to audit or delete part of their own history should be able to do it
with the tools they already have, not with a blurt subcommand we happened to write.

WHAT CAN GO WRONG
-----------------
  * **The disk is full, or the directory is not writable.** Journalling is a
    bookkeeping nicety; dictation is the product. Every failure here degrades to a
    single warning on stderr and is never raised at the caller, which is the
    dictation worker thread.
  * **The warning would repeat on every utterance.** A broken disk would otherwise
    print once per dictation forever. ``_warn_once`` keeps it to one message per
    process per failure kind.
  * **A partially-written line.** Records are appended as one ``write()`` of a
    single line by a single process, so a torn line needs a crash mid-syscall. It
    is still possible, so :func:`load_records` skips unparseable lines rather than
    refusing to load the file -- one bad line must not cost the user their history.
  * **Unbounded growth.** ``history_limit`` caps the record count. Trimming is
    amortized: a cheap file-size check decides whether to pay for the real
    read-and-rewrite, so the steady-state cost of a dictation stays one append.
    The consequence is that the file can briefly hold somewhat more than the limit,
    which is a trade we are making knowingly.
  * **``os.replace`` is only atomic within a filesystem**, so the trim's temp file
    is created in the destination directory rather than ``/tmp`` -- on modern macOS
    those are separate volumes. Same reasoning as :mod:`blurt.config`.

Python 3.9 floor: lazy annotations, typing generics only, no PEP 604 unions.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Set

__all__ = [
    "HistoryRecord",
    "default_history_path",
    "append_record",
    "load_records",
    "purge_history",
    "history_size",
]


#: Modes a record can be attributed to. Assistant commands are speech too, and
#: their vocabulary is exactly as worth learning as dictation's, so both are kept
#: and the mode is recorded rather than used to filter at write time.
_VALID_MODES = frozenset({"dictate", "assistant"})
_DEFAULT_MODE = "dictate"

#: Rough per-record size, used only to decide *whether* to check the real record
#: count. Wrong by 2x in either direction costs nothing but the timing of a trim.
_ASSUMED_RECORD_BYTES = 400

#: Trim only once the file is meaningfully over the limit, so a user running at
#: exactly ``history_limit`` records does not rewrite the whole file every time
#: they speak.
_TRIM_SLACK = 1.25

#: Failure kinds we have already complained about in this process.
_warned: Set[str] = set()


def _warn_once(kind: str, message: str) -> None:
    """Print a warning to stderr at most once per failure kind. Never raises."""
    if kind in _warned:
        return
    _warned.add(kind)
    try:
        print("blurt: history: " + message, file=sys.stderr)
    except Exception:  # pragma: no cover - stderr detached (launchd, py2app)
        pass


@dataclass(frozen=True)
class HistoryRecord:
    """One journalled dictation.

    Mirrors :class:`blurt.types.Transcript` plus the two things a transcript does
    not carry and an analysis needs: when it happened, and which hotkey produced
    it. ``raw`` is empty when the user has ``keep_raw_history`` switched off.
    """

    timestamp: float          # unix seconds, when the dictation finished
    mode: str                 # "dictate" | "assistant"
    raw: str                  # exactly what the engine produced, "" if not kept
    cleaned: str              # what was actually inserted
    engine: str               # e.g. "faster-whisper base.en"
    audio_seconds: float
    latency_seconds: float

    @classmethod
    def from_transcript(
        cls,
        transcript: Any,
        mode: str = _DEFAULT_MODE,
        timestamp: Optional[float] = None,
    ) -> "HistoryRecord":
        """Build a record from a :class:`~blurt.types.Transcript`.

        Reads attributes defensively so a caller passing something transcript-like
        (a test double, a future type with extra fields) works without ceremony.
        """
        return cls(
            timestamp=float(timestamp if timestamp is not None else time.time()),
            mode=mode if mode in _VALID_MODES else _DEFAULT_MODE,
            raw=str(getattr(transcript, "raw", "") or ""),
            cleaned=str(getattr(transcript, "cleaned", "") or ""),
            engine=str(getattr(transcript, "engine", "") or ""),
            audio_seconds=float(getattr(transcript, "audio_seconds", 0.0) or 0.0),
            latency_seconds=float(getattr(transcript, "latency_seconds", 0.0) or 0.0),
        )

    def text(self) -> str:
        """The best text to analyse for this record.

        Prefers ``raw``: it is what the engine actually heard, before the
        dictionary rewrote anything, which is the only view in which a
        misrecognition is still visible. Falls back to ``cleaned`` when raw
        history is disabled -- degraded but far better than skipping the record.
        """
        return self.raw or self.cleaned

    def to_json(self) -> str:
        """Serialise to one line of JSON. Keys are spelled out, not abbreviated."""
        return json.dumps(
            {
                "timestamp": round(self.timestamp, 3),
                "mode": self.mode,
                "raw": self.raw,
                "cleaned": self.cleaned,
                "engine": self.engine,
                "audio_seconds": round(self.audio_seconds, 3),
                "latency_seconds": round(self.latency_seconds, 3),
            },
            ensure_ascii=False,
            sort_keys=True,
        )


def default_history_path() -> pathlib.Path:
    """Return the journal location, honouring ``XDG_DATA_HOME``.

    Per the XDG spec a relative ``XDG_DATA_HOME`` is invalid and must be ignored,
    so we fall back to ``~/.local/share`` in that case -- matching how
    :func:`blurt.config.default_config_path` treats ``XDG_CONFIG_HOME``.
    """
    raw = os.environ.get("XDG_DATA_HOME", "")
    if raw:
        base = pathlib.Path(os.path.expanduser(raw))
        if base.is_absolute():
            return base / "blurt" / "history.jsonl"
        _warn_once("xdg", "ignoring relative XDG_DATA_HOME=%r" % (raw,))
    return (
        pathlib.Path(os.path.expanduser("~"))
        / ".local"
        / "share"
        / "blurt"
        / "history.jsonl"
    )


def _resolve(path: Optional[pathlib.Path]) -> pathlib.Path:
    return pathlib.Path(path) if path is not None else default_history_path()


def _coerce(obj: Any) -> Optional[HistoryRecord]:
    """Turn one parsed JSON object into a record, or None if it is not one.

    Tolerant on purpose. A record written by a newer blurt with extra fields, or
    one where a float arrived as a string, is still worth keeping -- the analysis
    downstream cares about the text far more than about the numbers.
    """
    if not isinstance(obj, dict):
        return None

    cleaned = obj.get("cleaned")
    raw = obj.get("raw")
    if not isinstance(cleaned, str):
        cleaned = ""
    if not isinstance(raw, str):
        raw = ""
    if not (raw or cleaned):
        return None  # a record with no text teaches us nothing

    def _number(key: str) -> float:
        value = obj.get(key)
        if isinstance(value, bool):
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0

    mode = obj.get("mode")
    engine = obj.get("engine")
    return HistoryRecord(
        timestamp=_number("timestamp"),
        mode=mode if isinstance(mode, str) and mode in _VALID_MODES else _DEFAULT_MODE,
        raw=raw,
        cleaned=cleaned,
        engine=engine if isinstance(engine, str) else "",
        audio_seconds=_number("audio_seconds"),
        latency_seconds=_number("latency_seconds"),
    )


def _ensure_directory(directory: pathlib.Path) -> None:
    """Create the journal directory ``0700``, tightening it if it already exists.

    ``mkdir(mode=...)`` is a no-op for an existing directory, and a user could
    plausibly have created ``~/.local/share/blurt`` themselves with a laxer umask.
    A best-effort ``chmod`` closes that: failing to tighten someone else's
    directory is not a reason to refuse to journal.
    """
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(str(directory), 0o700)
    except OSError:
        pass


def append_record(
    record: HistoryRecord,
    path: Optional[pathlib.Path] = None,
    limit: int = 2000,
) -> bool:
    """Append one record to the journal. Returns True if it was written.

    NEVER RAISES. This is called from the dictation worker thread immediately
    after the user's words have been delivered, and a journalling failure must
    not cost them a dictation they already spoke. Every problem degrades to
    ``False`` plus one warning per process.

    ``limit`` caps the number of records kept; see the module docstring on why
    trimming is amortized rather than exact.
    """
    target = _resolve(path)
    line = record.to_json() + "\n"

    try:
        _ensure_directory(target.parent)
        # Open with 0600 from the start rather than chmod-ing afterwards: there
        # must be no window in which the file exists and is group-readable.
        fd = os.open(
            str(target), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
        )
        try:
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(line)
        except BaseException:
            # fdopen owns the descriptor on success; on failure it may not have.
            try:
                os.close(fd)
            except OSError:
                pass
            raise
    except OSError as exc:
        _warn_once(
            "append",
            "could not write %s (%s); history is not being recorded" % (target, exc),
        )
        return False

    _maybe_trim(target, limit)
    return True


def _maybe_trim(path: pathlib.Path, limit: int) -> None:
    """Rewrite the journal keeping only the newest ``limit`` records, if needed.

    The size check is the whole point: it makes the common case (append, do
    nothing else) cost one ``stat``. Never raises -- a journal we cannot trim is
    a journal that grows, which is untidy but not harmful.
    """
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return
    if limit <= 0:
        return

    threshold = int(limit * _TRIM_SLACK * _ASSUMED_RECORD_BYTES)
    try:
        if path.stat().st_size <= threshold:
            return
    except OSError:
        return

    try:
        with open(str(path), "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError as exc:
        _warn_once("trim", "could not read %s to trim it (%s)" % (path, exc))
        return

    if len(lines) <= limit:
        # Long records rather than too many of them. Nothing to do, and the size
        # check will keep firing until the count genuinely grows -- a stat per
        # dictation, which is free.
        return

    keep = lines[-limit:]
    directory = path.parent
    tmp_name = None
    try:
        fd, tmp_name = tempfile.mkstemp(
            dir=str(directory), prefix="." + path.name + ".", suffix=".tmp"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.writelines(keep)
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, str(path))
    except OSError as exc:
        _warn_once("trim", "could not trim %s (%s)" % (path, exc))
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def load_records(
    path: Optional[pathlib.Path] = None,
    limit: Optional[int] = None,
) -> List[HistoryRecord]:
    """Read the journal, oldest first. Returns ``[]`` when there is nothing to read.

    NEVER RAISES. A missing file is the ordinary first-run state and produces no
    output at all. Unparseable lines are skipped individually, because one torn
    line must not cost the user everything else they have said.

    ``limit`` keeps only the newest N records.
    """
    target = _resolve(path)

    try:
        with open(str(target), "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        return []
    except OSError as exc:
        _warn_once("read", "could not read %s (%s)" % (target, exc))
        return []

    if limit is not None:
        try:
            bound = int(limit)
        except (TypeError, ValueError):
            bound = 0
        if bound > 0:
            lines = lines[-bound:]

    records: List[HistoryRecord] = []
    skipped = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            skipped += 1
            continue
        record = _coerce(obj)
        if record is None:
            skipped += 1
            continue
        records.append(record)

    if skipped:
        _warn_once(
            "skipped",
            "skipped %d unreadable line(s) in %s" % (skipped, target),
        )
    return records


def history_size(path: Optional[pathlib.Path] = None) -> int:
    """Number of records currently on disk. 0 when there is no journal.

    Counts lines rather than parsing them, so this stays cheap enough for
    ``blurt doctor`` to call unconditionally. Never raises.
    """
    target = _resolve(path)
    try:
        with open(str(target), "r", encoding="utf-8", errors="replace") as handle:
            return sum(1 for line in handle if line.strip())
    except FileNotFoundError:
        return 0
    except OSError:
        return 0


def purge_history(path: Optional[pathlib.Path] = None) -> bool:
    """Delete the journal. Returns True if a file was removed.

    Deliberately a plain unlink rather than an overwrite-then-unlink: on a
    copy-on-write filesystem (APFS is one) overwriting in place does not reliably
    destroy the old blocks anyway, so a scrub would be security theatre. Say what
    it does and let the user reach for FileVault, which actually solves this.
    """
    target = _resolve(path)
    try:
        os.unlink(str(target))
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        _warn_once("purge", "could not delete %s (%s)" % (target, exc))
        return False
