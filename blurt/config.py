"""JSON-backed user configuration for blurt.

Stores settings at ``$XDG_CONFIG_HOME/blurt/config.json`` (falling back to
``~/.config/blurt/config.json``). The guiding rule here is that a bad config
file must never stop the user from dictating: every problem degrades to a
default plus a warning on stderr, never an exception.

Behaviour:
  * Missing file          -> return defaults, write nothing.
  * Corrupt JSON          -> warn, rename it to ``config.json.bak``, use defaults.
  * Unknown keys          -> silently ignored (forward compatibility with newer
                             versions of blurt that add settings).
  * Bad value for a key   -> warn, use the default for that key only.
  * ``save_config``       -> temp file in the same directory + ``os.replace``,
                             so a crash or a full disk mid-write cannot leave a
                             half-written config behind.

What can go wrong on macOS:
  * ``~/.config`` may not exist yet on a fresh Mac; we create it on save only.
  * If blurt is ever run from a sandboxed wrapper (or ``HOME`` is redirected to
    a read-only container), ``mkdir``/``open`` raise ``OSError``. ``save_config``
    lets that propagate so the caller can surface it; ``load_config`` swallows
    it and returns defaults.
  * ``os.replace`` is only atomic within a single filesystem, which is why the
    temp file is created in the destination directory rather than ``/tmp``
    (``/tmp`` is a separate volume from the data volume on modern macOS).
  * If the config directory lives in iCloud Drive or Dropbox, the sync daemon
    can briefly hold or evict the file; a failed read is treated as "no config"
    rather than an error.

Python 3.9 compatible: no PEP 604 unions, no builtin generics at runtime.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional

__all__ = [
    "Config",
    "VALID_ENGINES",
    "VALID_CLEANUP_LEVELS",
    "default_config_path",
    "load_config",
    "save_config",
]


VALID_ENGINES: FrozenSet[str] = frozenset({"auto", "faster-whisper", "apple-speech"})
VALID_CLEANUP_LEVELS: FrozenSet[str] = frozenset({"none", "light", "standard"})

# Sample rates we are willing to hand to sounddevice / faster-whisper. Whisper
# itself wants 16 kHz; anything else means a resample somewhere downstream.
_MIN_SAMPLE_RATE = 8000
_MAX_SAMPLE_RATE = 48000

# Upper bound on the various millisecond knobs. One minute of preroll or paste
# delay is already absurd; beyond that it is certainly a typo or a unit mixup.
_MAX_MS = 60000

# More threads than this on any Mac blurt supports is a misconfiguration.
_MAX_THREADS = 64


@dataclass
class Config:
    """User-tunable settings. All fields have working defaults."""

    engine: str = "auto"            # "auto" | "faster-whisper" | "apple-speech"
    model: str = "auto"             # "auto" | "tiny.en" | "base.en" | "small.en" ...
    hotkey: str = "right_option"
    cleanup_level: str = "light"    # "none" | "light" | "standard"
    sample_rate: int = 16000
    preroll_ms: int = 500
    min_hold_ms: int = 200
    paste_delay_ms: int = 120
    clipboard_restore_ms: int = 400
    cpu_threads: int = 0            # 0 = auto
    keep_raw_history: bool = True
    dictionary: Dict[str, str] = field(default_factory=dict)


def _warn(message: str) -> None:
    """Print a warning to stderr. Never raises, even if stderr is closed."""
    try:
        print("blurt: config: " + message, file=sys.stderr)
    except Exception:  # pragma: no cover - stderr detached (launchd, py2app)
        pass


def default_config_path() -> pathlib.Path:
    """Return the config file location, honouring ``XDG_CONFIG_HOME``.

    Per the XDG spec a relative ``XDG_CONFIG_HOME`` is invalid and must be
    ignored, so we fall back to ``~/.config`` in that case.
    """
    raw = os.environ.get("XDG_CONFIG_HOME", "")
    if raw:
        base = pathlib.Path(os.path.expanduser(raw))
        if base.is_absolute():
            return base / "blurt" / "config.json"
        _warn("ignoring relative XDG_CONFIG_HOME=%r" % (raw,))
    return pathlib.Path(os.path.expanduser("~")) / ".config" / "blurt" / "config.json"


def _pick_str(
    data: Dict[str, Any],
    key: str,
    default: str,
    allowed: Optional[FrozenSet[str]] = None,
) -> str:
    """Read a string field, falling back to ``default`` with a warning."""
    if key not in data:
        return default
    value = data[key]
    if not isinstance(value, str):
        _warn("%s must be a string, got %s; using %r" % (key, type(value).__name__, default))
        return default
    value = value.strip()
    if not value:
        _warn("%s is empty; using %r" % (key, default))
        return default
    if allowed is not None and value not in allowed:
        _warn(
            "%s=%r is not one of %s; using %r"
            % (key, value, "/".join(sorted(allowed)), default)
        )
        return default
    return value


def _pick_int(
    data: Dict[str, Any],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Read an int field, clamping nothing: out-of-range means use the default."""
    if key not in data:
        return default
    value = data[key]
    # bool is a subclass of int; treating True as 1 here would hide a real typo.
    if isinstance(value, bool) or not isinstance(value, int):
        _warn("%s must be an integer, got %s; using %d" % (key, type(value).__name__, default))
        return default
    if not (minimum <= value <= maximum):
        _warn("%s=%d is outside %d..%d; using %d" % (key, value, minimum, maximum, default))
        return default
    return value


def _pick_bool(data: Dict[str, Any], key: str, default: bool) -> bool:
    if key not in data:
        return default
    value = data[key]
    if not isinstance(value, bool):
        _warn("%s must be true or false, got %s; using %r" % (key, type(value).__name__, default))
        return default
    return value


def _pick_dictionary(data: Dict[str, Any], key: str) -> Dict[str, str]:
    """Read the replacement dictionary, dropping individual bad entries."""
    if key not in data:
        return {}
    value = data[key]
    if not isinstance(value, dict):
        _warn("%s must be an object, got %s; ignoring it" % (key, type(value).__name__))
        return {}
    result: Dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            _warn("ignoring non-string %s entry %r" % (key, raw_key))
            continue
        if not raw_key.strip():
            _warn("ignoring empty %s key" % (key,))
            continue
        result[raw_key] = raw_value
    return result


def _from_dict(data: Dict[str, Any]) -> Config:
    """Build a Config from parsed JSON. Unknown keys are ignored."""
    defaults = Config()
    known = set(f.name for f in dataclasses.fields(Config))
    unknown = sorted(k for k in data.keys() if k not in known)
    if unknown:
        # Not a problem: an older blurt reading a newer config, or a leftover
        # key from a removed feature. Mention it once, quietly, and move on.
        _warn("ignoring unknown key(s): %s" % (", ".join(unknown),))

    return Config(
        engine=_pick_str(data, "engine", defaults.engine, VALID_ENGINES),
        model=_pick_str(data, "model", defaults.model),
        hotkey=_pick_str(data, "hotkey", defaults.hotkey),
        cleanup_level=_pick_str(
            data, "cleanup_level", defaults.cleanup_level, VALID_CLEANUP_LEVELS
        ),
        sample_rate=_pick_int(
            data, "sample_rate", defaults.sample_rate, _MIN_SAMPLE_RATE, _MAX_SAMPLE_RATE
        ),
        preroll_ms=_pick_int(data, "preroll_ms", defaults.preroll_ms, 0, _MAX_MS),
        min_hold_ms=_pick_int(data, "min_hold_ms", defaults.min_hold_ms, 0, _MAX_MS),
        paste_delay_ms=_pick_int(data, "paste_delay_ms", defaults.paste_delay_ms, 0, _MAX_MS),
        clipboard_restore_ms=_pick_int(
            data, "clipboard_restore_ms", defaults.clipboard_restore_ms, 0, _MAX_MS
        ),
        cpu_threads=_pick_int(data, "cpu_threads", defaults.cpu_threads, 0, _MAX_THREADS),
        keep_raw_history=_pick_bool(data, "keep_raw_history", defaults.keep_raw_history),
        dictionary=_pick_dictionary(data, "dictionary"),
    )


def _backup_corrupt(path: pathlib.Path) -> None:
    """Move an unparseable config aside so the next save starts clean.

    Renaming (rather than copying) means we warn once instead of on every
    launch. Best effort: if the rename fails we simply carry on with defaults.
    """
    backup = path.with_name(path.name + ".bak")
    try:
        os.replace(str(path), str(backup))
        _warn("moved unreadable config to %s" % (backup,))
    except OSError as exc:
        _warn("could not back up %s (%s)" % (path, exc))


def load_config(path: Optional[pathlib.Path] = None) -> Config:
    """Load the config, returning defaults for anything missing or invalid.

    Never raises and never creates the file. A missing config is the normal
    first-run case and produces no output at all.
    """
    target = pathlib.Path(path) if path is not None else default_config_path()

    try:
        with open(str(target), "r", encoding="utf-8") as handle:
            text = handle.read()
    except FileNotFoundError:
        return Config()
    except (OSError, UnicodeDecodeError) as exc:
        # Unreadable (permissions, iCloud eviction) or not valid UTF-8. Don't
        # touch the file: the bytes may still be recoverable by the user.
        _warn("could not read %s (%s); using defaults" % (target, exc))
        return Config()

    try:
        data = json.loads(text)
    except ValueError as exc:  # JSONDecodeError on 3.9, but keep it broad
        _warn("%s is not valid JSON (%s); using defaults" % (target, exc))
        _backup_corrupt(target)
        return Config()

    if not isinstance(data, dict):
        _warn(
            "%s must contain a JSON object, got %s; using defaults"
            % (target, type(data).__name__)
        )
        _backup_corrupt(target)
        return Config()

    return _from_dict(data)


def save_config(cfg: Config, path: Optional[pathlib.Path] = None) -> None:
    """Write the config atomically, creating parent directories as needed.

    Raises ``OSError`` if the directory cannot be created or written; callers
    that treat saving as optional should catch it.
    """
    target = pathlib.Path(path) if path is not None else default_config_path()
    directory = target.parent
    directory.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(dataclasses.asdict(cfg), indent=2, sort_keys=True) + "\n"

    # Temp file must share the destination filesystem for os.replace to be atomic.
    fd, tmp_name = tempfile.mkstemp(
        dir=str(directory), prefix="." + target.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, 0o600)  # may hold a personal replacement dictionary
        os.replace(tmp_name, str(target))
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

    # Best effort: make the rename itself durable. Not supported everywhere.
    try:
        dir_fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass
