"""Detect what machine blurt is actually running on, and size the ASR model to it.

Everything here runs during startup, before the menu bar item appears. So the
governing rule of this module is: NEVER RAISE. Every probe is wrapped, every
failure degrades to a conservative default. A machine we cannot measure gets
treated as a slow machine, which is the safe direction to be wrong in -- the
worst outcome is a smaller model than necessary, not a crash or a 24-second
transcription.

What can go wrong on macOS:

  - The Rosetta trap. A Python interpreter running under Rosetta 2 on Apple
    Silicon reports platform.machine() == "x86_64". Believing it would label an
    M-series Mac as Intel and hand it a small model plus a bad thread count. We
    correct for this by asking the kernel two questions: `sysctl.proc_translated`
    (1 = this process is being translated) and `hw.optional.arm64` (1 = the
    silicon is arm64). Note that `hw.optional.arm64` is itself translated away
    under Rosetta on some releases, which is exactly why we check both.

  - Absent sysctl keys are NORMAL, not errors. On a real Intel Mac neither
    `sysctl.proc_translated` nor `hw.optional.arm64` exists, and `sysctl -n`
    exits non-zero with "unknown oid" on stderr. On Apple Silicon,
    `machdep.cpu.brand_string` does not exist. Absent always means "no" or
    "fall back", never "fail".

  - sysctl itself can be missing, slow, or blocked in a sandboxed or restricted
    environment. Every call has a timeout and catches OSError, so a hung or
    absent binary costs us a default value rather than a hang at launch.

  - platform.mac_ver() can return empty strings when Python is confused about
    its host. We parse defensively and fall back to (0, 0, 0).

Model sizing is driven by measurements on the Intel floor machine (i7-7567U,
2 physical cores, faster-whisper int8, beam_size=1, model resident).

THREAD COUNT IS NOT A FREE KNOB. Measured, same machine, base.en:

    threads=2 (physical cores)   3s speech -> 1.35s   11s speech -> 1.80s
    threads=4 (hyperthreads)     3s speech -> 3.86s   11s speech -> 10.74s  (median)

Hyperthread siblings contend for the same AVX2/FMA execution ports, so
oversubscribing does not just fail to help -- it destroys tail latency, which
is the part users actually feel. recommend_threads() therefore returns PHYSICAL
cores, never logical ones. An earlier revision of this file used cores * 2 and
was measurably wrong.

Model sizing at threads=2:

    tiny.en   ~1.15s for 3s audio   -- only ~200ms faster than base.en, and
                                       measurably less accurate ("And so am I
                                       fellow Americans" vs "And so, my fellow
                                       Americans!"). Not worth the trade.
    base.en   ~1.35s for 3s audio   | ~1.80s for 11s audio   <- the default
    small.en  ~8.70s for 11s audio  + ~20s cold load          <- unusable here

So base.en is both the floor AND the ceiling on the "slow" tier: never downgrade
to tiny.en by default, never offer small.en at all.

Note that the short clip is not meaningfully faster than the long one: Whisper
pads every input to a 30-second window, so latency is roughly constant no matter
how briefly the user speaks. Budget in fixed milliseconds, not realtime factors.

Python 3.9 floor: lazy annotations, typing.Optional / typing.Tuple only.
"""

from __future__ import annotations

import platform
import subprocess
from typing import Optional, Tuple

from .types import Hardware

# Conservative fallbacks, used whenever a probe fails. These deliberately
# describe a weak machine: being wrong in this direction costs accuracy, being
# wrong in the other direction costs a 24-second transcription.
_DEFAULT_PHYSICAL_CORES = 2
_DEFAULT_RAM_GB = 8.0
_DEFAULT_CPU_BRAND = "Unknown CPU"

# sysctl should answer instantly; anything slower is a broken environment.
_SYSCTL_TIMEOUT_SECONDS = 2.0

_BYTES_PER_GB = 1024 ** 3

_MAX_THREADS = 8


def _sysctl(key: str) -> Optional[str]:
    """Read one sysctl key, or return None.

    None means "absent or unreadable" and is an expected, ordinary answer -- an
    unknown key exits non-zero, which we treat as absence rather than failure.
    This function never raises.
    """
    try:
        proc = subprocess.run(
            ["/usr/sbin/sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=_SYSCTL_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        # Covers sysctl missing (FileNotFoundError), permission problems, and
        # TimeoutExpired. All of them mean: we do not get to know this value.
        return None

    if proc.returncode != 0:
        return None

    value = (proc.stdout or "").strip()
    return value or None


def _sysctl_int(key: str) -> Optional[int]:
    """Read a sysctl key as an int, or return None if absent/non-numeric."""
    raw = _sysctl(key)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _detect_under_rosetta() -> bool:
    """True if this process is being translated by Rosetta 2.

    The key is absent on real Intel Macs and on natively-running arm64 Python;
    absent means "no".
    """
    return _sysctl_int("sysctl.proc_translated") == 1


def _detect_apple_silicon(under_rosetta: bool) -> bool:
    """True if the underlying silicon is arm64, regardless of how we are running."""
    if under_rosetta:
        # Only Apple Silicon can translate x86_64, so this is conclusive.
        return True
    if _sysctl_int("hw.optional.arm64") == 1:
        return True
    # Last resort: trust the interpreter. Reliable when we are NOT translated,
    # which the checks above have already established.
    return platform.machine().lower() in ("arm64", "aarch64")


def _detect_cpu_brand(is_apple_silicon: bool) -> str:
    """Human-readable CPU name.

    machdep.cpu.brand_string exists on Intel but not on Apple Silicon, so we
    fall back to the hardware model identifier and finally to a generic label.
    """
    brand = _sysctl("machdep.cpu.brand_string")
    if brand:
        return brand

    model = _sysctl("hw.model")
    if model:
        return model

    return "Apple Silicon" if is_apple_silicon else _DEFAULT_CPU_BRAND


def _detect_physical_cores() -> int:
    """Physical (not logical) core count; the floor machine has 2."""
    cores = _sysctl_int("hw.physicalcpu")
    if cores is None or cores < 1:
        return _DEFAULT_PHYSICAL_CORES
    return cores


def _detect_ram_gb() -> float:
    """Installed RAM in GB, rounded to one decimal."""
    memsize = _sysctl_int("hw.memsize")
    if memsize is None or memsize <= 0:
        return _DEFAULT_RAM_GB
    return round(memsize / _BYTES_PER_GB, 1)


def _detect_macos_version() -> Tuple[int, int, int]:
    """macOS version as a 3-tuple, e.g. (13, 7, 8). (0, 0, 0) if unknown.

    platform.mac_ver() can hand back an empty string or a two-part version, so
    we pad and parse defensively rather than indexing blindly.
    """
    try:
        raw = platform.mac_ver()[0]
    except Exception:
        return (0, 0, 0)

    if not raw:
        return (0, 0, 0)

    parts = raw.split(".")
    numbers = []
    for index in range(3):
        try:
            numbers.append(int(parts[index]))
        except (IndexError, ValueError):
            numbers.append(0)

    return (numbers[0], numbers[1], numbers[2])


def _classify_tier(is_apple_silicon: bool, physical_cores: int) -> str:
    """Bucket the machine into "fast" | "medium" | "slow".

    Apple Silicon is always fast. Intel splits on physical core count: the floor
    machine (2 physical cores) must land in "slow", because small.en took ~8.7s
    for 11s of audio on it plus a ~20s cold load.

    Both Intel tiers currently resolve to base.en; the split exists so a 4+ core
    Intel machine can be offered small.en as an opt-in later without that toggle
    ever appearing on a 2-core machine that cannot carry it.
    """
    if is_apple_silicon:
        return "fast"
    if physical_cores >= 4:
        return "medium"
    return "slow"


def detect() -> Hardware:
    """Probe the machine. Never raises; unknown values degrade to safe defaults."""
    under_rosetta = _detect_under_rosetta()
    is_apple_silicon = _detect_apple_silicon(under_rosetta)
    physical_cores = _detect_physical_cores()

    return Hardware(
        arch="arm64" if is_apple_silicon else "x86_64",
        is_apple_silicon=is_apple_silicon,
        under_rosetta=under_rosetta,
        cpu_brand=_detect_cpu_brand(is_apple_silicon),
        physical_cores=physical_cores,
        ram_gb=_detect_ram_gb(),
        macos_version=_detect_macos_version(),
        tier=_classify_tier(is_apple_silicon, physical_cores),
    )


def recommend_model(hw: Hardware) -> str:
    """Pick the largest Whisper model this machine can run without embarrassment.

    base.en is the default on BOTH Intel tiers, and that is deliberate in both
    directions:

      - We never downgrade a slow machine to tiny.en. tiny.en buys only ~200ms
        on the floor machine (fixed pipeline overhead swamps the model savings)
        and pays for it with real transcription errors. Bad trade.
      - We never offer small.en to a slow machine. Measured at ~8.7s for 11s of
        audio plus a ~20s cold load -- a 30-second dictation would take 20-25s
        to come back. That is not a product.

    An unrecognized tier is treated as slow, which is the safe direction.
    """
    if hw.tier == "fast":
        return "small.en"
    return "base.en"


def recommend_threads(hw: Hardware) -> int:
    """Thread count for the ASR engine: PHYSICAL cores, never logical.

    Oversubscribing hyperthreads is actively harmful here. Measured on the floor
    machine (2 physical / 4 logical), base.en:

        threads=2   3s speech -> 1.35s    11s speech ->  1.80s
        threads=4   3s speech -> 3.86s    11s speech -> 10.74s   (medians)

    Hyperthread siblings contend for the same AVX2/FMA execution ports. The
    damage lands hardest on tail latency, which is precisely what a user
    experiences as "this app is unreliable". Capped at _MAX_THREADS because
    faster-whisper stops scaling and starts contending with the audio callback
    well before high core counts.
    """
    return max(1, min(hw.physical_cores, _MAX_THREADS))
