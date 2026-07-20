"""Tests for gain normalization.

The behaviour under test is worth stating plainly, because it is the fix for a
measured 26.5% word error rate: quiet capture gets scaled up before it reaches
the model, but near-silence does NOT, and the gain is bounded.

These tests deliberately avoid touching a microphone or a stream. They exercise
the pure function only, so they run anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from blurt.audio import (
    _MAX_GAIN,
    _NORMALIZE_FLOOR_PEAK,
    _TARGET_PEAK,
    _normalize_gain,
)


def _tone(peak: float, samples: int = 1600) -> np.ndarray:
    """A sine at a specified peak amplitude."""
    t = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    wave = np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
    return (wave * peak).astype(np.float32)


def test_quiet_audio_is_amplified_toward_the_target_peak():
    quiet = _tone(0.01)
    out = _normalize_gain(quiet)
    assert float(np.max(np.abs(out))) == pytest.approx(_TARGET_PEAK, abs=0.02)


def test_the_measured_failure_level_gets_amplified():
    # rms 0.0002 is the level at which base.en dropped the first seven words of
    # a test utterance. Speech at that rms has a peak roughly 10x higher, so
    # ~0.002 is the realistic peak to test -- NOT 0.0002, which would be below
    # the silence floor and correctly left alone.
    quiet = _tone(0.002)
    out = _normalize_gain(quiet)
    assert float(np.max(np.abs(out))) > float(np.max(np.abs(quiet))) * 10


def test_healthy_audio_is_left_alone():
    healthy = _tone(0.95)
    out = _normalize_gain(healthy)
    assert np.allclose(out, healthy)


def test_loud_audio_is_not_attenuated():
    # Normalization only ever raises quiet audio; it must never turn down a
    # signal that is already loud, which would be a surprising side effect.
    # Compared against the input's own peak rather than the literal 0.99,
    # because float32(0.99) is 0.98999953 and a bare >= 0.99 fails on precision.
    loud = _tone(0.99)
    out = _normalize_gain(loud)
    assert float(np.max(np.abs(out))) >= float(np.max(np.abs(loud)))


def test_near_silence_is_NOT_amplified():
    # The important guard. Amplifying a room-tone-only buffer to full scale
    # invites the model to hallucinate speech out of noise.
    almost_nothing = _tone(_NORMALIZE_FLOOR_PEAK / 2.0)
    out = _normalize_gain(almost_nothing)
    assert np.allclose(out, almost_nothing)


def test_digital_silence_is_untouched_and_does_not_divide_by_zero():
    silence = np.zeros(1600, dtype=np.float32)
    out = _normalize_gain(silence)
    assert np.allclose(out, 0.0)
    assert np.isfinite(out).all()


def test_gain_is_capped():
    # Just above the floor so it IS normalized, but quiet enough that an
    # uncapped gain would be enormous.
    tiny = _tone(_NORMALIZE_FLOOR_PEAK * 1.01)
    out = _normalize_gain(tiny)
    applied = float(np.max(np.abs(out))) / float(np.max(np.abs(tiny)))
    assert applied <= _MAX_GAIN + 1e-6


def test_output_never_clips():
    for peak in (0.0006, 0.01, 0.1, 0.5, 0.9, 0.95, 1.0):
        out = _normalize_gain(_tone(peak))
        assert float(np.max(np.abs(out))) <= 1.0 + 1e-6, "clipped at peak=%s" % peak


def test_empty_input_is_safe():
    out = _normalize_gain(np.zeros(0, dtype=np.float32))
    assert out.size == 0


def test_dtype_stays_float32():
    # The engine is handed this array directly; a float64 buffer would either be
    # rejected or silently converted on every utterance.
    out = _normalize_gain(_tone(0.01))
    assert out.dtype == np.float32


def test_waveform_shape_is_preserved():
    # Normalization must be a pure scalar multiply -- same length, same zero
    # crossings, no filtering. Anything else is a bug.
    quiet = _tone(0.01)
    out = _normalize_gain(quiet)
    assert out.shape == quiet.shape
    assert np.array_equal(np.sign(out), np.sign(quiet))
