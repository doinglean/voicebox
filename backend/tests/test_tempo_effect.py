"""Pitch-preserving tempo effect and the `speed` request sugar."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from backend import models
from backend.utils.effects import (
    EFFECT_REGISTRY,
    apply_effects,
    build_pedalboard,
    get_available_effects,
    validate_effects_chain,
)

SR = 24000


def _tone(seconds: float = 1.0, hz: float = 220.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False, dtype=np.float32)
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_tempo_is_advertised_to_the_ui():
    listed = {e["type"]: e for e in get_available_effects()}
    assert "tempo" in listed
    speed = listed["tempo"]["params"]["speed"]
    assert speed["default"] == 1.0
    assert speed["min"] < 1.0 < speed["max"]


def test_tempo_validates_like_any_effect():
    assert validate_effects_chain([{"type": "tempo", "enabled": True, "params": {"speed": 0.85}}]) is None
    assert validate_effects_chain([{"type": "tempo", "params": {"speed": 3.0}}]) is not None
    assert validate_effects_chain([{"type": "tempo", "params": {"bogus": 1}}]) is not None


@pytest.mark.parametrize(("speed", "expected_ratio"), [(0.8, 1.25), (1.0, 1.0), (1.25, 0.8)])
def test_tempo_changes_length_not_shape(speed, expected_ratio):
    audio = _tone(1.0)
    out = apply_effects(audio, SR, [{"type": "tempo", "params": {"speed": speed}}])
    assert out.ndim == 1
    assert out.dtype == np.float32
    assert len(out) / len(audio) == pytest.approx(expected_ratio, rel=0.02)


def test_tempo_disabled_is_a_noop():
    audio = _tone(0.5)
    out = apply_effects(audio, SR, [{"type": "tempo", "enabled": False, "params": {"speed": 0.5}}])
    assert len(out) == len(audio)


def test_tempo_composes_with_plugin_effects_in_order():
    audio = _tone(1.0)
    chain = [
        {"type": "gain", "params": {"gain_db": -6.0}},
        {"type": "tempo", "params": {"speed": 0.5}},
        {"type": "gain", "params": {"gain_db": -6.0}},
    ]
    out = apply_effects(audio, SR, chain)
    assert len(out) / len(audio) == pytest.approx(2.0, rel=0.02)
    # Two -6 dB gains ≈ -12 dB overall (0.25 amplitude), regardless of the stretch in between.
    assert np.max(np.abs(out)) == pytest.approx(0.3 * 10 ** (-12 / 20), rel=0.1)


def test_build_pedalboard_skips_function_effects():
    board = build_pedalboard([{"type": "tempo", "params": {"speed": 0.8}}, {"type": "gain", "params": {}}])
    assert len(board) == 1
    assert "fn" in EFFECT_REGISTRY["tempo"]
    assert "cls" not in EFFECT_REGISTRY["tempo"]


# --- `speed` sugar on the request models --------------------------------------


def test_with_speed_appends_tempo_and_keeps_none():
    assert models.with_speed(None, None) is None
    assert models.with_speed(None, 1.0) is None
    chain = models.with_speed(None, 0.85)
    assert chain == [{"type": "tempo", "enabled": True, "params": {"speed": 0.85}}]

    existing = [{"type": "gain", "enabled": True, "params": {"gain_db": 2.0}}]
    merged = models.with_speed(existing, 1.2)
    assert merged[0] == existing[0]
    assert merged[1]["type"] == "tempo"
    assert existing == [{"type": "gain", "enabled": True, "params": {"gain_db": 2.0}}]  # not mutated


def test_speed_is_validated_on_requests():
    assert models.GenerationRequest(profile_id="p", text="hi", speed=0.9).speed == 0.9
    assert models.SpeakRequest(text="hi", speed=1.3).speed == 1.3
    with pytest.raises(ValidationError):
        models.GenerationRequest(profile_id="p", text="hi", speed=0.1)
    with pytest.raises(ValidationError):
        models.SpeakRequest(text="hi", speed=2.0)
