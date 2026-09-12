"""Engine tuning knobs (Chatterbox exaggeration / cfg_weight / temperature).

Covers the plumbing end to end without loading any model:
request validation -> engine_params() -> chunked generation forwarding
(only to backends that declare the knobs) -> persistence helpers.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from pydantic import ValidationError

from backend import models
from backend.backends.chatterbox_backend import ChatterboxTTSBackend
from backend.backends.chatterbox_turbo_backend import ChatterboxTurboTTSBackend
from backend.services.history import _dump_engine_params, load_engine_params
from backend.utils.chunked_tts import accepted_engine_params, generate_chunked

# --- request models ----------------------------------------------------------


def test_generation_request_defaults_leave_params_unset():
    req = models.GenerationRequest(profile_id="p", text="hi")
    assert req.exaggeration is None
    assert req.engine_params() is None


def test_generation_request_collects_only_explicit_params():
    req = models.GenerationRequest(profile_id="p", text="hi", exaggeration=0.8, temperature=0.6)
    assert req.engine_params() == {"exaggeration": 0.8, "temperature": 0.6}


@pytest.mark.parametrize(
    ("field", "value"),
    [("exaggeration", 1.5), ("exaggeration", -0.1), ("cfg_weight", 2.0), ("temperature", 0.0), ("temperature", 5.0)],
)
def test_generation_request_rejects_out_of_range(field, value):
    with pytest.raises(ValidationError):
        models.GenerationRequest(profile_id="p", text="hi", **{field: value})


def test_speak_request_accepts_params():
    req = models.SpeakRequest(text="hi", cfg_weight=0.3)
    assert req.cfg_weight == 0.3


def test_generation_response_parses_stored_json():
    resp = models.GenerationResponse(
        id="g",
        profile_id="p",
        text="t",
        language="de",
        created_at="2026-01-01T00:00:00",
        engine_params='{"exaggeration": 0.7}',
    )
    assert resp.engine_params == {"exaggeration": 0.7}

    resp = models.GenerationResponse(
        id="g", profile_id="p", text="t", language="de", created_at="2026-01-01T00:00:00", engine_params="not json"
    )
    assert resp.engine_params is None


# --- persistence helpers ------------------------------------------------------


def test_dump_and_load_roundtrip():
    assert _dump_engine_params(None) is None
    assert _dump_engine_params({}) is None
    raw = _dump_engine_params({"cfg_weight": 0.25})
    assert load_engine_params(raw) == {"cfg_weight": 0.25}
    assert load_engine_params(None) is None
    assert load_engine_params("{}") is None
    assert load_engine_params("garbage") is None


# --- forwarding ---------------------------------------------------------------


class _KnobBackend:
    """Backend whose generate() declares the Chatterbox knobs."""

    def __init__(self):
        self.calls: list[dict] = []

    async def generate(
        self, text, voice_prompt, language="en", seed=None, instruct=None, exaggeration=None, cfg_weight=None
    ):
        self.calls.append({"exaggeration": exaggeration, "cfg_weight": cfg_weight})
        return np.zeros(240, dtype=np.float32), 24000


class _PlainBackend:
    """Backend without any knobs (e.g. Kokoro/Qwen)."""

    def __init__(self):
        self.calls = 0

    async def generate(self, text, voice_prompt, language="en", seed=None, instruct=None):
        self.calls += 1
        return np.zeros(240, dtype=np.float32), 24000


def test_accepted_engine_params_filters_by_signature():
    params = {"exaggeration": 0.9, "cfg_weight": 0.2, "temperature": 0.5, "bogus": 1}
    assert accepted_engine_params(_KnobBackend(), params) == {"exaggeration": 0.9, "cfg_weight": 0.2}
    assert accepted_engine_params(_PlainBackend(), params) == {}
    assert accepted_engine_params(_PlainBackend(), None) == {}
    assert accepted_engine_params(_KnobBackend(), {"exaggeration": None}) == {}


async def test_generate_chunked_forwards_knobs_to_every_chunk():
    backend = _KnobBackend()
    text = "Erster Satz. " * 20  # long enough to be split into several chunks
    await generate_chunked(
        backend,
        text,
        {"ref_audio": None},
        language="de",
        max_chunk_chars=100,
        engine_params={"exaggeration": 0.9, "cfg_weight": 0.2, "temperature": 0.5},
    )
    assert len(backend.calls) > 1
    assert all(c == {"exaggeration": 0.9, "cfg_weight": 0.2} for c in backend.calls)


async def test_generate_chunked_ignores_knobs_for_plain_backend():
    backend = _PlainBackend()
    await generate_chunked(backend, "kurz", {"ref_audio": None}, engine_params={"exaggeration": 0.9})
    assert backend.calls == 1


# --- real backends declare the knobs ----------------------------------------


def test_chatterbox_backends_declare_knobs():
    for cls in (ChatterboxTTSBackend, ChatterboxTurboTTSBackend):
        params = inspect.signature(cls.generate).parameters
        assert {"exaggeration", "cfg_weight", "temperature"} <= set(params), cls.__name__
