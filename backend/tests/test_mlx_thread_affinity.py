"""MLX calls must all run on one dedicated OS thread.

MLX's Metal streams are thread-local (issue #699 / #675): a model loaded on
one executor thread and used from another raises
``There is no Stream(gpu, N) in current thread``. These tests pin the
contract without needing MLX itself — the backends are exercised with fake
models that merely record which thread touched them.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from backend.backends import mlx_thread
from backend.backends.mlx_backend import MLXSTTBackend, MLXTTSBackend
from backend.backends.qwen_llm_backend import MLXQwenLLMBackend


def _worker_ident() -> int:
    """Thread ident of the MLX worker (forces the thread to start)."""
    return mlx_thread.submit(threading.get_ident).result()


class _FakeTTSModel:
    def __init__(self, seen: list[int]):
        self.seen = seen

    def generate(self, text, ref_audio=None, ref_text="", lang_code="auto"):
        self.seen.append(threading.get_ident())
        yield SimpleNamespace(audio=np.zeros(24, dtype=np.float32), sample_rate=24000)


class _FakeSTTModel:
    def __init__(self, seen: list[int]):
        self.seen = seen

    def generate(self, audio_path, **decode_options):
        self.seen.append(threading.get_ident())
        return {"text": " hello "}


# --- helpers ----------------------------------------------------------------


def test_worker_thread_is_named_and_stable():
    first = _worker_ident()
    second = _worker_ident()
    assert first == second

    name = mlx_thread.submit(lambda: threading.current_thread().name).result()
    assert name.startswith("mlx-worker")


def test_run_sync_is_reentrant_from_worker():
    # Calling run_sync from inside the worker must execute inline instead of
    # waiting on the single-worker executor (which would deadlock).
    def inner():
        return threading.get_ident()

    def outer():
        assert mlx_thread.is_on_mlx_thread()
        return mlx_thread.run_sync(inner)

    fut = mlx_thread.submit(outer)
    assert fut.result(timeout=5) == _worker_ident()


async def test_run_on_mlx_thread_awaits_result():
    ident = await mlx_thread.run_on_mlx_thread(threading.get_ident)
    assert ident == _worker_ident()


# --- TTS backend -------------------------------------------------------------


async def test_tts_load_generate_unload_share_one_thread(monkeypatch):
    seen: list[int] = []
    backend = MLXTTSBackend(model_size="0.6B")

    def fake_load(self, model_size):
        seen.append(threading.get_ident())
        self.model = _FakeTTSModel(seen)
        self._current_model_size = model_size
        self.model_size = model_size

    monkeypatch.setattr(MLXTTSBackend, "_load_model_sync", fake_load)

    await backend.load_model_async("0.6B")
    audio, sr = await backend.generate("hi", {"ref_audio": None, "ref_text": ""}, "en")
    assert sr == 24000
    assert audio.shape == (24,)

    # Switching sizes must unload + reload on the same thread, in one shot.
    await backend.load_model_async("1.7B")
    assert backend._current_model_size == "1.7B"
    await backend.generate("again", {"ref_audio": None, "ref_text": ""}, "en")

    # Unload from the event-loop thread is routed to the worker as well.
    backend.unload_model()
    assert backend.model is None
    assert backend._current_model_size is None

    worker = _worker_ident()
    assert seen, "fake model was never touched"
    assert set(seen) == {worker}
    assert threading.get_ident() != worker


async def test_tts_generate_lazily_loads_on_worker(monkeypatch):
    seen: list[int] = []

    def fake_load(self, model_size):
        seen.append(threading.get_ident())
        self.model = _FakeTTSModel(seen)
        self._current_model_size = model_size
        self.model_size = model_size

    monkeypatch.setattr(MLXTTSBackend, "_load_model_sync", fake_load)
    backend = MLXTTSBackend(model_size="0.6B")

    # No explicit load: generate() must load inside the same worker submission.
    await backend.generate("lazy", {"ref_audio": None, "ref_text": ""}, "en")
    assert backend.is_loaded()
    assert set(seen) == {_worker_ident()}


async def test_tts_concurrent_generates_serialize(monkeypatch):
    """Two overlapping generate() calls never interleave load/inference."""
    active = 0
    max_active = 0
    lock = threading.Lock()

    class _SlowModel(_FakeTTSModel):
        def generate(self, *a, **kw):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                yield from super().generate(*a, **kw)
            finally:
                with lock:
                    active -= 1

    seen: list[int] = []

    def fake_load(self, model_size):
        self.model = _SlowModel(seen)
        self._current_model_size = model_size
        self.model_size = model_size

    monkeypatch.setattr(MLXTTSBackend, "_load_model_sync", fake_load)
    backend = MLXTTSBackend(model_size="0.6B")
    prompt = {"ref_audio": None, "ref_text": ""}
    await asyncio.gather(*(backend.generate(f"t{i}", prompt, "en") for i in range(4)))
    assert max_active == 1
    assert set(seen) == {_worker_ident()}


# --- STT backend -------------------------------------------------------------


async def test_stt_transcribe_runs_on_worker(monkeypatch):
    seen: list[int] = []

    def fake_load(self, model_size):
        seen.append(threading.get_ident())
        self.model = _FakeSTTModel(seen)
        self.model_size = model_size

    monkeypatch.setattr(MLXSTTBackend, "_load_model_sync", fake_load)
    backend = MLXSTTBackend(model_size="base")

    text = await backend.transcribe("/tmp/does-not-matter.wav", language="de")
    assert text == "hello"

    # Size override triggers a reload — still on the worker.
    await backend.transcribe("/tmp/x.wav", model_size="turbo")
    assert backend.model_size == "turbo"

    backend.unload_model()
    assert backend.model is None
    assert set(seen) == {_worker_ident()}


# --- LLM backend -------------------------------------------------------------


async def test_llm_load_generate_unload_share_one_thread(monkeypatch):
    seen: list[int] = []

    def fake_load(self, model_size):
        seen.append(threading.get_ident())
        self.model = object()
        self.tokenizer = object()
        self._current_model_size = model_size
        self.model_size = model_size

    def fake_generate(self, prompt, system, max_tokens, temperature, examples=None):
        seen.append(threading.get_ident())
        return f"echo:{prompt}"

    monkeypatch.setattr(MLXQwenLLMBackend, "_load_model_sync", fake_load)
    monkeypatch.setattr(MLXQwenLLMBackend, "_generate_sync", fake_generate)

    backend = MLXQwenLLMBackend(model_size="0.6B")
    assert await backend.generate("ping") == "echo:ping"
    assert await backend.generate("pong", model_size="1.7B") == "echo:pong"
    assert backend._current_model_size == "1.7B"

    backend.unload_model()
    assert backend.model is None

    assert set(seen) == {_worker_ident()}


@pytest.mark.parametrize("cls", [MLXTTSBackend, MLXSTTBackend, MLXQwenLLMBackend])
def test_unload_when_nothing_loaded_is_a_noop(cls):
    backend = cls()
    backend.unload_model()  # must not raise or deadlock
    assert backend.model is None
