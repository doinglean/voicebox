"""Single dedicated worker thread for every MLX call.

MLX's Metal backend keeps a *per-thread* stream registry: the GPU stream a
model was loaded on is only visible from the OS thread that created it.
``asyncio.to_thread`` hands work to the event loop's default executor, which
round-robins across several worker threads, so a model loaded on thread A
and later used from thread B raises::

    RuntimeError: There is no Stream(gpu, N) in current thread.

(see https://github.com/jamiepine/voicebox/issues/699 and #675). Every MLX
touch — model load, unload, TTS generate, Whisper transcribe, mlx-lm
generate — therefore has to run on the *same* thread for the whole process
lifetime. This module owns that thread.

Usage from async code::

    audio = await run_on_mlx_thread(self._generate_sync, text)

Usage from sync code (e.g. ``unload_model()`` called from a route)::

    run_sync(self._unload_model_sync)

``run_sync`` is re-entrant: when it is already executing on the MLX worker
it calls the function inline instead of deadlocking the single-worker pool.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

_worker_ident: int | None = None


def _remember_worker_ident() -> None:
    global _worker_ident
    _worker_ident = threading.get_ident()


_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="mlx-worker",
    initializer=_remember_worker_ident,
)


def is_on_mlx_thread() -> bool:
    """True when called from the dedicated MLX worker thread."""
    return _worker_ident is not None and threading.get_ident() == _worker_ident


def submit[T](func: Callable[..., T], *args: Any, **kwargs: Any) -> Future[T]:
    """Schedule ``func(*args, **kwargs)`` on the MLX worker; returns a Future."""
    return _executor.submit(func, *args, **kwargs)


async def run_on_mlx_thread[T](func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Await ``func(*args, **kwargs)`` executed on the MLX worker thread."""
    loop = asyncio.get_running_loop()
    return await asyncio.wrap_future(submit(func, *args, **kwargs), loop=loop)


def run_sync[T](func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run ``func`` on the MLX worker and block until it returns.

    Safe to call from any thread, including the worker itself (in which case
    the call is made inline — submitting to a single-worker executor from
    its own thread would wait on itself forever).
    """
    if is_on_mlx_thread():
        return func(*args, **kwargs)
    return submit(func, *args, **kwargs).result()
