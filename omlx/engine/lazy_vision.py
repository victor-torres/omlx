"""Keep a VLM's vision tower out of memory until an image arrives.

Once the model is loaded and every post-load transform has run, the tower's
arrays are written to a private snapshot and swapped for one-element
placeholders of the same dtype, so code that reads the tower's dtype keeps
working. The first call to the tower loads the snapshot back; the engine
releases it again after it has been idle for a while. The snapshot holds the
exact arrays the eager engine would use, quantization and recasts included.

The tower instance gets a subclass whose ``__call__`` loads first. Model code
calls the tower in several places (mlx-vlm's ``get_input_embeddings``,
``encode_image``, the engine's own feature path), and all of them go
through ``__call__``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

logger = logging.getLogger(__name__)


class LazyVisionTower:
    def __init__(self, tower: Any, snapshot: Path, idle_seconds: float):
        self.tower = tower
        self.snapshot = Path(snapshot)
        self.idle_seconds = float(idle_seconds)
        # Towers run on the global and per-engine MLX executors; the lock keeps a
        # release from landing while another thread is swapping arrays in.
        self._lock = threading.RLock()
        self._placeholders: list[tuple[str, mx.array]] = []
        self.loaded = True
        self.last_used = 0.0
        self.nbytes = 0
        self.load_count = 0

    def strip(self) -> None:
        """Move the tower's arrays to the snapshot. Runs on the MLX executor."""
        params = tree_flatten(self.tower.parameters())
        self.snapshot.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(self.snapshot), dict(params))
        self._placeholders = [(k, mx.zeros((1,), dtype=v.dtype)) for k, v in params]
        # Realized here, so no other thread meets an array of this stream.
        mx.eval([v for _, v in self._placeholders])
        self.nbytes = sum(v.nbytes for _, v in params)
        del params
        self.release()
        base = type(self.tower)
        lazy = self

        def call(tower, *args, **kwargs):
            return lazy.call(base.__call__, tower, *args, **kwargs)

        self.tower.__class__ = type(
            f"Lazy{base.__name__}", (base,), {"__call__": call}
        )
        logger.info(
            "Lazy vision: %d tensors (%.2f GiB) moved to %s",
            len(self._placeholders),
            self.nbytes / 2**30,
            self.snapshot,
        )

    def release(self) -> None:
        with self._lock:
            self.tower.update(tree_unflatten(self._placeholders))
            self.loaded = False
            # oMLX raises the buffer cache limit to most of RAM; without this
            # the freed buffers stay pooled and the memory is not returned.
            mx.clear_cache()

    def ensure_loaded(self) -> None:
        with self._lock:
            self.last_used = time.monotonic()
            if self.loaded:
                return
            start = time.monotonic()
            weights = mx.load(str(self.snapshot))
            mx.eval(list(weights.values()))
            self.tower.update(tree_unflatten(list(weights.items())))
            self.loaded = True
            self.load_count += 1
            logger.info("Lazy vision: tower loaded in %.2f s", time.monotonic() - start)

    def call(self, base_call, tower, *args, **kwargs):
        with self._lock:
            self.ensure_loaded()
            return base_call(tower, *args, **kwargs)

    def release_if_idle(self) -> bool:
        """Release the tower once it has been idle long enough. MLX executor."""
        with self._lock:
            if not self.loaded or time.monotonic() - self.last_used < self.idle_seconds:
                return False
            self.release()
        logger.info("Lazy vision: tower released after %.0f s idle", self.idle_seconds)
        return True

    def close(self) -> None:
        try:
            os.unlink(self.snapshot)
        except FileNotFoundError:
            pass
