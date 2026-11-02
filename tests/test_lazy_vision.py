"""Lazy vision tower: snapshot, placeholders, reload on call, idle release."""

import time

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.engine.lazy_vision import LazyVisionTower
from omlx.model_settings import ModelSettings


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 16)


class Tower(nn.Module):
    spatial_merge_size = 2

    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.blocks = [nn.Linear(16, 16) for _ in range(2)]

    def __call__(self, x, scale=1.0):
        x = self.patch_embed.proj(x)
        for block in self.blocks:
            x = block(x)
        return x * scale


@pytest.fixture
def tower():
    mx.random.seed(0)
    t = Tower()
    t.set_dtype(mx.bfloat16)
    mx.eval(t.parameters())
    return t


def _param_bytes(module):
    from mlx.utils import tree_flatten

    return sum(v.nbytes for _, v in tree_flatten(module.parameters()))


def test_strip_then_call_matches_eager(tower, tmp_path):
    x = mx.random.normal((3, 8)).astype(mx.bfloat16)
    expected = tower(x, scale=2.0)
    full = _param_bytes(tower)

    lazy = LazyVisionTower(tower, tmp_path / "tower.safetensors", idle_seconds=60)
    lazy.strip()

    assert not lazy.loaded
    assert (tmp_path / "tower.safetensors").exists()
    assert lazy.nbytes == full
    assert _param_bytes(tower) < full // 10
    # Code that reads the tower's dtype or plain attributes still works.
    assert tower.patch_embed.proj.weight.dtype == mx.bfloat16
    assert tower.spatial_merge_size == 2
    assert isinstance(tower, Tower)

    out = tower(x, scale=2.0)
    assert lazy.loaded and lazy.load_count == 1
    assert mx.array_equal(out, expected).item()
    assert _param_bytes(tower) == full


def test_release_only_after_idle(tower, tmp_path):
    lazy = LazyVisionTower(tower, tmp_path / "tower.safetensors", idle_seconds=60)
    lazy.strip()
    tower(mx.zeros((1, 8), dtype=mx.bfloat16))

    assert not lazy.release_if_idle()
    assert lazy.loaded

    lazy.last_used = time.monotonic() - 61
    assert lazy.release_if_idle()
    assert not lazy.loaded
    assert not lazy.release_if_idle()

    tower(mx.zeros((1, 8), dtype=mx.bfloat16))
    assert lazy.load_count == 2


def test_close_removes_snapshot(tower, tmp_path):
    lazy = LazyVisionTower(tower, tmp_path / "sub" / "tower.safetensors", 60)
    lazy.strip()
    lazy.close()
    assert not (tmp_path / "sub" / "tower.safetensors").exists()
    lazy.close()


def test_failed_snapshot_leaves_tower_eager(tower, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    lazy = LazyVisionTower(tower, blocker / "tower.safetensors", 60)
    with pytest.raises(Exception):
        lazy.strip()
    assert type(tower) is Tower
    assert tower.patch_embed.proj.weight.shape == (16, 8)


def test_settings_defaults_and_validation():
    settings = ModelSettings()
    assert settings.lazy_vision_enabled is False
    assert settings.lazy_vision_idle_seconds == 60
    ModelSettings(lazy_vision_enabled=True, lazy_vision_idle_seconds=5)
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            ModelSettings(lazy_vision_enabled=True, lazy_vision_idle_seconds=bad)
