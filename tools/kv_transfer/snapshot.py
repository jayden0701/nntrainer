"""Portable layer-major KV snapshots produced by the CausalLM CPU bridge."""

import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np


F32 = np.dtype("<f4")
PLANES = {"keys": "keys.f32", "values": "values.f32"}


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def read_json(path):
    with path.open(encoding="utf-8") as file:
        result = json.load(file)
    if not isinstance(result, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


@contextmanager
def new_directory(path):
    """Publish a complete artifact, and discard staging on ordinary failure."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"destination already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    try:
        yield staging
        if path.exists():
            raise FileExistsError(f"destination already exists: {path}")
        staging.rename(path)
    finally:
        if staging.exists():
            rmtree(staging)


def open_plane(path, shape, mode="r"):
    size = math.prod(shape) * F32.itemsize
    if mode == "r" and path.stat().st_size != size:
        raise ValueError(f"{path} must hold exactly {size} bytes")
    return np.memmap(path, dtype=F32, mode=mode, shape=shape)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    layers: int
    kv_heads: int
    head_dim: int
    theta: float
    rotary_dim: int

    @classmethod
    def parse(cls, obj):
        model_id = obj["model_id"]
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a nonempty string")
        layers = positive_int(obj["layers"], "layers")
        kv_heads = positive_int(obj["kv_heads"], "kv_heads")
        head_dim = positive_int(obj["head_dim"], "head_dim")
        if obj["key_space"] != "post_rope":
            raise ValueError("snapshot keys must be in post_rope space")
        rope = obj["rope"]
        if rope["layout"] != "split_half":
            raise ValueError("only unscaled split_half RoPE is supported")
        theta = rope["theta"]
        if type(theta) not in (float, int) or not math.isfinite(theta) or theta <= 0:
            raise ValueError("RoPE theta must be finite and positive")
        rotary_dim = positive_int(rope["rotary_dim"], "rotary_dim")
        if rotary_dim % 2 or rotary_dim > head_dim:
            raise ValueError("rotary_dim must be even and at most head_dim")
        return cls(model_id, layers, kv_heads, head_dim, float(theta), rotary_dim)

    def metadata(self):
        return {
            "model_id": self.model_id, "layers": self.layers,
            "kv_heads": self.kv_heads, "head_dim": self.head_dim,
            "key_space": "post_rope",
            "rope": {"layout": "split_half", "theta": self.theta,
                     "rotary_dim": self.rotary_dim},
        }

    def shape(self, tokens):
        return self.layers, tokens, self.kv_heads, self.head_dim


class Snapshot:
    def __init__(self, path):
        self.path = Path(path)
        meta = read_json(self.path / "metadata.json")
        if type(meta.get("version")) is not int or meta["version"] != 1:
            raise ValueError(f"{self.path}: unsupported snapshot version")
        self.spec = ModelSpec.parse(meta)
        self.tokens = positive_int(meta["tokens"], "tokens")
        self.token_ids = self._read_ids(meta, "token_ids")
        self.positions = self._read_ids(meta, "positions")
        if any(a >= b for a, b in zip(self.positions, self.positions[1:])):
            raise ValueError(f"{self.path}: positions must be strictly increasing")
        expected = math.prod(self.spec.shape(self.tokens)) * F32.itemsize
        for filename in PLANES.values():
            plane_path = self.path / filename
            if plane_path.stat().st_size != expected:
                raise ValueError(f"{plane_path} must hold exactly {expected} bytes")

    def plane(self, channel):
        return open_plane(self.path / PLANES[channel], self.spec.shape(self.tokens))

    def _read_ids(self, meta, key):
        values = meta[key]
        if (not isinstance(values, list) or len(values) != self.tokens
                or any(type(value) is not int or value < 0 or value > 2**53 - 1
                       for value in values)):
            raise ValueError(f"{self.path}: invalid {key} for {self.tokens} tokens")
        return tuple(values)


@contextmanager
def create_snapshot(path, spec, token_ids, positions):
    tokens = len(token_ids)
    with new_directory(path) as staging:
        planes = {
            key: open_plane(staging / filename, spec.shape(tokens), "w+")
            for key, filename in PLANES.items()
        }
        try:
            yield planes
            for plane in planes.values():
                plane.flush()
        finally:
            # Release file handles before publishing the directory on Windows.
            for plane in planes.values():
                plane._mmap.close()
        write_json(staging / "metadata.json", {
            "version": 1, **spec.metadata(), "tokens": tokens,
            "token_ids": list(token_ids), "positions": list(positions),
        })


def rotate_keys(keys, positions, spec, *, inverse=False):
    """Rotate the first rotary_dim coordinates of rows shaped [T, ..., D]."""
    keys = np.asarray(keys)
    if keys.shape[0] != len(positions) or keys.shape[-1] != spec.head_dim:
        raise ValueError("key shape does not match positions and head dimension")
    half = spec.rotary_dim // 2
    freq = np.power(spec.theta, -np.arange(half, dtype=np.float64) / half)
    angles = np.multiply.outer(np.asarray(positions, dtype=np.float64), freq)
    if inverse:
        angles *= -1
    trig_shape = (len(positions),) + (1,) * (keys.ndim - 2) + (half,)
    cos = np.cos(angles).astype(keys.dtype).reshape(trig_shape)
    sin = np.sin(angles).astype(keys.dtype).reshape(trig_shape)
    first, second = keys[..., :half], keys[..., half:2 * half]
    result = keys.copy()
    result[..., :half] = first * cos - second * sin
    result[..., half:2 * half] = second * cos + first * sin
    return result
