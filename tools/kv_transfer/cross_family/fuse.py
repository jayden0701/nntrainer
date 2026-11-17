"""Small CPU adapter and portable KV-cache application for the cross-family PoC."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..snapshot import ModelSpec, Snapshot, create_snapshot


SOURCE = ModelSpec("Qwen/Qwen3-0.6B", 28, 8, 128, 1000000.0, 128)
RECEIVER = ModelSpec("HuggingFaceTB/SmolLM2-135M", 30, 3, 64, 100000.0, 64)


class StreamFuser(nn.Module):
    def __init__(self, width):
        super().__init__()
        source_dim = SOURCE.kv_heads * SOURCE.head_dim
        receiver_dim = RECEIVER.kv_heads * RECEIVER.head_dim
        self.project = nn.Linear(source_dim + receiver_dim, width)
        self.residual = nn.Linear(width, receiver_dim)
        self.weight = nn.Linear(width, RECEIVER.kv_heads)
        self.gate_logit = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, source, receiver):
        # Preserve positions when moving [B,H,T,D] to flattened [B,T,H*D].
        t = receiver.transpose(1, 2)
        s = source.transpose(1, 2)
        sf = F.layer_norm(s.flatten(-2), (s.shape[-2] * s.shape[-1],))
        tf = F.layer_norm(t.flatten(-2), (t.shape[-2] * t.shape[-1],))
        h = F.silu(self.project(torch.cat((sf, tf), dim=-1)))
        delta = self.residual(h).reshape_as(t)
        head_weight = self.weight(h).sigmoid().unsqueeze(-1)
        # Scale the normalized branch by the receiver's per-token/head RMS.
        scale = t.square().mean(dim=-1, keepdim=True).add(1e-8).sqrt()
        return (t + scale * self.gate_logit.sigmoid() * head_weight * delta).transpose(1, 2)


class CacheFuser(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.ModuleList((StreamFuser(width), StreamFuser(width)))
            for _ in range(SOURCE.layers)
        )

    def forward(self, source, receiver):
        if len(source) != SOURCE.layers or len(receiver) != RECEIVER.layers:
            raise ValueError("unexpected cache layer counts")
        offset = len(receiver) - len(source)
        return receiver[:offset] + tuple(
            tuple(fuse(s, t) for fuse, s, t in zip(layer, source_kv, receiver_kv))
            for layer, source_kv, receiver_kv in zip(
                self.layers, source, receiver[offset:]
            )
        )


def read_cache(snapshot, spec):
    if snapshot.spec != spec:
        raise ValueError(f"{snapshot.path}: expected {spec}, got {snapshot.spec}")
    keys = snapshot.plane("keys")
    values = snapshot.plane("values")
    return tuple(
        (
            torch.from_numpy(np.array(key.transpose(1, 0, 2)[None])),
            torch.from_numpy(np.array(value.transpose(1, 0, 2)[None])),
        )
        for key, value in zip(keys, values)
    )


def write_cache(path, cache, spec, ids, positions):
    with create_snapshot(path, spec, ids, positions) as planes:
        for index, (keys, values) in enumerate(cache):
            for name, tensor in (("keys", keys), ("values", values)):
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"nonfinite {name} in layer {index}")
                planes[name][index] = tensor[0].transpose(0, 1).detach().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--receiver", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["version"] != 1:
        raise ValueError("unsupported fuser checkpoint version")
    model = CacheFuser(checkpoint["width"])
    model.load_state_dict(checkpoint["state"])
    model.eval()

    source = Snapshot(args.source)
    receiver = Snapshot(args.receiver)
    if source.positions != receiver.positions:
        raise ValueError("source IDs must be aligned to the receiver's positions")
    positions = receiver.positions
    if positions != tuple(range(receiver.tokens)):
        raise ValueError("this adapter was trained on zero-based consecutive positions")

    with torch.no_grad():
        fused = model(read_cache(source, SOURCE), read_cache(receiver, RECEIVER))
        write_cache(args.output, fused, receiver.spec, receiver.token_ids, positions)


if __name__ == "__main__":
    main()
