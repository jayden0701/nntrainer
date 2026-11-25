#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Calibrate compact Qwen3 value maps at the target's current query state."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmark import file_digest, verify_prefix
from dataset import read_pairs
from mapper import Mapping, predict
from snapshot import Snapshot, new_directory, read_json, write_json


def solve_correction(gram, cross, metric, budget):
    size = len(gram)
    ridge = max(1e-8, 0.01 * np.trace(gram) / size)
    if budget <= 0 or not np.any(cross):
        return np.zeros_like(cross), 0.0
    system = gram + ridge * np.eye(size)

    # Whiten M so all multiplier trials reduce to diagonal solves.
    factor = np.linalg.cholesky(metric)
    whitened = np.linalg.solve(factor, system)
    whitened = np.linalg.solve(factor, whitened.T).T
    eigenvalues, vectors = np.linalg.eigh((whitened + whitened.T) * 0.5)
    rhs = vectors.T @ np.linalg.solve(factor, cross)
    squared = np.sum(rhs ** 2, axis=1)

    def energy(multiplier):
        return float(np.sum(squared / (eigenvalues + multiplier) ** 2))

    multiplier = 0.0
    if energy(0.0) > budget:
        low, high = 0.0, max(1.0, float(np.sqrt(squared.sum() / budget)))
        for _ in range(56):
            mid = (low + high) / 2
            if energy(mid) > budget:
                low = mid
            else:
                high = mid
        multiplier = high
    update = np.linalg.solve(factor.T, vectors @ (rhs / (eigenvalues[:, None] + multiplier)))
    return update, energy(multiplier)


def value_features(snapshot, layers, positions):
    values = snapshot.plane("values")
    return np.stack([values[layer, positions] for layer in layers], axis=-2).reshape(
        -1, snapshot.spec.kv_heads, len(layers) * snapshot.spec.head_dim)


def with_intercept(features):
    return np.concatenate((features, np.ones((*features.shape[:-1], 1), dtype=features.dtype)), axis=-1)


def sampling_plan(pairs, count, seed, stride):
    offsets = np.cumsum([0, *((pair.source.tokens + stride - 1) // stride for pair in pairs)])
    selection = np.sort(np.random.default_rng(seed).choice(offsets[-1], min(count, offsets[-1]), replace=False))
    return [(pair, (selection[begin:end] - offsets[i]) * stride)
            for i, (pair, begin, end) in enumerate(zip(
                pairs, np.searchsorted(selection, offsets[:-1]),
                np.searchsorted(selection, offsets[1:]))) if begin != end]


def energy_metric(sampled, layers, layer, weights, bias):
    features = with_intercept(np.concatenate([
        value_features(pair.source, layers, positions) for pair, positions in sampled
    ])).transpose(1, 0, 2).astype(np.float64)
    expected = np.concatenate([
        pair.target.plane("values")[layer, positions] for pair, positions in sampled
    ]).transpose(1, 0, 2)
    coefficients = np.concatenate((weights, bias[:, None, :]), axis=1)
    residual = features @ coefficients - expected
    metric = features.transpose(0, 2, 1) @ features / features.shape[1]
    metric += 1e-4 * np.eye(features.shape[-1])
    return metric, 0.01 * np.mean(np.sum(residual ** 2, axis=-1), axis=1)


@dataclass
class Handoff:
    source: Snapshot
    target: Snapshot
    queries: list


def query_readout(layer, hidden, rotary, mapped_key, mapped_value, target_key, target_value):
    import torch
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

    attn = layer.self_attn
    normalized = layer.input_layernorm(hidden)
    shape = (*hidden.shape[:-1], -1, attn.head_dim)
    q = attn.q_norm(attn.q_proj(normalized).view(shape)).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(normalized).view(shape)).transpose(1, 2)
    v = attn.v_proj(normalized).view(shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *rotary)
    query_count = q.shape[-2]
    mask = torch.ones((query_count, query_count), dtype=torch.bool).triu(1)
    query_scores = (q @ repeat_kv(k, attn.num_key_value_groups).transpose(-2, -1)) * attn.scaling
    query_scores.masked_fill_(mask, float("-inf"))

    def read(history_key, history_value):
        history_scores = (q @ history_key.transpose(-2, -1)) * attn.scaling
        weights = torch.cat((history_scores, query_scores), dim=-1).softmax(dim=-1)
        history_weights, query_weights = weights.split((history_key.shape[-2], query_count), dim=-1)
        return history_weights, history_weights @ history_value + query_weights @ repeat_kv(v, attn.num_key_value_groups)

    weights, mapped = read(mapped_key, mapped_value)
    _, target = read(target_key, target_value)
    return weights[0], mapped[0], target[0] - mapped[0]


def history_planes(mapping, handoff, layer, groups):
    import torch
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv

    planes = []
    for index, name in enumerate(("keys", "values")):
        snapshot = handoff.source
        predicted = predict(snapshot, snapshot.plane(name), mapping.geometry, name,
                            slice(0, snapshot.tokens), layer, mapping.weights[index, layer],
                            mapping.bias[index, layer])
        expected = handoff.target.plane(name)[layer].copy()
        for values in (predicted, expected):
            planes.append(repeat_kv(torch.from_numpy(values).transpose(0, 1).unsqueeze(0), groups))
    return planes[0], planes[2], planes[1], planes[3]


def collect_layer(model_layer, mapping, handoffs, layer):
    import torch

    geometry = mapping.geometry
    groups = model_layer.self_attn.num_key_value_groups
    heads = geometry.target.kv_heads
    gram = np.zeros((heads, geometry.features + 1, geometry.features + 1))
    cross = np.zeros((heads, geometry.features + 1, geometry.outputs))
    traces = []
    for handoff in handoffs:
        features = with_intercept(value_features(handoff.source, geometry.layers[layer], slice(None)))
        inputs = torch.from_numpy(features.transpose(1, 0, 2).copy())
        kv = history_planes(mapping, handoff, layer, groups)
        for query in handoff.queries:
            weights, mapped, residual = query_readout(model_layer, query["hidden"], query["rotary"], *kv)
            pooled = (weights.reshape(heads, -1, weights.shape[-1]) @ inputs).double().numpy()
            errors = residual.reshape(heads, -1, geometry.outputs).double().numpy()
            normalized = 1.0 / (len(handoff.queries) * pooled.shape[1])
            gram += normalized * (pooled.transpose(0, 2, 1) @ pooled)
            cross += normalized * (pooled.transpose(0, 2, 1) @ errors)
            traces.append((query, pooled, mapped, errors, normalized))
    return gram, cross, traces


def calibrate(args):
    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    mapping = Mapping(args.initial)
    cfg = read_json(args.initial / "metadata.json")
    geometry = mapping.geometry
    if geometry.layout != "headwise":
        raise ValueError("Readout calibration requires a compact initial mapper")
    hf = AutoModelForCausalLM.from_pretrained(args.hf_target, dtype=torch.float32,
                                             local_files_only=True).eval()
    model = hf.model
    model_cfg = model.config
    spec = geometry.target
    if (model_cfg.model_type != "qwen3" or len(model.layers) != spec.layers
            or model_cfg.num_key_value_heads != spec.kv_heads or model_cfg.head_dim != spec.head_dim
            or spec.rotary_dim != spec.head_dim or model_cfg.rope_theta != spec.theta
            or model_cfg.rope_scaling is not None or any(layer.self_attn.sliding_window for layer in model.layers)):
        raise ValueError("Reference model must match the target's full-attention Qwen3 geometry")

    handoffs = []
    data = read_json(args.manifest)["documents"]
    for index, document in enumerate(data):
        source = Snapshot(args.caches / f"source-{index:03d}")
        target = Snapshot(args.caches / f"target-{index:03d}")
        verify_prefix(source, document["history"], geometry.source)
        verify_prefix(target, document["history"], spec)
        queries = []
        for query in document["queries"]:
            count = len(query["token_ids"])
            positions = torch.arange(source.tokens, source.tokens + count)[None, :]
            hidden = model.embed_tokens(torch.tensor([query["token_ids"]], dtype=torch.long))
            queries.append({"hidden": hidden, "rotary": model.rotary_emb(hidden, positions)})
        handoffs.append(Handoff(source, target, queries))

    paired = read_pairs(args.initial_data)
    sampled = sampling_plan(paired, 4096, args.seed, cfg["token_stride"])
    reports = []
    with new_directory(args.output) as staging:
        weights, bias = geometry.arrays(staging, "w+")
        weights[:] = mapping.weights
        bias[:] = mapping.bias
        for index, layer in enumerate(model.layers):
            initial_weights, initial_bias = mapping.weights[1, index], mapping.bias[1, index]
            metric, budget = energy_metric(sampled, geometry.layers[index], index, initial_weights, initial_bias)
            gram, cross, traces = collect_layer(layer, mapping, handoffs, index)
            before = after = 0.0
            for head in range(spec.kv_heads):
                correction, _ = solve_correction(gram[head], cross[head], metric[head], budget[head])
                weights[1, index, head] += correction[:-1]
                bias[1, index, head] += correction[-1]
            # Propagate with the actual exported coefficient precision.
            correction = np.concatenate((weights[1, index] - initial_weights,
                                         (bias[1, index] - initial_bias)[:, None, :]), axis=1)
            if not np.isfinite(correction).all():
                raise ValueError(f"Non-finite folded value correction at layer {index}")
            updates = np.sum(correction * (metric @ correction), axis=(1, 2))
            for query, pooled, mapped, errors, normalizer in traces:
                delta = pooled @ correction
                before += normalizer * float(np.sum(errors ** 2))
                after += normalizer * float(np.sum((errors - delta) ** 2))
                readout = mapped + torch.from_numpy(delta.reshape(mapped.shape).astype(np.float32))
                hidden = query["hidden"] + layer.self_attn.o_proj(readout.transpose(0, 1).reshape(
                    1, query["hidden"].shape[1], -1).contiguous())
                query["hidden"] = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            report = {"layer": index, "weighted_readout_sse_before": before,
                      "weighted_readout_sse_after": after, "energy": updates.tolist(), "budget": budget.tolist()}
            reports.append(report)
            print(f"layer {index:02d}: {before:.6f} -> {after:.6f}", flush=True)
        weights.flush()
        bias.flush()
        cfg["readout_calibration"] = {
            "documents": [d["id"] for d in data], "queries": sum(len(h.queries) for h in handoffs),
            "manifest_sha256": file_digest(args.manifest), "seed": args.seed, "layers": reports,
            "initial_sampled_tokens": sum(len(rows) for _, rows in sampled),
        }
        write_json(staging / "metadata.json", cfg)


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("initial", "initial-data", "manifest", "caches", "hf-target", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()
    with torch.inference_mode():
        calibrate(args)


if __name__ == "__main__":
    main()
