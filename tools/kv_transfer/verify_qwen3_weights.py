#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare every FP32 layer-graph weight against its original Qwen3 Safetensor."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def verify(hf_dir, native_dir):
    root = Path(__file__).resolve().parents[2]
    path = root / "Applications/CausalLM/res/qwen3/qwen3-0.6b/weight_converter_layergraph.py"
    module = importlib.util.spec_from_file_location("qwen3_converter", path)
    converter = importlib.util.module_from_spec(module)
    module.loader.exec_module(converter)

    if (hf_dir / "config.json").read_bytes() != (native_dir / "config.json").read_bytes():
        raise ValueError("model configs differ")
    config = json.loads((hf_dir / "config.json").read_text())
    nntr = json.loads((native_dir / "nntr_config.json").read_text())
    if config["model_type"] != "qwen3" or nntr["model_tensor_type"] != "FP32-FP32":
        raise ValueError("expected native Qwen3 FP32 weights")
    binary = native_dir / nntr["model_file_name"]
    native = np.memmap(binary, mode="r", dtype="<f4")
    shards = {}
    for file in hf_dir.glob("*.safetensors"):
        with safe_open(file, framework="pt") as shard:
            for name in shard.keys():
                if name in shards:
                    raise ValueError(f"duplicate checkpoint tensor {name}")
                shards[name] = file
    tied = config.get("tie_word_embeddings", False)
    order = list(converter.weight_order(config["num_hidden_layers"], tied))
    expected = {name for name, _, _ in order}
    if tied and "lm_head.weight" in shards:
        with safe_open(shards["lm_head.weight"], framework="pt") as head, safe_open(
                shards["model.embed_tokens.weight"], framework="pt") as embed:
            if not torch.equal(head.get_tensor("lm_head.weight"), embed.get_tensor("model.embed_tokens.weight")):
                raise ValueError("explicit LM head differs from the tied embedding")
        expected.add("lm_head.weight")
    if set(shards) != expected:
        raise ValueError("unexpected or missing tensors in the checkpoint")
    offset = 0
    for name, _, transpose in order:
        with safe_open(shards[name], framework="pt") as shard:
            tensor = shard.get_tensor(name)
            shape = tuple(reversed(tensor.shape)) if transpose else tensor.shape
            packed = native[offset:offset + tensor.numel()].reshape(shape)
            unpacked = packed.T if transpose else packed
            for start in range(0, tensor.shape[0], 1024):
                rows = slice(start, start + 1024)
                reference = tensor[rows].float().numpy()
                if not np.array_equal(reference.view(np.uint32), unpacked[rows].view(np.uint32)):
                    raise ValueError(f"{name}: differing FP32 bits in rows {rows}")
            offset += tensor.numel()
    if offset != len(native):
        raise ValueError("trailing native weight data")
    return {"file": binary.name, "tensors": len(order), "bytes": binary.stat().st_size,
            "bitwise_equal": True}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("hf", type=Path)
    p.add_argument("native", type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(verify(args.hf, args.native), indent=2))
