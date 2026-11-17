#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
#
## @file   weight_converter_layergraph.py
## @brief  HF Qwen3 -> nntrainer layer-graph .bin converter (int4-quantize source).
## @author Jijoong Moon <jijoong.moon@samsung.com>
##
## numpy-only converter: HF Qwen3 (single or sharded Safetensors) -> nntrainer layer-graph
## FP32 positional .bin, matching the weight order of the torch
## weight_converter.py in this directory (and res/qwen3/qwen3-4b/).
##
## This is STEP 1 of the QS4CX-FP16 pipeline (see build_qs4cx.sh): it produces
## the FP32-weight / FP16-norm source that nntr_quantize quantizes to the
## 1K-benchmark recipe (embedding Q6_K, FC QS4CX, lm_head Q6_K, QS4CX-FP16).
## Use --data_type float32 --norm_fp16 for that source. No torch/transformers
## lib needed (reads Safetensors directly; bf16 -> fp32 is an exact widen).
##
## Qwen3 graph order (whole model):
##   embed_tokens            (as-is, [vocab, hidden], no transpose)
##   for each layer:
##     input_layernorm       (raw gamma -- Qwen RMS uses w, NOT (1+w))
##     q_proj  -> transpose [in, out]
##     q_norm  gamma
##     k_proj  -> transpose
##     k_norm  gamma
##     v_proj  -> transpose
##     o_proj  -> transpose
##     post_attention_layernorm
##     gate_proj -> transpose
##     up_proj   -> transpose
##     down_proj -> transpose
##   model.norm.weight
##   (lm_head: written transposed only when NOT tied)
import argparse
import json
import struct
from pathlib import Path

import numpy as np


def load_safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    data_offset = 8 + n
    return hdr, data_offset


def make_reader(path, hdr, data_offset):
    DTYPE = {
        "F32": (np.float32, 4),
        "F16": (np.float16, 2),
        "BF16": (np.uint16, 2),  # widened to fp32 below
        "I8": (np.int8, 1),
    }

    def read(name):
        meta = hdr[name]
        dt = meta["dtype"]
        shape = meta["shape"]
        beg, end = meta["data_offsets"]
        np_dt, _ = DTYPE[dt]
        with open(path, "rb") as f:
            f.seek(data_offset + beg)
            raw = f.read(end - beg)
        arr = np.frombuffer(raw, dtype=np_dt)
        if dt == "BF16":
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        else:
            arr = arr.astype(np.float32)
        return arr.reshape(shape)

    return read


def model_reader(model_path):
    model_path = Path(model_path)
    single = model_path / "model.safetensors"
    if single.is_file():
        header, offset = load_safetensors_header(single)
        reader = make_reader(single, header, offset)
        return header.keys(), reader

    index = model_path / "model.safetensors.index.json"
    with index.open() as stream:
        tensor_shards = json.load(stream)["weight_map"]

    readers = {}
    shard_keys = {}
    for filename in sorted(set(tensor_shards.values())):
        shard = model_path / filename
        header, offset = load_safetensors_header(shard)
        readers[filename] = make_reader(shard, header, offset)
        shard_keys[filename] = header.keys()
    for tensor, filename in tensor_shards.items():
        if tensor not in shard_keys[filename]:
            raise ValueError(f"{tensor} is missing from indexed shard {filename}")

    def read(name):
        return readers[tensor_shards[name]](name)

    return tensor_shards.keys(), read


def weight_order(layers, tied):
    yield "model.embed_tokens.weight", False, False
    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        # Attention q/k norms follow their projections in layer-graph order.
        for name, norm, transpose in (
            ("input_layernorm", True, False),
            ("self_attn.q_proj", False, True),
            ("self_attn.q_norm", True, False),
            ("self_attn.k_proj", False, True),
            ("self_attn.k_norm", True, False),
            ("self_attn.v_proj", False, True),
            ("self_attn.o_proj", False, True),
            ("post_attention_layernorm", True, False),
            ("mlp.gate_proj", False, True),
            ("mlp.up_proj", False, True),
            ("mlp.down_proj", False, True),
        ):
            yield prefix + name + ".weight", norm, transpose
    yield "model.norm.weight", True, False
    if not tied:
        yield "lm_head.weight", False, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                    help="HF model dir containing Safetensors weights and config.json")
    ap.add_argument("--output_name", required=True)
    ap.add_argument("--data_type", choices=["float32", "float16"],
                    default="float32")
    ap.add_argument("--norm_fp16", action="store_true",
                    help="write RMSNorm gammas as FP16 (mixed bin for the "
                         "FP32-weight / FP16-activation quantize path)")
    args = ap.parse_args()

    out_dtype = np.float32 if args.data_type == "float32" else np.float16
    norm_fp16 = args.norm_fp16

    with (Path(args.model_path) / "config.json").open() as stream:
        cfg = json.load(stream)
    n_layers = cfg["num_hidden_layers"]
    tied = bool(cfg.get("tie_word_embeddings", False))

    tensors, read = model_reader(args.model_path)
    order = list(weight_order(n_layers, tied))
    for name, _, _ in order:
        if name not in tensors:
            raise ValueError(f"required tensor {name} is missing from {args.model_path}")

    total = 0
    with open(args.output_name, "wb") as stream:
        for name, is_rms, transpose in order:
            array = read(name)
            if transpose:
                array = array.T
            dtype = np.float16 if is_rms and norm_fp16 else out_dtype
            array = np.ascontiguousarray(array, dtype=dtype)
            array.tofile(stream)
            total += array.nbytes

    print(f"wrote {args.output_name}: {total} bytes "
          f"({total/1e9:.2f} GB), {n_layers} layers, "
          f"dtype={args.data_type}, norm_fp16={norm_fp16}, tied={tied}")


if __name__ == "__main__":
    main()
