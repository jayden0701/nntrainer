"""Convert supported dense, unscaled Llama HF weights to nntrainer FP32."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def convert(source, output):
    config = json.loads((source / "config.json").read_text())
    if (config.get("model_type") != "llama" or config.get("rope_scaling")
            or config.get("attention_bias", False) or config.get("mlp_bias", False)
            or config.get("hidden_act", "silu") != "silu"):
        raise ValueError("Only dense Llama, SiLU, bias-free projections and unscaled RoPE are supported")
    reader_path = (Path(__file__).resolve().parents[2] / "Applications/CausalLM/res/"
                   "qwen3/qwen3-0.6b/weight_converter_layergraph.py")
    spec = importlib.util.spec_from_file_location("nntr_safetensors_reader", reader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    keys, read = module.model_reader(source)
    order = [("model.embed_tokens.weight", False)]
    for layer in range(config["num_hidden_layers"]):
        for name, transpose in (
            ("input_layernorm", False), ("self_attn.q_proj", True),
            ("self_attn.k_proj", True), ("self_attn.v_proj", True),
            ("self_attn.o_proj", True), ("post_attention_layernorm", False),
            ("mlp.gate_proj", True), ("mlp.up_proj", True), ("mlp.down_proj", True),
        ):
            order.append((f"model.layers.{layer}.{name}.weight", transpose))
    order.append(("model.norm.weight", False))
    if not config.get("tie_word_embeddings", False):
        order.append(("lm_head.weight", True))
    required = {name for name, _ in order}
    missing = required - set(keys)
    extra = set(keys) - required - {"__metadata__"}
    if config.get("tie_word_embeddings", False) and "lm_head.weight" in extra:
        if not np.array_equal(read("lm_head.weight"), read("model.embed_tokens.weight")):
            raise ValueError("Declared tied embeddings disagree with lm_head")
        extra.remove("lm_head.weight")
    if missing or extra:
        raise ValueError(f"Unsupported weight layout: missing={sorted(missing)}, extra={sorted(extra)}")
    with output.open("xb") as stream:
        for name, transpose in order:
            array = read(name)
            np.ascontiguousarray(array.T if transpose else array, dtype="<f4").tofile(stream)
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size,
                      "tensors": len(order), "dtype": "float32"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.source, args.output)
