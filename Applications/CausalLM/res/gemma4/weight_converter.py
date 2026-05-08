#!/usr/bin/env python3
"""
Weight converter for Gemma4 text model - converts from HuggingFace format to nntrainer format.

This script converts Gemma4 text model weights to the format expected by nntrainer,
with all weights saved in float32 format.

Key features:
- 35 layers with interleaved sliding/full attention
- KV sharing from layer 15 onwards (last 20 layers share KV from earlier layers)
- Double-wide MLP for KV-shared layers
- Per-layer input embeddings
- Full attention layers use head_dim=512, sliding attention uses head_dim=256

Reference: Gemma4ForConditionalGeneration with:
- 35 text layers
- Hidden size: 1536
- Vocab size: 262144
- Sliding attention: head_dim=256, num_kv_heads=1
- Full attention: head_dim=512, num_kv_heads=1
"""

import argparse
import torch
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM


def save_gemma4_for_nntrainer(params, config, dtype, file):
    """Convert and save Gemma4 text model weights as nntrainer format"""
    text_config = config.text_config
    n_layers = text_config.num_hidden_layers  # 35
    hidden_size = text_config.hidden_size  # 1536
    vocab_size = text_config.vocab_size  # 262144
    hidden_size_per_layer_input = text_config.hidden_size_per_layer_input  # 256
    intermediate_size = text_config.intermediate_size  # 6144
    num_kv_shared_layers = text_config.num_kv_shared_layers  # 20
    layer_types = text_config.layer_types
    use_double_wide_mlp = text_config.use_double_wide_mlp
    
    first_kv_shared_layer_idx = n_layers - num_kv_shared_layers  # 15
    
    # Head dimensions
    sliding_head_dim = text_config.head_dim  # 256
    full_head_dim = text_config.global_head_dim  # 512
    
    print(f"\nConverting Gemma4 text model weights:")
    print(f"  Number of layers: {n_layers}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Hidden size per layer input: {hidden_size_per_layer_input}")
    print(f"  Intermediate size: {intermediate_size}")
    print(f"  KV shared layers: {num_kv_shared_layers} (starting from layer {first_kv_shared_layer_idx})")
    print(f"  Double wide MLP: {use_double_wide_mlp}")
    print(f"  Sliding head dim: {sliding_head_dim}")
    print(f"  Full head dim: {full_head_dim}")
    print(f"  Output dtype: {dtype}")
    print("=" * 80)
    
    # Debug: Print all language_model-related keys
    print("\nAvailable language_model keys in state_dict:")
    lm_keys = [k for k in params.keys() if 'language_model' in k or k.startswith('model.')]
    for key in lm_keys[:30]:
        print(f"  {key}")
    if len(lm_keys) > 30:
        print(f"  ... and {len(lm_keys) - 30} more keys")
    print("=" * 80)
    
    total_bytes = 0
    
    def save_weight(name, weight, is_rms=False):
        """Save weight with optional RMSNorm conversion"""
        # Convert to float32 first to handle bfloat16
        weight = weight.float()
        
        if is_rms:
            print(f"  {name:60s} | RMSNorm (gamma) | shape={weight.shape} | dtype={weight.dtype}")
        else:
            print(f"  {name:60s} | weight | shape={weight.shape} | dtype={weight.dtype}")
        
        # Ensure float32 dtype
        weight_np = np.array(weight, dtype=dtype)
        weight_np.tofile(file)
        bytes_saved = weight_np.nbytes
        print(f"    Saved {bytes_saved:,} bytes ({bytes_saved / 1024:.2f} KB)")
        return bytes_saved
    
    def save_projection(layer_prefix, proj_name, transpose=True):
        """Save projection layer weights"""
        weight = params[f"{layer_prefix}{proj_name}.weight"]
        if transpose:
            # nntrainer expects (out_features, in_features) format
            # HuggingFace: (in_features, out_features) for most layers
            weight = weight.permute(1, 0)
        return save_weight(f"{proj_name}", weight)
    
    def is_kv_shared_layer(layer_idx):
        """Check if layer is KV-shared"""
        return layer_idx >= first_kv_shared_layer_idx
    
    def is_full_attention_layer(layer_idx):
        """Check if layer uses full attention"""
        return layer_types[layer_idx] == "full_attention"
    
    def get_head_dim(layer_idx):
        """Get head dimension for layer"""
        if is_full_attention_layer(layer_idx):
            return full_head_dim
        return sliding_head_dim
    
    def get_intermediate_size(layer_idx):
        """Get intermediate size for MLP (double-wide for KV-shared layers)"""
        if use_double_wide_mlp and is_kv_shared_layer(layer_idx):
            return intermediate_size * 2
        return intermediate_size
    
    # 1. Main embedding
    print("\n" + "=" * 80)
    print("1. Main Embedding (embedding0)")
    print("=" * 80)
    # Gemma4 uses scaled embedding, stored in language_model.embed_tokens
    embedding_weight = params.get("model.language_model.embed_tokens.weight", 
                                   params.get("model.embed_tokens.weight", 
                                              params.get("embed_tokens.weight")))
    if embedding_weight is None:
        # Try direct key
        for key in params.keys():
            if "embed_tokens" in key and "per_layer" not in key:
                embedding_weight = params[key]
                print(f"  Found embedding at: {key}")
                break
    bytes_saved = save_weight("embedding0", embedding_weight)
    total_bytes += bytes_saved
    
    # 2. Per-layer input embedding
    print("\n" + "=" * 80)
    print("2. Per-layer Input Embedding (per_layer_input_embedding)")
    print("=" * 80)
    ple_weight = None
    for key in params.keys():
        if "embed_tokens_per_layer" in key:
            ple_weight = params[key]
            print(f"  Found per-layer embedding at: {key}")
            break
    if ple_weight is not None:
        bytes_saved = save_weight("per_layer_input_embedding", ple_weight)
        total_bytes += bytes_saved
    else:
        print("  WARNING: per_layer_input_embedding not found!")
    
    # 3. Per-layer input projection
    print("\n" + "=" * 80)
    print("3. Per-layer Input Projection (per_layer_input_projection)")
    print("=" * 80)
    plp_weight = None
    for key in params.keys():
        if "per_layer_model_projection" in key:
            plp_weight = params[key]
            print(f"  Found per-layer projection at: {key}")
            break
    if plp_weight is not None:
        # Transpose: (num_layers * hidden_size_per_layer_input, hidden_size) -> (hidden_size, num_layers * hidden_size_per_layer_input)
        plp_weight = plp_weight.permute(1, 0)
        bytes_saved = save_weight("per_layer_input_projection", plp_weight)
        total_bytes += bytes_saved
    else:
        print("  WARNING: per_layer_input_projection not found!")

    # 4. Per-layer projection norm
    print("\n" + "=" * 80)
    print("4. Per-layer Projection Norm (per_layer_input_projection)")
    print("=" * 80)
    plp_weight = None
    for key in params.keys():
        if "per_layer_projection_norm" in key:
            plp_weight = params[key]
            print(f"  Found per_layer_projection_norm at: {key}")
            break
    if plp_weight is not None:
        # Transpose: (num_layers * hidden_size_per_layer_input, hidden_size) -> (hidden_size, num_layers * hidden_size_per_layer_input)
        bytes_saved = save_weight("per_layer_projection_norm", plp_weight, is_rms=True)
        total_bytes += bytes_saved
    else:
        print("  WARNING: per_layer_projection_norm not found!")

    
        
    # 5. Decoder layers (0-34)
    for layer_idx in range(n_layers):
        print(f"\n" + "=" * 80)
        print(f"Decoder Layer {layer_idx} ({layer_types[layer_idx]})")
        if is_kv_shared_layer(layer_idx):
            print(f"  [KV-SHARED LAYER - no wk/wv/k_norm]")
        if use_double_wide_mlp and is_kv_shared_layer(layer_idx):
            print(f"  [DOUBLE-WIDE MLP - intermediate_size={get_intermediate_size(layer_idx)}]")
        print("=" * 80)
        
        # Find layer prefix
        layer_prefix = None
        for key in params.keys():
            if f"layers.{layer_idx}." in key:
                layer_prefix = key.split(f"layers.{layer_idx}.")[0] + f"layers.{layer_idx}."
                break
        
        if layer_prefix is None:
            print(f"  ERROR: Could not find layer {layer_idx} in state dict!")
            continue
        
        print(f"  Layer prefix: {layer_prefix}")
        
        head_dim = get_head_dim(layer_idx)
        layer_intermediate_size = get_intermediate_size(layer_idx)
        is_kv_shared = is_kv_shared_layer(layer_idx)
        
        # Attention norm (input_layernorm)
        bytes_saved = save_weight(
            f"layer{layer_idx}_attention_norm",
            params[f"{layer_prefix}input_layernorm.weight"],
            is_rms=True
        )
        total_bytes += bytes_saved
        
        # Wq (query projection)
        # Shape: (hidden_size, num_heads * head_dim) -> transpose to (num_heads * head_dim, hidden_size)
        wq_weight = params[f"{layer_prefix}self_attn.q_proj.weight"]
        wq_weight = wq_weight.permute(1, 0)
        bytes_saved = save_weight(f"layer{layer_idx}_wq", wq_weight)
        total_bytes += bytes_saved
        

        
        # Wk, Wv (only for non-KV-shared layers)
        if not is_kv_shared:
            # Wk (key projection)
            wk_weight = params[f"{layer_prefix}self_attn.k_proj.weight"]
            wk_weight = wk_weight.permute(1, 0)
            bytes_saved = save_weight(f"layer{layer_idx}_wk", wk_weight)
            total_bytes += bytes_saved
            
            # Wv (value projection)
            wv_weight = params[f"{layer_prefix}self_attn.v_proj.weight"]
            wv_weight = wv_weight.permute(1, 0)
            bytes_saved = save_weight(f"layer{layer_idx}_wv", wv_weight)
            total_bytes += bytes_saved
            


        # Q norm (always present)
        q_norm_weight = params[f"{layer_prefix}self_attn.q_norm.weight"]
        bytes_saved = save_weight(f"layer{layer_idx}_q_norm", q_norm_weight, is_rms=True)
        total_bytes += bytes_saved

        # k_norm (only for non-KV-shared layers)
        if not is_kv_shared:
            # K norm
            k_norm_weight = params[f"{layer_prefix}self_attn.k_norm.weight"]
            bytes_saved = save_weight(f"layer{layer_idx}_k_norm", k_norm_weight, is_rms=True)
            total_bytes += bytes_saved
        
        # Attention output projection
        wo_weight = params[f"{layer_prefix}self_attn.o_proj.weight"]
        wo_weight = wo_weight.permute(1, 0)
        bytes_saved = save_weight(f"layer{layer_idx}_attention_out", wo_weight)
        total_bytes += bytes_saved
        
        # Post-attention norm
        bytes_saved = save_weight(
            f"layer{layer_idx}_post_attention_norm",
            params[f"{layer_prefix}post_attention_layernorm.weight"],
            is_rms=True
        )
        total_bytes += bytes_saved
        
        # Pre-FFN norm
        bytes_saved = save_weight(
            f"layer{layer_idx}_pre_ffn_norm",
            params[f"{layer_prefix}pre_feedforward_layernorm.weight"],
            is_rms=True
        )
        total_bytes += bytes_saved
        
        # FFN gate projection
        gate_weight = params[f"{layer_prefix}mlp.gate_proj.weight"]
        gate_weight = gate_weight.permute(1, 0)
        bytes_saved = save_weight(f"layer{layer_idx}_ffn_gate", gate_weight)
        total_bytes += bytes_saved
        
        # FFN up projection
        up_weight = params[f"{layer_prefix}mlp.up_proj.weight"]
        up_weight = up_weight.permute(1, 0)
        bytes_saved = save_weight(f"layer{layer_idx}_ffn_up", up_weight)
        total_bytes += bytes_saved
        
        # FFN down projection
        down_weight = params[f"{layer_prefix}mlp.down_proj.weight"]
        down_weight = down_weight.permute(1, 0)
        bytes_saved = save_weight(f"layer{layer_idx}_ffn_down", down_weight)
        total_bytes += bytes_saved
        
        # Post-FFN norm
        bytes_saved = save_weight(
            f"layer{layer_idx}_post_ffn_norm",
            params[f"{layer_prefix}post_feedforward_layernorm.weight"],
            is_rms=True
        )
        total_bytes += bytes_saved
        
        # Per-layer input gate
        plig_weight = params.get(f"{layer_prefix}per_layer_input_gate.weight")
        if plig_weight is not None:
            plig_weight = plig_weight.permute(1, 0)
            bytes_saved = save_weight(f"layer{layer_idx}_per_layer_input_gate", plig_weight)
            total_bytes += bytes_saved
        else:
            print(f"  WARNING: layer{layer_idx}_per_layer_input_gate not found!")
        
        # Per-layer input projection
        plip_weight = params.get(f"{layer_prefix}per_layer_projection.weight")
        if plip_weight is not None:
            plip_weight = plip_weight.permute(1, 0)
            bytes_saved = save_weight(f"layer{layer_idx}_per_layer_input_proj", plip_weight)
            total_bytes += bytes_saved
        else:
            print(f"  WARNING: layer{layer_idx}_per_layer_input_proj not found!")
        
        # Post per-layer input norm
        ppln_weight = params.get(f"{layer_prefix}post_per_layer_input_norm.weight")
        if ppln_weight is not None:
            bytes_saved = save_weight(f"layer{layer_idx}_post_per_layer_input_norm", ppln_weight, is_rms=True)
            total_bytes += bytes_saved
        else:
            print(f"  WARNING: layer{layer_idx}_post_per_layer_input_norm not found!")
        
        # layer scalar

        ppln_weight = params.get(f"{layer_prefix}layer_scalar")
        if ppln_weight is not None:
            bytes_saved = save_weight(f"layer{layer_idx}_layer_scalar", ppln_weight)
            total_bytes += bytes_saved
        else:
            print(f"  WARNING: layer{layer_idx}_layer_scalar not found!")

    
    # 6. Output norm
    print(f"\n" + "=" * 80)
    print("6. Output Norm")
    print("=" * 80)
    output_norm_weight = None
    for key in params.keys():
        if "language_model.norm.weight" in key:
            output_norm_weight = params[key]
            break
        elif key.endswith("model.norm.weight"):
            output_norm_weight = params[key]
            break
    if output_norm_weight is not None:
        bytes_saved = save_weight("output_norm", output_norm_weight, is_rms=True)
        total_bytes += bytes_saved
    else:
        print("  WARNING: output_norm not found!")
    
    print(f"\n" + "=" * 80)
    print(f"Total bytes saved: {total_bytes:,} ({total_bytes / 1024 / 1024 / 1024:.2f} GB)")
    print("=" * 80)
    
    return total_bytes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Gemma4 text model weights to nntrainer format")
    parser.add_argument(
        "--model_path",
        type=str,
        default=".",
        help="Path to Gemma4 model directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./nntr_gemma4_fp32.bin",
        help="Output binary file path"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Output data type (default: float32)"
    )
    
    args = parser.parse_args()
    
    print(f"Loading Gemma4 model from: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="float",  # Load in float32 for accuracy
        trust_remote_code=True
    )
    model.eval()
    
    print(f"\nModel Configuration:")
    print(f"  Text layers: {config.text_config.num_hidden_layers}")
    print(f"  Text hidden size: {config.text_config.hidden_size}")
    print(f"  Text vocab size: {config.text_config.vocab_size}")
    print(f"  KV shared layers: {config.text_config.num_kv_shared_layers}")
    print(f"  Layer types: {config.text_config.layer_types}")
    
    # Extract state dict
    state_dict = model.state_dict()
    
    print(f"\nTotal parameters in model: {sum(p.numel() for p in state_dict.values()):,}")
    
    # Convert and save
    try:
        with open(args.output, "wb") as f:
            save_gemma4_for_nntrainer(state_dict, config, args.dtype, f)
        print(f"\n✓ Successfully saved weights to: {args.output}")
    except Exception as e:
        print(f"\n✗ Error saving weights: {e}")
        import traceback
        traceback.print_exc()