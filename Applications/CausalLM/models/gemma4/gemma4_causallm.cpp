// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   gemma4_causallm.cpp
 * @date   07 Apr 2026
 * @brief  This defines a Gemma4 causal language model.
 * @see    https://github.com/nnstreamer/
 * @author OpenAI Codex
 * @bug    No known bugs except for NYI items
 */

#include <gemma4_causallm.h>

#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>
#include <reshaped_rms_norm.h>

namespace causallm {

json &Gemma4Transformer::sanitizeConfig(json &cfg) {
  if (!cfg.contains("tie_word_embeddings")) {
    cfg["tie_word_embeddings"] = true;
  }

  if (!cfg.contains("head_dim") && cfg.contains("hidden_size") &&
      cfg.contains("num_attention_heads")) {
    cfg["head_dim"] = cfg["hidden_size"].get<unsigned int>() /
                      cfg["num_attention_heads"].get<unsigned int>();
  }

  return cfg;
}

json &Gemma4Transformer::sanitizeGenerationConfig(json &gen_cfg,
                                                  const json &cfg) {
  if (!gen_cfg.contains("eos_token_id")) {
    if (cfg.contains("eos_token_id")) {
      auto eos = cfg["eos_token_id"];
      if (eos.is_number()) {
        gen_cfg["eos_token_id"] =
          std::vector<unsigned int>{eos.get<unsigned int>()};
      } else {
        gen_cfg["eos_token_id"] = eos;
      }
    }
  } else {
    auto eos = gen_cfg["eos_token_id"];
    if (eos.is_number()) {
      gen_cfg["eos_token_id"] =
        std::vector<unsigned int>{eos.get<unsigned int>()};
    }
  }

  return gen_cfg;
}

void Gemma4Transformer::setupParameters(json &cfg, json &generation_cfg,
                                        json &nntr_cfg) {
  Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);

  if (cfg.contains("layer_types")) {
    layer_types = cfg["layer_types"].get<std::vector<std::string>>();
  }

  if (cfg.contains("attn_logit_softcapping") &&
      !cfg["attn_logit_softcapping"].is_null()) {
    ATTN_LOGIT_SOFTCAPPING = cfg["attn_logit_softcapping"].get<float>();
  }

  GLOBAL_HEAD_DIM = cfg.contains("global_head_dim") &&
                      !cfg["global_head_dim"].is_null()
                    ? cfg["global_head_dim"].get<unsigned int>()
                    : HEAD_DIM;

  NUM_GLOBAL_KEY_VALUE_HEADS =
    cfg.contains("num_global_key_value_heads") &&
      !cfg["num_global_key_value_heads"].is_null()
      ? cfg["num_global_key_value_heads"].get<unsigned int>()
      : NUM_KEY_VALUE_HEADS;

  ATTENTION_K_EQ_V = cfg.contains("attention_k_eq_v") &&
                     cfg["attention_k_eq_v"].get<bool>();

  FULL_ATTENTION_ROPE_THETA = ROPE_THETA;
  SLIDING_ATTENTION_ROPE_THETA = ROPE_THETA;

  if (cfg.contains("rope_parameters") && cfg["rope_parameters"].is_object()) {
    const auto &rope_params = cfg["rope_parameters"];
    if (rope_params.contains("full_attention") &&
        rope_params["full_attention"].contains("rope_theta")) {
      FULL_ATTENTION_ROPE_THETA =
        rope_params["full_attention"]["rope_theta"].get<unsigned int>();
    }
    if (rope_params.contains("sliding_attention") &&
        rope_params["sliding_attention"].contains("rope_theta")) {
      SLIDING_ATTENTION_ROPE_THETA =
        rope_params["sliding_attention"]["rope_theta"].get<unsigned int>();
    }
  }
}

std::vector<LayerHandle>
Gemma4Transformer::createTransformerDecoderBlock(const int layer_id,
                                                 std::string input_name) {

  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("input_layers", input_name),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  auto att_layer =
    createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                    "layer" + std::to_string(layer_id) + "_attention_norm",
                    "layer" + std::to_string(layer_id) + "_attention_norm",
                    "layer" + std::to_string(layer_id) + "_attention_norm");
  layers.insert(layers.end(), att_layer.begin(), att_layer.end());

  layers.push_back(createLayer(
    "rms_norm", {withKey("name", "layer" + std::to_string(layer_id) +
                                   "_post_attention_norm"),
                 withKey("input_layers",
                         "layer" + std::to_string(layer_id) + "_attention_out"),
                 withKey("epsilon", std::to_string(NORM_EPS)),
                 withKey("packed", "false")}));

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_post_attention"),
     withKey("input_layers", input_name + ",layer" + std::to_string(layer_id) +
                               "_post_attention_norm")}));

  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "pre_ffn_norm"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_post_attention"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  auto ffn_layer =
    createMlp(layer_id, DIM, INTERMEDIATE_SIZE,
              "layer" + std::to_string(layer_id) + "pre_ffn_norm");
  layers.insert(layers.end(), ffn_layer.begin(), ffn_layer.end());

  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "post_ffn_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output"),
     withKey("input_layers", "layer" + std::to_string(layer_id) +
                               "_post_attention,layer" +
                               std::to_string(layer_id) + "post_ffn_norm")}));

  return layers;
}

std::vector<LayerHandle> Gemma4Transformer::createAttention(
  const int layer_id, int seq_len, int n_heads, int head_dim,
  std::string query_name, std::string key_name, std::string value_name) {
  std::vector<LayerHandle> layers;

  auto Q = "layer" + std::to_string(layer_id) + "_wq";
  auto Q_norm = "layer" + std::to_string(layer_id) + "_q_norm";
  auto K = "layer" + std::to_string(layer_id) + "_wk";
  auto K_norm = "layer" + std::to_string(layer_id) + "_k_norm";
  auto V = "layer" + std::to_string(layer_id) + "_wv";
  auto A = "layer" + std::to_string(layer_id) + "_attention";
  auto O = "layer" + std::to_string(layer_id) + "_attention_out";

  bool is_sliding = true;
  if (!layer_types.empty() && layer_id < static_cast<int>(layer_types.size())) {
    is_sliding = layer_types[layer_id] == "sliding_attention";
  }

  int curr_head_dim = is_sliding ? HEAD_DIM : GLOBAL_HEAD_DIM;
  int curr_kv_heads = (is_sliding || !ATTENTION_K_EQ_V) ? NUM_KEY_VALUE_HEADS
                                                         : NUM_GLOBAL_KEY_VALUE_HEADS;

  // Q layer [B, S, H] -> [B, S, Nq*Dh]
  std::vector<std::string> q_params = {withKey("name", Q),
                                       withKey("unit", curr_head_dim * n_heads),
                                       withKey("disable_bias", "true"),
                                       withKey("input_layers", query_name),
                                       withKey("weight_initializer", "ones"),
                                       withKey("weight_dtype", FC_LAYER_DTYPE)};
  layers.push_back(createLayer("fully_connected", q_params));

  // K layer [B, S, H] -> [B, S, Nk*Dh]
  std::vector<std::string> k_params = {
    withKey("name", K),
    withKey("unit", curr_head_dim * curr_kv_heads),
    withKey("disable_bias", "true"),
    withKey("input_layers", key_name),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  layers.push_back(createLayer("fully_connected", k_params));

  // V layer [B, S, H] -> [B, S, Nk*Dh]
  // TODO: Gemma4 `attention_k_eq_v` shares K/V projection in full-attention
  // layers. NNTrainer currently keeps a dedicated V projection for parity with
  // existing MHA core path.
  std::vector<std::string> v_params = {
    withKey("name", V),
    withKey("unit", curr_head_dim * curr_kv_heads),
    withKey("disable_bias", "true"),
    withKey("input_layers", value_name),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  layers.push_back(createLayer("fully_connected", v_params));

  // q_norm on per-head projection [B, S, Nq*Dh]
  std::vector<std::string> q_norm_params = {
    withKey("name", Q_norm), withKey("input_layers", Q),
    withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(curr_head_dim))};
  layers.push_back(createLayer("reshaped_rms_norm", q_norm_params));

  // k_norm on per-head projection [B, S, Nk*Dh]
  std::vector<std::string> k_norm_params = {
    withKey("name", K_norm), withKey("input_layers", K),
    withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(curr_head_dim))};
  layers.push_back(createLayer("reshaped_rms_norm", k_norm_params));

  unsigned int window_size = is_sliding ? SLIDING_WINDOW : UINT_MAX;
  unsigned int rope_theta =
    is_sliding ? SLIDING_ATTENTION_ROPE_THETA : FULL_ATTENTION_ROPE_THETA;

  // Attention core receives [Q_norm, K_norm, V]
  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", curr_kv_heads),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
    withKey("sliding_window", window_size),
    withKey("rope_theta", std::to_string(rope_theta)),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("attn_logit_softcapping", std::to_string(ATTN_LOGIT_SOFTCAPPING)),
    withKey("is_causal", IS_CAUSAL ? "true" : "false"),
    withKey("input_layers", {Q_norm, K_norm, V})};
  layers.push_back(createLayer("mha_core", a_params));

  // O layer [B, S, Nq*Dh] -> [B, S, H]
  std::vector<std::string> o_params = {withKey("name", O),
                                       withKey("unit", DIM),
                                       withKey("disable_bias", "true"),
                                       withKey("input_layers", A),
                                       withKey("weight_initializer", "ones"),
                                       withKey("weight_dtype", FC_LAYER_DTYPE)};
  layers.push_back(createLayer("fully_connected", o_params));

  return layers;
}

std::vector<LayerHandle> Gemma4Transformer::createMlp(const int layer_id,
                                                      int dim, int hidden_dim,
                                                      std::string input_name) {
  std::vector<LayerHandle> layers;

  // TODO: Gemma4 supports per-layer double-width MLP for KV-shared tail layers.
  // This implementation currently follows the Gemma3 MLP path.

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("input_layers", input_name), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  layers.push_back(createLayer(
    "activation",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate_gelu"),
     withKey("activation", "tanh_gelu"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_ffn_gate")}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("input_layers", input_name), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  layers.push_back(createLayer(
    "multiply",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_geglu"),
     withKey("input_layers", "layer" + std::to_string(layer_id) +
                               "_ffn_gate_gelu,layer" +
                               std::to_string(layer_id) + "_ffn_up")}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", dim), withKey("disable_bias", "true"),
     withKey("input_layers", "layer" + std::to_string(layer_id) + "_ffn_geglu"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  return layers;
}

void Gemma4Transformer::registerCustomLayers() {
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

void Gemma4CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  Gemma4Transformer::registerCustomLayers();
}

} // namespace causallm
