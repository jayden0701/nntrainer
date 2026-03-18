// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   timm_vit_transformer.cpp
 * @date   28 Jan 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief   This timm_vit_transformer.cpp constructs a class for timm ViT model
 * compatible with the PyTorch timm library.
 */

#include "t5gemma2.h"
#include "t5gemma2_processor.h"
#include <app_context.h>
#include <cfloat>
#include <engine.h>
#include <factory.h>
#include <llm_util.hpp>
#include <random>
#include <fused_fc_reshaped_rms_norm.h>
#include <reshaped_rms_norm.h>

namespace causallm {

void T5Gemma2Transformer::setupParameters(json &cfg, json &generation_cfg,
                                          json &nntr_cfg) {
  BATCH_SIZE = nntr_cfg.value("batch_size", 1);
  MODEL_TENSOR_TYPE = nntr_cfg.value("model_tensor_type", "FP32-FP32");
  EMBEDDING_DTYPE = nntr_cfg.value("embedding_dtype", "FP32");
  FC_LAYER_DTYPE = nntr_cfg.value("fc_layer_dtype", "FP32");

  // we use same tokenizer over all
  NUM_VOCAB = cfg.value("vocab_size", 262144);

  TOKEN_INDEX_EOI = cfg.value("eoi_token_index", 256000);
  TOKEN_INDEX_IMAGE = cfg.value("image_token_index", 256001);
  EOS_TOKEN_ID = cfg.value("eos_token_id", 1);
  BOS_TOKEN_ID = cfg.value("bos_token_id", 2);

  // Decoder configuration
  if (cfg.contains("decoder")) {
    json &decoder_cfg = cfg["decoder"];
    DIM = decoder_cfg.value("hidden_size", 768);
    INTERMEDIATE_SIZE = decoder_cfg.value("intermediate_size", 3072);
    NUM_LAYERS = decoder_cfg.value("num_hidden_layers", 12);
    NUM_HEADS = decoder_cfg.value("num_attention_heads", 12);
    HEAD_DIM = decoder_cfg.value("head_dim", DIM / NUM_HEADS);
    NUM_KEY_VALUE_HEADS = decoder_cfg.value("num_key_value_heads", NUM_HEADS);
    MAX_POSITION_EMBEDDINGS = decoder_cfg.value("max_position_embeddings", 196);
    NORM_EPS = decoder_cfg.value("rms_norm_eps", 1e-6);
    IS_CAUSAL = decoder_cfg.value("use_bidirectional_attention", false);
    GQA_SIZE = NUM_HEADS / NUM_KEY_VALUE_HEADS;

    // Sliding window
    if (decoder_cfg.contains("sliding_window") &&
        !decoder_cfg["sliding_window"].is_null()) {
      SLIDING_WINDOW = decoder_cfg["sliding_window"].get<unsigned int>();
    } else {
      SLIDING_WINDOW = UINT_MAX;
    }

    // RoPE parameters (use sliding attention defaults)
    if (decoder_cfg.contains("rope_parameters") &&
        decoder_cfg["rope_parameters"].contains("sliding_attention")) {
      json &rope_cfg = decoder_cfg["rope_parameters"]["sliding_attention"];
      ROPE_THETA = rope_cfg.value("rope_theta", 10000);
    } else {
      ROPE_THETA = 10000;
    }
  } else {
    // Fallback to top-level config for compatibility
    DIM = cfg.value("hidden_size", 768);
    INTERMEDIATE_SIZE = cfg.value("intermediate_size", 3072);
    NUM_LAYERS = cfg.value("num_hidden_layers", 12);
    NUM_HEADS = cfg.value("num_attention_heads", 12);
    HEAD_DIM = cfg.value("head_dim", DIM / NUM_HEADS);
    NUM_KEY_VALUE_HEADS = cfg.value("num_key_value_heads", NUM_HEADS);
    MAX_POSITION_EMBEDDINGS = cfg.value("max_position_embeddings", 196);
    ROPE_THETA = cfg.value("rope_theta", 10000);
    NORM_EPS = cfg.value("norm_eps", 1e-6);
    GQA_SIZE = NUM_HEADS / NUM_KEY_VALUE_HEADS;
    IS_CAUSAL = cfg.value("is_causal", false);
    SLIDING_WINDOW =
      cfg.contains("sliding_window") && !cfg["sliding_window"].is_null()
        ? cfg["sliding_window"].get<unsigned int>()
        : UINT_MAX;
  }

  TIE_WORD_EMBEDDINGS = cfg.value("tie_word_embeddings", false);

  INIT_SEQ_LEN = nntr_cfg.value("init_seq_len", 224);
  MAX_SEQ_LEN = nntr_cfg.value("max_seq_len", 224);
  NUM_TO_GENERATE = nntr_cfg.value("num_to_generate", 0);

  // ==========ENCODER CONFIGURATION ==========
  if (cfg.contains("encoder")) {
    json &encoder_cfg = cfg["encoder"];

    // Encoder text config (for processing image features in text space)
    if (encoder_cfg.contains("text_config")) {
      json &enc_text_cfg = encoder_cfg["text_config"];
      ENC_NUM_LAYERS = enc_text_cfg.value("num_hidden_layers", 18);
      ENC_NUM_HEADS = enc_text_cfg.value("num_attention_heads", 4);
      ENC_NUM_KEY_VALUE_HEADS = enc_text_cfg.value("num_key_value_heads", 1);
      ENC_HEAD_DIM = enc_text_cfg.value("head_dim", 256);
      ENC_HIDDEN_SIZE = enc_text_cfg.value("hidden_size", 640);
      ENC_INTERMEDIATE_SIZE = enc_text_cfg.value("intermediate_size", 2048);
      ENC_MAX_POSITION_EMBEDDINGS =
        enc_text_cfg.value("max_position_embeddings", 32768);
      ENC_NORM_EPS = enc_text_cfg.value("rms_norm_eps", 1e-6f);
      ENC_SLIDING_WINDOW = enc_text_cfg.value("sliding_window", 512);
      ENC_IS_BIDIRECTIONAL =
        enc_text_cfg.value("use_bidirectional_attention", false);
      ENC_USE_CROSS_ATTENTION =
        enc_text_cfg.value("add_cross_attention", false);
      ENC_SLIDING_WINDOW_PATTERN =
        enc_text_cfg.value("_sliding_window_pattern", 6);
      ENC_GQA_SIZE = ENC_NUM_HEADS / ENC_NUM_KEY_VALUE_HEADS;

      ENC_MM_TOKENS_PER_IMAGE = encoder_cfg.value("mm_tokens_per_image", 256);

      // RoPE parameters for encoder
      if (enc_text_cfg.contains("rope_parameters")) {
        json &rope_params = enc_text_cfg["rope_parameters"];

        // Sliding attention RoPE parameters
        if (rope_params.contains("sliding_attention")) {
          json &rope_cfg = rope_params["sliding_attention"];
          ENC_ROPE_THETA_SLIDING = rope_cfg.value("rope_theta", 10000.0f);
        } else {
          ENC_ROPE_THETA_SLIDING = 10000.0f;
        }

        // Full attention RoPE parameters
        if (rope_params.contains("full_attention")) {
          json &rope_cfg = rope_params["full_attention"];
          ENC_ROPE_THETA = rope_cfg.value("rope_theta", 1000000.0f);
          ENC_ROPE_FACTOR= rope_cfg.value("factor", 8.0f);
        } else {
          ENC_ROPE_THETA = 1000000.0f;
        }
      } else {
        // Default values
        ENC_ROPE_THETA_SLIDING = 10000.0f;
        ENC_ROPE_THETA = 1000000.0f;
      }
    }

    // Vision config (SigLIP)
    if (encoder_cfg.contains("vision_config")) {
      json &vision_cfg = encoder_cfg["vision_config"];
      VISION_NUM_LAYERS = vision_cfg.value("num_hidden_layers", 27);
      VISION_NUM_HEADS = vision_cfg.value("num_attention_heads", 16);
      VISION_HIDDEN_SIZE = vision_cfg.value("hidden_size", 1152);
      VISION_INTERMEDIATE_SIZE = vision_cfg.value("intermediate_size", 4304);
      VISION_IMAGE_SIZE = vision_cfg.value("image_size", 896);
      VISION_PATCH_SIZE = vision_cfg.value("patch_size", 14);
      VISION_NUM_CHANNELS = vision_cfg.value("num_channels", 3);
      VISION_NORM_EPS = vision_cfg.value("layer_norm_eps", 1e-6f);
      VISION_HEAD_DIM = VISION_HIDDEN_SIZE / VISION_NUM_HEADS;
      VISION_NUM_PATCHES = (VISION_IMAGE_SIZE / VISION_PATCH_SIZE) *
                           (VISION_IMAGE_SIZE / VISION_PATCH_SIZE);
    }
  }

  // ========== DECODER CONFIGURATION ==========
  if (cfg.contains("decoder")) {
    json &decoder_cfg = cfg["decoder"];
    DEC_NUM_LAYERS = decoder_cfg.value("num_hidden_layers", 18);
    DEC_NUM_HEADS = decoder_cfg.value("num_attention_heads", 4);
    DEC_NUM_KEY_VALUE_HEADS = decoder_cfg.value("num_key_value_heads", 1);
    DEC_HEAD_DIM = decoder_cfg.value("head_dim", 256);
    DEC_HIDDEN_SIZE = decoder_cfg.value("hidden_size", 640);
    DEC_INTERMEDIATE_SIZE = decoder_cfg.value("intermediate_size", 2048);
    DEC_MAX_POSITION_EMBEDDINGS =
      decoder_cfg.value("max_position_embeddings", 32768);
    DEC_NORM_EPS = decoder_cfg.value("rms_norm_eps", 1e-6f);
    DEC_SLIDING_WINDOW = decoder_cfg.value("sliding_window", 512);
    DEC_IS_CAUSAL = !decoder_cfg.value("use_bidirectional_attention", false);

    DEC_QUERY_PRE_ATTN_SCALAR = decoder_cfg.value("query_pre_attn_scalar", 256);
    DEC_SLIDING_WINDOW_PATTERN =
      decoder_cfg.value("_sliding_window_pattern", 6);
    // DEC_ATTN_LOGIT_SOFTCAPPING =
    //   decoder_cfg.value("attn_logit_softcapping", 0.0);
    // DEC_FINAL_LOGIT_SOFTCAPPING =
    //   decoder_cfg.value("final_logit_softcapping", 0.0);

    // RoPE parameters for decoder
    if (decoder_cfg.contains("rope_parameters")) {
      json &rope_params = decoder_cfg["rope_parameters"];

      // Sliding attention RoPE parameters
      if (rope_params.contains("sliding_attention")) {
        json &rope_cfg = rope_params["sliding_attention"];
        DEC_ROPE_THETA_SLIDING = rope_cfg.value("rope_theta", 10000.0f);
      } else {
        DEC_ROPE_THETA_SLIDING = 10000.0f;
      }

      // Full attention RoPE parameters
      if (rope_params.contains("full_attention")) {
        json &rope_cfg = rope_params["full_attention"];
        DEC_ROPE_THETA = rope_cfg.value("rope_theta", 1000000.0f);
      } else {
        DEC_ROPE_THETA = 1000000.0f;
      }
    } else {
      // Default values
      DEC_ROPE_THETA_SLIDING = 10000.0f;
      DEC_ROPE_THETA = 1000000.0f;
    }
  }

  // Image configuration
  IMG_SIZE = VISION_IMAGE_SIZE;
  PATCH_SIZE = VISION_PATCH_SIZE;
  NUM_PATCHES = VISION_NUM_PATCHES;
  IMG_CHANNELS = VISION_NUM_CHANNELS;
}

std::vector<LayerHandle> T5Gemma2Transformer::createPatchEmbed() {
  std::vector<LayerHandle> layers;

  int embed_dim = DIM;

  layers.push_back(createLayer(
    "input", {withKey("name", "input_image"),
              withKey("input_shape", std::to_string(IMG_CHANNELS) + ":" +
                                       std::to_string(IMG_SIZE) + ":" +
                                       std::to_string(IMG_SIZE))}));

  std::vector<std::string> conv_params = {
    withKey("name", "patch_embed/conv"),
    withKey("kernel_size",
            {std::to_string(PATCH_SIZE), std::to_string(PATCH_SIZE)}),
    withKey("filters", std::to_string(embed_dim)),
    withKey("stride", {std::to_string(PATCH_SIZE), std::to_string(PATCH_SIZE)}),
    withKey("padding", "valid"),
    withKey("input_layers", "input_image")};
  layers.push_back(createLayer("conv2d", conv_params));

  layers.push_back(createLayer(
    "reshape", {withKey("name", "patch_embed/flatten"),
                withKey("target_shape", "1:" + std::to_string(embed_dim) + ":" +
                                          std::to_string(NUM_PATCHES)),
                withKey("input_layers", "patch_embed/conv")}));

  layers.push_back(
    createLayer("permute", {withKey("name", "patch_embed/transpose"),
                            withKey("direction", {1, 3, 2}),
                            withKey("input_layers", "patch_embed/flatten")}));

  layers.push_back(createLayer(
    "weight",
    {withKey("name", "pos_embed/weights"),
     withKey("weight_dim", "1:1:" + std::to_string(NUM_PATCHES) + ":" +
                             std::to_string(embed_dim)),
     withKey("tensor_dtype", "FP32"), withKey("weight_name", "pos_embed")}));

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "pos_embed/add"),
     withKey("input_layers", {"patch_embed/transpose", "pos_embed/weights"})}));

  return layers;
}

std::vector<LayerHandle> T5Gemma2Transformer::createMergedAttention(
  std::string prefix, const int layer_id, int seq_len, int n_heads,
  int head_dim, int gqa_size, std::string query_name, std::string key_name,
  std::string value_name, std::string cross_key_name, std::string cross_value_name) {

  std::vector<LayerHandle> layers;

  // === Self-Attention Projections ===
  auto Q = prefix + "wq";
  auto K = prefix + "wk";
  auto V = prefix + "wv";
  auto O = prefix + "attention_out";

  auto Q_norm = Q + "_norm";
  auto K_norm = K + "_norm";

  // Query projection (decoder hidden states -> Q)
  std::vector<std::string> q_params = {
    withKey("name", Q), withKey("unit", head_dim * n_heads),
    withKey("disable_bias", "true"), withKey("input_layers", query_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", q_params));

  // Key projection (decoder hidden states -> K)
  std::vector<std::string> k_params = {
    withKey("name", K), withKey("unit", head_dim * n_heads / gqa_size),
    withKey("disable_bias", "true"), withKey("input_layers", key_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", k_params));

  // Value projection (decoder hidden states -> V)
  std::vector<std::string> v_params = {
    withKey("name", V), withKey("unit", head_dim * n_heads / gqa_size),
    withKey("disable_bias", "true"), withKey("input_layers", value_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", v_params));

  // Q normalization (RMSNorm)
  std::vector<std::string> q_norm_params = {
    withKey("name", Q_norm), withKey("input_layers", Q),
    withKey("packed", "false"),
    withKey("epsilon", std::to_string(DEC_NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim))};
  layers.push_back(createLayer("reshaped_rms_norm", q_norm_params));

  // K normalization (RMSNorm)
  std::vector<std::string> k_norm_params = {
    withKey("name", K_norm), withKey("input_layers", K),
    withKey("packed", "false"),
    withKey("epsilon", std::to_string(DEC_NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim))};
  layers.push_back(createLayer("reshaped_rms_norm", k_norm_params));

  // === Cross-Attention Projections ===
  auto cross_K = prefix + "cross_wk";
  auto cross_V = prefix + "cross_wv";
  auto cross_K_norm = cross_K + "_norm";

  // Cross Key projection (encoder hidden states -> cross_K)
  std::vector<std::string> cross_k_params = {
    withKey("name", cross_K), withKey("unit", head_dim * n_heads / gqa_size),
    withKey("disable_bias", "true"), withKey("input_layers", cross_key_name),
    withKey("weight_initializer", "ones"),};
  layers.push_back(createLayer("fully_connected", cross_k_params));

  // Cross Value projection (encoder hidden states -> cross_V)
  std::vector<std::string> cross_v_params = {
    withKey("name", cross_V), withKey("unit", head_dim * n_heads / gqa_size),
    withKey("disable_bias", "true"), withKey("input_layers", cross_value_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", cross_v_params));

  // Cross K normalization (RMSNorm)
  std::vector<std::string> cross_k_norm_params = {
    withKey("name", cross_K_norm), withKey("input_layers", cross_K),
    withKey("packed", "false"),
    withKey("epsilon", std::to_string(DEC_NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim))};
  layers.push_back(createLayer("reshaped_rms_norm", cross_k_norm_params));

  // === Concatenation of Key and Value for Merged Attention ===
  // Concat: [self_K, cross_K] along sequence dimension
  auto concat_K = prefix + "concat_key";
  std::vector<std::string> concat_k_params = {
    withKey("name", concat_K),
    withKey("input_layers", {K_norm, cross_K_norm}),
    withKey("axis", "2")};  // Concat along sequence dimension (height)
  layers.push_back(createLayer("concat", concat_k_params));

  // Concat: [self_V, cross_V] along sequence dimension
  auto concat_V = prefix + "concat_value";
  std::vector<std::string> concat_v_params = {
    withKey("name", concat_V),
    withKey("input_layers", {V, cross_V}),
    withKey("axis", "2")};  // Concat along sequence dimension (height)
  layers.push_back(createLayer("concat", concat_v_params));

  // === Merged Attention ===
  auto A = prefix + "attention";

  // Determine RoPE theta based on layer type
  bool is_full_attention = ((layer_id + 1) % DEC_SLIDING_WINDOW_PATTERN == 0);

  // Merged attention: Q attends to [self_K + cross_K]
  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", n_heads / gqa_size),
    withKey("max_timestep", std::to_string(MAX_SEQ_LEN + INIT_SEQ_LEN)),  // decoder + encoder seq len
    withKey("sliding_window",
            is_full_attention ? UINT_MAX : DEC_SLIDING_WINDOW),
    withKey("use_rope", "false"),
    withKey("rope_theta",
            is_full_attention ? DEC_ROPE_THETA : DEC_ROPE_THETA_SLIDING),
    withKey("max_new_tokens", std::to_string(0)),
    withKey("is_causal", "true"),  // Decoder is causal
    withKey("input_layers", {Q_norm, concat_K, concat_V})};
  layers.push_back(createLayer("mha_core", a_params));

  // === Output Projection ===
  std::vector<std::string> o_params = {
    withKey("name", O), withKey("unit", DEC_HIDDEN_SIZE),
    withKey("disable_bias", "true"), withKey("input_layers", A),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", o_params));

  return layers;
}



// TODO : vision에서 코드 재활용 가능한지는 나중에 확인
std::vector<LayerHandle> T5Gemma2Transformer::createSelfAttention(
  std::string prefix, const int layer_id, int seq_len, int n_heads,
  int head_dim, int gqa_size, std::string query_name, std::string key_name,
  std::string value_name) {

  std::vector<LayerHandle> layers;

  auto Q = prefix + "wq";
  auto K = prefix + "wk";
  auto V = prefix + "wv";
  auto A = prefix + "attention";
  auto O = prefix + "attention_out";

  auto Q_norm = Q + "_norm";
  auto K_norm = K + "_norm";

  // V layer
  std::vector<std::string> v_params = {
    withKey("name", V), withKey("unit", head_dim * n_heads / gqa_size),
    withKey("disable_bias", "true"), withKey("input_layers", value_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", v_params));

  // K projection + reshaped RMSNorm
  std::vector<std::string> k_params = {
    withKey("name", K_norm), withKey("unit", head_dim * n_heads / gqa_size),
    withKey("disable_bias", "true"), withKey("input_layers", key_name),
    withKey("weight_initializer", "ones"),
    withKey("epsilon", std::to_string(ENC_NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim))};
  layers.push_back(createLayer("fused_fc_reshaped_rms_norm", k_params));

  // Q projection + reshaped RMSNorm
  std::vector<std::string> q_params = {
    withKey("name", Q_norm), withKey("unit", head_dim * n_heads),
    withKey("disable_bias", "true"), withKey("input_layers", query_name),
    withKey("weight_initializer", "ones"),
    withKey("epsilon", std::to_string(ENC_NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim))};
  layers.push_back(createLayer("fused_fc_reshaped_rms_norm", q_params));

  // Determine RoPE theta based on layer type
  bool is_full_attention = ((layer_id + 1) % ENC_SLIDING_WINDOW_PATTERN == 0);

  // Attention core layer
  std::vector<std::string> a_params = {
    withKey("name", A), withKey("num_heads", n_heads),
    withKey("num_heads_kv", n_heads / gqa_size),
    withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
    withKey("sliding_window",
            is_full_attention ? UINT_MAX : ENC_SLIDING_WINDOW),
    withKey("use_rope", "true"),
    withKey("rope_theta",
            is_full_attention ? ENC_ROPE_THETA : ENC_ROPE_THETA_SLIDING),
    withKey("rope_scaling_type", is_full_attention ? "linear" : "default"),
    withKey("rope_scaling_factor", ENC_ROPE_FACTOR),
    // set "max_new_tokens" to 1 for encoding mode 
    withKey("max_new_tokens", std::to_string(1)),
    withKey("is_causal", "false"),
    withKey("input_layers", {Q_norm, K_norm, V})};
  layers.push_back(createLayer("mha_core", a_params));

  // O layer
  std::vector<std::string> o_params = {
    withKey("name", O), withKey("unit", ENC_HIDDEN_SIZE),
    withKey("disable_bias", "true"), withKey("input_layers", A),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", o_params));

  return layers;
}

std::vector<LayerHandle>
T5Gemma2Transformer::createMlp(std::string prefix, int dim,
                               int intermediate_dim, std::string input_name) {
  std::vector<LayerHandle> layers;

  // Gate projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", prefix + "ffn_gate"), withKey("unit", intermediate_dim),
     withKey("disable_bias", "true"), withKey("input_layers", input_name),
     withKey("weight_initializer", "ones")}));

  // Up projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", prefix + "ffn_up"), withKey("unit", intermediate_dim),
     withKey("disable_bias", "true"), withKey("input_layers", input_name),
     withKey("weight_initializer", "ones")}));

  // Fused GeGLU: tanh_gelu(gate) * up
  layers.push_back(createLayer(
    "geglu", {withKey("name", prefix + "ffn_geglu"),
               withKey("input_layers",
                       prefix + "ffn_gate" + "," + prefix + "ffn_up")}));

  // Down projection
  layers.push_back(createLayer(
    "fully_connected", {withKey("name", prefix + "ffn_down"),
                        withKey("unit", dim), withKey("disable_bias", "true"),
                        withKey("input_layers", prefix + "ffn_geglu"),
                        withKey("weight_initializer", "ones")}));

  return layers;
}

void T5Gemma2Transformer::constructModel() {
  // Deprecated: Not used for lazy initialization approach
  // Models are created separately in createEncoderModel() and
  // createDecoderModel()
}

void T5Gemma2Transformer::initialize() {
  registerCustomLayers();

  // Create and compile encoder model (no memory allocation yet)
  createEncoderModel();

  // Create and compile decoder model (no memory allocation yet)
  createDecoderModel();

  is_initialized = true;

  std::cout << "[Init] Models compiled (no memory allocated yet)" << std::endl;
  std::cout
    << "[Init] Memory will be allocated on load(initialize() is called there)"
    << std::endl;
}

void T5Gemma2Transformer::registerCustomLayers() {
  Transformer::registerCustomLayers();

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

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::FusedFCReshapedRMSNormLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

void T5Gemma2Transformer::run(const WSTR prompt, bool do_sample,
                              const WSTR system_prompt,
                              const WSTR tail_prompt) {

  if (!is_initialized) {
    throw std::runtime_error("T5Gemma2 model is not initialized. Please call "
                             "initialize() before run().");
  }

  // Convert WSTR to std::string if needed (for Windows compatibility)
  std::string input_prompt;
#ifdef _WIN32
  input_prompt = std::string(prompt.begin(), prompt.end());
#else
  input_prompt = prompt;
#endif

  // Process input prompt (extracts text and images)
  auto processor_output = processor->process(input_prompt);

  std::cout << "\n========== T5Gemma2 Inference ==========\n" << std::endl;
  std::cout << "Input: " << input_prompt << std::endl;
  std::cout << "  processed_text length: "
            << processor_output.processed_text.size() << std::endl;
  std::cout << "  pixel_values size: " << processor_output.pixel_values.size()
            << std::endl;

  // Check if we have image input or text-only input
  if (!processor_output.pixel_values.empty()) {
    // === IMAGE INPUT MODE ===
    // TODO fix here
    std::cout << "\n[Mode: Image-Text Multimodal not implemented yett]\n"
              << std::endl;
    return;

    // Prepare model input for image
    float *image_input =
      (float *)malloc(sizeof(float) * processor_output.pixel_values.size());
    if (!image_input) {
      throw std::runtime_error("Failed to allocate memory for image input.");
    }

    std::copy(processor_output.pixel_values.begin(),
              processor_output.pixel_values.end(), image_input);

    // Run encoder inference to get image representations
    std::vector<float *> input_tensors = {image_input};
    std::vector<float *> label_tensors;

    // Encoder output shape: [NUM_PATCHES, ENC_HIDDEN_SIZE]
    auto encoder_output =
      model->incremental_inference(BATCH_SIZE, input_tensors, label_tensors,
                                   NUM_PATCHES, 0, NUM_PATCHES, false);

    std::cout << "[Encoder] Processed image to " << NUM_PATCHES
              << " patches with dimension " << ENC_HIDDEN_SIZE << std::endl;
    std::cout << "[Encoder] First 5 output values: ";
    for (int i = 0; i < std::min(5, ENC_HIDDEN_SIZE); ++i) {
      std::cout << encoder_output[0][i] << " ";
    }
    std::cout << std::endl;

    free(image_input);

    // TODO: Implement decoder to generate text conditioned on encoder output
    std::cout << "\n[Note] Decoder generation from encoder output is not yet "
                 "implemented.\n"
              << std::endl;

  } else {
    // === TEXT-ONLY INPUT MODE (Lazy Initialization) ===
    std::cout << "\n[Mode: Text-Only with Lazy Initialization]\n" << std::endl;

    if (!tokenizer) {
      throw std::runtime_error(
        "Tokenizer is not set. Cannot process text-only input.");
    }

    // Tokenize the processed text using the tokenizer
    std::vector<int> input_ids =
      tokenizer->Encode(processor_output.processed_text);
    unsigned int input_len = input_ids.size();


    ACTUAL_SEQ_LEN = input_len;

    if (input_len == 0) {
      throw std::runtime_error("Input text resulted in empty token sequence.");
    }

    std::cout << "[Pipeline] Tokenized "
              << processor_output.processed_text.size() << " characters to "
              << input_len << " tokens" << std::endl;

    // Truncate if necessary
    unsigned int max_input_len = MAX_SEQ_LEN - NUM_TO_GENERATE;
    if (input_len > max_input_len) {
      std::cout << "[Warning] Input length " << input_len << " exceeds maximum "
                << max_input_len << ". Truncating." << std::endl;
      input_len = max_input_len;
    }

    // Allocate input tensor
    float *text_input =
      (float *)malloc(sizeof(float) * BATCH_SIZE * MAX_SEQ_LEN);
    if (!text_input) {
      throw std::runtime_error("Failed to allocate memory for text input.");
    }

    // Prepare input buffer with token IDs
    for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
      for (unsigned int i = 0; i < input_len; ++i) {
        text_input[b * MAX_SEQ_LEN + i] = static_cast<float>(input_ids[i]);
      }
    }

    std::cout << "\n[Pipeline] ========== Lazy Initialization Flow =========="
              << std::endl;
    std::cout << "[Pipeline] Step 1: Run encoder with " << input_len
              << " tokens" << std::endl;
    std::cout << "[Pipeline] Step 2: Run decoder with encoder output"
              << std::endl;

    // === STEP 1: ENCODER (lazy init) ===
    auto encoder_output = runEncoder(text_input, input_len);

    free(text_input);

    std::cout << "\n[Memory Summary]" << std::endl;
    std::cout << "  Peak memory usage: max(encoder_mem, decoder_mem)"
              << std::endl;
    std::cout << "  (Not encoder_mem + decoder_mem)" << std::endl;
    std::cout << "\n[Pipeline] ========== End of Inference =========="
              << std::endl;

    // TODO : free ENCODER, and initialize decoder here

    // Step 2 - Decoder
    std::cout << runDecoder(encoder_output) << std::endl;
  }

  std::cout << "\n========== End of Inference ==========\n" << std::endl;
}

std::vector<LayerHandle>
T5Gemma2Transformer::createEncoder(const std::string &input_name) {
  std::vector<LayerHandle> layers;

  // Create embedding layer (with scaling as in Gemma3TextScaledWordEmbedding)
  // Embedding scale = sqrt(hidden_size)
  float embed_scale = std::sqrt(ENC_HIDDEN_SIZE);
  layers.push_back(createLayer(
    "embedding_layer", {withKey("name", "encoder_embedding"),
                        withKey("in_dim", std::to_string(NUM_VOCAB)),
                        withKey("out_dim", std::to_string(ENC_HIDDEN_SIZE)),
                        withKey("scale", std::to_string(embed_scale)),
                        withKey("weight_dtype", EMBEDDING_DTYPE),
                        withKey("input_layers", input_name)}));

  std::string residual_checkpoint = "encoder_embedding";

  // Create encoder layers (T5Gemma2 has NUM_LAYERS encoder layers)
  for (int i = 0; i < NUM_LAYERS; i++) {
    // Create attention type based on layer pattern
    // T5Gemma2 uses mix of sliding_attention and full_attention
    bool is_full_attention = ((i + 1) % 6 == 0);

    std::string prefix = "encoder_layer" + std::to_string(i) + "_";

    // Input layernorm
    layers.push_back(createLayer(
      "rms_norm", {withKey("name", prefix + "pre_attention_layernorm"),
                   withKey("epsilon", std::to_string(ENC_NORM_EPS)),
                   withKey("input_layers", residual_checkpoint),
                   withKey("packed", "false")}));

    // Self-attention
    // prefix + "attention_out"
    auto att_layers = createSelfAttention(
      prefix, i, INIT_SEQ_LEN, ENC_NUM_HEADS, ENC_HEAD_DIM, ENC_GQA_SIZE,
      prefix + "pre_attention_layernorm", prefix + "pre_attention_layernorm",
      prefix + "pre_attention_layernorm");
    layers.insert(layers.end(), att_layers.begin(), att_layers.end());

    // Post-attention layernorm
    layers.push_back(createLayer(
      "rms_norm", {withKey("name", prefix + "post_attention_layernorm"),
                   withKey("epsilon", std::to_string(NORM_EPS)),
                   withKey("input_layers", prefix + "attention_out"),
                   withKey("packed", "false")}));

    // Residual connection after attention
    layers.push_back(createLayer(
      "addition",
      {withKey("name", prefix + "attention_residual"),
       withKey("input_layers",
               {residual_checkpoint, prefix + "post_attention_layernorm"})}));

    residual_checkpoint = prefix + "attention_residual";

    // Pre-FFN layernorm
    layers.push_back(createLayer(
      "rms_norm", {withKey("name", prefix + "pre_feedforward_layernorm"),
                   withKey("epsilon", std::to_string(NORM_EPS)),
                   withKey("input_layers", residual_checkpoint),
                   withKey("packed", "false")}));

    // MLP (SwiGLU) - need to modify createMlp for encoder
    auto mlp_layers = createMlp(prefix, ENC_HIDDEN_SIZE, ENC_INTERMEDIATE_SIZE,
                                prefix + "pre_feedforward_layernorm");
    layers.insert(layers.end(), mlp_layers.begin(), mlp_layers.end());

    // Post-FFN layernorm
    layers.push_back(createLayer(
      "rms_norm", {withKey("name", prefix + "post_feedforward_layernorm"),
                   withKey("epsilon", std::to_string(NORM_EPS)),
                   withKey("input_layers", prefix + "ffn_down"),
                   withKey("packed", "false")}));

    // Residual connection after FFN
    layers.push_back(createLayer(
      "addition",
      {withKey("name", prefix + "ffn_residual"),
       withKey("input_layers",
               {residual_checkpoint, prefix + "post_feedforward_layernorm"})}));

    residual_checkpoint = prefix + "ffn_residual";
  }

  // Final normalization layer
  layers.push_back(
    createLayer("rms_norm", {withKey("name", "encoder_output_norm"),
                             withKey("epsilon", std::to_string(NORM_EPS)),
                             withKey("input_layers", residual_checkpoint),
                             withKey("packed", "false")}));

  return layers;
}

bool T5Gemma2Transformer::checkImageInput(const std::string &input_text) {
  // Check if BOI_TOKEN is present in the input text
  // if so, update HAS_IMAGE_INPUT
  HAS_IMAGE_INPUT = (input_text.find(BOI_TOKEN) != std::string::npos);
  return HAS_IMAGE_INPUT;
}

std::vector<unsigned int> T5Gemma2Transformer::generate(float *logits,
                                                        bool do_sample) {
  std::vector<unsigned int> outputs;

  for (unsigned int iteration = 0; iteration < BATCH_SIZE; ++iteration) {
    // Use argmax (do_sample = false) or sampling
    if (do_sample == false) {
      unsigned int argmax_idx =
        std::distance(logits, std::max_element(logits, logits + NUM_VOCAB));
      outputs.push_back(argmax_idx);
    } else {
      // Apply softmax to logits
      float max_logits = *std::max_element(logits, logits + NUM_VOCAB);
      float sum_exp_logits = 0;

      for (unsigned int i = 0; i < NUM_VOCAB; i++) {
        float exp_x = std::exp(logits[i] - max_logits);
        sum_exp_logits += exp_x;
        logits[i] = exp_x;
      }

      // Normalize to get probabilities
      for (unsigned int i = 0; i < NUM_VOCAB; ++i) {
        logits[i] /= sum_exp_logits;
      }

      // Sample from final logits using discrete distribution
      std::discrete_distribution<int> dist(logits, logits + NUM_VOCAB);
      std::mt19937 rng(std::random_device{}());
      unsigned int sampled_idx = dist(rng);

      outputs.push_back(sampled_idx);
    }

    // Move to next batch
    logits = logits + NUM_VOCAB;
  }

  return outputs;
}

// TODO : merge with createEncoder()
void T5Gemma2Transformer::createEncoderModel() {
  if (encoder_compiled) {
    std::cout << "[EncoderModel] Already compiled" << std::endl;
    return;
  }

  encoder_model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  std::vector<LayerHandle> encoder_layers;

  // Input layer
  encoder_layers.push_back(createLayer(
    "input", {withKey("name", "encoder_input"),
              withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));

  // Encoder layers
  auto encoder_block = createEncoder("encoder_input");
  encoder_layers.insert(encoder_layers.end(), encoder_block.begin(),
                        encoder_block.end());

  // Identity layer for debugging
  encoder_layers.push_back(
    createLayer("identity", {withKey("name", "encoder_output"),
                             withKey("input_layers", "encoder_output_norm")}));

  // Add all layers to model
  for (auto &layer : encoder_layers) {
    encoder_model->addLayer(layer);
  }

  // Model properties
  std::vector<std::string> model_props = {
    withKey("batch_size", "1"), withKey("epochs", "1"),
    withKey("model_tensor_type", MODEL_TENSOR_TYPE)};
  encoder_model->setProperty(model_props);

  // Compile only (no memory allocation)
  if (encoder_model->compile(ml::train::ExecutionMode::INFERENCE)) {
    throw std::runtime_error("Encoder compilation failed.");
  }

  encoder_compiled = true;
  std::cout << "[EncoderModel] Compiled successfully (no memory allocated yet)"
            << std::endl;
}

void T5Gemma2Transformer::createDecoderModel() {
  if (decoder_compiled) {
    std::cout << "[DecoderModel] Already compiled" << std::endl;
    return;
  }

  decoder_model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  std::vector<LayerHandle> decoder_layers;

  // Input layer for decoder tokens
  
  // TODO : SEQ_LEN set to 1 for concat, could need to change for multiple input
  decoder_layers.push_back(createLayer(
    "input", {withKey("name", "decoder_token_input"),
              withKey("input_shape", "1:1:" + std::to_string(1))}));



  // TODO : can we change this to actual expected encoder length? 
  // (this could be calculated before createModel via using prompt)
  decoder_layers.push_back(createLayer(
    "input",
    {withKey("name", "encoder_input"),
     withKey("input_shape", "1:" + std::to_string(INIT_SEQ_LEN) + ":" +std::to_string(ENC_HIDDEN_SIZE))}));

  // Embedding layer
  float embed_scale = std::sqrt(DEC_HIDDEN_SIZE);
  decoder_layers.push_back(createLayer(
    "embedding_layer", {withKey("name", "decoder_embedding"),
                        withKey("in_dim", std::to_string(NUM_VOCAB)),
                        withKey("out_dim", std::to_string(DEC_HIDDEN_SIZE)),
                        withKey("scale", std::to_string(embed_scale)),
                        withKey("weight_dtype", EMBEDDING_DTYPE),
                        withKey("input_layers", "decoder_token_input")}));

  std::string residual_checkpoint = "decoder_embedding";

  // Create decoder layers
  for (int i = 0; i < DEC_NUM_LAYERS; i++) {
    std::string prefix = "decoder_layer" + std::to_string(i) + "_";

    // Input layernorm
    decoder_layers.push_back(createLayer(
      "rms_norm", {withKey("name", prefix + "pre_attention_layernorm"),
                   withKey("epsilon", std::to_string(DEC_NORM_EPS)),
                   withKey("input_layers", residual_checkpoint),
                   withKey("packed", "false")}));

    // merged attention (causal)
    auto att_layers = createMergedAttention(
      prefix, i, MAX_SEQ_LEN, DEC_NUM_HEADS, DEC_HEAD_DIM,
      DEC_NUM_HEADS / DEC_NUM_KEY_VALUE_HEADS,
      prefix + "pre_attention_layernorm", prefix + "pre_attention_layernorm",
      prefix + "pre_attention_layernorm", "encoder_input", "encoder_input");
    decoder_layers.insert(decoder_layers.end(), att_layers.begin(),
                          att_layers.end());

    // Post-attention layernorm
    decoder_layers.push_back(createLayer(
      "rms_norm", {withKey("name", prefix + "post_attention_layernorm"),
                   withKey("epsilon", std::to_string(NORM_EPS)),
                   withKey("input_layers", prefix + "attention_out"),
                   withKey("packed", "false")}));

    // Residual connection after attention
    decoder_layers.push_back(createLayer(
      "addition",
      {withKey("name", prefix + "attention_residual"),
       withKey("input_layers",
               {residual_checkpoint, prefix + "post_attention_layernorm"})}));

    residual_checkpoint = prefix + "attention_residual";

    // Pre-FFN layernorm
    decoder_layers.push_back(createLayer(
      "rms_norm", {withKey("name", prefix + "pre_feedforward_layernorm"),
                   withKey("epsilon", std::to_string(NORM_EPS)),
                   withKey("input_layers", residual_checkpoint),
                   withKey("packed", "false")}));

    // MLP
    auto mlp_layers = createMlp(prefix, DEC_HIDDEN_SIZE, DEC_INTERMEDIATE_SIZE,
                                prefix + "pre_feedforward_layernorm");
    decoder_layers.insert(decoder_layers.end(), mlp_layers.begin(),
                          mlp_layers.end());

    // Post-FFN layernorm
    decoder_layers.push_back(createLayer(
      "rms_norm", {withKey("name", prefix + "post_feedforward_layernorm"),
                   withKey("epsilon", std::to_string(NORM_EPS)),
                   withKey("input_layers", prefix + "ffn_down"),
                   withKey("packed", "false")}));

    // Residual connection after FFN
    decoder_layers.push_back(createLayer(
      "addition",
      {withKey("name", prefix + "ffn_residual"),
       withKey("input_layers",
               {residual_checkpoint, prefix + "post_feedforward_layernorm"})}));

    residual_checkpoint = prefix + "ffn_residual";
  }

  // Final normalization
  decoder_layers.push_back(
    createLayer("rms_norm", {withKey("name", "decoder_output_norm"),
                             withKey("epsilon", std::to_string(DEC_NORM_EPS)),
                             withKey("input_layers", residual_checkpoint),
                             withKey("packed", "false")}));

  // LM head (project to vocabulary)
  decoder_layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "lm_head"), withKey("unit", std::to_string(NUM_VOCAB)),
     withKey("disable_bias", "true"),
     withKey("input_layers", "decoder_output_norm"),
     withKey("weight_initializer", "ones")}));

  // Add all layers to model
  for (auto &layer : decoder_layers) {
    decoder_model->addLayer(layer);
  }

  // Model properties
  std::vector<std::string> model_props = {
    withKey("batch_size", "1"), withKey("epochs", "1"),
    withKey("model_tensor_type", MODEL_TENSOR_TYPE)};
  decoder_model->setProperty(model_props);

  // Compile only (no memory allocation)
  if (decoder_model->compile(ml::train::ExecutionMode::INFERENCE)) {
    throw std::runtime_error("Decoder compilation failed.");
  }

  decoder_compiled = true;
  std::cout << "[DecoderModel] Compiled successfully (no memory allocated yet)"
            << std::endl;
}

std::vector<float> T5Gemma2Transformer::runEncoder(float *input_data,
                                                   unsigned int input_len) {
  if (!encoder_initialized) {
    throw std::runtime_error(
      "Encoder not initialized. Call initialize() first.");
  }

  std::cout << "\n========== Encoder Inference ==========" << std::endl;

  // Memory tracking (getMemorySize may not be available in all versions)
  encoder_memory_size = 0;
  std::cout << "[Encoder] Memory allocated (size tracking not available)"
            << std::endl;

  // Prepare input tensors
  std::vector<float *> input_tensors = {input_data};
  std::vector<float *> label_tensors;


  std::vector<ml::train::TensorDim> input_dims;
  ml::train::TensorDim input_dim(1, 1, input_len, ENC_HIDDEN_SIZE);
  input_dims.push_back(input_dim);
  // encoder_model->resetInputDimension(input_dims);

  // Inference
  auto start_time = std::chrono::high_resolution_clock::now();

  auto encoder_output = encoder_model->incremental_inference(
    1, input_tensors, label_tensors, input_len, 0, input_len, false);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    end_time - start_time);

  // Save encoder output
  size_t output_size = input_len * ENC_HIDDEN_SIZE;
  std::vector<float> saved_output(output_size);
  std::copy(encoder_output[0], encoder_output[0] + output_size,
            saved_output.begin());

  std::cout << "[Encoder] Inference completed in " << duration.count() << " ms"
            << std::endl;
  std::cout << "[Encoder] Output shape: [" << input_len << ", "
            << ENC_HIDDEN_SIZE << "]" << std::endl;
  std::cout << "[Encoder] First 5 values: ";
  for (int i = 0; i < std::min(5, (int)output_size); ++i) {
    std::cout << saved_output[i] << " ";
  }
  std::cout << std::endl;

  // Note: Memory will be automatically deallocated when encoder_model is reset
  // or destroyed For now, we rely on RAII - memory will be freed when the model
  // is recreated or destroyed
  std::cout << "[Encoder] Inference complete (memory held for potential reuse)"
            << std::endl;


              std::cout << "=================[ LLM with NNTrainer ]===================\n";
  std::cout << "prefill: " << input_len << " tokens, "
            << duration.count() << " ms, "
            << ((double)input_len / duration.count() * 1000) << " TPS\n";

  return saved_output;
}

std::string
T5Gemma2Transformer::runDecoder(const std::vector<float> &encoder_output) {
  if (!decoder_initialized) {
    throw std::runtime_error(
      "Decoder not initialized. Call initialize() first.");
  }

  std::cout << "\n========== Decoder Inference ==========" << std::endl;

  // Memory tracking
  decoder_memory_size = 0;
  std::cout << "[Decoder] Memory allocated (size tracking not available)"
            << std::endl;

  // Prepare input
  float *decoder_tokens =
    (float *)malloc(sizeof(float) * (NUM_TO_GENERATE + 1));
  if (!decoder_tokens) {
    throw std::runtime_error("Failed to allocate decoder tokens");
  }

  // Initialize with BOS token
  decoder_tokens[0] = static_cast<float>(BOS_TOKEN_ID);

  // Convert encoder_output to pointer (non-const)
  // Do this to keep original Encoded output Data const
  std::vector<float> encoder_output_mutable(encoder_output.begin(),
                                            encoder_output.end());

  // Decoder inputs: [decoder_tokens, encoder_output]
  // TODO : 지금은 topo sort되기 전이라 뒤집어 있지만 나중에 다시 뒤집어야 함
  std::vector<float *> decoder_inputs = {encoder_output_mutable.data(), decoder_tokens
                                         };
  std::vector<float *> label_tensors;
  std::vector<unsigned int> generated_tokens;

  // Inference
  auto start_time = std::chrono::high_resolution_clock::now();


// TMP code for custom setting from-to in some layers
  std::unordered_map<std::string, unsigned int> custom_to_map;
for(int i=0; i<DEC_NUM_LAYERS; ++i)
{
  std::string prefix = "decoder_layer" + std::to_string(i) + "_";
  auto cross_K = prefix + "cross_wk";
  auto cross_V = prefix + "cross_wv";
  auto cross_K_norm = cross_K + "_norm";
  
  custom_to_map.insert({cross_K,ACTUAL_SEQ_LEN});
  custom_to_map.insert({cross_V,ACTUAL_SEQ_LEN});
  custom_to_map.insert({cross_K_norm,ACTUAL_SEQ_LEN});


}
          
  // Token Generation (no prefill)
  for (unsigned int i = 0; i < NUM_TO_GENERATE; ++i) {
    auto gen_output = decoder_model->incremental_inference(
      1, decoder_inputs, label_tensors, 1, i, i + 1, false, &custom_to_map);

    unsigned int new_token = generate(gen_output[0], false)[0];

    // TODO : EOS나 완료 관련 메커니즘 강화
    // if (new_token == EOS_TOKEN_ID) { // EOS
    //   std::cout << "[Decoder] Reached EOS at position " << i << std::endl;
    //   break;
    // }

    generated_tokens.push_back(new_token);

   
    decoder_tokens[0] = static_cast<float>(new_token);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    end_time - start_time);

  std::cout << "[Decoder] Generated " << generated_tokens.size()
            << " tokens in " << duration.count() << " ms" << std::endl;

  // Decode (convert unsigned int to int)
  std::vector<int> generated_tokens_int(generated_tokens.begin(),
                                        generated_tokens.end());
  std::string decoded_text = tokenizer->Decode(generated_tokens_int);

  // Cleanup
  free(decoder_tokens);

  std::cout << "[Decoder] Inference complete (memory held for potential reuse)"
            << std::endl;

  return decoded_text;
}

void T5Gemma2Transformer::loadEncoderWeights(const std::string &weight_path) {
  if (!encoder_compiled) {
    throw std::runtime_error("Encoder not compiled.");
  }

  // Initialize encoder (allocate memory at this point)
  if (encoder_model->initialize(ml::train::ExecutionMode::INFERENCE)) {
    throw std::runtime_error("Encoder initialization failed.");
  }
  encoder_initialized = true;

  // encoder_model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

  if (encoder_weights_loaded) {
    std::cout << "[EncoderWeights] Already loaded, skipping" << std::endl;
    return;
  }

  std::cout << "\n========== Loading Encoder Weights ==========" << std::endl;

  try {
    auto start_time = std::chrono::high_resolution_clock::now();

    encoder_model->load(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

    encoder_weights_loaded = true;
    encoder_weight_path = weight_path;

    std::cout << "[EncoderWeights] Loaded from: " << weight_path << std::endl;
    std::cout << "[EncoderWeights] Loading time: " << duration.count() << " ms"
              << std::endl;

  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to load encoder weights from " +
                             weight_path + " | Reason: " + e.what());
  }

  std::cout << "=============================================\n" << std::endl;
}

void T5Gemma2Transformer::loadDecoderWeights(const std::string &weight_path) {
  if (!decoder_compiled) {
    throw std::runtime_error("Decoder not compiled.");
  }

  // Initialize decoder (allocate memory at this point)
  if (decoder_model->initialize(ml::train::ExecutionMode::INFERENCE)) {
    throw std::runtime_error("Decoder initialization failed.");
  }

  decoder_initialized = true;

  // decoder_model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

  // std::cout
  //   << "\n========== Loading Decoder Weights Not implemented yet =========="
  //   << std::endl;
  // return;

  if (decoder_weights_loaded) {
    std::cout << "[DecoderWeights] Already loaded, skipping" << std::endl;
    return;
  }

  std::cout << "\n========== Loading Decoder Weights ==========" << std::endl;

  try {
    auto start_time = std::chrono::high_resolution_clock::now();

    decoder_model->load(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

    decoder_weights_loaded = true;
    decoder_weight_path = weight_path;

    std::cout << "[DecoderWeights] Loaded from: " << weight_path << std::endl;
    std::cout << "[DecoderWeights] Loading time: " << duration.count() << " ms"
              << std::endl;

  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to load decoder weights from " +
                             weight_path + " | Reason: " + e.what());
  }

  std::cout << "=============================================\n" << std::endl;
}

void T5Gemma2Transformer::load_weight(const std::string &weight_path) {
  if (!is_initialized) {
    throw std::runtime_error("T5Gemma2 model is not initialized. Please call "
                             "initialize() before load_weight().");
  }

  std::cout << "\n========== Loading T5Gemma2 Weights (Lazy) =========="
            << std::endl;

  // Parse weight path to extract encoder/decoder paths
  // Expected format: <path>/model_encoder.bin and <path>/model_decoder.bin
  // or just one path for both models

  std::string base_path = weight_path;
  std::string encoder_path = weight_path;
  std::string decoder_path = weight_path;

  // Check if the path indicates separate encoder/decoder files
  // Try to detect patterns like "encoder.bin" or "_encoder.bin"
  size_t encoder_pos = weight_path.find("encoder");
  size_t decoder_pos = weight_path.find("decoder");

  if (encoder_pos != std::string::npos) {
    // The path specifies encoder weights
    encoder_path = weight_path;
    // Derive decoder path
    decoder_path = weight_path;
    decoder_path.replace(encoder_pos, 7, "decoder");
    std::cout << "[LoadWeight] Detected separate encoder/decoder files"
              << std::endl;
  } else if (decoder_pos != std::string::npos) {
    // The path specifies decoder weights
    decoder_path = weight_path;
    // Derive encoder path
    encoder_path = weight_path;
    encoder_path.replace(decoder_pos, 7, "encoder");
    std::cout << "[LoadWeight] Detected separate decoder/encoder files"
              << std::endl;
  } else {
    // Try to construct separate paths from the base path
    // Look for the last extension
    size_t dot_pos = weight_path.find_last_of('.');
    if (dot_pos != std::string::npos) {
      std::string ext = weight_path.substr(dot_pos);
      std::string base = weight_path.substr(0, dot_pos);
      encoder_path = base + "_encoder" + ext;
      decoder_path = base + "_decoder" + ext;
      std::cout << "[LoadWeight] Constructed separate paths:" << std::endl;
      std::cout << "  Encoder: " << encoder_path << std::endl;
      std::cout << "  Decoder: " << decoder_path << std::endl;
    }
  }

  // Load encoder weights first (lower memory footprint)
  std::cout << "\n[LoadWeight] Step 1: Loading encoder weights..." << std::endl;
  loadEncoderWeights(encoder_path);

  // Load decoder weights
  std::cout << "\n[LoadWeight] Step 2: Loading decoder weights..." << std::endl;
  loadDecoderWeights(decoder_path);

  std::cout
    << "\n[LoadWeight] ========== All weights loaded successfully =========="
    << std::endl;
  std::cout << "[LoadWeight] Memory optimized: Only one model's weights in "
               "memory at a time during loading"
            << std::endl;
  std::cout
    << "====================================================================\n"
    << std::endl;
}

} // namespace causallm
