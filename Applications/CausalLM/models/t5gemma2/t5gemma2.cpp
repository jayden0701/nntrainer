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
#include <factory.h>
#include <llm_util.hpp>

namespace causallm {

void T5Gemma2Transformer::setupParameters(json &cfg, json &generation_cfg,
                                         json &nntr_cfg) {
  BATCH_SIZE = nntr_cfg.value("batch_size", 1);
  MODEL_TENSOR_TYPE = nntr_cfg.value("model_tensor_type", "FP32-FP32");
  EMBEDDING_DTYPE = nntr_cfg.value("embedding_dtype", "FP32");
  FC_LAYER_DTYPE = nntr_cfg.value("fc_layer_dtype", "FP32");

  // T5Gemma2 has nested config structure - access decoder config for text model parameters
  NUM_VOCAB = cfg.value("vocab_size", 1000);
  
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
    if (decoder_cfg.contains("sliding_window") && !decoder_cfg["sliding_window"].is_null()) {
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

  // Encoder configuration (vision model)
  if (cfg.contains("encoder") && cfg["encoder"].contains("vision_config")) {
    json &vision_cfg = cfg["encoder"]["vision_config"];
    IMG_SIZE = vision_cfg.value("image_size", 224);
    PATCH_SIZE = vision_cfg.value("patch_size", 14);
    IMG_CHANNELS = vision_cfg.value("num_channels", 3);
    
    // Calculate number of patches: (IMG_SIZE / PATCH_SIZE) ^ 2
    NUM_PATCHES = (IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE);
  } else {
    // Fallback to top-level config for compatibility
    IMG_SIZE = cfg.value("img_size", 224);
    PATCH_SIZE = cfg.value("patch_size", 16);
    NUM_PATCHES = cfg.value("num_patches", 196);
    IMG_CHANNELS = 3;
  }
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

std::vector<LayerHandle>
T5Gemma2Transformer::createAttention(const int layer_id,
                                     const std::string &input_name) {
  std::vector<LayerHandle> layers;

  std::string prefix = "layer" + std::to_string(layer_id) + "_";

  layers.push_back(createLayer("layer_normalization",
                               {withKey("name", prefix + "attention_norm"),
                                withKey("axis", "3"),
                                withKey("epsilon", std::to_string(NORM_EPS)),
                                withKey("input_layers", input_name)}));

  auto q = prefix + "qkv_q", k = prefix + "qkv_k", v = prefix + "qkv_v",
       a = prefix + "attention", o = prefix + "attention_out";

  layers.push_back(
    createLayer("fully_connected",
                {withKey("name", q), withKey("unit", std::to_string(DIM)),
                 withKey("disable_bias", "false"),
                 withKey("input_layers", prefix + "attention_norm")}));

  layers.push_back(
    createLayer("fully_connected",
                {withKey("name", k), withKey("unit", std::to_string(DIM)),
                 withKey("disable_bias", "false"),
                 withKey("input_layers", prefix + "attention_norm")}));

  layers.push_back(
    createLayer("fully_connected",
                {withKey("name", v), withKey("unit", std::to_string(DIM)),
                 withKey("disable_bias", "false"),
                 withKey("input_layers", prefix + "attention_norm")}));

  layers.push_back(createLayer(
    "mha_core",
    {withKey("name", a), withKey("num_heads", std::to_string(NUM_HEADS)),
     withKey("num_heads_kv", std::to_string(NUM_HEADS)),
     withKey("max_timestep", std::to_string(NUM_PATCHES + 1)),
     withKey("is_causal", "false"), withKey("use_rope", "false"),
     withKey("input_layers", {q, k, v})}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", o), withKey("unit", std::to_string(DIM)),
     withKey("disable_bias", "false"), withKey("input_layers", a)}));

  return layers;
}

std::vector<LayerHandle>
T5Gemma2Transformer::createMlp(const int layer_id,
                               const std::string &input_name) {
  std::vector<LayerHandle> layers;

  std::string prefix = "layer" + std::to_string(layer_id) + "_";

  layers.push_back(
    createLayer("layer_normalization",
                {withKey("name", prefix + "ffn_norm"), withKey("axis", "3"),
                 withKey("epsilon", std::to_string(NORM_EPS)),
                 withKey("input_layers", input_name)}));

  layers.push_back(createLayer(
    "fully_connected", {withKey("name", prefix + "ffn_up"),
                        withKey("unit", std::to_string(INTERMEDIATE_SIZE)),
                        withKey("disable_bias", "false"),
                        withKey("input_layers", prefix + "ffn_norm")}));

  layers.push_back(
    createLayer("activation", {withKey("name", prefix + "ffn_gelu"),
                               withKey("activation", "gelu"),
                               withKey("input_layers", prefix + "ffn_up")}));

  layers.push_back(createLayer("fully_connected",
                               {withKey("name", prefix + "ffn_down"),
                                withKey("unit", std::to_string(DIM)),
                                withKey("disable_bias", "false"),
                                withKey("input_layers", prefix + "ffn_gelu")}));

  return layers;
}

std::vector<LayerHandle>
T5Gemma2Transformer::createTransformerDecoderBlock(const int layer_id,
                                                   std::string input_name) {
  std::vector<LayerHandle> layers;

  std::string prefix = "layer" + std::to_string(layer_id) + "_";

  auto attention = createAttention(layer_id, input_name);
  layers.insert(layers.end(), attention.begin(), attention.end());

  layers.push_back(createLayer(
    "addition",
    {withKey("name", prefix + "attention_residual"),
     withKey("input_layers", {input_name, prefix + "attention_out"})}));

  auto mlp = createMlp(layer_id, prefix + "attention_residual");
  layers.insert(layers.end(), mlp.begin(), mlp.end());

  layers.push_back(createLayer(
    "addition", {withKey("name", prefix + "ffn_residual"),
                 withKey("input_layers", {prefix + "attention_residual",
                                          prefix + "ffn_down"})}));

  return layers;
}

void T5Gemma2Transformer::constructModel() {
  std::vector<LayerHandle> layers;

  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  auto patch_embed_layers = createPatchEmbed();
  layers.insert(layers.end(), patch_embed_layers.begin(),
                patch_embed_layers.end());

  std::string last_output = "pos_embed/add";
  for (int i = 0; i < NUM_LAYERS; i++) {
    auto block_layers = createTransformerDecoderBlock(i, last_output);
    layers.insert(layers.end(), block_layers.begin(), block_layers.end());
    last_output = "layer" + std::to_string(i) + "_ffn_residual";
  }

  layers.push_back(createLayer(
    "layer_normalization",
    {withKey("name", "output_norm"), withKey("axis", "3"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("input_layers",
             "layer" + std::to_string(NUM_LAYERS - 1) + "_ffn_residual")}));

  for (auto &layer : layers) {
    model->addLayer(layer);
  }
}

void T5Gemma2Transformer::initialize() {
  registerCustomLayers();

  constructModel();

  std::vector<std::string> model_props = {
    withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
    withKey("model_tensor_type", MODEL_TENSOR_TYPE)};

  model->setProperty(model_props);

  if (model->compile(ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("Model compilation failed.");
  }

  if (model->initialize(ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("Model initialization failed.");
  }

  is_initialized = true;

  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);
}

void T5Gemma2Transformer::registerCustomLayers() {
  Transformer::registerCustomLayers();
}

void T5Gemma2Transformer::run(const WSTR prompt, bool do_sample,
                              const WSTR system_prompt, const WSTR tail_prompt) {

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

  // Create processor instance
  // TODO : maybe change this to init
  nntrainer::T5Gemma2Processor processor(256, 256000);

  // Process input prompt (extracts text and images)
  auto processor_output = processor.process(input_prompt);

  std::cout << "[T5Gemma2Transformer] Processed input:" << std::endl;
  std::cout << "  input_ids length: " << processor_output.input_ids.size() << std::endl;
  std::cout << "  pixel_values size: " << processor_output.pixel_values.size() << std::endl;

  // Prepare model input
  std::vector<float *> input;

  if (!processor_output.pixel_values.empty()) {
    // Image input: use pixel_values from processor
    float *input_sample = (float *)malloc(sizeof(float) * processor_output.pixel_values.size());
    if (!input_sample) {
      throw std::runtime_error("Failed to allocate memory for input_sample.");
    }

    std::copy(processor_output.pixel_values.begin(), processor_output.pixel_values.end(), input_sample);
    input.push_back(input_sample);

    std::vector<float *> label;
    std::vector<float *> output = model->incremental_inference(
      BATCH_SIZE, input, label, NUM_PATCHES, 0, NUM_PATCHES, false);

    std::cout << "First 10 output values: ";
    for (int i = 0; i < std::min(10, DIM); ++i) {
      std::cout << "[" << i << "]=" << output[0][i] << " ";
    }
    std::cout << std::endl;

    free(input_sample);
  } else {
    // Text-only input
    std::cout << "[T5Gemma2Transformer] Text-only input - processing not yet implemented" << std::endl;
    // TODO: Implement text-only inference path
  }
}

bool T5Gemma2Transformer::checkImageInput(const std::string &input_text) {
  // Check if BOI_TOKEN is present in the input text
  // if so, update HAS_IMAGE_INPUT
  HAS_IMAGE_INPUT = (input_text.find(BOI_TOKEN) != std::string::npos);
  return HAS_IMAGE_INPUT;
}

} // namespace causallm
