// SPDX-License-Identifier: Apache-2.0

#include "sequence_classification_bert.h"

#include <app_context.h>
#include <codecvt>
#include <embedding_pooling_layer.h>
#include <engine.h>
#include <llm_util.hpp>
#include <locale>
#include <stdexcept>

namespace causallm {

SequenceClassificationBert::SequenceClassificationBert(json &cfg,
                                                       json &generation_cfg,
                                                       json &nntr_cfg) :
  Transformer(sanitizeConfig(cfg), generation_cfg, nntr_cfg,
              ModelType::EMBEDDING),
  BertTransformer(cfg, generation_cfg, nntr_cfg) {
  NUM_CLASSES = cfg.contains("num_classes")
                  ? cfg["num_classes"].get<unsigned int>()
                  : (cfg.contains("num_labels")
                       ? cfg["num_labels"].get<unsigned int>()
                       : 2);
  HIDDEN_DROPOUT_PROB = cfg.contains("hidden_dropout_prob")
                          ? cfg["hidden_dropout_prob"].get<float>()
                          : 0.1f;
}

void SequenceClassificationBert::constructModel() {
  BertTransformer::constructModel();

  model->addLayer(createLayer(
    "embedding_pooling",
    {withKey("name", "cls_pooling"), withKey("word_embedding_dimension", DIM),
     withKey("pooling_mode_cls_token", "true"),
     withKey("input_layers", "layer" + std::to_string(NUM_LAYERS - 1) +
                               "_ffn_norm")}));

  model->addLayer(createLayer(
    "fully_connected",
    {withKey("name", "bert_pooler"), withKey("unit", DIM),
     withKey("disable_bias", "false"), withKey("input_layers", "cls_pooling")}));

  model->addLayer(createLayer("activation", {withKey("name", "bert_pooler_act"),
                                               withKey("activation", "tanh"),
                                               withKey("input_layers", "bert_pooler")}));

  model->addLayer(createLayer(
    "dropout", {withKey("name", "bert_classifier_dropout"),
                 withKey("rate", HIDDEN_DROPOUT_PROB),
                 withKey("input_layers", "bert_pooler_act")}));

  model->addLayer(createLayer(
    "fully_connected",
    {withKey("name", "output_of_classifier"), withKey("unit", NUM_CLASSES),
     withKey("disable_bias", "false"),
     withKey("input_layers", "bert_classifier_dropout")}));
}

void SequenceClassificationBert::registerCustomLayers() {
  BertTransformer::registerCustomLayers();

  const auto &ct_engine = nntrainer::Engine::Global();
  const auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::EmbeddingPoolingLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

std::vector<float *> SequenceClassificationBert::encode(
  const WSTR prompt, const WSTR system_prompt, const WSTR tail_prompt) {
  if (!is_initialized) {
    throw std::runtime_error(
      "SequenceClassificationBert is not initialized. Please call initialize() "
      "before encode().");
  }

#if defined(_WIN32)
  std::wstring prompt_ = system_prompt + prompt + tail_prompt;
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  auto tokenized = tokenizer->Encode(converter.to_bytes(prompt_), true);
#else
  std::string prompt_ = system_prompt + prompt + tail_prompt;
  auto tokenized = tokenizer->Encode(prompt_, true);
#endif

  unsigned int input_len =
    std::min(static_cast<unsigned int>(tokenized.size()), INIT_SEQ_LEN);

  float *input_sample =
    (float *)malloc(sizeof(float) * BATCH_SIZE * INIT_SEQ_LEN);
  float *position_ids =
    (float *)malloc(sizeof(float) * BATCH_SIZE * INIT_SEQ_LEN);
  float *token_type_ids =
    (float *)malloc(sizeof(float) * BATCH_SIZE * INIT_SEQ_LEN);

  if (!input_sample || !position_ids || !token_type_ids) {
    free(input_sample);
    free(position_ids);
    free(token_type_ids);
    throw std::runtime_error("Failed to allocate input buffers");
  }

  std::fill(input_sample, input_sample + BATCH_SIZE * INIT_SEQ_LEN, 0.0f);
  std::fill(position_ids, position_ids + BATCH_SIZE * INIT_SEQ_LEN, 0.0f);
  std::fill(token_type_ids, token_type_ids + BATCH_SIZE * INIT_SEQ_LEN, 0.0f);

  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    for (unsigned int i = 0; i < input_len; ++i) {
      input_sample[static_cast<size_t>(b) * INIT_SEQ_LEN + i] =
        static_cast<float>(tokenized[i]);
      position_ids[static_cast<size_t>(b) * INIT_SEQ_LEN + i] =
        static_cast<float>(i);
    }
  }

  std::vector<float *> input = {input_sample, position_ids, token_type_ids};
  std::vector<float *> label;

  auto output = model->incremental_inference(BATCH_SIZE, input, label, input_len,
                                             0, input_len, false);

  free(input_sample);
  free(position_ids);
  free(token_type_ids);

  return output;
}

} // namespace causallm
