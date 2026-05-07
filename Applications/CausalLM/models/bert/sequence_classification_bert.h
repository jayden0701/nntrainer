// SPDX-License-Identifier: Apache-2.0
#ifndef __SEQUENCE_CLASSIFICATION_BERT_H__
#define __SEQUENCE_CLASSIFICATION_BERT_H__

#include <bert_transformer.h>

namespace causallm {

class SequenceClassificationBert : public BertTransformer {
public:
  static constexpr const char *architectures = "BertForSequenceClassification";

  SequenceClassificationBert(json &cfg, json &generation_cfg, json &nntr_cfg);
  ~SequenceClassificationBert() override = default;

  void constructModel() override;
  void registerCustomLayers() override;
  std::vector<float *> encode(const WSTR prompt, const WSTR system_prompt = "",
                              const WSTR tail_prompt = "") override;

private:
  unsigned int NUM_CLASSES = 2;
  float HIDDEN_DROPOUT_PROB = 0.1f;
};

} // namespace causallm

#endif
