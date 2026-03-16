// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   encoder_decoder.cpp
 * @date   15 Mar 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines shared behavior for encoder-decoder transformer
 *         models.
 */

#include <encoder_decoder.h>

namespace causallm {

EncoderDecoder::EncoderDecoder(json &cfg, json &generation_cfg, json &nntr_cfg,
                               ModelType model_type) :
  Transformer(cfg, generation_cfg, nntr_cfg, model_type) {}

EncoderDecoder::~EncoderDecoder() = default;

} // namespace causallm
