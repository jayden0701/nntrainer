// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   encoder_decoder.h
 * @date   15 Mar 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines a base class for encoder-decoder transformer
 *         models.
 */

#ifndef __ENCODER_DECODER_H__
#define __ENCODER_DECODER_H__

#include <transformer.h>

namespace causallm {

/**
 * @brief EncoderDecoder Class
 */
class EncoderDecoder : virtual public Transformer {
public:
  EncoderDecoder(json &cfg, json &generation_cfg, json &nntr_cfg,
                 ModelType model_type = ModelType::MODEL) :
    Transformer(cfg, generation_cfg, nntr_cfg, model_type) {}

  virtual ~EncoderDecoder() = default;
};

} // namespace causallm

#endif // __ENCODER_DECODER_H__

