// SPDX-License-Identifier: Apache-2.0
// clang-format off
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file constant_derivative_loss_layer.h
 * @date 05 Oct 2021
 * @brief Constant derivative loss implementation
 * @note Feeds an arbitrary derivative value to the last layer.
 * @see https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
// clang-format on
#ifndef __CONSTANT_DERIVATIVE_LOSS_LAYER_H__
#define __CONSTANT_DERIVATIVE_LOSS_LAYER_H__
#ifdef __cplusplus

#include <loss_layer.h>

namespace nntrainer {

/**
 * @class   ConstantDerivativeLossLayer
 * @brief   Constant derivative loss layer
 */
class ConstantDerivativeLossLayer final : public LossLayer {
public:
  /**
   * @brief     Constructor of ConstantDerivativeLossLayer
   */
  ConstantDerivativeLossLayer();

  /**
   * @brief     Destructor of ConstantDerivativeLossLayer
   */
  ~ConstantDerivativeLossLayer();

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return ConstantDerivativeLossLayer::type;
  };

  static constexpr const char *type = "constant_derivative";
};
} // namespace nntrainer

#endif /* __cplusplus */

#endif // __CONSTANT_DERIVATIVE_LOSS_LAYER_H__
