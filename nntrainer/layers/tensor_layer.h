// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   tensor_layer.h
 * @date   17 Jan 2025
 * @brief  Tensor layer
 * @see    https://github.com/nntrainer/nntrainer
 *
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __TENSOR_LAYER_H__
#define __TENSOR_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @class   TensorLayer
 * @brief   Tensor forwarding layer
 */
class TensorLayer : public Layer {
public:
  /**
   * @brief     Constructor of TensorLayer
   */
  TensorLayer();

  /**
   * @brief     Destructor of TensorLayer
   */
  ~TensorLayer() = default;

  /**
   *  @brief  Move constructor of TensorLayer.
   *  @param[in] rhs
   */
  TensorLayer(TensorLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @param[in] rhs TensorLayer to be moved.
   */
  TensorLayer &operator=(TensorLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return false; };

  /**
   * @brief Initialize the in-place settings of the layer
   * @return InPlaceType
   */
  InPlaceType initializeInPlace() final {
    is_inplace = true;
    return InPlaceType::NON_RESTRICTING;
  }

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return TensorLayer::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "tensor";

private:
  std::tuple<std::vector<props::TensorDimension>,
             std::vector<props::TensorDataType>, std::vector<props::TensorName>,
             std::vector<props::TensorLife>>
    tensor_props;

  std::vector<unsigned int> tensor_idx;
  unsigned int n_tensor;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __INPUT_LAYER_H__ */
