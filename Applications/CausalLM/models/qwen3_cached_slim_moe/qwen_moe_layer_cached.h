// SPDX-License-Identifier: Apache-2.0
// clang-format off
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   qwen_moe_layer_cached.h
 * @date   09 June 2025
 * @brief  Cached slim Mixture-of-Experts layer for Qwen3 MoE.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This file is part of the Mixture-of-Experts implementation.
 *         It does not support shared experts.
 *         This layer is implemented based on LLaMA-MoE.
 *         For more information, please refer to:
 *         https://arxiv.org/pdf/2406.16554
 * @todo   This layer does not support backward propagation yet.
 */
// clang-format on

#ifndef __QWEN3_CACHED_SLIM_MOE_LAYER_H__
#define __QWEN3_CACHED_SLIM_MOE_LAYER_H__
#ifdef __cplusplus

#pragma once
#ifndef WIN_EXPORT
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif
#endif

#include <acti_func.h>
#include <causallm_common_properties.h>
#include <common_properties.h>
#include <layer_impl.h>
#include <list>

namespace causallm {

/**
 * @class   CachedSlimMoELayer
 * @brief   Cached slim MoE layer.
 */
class WIN_EXPORT CachedSlimMoELayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of cached slim Mixture-of-Experts layer
   */
  CachedSlimMoELayer();

  /**
   * @brief     Destructor of cached slim Mixture-of-Experts layer
   */
  ~CachedSlimMoELayer() = default;

  /**
   * @brief  Move constructor.
   *  @param[in] CachedSlimMoELayer &&
   */
  CachedSlimMoELayer(CachedSlimMoELayer &&rhs) = delete;

  /**
   * @brief  Move assignment operator.
   * @param[in] rhs CachedSlimMoELayer to be moved.
   */
  CachedSlimMoELayer &operator=(CachedSlimMoELayer &&rhs) = delete;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned)
   */
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, const ml::train::ExportMethods
   * &methods)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return CachedSlimMoELayer::type;
  };

  /**
   * @brief Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  static constexpr const char *type =
    "moe_cached_slim"; /**< type of the layer */

private:
  unsigned int num_experts;      /**< number of experts */
  unsigned int topk;             /**< number of experts per token, i.e., topk */
  nntrainer::ActiFunc acti_func; /**< activation function for the expert */
  std::tuple<props::NumExperts, props::NumExpertsPerToken,
             nntrainer::props::Unit, props::MoEActivation>
    moe_props;

  // weight indices
  std::vector<unsigned int> expert_gate_proj_indices;
  std::vector<unsigned int> expert_up_proj_indices;
  std::vector<unsigned int> expert_down_proj_indices;

  std::list<int> loaded_expert_deque;
  std::unordered_map<int, std::list<int>::iterator> iteration_map;
  std::unordered_map<int, double> expert_predict_scores;
  std::vector<bool> need_load;
  std::mutex cache_mutex;

  unsigned int gate_idx;

  // Intermediate tensor indices
  unsigned int router_logits_idx;
  unsigned int expert_mask_idx;
  /// Expert forward computation without memory copies.
  inline void compute_expert_forward(
    const nntrainer::Tensor &input, nntrainer::Tensor &output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size);
};
} // namespace causallm

#endif /* __cplusplus */
#endif /* __QWEN3_CACHED_SLIM_MOE_LAYER_H__ */
