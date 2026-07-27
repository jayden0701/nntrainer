// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   iotensor_wrapper.hpp
 * @brief  Wrapper around QNN IO tensor setup utilities
 * @author Joonseok Oh <jrock.oh@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#ifndef __QNN_IOTENSOR_WRAPPER_H__
#define __QNN_IOTENSOR_WRAPPER_H__

#include "IOTensor.hpp"
#include "QnnSampleAppUtils.hpp"
#include "QnnWrapperUtils.hpp"

#include <QnnTypeMacros.hpp>
#include <nntrainer_log.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>

namespace nntrainer {

/** @brief Wraps QNN IO tensor allocation and teardown without copying data. */
class IOTensorWrapper {
public:
  /** Owns copied host descriptor metadata, not RPC backing or memHandles. */
  struct TensorDeleter {
    uint32_t tensor_count{0};

    void operator()(Qnn_Tensor_t *tensors) const noexcept {
      if (tensors != nullptr) {
        qnn_wrapper_api::freeQnnTensors(tensors, tensor_count);
      }
    }
  };

  using TensorOwner = std::unique_ptr<Qnn_Tensor_t[], TensorDeleter>;

  static TensorOwner makeTensorOwner(uint32_t tensor_count) {
    return TensorOwner(nullptr, TensorDeleter{tensor_count});
  }

  qnn::tools::iotensor::StatusCode
  setupInputAndOutputTensors(TensorOwner &inputs, TensorOwner &outputs,
                             const qnn_wrapper_api::GraphInfo_t &graph_info) {
    inputs = makeTensorOwner(graph_info.numInputTensors);
    outputs = makeTensorOwner(graph_info.numOutputTensors);

    auto status = setupTensorsNoCopy(inputs, graph_info.numInputTensors,
                                     graph_info.inputTensors);
    if (status != qnn::tools::iotensor::StatusCode::SUCCESS) {
      ml_loge("Failure in setting up input tensors");
      return status;
    }

    status = setupTensorsNoCopy(outputs, graph_info.numOutputTensors,
                                graph_info.outputTensors);
    if (status != qnn::tools::iotensor::StatusCode::SUCCESS) {
      ml_loge("Failure in setting up output tensors");
    }
    return status;
  }

private:
  static bool hasCompleteV1Metadata(const Qnn_Tensor_t &tensor) noexcept {
    if (tensor.version != QNN_TENSOR_VERSION_1 || tensor.v1.name == nullptr ||
        tensor.v1.name[0] == '\0' ||
        (tensor.v1.rank > 0 && tensor.v1.dimensions == nullptr) ||
        tensor.v1.rank >
          std::numeric_limits<size_t>::max() / sizeof(uint32_t)) {
      return false;
    }

    size_t element_count = 1;
    for (uint32_t axis = 0; axis < tensor.v1.rank; ++axis) {
      const auto dimension = tensor.v1.dimensions[axis];
      if (dimension == 0 ||
          element_count > std::numeric_limits<size_t>::max() / dimension) {
        return false;
      }
      element_count *= dimension;
    }

    const auto &quant_params = tensor.v1.quantizeParams;
    switch (quant_params.quantizationEncoding) {
    case QNN_QUANTIZATION_ENCODING_UNDEFINED:
      return true;
    case QNN_QUANTIZATION_ENCODING_SCALE_OFFSET:
      return std::isfinite(quant_params.scaleOffsetEncoding.scale) &&
             quant_params.scaleOffsetEncoding.scale > 0.0f;
    case QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET: {
      const auto &axis_params = quant_params.axisScaleOffsetEncoding;
      if (axis_params.axis >= tensor.v1.rank ||
          axis_params.numScaleOffsets == 0 ||
          axis_params.numScaleOffsets !=
            tensor.v1.dimensions[axis_params.axis] ||
          axis_params.numScaleOffsets >
            std::numeric_limits<size_t>::max() / sizeof(Qnn_ScaleOffset_t) ||
          axis_params.scaleOffset == nullptr) {
        return false;
      }
      for (uint32_t index = 0; index < axis_params.numScaleOffsets; ++index) {
        const auto scale = axis_params.scaleOffset[index].scale;
        if (!std::isfinite(scale) || scale <= 0.0f) {
          return false;
        }
      }
      return true;
    }
    default:
      return false;
    }
  }

  static bool hasMatchingCopiedMetadata(const Qnn_Tensor_t &source,
                                        const Qnn_Tensor_t &copy) noexcept {
    if (!hasCompleteV1Metadata(copy) || copy.v1.id != source.v1.id ||
        std::strcmp(copy.v1.name, source.v1.name) != 0 ||
        copy.v1.type != source.v1.type ||
        copy.v1.dataFormat != source.v1.dataFormat ||
        copy.v1.dataType != source.v1.dataType ||
        copy.v1.rank != source.v1.rank ||
        copy.v1.quantizeParams.encodingDefinition !=
          source.v1.quantizeParams.encodingDefinition ||
        copy.v1.quantizeParams.quantizationEncoding !=
          source.v1.quantizeParams.quantizationEncoding) {
      return false;
    }

    for (uint32_t axis = 0; axis < source.v1.rank; ++axis) {
      if (copy.v1.dimensions[axis] != source.v1.dimensions[axis]) {
        return false;
      }
    }

    if (source.v1.quantizeParams.quantizationEncoding ==
        QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
      return copy.v1.quantizeParams.scaleOffsetEncoding.scale ==
               source.v1.quantizeParams.scaleOffsetEncoding.scale &&
             copy.v1.quantizeParams.scaleOffsetEncoding.offset ==
               source.v1.quantizeParams.scaleOffsetEncoding.offset;
    }
    if (source.v1.quantizeParams.quantizationEncoding ==
        QNN_QUANTIZATION_ENCODING_UNDEFINED) {
      return true;
    }

    const auto &source_axis = source.v1.quantizeParams.axisScaleOffsetEncoding;
    const auto &copy_axis = copy.v1.quantizeParams.axisScaleOffsetEncoding;
    if (copy_axis.axis != source_axis.axis ||
        copy_axis.numScaleOffsets != source_axis.numScaleOffsets) {
      return false;
    }
    for (uint32_t index = 0; index < source_axis.numScaleOffsets; ++index) {
      if (copy_axis.scaleOffset[index].scale !=
            source_axis.scaleOffset[index].scale ||
          copy_axis.scaleOffset[index].offset !=
            source_axis.scaleOffset[index].offset) {
        return false;
      }
    }
    return true;
  }

  qnn::tools::iotensor::StatusCode
  setupTensorsNoCopy(TensorOwner &tensors, uint32_t tensor_count,
                     const Qnn_Tensor_t *tensor_wrappers) {
    tensors.reset();
    if (tensor_count == 0) {
      QNN_INFO("tensor count is 0. Nothing to setup.");
      return qnn::tools::iotensor::StatusCode::SUCCESS;
    }
    if (tensor_wrappers == nullptr) {
      ml_loge("tensorWrappers is nullptr");
      return qnn::tools::iotensor::StatusCode::FAILURE;
    }
    if (tensor_count >
        std::numeric_limits<size_t>::max() / sizeof(Qnn_Tensor_t)) {
      ml_loge("QNN tensor descriptor count is too large");
      return qnn::tools::iotensor::StatusCode::FAILURE;
    }

    auto *raw_tensors = static_cast<Qnn_Tensor_t *>(
      std::calloc(tensor_count, sizeof(Qnn_Tensor_t)));
    if (raw_tensors == nullptr) {
      ml_loge("Memory allocation failed for QNN tensor descriptors");
      return qnn::tools::iotensor::StatusCode::FAILURE;
    }
    tensors.reset(raw_tensors);

    for (uint32_t tensor_index = 0; tensor_index < tensor_count;
         ++tensor_index) {
      tensors[tensor_index] = QNN_TENSOR_INIT;
    }

    for (uint32_t tensor_index = 0; tensor_index < tensor_count;
         ++tensor_index) {
      const auto &source = tensor_wrappers[tensor_index];
      if (!hasCompleteV1Metadata(source)) {
        ml_loge("Invalid or unsupported QNN tensor metadata at index %u",
                tensor_index);
        return qnn::tools::iotensor::StatusCode::FAILURE;
      }
      if (!qnn::tools::sample_app::deepCopyQnnTensorInfo(
            tensors.get() + tensor_index, &source) ||
          !hasMatchingCopiedMetadata(source, tensors[tensor_index])) {
        ml_loge("Failed to copy complete QNN tensor metadata at index %u",
                tensor_index);
        return qnn::tools::iotensor::StatusCode::FAILURE;
      }
      QNN_TENSOR_SET_MEM_TYPE(tensors.get() + tensor_index,
                              QNN_TENSORMEMTYPE_MEMHANDLE);
    }
    return qnn::tools::iotensor::StatusCode::SUCCESS;
  }
};

} // namespace nntrainer

#endif
