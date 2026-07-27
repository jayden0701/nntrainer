// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   QNNGraph.cpp
 * @date   10 Jan 2025
 * @brief  This is QNN Graph Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include "QNNGraph.h"
#include "QnnTypes.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <inttypes.h>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <common_properties.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>

namespace nntrainer {

namespace {

std::shared_ptr<QNNVar> getQNNVar(RunLayerContext &context) {
  auto backend =
    std::static_pointer_cast<QNNBackendVar>(context.getContextData());
  NNTR_THROW_IF(!backend, std::runtime_error)
    << "QNN layer context has no backend data";

  auto qnn_data = backend->getVar();
  NNTR_THROW_IF(!qnn_data, std::runtime_error)
    << "QNN backend data is unavailable";
  return qnn_data;
}

bool isCompatibleQnnDataType(Tdatatype tensor_type,
                             Qnn_DataType_t qnn_type) noexcept {
  switch (tensor_type) {
  case Tdatatype::UINT8:
    return qnn_type == QNN_DATATYPE_UINT_8 ||
           qnn_type == QNN_DATATYPE_UFIXED_POINT_8;
  case Tdatatype::UINT16:
    return qnn_type == QNN_DATATYPE_UINT_16 ||
           qnn_type == QNN_DATATYPE_UFIXED_POINT_16;
  case Tdatatype::FP16:
    return qnn_type == QNN_DATATYPE_FLOAT_16;
  case Tdatatype::FP32:
    return qnn_type == QNN_DATATYPE_FLOAT_32;
  default:
    return false;
  }
}

size_t getQnnTensorElementCount(const Qnn_Tensor_t &qnn_tensor,
                                const char *kind, size_t index) {
  size_t element_count = 1;
  for (uint32_t axis = 0; axis < qnn_tensor.v1.rank; ++axis) {
    const auto dimension = qnn_tensor.v1.dimensions[axis];
    NNTR_THROW_IF(dimension == 0, std::invalid_argument)
      << "QNN " << kind << " tensor " << index
      << " has a zero-sized dimension at axis " << axis;
    NNTR_THROW_IF(element_count >
                    std::numeric_limits<size_t>::max() / dimension,
                  std::length_error)
      << "QNN " << kind << " tensor " << index << " element count overflows";
    element_count *= dimension;
  }
  return element_count;
}

size_t getQnnTensorElementSize(Qnn_DataType_t qnn_type) {
  switch (qnn_type) {
  case QNN_DATATYPE_UINT_8:
  case QNN_DATATYPE_UFIXED_POINT_8:
    return 1;
  case QNN_DATATYPE_UINT_16:
  case QNN_DATATYPE_UFIXED_POINT_16:
  case QNN_DATATYPE_FLOAT_16:
    return 2;
  case QNN_DATATYPE_FLOAT_32:
    return 4;
  default:
    NNTR_THROW_IF(true, std::invalid_argument)
      << "Unsupported QNN tensor data type: "
      << static_cast<unsigned int>(qnn_type);
    return 0;
  }
}

struct QnnTensorBuffer {
  void *data;
  size_t required_bytes;
};

QnnTensorBuffer getQnnTensorBuffer(Tensor &tensor,
                                   const Qnn_Tensor_t &qnn_tensor,
                                   const char *kind, size_t index) {
  NNTR_THROW_IF(qnn_tensor.version != QNN_TENSOR_VERSION_1,
                std::invalid_argument)
    << "Unsupported QNN " << kind << " tensor descriptor version at index "
    << index;

  const auto tensor_type = tensor.getDataType();
  NNTR_THROW_IF(!isCompatibleQnnDataType(tensor_type, qnn_tensor.v1.dataType),
                std::invalid_argument)
    << "NNTrainer and QNN " << kind
    << " tensor data types do not match at index " << index
    << ": nntrainer=" << static_cast<int>(tensor_type)
    << ", qnn=" << static_cast<unsigned int>(qnn_tensor.v1.dataType);

  const size_t element_count =
    getQnnTensorElementCount(qnn_tensor, kind, index);
  NNTR_THROW_IF(tensor.size() != element_count, std::invalid_argument)
    << "NNTrainer and QNN " << kind
    << " tensor element counts do not match at index " << index
    << ": nntrainer=" << tensor.size() << ", qnn=" << element_count;

  const size_t element_size = getQnnTensorElementSize(qnn_tensor.v1.dataType);
  NNTR_THROW_IF(element_count >
                  std::numeric_limits<size_t>::max() / element_size,
                std::length_error)
    << "QNN " << kind << " tensor " << index << " byte count overflows";
  const size_t required_bytes = element_count * element_size;
  NNTR_THROW_IF(tensor.bytes() < required_bytes, std::invalid_argument)
    << "NNTrainer " << kind << " tensor " << index
    << " backing is too small: available=" << tensor.bytes()
    << ", required=" << required_bytes;

  void *buffer = tensor.getData<void>();
  NNTR_THROW_IF(buffer == nullptr, std::runtime_error)
    << "NNTrainer " << kind << " tensor " << index << " has no backing buffer";
  return {buffer, required_bytes};
}

using QuantParamMap = std::map<std::string, std::pair<float, int>>;

template <typename QuantProperty>
QuantParamMap
makeQuantParamMap(const std::vector<QuantProperty> &quant_properties,
                  const char *kind) {
  QuantParamMap quant_params;
  for (const auto &property : quant_properties) {
    const auto parameter = property.get();
    const auto inserted =
      quant_params.emplace(parameter.first, parameter.second);
    NNTR_THROW_IF(!inserted.second, std::invalid_argument)
      << "Duplicate QNN " << kind
      << " tensor quantization parameter: " << parameter.first;
  }
  return quant_params;
}

void applyQuantParam(Qnn_Tensor_t &tensor, const QuantParamMap &quant_params,
                     const char *kind, size_t index) {
  NNTR_THROW_IF(tensor.version != QNN_TENSOR_VERSION_1, std::invalid_argument)
    << "Unsupported QNN " << kind << " tensor descriptor version at index "
    << index;

  const char *name = tensor.v1.name;
  NNTR_THROW_IF(name == nullptr || name[0] == '\0', std::runtime_error)
    << "QNN " << kind << " tensor " << index << " has no name";

  switch (tensor.v1.quantizeParams.quantizationEncoding) {
  case QNN_QUANTIZATION_ENCODING_UNDEFINED:
  case QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET:
    return;
  case QNN_QUANTIZATION_ENCODING_SCALE_OFFSET:
    break;
  default:
    NNTR_THROW_IF(true, std::invalid_argument)
      << "Unsupported QNN " << kind << " tensor quantization encoding at index "
      << index << ": "
      << static_cast<unsigned int>(
           tensor.v1.quantizeParams.quantizationEncoding);
  }

  const auto parameter = quant_params.find(name);
  NNTR_THROW_IF(parameter == quant_params.end(), std::invalid_argument)
    << "Missing scalar quantization parameters for QNN " << kind << " tensor "
    << name;
  NNTR_THROW_IF(!std::isfinite(parameter->second.first) ||
                  parameter->second.first <= 0.0f,
                std::invalid_argument)
    << "Invalid scalar quantization scale for QNN " << kind << " tensor "
    << name << ": " << parameter->second.first;
  tensor.v1.quantizeParams.scaleOffsetEncoding.scale = parameter->second.first;
  tensor.v1.quantizeParams.scaleOffsetEncoding.offset =
    parameter->second.second;
}

void logSecondaryAfterExecuteFailure(
  const char *graph_name, bool returned_failure,
  const std::exception_ptr &after_execute_exception) noexcept {
  if (after_execute_exception != nullptr) {
    try {
      std::rethrow_exception(after_execute_exception);
    } catch (const std::exception &error) {
      ml_loge("QNN afterExecute threw after graph execution failure: "
              "graph=%s, reason=%s",
              graph_name, error.what());
    } catch (...) {
      ml_loge("QNN afterExecute threw an unknown exception after graph "
              "execution failure: graph=%s",
              graph_name);
    }
    return;
  }
  if (returned_failure) {
    ml_loge("QNN afterExecute returned false after graph execution failure: "
            "graph=%s",
            graph_name);
  }
}

} // namespace

QNNGraph::QNNGraph() :
  LayerImpl(), graph_props({}, {}, {}, props::FilePath(), {}, {}) {}

void QNNGraph::finalize(InitLayerContext &context) {
  bin_path = std::get<props::FilePath>(graph_props).get();

  auto &dims = std::get<std::vector<props::TensorDimension>>(graph_props);
  t_dims.assign(dims.begin(), dims.end());

  t_dtype = std::get<std::vector<props::TensorDataType>>(graph_props);

  t_type = std::get<std::vector<props::TensorType>>(graph_props);

  NNTR_THROW_IF(t_dims.size() != t_dtype.size(), std::invalid_argument)
    << "Size of Dimension, DataTypes must be same!";
  NNTR_THROW_IF(t_dims.size() != t_type.size(), std::invalid_argument)
    << "Size of Dimension, Types must be same!";

  std::vector<TensorDim> out_dim;

  for (unsigned int i = 0; i < t_dims.size(); ++i) {
    t_dims[i].setFormat(context.getFormat());
    t_dims[i].setDataType(t_dtype[i]);

    std::string name = "w_" + std::to_string(i);

    switch (t_type[i]) {
    case nntrainer::TensorType_::OUT_TENSOR:
      out_dim.push_back(t_dims[i]);
      break;
    case nntrainer::TensorType_::IN_TENSOR:
      tensor_idx.push_back(
        context.requestTensor(t_dims[i], name, Initializer::NONE, true,
                              TensorLifespan::FORWARD_FUNC_LIFESPAN));
      break;
    default:
      break;
    }
  }

  /// @todo fc actaully supports multidimensions. EffDimFlag shouldn't be fixed
  /// like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  /** set output dimensions */
  context.setOutputDimensions(out_dim);
}

void QNNGraph::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, graph_props);

  LayerImpl::setProperty(remain_props);
}

void QNNGraph::read(std::ifstream &file, RunLayerContext &run_context,
                    bool opt_var, ml::train::ExecutionMode mode, bool trainable,
                    TensorDim::DataType defineWeightDataType, bool fsu,
                    size_t start_offset, bool read_from_offset, int file_fd) {}

void QNNGraph::forwarding(RunLayerContext &context, bool training) {
  (void)training;

  auto qnn_data = getQNNVar(context);
  NNTR_THROW_IF(!qnn_data->RpcMem, std::runtime_error)
    << "QNN RPC memory manager is unavailable";

  auto context_ref = qnn_data->findContext(bin_path);
  if (!context_ref) {
    ml_logw("Context is not created. Create Now");
    NNTR_THROW_IF(qnn_data->makeContext(bin_path) != StatusCode::SUCCESS,
                  std::runtime_error)
      << "Failed to create QNN context from " << bin_path;
    context_ref = qnn_data->findContext(bin_path);
  }

  NNTR_THROW_IF(!context_ref, std::runtime_error)
    << "QNN context is unavailable after creation for " << bin_path;

  auto *graph_info = qnn_data->graphRetrieve(bin_path, context.getName());
  NNTR_THROW_IF(graph_info == nullptr, std::invalid_argument)
    << "Cannot retrieve graph " << context.getName() << " from " << bin_path;

  const char *graph_name =
    graph_info->graphName == nullptr ? "<unknown>" : graph_info->graphName;
  auto &context_info = context_ref->get();
  NNTR_THROW_IF(context_info.m_context == nullptr, std::runtime_error)
    << "QNN context handle is unavailable for " << bin_path;
  NNTR_THROW_IF(graph_info->graph == nullptr, std::runtime_error)
    << "QNN graph handle is unavailable for " << graph_name;

  NNTR_THROW_IF(context.getNumInputs() != graph_info->numInputTensors,
                std::invalid_argument)
    << "Number of NNtrainer's inputs " << context.getNumInputs()
    << " does not match with number of QNN's input tensors "
    << graph_info->numInputTensors << "!";

  NNTR_THROW_IF(context.getNumOutputs() != graph_info->numOutputTensors,
                std::invalid_argument)
    << "Number of NNtrainer's outputs " << context.getNumOutputs()
    << " does not match with number of QNN's output tensors "
    << graph_info->numOutputTensors << "!";

  auto inputs = IOTensorWrapper::makeTensorOwner(graph_info->numInputTensors);
  auto outputs = IOTensorWrapper::makeTensorOwner(graph_info->numOutputTensors);
  NNTR_THROW_IF(qnn_data->m_ioTensor.setupInputAndOutputTensors(inputs, outputs,
                                                                *graph_info) !=
                  qnn::tools::iotensor::StatusCode::SUCCESS,
                std::runtime_error)
    << "Failed to set up QNN I/O tensor descriptors for graph " << graph_name;

  const auto input_quant_params = makeQuantParamMap(
    std::get<std::vector<props::InputQuantParam>>(graph_props), "input");
  const auto output_quant_params = makeQuantParamMap(
    std::get<std::vector<props::OutputQuantParam>>(graph_props), "output");

  for (size_t i = 0; i < context.getNumInputs(); ++i) {
    applyQuantParam(inputs[i], input_quant_params, "input", i);
    const auto buffer =
      getQnnTensorBuffer(context.getInput(i), inputs[i], "input", i);
    qnn_data->RpcMem->registerQnnTensor(
      buffer.data, inputs[i], context_info.m_context, buffer.required_bytes);
  }

  for (size_t i = 0; i < context.getNumOutputs(); ++i) {
    applyQuantParam(outputs[i], output_quant_params, "output", i);
    const auto buffer =
      getQnnTensorBuffer(context.getOutput(i), outputs[i], "output", i);
    qnn_data->RpcMem->registerQnnTensor(
      buffer.data, outputs[i], context_info.m_context, buffer.required_bytes);
  }

  auto &qnn_interface = qnn_data->m_qnnFunctionPointers.qnnInterface;
  NNTR_THROW_IF(qnn_interface.graphExecute == nullptr, std::runtime_error)
    << "QNN graphExecute function is unavailable";

  QnnGraph_Config_t **custom_graph_configs = nullptr;
  uint32_t config_count = 0;
  auto *extension = qnn_data->m_backendExtensions == nullptr
                      ? nullptr
                      : qnn_data->m_backendExtensions->interface();
  if (extension != nullptr) {
    NNTR_THROW_IF(!extension->beforeExecute(graph_name, &custom_graph_configs,
                                            &config_count),
                  std::runtime_error)
      << "QNN extension beforeExecute failed for graph " << graph_name;

    NNTR_THROW_IF(config_count > 0 && custom_graph_configs == nullptr,
                  std::runtime_error)
      << "QNN extension returned an invalid graph config list for "
      << graph_name << ": count=" << config_count;
    NNTR_THROW_IF(static_cast<uint64_t>(config_count) + 1 >
                    static_cast<uint64_t>(std::numeric_limits<size_t>::max()),
                  std::length_error)
      << "QNN extension returned too many graph configs for " << graph_name;

    if (config_count > 0) {
      std::vector<const QnnGraph_Config_t *> graph_configs(
        static_cast<size_t>(config_count) + 1, nullptr);
      for (uint32_t i = 0; i < config_count; ++i) {
        NNTR_THROW_IF(custom_graph_configs[i] == nullptr, std::runtime_error)
          << "QNN extension returned a null graph config at index " << i
          << " for " << graph_name;
        graph_configs[i] = custom_graph_configs[i];
      }

      NNTR_THROW_IF(qnn_interface.graphSetConfig == nullptr, std::runtime_error)
        << "QNN graphSetConfig function is unavailable";
      const auto config_status =
        qnn_interface.graphSetConfig(graph_info->graph, graph_configs.data());
      if (config_status != QNN_SUCCESS) {
        const auto error_code = static_cast<uint64_t>(config_status);
        const auto public_error_code =
          static_cast<unsigned int>(QNN_GET_ERROR_CODE(config_status));
        ml_loge("QNN graphSetConfig failed: error=%" PRIu64
                ", public_error=%u, binary=%s, graph=%s",
                error_code, public_error_code, bin_path.c_str(), graph_name);
        NNTR_THROW_IF(true, std::runtime_error)
          << "QNN graphSetConfig failed: error=" << error_code
          << ", public_error=" << public_error_code << ", binary=" << bin_path
          << ", graph=" << graph_name;
      }
    }
  }

  Qnn_ErrorHandle_t execute_status = QNN_GRAPH_NO_ERROR;
  std::exception_ptr execute_exception;
  try {
    execute_status = qnn_interface.graphExecute(
      graph_info->graph, inputs.get(), graph_info->numInputTensors,
      outputs.get(), graph_info->numOutputTensors,
      qnn_data->m_profileBackendHandle, nullptr);
  } catch (...) {
    execute_exception = std::current_exception();
  }

  bool after_execute_succeeded = true;
  std::exception_ptr after_execute_exception;
  if (extension != nullptr) {
    try {
      after_execute_succeeded = extension->afterExecute();
    } catch (...) {
      after_execute_exception = std::current_exception();
    }
  }

  const bool after_execute_failed =
    !after_execute_succeeded || after_execute_exception != nullptr;
  if (execute_exception != nullptr) {
    if (after_execute_failed) {
      logSecondaryAfterExecuteFailure(graph_name, !after_execute_succeeded,
                                      after_execute_exception);
    }
    std::rethrow_exception(execute_exception);
  }

  if (execute_status != QNN_GRAPH_NO_ERROR) {
    if (after_execute_failed) {
      logSecondaryAfterExecuteFailure(graph_name, !after_execute_succeeded,
                                      after_execute_exception);
    }

    const auto error_code = static_cast<uint64_t>(execute_status);
    const auto public_error_code =
      static_cast<unsigned int>(QNN_GET_ERROR_CODE(execute_status));
    ml_loge("[QNNGraph] graphExecute failed: error=%" PRIu64
            ", public_error=%u, binary=%s, graph=%s, context=%p, "
            "graph_handle=%p",
            error_code, public_error_code, bin_path.c_str(), graph_name,
            static_cast<void *>(context_info.m_context),
            static_cast<void *>(graph_info->graph));

    NNTR_THROW_IF(true, std::runtime_error)
      << "QNN graphExecute failed: error=" << error_code
      << ", public_error=" << public_error_code << ", binary=" << bin_path
      << ", graph=" << graph_name
      << ", context=" << static_cast<void *>(context_info.m_context)
      << ", graph_handle=" << static_cast<void *>(graph_info->graph);
  }

  if (after_execute_exception != nullptr) {
    try {
      std::rethrow_exception(after_execute_exception);
    } catch (const std::exception &error) {
      NNTR_THROW_IF(true, std::runtime_error)
        << "QNN extension afterExecute threw for graph " << graph_name << ": "
        << error.what();
    } catch (...) {
      NNTR_THROW_IF(true, std::runtime_error)
        << "QNN extension afterExecute threw for graph " << graph_name;
    }
  }
  NNTR_THROW_IF(!after_execute_succeeded, std::runtime_error)
    << "QNN extension afterExecute failed for graph " << graph_name;
}

} // namespace nntrainer
