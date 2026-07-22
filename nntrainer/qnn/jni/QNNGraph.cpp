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
#include <cstdint>
#include <fcntl.h>
#include <inttypes.h>
#include <memory>
#include <sys/mman.h>
#include <unistd.h>

#include <sys/resource.h>
#include <thread>

#include "QnnSampleAppUtils.hpp"
#include "Utils/DataUtil.hpp"
#include <common_properties.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

std::shared_ptr<QNNVar> getQNNVar(RunLayerContext &context) {
  std::shared_ptr<QNNVar> qc_var =
    (std::static_pointer_cast<QNNBackendVar>(context.getContextData()))
      ->getVar();
  return qc_var;
}

static void *getQnnTensorBuffer(Tensor &tensor, const char *kind,
                                size_t index) {
  switch (tensor.getDataType()) {
  case Tdatatype::UINT4:
  case Tdatatype::UINT8:
  case Tdatatype::UINT16:
  case Tdatatype::FP32:
    break;
  default:
    NNTR_THROW_IF(true, std::invalid_argument)
      << "Unsupported QNN " << kind << " tensor data type at index " << index
      << ": " << static_cast<int>(tensor.getDataType());
    return nullptr;
  }

  void *buffer = tensor.getData<void>();
  NNTR_THROW_IF(buffer == nullptr, std::runtime_error)
    << "NNTrainer " << kind << " tensor " << index << " has no backing buffer";
  return buffer;
}

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
  auto qc_var = getQNNVar(context);
  NNTR_THROW_IF(!qc_var->RpcMem, std::runtime_error)
    << "QNN RPC memory manager is unavailable";
  auto execution_guard = qc_var->RpcMem->acquireExecutionGuard();

  auto context_ref = qc_var->findContext(bin_path);
  if (!context_ref) {
    execution_guard.unlock();
    {
      auto cleanup_guard = qc_var->RpcMem->acquireCleanupGuard();
      context_ref = qc_var->findContext(bin_path);
      if (!context_ref) {
        ml_logw("Context is not created. Create Now");
        NNTR_THROW_IF(qc_var->makeContext(bin_path) != StatusCode::SUCCESS,
                      std::runtime_error)
          << "Failed to create QNN context from " << bin_path;
      }
    }
    execution_guard = qc_var->RpcMem->acquireExecutionGuard();
    context_ref = qc_var->findContext(bin_path);
  }

  NNTR_THROW_IF(!context_ref, std::runtime_error)
    << "QNN context is unavailable after creation for " << bin_path;

  auto graphInfo = qc_var->graphRetrieve(bin_path, context.getName());
  NNTR_THROW_IF(!graphInfo, std::invalid_argument)
    << "cannot retrieve graph " << context.getName() << " from " << bin_path;

  Qnn_Context_Graph_t &context_i = context_ref->get();

  const char *graph_name =
    graphInfo->graphName == nullptr ? "<unknown>" : graphInfo->graphName;

  NNTR_THROW_IF(context.getNumInputs() != graphInfo->numInputTensors,
                std::invalid_argument)
    << "Number of NNtrainer's inputs " << context.getNumInputs()
    << " does not match with number of QNN's input tensors "
    << graphInfo->numInputTensors << "!";

  NNTR_THROW_IF(context.getNumOutputs() != graphInfo->numOutputTensors,
                std::invalid_argument)
    << "Number of NNtrainer's outputs " << context.getNumOutputs()
    << " does not match with number of QNN's output tensors "
    << graphInfo->numOutputTensors << "!";

  auto inputs = IOTensorWrapper::makeTensorOwner(graphInfo->numInputTensors);
  auto outputs = IOTensorWrapper::makeTensorOwner(graphInfo->numOutputTensors);

  NNTR_THROW_IF(qc_var->m_ioTensor.setupInputAndOutputTensors(
                  inputs, outputs, *graphInfo) != iotensor::StatusCode::SUCCESS,
                std::runtime_error)
    << "Failed to set up QNN I/O tensor descriptors for graph " << graph_name;

  auto input_quant_params =
    std::get<std::vector<props::InputQuantParam>>(graph_props);
  std::map<std::string, std::pair<float, int>> input_quant_param_map;
  for (auto &param : input_quant_params) {
    auto p = param.get();
    input_quant_param_map[p.first] = p.second;
  }
  auto output_quant_params =
    std::get<std::vector<props::OutputQuantParam>>(graph_props);
  std::map<std::string, std::pair<float, int>> output_quant_param_map;
  for (auto &param : output_quant_params) {
    auto p = param.get();
    output_quant_param_map[p.first] = p.second;
  }

  for (size_t i = 0; i < context.getNumInputs(); ++i) {
    const char *key = inputs[i].v1.name;
    NNTR_THROW_IF(key == nullptr, std::runtime_error)
      << "QNN input tensor " << i << " has no name";
    NNTR_THROW_IF(input_quant_param_map.find(key) ==
                    input_quant_param_map.end(),
                  std::invalid_argument)
      << key;
    auto value = input_quant_param_map[key];
    inputs[i].v1.quantizeParams.scaleOffsetEncoding.scale = value.first;
    inputs[i].v1.quantizeParams.scaleOffsetEncoding.offset = value.second;
    // NNTrainer QNN tensors already use RPC shared-memory backing. Bind that
    // allocation directly instead of copying into a raw client buffer.
    void *buffer = getQnnTensorBuffer(context.getInput(i), "input", i);
    qc_var->RpcMem->registerQnnTensor(buffer, inputs[i], context_i.m_context,
                                      execution_guard);
  }

  for (size_t i = 0; i < context.getNumOutputs(); ++i) {
    const char *key = outputs[i].v1.name;
    NNTR_THROW_IF(key == nullptr, std::runtime_error)
      << "QNN output tensor " << i << " has no name";
    NNTR_THROW_IF(output_quant_param_map.find(key) ==
                    output_quant_param_map.end(),
                  std::invalid_argument)
      << key;
    auto value = output_quant_param_map[key];
    outputs[i].v1.quantizeParams.scaleOffsetEncoding.scale = value.first;
    outputs[i].v1.quantizeParams.scaleOffsetEncoding.offset = value.second;
    // Outputs are written directly into the NNTrainer RPC allocation as well.
    void *buffer = getQnnTensorBuffer(context.getOutput(i), "output", i);
    qc_var->RpcMem->registerQnnTensor(buffer, outputs[i], context_i.m_context,
                                      execution_guard);
  }

  Qnn_ErrorHandle_t executeStatus = QNN_GRAPH_NO_ERROR;
  QnnGraph_Config_t **customGraphConfigs{nullptr};
  uint32_t configCount{0};
  auto backend_extensions = qc_var->m_backendExtensions;
  if (nullptr != backend_extensions && backend_extensions->interface()) {
    if (!backend_extensions->interface()->beforeExecute(
          graphInfo->graphName, &customGraphConfigs, &configCount)) {
      QNN_ERROR("Extensions Failure in beforeExecute()");
    }
    if (customGraphConfigs) {
      std::vector<const QnnGraph_Config_t *> graphConfigsPointers(
        configCount + 1, nullptr);
      for (size_t idx = 0u; idx < configCount; idx++) {
        graphConfigsPointers[idx] = customGraphConfigs[idx];
      }
      if (QNN_SUCCESS !=
          qc_var->m_qnnFunctionPointers.qnnInterface.graphSetConfig(
            graphInfo->graph, graphConfigsPointers.data())) {
        QNN_ERROR("Failure in setGraphConfigsBeforeExecute()");
      }
    }
  }
  executeStatus = qc_var->m_qnnFunctionPointers.qnnInterface.graphExecute(
    graphInfo->graph, inputs.get(), graphInfo->numInputTensors, outputs.get(),
    graphInfo->numOutputTensors, qc_var->m_profileBackendHandle, nullptr);

  if (nullptr != backend_extensions && backend_extensions->interface()) {
    if (!backend_extensions->interface()->afterExecute()) {
      QNN_ERROR("Extensions Failure in afterExecute()");
    }
  }

  if (QNN_GRAPH_NO_ERROR != executeStatus) {
    const auto error_code = static_cast<uint64_t>(executeStatus);
    const auto public_error_code =
      static_cast<unsigned int>(QNN_GET_ERROR_CODE(executeStatus));
    ml_loge("[QNNGraph] graphExecute failed: error=%" PRIu64 ", public_error=%u"
            ", binary=%s, graph=%s, context=%p, graph_handle=%p",
            error_code, public_error_code, bin_path.c_str(), graph_name,
            static_cast<void *>(context_i.m_context),
            static_cast<void *>(graphInfo->graph));

    NNTR_THROW_IF(true, std::runtime_error)
      << "QNN graphExecute failed: error=" << error_code
      << ", public_error=" << public_error_code << ", binary=" << bin_path
      << ", graph=" << graph_name
      << ", context=" << static_cast<void *>(context_i.m_context)
      << ", graph_handle=" << static_cast<void *>(graphInfo->graph);
  }
}

} // namespace nntrainer
