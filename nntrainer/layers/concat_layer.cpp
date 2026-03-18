// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   concat_layer.cpp
 * @date   27 Oct 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Concat Layer Class for Neural Network
 *
 * @todo merge concat and split layer to a common implementation
 */

#include <cstring>
#include <type_traits>
#include <vector>

#include <concat_layer.h>
#include <layer_context.h>
#include <nntr_threads.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor_dim.h>
#include <util_func.h>

namespace nntrainer {
ConcatLayer::ConcatLayer() :
  Layer(), concat_dimension(1), leading_helper_dim(1) {}

static constexpr size_t SINGLE_INOUT_IDX = 0;

void ConcatLayer::finalize(InitLayerContext &context) {
  auto &concat_dimension_prop = std::get<props::ConcatDimension>(concat_props);
  /** for backward compatibility, default concat dimension will be channel */
  /// @todo this is hacky way to force concat dimension to width if channel
  /// dimension is taken, this is because recurrent realizer, return sequence
  /// exploits concat layer but have no control over where to stack/axis
  concat_dimension = context.getInputDimensions().front().channel() > 1 ? 3 : 1;
  if (!concat_dimension_prop.empty())
    concat_dimension = concat_dimension_prop.get();

  /**
   * The concat is only done along the axis dimension.
   * For example, consider 2 inputs a, b with dimensions [b,c,h,w] each
   * 1. concat_dimension = 1, output_dim = [b,c_a+c_b,h,w]
   * 2. concat_dimension = 2, output_dim = [b,c,h_a+h_b,w]
   * 3. concat_dimension = 3, output_dim = [b,c,h,w_a+w_b]
   */
  auto const &input_dims = context.getInputDimensions();
  const TensorDim &input_dim_0 = input_dims[SINGLE_INOUT_IDX];
  unsigned int concat_dim_val = input_dim_0.getTensorDim(concat_dimension);

  for (unsigned int idx = 1; idx < input_dims.size(); ++idx) {
    const TensorDim &dim = input_dims[idx];

    for (unsigned int i = 0; i < ml::train::TensorDim::getNumDim(); ++i) {
      if (i == concat_dimension)
        continue;
      NNTR_THROW_IF(input_dim_0[i] != dim[i], std::runtime_error)
        << "Error: concat layer requires same shape from all input layers "
           "along non-concat dimension";
    }
    concat_dim_val += dim[concat_dimension];
  }

  TensorDim output_dim = input_dim_0;
  output_dim.setTensorDim(concat_dimension, concat_dim_val);

  context.setOutputDimensions({output_dim});

  /**
   * The following helper shapes facilitate efficient concatenation and split of
   * the data.
   *
   * The helper shapes are created by consolidating all the dimensions before
   * the concat dimension to the first axis and all the remaining dimensions to
   * the last axis.
   *
   * @note This is possible since the data starting from the concat dimension to
   * the end is always continuous.
   *
   * @example the following shows how the helper dimension will look with given
   * inputs and concat dimension.
   *
   *          | cat_dim 1 | cat_dim 2 | cat_dim 3
   *  --------|-----------|-----------|-----------
   *  input0  |  2:1:2:3  |  1:2:1:3  |  1:2:2:3
   *  input1  |  2:3:2:3  |  1:2:3:3  |  1:2:2:1
   *  --------|-----------|-----------|-----------
   *  helper0 |  2:1:1:6  |  2:1:1:3  |  4:1:1:3
   *  helper1 |  2:1:1:18 |  2:1:1:9  |  4:1:1:1
   *
   */
  /// Setup output_reshape_helper (how output should be reshaped)
  output_reshape_helper.channel(1);
  output_reshape_helper.height(1);
  output_reshape_helper.width(1);
  for (unsigned int axis = concat_dimension;
       axis < ml::train::TensorDim::getNumDim(); ++axis) {
    output_reshape_helper.width(output_reshape_helper.width() *
                                output_dim.getTensorDim(axis));
  }

  /// Setup input_reshape_helper (how inputs should be reshaped)
  input_reshape_helper.resize(input_dims.size());

  for (unsigned int idx = 0; idx < input_reshape_helper.size(); idx++) {
    input_reshape_helper[idx].channel(1);
    input_reshape_helper[idx].height(1);
    input_reshape_helper[idx].width(1);

    for (unsigned int axis = concat_dimension;
         axis < ml::train::TensorDim::getNumDim(); ++axis) {

      input_reshape_helper[idx].width(input_reshape_helper[idx].width() *
                                      input_dims[idx].getTensorDim(axis));
    }
  }

  leading_helper_dim = 1;
  for (unsigned int idx = 1; idx < concat_dimension; ++idx) {
    leading_helper_dim *= output_dim.getTensorDim(idx);
  }

  setBatch(input_dims[SINGLE_INOUT_IDX].batch());
}

void ConcatLayer::forwarding(RunLayerContext &context, bool training) {
  /**
   * Forwarding in ConcatLayer works as follows
   *
   *    in1        in2       in3                  output
   * |---0---| |----3----| |--6--|      |---0---||----3----||--6--|
   * |---1---| |----4----| |--7--|  =>  |---1---||----4----||--7--|
   * |---2---| |----5----| |--8--|      |---2---||----5----||--8--|
   *
   * @note For each input tensor, it iterates batches and copies the entire
   * width size to the corresponding output position. In the diagram above, the
   * row would be a batch, and the column would be a width. the number of each
   * block in the diagram indicates the order of copy to output.
   *
   * @todo avoid copy by creating input here as a shared_tensor of the output
   * here and then this layer can be in_place as well
   */
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  const TensorDim out_dim = output.getDim();
  output.reshape(output_reshape_helper);
  unsigned int output_width_offset = 0;
  TensorDim::TensorType tensor_type = output.getTensorType();

  for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
    Tensor &input = context.getInput(idx);
    const TensorDim in_dim = input.getDim();
    auto const &irh = input_reshape_helper[idx];
    input.reshape(irh);
    unsigned int data_copy_size = irh.width();

    /** loop over the dimensions before the concat dimension */
    if (in_dim.getDataType() == TensorDim::DataType::FP32) {
      /** copy continous tensor data (reshaped width) */
      for (unsigned int batch = 0; batch < output.batch(); batch++) {
        Tensor dest_tensor = Tensor::Map<float>(
          output.getAddress<float>(batch, 0, 0, output_width_offset),
          data_copy_size * sizeof(float),
          {1, 1, 1, data_copy_size, tensor_type});
        const Tensor source_tensor =
          Tensor::Map<float>(input.getAddress<float>(batch, 0, 0, 0),
                             data_copy_size * sizeof(float),
                             {1, 1, 1, data_copy_size, tensor_type});
        dest_tensor.copy(source_tensor);
      }
    } else if (in_dim.getDataType() == TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      /** copy continous tensor data (reshaped width) */
      for (unsigned int batch = 0; batch < output.batch(); batch++) {
        Tensor dest_tensor = Tensor::Map<_FP16>(
          output.getAddress<_FP16>(batch, 0, 0, output_width_offset),
          data_copy_size * sizeof(_FP16),
          {1, 1, 1, data_copy_size, tensor_type});
        const Tensor source_tensor =
          Tensor::Map<_FP16>(input.getAddress<_FP16>(batch, 0, 0, 0),
                             data_copy_size * sizeof(_FP16),
                             {1, 1, 1, data_copy_size, tensor_type});
        dest_tensor.copy(source_tensor);
      }
#else
      throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
    }

    output_width_offset += irh.width();
    input.reshape(in_dim);
  }

  output.reshape(out_dim);
}

void ConcatLayer::incremental_forwarding(RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {
  /**
   * @todo avoid copy by creating input here as a shared_tensor of the output
   * here and then this layer can be in_place as well
   */
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  const TensorDim out_dim = output.getDim();
  const TensorDim::TensorType tensor_type = output.getTensorType();
  if (concat_dimension == 3)
    output.reshape(output_reshape_helper);

  auto copy_incremental_step = [&](auto *type_tag, unsigned int elem_size) {
    unsigned int output_concat_offset = 0;

    for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
      Tensor &input = context.getInput(idx);
      const TensorDim in_dim = input.getDim();

      if (concat_dimension == 1) {
        const unsigned int step_height = to - from;
        const unsigned int step_width = step_height * in_dim.width();

        for (unsigned int batch = 0; batch < out_dim.batch(); ++batch) {
          for (unsigned int channel = 0; channel < in_dim.channel();
               ++channel) {
            Tensor dest_tensor = Tensor::Map<std::decay_t<decltype(*type_tag)>>(
              output.getAddress<std::decay_t<decltype(*type_tag)>>(
                batch, output_concat_offset + channel, from, 0),
              step_width * elem_size, {1, 1, 1, step_width, tensor_type});
            const Tensor source_tensor =
              Tensor::Map<std::decay_t<decltype(*type_tag)>>(
                input.getAddress<std::decay_t<decltype(*type_tag)>>(
                  batch, channel, from, 0),
                step_width * elem_size, {1, 1, 1, step_width, tensor_type});
            dest_tensor.copy(source_tensor);
          }
        }

        output_concat_offset += in_dim.channel();
      } else if (concat_dimension == 2) {
        const unsigned int step_height = to - from;
        const unsigned int step_width = step_height * in_dim.width();

        for (unsigned int batch = 0; batch < out_dim.batch(); ++batch) {
          for (unsigned int channel = 0; channel < in_dim.channel();
               ++channel) {
            Tensor dest_tensor = Tensor::Map<std::decay_t<decltype(*type_tag)>>(
              output.getAddress<std::decay_t<decltype(*type_tag)>>(
                batch, channel, output_concat_offset + from, 0),
              step_width * elem_size, {1, 1, 1, step_width, tensor_type});
            const Tensor source_tensor =
              Tensor::Map<std::decay_t<decltype(*type_tag)>>(
                input.getAddress<std::decay_t<decltype(*type_tag)>>(
                  batch, channel, from, 0),
                step_width * elem_size, {1, 1, 1, step_width, tensor_type});
            dest_tensor.copy(source_tensor);
          }
        }

        output_concat_offset += in_dim.height();
      } else if (concat_dimension == 3) {
        const auto &irh = input_reshape_helper[idx];
        const unsigned int data_copy_size = irh.width();
        const unsigned int batch_channel = out_dim.batch() * out_dim.channel();

        input.reshape(irh);
        for (unsigned int batch = batch_channel * from;
             batch < batch_channel * to; batch++) {
          for (unsigned int count = 0; count < irh.height(); count++) {
            Tensor dest_tensor = Tensor::Map<std::decay_t<decltype(*type_tag)>>(
              output.getAddress<std::decay_t<decltype(*type_tag)>>(
                batch, 0, output_concat_offset + count, 0),
              data_copy_size * elem_size,
              {1, 1, 1, data_copy_size, tensor_type});
            const Tensor source_tensor =
              Tensor::Map<std::decay_t<decltype(*type_tag)>>(
                input.getAddress<std::decay_t<decltype(*type_tag)>>(batch, 0,
                                                                    count, 0),
                data_copy_size * elem_size,
                {1, 1, 1, data_copy_size, tensor_type});
            dest_tensor.copy(source_tensor);
          }
        }
        input.reshape(in_dim);
        output_concat_offset += irh.height();
      } else {
        throw std::invalid_argument(
          "ConcatLayer incremental_forwarding supports axis 1, 2, and 3 only");
      }
    }
  };

  if (out_dim.getDataType() == TensorDim::DataType::FP32) {
    copy_incremental_step(output.getData<float>(), sizeof(float));
  } else if (out_dim.getDataType() == TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    copy_incremental_step(output.getData<_FP16>(), sizeof(_FP16));
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  } else {
    throw std::invalid_argument(
      "ConcatLayer incremental_forwarding supports fp32/fp16 only");
  }

  if (concat_dimension == 3)
    output.reshape(out_dim);
}

void ConcatLayer::calcDerivative(RunLayerContext &context) {
  /**
   * calcDerivative in ConcatLayer works as follows
   *
   *           output                    in1        in2       in3
   * |---0---||----3----||--6--|      |---0---| |----3----| |--6--|
   * |---1---||----4----||--7--|  =>  |---1---| |----4----| |--7--|
   * |---2---||----5----||--8--|      |---2---| |----5----| |--8--|
   *
   * @note For each input tensor, it iterates batches and copies the entire
   * input width size from the output tensor to the corresponding input. In the
   * diagram above, the row would be a batch, and the column would be a width.
   * The number of each block in the diagram indicates the order of copy to
   * inputs.
   *
   * @todo avoid copy by creating input here as a shared_tensor of the output
   * here and then this layer can be in_place as well
   */
  Tensor output = context.getIncomingDerivative(SINGLE_INOUT_IDX);

  output.reshape(output_reshape_helper);
  unsigned int output_width_offset = 0;
  TensorDim::TensorType tensor_type = output.getTensorType();

  for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
    Tensor &input = context.getOutgoingDerivative(idx);
    const TensorDim in_dim = input.getDim();
    auto const &irh = input_reshape_helper[idx];
    input.reshape(irh);
    unsigned int data_copy_size = irh.width();

    if (in_dim.getDataType() == TensorDim::DataType::FP32) {
      /** loop over the dimensions before the concat dimension */
      for (unsigned int batch = 0; batch < output.batch(); batch++) {
        /** copy continous data (reshaped width size) in a tensor */
        const Tensor source_tensor = Tensor::Map<float>(
          output.getAddress<float>(batch, 0, 0, output_width_offset),
          data_copy_size * sizeof(float),
          {1, 1, 1, data_copy_size, tensor_type});
        Tensor dest_tensor =
          Tensor::Map<float>(input.getAddress<float>(batch, 0, 0, 0),
                             data_copy_size * sizeof(float),
                             {1, 1, 1, data_copy_size, tensor_type});
        dest_tensor.copy(source_tensor);
      }
    } else if (in_dim.getDataType() == TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      /** loop over the dimensions before the concat dimension */
      for (unsigned int batch = 0; batch < output.batch(); batch++) {
        /** copy continous data (reshaped width size) in a tensor */
        const Tensor source_tensor = Tensor::Map<_FP16>(
          output.getAddress<_FP16>(batch, 0, 0, output_width_offset),
          data_copy_size * sizeof(_FP16),
          {1, 1, 1, data_copy_size, tensor_type});
        Tensor dest_tensor =
          Tensor::Map<_FP16>(input.getAddress<_FP16>(batch, 0, 0, 0),
                             data_copy_size * sizeof(_FP16),
                             {1, 1, 1, data_copy_size, tensor_type});
        dest_tensor.copy(source_tensor);
      }
#else
      throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
    }

    input.reshape(in_dim);
    output_width_offset += irh.width();
  }
}

void ConcatLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, concat_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[ConcatLayer] Unknown Layer Properties count " +
         std::to_string(values.size());
}

void ConcatLayer::exportTo(Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  Layer::exportTo(exporter, method);
  exporter.saveResult(concat_props, method, this);
}

} /* namespace nntrainer */
