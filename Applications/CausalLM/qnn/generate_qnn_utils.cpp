// SPDX-License-Identifier: Apache-2.0
/**
 * @file   generate_qnn_utils.cpp
 * @brief  Helper utilities for QNN generation: KV cache bookkeeping,
 *         attention mask/rope buffer prep, and sampling.
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include "generate_qnn_utils.h"
#include "android_memory_allocator.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <model.h>
#include <tokenizers_cpp.h>

std::vector<IO_TensorType>
run_qnn_inference(ModelHandle &model, unsigned int batch,
                  const std::vector<IO_TensorType> &inputs,
                  const GraphInfo &graph_info) {
  // 1. Strip dtype tags: the main nntrainer inference API is type-erased and
  //    treats float* purely as an opaque buffer carrier (the real dtype is
  //    recovered from the graph's input dims inside mapExternalTensor). The
  //    bound pointer address is preserved across the reinterpret_cast, so no
  //    runtime behavior changes vs. the OLD variant-typed API.
  std::vector<float *> float_inputs;
  float_inputs.reserve(inputs.size());
  for (const auto &in : inputs) {
    float *ptr =
      std::visit([](auto *p) { return reinterpret_cast<float *>(p); }, in);
    float_inputs.push_back(ptr);
  }

  std::vector<float *> float_outputs = model->inference(batch, float_inputs);

  // 2. Re-tag each output buffer with the graph's declared output dtype so
  //    callers can keep using std::get<uint16_t*>/std::get<uint8_t*>.
  std::vector<IO_TensorType> outputs;
  outputs.reserve(float_outputs.size());
  for (size_t i = 0; i < float_outputs.size(); ++i) {
    std::string dtype = (i < graph_info.raw_outputs.size())
                          ? graph_info.raw_outputs[i].data_type
                          : std::string();
    if (dtype == "QNN_DATATYPE_UFIXED_POINT_8") {
      outputs.emplace_back(reinterpret_cast<uint8_t *>(float_outputs[i]));
    } else if (dtype == "QNN_DATATYPE_UFIXED_POINT_16" ||
               dtype == "QNN_DATATYPE_FLOAT_16") {
      outputs.emplace_back(reinterpret_cast<uint16_t *>(float_outputs[i]));
    } else {
      // Unknown / FP32 / unspecified: keep as float* (matches OLD default).
      outputs.emplace_back(float_outputs[i]);
    }
  }
  return outputs;
}

namespace {

constexpr int kQnnKvNumColumns = 128;

constexpr float kRopeQuantScale = 3.051804378628731e-05f;
constexpr int kRopeQuantOffset = -32768;

uint16_t quantize_rope_value(double value, double attention_factor) {
  const double q =
    (value * attention_factor) / kRopeQuantScale - kRopeQuantOffset;
  if (q <= 0.0)
    return 0;
  if (q >= 65535.0)
    return 65535;
  return static_cast<uint16_t>(std::lrint(q));
}

} // namespace

std::tuple<uint16_t *, uint16_t *>
get_cos_sin(int context_size, int pos_dim, const double theta,
            const std::string &rope_type, double partial_rotary_factor,
            double rope_scaling_factor, int rope_head_dim) {
  double attention_factor = 1.0;
  const double scaling_factor =
    rope_scaling_factor > 0.0 ? rope_scaling_factor : 1.0;
  const int frequency_dim = rope_head_dim > 0 ? rope_head_dim : pos_dim * 2;
  // inv_freq indexes angle pairs, so the exponent advances by 2/head_dim.
  const double exponent = 2.0 / static_cast<double>(frequency_dim);
  std::vector<double> inv_freq(pos_dim, 0.0);

  if (rope_type == "default") {
    for (int j = 0; j < pos_dim; j++)
      inv_freq[j] = 1.0 / std::pow(theta, j * exponent);
  } else if (rope_type == "linear" || rope_type == "proportional") {
    int rotary_freq_count = pos_dim;

    if (rope_type == "proportional") {
      double proportion = partial_rotary_factor;
      if (proportion < 0.0)
        proportion = 0.0;
      if (proportion > 1.0)
        proportion = 1.0;
      rotary_freq_count = static_cast<int>(
        std::floor(proportion * static_cast<double>(frequency_dim) / 2.0));
      rotary_freq_count = std::max(0, std::min(pos_dim, rotary_freq_count));
    }

    for (int j = 0; j < rotary_freq_count; j++)
      inv_freq[j] = (1.0 / std::pow(theta, j * exponent)) / scaling_factor;
  } else {
    for (int j = 0; j < pos_dim; j++)
      inv_freq[j] = 1.0 / std::pow(theta, j * exponent);
  }

  // Partial RoPE (Gemma 4 full attention uses partial_rotary_factor=0.25):
  // only the first `effective_dim` lanes carry an actual rotation; the
  // remaining lanes are identity (cos=1, sin=0) so multiplication is
  // pass-through. Without this the non-rotary lanes get a small bogus
  // rotation that compounds across the 35 transformer layers and the
  // model collapses into repetition after a few dozen tokens.
  int effective_dim = pos_dim;
  if (partial_rotary_factor > 0.0 && partial_rotary_factor < 1.0) {
    effective_dim =
      static_cast<int>(std::floor(pos_dim * partial_rotary_factor));
    if (effective_dim < 0)
      effective_dim = 0;
    if (effective_dim > pos_dim)
      effective_dim = pos_dim;
  }

  uint16_t *cos_val =
    (uint16_t *)allocate(sizeof(uint16_t) * context_size * pos_dim);
  uint16_t *sin_val =
    (uint16_t *)allocate(sizeof(uint16_t) * context_size * pos_dim);

  // Quantized identity values for non-rotary lanes.
  const uint16_t cos_one = quantize_rope_value(1.0, attention_factor);
  const uint16_t sin_zero = quantize_rope_value(0.0, attention_factor);

  for (int i = 0; i < context_size; i++) {
    for (int j = 0; j < effective_dim; j++) {
      const double freq = i * inv_freq[j];
      cos_val[i * pos_dim + j] =
        quantize_rope_value(std::cos(freq), attention_factor);
      sin_val[i * pos_dim + j] =
        quantize_rope_value(std::sin(freq), attention_factor);
    }
    for (int j = effective_dim; j < pos_dim; j++) {
      cos_val[i * pos_dim + j] = cos_one;
      sin_val[i * pos_dim + j] = sin_zero;
    }
  }
  return std::make_tuple(cos_val, sin_val);
}

bool qnn_starts_with(const std::string &value, const std::string &prefix) {
  return value.compare(0, prefix.size(), prefix) == 0;
}

int find_tensor_index_or_minus_one(const TensorInfoList &tensor_infos,
                                   const std::string &tensor_name) {
  for (size_t idx = 0; idx < tensor_infos.size(); idx++) {
    if (tensor_infos[idx].name == tensor_name) {
      return static_cast<int>(idx);
    }
  }
  return -1;
}

std::string kv_output_to_input_name(const std::string &output_name) {
  if (output_name.size() >= 4 &&
      output_name.compare(output_name.size() - 4, 4, "_out") == 0) {
    return output_name.substr(0, output_name.size() - 4) + "_in";
  }
  return output_name;
}

int get_kv_row_length(const TensorInfo &tensor_info, bool is_key,
                      const std::string &tensor_name) {
  if (tensor_info.dimensions.size() < 2) {
    throw std::runtime_error("Unexpected KV dims for " + tensor_name);
  }

  if (is_key) {
    return tensor_info.dimensions.back();
  }

  return tensor_info.dimensions[tensor_info.dimensions.size() - 2];
}

void copy_kv_cache_window(uint8_t *dest, int dest_row_length,
                          const uint8_t *src, int src_row_length,
                          int history_length, bool is_key, int num_columns) {
  if (dest == nullptr || src == nullptr || history_length <= 0 ||
      dest_row_length <= 0 || src_row_length <= 0) {
    return;
  }

  const int available_history = std::min(history_length, src_row_length);
  const int copy_length = std::min(available_history, dest_row_length);
  const int src_start = available_history - copy_length;
  const bool align_to_tail =
    history_length >= src_row_length && dest_row_length > copy_length;
  const int dest_start = align_to_tail ? dest_row_length - copy_length : 0;

  if (is_key) {
    for (int col = 0; col < num_columns; ++col) {
      std::memcpy(dest + col * dest_row_length + dest_start,
                  src + col * src_row_length + src_start, copy_length);
    }
  } else {
    std::memcpy(dest + dest_start * num_columns, src + src_start * num_columns,
                copy_length * num_columns);
  }
}

std::vector<QnnKvOutputBinding> build_kv_output_bindings(
  const TensorInfoList &outputs,
  const std::unordered_map<std::string, int> &generation_kv_index_by_name,
  const std::string &graph_name, int kv_per_layer) {
  std::vector<QnnKvOutputBinding> bindings;
  for (size_t idx = 0; idx < outputs.size(); idx++) {
    const auto &name = outputs[idx].name;
    if (!qnn_starts_with(name, "past_")) {
      continue;
    }

    const auto input_name = kv_output_to_input_name(name);
    auto it = generation_kv_index_by_name.find(input_name);
    if (it == generation_kv_index_by_name.end()) {
      throw std::runtime_error(graph_name +
                               " KV output has no generation input: " + name);
    }

    const int kv_index = it->second;
    bindings.push_back({static_cast<int>(idx), kv_index,
                        kv_index / kv_per_layer,
                        qnn_starts_with(name, "past_key_")});
  }
  return bindings;
}

void append_outputs_to_kv_cache(const std::vector<IO_TensorType> &step_outputs,
                                const std::vector<QnnKvOutputBinding> &bindings,
                                const std::vector<uint8_t *> &kvs,
                                const std::vector<int> &kv_row_lengths,
                                int target_position, int rows,
                                int src_row_length,
                                const std::string &graph_name,
                                const std::vector<int> *kv_columns) {
  for (const auto &binding : bindings) {
    if (binding.output_index < 0 ||
        binding.output_index >= static_cast<int>(step_outputs.size()) ||
        binding.kv_index < 0 ||
        binding.kv_index >= static_cast<int>(kvs.size()) ||
        binding.layer_index < 0 ||
        binding.layer_index >= static_cast<int>(kv_row_lengths.size()) ||
        (kv_columns != nullptr &&
         binding.layer_index >= static_cast<int>(kv_columns->size()))) {
      throw std::runtime_error(graph_name +
                               " output KV binding is out of range");
    }
  }

#pragma omp parallel for
  for (int binding_idx = 0; binding_idx < static_cast<int>(bindings.size());
       binding_idx++) {
    const auto &binding = bindings[binding_idx];
    const int dest_row_length = kv_row_lengths[binding.layer_index];
    const int num_columns =
      kv_columns ? (*kv_columns)[binding.layer_index] : kQnnKvNumColumns;
    auto output = std::get<uint8_t *>(step_outputs[binding.output_index]);
    auto dest = kvs[binding.kv_index];

    int target_idx = target_position;
    const int valid_before = std::min(target_position, dest_row_length);
    const int shift = valid_before + rows - dest_row_length;
    if (shift > 0) {
      target_idx = valid_before - shift;
      if (binding.is_key) {
        for (int col = 0; col < num_columns; ++col) {
          uint8_t *col_base = dest + col * dest_row_length;
          std::memmove(col_base, col_base + shift, dest_row_length - shift);
        }
      } else {
        std::memmove(dest, dest + shift * num_columns,
                     (dest_row_length - shift) * num_columns);
      }
    }

    if (binding.is_key) {
      process_key(output, rows, num_columns, dest, target_idx, dest_row_length,
                  src_row_length);
    } else {
      process_value(output, rows, num_columns, dest, target_idx);
    }
  }
}

void process_key(uint8_t *pointer, int row, int column, uint8_t *dest, int idx,
                 int dest_row_length, int src_row_length) {
  // format: 1:1:col:row
  for (int i = 0; i < column; i++) {
    std::memcpy(dest + i * dest_row_length + idx, pointer + i * src_row_length,
                row);
  }
}

void process_value(uint8_t *pointer, int row, int column, uint8_t *dest,
                   int idx) {
  // format: 1:1:row:col
  for (int i = 0; i < row; i++) {
    std::memcpy(dest + (idx + i) * column, pointer + i * column, column);
  }
}

void fill_attention_mask_with_length(int rows, int columns, int length,
                                     uint16_t *attention_mask) {

  int index = 0;
  // TODO use std::fill_n?
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      if (i >= length) {
        attention_mask[index] = std::numeric_limits<uint16_t>::min();
      } else if (j >= columns - rows && j <= columns - rows + i) {
        attention_mask[index] = std::numeric_limits<uint16_t>::max();
      } else {
        attention_mask[index] = std::numeric_limits<uint16_t>::min();
      }
      index++;
    }
  }
}

void fill_attention_mask_with_prev_length(int rows, int columns, int length,
                                          uint16_t *attention_mask) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < length; j++) {
      attention_mask[i * columns + j] = std::numeric_limits<uint16_t>::max();
    }
  }
}

uint16_t *get_zero_memory(int size, int zero_point) {
  uint16_t *memory = (uint16_t *)allocate(size * sizeof(uint16_t));
  for (int i = 0; i < size; i++)
    memory[i] = zero_point;
  return memory;
}

void fill_generation_inputs_common(
  uint16_t *generation_attention_mask, int generation_attention_mask_elements,
  uint16_t *generation_sliding_attention_mask,
  int generation_sliding_attention_mask_elements,
  int generation_full_kv_past_length, int generation_sliding_kv_past_length,
  uint16_t *generation_position_ids_cos, uint16_t *generation_position_ids_sin,
  const uint16_t *position_ids_cos, const uint16_t *position_ids_sin,
  int pos_dim, uint16_t *generation_swa_position_ids_cos,
  uint16_t *generation_swa_position_ids_sin,
  const uint16_t *swa_position_ids_cos, const uint16_t *swa_position_ids_sin,
  int swa_pos_dim, int position, int rope_cache_seq_len) {
  if (position < 0 || position >= rope_cache_seq_len) {
    throw std::runtime_error("Generation position is out of rope cache");
  }

  std::fill_n(generation_attention_mask, generation_attention_mask_elements, 0);
  std::fill_n(generation_sliding_attention_mask,
              generation_sliding_attention_mask_elements, 0);

  // Self-attention slot: the current token's own KV position.
  // Using full_kv_past_length (= per-row elements - 1) rather than
  // attention_mask_elements - 1 ensures the slot lands in row 0 when the
  // generation graph batches multiple sequence positions
  // (generation_seq_len>1).
  generation_attention_mask[generation_full_kv_past_length] =
    std::numeric_limits<uint16_t>::max();
  generation_sliding_attention_mask[generation_sliding_kv_past_length] =
    std::numeric_limits<uint16_t>::max();

  for (int i = 0; i < position && i < generation_full_kv_past_length; i++) {
    generation_attention_mask[i] = std::numeric_limits<uint16_t>::max();
  }
  for (int i = 0; i < position && i < generation_sliding_kv_past_length; i++) {
    generation_sliding_attention_mask[i] = std::numeric_limits<uint16_t>::max();
  }

  std::memcpy(generation_position_ids_cos,
              position_ids_cos + position * pos_dim,
              pos_dim * sizeof(uint16_t));
  std::memcpy(generation_position_ids_sin,
              position_ids_sin + position * pos_dim,
              pos_dim * sizeof(uint16_t));
  std::memcpy(generation_swa_position_ids_cos,
              swa_position_ids_cos + position * swa_pos_dim,
              swa_pos_dim * sizeof(uint16_t));
  std::memcpy(generation_swa_position_ids_sin,
              swa_position_ids_sin + position * swa_pos_dim,
              swa_pos_dim * sizeof(uint16_t));
}

void fill_generation_inputs(
  float *generation_sample, int current_token,
  uint16_t *generation_attention_mask, int generation_attention_mask_elements,
  uint16_t *generation_sliding_attention_mask,
  int generation_sliding_attention_mask_elements,
  int generation_full_kv_past_length, int generation_sliding_kv_past_length,
  uint16_t *generation_position_ids_cos, uint16_t *generation_position_ids_sin,
  const uint16_t *position_ids_cos, const uint16_t *position_ids_sin,
  int pos_dim, uint16_t *generation_swa_position_ids_cos,
  uint16_t *generation_swa_position_ids_sin,
  const uint16_t *swa_position_ids_cos, const uint16_t *swa_position_ids_sin,
  int swa_pos_dim, int position, int rope_cache_seq_len) {
  generation_sample[0] = current_token;
  fill_generation_inputs_common(
    generation_attention_mask, generation_attention_mask_elements,
    generation_sliding_attention_mask,
    generation_sliding_attention_mask_elements, generation_full_kv_past_length,
    generation_sliding_kv_past_length, generation_position_ids_cos,
    generation_position_ids_sin, position_ids_cos, position_ids_sin, pos_dim,
    generation_swa_position_ids_cos, generation_swa_position_ids_sin,
    swa_position_ids_cos, swa_position_ids_sin, swa_pos_dim, position,
    rope_cache_seq_len);
}

void fill_generation_inputs_u16(
  uint16_t *generation_attention_mask, int generation_attention_mask_elements,
  uint16_t *generation_sliding_attention_mask,
  int generation_sliding_attention_mask_elements,
  int generation_full_kv_past_length, int generation_sliding_kv_past_length,
  uint16_t *generation_position_ids_cos, uint16_t *generation_position_ids_sin,
  const uint16_t *position_ids_cos, const uint16_t *position_ids_sin,
  int pos_dim, uint16_t *generation_swa_position_ids_cos,
  uint16_t *generation_swa_position_ids_sin,
  const uint16_t *swa_position_ids_cos, const uint16_t *swa_position_ids_sin,
  int swa_pos_dim, int position, int rope_cache_seq_len) {
  fill_generation_inputs_common(
    generation_attention_mask, generation_attention_mask_elements,
    generation_sliding_attention_mask,
    generation_sliding_attention_mask_elements, generation_full_kv_past_length,
    generation_sliding_kv_past_length, generation_position_ids_cos,
    generation_position_ids_sin, position_ids_cos, position_ids_sin, pos_dim,
    generation_swa_position_ids_cos, generation_swa_position_ids_sin,
    swa_position_ids_cos, swa_position_ids_sin, swa_pos_dim, position,
    rope_cache_seq_len);
}

int sample(uint16_t *pointer, int length, int *tokens, int number_of_tokens,
           float logit_scale, int logit_offset, float repetition_penalty,
           float temperature, float top_p, int top_k,
           std::mt19937 &random_engine, float final_logit_softcapping,
           const int32_t *allowed_token_bitmask,
           size_t allowed_token_bitmask_size) {
  if (pointer == nullptr)
    throw std::invalid_argument("Sampling logits must not be null");
  if (length <= 0)
    throw std::invalid_argument("Sampling vocabulary size must be positive");
  if (number_of_tokens < 0 || (number_of_tokens > 0 && tokens == nullptr)) {
    throw std::invalid_argument("Sampling token history is invalid");
  }
  if (top_k <= 0)
    throw std::invalid_argument("Sampling top_k must be positive");
  if (!std::isfinite(logit_scale) || logit_scale <= 0.0f)
    throw std::invalid_argument("Sampling logit_scale must be positive");
  if (!std::isfinite(repetition_penalty) || repetition_penalty <= 0.0f)
    throw std::invalid_argument("Sampling repetition_penalty must be positive");
  if (!std::isfinite(temperature) || temperature < 0.0f)
    throw std::invalid_argument("Sampling temperature must be non-negative");
  if (!std::isfinite(top_p) || top_p < 0.0f || top_p > 1.0f)
    throw std::invalid_argument("Sampling top_p must be in [0, 1]");
  if (!std::isfinite(final_logit_softcapping) ||
      final_logit_softcapping < 0.0f) {
    throw std::invalid_argument(
      "Sampling final_logit_softcapping must be non-negative");
  }
  if ((allowed_token_bitmask == nullptr) != (allowed_token_bitmask_size == 0)) {
    throw std::invalid_argument("Sampling token mask pointer/size mismatch");
  }

  const size_t required_mask_size = (static_cast<size_t>(length) + 31U) / 32U;
  if (allowed_token_bitmask != nullptr &&
      allowed_token_bitmask_size < required_mask_size) {
    throw std::invalid_argument(
      "Sampling token mask does not cover the vocabulary");
  }

  std::unordered_set<int> repeated_tokens;
  repeated_tokens.reserve(static_cast<size_t>(number_of_tokens));
  for (int i = 0; i < number_of_tokens; ++i) {
    if (tokens[i] >= 0 && tokens[i] < length)
      repeated_tokens.insert(tokens[i]);
  }

  struct Candidate {
    float logit;
    int token;
  };
  auto lower_priority = [](const Candidate &lhs, const Candidate &rhs) {
    if (lhs.logit != rhs.logit)
      return lhs.logit > rhs.logit;
    return lhs.token < rhs.token;
  };
  std::priority_queue<Candidate, std::vector<Candidate>,
                      decltype(lower_priority)>
    top_candidates(lower_priority);

  const auto is_allowed = [&](int token_id) {
    if (allowed_token_bitmask == nullptr)
      return true;
    const size_t block = static_cast<size_t>(token_id) / 32U;
    const auto bits = static_cast<uint32_t>(allowed_token_bitmask[block]);
    const auto bit = static_cast<unsigned int>(token_id) % 32U;
    return ((bits >> bit) & 1U) != 0;
  };

  const bool use_softcap = final_logit_softcapping > 0.0f;
  const float inverse_softcap =
    use_softcap ? 1.0f / final_logit_softcapping : 0.0f;
  for (int token_id = 0; token_id < length; ++token_id) {
    if (!is_allowed(token_id))
      continue;

    float logit =
      (static_cast<float>(pointer[token_id]) + logit_offset) * logit_scale;
    if (use_softcap) {
      logit = final_logit_softcapping * std::tanh(logit * inverse_softcap);
    }
    if (repeated_tokens.count(token_id) != 0) {
      logit =
        logit > 0.0f ? logit / repetition_penalty : logit * repetition_penalty;
    }
    if (!std::isfinite(logit))
      throw std::invalid_argument("Sampling produced a non-finite logit");

    const Candidate candidate{logit, token_id};
    if (top_candidates.size() < static_cast<size_t>(top_k)) {
      top_candidates.push(candidate);
    } else {
      const Candidate &lowest = top_candidates.top();
      if (candidate.logit > lowest.logit ||
          (candidate.logit == lowest.logit && candidate.token < lowest.token)) {
        top_candidates.pop();
        top_candidates.push(candidate);
      }
    }
  }

  if (top_candidates.empty())
    throw std::runtime_error("Sampling mask does not allow any token");

  std::vector<Candidate> candidates;
  candidates.reserve(top_candidates.size());
  while (!top_candidates.empty()) {
    candidates.push_back(top_candidates.top());
    top_candidates.pop();
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &lhs, const Candidate &rhs) {
              if (lhs.logit != rhs.logit)
                return lhs.logit > rhs.logit;
              return lhs.token < rhs.token;
            });

  if (temperature <= 1e-5f)
    return candidates.front().token;

  for (auto &candidate : candidates)
    candidate.logit /= temperature;

  const double max_logit = static_cast<double>(candidates.front().logit);
  std::vector<double> weights;
  weights.reserve(candidates.size());
  double weight_sum = 0.0;
  for (const auto &candidate : candidates) {
    const double weight =
      std::exp(static_cast<double>(candidate.logit) - max_logit);
    weights.push_back(weight);
    weight_sum += weight;
  }
  if (!std::isfinite(weight_sum) || weight_sum <= 0.0)
    throw std::runtime_error("Sampling probability normalization failed");

  size_t nucleus_size = 0;
  double cumulative_probability = 0.0;
  do {
    cumulative_probability += weights[nucleus_size] / weight_sum;
    ++nucleus_size;
  } while (nucleus_size < weights.size() &&
           cumulative_probability < static_cast<double>(top_p));

  std::discrete_distribution<size_t> distribution(
    weights.begin(), weights.begin() + nucleus_size);
  return candidates[distribution(random_engine)].token;
}
