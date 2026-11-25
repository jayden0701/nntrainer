// SPDX-License-Identifier: Apache-2.0
/** @file cache_bridge.cpp
 * @brief Stream reconstructed CPU KV into an unpublished target cache step.
 */
#include "cache_bridge.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

#include <causal_lm.h>
#include <cpu_backend.h>
#include <fp16.h>
#include <kv_cache/kv_cache_manager.h>
#include <layer_context.h>
#include <llm_util.hpp>

namespace causallm {
namespace {
using Clock = std::chrono::steady_clock;
using Tokens = std::vector<uint32_t>;
using nntrainer::kv::KvCacheManager;

double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
    .count();
}

struct ModelGeometry {
  std::string id;
  unsigned layers, heads, dimension;
  double theta;

  bool operator==(const ModelGeometry &other) const {
    return id == other.id && layers == other.layers && heads == other.heads &&
           dimension == other.dimension && theta == other.theta;
  }
};

ModelGeometry readGeometry(const json &model) {
  const auto &rope = model.at("rope");
  auto dimension = model.at("head_dim").get<unsigned>();
  auto theta = rope.at("theta").get<double>();
  if (model.at("key_space") != "post_rope" ||
      rope.at("layout") != "split_half" || rope.at("rotary_dim") != dimension ||
      !dimension || dimension % 2 || !std::isfinite(theta) || theta <= 0)
    throw std::invalid_argument(
      "CacheBridge requires unscaled split-half RoPE");
  return {model.at("model_id").get<std::string>(),
          model.at("layers").get<unsigned>(),
          model.at("kv_heads").get<unsigned>(), dimension, theta};
}

std::vector<float> readCoefficients(const std::filesystem::path &path,
                                    size_t count) {
  if (std::filesystem::file_size(path) != count * sizeof(float))
    throw std::invalid_argument("Unexpected CacheBridge coefficient file size");
  std::vector<float> data(count);
  std::ifstream file(path, std::ios::binary);
  if (!file.read(reinterpret_cast<char *>(data.data()),
                 count * sizeof(float)) ||
      std::any_of(data.begin(), data.end(),
                  [](float x) { return !std::isfinite(x); }))
    throw std::invalid_argument("Invalid CacheBridge coefficients");
  return data;
}

struct Mapper {
  unsigned k, dimension, width, features, outputs, units;
  std::vector<std::vector<unsigned>> layers;
  std::vector<float> weights, bias;

  Mapper(const std::filesystem::path &path, const ModelGeometry &source,
         const ModelGeometry &target) {
    json metadata;
    std::ifstream(path / "metadata.json") >> metadata;
    if (metadata.at("version") != 1 || metadata.at("fit_space") != "pre_rope" ||
        !(readGeometry(metadata.at("source")) == source) ||
        !(readGeometry(metadata.at("target")) == target) ||
        source.heads != target.heads || source.dimension != target.dimension)
      throw std::invalid_argument("CacheBridge mapper geometry mismatch");
    const bool compact = metadata.at("layout") == "headwise";
    if (!compact && metadata.at("layout") != "all-head")
      throw std::invalid_argument("Unknown CacheBridge mapper layout");
    layers =
      metadata.at("source_layers").get<std::vector<std::vector<unsigned>>>();
    if (layers.size() != target.layers || layers.empty() ||
        layers.front().empty() || layers.front().size() > source.layers)
      throw std::invalid_argument("Invalid CacheBridge mapper layer count");
    k = layers.front().size();
    for (const auto &row : layers) {
      std::vector<bool> seen(source.layers);
      if (row.size() != k)
        throw std::invalid_argument("Invalid CacheBridge source layer count");
      for (auto layer : row) {
        if (layer >= source.layers || seen[layer])
          throw std::invalid_argument("Invalid CacheBridge source layer");
        seen[layer] = true;
      }
    }
    dimension = source.dimension;
    width = source.heads * dimension;
    units = compact ? source.heads : 1;
    outputs = width / units;
    features = k * outputs;
    const size_t coefficients = size_t{2} * target.layers * units * outputs;
    weights = readCoefficients(path / "weights.f32", coefficients * features);
    bias = readCoefficients(path / "bias.f32", coefficients);
  }
};

struct RotaryBlock {
  std::vector<float> cosine;
  std::vector<float> sine;

  RotaryBlock(unsigned rows, unsigned dim) :
    cosine(size_t{rows} * (dim / 2)), sine(cosine.size()) {}

  void fill(unsigned from, unsigned rows, unsigned dim, double theta) {
    const unsigned half = dim / 2;
    for (unsigned d = 0; d < half; ++d) {
      const double frequency = std::pow(theta, -2.0 * d / dim);
      for (unsigned row = 0; row < rows; ++row) {
        const double angle = (from + row) * frequency;
        cosine[row * half + d] = std::cos(angle);
        sine[row * half + d] = std::sin(angle);
      }
    }
  }

  void rotate(float *head, unsigned dim, unsigned row, bool inverse) const {
    const auto half = dim / 2;
    for (unsigned d = 0; d < half; ++d) {
      const float c = cosine[row * half + d];
      const float s = (inverse ? -1 : 1) * sine[row * half + d];
      const float left = head[d], right = head[d + half];
      head[d] = left * c - right * s;
      head[d + half] = right * c + left * s;
    }
  }

  size_t bytes() const { return (cosine.size() + sine.size()) * sizeof(float); }
};

struct TransferStats {
  double gather_ms = 0, affine_ms = 0, install_ms = 0;
  size_t scratch = 0;
};

/** Install each result block directly into write views of one target step. */
TransferStats install(KvCacheManager &source, KvCacheManager &target,
                      const ModelGeometry &source_geometry,
                      const ModelGeometry &target_geometry,
                      const Tokens &tokens, const Mapper &mapper,
                      unsigned block) {
  if (&source == &target || source.hasActiveStep() || !source.valid() ||
      source.committedLength() != tokens.size() ||
      source.committedTokens() != tokens || target.committedLength() != 0)
    throw std::invalid_argument(
      "CacheBridge requires a committed source and empty target");

  auto step = target.beginStep(tokens.size(), tokens.data());
  block = std::min<unsigned>(block, tokens.size());
  std::vector<float> input(size_t{block} * mapper.features);
  std::vector<float> output(size_t{block} * mapper.outputs);
  RotaryBlock source_rope(block, mapper.dimension),
    target_rope(block, mapper.dimension);
  TransferStats stats;
  stats.scratch = (input.size() + output.size()) * sizeof(float) +
                  source_rope.bytes() + target_rope.bytes();
  using Plane = std::shared_ptr<nntrainer::MemoryData>;
  std::vector<Plane> source_keys, source_values;
  for (unsigned layer = 0; layer < source_geometry.layers; ++layer) {
    source_keys.push_back(source.keyMemory(layer));
    source_values.push_back(source.valueMemory(layer));
  }
  using Views = std::vector<nntrainer::kv::KvView>;
  std::vector<Views> target_keys, target_values;
  for (unsigned layer = 0; layer < target_geometry.layers; ++layer) {
    target_keys.push_back(target.writeKey(step, layer));
    target_values.push_back(target.writeValue(step, layer));
  }
  for (unsigned from = 0; from < tokens.size(); from += block) {
    const unsigned rows = std::min<unsigned>(block, tokens.size() - from);
    auto start = Clock::now();
    source_rope.fill(from, rows, mapper.dimension, source_geometry.theta);
    target_rope.fill(from, rows, mapper.dimension, target_geometry.theta);
    stats.gather_ms += elapsed(start);
    for (unsigned plane = 0; plane < 2; ++plane) {
      const auto &source_planes = plane ? source_values : source_keys;
      auto &target_planes = plane ? target_values : target_keys;
      for (unsigned layer = 0; layer < target_geometry.layers; ++layer) {
        auto *destination =
          static_cast<uint16_t *>(target_planes[layer].at(0).writableData());
        for (unsigned unit = 0; unit < mapper.units; ++unit) {
          start = Clock::now();
          for (unsigned j = 0; j < mapper.k; ++j) {
            const auto *data =
              source_planes[mapper.layers[layer][j]]->getAddr<uint16_t>();
            for (unsigned row = 0; row < rows; ++row) {
              const auto *src = data + size_t{from + row} * mapper.width +
                                unit * mapper.outputs;
              auto *dst = input.data() + size_t{row} * mapper.features +
                          j * mapper.outputs;
              for (unsigned d = 0; d < mapper.outputs; ++d)
                dst[d] = nntrainer::compute_fp16_to_fp32(src[d]);
              if (plane == 0)
                for (unsigned d = 0; d < mapper.outputs; d += mapper.dimension)
                  source_rope.rotate(dst + d, mapper.dimension, row, true);
            }
          }
          stats.gather_ms += elapsed(start);

          start = Clock::now();
          const size_t offset =
            ((size_t{plane} * target_geometry.layers + layer) * mapper.units +
             unit) *
            mapper.outputs;
          nntrainer::sgemm(0, false, false, rows, mapper.outputs,
                           mapper.features, 1.0f, input.data(), mapper.features,
                           mapper.weights.data() + offset * mapper.features,
                           mapper.outputs, 0.0f, output.data(), mapper.outputs);
          stats.affine_ms += elapsed(start);

          start = Clock::now();
          for (unsigned row = 0; row < rows; ++row) {
            auto *src = output.data() + size_t{row} * mapper.outputs;
            auto *dst = destination + size_t{from + row} * mapper.width +
                        unit * mapper.outputs;
            for (unsigned d = 0; d < mapper.outputs; ++d)
              src[d] += mapper.bias[offset + d];
            if (plane == 0)
              for (unsigned d = 0; d < mapper.outputs; d += mapper.dimension)
                target_rope.rotate(src + d, mapper.dimension, row, false);
            for (unsigned d = 0; d < mapper.outputs; ++d) {
              if (!std::isfinite(src[d]) || std::abs(src[d]) > 65504.0f)
                throw std::runtime_error(
                  "CacheBridge output exceeds FP16 range");
              dst[d] = nntrainer::compute_fp32_to_fp16(src[d]);
            }
          }
          stats.install_ms += elapsed(start);
        }
      }
    }
  }
  for (unsigned layer = 0; layer < target_geometry.layers; ++layer)
    target.markReady(step, layer);
  target.commit(std::move(step));
  return stats;
}

double logProbability(const float *logits, unsigned vocab, uint32_t token) {
  const float maximum = *std::max_element(logits, logits + vocab);
  double mass = 0;
  for (unsigned i = 0; i < vocab; ++i)
    mass += std::exp(static_cast<double>(logits[i]) - maximum);
  const double result =
    static_cast<double>(logits[token]) - maximum - std::log(mass);
  if (!std::isfinite(result))
    throw std::runtime_error("CacheBridge scoring produced non-finite logits");
  return result;
}
} // namespace

/** Runs one synchronous request at a time on two private model caches. */
class CacheBridge {
public:
  CacheBridge(CausalLM &source, CausalLM &target, const json &config) :
    source(source),
    target(target),
    source_geometry(
      geometry(source, config.at("source_id").get<std::string>())),
    target_geometry(
      geometry(target, config.at("target_id").get<std::string>())),
    block(config.value("block_tokens", 128u)),
    head(findHead(target)) {
    if (!block || block > 2048 || source.NUM_VOCAB != target.NUM_VOCAB)
      throw std::invalid_argument(
        "Invalid CacheBridge block or vocabulary size");
    if (!config.at("mappers").is_object())
      throw std::invalid_argument("CacheBridge mappers must be a named object");
    for (auto &[name, path] : config.at("mappers").items()) {
      if (name == "cold" || name == "exact")
        throw std::invalid_argument("Reserved CacheBridge mode name");
      mappers.try_emplace(name, path.get<std::string>(), source_geometry,
                          target_geometry);
    }
  }

  json run(const json &request) {
    const auto history = readTokens(request.at("history"), source.NUM_VOCAB);
    const auto query = readTokens(request.at("query"), target.NUM_VOCAB);
    const auto choices = request.value("options", json::array());
    if (!choices.is_array())
      throw std::invalid_argument("CacheBridge options must be an array");
    std::vector<Tokens> options;
    for (const auto &choice : choices)
      options.push_back(readTokens(choice, target.NUM_VOCAB));
    if (query.empty() || history.size() > source.MAX_SEQ_LEN ||
        history.size() + query.size() > target.MAX_SEQ_LEN)
      throw std::invalid_argument("CacheBridge request exceeds model context");
    for (const auto &option : options) {
      if (option.empty() || history.size() + query.size() + option.size() - 1 >
                              target.MAX_SEQ_LEN)
        throw std::invalid_argument("CacheBridge option exceeds model context");
    }
    Tokens prompt = history;
    prompt.insert(prompt.end(), query.begin(), query.end());

    if ((request.contains("max_new_tokens") &&
         !request.at("max_new_tokens").is_number_unsigned()) ||
        (request.contains("seed") && !request.at("seed").is_number_unsigned()))
      throw std::invalid_argument("Invalid CacheBridge generation parameters");
    const auto cap = request.value("max_new_tokens", uint64_t{0});
    const auto stops =
      readTokens(request.value("stop_ids", json::array()), target.NUM_VOCAB);
    if ((request.contains("max_new_tokens") && cap == 0) ||
        (cap && (!options.empty() || stops.empty() ||
                 cap - 1 > target.MAX_SEQ_LEN - prompt.size())) ||
        (!cap && (!stops.empty() || request.contains("seed"))))
      throw std::invalid_argument("Invalid CacheBridge generation controls");
    const auto seed = request.value("seed", uint64_t{0});

    const auto modes = request.at("modes").get<std::vector<std::string>>();
    const bool reuse = request.value("reuse_history", false);
    if (modes.empty() || (reuse && (modes.size() != 1 || options.empty())))
      throw std::invalid_argument(
        "History reuse requires a single scoring mode");
    const auto prepare = Clock::now();
    if (!history.empty() && !mappers.empty()) {
      auto *source_cache = source.kv_cache.coreCPU();
      if (!source_cache->valid() ||
          source_cache->committedTokens() != history) {
        source_cache->reset();
        forward(source, history, 0, false);
      }
    }
    json result = {{"id", request.at("id")},
                   {"history_tokens", history.size()},
                   {"query_tokens", query.size()},
                   {"source_prefill_ms", elapsed(prepare)}};
    auto run_mode = [&](const std::string &name, const Mapper *mapper) {
      auto &cache = *target.kv_cache.coreCPU();
      const bool reuse_ready = reuse && prefix_mode == name &&
                               prefix_history == history && cache.valid() &&
                               cache.committedLength() >= history.size();
      if (!reuse_ready)
        cache.reset();
      prefix_history.clear();
      prefix_mode.clear();
      const auto start = Clock::now();
      TransferStats stats;
      std::string fallback;
      if (!reuse_ready && mapper && !history.empty()) {
        try {
          stats = install(*source.kv_cache.coreCPU(), cache, source_geometry,
                          target_geometry, history, *mapper, block);
        } catch (const std::exception &error) {
          fallback = error.what();
          cache.reset();
        }
      } else if (!reuse_ready && !history.empty() &&
                 (name == "exact" || (reuse && name == "cold"))) {
        forward(target, history, 0, false);
      }
      const auto install_ms = elapsed(start);
      const auto query_start = Clock::now();
      auto logits = cache.committedLength()
                      ? forward(target, query, history.size(), true)
                      : forward(target, prompt, 0, true);
      SamplingRng rng{seed};
      const auto token =
        cap
          ? sample(logits, rng)
          : static_cast<uint32_t>(
              std::max_element(logits.begin(), logits.end()) - logits.begin());
      const auto ttft = elapsed(start);
      json mode = {{"token", token},
                   {"ttft_ms", ttft},
                   {"target_query_ms", elapsed(query_start)},
                   {"cache_prepare_ms", install_ms},
                   {"gather_rope_ms", stats.gather_ms},
                   {"affine_ms", stats.affine_ms},
                   {"install_ms", stats.install_ms},
                   {"mapper_scratch_bytes", stats.scratch}};
      if (!fallback.empty())
        mode["fallback"] = fallback;
      if (reuse && fallback.empty()) {
        prefix_history = history;
        prefix_mode = name;
        mode["reused_history"] = reuse_ready;
      }
      if (cap)
        generate(mode, token, stops, static_cast<size_t>(cap), prompt.size(),
                 start, rng);
      if (!options.empty())
        mode["scores"] = scoreOptions(options, logits, prompt.size());
      result[name] = std::move(mode);
    };

    for (const auto &name : modes) {
      if (result.contains(name))
        throw std::invalid_argument("Duplicate CacheBridge mode");
      if (name == "cold" || name == "exact")
        run_mode(name, nullptr);
      else
        run_mode(name, &mappers.at(name));
    }
    return result;
  }

private:
  static ModelGeometry geometry(CausalLM &lm, const std::string &id) {
    const auto &config = lm.system_prefix_config_;
    if (config.value("model_type", std::string()) != "qwen3" ||
        config.value("rope_interleaved", false) ||
        config.value("partial_rotary_factor", 1.0) != 1.0 ||
        (config.contains("rope_scaling") &&
         !config["rope_scaling"].is_null()) ||
        lm.BATCH_SIZE != 1 || lm.SKIP_PREFILL || causallm_engine() != "cpu")
      throw std::invalid_argument(
        "CacheBridge requires batch-one CPU Qwen3 with default RoPE");
    if (lm.NUM_LAYERS <= 0 || lm.NUM_KEY_VALUE_HEADS <= 0 || lm.HEAD_DIM <= 0)
      throw std::invalid_argument("Invalid CacheBridge model geometry");
    const ModelGeometry model{id, static_cast<unsigned>(lm.NUM_LAYERS),
                              static_cast<unsigned>(lm.NUM_KEY_VALUE_HEADS),
                              static_cast<unsigned>(lm.HEAD_DIM),
                              static_cast<double>(lm.ROPE_THETA)};
    if (!lm.kv_cache.isAllocated())
      lm.allocateAndBindKVCache();
    auto *manager = lm.kv_cache.coreCPU();
    if (!manager || manager->paged())
      throw std::invalid_argument(
        "CacheBridge requires the contiguous CPU KV manager");
    for (unsigned l = 0; l < model.layers; ++l) {
      const auto &layer = manager->layerSpec(l);
      if (layer.encoding != nntrainer::kv::Encoding::Float16 ||
          layer.capacity != layer.max_tokens || layer.source >= 0 ||
          layer.num_kv_heads != model.heads ||
          layer.head_dim != model.dimension || !layer.has_producer)
        throw std::invalid_argument("Invalid CacheBridge FP16 KV geometry");
    }
    return model;
  }

  static Tokens readTokens(const json &ids, unsigned vocab) {
    if (!ids.is_array())
      throw std::invalid_argument("CacheBridge tokens must be an array");
    Tokens tokens;
    tokens.reserve(ids.size());
    for (const auto &id : ids) {
      if (!id.is_number_unsigned() || id.get<uint64_t>() >= vocab)
        throw std::invalid_argument("CacheBridge token outside the vocabulary");
      tokens.push_back(id.get<uint32_t>());
    }
    return tokens;
  }

  static nntrainer::RunLayerContext *findHead(CausalLM &lm) {
    nntrainer::RunLayerContext *head = nullptr;
    lm.model->forEachLayer(
      [&](ml::train::Layer &layer, nntrainer::RunLayerContext &ctx, void *) {
        if (layer.getName() == "output_of_causallm" &&
            layer.getType() == "tie_word_embeddings" &&
            ctx.getNumInputs() == 1 && ctx.getNumWeights() >= 1 &&
            ctx.getInput(0).getDataType() ==
              ml::train::TensorDim::DataType::FP32 &&
            ctx.getWeight(0).getDataType() ==
              ml::train::TensorDim::DataType::FP32)
          head = &ctx;
      },
      nullptr);
    return head;
  }

  static std::vector<float> forward(CausalLM &lm, const Tokens &tokens,
                                    unsigned position, bool logits) {
    lm.setKVCachePosition(position);
    std::vector<float> values(lm.INIT_SEQ_LEN);
    auto inputs = lm.kvInferenceInputs(values.data());
    std::vector<float> result;
    for (unsigned index = 0; index < tokens.size();) {
      const unsigned count =
        std::min<unsigned>(lm.INIT_SEQ_LEN, tokens.size() - index);
      std::copy_n(tokens.begin() + index, count, values.begin());
      lm.beforePrefillBlock(position, position + count);
      auto out = lm.incrementalInference(
        1, inputs, count, position, position + count, tokens.data() + index);
      position += count;
      index += count;
      if (logits && index == tokens.size()) {
        if (out.empty())
          throw std::runtime_error("CacheBridge forward returned no logits");
        try {
          result.assign(out.front(), out.front() + lm.NUM_VOCAB);
        } catch (...) {
          for (auto *row : out)
            lm.releaseLogitsBuf(row);
          throw;
        }
      }
      for (auto *row : out)
        lm.releaseLogitsBuf(row);
    }
    return result;
  }

  static uint32_t sample(const std::vector<float> &logits, SamplingRng &rng) {
    // Fixed settings keep generation comparisons paired across cache modes.
    return applyTKP(logits.data(), logits.size(), 0.6f, 20, 0.95f, rng);
  }

  void generate(json &mode, uint32_t first, const Tokens &stops, size_t cap,
                size_t position, Clock::time_point start, SamplingRng &rng) {
    Tokens generated{first};
    std::vector<double> step_ms;
    auto stopped = [&] {
      return std::find(stops.begin(), stops.end(), generated.back()) !=
             stops.end();
    };
    const auto decoding = Clock::now();
    while (generated.size() < cap && !stopped()) {
      const auto step = Clock::now();
      const auto logits =
        forward(target, Tokens{generated.back()}, position++, true);
      generated.push_back(sample(logits, rng));
      step_ms.push_back(elapsed(step));
    }
    mode["decode_ms"] = elapsed(decoding);
    mode["e2e_ms"] = elapsed(start);
    mode["finish"] = stopped() ? "eos" : "limit";
    mode["output_ids"] = std::move(generated);
    mode["step_ms"] = std::move(step_ms);
  }

  std::vector<double> scoreOptions(const std::vector<Tokens> &options,
                                   const std::vector<float> &logits,
                                   unsigned position) {
    std::vector<double> scores;
    for (const auto &option : options) {
      double sum =
        logProbability(logits.data(), target.NUM_VOCAB, option.front());
      unsigned scored = 1;
      while (scored < option.size()) {
        const unsigned remaining = option.size() - scored;
        const unsigned count =
          head ? std::min({remaining, target.INIT_SEQ_LEN, 64u}) : 1;
        Tokens feed(option.begin() + scored - 1,
                    option.begin() + scored - 1 + count);
        auto last = forward(target, feed, position + scored - 1, true);
        if (count == 1) {
          sum += logProbability(last.data(), target.NUM_VOCAB, option[scored]);
        } else {
          auto &hidden = head->getInput(0);
          auto &weight = head->getWeight(0);
          auto dim = hidden.getDim();
          dim.height(count);
          auto input = hidden.getSharedDataTensor(dim, 0, true);
          dim.width(target.NUM_VOCAB);
          std::vector<float> rows(size_t{count} * target.NUM_VOCAB);
          auto output = nntrainer::Tensor::Map(
            rows.data(), rows.size() * sizeof(float), dim, 0);
          input.dot(weight, output, false, true);
          for (unsigned row = 0; row < count; ++row)
            sum += logProbability(rows.data() + size_t{row} * target.NUM_VOCAB,
                                  target.NUM_VOCAB, option[scored + row]);
        }
        scored += count;
      }
      scores.push_back(sum / option.size());
    }
    return scores;
  }

  CausalLM &source, &target;
  const ModelGeometry source_geometry, target_geometry;
  const unsigned block;
  std::map<std::string, Mapper> mappers;
  nntrainer::RunLayerContext *head;
  Tokens prefix_history;
  std::string prefix_mode;
};

void runCacheBridge(CausalLM &source, CausalLM &target, const json &config) {
  CacheBridge bridge(source, target, config);
  const auto &requests = config.at("requests");
  if (!requests.is_array() || requests.empty())
    throw std::invalid_argument("CacheBridge requires at least one request");
  for (const auto &request : requests) {
    const auto result = bridge.run(request).dump();
    std::cout << "[CACHEBRIDGE] " << result << std::endl;
  }
}
} // namespace causallm
