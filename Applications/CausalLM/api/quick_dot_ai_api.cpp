// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    quick_dot_ai_api.cpp
 * @date    21 Jan 2026
 * @brief   This is a C API for CausalLM application
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "quick_dot_ai_api.h"
#include "quick_dot_ai_api_internal.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#ifndef _WIN32
#include <cxxabi.h>
#endif

#include "callback_streamer.h"
#include "causal_lm.h"
#include "chat_template.h"
#include "gemma3_causallm.h"
#include "gemma4_causallm.h"
#include "gptoss_cached_slim_causallm.h"
#include "gptoss_causallm.h"
#include "json.hpp"
#include "model_callbacks.h"
#include "model_config_internal.h"
#include "model_descriptor.h"
#include "multilingual_tinybert_16mb.h"
#include "openai_request.h"
#include "qwen2_causallm.h"
#include "qwen3_cached_slim_moe_causallm.h"
#include "qwen3_causallm.h"
#include "qwen3_moe_causallm.h"
#include "qwen3_slim_moe_causallm.h"
#include "sentence_transformer.h"
#include "xgrammar_manager.h"
#include "xgrammar_wrapper.h"
#include <factory.h>
#ifdef ENABLE_QNN_MODELS
#include "gemma4_e2b_qnn.h"
#include "quick_dot_ai_qnn.h"
#endif
#include <fstream>
#include <sys/stat.h>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "QuickAI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define LOGD(fmt, ...) fprintf(stdout, fmt "\n", ##__VA_ARGS__)
#define LOGE(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#endif

using json = nlohmann::json;

static int set_environment_variable(const char *name, const char *value) {
#ifdef _WIN32
  return _putenv_s(name, value);
#else
  return setenv(name, value, 1);
#endif
}

/**
 * @brief Per-handle state for a loaded CausalLM model instance.
 *
 * Each handle may carry one or more sub-models so that compositions like a
 *
 * vision encoder plus LLM can live behind a single owner. The parallel model
 *
 * metadata vectors use the same index. Generation selects the text model and
 *
 * optionally drives the complete composition through quickAiRunOpenAI().
 *
 * Note: the legacy non-handle API (loadModel / ...) is
 * implemented on top of a single static "default" instance of this struct
 * so that existing callers (e.g. test_api) keep working unchanged.
 */
struct CausalLmModel {
  std::mutex mtx;
  std::mutex cancellation_mtx;
  bool run_announced = false;
  bool run_active = false;
  bool cancellation_pending = false;
  std::vector<std::unique_ptr<causallm::Transformer>> models;
  std::vector<causallm::Transformer *> cancellation_targets;
  std::vector<std::string> architectures;
  std::vector<std::string> model_dirs;
  std::vector<std::optional<causallm::ChatTemplate>> chat_templates;
  std::unique_ptr<causallm::XGrammarManager> grammar_manager;
  std::unordered_map<std::string, std::string> dynamic_grammar_schemas;
  std::string last_output;
  std::string native_lib_dir;
  std::vector<double> initialization_duration_ms;
  bool verbose = false;
  std::string chat_template_name;
  bool initialized = false;
  int kv_len = 0;
};

/**
 * Clear owned models without racing a cross-thread cancellation request.
 *
 * Callers must already hold @c h.mtx; cancelModelHandle only takes the
 *
 * cancellation mutex, so it never waits for a long-running inference.
 */
static void clear_handle_models(CausalLmModel &h) {
  std::lock_guard<std::mutex> cancellation_lock(h.cancellation_mtx);
  h.run_announced = false;
  h.run_active = false;
  h.cancellation_pending = false;
  h.cancellation_targets.clear();
  h.models.clear();
}

static bool is_valid_backend(BackendType backend) {
  return backend >= CAUSAL_LM_BACKEND_CPU && backend <= CAUSAL_LM_BACKEND_NPU;
}

static bool is_valid_quantization(ModelQuantizationType quantization) {
  return quantization >= CAUSAL_LM_QUANTIZATION_UNKNOWN &&
         quantization <= CAUSAL_LM_QUANTIZATION_W32A32;
}

/** Publish stable, non-owning targets after a load has fully succeeded. */
static void publish_cancellation_targets(CausalLmModel &h) {
  std::lock_guard<std::mutex> cancellation_lock(h.cancellation_mtx);
  h.cancellation_targets.clear();
  h.cancellation_targets.reserve(h.models.size());
  for (const auto &model : h.models) {
    if (model)
      h.cancellation_targets.push_back(model.get());
  }
}

/** Own the cancellation state for the complete parse-to-generation request. */
class ScopedRunRequest final {
public:
  explicit ScopedRunRequest(CausalLmModel &handle) : handle_(handle) {
    std::lock_guard<std::mutex> cancellation_lock(handle_.cancellation_mtx);
    if (!handle_.run_announced) {
      handle_.run_announced = true;
      handle_.cancellation_pending = false;
    }
  }

  ScopedRunRequest(const ScopedRunRequest &) = delete;
  ScopedRunRequest &operator=(const ScopedRunRequest &) = delete;

  ~ScopedRunRequest() {
    std::lock_guard<std::mutex> cancellation_lock(handle_.cancellation_mtx);
    if (!handle_.run_active) {
      handle_.run_announced = false;
      handle_.cancellation_pending = false;
    }
  }

private:
  CausalLmModel &handle_;
};

/**
 * Mark a handle as actively generating and prepare every component for one

 * * cancellation epoch. The cancellation mutex serializes the transition with

 * * cancelModelHandle(), preventing one request's stop from reaching the next.

 */
class ScopedGeneration final {
public:
  explicit ScopedGeneration(CausalLmModel &handle) : handle_(handle) {
    std::lock_guard<std::mutex> cancellation_lock(handle_.cancellation_mtx);
    for (auto *model : handle_.cancellation_targets) {
      if (model != nullptr)
        model->prepareForRun();
    }
    handle_.run_active = true;
    if (handle_.cancellation_pending) {
      handle_.cancellation_pending = false;
      for (auto *model : handle_.cancellation_targets) {
        if (model != nullptr)
          model->requestStop();
      }
    }
  }

  ScopedGeneration(const ScopedGeneration &) = delete;
  ScopedGeneration &operator=(const ScopedGeneration &) = delete;

  ~ScopedGeneration() {
    std::lock_guard<std::mutex> cancellation_lock(handle_.cancellation_mtx);
    handle_.run_active = false;
    handle_.run_announced = false;
    handle_.cancellation_pending = false;
  }

private:
  CausalLmModel &handle_;
};

static std::mutex g_registry_mutex;
static std::mutex g_options_mutex;
static bool g_default_verbose = false;
static std::string g_default_chat_template_name;

// Default handle backing the legacy non-handle API.
static CausalLmModel &get_default_handle() {
  static CausalLmModel instance;
  return instance;
}

class XGrammarLogitsProcessor final : public causallm::LogitsProcessor {
public:
  explicit XGrammarLogitsProcessor(causallm::XGrammar *grammar,
                                   std::function<void()> on_completed = {}) :
    grammar_(grammar), on_completed_(std::move(on_completed)) {}

  void process(float *logits, unsigned int vocab_size,
               unsigned int batch_index) override {
    if (batch_index != 0 || grammar_ == nullptr ||
        !grammar_->isGrammarEnabled()) {
      return;
    }
    grammar_->applyGrammarMask(logits, static_cast<int>(vocab_size));
  }

  void acceptToken(unsigned int token_id, unsigned int batch_index) override {
    if (batch_index != 0 || grammar_ == nullptr ||
        grammar_->getGrammarMatcher() == nullptr) {
      return;
    }
    auto *matcher = grammar_->getGrammarMatcher();
    if (!matcher->AcceptToken(static_cast<int32_t>(token_id))) {
      failed_ = true;
      if (on_completed_) {
        on_completed_();
      }
      return;
    }
    if (matcher->IsCompleted() || matcher->IsTerminated()) {
      if (on_completed_) {
        on_completed_();
      }
      return;
    }
    grammar_->getGrammarMatcher()->FillNextTokenBitmask(
      &grammar_->getBitmaskTensor());
  }

  void reset() override {
    failed_ = false;
    if (grammar_ != nullptr) {
      grammar_->resetGrammar();
    }
  }

  bool failed() const { return failed_; }

private:
  causallm::XGrammar *grammar_;
  std::function<void()> on_completed_;
  bool failed_ = false;
};

#ifdef ENABLE_QNN_MODELS
static causallm::Quick_Dot_AI_QNN *as_qnn_model(causallm::Transformer *model) {
  return dynamic_cast<causallm::Quick_Dot_AI_QNN *>(model);
}
#endif

static bool model_supports_text_output(causallm::Transformer *model) {
  return model != nullptr && model->supportsTextGeneration();
}

static bool get_model_output(causallm::Transformer *model,
                             std::string &output) {
  if (!model_supports_text_output(model))
    return false;
  output = model->getOutput(0);
  return true;
}

static bool set_model_streamer(causallm::Transformer *model,
                               ::BaseStreamer *streamer) {
  if (!model_supports_text_output(model))
    return false;
  model->setStreamer(streamer);
  return true;
}

static bool request_model_stop(causallm::Transformer *model) {
  if (!model_supports_text_output(model))
    return false;
  model->requestStop();
  return true;
}

static std::map<std::string, std::string> g_model_path_map = {
  {"QWEN3-0.6B", "qwen3-0.6b"},
  {"QWEN3-1.7B-Q40", "qwen3-1.7b-q40-arm"},
  {"TINY_BERT", "tiny_bert"},
  {"FUNCTION_GEMMA", "function_gemma"},
  {"GEMMA4_CPU", "gemma4_cpu"},
  {"OURO_EMBEDDING", "ouro_embedding"},
#ifdef ENABLE_QNN_MODELS
  {"GEMMA4-E2B-QNN", "gemma-4-e2b-qnn"},
  {"VJEPA2-QNN", "vjepa2-qnn"},
#endif
};

/**
 * @brief RegisteredModel
 */
struct RegisteredModel {
  std::string arch_name;
  ModelRuntimeConfig config;
};
static std::map<std::string, RegisteredModel> g_model_registry;
static std::map<std::string, ModelArchConfig> g_arch_config_map;

// Internal C++ registration functions — called from model_config.cpp
// These bypass extern "C" PLT and write directly to our static maps.
namespace quick_dot_ai {

void register_arch(const char *arch_name, ModelArchConfig config) {
  std::lock_guard<std::mutex> lock(g_registry_mutex);
  std::string name(arch_name);
  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  g_arch_config_map[name] = config;
}

void register_model(const char *model_name, const char *arch_name,
                    ModelRuntimeConfig config) {
  std::lock_guard<std::mutex> lock(g_registry_mutex);
  std::string name(model_name);
  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  std::string aname(arch_name);
  std::transform(aname.begin(), aname.end(), aname.begin(), ::toupper);
  g_model_registry[name] = {aname, config};
}

} // namespace quick_dot_ai

// ---------------------------------------------------------------------------
// T4: string-id descriptor registry + catalog JSON
// ---------------------------------------------------------------------------

// Lazily-constructed (Meyers singleton) so cross-library self-registration
// from model plugins (libquick_dot_ai.so) — whose static constructors run
// BEFORE this lib's globals would be constructed — lands in a live registry
// instead of an uninitialized one (static-init-order fiasco).
static std::mutex &descriptor_mutex() {
  static std::mutex m;
  return m;
}
static std::vector<ModelDescriptor> &descriptor_registry() {
  static std::vector<ModelDescriptor> v;
  return v;
}

namespace quick_dot_ai {
void register_model_descriptor(const ModelDescriptor *desc) {
  if (!desc || !desc->id)
    return;
  std::lock_guard<std::mutex> lock(descriptor_mutex());
  for (auto &d : descriptor_registry()) {
    if (std::strcmp(d.id, desc->id) == 0) {
      LOGE("register_model_descriptor: duplicate id '%s', overwriting",
           desc->id);
      d = *desc;
      return;
    }
  }
  descriptor_registry().push_back(*desc);
}
} // namespace quick_dot_ai

/** Find a descriptor by string id. Returns a copy while locked, or nullopt. */
static std::optional<ModelDescriptor> find_descriptor_by_id(const char *id) {
  if (!id)
    return std::nullopt;
  std::lock_guard<std::mutex> lk(descriptor_mutex());
  for (const auto &d : descriptor_registry())
    if (std::strcmp(d.id, id) == 0)
      return d; // copy while locked
  return std::nullopt;
}

// Library-owned buffer: rebuilt on every call and returned via c_str().
// The pointer is valid only until the next call to getModelCatalogJson().
static std::string g_catalog_json_cache;

/**
 * Returns a pointer to a library-owned buffer containing a JSON array of
 * registered model descriptors. The buffer is valid only until the next
 * call to getModelCatalogJson(). Callers must copy the contents immediately
 * (e.g. via JNI NewStringUTF) and must not hold the pointer across calls.
 * Not safe for concurrent access to the returned pointer.
 */
extern "C" const char *getModelCatalogJson(void) {
  auto json_escape = [](const char *s) -> std::string {
    if (!s)
      return "";
    std::string out;
    for (; *s; ++s) {
      if (*s == '"')
        out += "\\\"";
      else if (*s == '\\')
        out += "\\\\";
      else
        out += *s;
    }
    return out;
  };

  std::lock_guard<std::mutex> lock(descriptor_mutex());
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < descriptor_registry().size(); ++i) {
    const auto &d = descriptor_registry()[i];
    if (i)
      os << ",";
    os << "{\"id\":\"" << json_escape(d.id) << "\",\"family\":\""
       << json_escape(d.family) << "\",\"display_name\":\""
       << json_escape(d.display_name ? d.display_name : d.id)
       << "\",\"runtime\":" << static_cast<int>(d.runtime)
       << ",\"backend_mask\":" << d.backend_mask
       << ",\"capabilities\":" << d.capabilities;
    if (d.sd_variant_id)
      os << ",\"sd_variant_id\":\"" << json_escape(d.sd_variant_id) << "\"";
    os << "}";
  }
  os << "]";
  g_catalog_json_cache = os.str();
  return g_catalog_json_cache.c_str();
}

// Helper to register models (similar to main.cpp) ensuring factory is
// populated. Factory registration is singleton and persistent, but we do it
// once here to be sure. Since mquiain.cpp is not linked, we must duplicate
// registration or share it. Assuming this lib is used independently of
// main.cpp.
static void register_models() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    causallm::Factory::Instance().registerModel(
      "LlamaForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::CausalLM>(cfg, generation_cfg,
                                                    nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "Qwen2ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Qwen2CausalLM>(cfg, generation_cfg,
                                                         nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "Qwen3ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Qwen3CausalLM>(cfg, generation_cfg,
                                                         nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "Qwen3MoeForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Qwen3MoECausalLM>(cfg, generation_cfg,
                                                            nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "Qwen3SlimMoeForCausalLM",
      [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Qwen3SlimMoECausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "Qwen3CachedSlimMoeForCausalLM",
      [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Qwen3CachedSlimMoECausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "GptOssForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::GptOssForCausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "GptOssCachedSlimCausalLM",
      [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::GptOssCachedSlimCausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "Gemma3ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Gemma3CausalLM>(cfg, generation_cfg,
                                                          nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "Gemma4ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Gemma4CausalLM>(cfg, generation_cfg,
                                                          nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "MultilingualTinyBert", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::MultilingualTinyBert>(
          cfg, generation_cfg, nntr_cfg);
      });
    // Ouro (Universal-Transformer) self-registers "OuroModel" from its own
    // TU (src/models/ouro/ouro_embedding.cpp) via __attribute__((constructor)),
    // same pattern as the QNN models below.

#ifdef ENABLE_QNN_MODELS
    causallm::Factory::Instance().registerModel(
      "Gemma4_E2B_QNN", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Gemma4_E2B_QNN>(cfg, generation_cfg,
                                                          nntr_cfg);
      });
#endif
    // Register built-in configurations
    quick_dot_ai::register_builtin_configs();
  });
}

static const char *get_model_name_from_type(ModelType type) {
  switch (type) {
  case CAUSAL_LM_MODEL_QWEN3_0_6B:
    return "QWEN3-0.6B";
  case CAUSAL_LM_MODEL_QWEN3_1_7B_Q40:
    return "QWEN3-1.7B-Q40";
  case CAUSAL_LM_MODEL_TINY_BERT:
    return "TINY_BERT";
  case CAUSAL_LM_MODEL_FUNCTION_GEMMA:
    return "FUNCTION_GEMMA";
  case CAUSAL_LM_MODEL_GEMMA4_CPU:
    return "GEMMA4_CPU";
  case CAUSAL_LM_MODEL_OURO_EMBEDDING:
    return "OURO_EMBEDDING";
#ifdef ENABLE_QNN_MODELS
  case CAUSAL_LM_MODEL_GEMMA4_E2B_QNN:
    return "GEMMA4-E2B-QNN";
  case CAUSAL_LM_MODEL_VJEPA2_QNN:
    return "VJEPA2-QNN";
#endif
  default:
    return nullptr;
  }
}

static size_t text_generation_model_index(const CausalLmModel &h) {
  // Convention: a multi-model handle is [vision producer, text LLM, ...];
  // text generation runs on the LLM at index 1.
  return (h.models.size() > 1) ? 1 : 0;
}

static void reset_handle_session_state(CausalLmModel &h) { h.kv_len = 0; }

static void update_handle_session_after_run(CausalLmModel &h,
                                            size_t model_index) {
  if (model_index >= h.models.size() || model_index >= h.architectures.size())
    return;
  const auto *cb =
    ModelCallbackRegistry::instance().lookup(h.architectures[model_index]);
  if (!cb || !cb->read_kv_len)
    return;
  h.kv_len = cb->read_kv_len(h.models[model_index].get());
}

#ifdef ENABLE_QNN_MODELS
static causallm::Quick_Dot_AI_QNN *find_qnn_kv_cache_model(CausalLmModel &h) {
  for (auto &m : h.models) {
    auto *q = dynamic_cast<causallm::Quick_Dot_AI_QNN *>(m.get());
    if (q && q->supportsKvCachePersistence())
      return q;
  }
  return nullptr;
}
#endif

static ErrorCode save_qnn_kv_cache_on_handle(CausalLmModel &h,
                                             const char *cache_path) {
#ifndef ENABLE_QNN_MODELS
  (void)h;
  (void)cache_path;
  return CAUSAL_LM_ERROR_UNSUPPORTED;
#else
  if (cache_path == nullptr || cache_path[0] == '\0') {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  std::lock_guard<std::mutex> lock(h.mtx);
  if (!h.initialized || h.models.empty()) {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  auto *model = find_qnn_kv_cache_model(h);
  if (model == nullptr) {
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }

  try {
    model->saveKvCache(cache_path);
    h.kv_len = model->getKvLen();
  } catch (const std::exception &e) {
    LOGE("saveQnnKvCacheHandle failed: %s", e.what());
    return CAUSAL_LM_ERROR_UNKNOWN;
  }

  return CAUSAL_LM_ERROR_NONE;
#endif
}

static ErrorCode load_qnn_kv_cache_on_handle(CausalLmModel &h,
                                             const char *cache_path) {
#ifndef ENABLE_QNN_MODELS
  (void)h;
  (void)cache_path;
  return CAUSAL_LM_ERROR_UNSUPPORTED;
#else
  if (cache_path == nullptr || cache_path[0] == '\0') {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  std::lock_guard<std::mutex> lock(h.mtx);
  if (!h.initialized || h.models.empty()) {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  auto *model = find_qnn_kv_cache_model(h);
  if (model == nullptr) {
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }

  try {
    model->loadKvCache(cache_path);
    h.kv_len = model->getKvLen();
  } catch (const std::exception &e) {
    LOGE("loadQnnKvCacheHandle failed: %s", e.what());
    return CAUSAL_LM_ERROR_UNKNOWN;
  }

  return CAUSAL_LM_ERROR_NONE;
#endif
}

static ErrorCode reset_qnn_kv_cache_on_handle(CausalLmModel &h) {
#ifndef ENABLE_QNN_MODELS
  (void)h;
  return CAUSAL_LM_ERROR_UNSUPPORTED;
#else
  std::lock_guard<std::mutex> lock(h.mtx);
  if (!h.initialized || h.models.empty()) {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  auto *model = find_qnn_kv_cache_model(h);
  if (model == nullptr) {
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }

  try {
    model->resetKvCache();
    reset_handle_session_state(h);
  } catch (const std::exception &e) {
    LOGE("resetQnnKvCacheHandle failed: %s", e.what());
    return CAUSAL_LM_ERROR_UNKNOWN;
  }

  return CAUSAL_LM_ERROR_NONE;
#endif
}

static std::string get_quantization_suffix(ModelQuantizationType type) {
  // Current external model directories are not quantization-suffixed.
  // Quantization is still used for descriptor/config selection.
  (void)type;
  return "";
}

static std::string resolve_model_path(const std::string &model_key,
                                      ModelQuantizationType quant_type) {
  std::string path_upper = model_key;
  std::transform(path_upper.begin(), path_upper.end(), path_upper.begin(),
                 ::toupper);

  std::string base_dir_name = "";

  // 1. Try to find base directory name from map
  if (g_model_path_map.find(path_upper) != g_model_path_map.end()) {
    base_dir_name = g_model_path_map[path_upper];
  } else {
    // Fallback: use lowercased key as base dir name if not found in map
    // or just return empty? For restricted API, we should probably fail
    // earlier, but here we can return constructed path.
    base_dir_name = path_upper;
    std::transform(base_dir_name.begin(), base_dir_name.end(),
                   base_dir_name.begin(), ::tolower);
  }

  std::string model_path =
    "/" + base_dir_name + get_quantization_suffix(quant_type);

  return model_path;
}

/**
 * @brief Rebase path-like keys of a sub-model nntr_config.json onto @p sub_dir.
 *
 * Called once per sub-model inside the multi-model branch of
 * load_into_handle so that downstream code (Factory::create, load_weight)
 * sees absolute paths — mirrors the inline fixups the single-model path
 * already performs for model_file_name / binary_config_path / ...
 *
 * Absolute values (leading '/') are left untouched so the caller can
 * override a specific file with a system-wide path if they want.
 */
static bool is_absolute_path(const std::string &path) {
  if (path.empty())
    return false;
  if (path[0] == '/' || path[0] == '\\')
    return true;
  return path.size() >= 3 &&
         std::isalpha(static_cast<unsigned char>(path[0])) && path[1] == ':' &&
         (path[2] == '/' || path[2] == '\\');
}

static std::string rebase_path(const std::string &path,
                               const std::string &base_dir) {
  if (path.empty() || is_absolute_path(path))
    return path;
  return base_dir + "/" + path;
}

static void fix_paths(json &nntr_cfg, const std::string &sub_dir) {
  static const char *kKeys[] = {
    "tokenizer_file",       "model_file_name",     "binary_config_path",
    "image_newline_path",   "embedding_file_name", "ple_file_name",
    "rotation_matrix_path",
  };
  for (const char *k : kKeys) {
    if (!nntr_cfg.contains(k) || !nntr_cfg[k].is_string())
      continue;
    std::string v = nntr_cfg[k].get<std::string>();
    nntr_cfg[k] = rebase_path(v, sub_dir);
  }
}

static bool check_file_exists(const std::string &path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

static bool initialize_handle_grammar(CausalLmModel &h, size_t model_index) {
  if (model_index >= h.models.size() || !h.models[model_index] ||
      model_index >= h.model_dirs.size()) {
    return false;
  }

  if (!h.grammar_manager) {
    h.grammar_manager = std::make_unique<causallm::XGrammarManager>();
  }
  auto *tokenizer = h.models[model_index]->getTokenizer();
  const unsigned int vocab_size = h.models[model_index]->getVocabSize();
  const std::string tokenizer_path =
    h.model_dirs[model_index] + "/tokenizer.json";
  std::ifstream tokenizer_file(tokenizer_path, std::ios::binary);
  if (!tokenizer_file.is_open()) {
    LOGE("Cannot initialize xgrammar: tokenizer metadata file is missing: %s",
         tokenizer_path.c_str());
    return false;
  }
  std::ostringstream tokenizer_json;
  tokenizer_json << tokenizer_file.rdbuf();

  std::string tokenizer_metadata;
  try {
    tokenizer_metadata =
      xgrammar::TokenizerInfo::DetectMetadataFromHF(tokenizer_json.str());
  } catch (const std::exception &e) {
    LOGE("Cannot detect xgrammar tokenizer metadata for model[%zu]: %s",
         model_index, e.what());
    return false;
  }
  try {
    if (!h.grammar_manager->initialize(tokenizer, vocab_size,
                                       tokenizer_metadata)) {
      return false;
    }

    const std::string toolset_path =
      h.model_dirs[model_index] + "/Toolset.json";
    if (check_file_exists(toolset_path) &&
        !h.grammar_manager->loadToolset(toolset_path, tokenizer, vocab_size)) {
      LOGE("Failed to load toolset for model[%zu]: %s", model_index,
           toolset_path.c_str());
      return false;
    }
  } catch (const std::exception &e) {
    LOGE("Cannot initialize xgrammar for model[%zu]: %s", model_index,
         e.what());
    return false;
  } catch (...) {
    LOGE("Cannot initialize xgrammar for model[%zu]", model_index);
    return false;
  }
  return true;
}

static void validate_models() {
  std::lock_guard<std::mutex> registry_lock(g_registry_mutex);
  LOGD("[DEBUG] Validating model files...");
  // Iterate over all known model names in map
  for (auto const &[key, val] : g_model_path_map) {
    // We want to check for each Quantization Type if it exists
    // List of quant types to check: UNKNOWN (default), W4A32, W16A16, W32A32
    std::vector<ModelQuantizationType> quant_types = {
      CAUSAL_LM_QUANTIZATION_UNKNOWN, CAUSAL_LM_QUANTIZATION_W4A32,
      CAUSAL_LM_QUANTIZATION_W16A16, CAUSAL_LM_QUANTIZATION_W32A32};

    for (auto qt : quant_types) {
      std::string quant_suffix = get_quantization_suffix(qt);

      std::string lookup_key = key;
      if (qt != CAUSAL_LM_QUANTIZATION_UNKNOWN) {
        std::transform(quant_suffix.begin(), quant_suffix.end(),
                       quant_suffix.begin(), ::toupper); // "-W4A32"
        lookup_key += quant_suffix;
      }

      // Resolve path for this combination
      std::string resolved_path = "./models" + resolve_model_path(key, qt);

      if (g_model_registry.find(lookup_key) != g_model_registry.end()) {
        // CASE 1: Configuration is registered in model_config.cpp
        // For these models, we only check if the binary weight file exists.
        // The configurations (config.json, etc.) are embedded in the library.
        RegisteredModel &rm = g_model_registry[lookup_key];
        std::string bin_file_name = rm.config.model_file_name;
        std::string full_path = resolved_path + "/" + bin_file_name;

        if (check_file_exists(full_path)) {
          LOGD("  [OK] Reg Config: %s -> %s", lookup_key.c_str(),
               full_path.c_str());
        } else {
          LOGD("  [FAIL] Reg Config: %s -> Missing binary: %s",
               lookup_key.c_str(), full_path.c_str());
        }
      } else {
        // CASE 2: No internal config, but model type exists (via map
        // iteration). For these models, we require external configuration files
        // (config.json, nntr_config.json) to be present in the directory.
        if (check_file_exists(resolved_path)) {
          bool has_config = check_file_exists(resolved_path + "/config.json");
          bool has_nntr =
            check_file_exists(resolved_path + "/nntr_config.json");

          if (has_config && has_nntr) {
            LOGD("  [OK] External Config: %s -> %s", lookup_key.c_str(),
                 resolved_path.c_str());
            // Optional: Parse nntr_config to check bin
            try {
              json nntr =
                causallm::LoadJsonFile(resolved_path + "/nntr_config.json");
              if (nntr.contains("model_file_name")) {
                std::string bin = nntr["model_file_name"];
                if (check_file_exists(resolved_path + "/" + bin)) {
                  LOGD("       (Binary confirmed: %s)", bin.c_str());
                } else {
                  LOGD("       (MISSING BINARY: %s)", bin.c_str());
                }
              }
            } catch (...) {
            }
          } else {
            LOGD("  [FAIL] External Config: %s -> Missing configs in %s",
                 lookup_key.c_str(), resolved_path.c_str());
          }
        }
      }
    }
  }
}

ErrorCode setOptions(Config config) {
  {
    std::lock_guard<std::mutex> lock(g_options_mutex);
    g_default_verbose = config.verbose;
    g_default_chat_template_name =
      config.chat_template_name != nullptr ? config.chat_template_name : "";
  }
  if (config.debug_mode) {
    // Ensure models are registered so we can validate them
    register_models();
    validate_models();
  }
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode registerModelArchitecture(const char *arch_name,
                                    ModelArchConfig config) {
  if (arch_name == nullptr)
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  std::lock_guard<std::mutex> lock(g_registry_mutex);
  std::string name(arch_name);
  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  g_arch_config_map[name] = config;
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode registerModel(const char *model_name, const char *arch_name,
                        ModelRuntimeConfig config) {
  if (model_name == nullptr || arch_name == nullptr)
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  std::lock_guard<std::mutex> lock(g_registry_mutex);
  std::string name(model_name);
  std::transform(name.begin(), name.end(), name.begin(), ::toupper);

  std::string aname(arch_name);
  std::transform(aname.begin(), aname.end(), aname.begin(), ::toupper);

  g_model_registry[name] = {aname, config};
  return CAUSAL_LM_ERROR_NONE;
}

/**
 * @brief Core loader shared by loadModel and loadModelHandle.
 *
 * Populates the given handle's model / architecture / init-duration
 * vectors on success. Takes the handle's own mutex so two concurrent
 * loads on the same handle are serialized, while loads on different
 * handles run in parallel. A separate registry mutex protects
 * g_model_registry / g_arch_config_map during lookup.
 *
 * Dispatch in CASE 2 (file-based):
 *   - If the top-level nntr_config.json has both "architectures" (string
 *     array) and "model_dirs" (string array) of equal non-zero length,
 *     loads one sub-model per entry (e.g. vision encoder + LLM).
 *   - Otherwise loads a single model from the resolved directory using
 *     the pre-existing flow.
 */
#ifdef ENABLE_QNN_MODELS
// Point the QNN HTP backend at htp_backend_ext_config.json for ANY QNN arch.
// Must run BEFORE constructing/initializing a QNN model — single OR multi-model
// sub-models (e.g. the vision encoder of a multimodal pair) — otherwise
// QNNContext falls back to the process cwd ("/" for an installed APK) and QNN
// layer registration throws not_supported ("Unable to load backend extensions
// config"). Honors an externally configured path if already set.
static void ensure_qnn_backend_ext_config(const std::string &base_dir) {
  const char *configured = getenv("QUICK_DOT_AI_QNN_BACKEND_EXT_CONFIG_PATH");
  if (configured != nullptr && configured[0] != '\0')
    return;
  std::string config_path = base_dir;
  // Strip a trailing "/models" so the config resolves next to the
  // model-collection root (mirrors the single-model resolution).
  while (!config_path.empty() && config_path.back() == '/')
    config_path.pop_back();
  if (config_path.length() >= 7 &&
      config_path.substr(config_path.length() - 7) == "/models") {
    config_path = config_path.substr(0, config_path.length() - 7);
  }
  config_path += "/htp_backend_ext_config.json";
  set_environment_variable("QUICK_DOT_AI_QNN_BACKEND_EXT_CONFIG_PATH",
                           config_path.c_str());
}
#endif

/** Internal overload: config_name already resolved (T4 byName path). */
static ErrorCode load_into_handle(CausalLmModel &h, BackendType compute,
                                  const char *target_model_name,
                                  ModelQuantizationType quant_type,
                                  const char *native_lib_dir,
                                  const char *model_base_path) {
  LOGD("[DEBUG] load_into_handle: START");
  LOGD("[DEBUG]   compute: %d", compute);
  LOGD("[DEBUG]   target_model_name: %s",
       target_model_name ? target_model_name : "(null)");
  LOGD("[DEBUG]   quant_type: %d", quant_type);

  if (target_model_name == nullptr || target_model_name[0] == '\0' ||
      model_base_path == nullptr || model_base_path[0] == '\0') {
    LOGE("load_into_handle: model name and model base path are required");
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  auto start_init = std::chrono::high_resolution_clock::now();

  // Ensure models/configs are registered (thread-safe via call_once)
  LOGD("[DEBUG] load_into_handle: Calling register_models...");
  register_models();
  LOGD("[DEBUG] load_into_handle: register_models done");

  std::lock_guard<std::mutex> lock(h.mtx);
  try {
    clear_handle_models(h);
    h.architectures.clear();
    h.model_dirs.clear();
    h.chat_templates.clear();
    h.grammar_manager = std::make_unique<causallm::XGrammarManager>();
    h.dynamic_grammar_schemas.clear();
    h.last_output.clear();
    h.initialization_duration_ms.clear();
    h.initialized = false;
    reset_handle_session_state(h);
    {
      std::lock_guard<std::mutex> options_lock(g_options_mutex);
      h.verbose = g_default_verbose;
      h.chat_template_name = g_default_chat_template_name;
    }

    // Check if it's a registered in-memory config
    std::string input_name = std::string(target_model_name);
    std::string input_name_upper = input_name;
    std::transform(input_name_upper.begin(), input_name_upper.end(),
                   input_name_upper.begin(), ::toupper);
    LOGD("[DEBUG] load_into_handle: input_name = %s", input_name.c_str());

    std::string quant_suffix = "";
    switch (quant_type) {
    case CAUSAL_LM_QUANTIZATION_W4A32:
      quant_suffix = "-W4A32";
      break;
    case CAUSAL_LM_QUANTIZATION_W16A16:
      quant_suffix = "-W16A16";
      break;
    case CAUSAL_LM_QUANTIZATION_W8A16:
      quant_suffix = "-W8A16";
      break;
    case CAUSAL_LM_QUANTIZATION_W32A32:
      quant_suffix = "-W32A32";
      break;
    default:
      break;
    }
    std::string lookup_name = input_name_upper + quant_suffix;
    LOGD("[DEBUG] load_into_handle: lookup_name = %s", lookup_name.c_str());

    json cfg;
    json generation_cfg;
    json nntr_cfg;
    std::optional<causallm::ChatTemplate> loaded_chat_template;
    std::string model_dir_path;
    std::string abs_model_dir;
    std::string base_dir = model_base_path;

#ifdef ENABLE_QNN_MODELS
    // Set the QNN backend-extensions config path up front so it is in effect
    // for BOTH the multi-model sub-model loop and the single-model path.
    ensure_qnn_backend_ext_config(base_dir);
#endif

    // Check in-memory map first
    // if (g_model_registry.find(lookup_name) != g_model_registry.end()) {

    // always goto case2
    if (0) {
      LOGD("[DEBUG] load_into_handle: CASE 1 - Internal config found for %s",
           lookup_name.c_str());
      // ------------------------------------------------------------------------
      // CASE 1: Model Configuration is Internal (Registered in
      // model_config.cpp)
      // ------------------------------------------------------------------------
      // In this case, we do NOT load config.json or nntr_config.json from disk.
      // We only locate the binary weight file.
      RegisteredModel &rm = g_model_registry[lookup_name];

      // Find architecture config
      if (g_arch_config_map.find(rm.arch_name) == g_arch_config_map.end()) {
        LOGE("[DEBUG] load_into_handle: Architecture '%s' not found for model "
             "'%s'",
             rm.arch_name.c_str(), lookup_name.c_str());
        return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
      }
      LOGD("[DEBUG] load_into_handle: arch_name = %s", rm.arch_name.c_str());
      ModelArchConfig &ac = g_arch_config_map[rm.arch_name];
      ModelRuntimeConfig &rc = rm.config;

      // Strategy: Resolve path to find the weight file
      model_dir_path =
        "./models" + resolve_model_path(target_model_name, quant_type);
      LOGD("[DEBUG] load_into_handle: model_dir_path = %s",
           model_dir_path.c_str());

      // Populate JSONs from Arch Struct
      cfg["vocab_size"] = ac.vocab_size;
      cfg["hidden_size"] = ac.hidden_size;
      cfg["intermediate_size"] = ac.intermediate_size;
      cfg["num_hidden_layers"] = ac.num_hidden_layers;
      cfg["num_attention_heads"] = ac.num_attention_heads;
      cfg["head_dim"] = ac.head_dim;
      cfg["num_key_value_heads"] = ac.num_key_value_heads > 0
                                     ? ac.num_key_value_heads
                                     : ac.num_attention_heads;
      cfg["max_position_embeddings"] = ac.max_position_embeddings;
      cfg["rope_theta"] = ac.rope_theta;
      cfg["rms_norm_eps"] = ac.rms_norm_eps;
      cfg["tie_word_embeddings"] = ac.tie_word_embeddings;
      if (ac.sliding_window != UINT_MAX) {
        cfg["sliding_window"] = ac.sliding_window;
      } else {
        cfg["sliding_window"] = nullptr;
      }
      cfg["sliding_window_pattern"] = ac.sliding_window_pattern;
      cfg["architectures"] = {std::string(ac.architecture)};

      if (ac.num_eos_token_ids > 0) {
        std::vector<unsigned int> eos_ids;
        for (unsigned int i = 0; i < ac.num_eos_token_ids; ++i)
          eos_ids.push_back(ac.eos_token_ids[i]);
        generation_cfg["eos_token_id"] = eos_ids;
      }
      generation_cfg["bos_token_id"] = ac.bos_token_id;

      // Populate JSONs from Runtime Struct
      generation_cfg["top_k"] = rc.top_k;
      generation_cfg["top_p"] = rc.top_p;
      generation_cfg["temperature"] = rc.temperature;
      generation_cfg["do_sample"] = false;

      nntr_cfg["batch_size"] = rc.batch_size;
      nntr_cfg["model_type"] = std::string(rc.model_type);
      nntr_cfg["model_tensor_type"] = std::string(rc.model_tensor_type);
      nntr_cfg["init_seq_len"] = rc.init_seq_len;
      nntr_cfg["max_seq_len"] = rc.max_seq_len;
      nntr_cfg["num_to_generate"] = rc.num_to_generate;
      nntr_cfg["fsu"] = rc.fsu;
      nntr_cfg["fsu_lookahead"] = rc.fsu_lookahead;
      nntr_cfg["embedding_dtype"] = std::string(rc.embedding_dtype);
      nntr_cfg["fc_layer_dtype"] = std::string(rc.fc_layer_dtype);
      nntr_cfg["model_file_name"] = std::string(rc.model_file_name);

      // tokenizer_file path is set later from abs_model_dir in the shared
      // post-processing block below.
      (void)rc.tokenizer_file;

      if (strlen(rc.lmhead_dtype) > 0) {
        nntr_cfg["lmhead_dtype"] = std::string(rc.lmhead_dtype);
      }

      std::vector<unsigned int> bad_ids;
      for (unsigned int i = 0; i < rc.num_bad_word_ids; ++i)
        bad_ids.push_back(rc.bad_word_ids[i]);
      nntr_cfg["bad_word_ids"] = bad_ids;
    } else {
      LOGD("[DEBUG] load_into_handle: CASE 2 - External config (file-based)");
      // --------------------------------------------------
      // CASE 2: External Model Configuration (File-based)
      // --------------------------------------------------
      // The model type is registered (enum), but specific configuration for
      // this quantization is not in memory. We must load config.json and
      // nntr_config.json from the model directory
      model_dir_path = resolve_model_path(target_model_name, quant_type);
      LOGD("[DEBUG] load_into_handle: model_dir_path = %s",
           model_dir_path.c_str());

      abs_model_dir = base_dir + model_dir_path;
      LOGD("[DEBUG] load_into_handle: abs_model_dir = %s",
           abs_model_dir.c_str());

      // Top-level nntr_config.json is read once and used for both
      //   (a) multi-model dispatch (architectures[] + model_dirs[]), and
      //   (b) the single-model fallback below.
      json top_nntr =
        causallm::LoadJsonFile(abs_model_dir + "/nntr_config.json");

      LOGD("[DEBUG] load_into_handle: abs_model_dir = %s",
           abs_model_dir.c_str());

      LOGD("[DEBUG] load_into_handle: top_nntr = %s",
           (abs_model_dir + "/nntr_config.json").c_str());

      const bool is_multi =
        top_nntr.contains("architectures") &&
        top_nntr["architectures"].is_array() &&
        top_nntr.contains("model_dirs") && top_nntr["model_dirs"].is_array() &&
        !top_nntr["architectures"].empty() &&
        top_nntr["architectures"].size() == top_nntr["model_dirs"].size();

      LOGD("[DEBUG] load_into_handle: abs_model_dir = %d %d %d %d %zu %zu",
           top_nntr.contains("architectures"),
           top_nntr["architectures"].is_array(),
           top_nntr.contains("model_dirs"), top_nntr["model_dirs"].is_array(),
           top_nntr["architectures"].size(), top_nntr["model_dirs"].size());

      if (is_multi) {
        // ----------------------------------------------------------------
        // Multi-model branch.
        //
        //   top_nntr_config.json:
        //     { "architectures": ["ArchA", "ArchB"],
        //       "model_dirs":   ["sub_a",  "sub_b"] }
        //
        // Each sub_dir = abs_model_dir + "/" + model_dirs[i] owns its own
        // config.json / generation_config.json / nntr_config.json +
        // weights. The top-level architectures[i] wins over any
        // "architectures" entry inside sub-config — one source of truth.
        // ----------------------------------------------------------------
        auto archs = top_nntr["architectures"].get<std::vector<std::string>>();
        auto dirs = top_nntr["model_dirs"].get<std::vector<std::string>>();
        LOGD("[DEBUG] load_into_handle: MULTI-MODEL spec (N=%zu)",
             archs.size());

        for (size_t i = 0; i < archs.size(); ++i) {
          const std::string &arch_i = archs[i];
          const std::string sub_dir = abs_model_dir + "/" + dirs[i];
          LOGD("[DEBUG]   [%zu] arch=%s dir=%s", i, arch_i.c_str(),
               sub_dir.c_str());

          json sub_cfg = causallm::LoadJsonFile(sub_dir + "/config.json");

          json sub_gen;
          if (check_file_exists(sub_dir + "/generation_config.json")) {
            sub_gen =
              causallm::LoadJsonFile(sub_dir + "/generation_config.json");
          } else {
            sub_gen = json::object();
          }

          json sub_nntr = causallm::LoadJsonFile(sub_dir + "/nntr_config.json");

          fix_paths(sub_nntr, sub_dir);

          // Optional per-sub-model overrides from the top-level config.
          // Lets callers flip flags like uses_embedding / add keys like
          // embedding_file_name without duplicating the sub-model's own
          // nntr_config.json. fix_paths is run again so any newly
          // introduced path-like key (e.g. embedding_file_name) is
          // resolved relative to sub_dir just like the native keys.
          if (top_nntr.contains("model_options") &&
              top_nntr["model_options"].is_array() &&
              i < top_nntr["model_options"].size() &&
              top_nntr["model_options"][i].is_object()) {
            for (auto it = top_nntr["model_options"][i].begin();
                 it != top_nntr["model_options"][i].end(); ++it) {
              sub_nntr[it.key()] = it.value();
              LOGD("[DEBUG]   override sub[%zu] %s", i, it.key().c_str());
            }
            fix_paths(sub_nntr, sub_dir);
          }
          if (sub_nntr.contains("lora_path")) {
            LOGD("lora_path : %s",
                 sub_nntr["lora_path"].get<std::string>().c_str());
            std::string lora_path =
              sub_dir + "/" + sub_nntr["lora_path"].get<std::string>();
            sub_nntr["lora_path"] = lora_path;
            LOGD("lora_path is now %s", lora_path.c_str());
          }

          auto m = causallm::Factory::Instance().create(arch_i, sub_cfg,
                                                        sub_gen, sub_nntr);
          if (!m) {
            LOGE("[DEBUG] load_into_handle: Factory::create returned nullptr "
                 "for sub-model %zu (arch=%s)",
                 i, arch_i.c_str());
            return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
          }

          auto sub_t0 = std::chrono::high_resolution_clock::now();
          if (native_lib_dir != nullptr && strlen(native_lib_dir) > 0) {
            set_environment_variable("ADSP_LIBRARY_PATH", native_lib_dir);
          }
          m->initialize();

          std::string weight_file =
            sub_nntr.contains("model_file_name")
              ? sub_nntr["model_file_name"].get<std::string>()
              : (sub_dir + "/pytorch_model.bin");
          m->load_weight(weight_file);
          auto sub_t1 = std::chrono::high_resolution_clock::now();
          double sub_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            sub_t1 - sub_t0)
                            .count();

          h.models.push_back(std::move(m));
          h.architectures.push_back(arch_i);
          h.model_dirs.push_back(sub_dir);
          h.initialization_duration_ms.push_back(sub_ms);
          LOGD("[DEBUG]   [%zu] loaded (%.1f ms)", i, sub_ms);

          // Keep templates parallel with models so handles and sub-models do
          // not overwrite each other's rendering state.
          if (causallm::ChatTemplate::Exists(sub_dir)) {
            try {
              h.chat_templates.emplace_back(
                causallm::ChatTemplate::Load(sub_dir));
              std::cout << "[Info] Chat template loaded from " << sub_dir
                        << std::endl;
            } catch (const std::exception &e) {
              std::cerr << "[Warning] Chat template load failed: " << e.what()
                        << ". Continuing without a model-provided template."
                        << std::endl;
              h.chat_templates.emplace_back(std::nullopt);
            }
          } else {
            h.chat_templates.emplace_back(std::nullopt);
          }
        }

        if (native_lib_dir != nullptr)
          h.native_lib_dir = native_lib_dir;
        const size_t text_model_index = text_generation_model_index(h);
        if (!initialize_handle_grammar(h, text_model_index)) {
          LOGE("[Warning] Grammar unavailable for text model[%zu]",
               text_model_index);
        }
        h.initialized = true;
        publish_cancellation_targets(h);

        auto finish_init = std::chrono::high_resolution_clock::now();
        auto e2e = std::chrono::duration_cast<std::chrono::milliseconds>(
                     finish_init - start_init)
                     .count();
        LOGD("[DEBUG] load_into_handle: MULTI-MODEL SUCCESS "
             "(%zu models, %lld ms e2e)",
             h.models.size(), (long long)e2e);
        return CAUSAL_LM_ERROR_NONE;
      }

      // -------------------- single-model fallback --------------------
      LOGD("single cfg : %s", (abs_model_dir + "/config.json").c_str());
      cfg = causallm::LoadJsonFile(abs_model_dir + "/config.json");

      if (check_file_exists(abs_model_dir + "/generation_config.json")) {
        generation_cfg =
          causallm::LoadJsonFile(abs_model_dir + "/generation_config.json");
      }

      nntr_cfg = std::move(top_nntr);

      if (nntr_cfg.contains("lora_path")) {
        nntr_cfg["lora_path"] = "";
      }

      LOGD("single tokenizer : %s",
           (abs_model_dir + "/tokenizer.json").c_str());

      if (nntr_cfg.contains("tokenizer_file")) {
        nntr_cfg["tokenizer_file"] = abs_model_dir + "/tokenizer.json";
      }
    }

    // Load chat template from model directory if available.
    if (causallm::ChatTemplate::Exists(abs_model_dir)) {
      try {
        loaded_chat_template = causallm::ChatTemplate::Load(abs_model_dir);
        LOGD("[Info] Chat template loaded from %s", abs_model_dir.c_str());
      } catch (const std::exception &e) {
        LOGE("[Warning] Chat template load failed: %s. Continuing without a "
             "model-provided template.",
             e.what());
        loaded_chat_template.reset();
      }
    } else {
      loaded_chat_template.reset();
      LOGE("[Warning] No model-provided chat template found in %s.",
           abs_model_dir.c_str());
    }

    // Construct weight file path
    std::string weight_file_name;
    if (nntr_cfg.contains("model_file_name")) {
      weight_file_name = nntr_cfg["model_file_name"].get<std::string>();
    } else {
      weight_file_name = "pytorch_model.bin";
    }

    const std::string weight_file =
      rebase_path(weight_file_name, abs_model_dir);
    LOGD("[DEBUG] load_into_handle: weight_file = %s", weight_file.c_str());
    std::cout << "-------------------" << abs_model_dir << "/" << std::endl;

    nntr_cfg["model_file_name"] = weight_file;
    if (nntr_cfg.contains("binary_config_path")) {
      std::string str = nntr_cfg["binary_config_path"].get<std::string>();
      nntr_cfg["binary_config_path"] = rebase_path(str, abs_model_dir);
      LOGD("[DEBUG] bianry config data: file = %s",
           nntr_cfg["binary_config_path"].get<std::string>().c_str());
    }
    if (nntr_cfg.contains("image_newline_path")) {
      std::string str = nntr_cfg["image_newline_path"].get<std::string>();
      nntr_cfg["image_newline_path"] = rebase_path(str, abs_model_dir);
      LOGD("[DEBUG] new line config data: file = %s",
           nntr_cfg["image_newline_path"].get<std::string>().c_str());
    }
    if (nntr_cfg.contains("embedding_file_name")) {
      std::string str = nntr_cfg["embedding_file_name"].get<std::string>();
      nntr_cfg["embedding_file_name"] = rebase_path(str, abs_model_dir);
    }
    if (nntr_cfg.contains("ple_file_name")) {
      std::string str = nntr_cfg["ple_file_name"].get<std::string>();
      nntr_cfg["ple_file_name"] = rebase_path(str, abs_model_dir);
    }

    // Determine architecture from config or ModelType
    // Priority: Config file architecture > ModelType mapping (fallback)
    std::string architecture;
    if (cfg.contains("architectures") && cfg["architectures"].is_array() &&
        !cfg["architectures"].empty()) {
      architecture = cfg["architectures"].get<std::vector<std::string>>()[0];
    } else {
      // No fallback mapping from specific ModelType instances to generic
      // architecture strings for now, as specific types should have config or
      // be loaded from valid file with config.json
      LOGE("[DEBUG] load_into_handle: No architecture found in config");
      return CAUSAL_LM_ERROR_INVALID_PARAMETER;
    }
    LOGD("[DEBUG] load_into_handle: architecture = %s", architecture.c_str());

    LOGD("[DEBUG] load_into_handle: Creating model via Factory...%s ",
         architecture.c_str());

    auto m = causallm::Factory::Instance().create(architecture, cfg,
                                                  generation_cfg, nntr_cfg);
    if (!m) {
      LOGE("[DEBUG] load_into_handle: Factory::create returned nullptr");
      return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
    }
    LOGD("[DEBUG] load_into_handle: Model created successfully");

    if (native_lib_dir != nullptr)
      h.native_lib_dir = native_lib_dir;

#ifdef ENABLE_QNN_MODELS
    // Point the QNN HTP backend at htp_backend_ext_config.json. This must run
    // for ANY QNN-backed architecture (e.g. Gemma4_E2B_QNN), not only
    // callback-registered ones: otherwise QNNContext falls back to the process
    // cwd — which is "/" for an installed APK — and QNN layer registration
    // throws not_supported ("Unable to load backend extensions config").
    // Honor an externally configured path if one is already set.
    ensure_qnn_backend_ext_config(base_dir);
#endif

    LOGD("[DEBUG] load_into_handle: Calling model->initialize()...");
    if (native_lib_dir != nullptr && strlen(native_lib_dir) > 0) {
      set_environment_variable("ADSP_LIBRARY_PATH", native_lib_dir);
    }
    m->initialize();
    LOGD("[DEBUG] load_into_handle: model->initialize() done");

    LOGD("[DEBUG] load_into_handle: Calling model->load_weight()...");
    // Multi-component models (e.g. Lfm2VLVJepa21BModel = ViT + projector +
    // LFM2) have no single model_file_name; they resolve each component's
    // weight file relative to the model directory, so hand them the directory
    // instead.
    if (architecture == "Lfm2VLVJepa21BModel") {
      m->load_weight(abs_model_dir);
    } else {
      m->load_weight(weight_file);
    }
    LOGD("[DEBUG] load_into_handle: model->load_weight() done");

    auto finish_init = std::chrono::high_resolution_clock::now();
    auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      finish_init - start_init);

    h.models.push_back(std::move(m));
    h.architectures.push_back(architecture);
    h.model_dirs.push_back(abs_model_dir);
    h.chat_templates.push_back(std::move(loaded_chat_template));
    h.initialization_duration_ms.push_back(
      static_cast<double>(init_duration.count()));
    h.initialized = true;

    if (!initialize_handle_grammar(h, 0)) {
      LOGE("[Warning] Grammar unavailable for model[0]");
    }
    publish_cancellation_targets(h);

    LOGD("[DEBUG] load_into_handle: SINGLE SUCCESS (init took %lld ms)",
         (long long)init_duration.count());
  } catch (...) {
    // RTTI may not match across shared libraries — query the current
    // exception's typeinfo directly via the Itanium ABI hook. This
    // works even when catching by concrete types fails due to typeinfo
    // duplication between libnntrainer.so and libquick_dot_ai_api.so.
#ifndef _WIN32
    const std::type_info *ti = abi::__cxa_current_exception_type();
    const char *raw = ti ? ti->name() : "(null)";
    int status = 0;
    char *demangled = (ti != nullptr)
                        ? abi::__cxa_demangle(raw, nullptr, nullptr, &status)
                        : nullptr;
    LOGE("[DEBUG] load_into_handle: unknown exception, type=%s",
         demangled ? demangled : raw);
    std::free(demangled);
#else
    LOGE("[DEBUG] load_into_handle: unknown exception");
#endif

    // Also try once more via rethrow — in case std::exception RTTI does
    // match from this catch-site (we already tried above but leaving
    // this as a second chance is cheap).
    try {
      throw;
    } catch (const std::exception &e) {
      LOGE("[DEBUG] load_into_handle: rethrown std::exception what()=%s",
           e.what());
    } catch (...) {
      LOGE("[DEBUG] load_into_handle: rethrown still non-std");
    }
    return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
  }
  LOGD("[DEBUG] load_into_handle: END (returning CAUSAL_LM_ERROR_NONE)");
  return CAUSAL_LM_ERROR_NONE;
}

/** ModelType overload: translates enum → config_name then delegates. */
static ErrorCode load_into_handle(CausalLmModel &h, BackendType compute,
                                  ModelType modeltype,
                                  ModelQuantizationType quant_type,
                                  const char *native_lib_dir,
                                  const char *model_base_path) {
  const char *target_model_name = get_model_name_from_type(modeltype);
  if (!target_model_name) {
    LOGE("[DEBUG] load_into_handle: Invalid modeltype %d", modeltype);
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  LOGD("[DEBUG] load_into_handle: target_model_name = %s %d", target_model_name,
       modeltype);
  return load_into_handle(h, compute, target_model_name, quant_type,
                          native_lib_dir, model_base_path);
}

/**
 * @brief Core metrics fetcher shared by getPerformanceMetrics and its
 *        handle-based counterpart.
 *
 * Reports models[0] runtime metrics. initialization_duration_ms is the
 * sum over all sub-models this handle owns.
 */
static ErrorCode metrics_on_handle(CausalLmModel &h,
                                   PerformanceMetrics *metrics) {
  if (metrics == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  std::lock_guard<std::mutex> lock(h.mtx);
  size_t metrics_model_idx = text_generation_model_index(h);
  if (!h.initialized || h.models.size() <= metrics_model_idx ||
      !h.models[metrics_model_idx]) {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  try {
    auto *model = h.models[metrics_model_idx].get();
    if (!model->hasRun()) {
      return CAUSAL_LM_ERROR_INFERENCE_NOT_RUN;
    }
    auto im = model->getPerformanceMetrics();
    metrics->prefill_tokens = im.prefill_tokens;
    metrics->prefill_duration_ms = im.prefill_duration_ms;
    metrics->generation_tokens = im.generation_tokens;
    metrics->generation_duration_ms = im.generation_duration_ms;
    metrics->total_duration_ms = im.total_duration_ms;
    metrics->peak_memory_kb = im.peak_memory_kb;

    double total_init = 0.0;
    for (double d : h.initialization_duration_ms)
      total_init += d;
    metrics->initialization_duration_ms = total_init;
  } catch (const std::exception &e) {
    LOGE("Exception in getPerformanceMetrics: %s", e.what());
    return CAUSAL_LM_ERROR_UNKNOWN;
  }

  return CAUSAL_LM_ERROR_NONE;
}

/*============================================================================
 * Legacy non-handle API implementation
 *============================================================================*/

ErrorCode loadModel(BackendType compute, ModelType modeltype,
                    ModelQuantizationType quant_type,
                    const char *model_base_path) {
  return load_into_handle(get_default_handle(), compute, modeltype, quant_type,
                          nullptr, model_base_path);
}

ErrorCode saveQnnKvCache(const char *cache_path) {
  return save_qnn_kv_cache_on_handle(get_default_handle(), cache_path);
}

ErrorCode loadQnnKvCache(const char *cache_path) {
  return load_qnn_kv_cache_on_handle(get_default_handle(), cache_path);
}

ErrorCode resetQnnKvCache(void) {
  return reset_qnn_kv_cache_on_handle(get_default_handle());
}

ErrorCode getPerformanceMetrics(PerformanceMetrics *metrics) {
  return metrics_on_handle(get_default_handle(), metrics);
}

/*============================================================================
 * Handle-based API implementation
 *============================================================================*/

ErrorCode loadModelHandle(BackendType compute, ModelType modeltype,
                          ModelQuantizationType quant_type,
                          const char *native_lib_dir,
                          const char *model_base_path,
                          CausalLmHandle *out_handle) {
  LOGD("[DEBUG] loadModelHandle:%d START", __LINE__);
  LOGD("[DEBUG] loadModelHandle:%d   compute: %d", __LINE__, compute);
  LOGD("[DEBUG] loadModelHandle:%d   modeltype: %d", __LINE__, modeltype);
  LOGD("[DEBUG] loadModelHandle:%d   quant_type: %d", __LINE__, quant_type);
  LOGD("[DEBUG] loadModelHandle:%d   native_lib_dir: %s", __LINE__,
       native_lib_dir ? native_lib_dir : "(null)");
  LOGD("[DEBUG] loadModelHandle:%d   out_handle ptr: %p", __LINE__,
       (void *)out_handle);

  if (out_handle == nullptr) {
    LOGE("[DEBUG] loadModelHandle:%d out_handle is nullptr", __LINE__);
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  *out_handle = nullptr;
  if (!is_valid_backend(compute) || !is_valid_quantization(quant_type))
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;

  std::unique_ptr<CausalLmModel> h;
  try {
    h = std::make_unique<CausalLmModel>();
  } catch (const std::exception &e) {
    LOGE("loadModelHandle: allocation failed: %s", e.what());
    return CAUSAL_LM_ERROR_UNKNOWN;
  }
  LOGD("[DEBUG] loadModelHandle:%d CausalLmModel allocated at %p", __LINE__,
       (void *)h.get());

  LOGD("[DEBUG] loadModelHandle:%d Calling load_into_handle...", __LINE__);
  ErrorCode ec = load_into_handle(*h, compute, modeltype, quant_type,
                                  native_lib_dir, model_base_path);
  LOGD("[DEBUG] loadModelHandle:%d load_into_handle returned: %d", __LINE__,
       ec);

  if (ec != CAUSAL_LM_ERROR_NONE) {
    LOGE("[DEBUG] loadModelHandle:%d load_into_handle failed, deleting handle",
         __LINE__);
    return ec;
  }
  *out_handle = h.release();
  LOGD("[DEBUG] loadModelHandle:%d SUCCESS, handle set to %p", __LINE__,
       (void *)*out_handle);
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode loadModelHandleByName(BackendType compute, const char *model_id,
                                ModelQuantizationType quant_type,
                                const char *native_lib_dir,
                                const char *model_base_path,
                                CausalLmHandle *out_handle) {
  if (out_handle == nullptr)
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  *out_handle = nullptr;
  if (model_id == nullptr || model_id[0] == '\0' ||
      !is_valid_backend(compute) || !is_valid_quantization(quant_type))
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;

  try {
    register_models(); // ensure factory + public descriptors registered

    auto d_opt = find_descriptor_by_id(model_id);
    if (!d_opt) {
      LOGE("loadModelHandleByName: unknown id '%s'", model_id);
      return CAUSAL_LM_ERROR_INVALID_PARAMETER;
    }
    const ModelDescriptor &d = *d_opt;
    if (!d.config_name) {
      LOGE("loadModelHandleByName: descriptor '%s' has null config_name",
           model_id);
      return CAUSAL_LM_ERROR_INVALID_PARAMETER;
    }
    if (((d.backend_mask >> static_cast<unsigned int>(compute)) & 1u) == 0u) {
      LOGE("loadModelHandleByName: backend %d not in mask 0x%x for '%s'",
           compute, d.backend_mask, model_id);
      return CAUSAL_LM_ERROR_INVALID_PARAMETER;
    }

    std::unique_ptr<CausalLmModel> h;
    try {
      h = std::make_unique<CausalLmModel>();
    } catch (const std::exception &e) {
      LOGE("loadModelHandleByName: allocation failed: %s", e.what());
      return CAUSAL_LM_ERROR_UNKNOWN;
    }

    ErrorCode ec = load_into_handle(*h, compute, d.config_name, quant_type,
                                    native_lib_dir, model_base_path);
    if (ec != CAUSAL_LM_ERROR_NONE) {
      return ec;
    }
    *out_handle = h.release();
    return CAUSAL_LM_ERROR_NONE;
  } catch (const std::exception &e) {
    LOGE("loadModelHandleByName: exception: %s", e.what());
    return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
  } catch (...) {
    LOGE("loadModelHandleByName: unknown exception");
    return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
  }
}

ErrorCode configureSpeculativeDecoding(CausalLmHandle h, bool use_sd) {
  if (!h)
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;

  std::lock_guard<std::mutex> lock(h->mtx);
  if (!h->initialized || h->models.empty() || h->architectures.empty())
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  if (!use_sd)
    return CAUSAL_LM_ERROR_NONE;

  const auto *cb =
    ModelCallbackRegistry::instance().lookup(h->architectures[0]);
  if (!cb || !cb->configure_speculative_decoding)
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  try {
    return cb->configure_speculative_decoding(h->models[0].get(), true);
  } catch (const std::exception &e) {
    LOGE("configureSpeculativeDecoding: %s", e.what());
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  } catch (...) {
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
}

ErrorCode loadMultimodalHandleByName(
  BackendType compute, const char *embedding_model_id, const char *llm_model_id,
  ModelQuantizationType quant_type, const char *native_lib_dir,
  const char *model_base_path, CausalLmHandle *out_handle) {
  if (out_handle == nullptr)
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  *out_handle = nullptr;
  if (embedding_model_id == nullptr || embedding_model_id[0] == '\0' ||
      llm_model_id == nullptr || llm_model_id[0] == '\0' ||
      !is_valid_backend(compute) || !is_valid_quantization(quant_type))
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;

#ifndef QUICKAI_ENABLE_EXPERIMENTAL_MULTIMODAL_NNTRAINER_API
  (void)compute;
  (void)quant_type;
  (void)native_lib_dir;
  (void)model_base_path;
  LOGE("loadMultimodalHandleByName: experimental multimodal nntrainer API is "
       "not enabled");
  return CAUSAL_LM_ERROR_UNSUPPORTED;
#else
  try {
    register_models();

    auto ev = find_descriptor_by_id(embedding_model_id);
    auto lv = find_descriptor_by_id(llm_model_id);
    if (!ev || !lv || !ev->config_name || !lv->config_name) {
      LOGE("loadMultimodalHandleByName: unknown id(s) emb='%s' llm='%s'",
           embedding_model_id, llm_model_id);
      return CAUSAL_LM_ERROR_INVALID_PARAMETER;
    }
    const unsigned int backend_bit = 1u << static_cast<unsigned int>(compute);
    if ((ev->backend_mask & backend_bit) == 0u ||
        (lv->backend_mask & backend_bit) == 0u ||
        (ev->capabilities & QDA_CAP_VISION_ENCODER) == 0u ||
        (lv->capabilities & QDA_CAP_OPENAI_API) == 0u ||
        (lv->capabilities & QDA_CAP_MULTIMODAL) == 0u) {
      LOGE("loadMultimodalHandleByName: incompatible descriptors or backend");
      return CAUSAL_LM_ERROR_UNSUPPORTED;
    }

    // Load each model into its own temporary single-model handle, then move the
    // sub-models into the combined handle in order [vision producer, LLM].
    // This reuses the proven single-model load path without modifying it.
    CausalLmModel tmp_vision;
    CausalLmModel tmp_llm;
    ErrorCode ec =
      load_into_handle(tmp_vision, compute, ev->config_name, quant_type,
                       native_lib_dir, model_base_path);
    if (ec != CAUSAL_LM_ERROR_NONE) {
      LOGE("loadMultimodalHandleByName: vision '%s' load failed (%d)",
           embedding_model_id, ec);
      return ec;
    }
    ec = load_into_handle(tmp_llm, compute, lv->config_name, quant_type,
                          native_lib_dir, model_base_path);
    if (ec != CAUSAL_LM_ERROR_NONE) {
      LOGE("loadMultimodalHandleByName: llm '%s' load failed (%d)",
           llm_model_id, ec);
      return ec;
    }
    if (tmp_vision.models.empty() || tmp_llm.models.empty())
      return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;

    auto *vision_model = tmp_vision.models[0].get();
    auto *llm_model = tmp_llm.models[0].get();
    if (!vision_model->supportsImageEncoding() ||
        !llm_model->supportsTextGeneration() ||
        !llm_model->supportsEmbeddingInput() ||
        llm_model->embeddingBytesPerToken() == 0) {
      LOGE("loadMultimodalHandleByName: model pair does not implement the "
           "required vision/embedding generation interfaces");
      return CAUSAL_LM_ERROR_UNSUPPORTED;
    }

    auto h = std::make_unique<CausalLmModel>();

    auto move_one = [](CausalLmModel &src, CausalLmModel &dst) {
      dst.models.push_back(std::move(src.models[0]));
      dst.architectures.push_back(
        src.architectures.empty() ? std::string() : src.architectures[0]);
      dst.model_dirs.push_back(src.model_dirs.empty() ? std::string()
                                                      : src.model_dirs[0]);
      if (!src.initialization_duration_ms.empty())
        dst.initialization_duration_ms.push_back(
          src.initialization_duration_ms[0]);
      dst.chat_templates.push_back(src.chat_templates.empty()
                                     ? std::optional<causallm::ChatTemplate>{}
                                     : std::move(src.chat_templates[0]));
    };
    move_one(tmp_vision, *h); // index 0 = vision producer
    move_one(tmp_llm, *h);    // index 1 = LLM consumer
    if (native_lib_dir != nullptr)
      h->native_lib_dir = native_lib_dir;
    h->verbose = tmp_llm.verbose;
    h->chat_template_name = tmp_llm.chat_template_name;
    if (!initialize_handle_grammar(*h, 1)) {
      LOGE("loadMultimodalHandleByName: grammar unavailable for the LLM");
    }
    h->initialized = true;
    publish_cancellation_targets(*h);

    *out_handle = h.release();
    return CAUSAL_LM_ERROR_NONE;
  } catch (const std::exception &e) {
    LOGE("loadMultimodalHandleByName: exception: %s", e.what());
    return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
  } catch (...) {
    LOGE("loadMultimodalHandleByName: unknown exception");
    return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
  }
#endif
}

ErrorCode saveQnnKvCacheHandle(CausalLmHandle handle, const char *cache_path) {
  if (handle == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  return save_qnn_kv_cache_on_handle(*handle, cache_path);
}

ErrorCode loadQnnKvCacheHandle(CausalLmHandle handle, const char *cache_path) {
  if (handle == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  return load_qnn_kv_cache_on_handle(*handle, cache_path);
}

ErrorCode resetQnnKvCacheHandle(CausalLmHandle handle) {
  if (handle == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  return reset_qnn_kv_cache_on_handle(*handle);
}

ErrorCode getPerformanceMetricsHandle(CausalLmHandle handle,
                                      PerformanceMetrics *metrics) {
  if (handle == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  return metrics_on_handle(*handle, metrics);
}

/*============================================================================
 * Internal streaming helper
 *============================================================================*/

static ErrorCode
run_model_streaming_on_handle(CausalLmModel &h, const std::string &raw_input,
                              CausalLmTokenCallback callback, void *user_data,
                              size_t model_index,
                              causallm::XGrammar *grammar = nullptr) {
  if (model_index >= h.models.size() || !h.models[model_index]) {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }
  auto *m = h.models[model_index].get();
  if (!model_supports_text_output(m)) {
    LOGE("[DEBUG] run_model_streaming_on_handle: model[%zu] does not expose "
         "text output",
         model_index);
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }

  CallbackStreamer streamer;
  callback_streamer_init(&streamer, callback, user_data);
  if (!set_model_streamer(m, &streamer.base))
    return CAUSAL_LM_ERROR_UNSUPPORTED;

  std::unique_ptr<XGrammarLogitsProcessor> grammar_processor;
  struct Detach {
    causallm::Transformer *t;
    bool detach_logits = false;
#ifdef ENABLE_QNN_MODELS
    causallm::Quick_Dot_AI_QNN *qnn_model = nullptr;
#endif
    ~Detach() {
      set_model_streamer(t, nullptr);
      if (detach_logits)
        t->setLogitsProcessor(nullptr);
#ifdef ENABLE_QNN_MODELS
      if (qnn_model != nullptr)
        qnn_model->resetXGrammar();
#endif
    }
  } detach_guard{m};

  bool qnn_grammar_attached = false;
#ifdef ENABLE_QNN_MODELS
  if (grammar != nullptr) {
    if (auto *qnn_model = as_qnn_model(m)) {
      qnn_model->setXGrammar(grammar);
      detach_guard.qnn_model = qnn_model;
      qnn_grammar_attached = true;
    }
  }
#endif
  if (grammar != nullptr && !qnn_grammar_attached) {
    grammar_processor = std::make_unique<XGrammarLogitsProcessor>(
      grammar, [m]() { request_model_stop(m); });
    m->setLogitsProcessor(grammar_processor.get());
    detach_guard.detach_logits = true;
  }

  try {
    LOGD("[DEBUG]   formatted input length: %zu", raw_input.length());
    m->run(raw_input, false, "", "", h.verbose);

    if (grammar_processor != nullptr && grammar_processor->failed()) {
      LOGE("[DEBUG] run_model_streaming_on_handle: grammar rejected a "
           "generated token");
      return CAUSAL_LM_ERROR_INFERENCE_FAILED;
    }
#ifdef ENABLE_QNN_MODELS
    if (detach_guard.qnn_model != nullptr &&
        detach_guard.qnn_model->hasXGrammarFailure()) {
      LOGE("[DEBUG] run_model_streaming_on_handle: QNN grammar rejected a "
           "generated token");
      return CAUSAL_LM_ERROR_INFERENCE_FAILED;
    }
#endif

    if (!get_model_output(m, h.last_output))
      return CAUSAL_LM_ERROR_UNSUPPORTED;
    update_handle_session_after_run(h, model_index);

    if (m->hasRun()) {
      auto im = m->getPerformanceMetrics();
      double total_init = 0.0;
      for (double d : h.initialization_duration_ms)
        total_init += d;

      LOGD("[PERF] Performance Metrics:");
      LOGD("[PERF]   prefill_tokens: %u", im.prefill_tokens);
      LOGD("[PERF]   prefill_duration_ms: %.2f", im.prefill_duration_ms);
      LOGD("[PERF]   generation_tokens: %u", im.generation_tokens);
      LOGD("[PERF]   generation_duration_ms: %.2f", im.generation_duration_ms);
      LOGD("[PERF]   total_duration_ms: %.2f", im.total_duration_ms);
      LOGD("[PERF]   peak_memory_kb: %zu", im.peak_memory_kb);
      LOGD("[PERF]   initialization_duration_ms: %.2f", total_init);

      if (im.prefill_duration_ms > 0) {
        double tokens_per_sec =
          (im.prefill_tokens * 1000.0) / im.prefill_duration_ms;
        LOGD("[PERF]   prefill_tokens_per_sec: %.2f", tokens_per_sec);
      }
      if (im.generation_duration_ms > 0) {
        double tokens_per_sec =
          (im.generation_tokens * 1000.0) / im.generation_duration_ms;
        LOGD("[PERF]   generation_tokens_per_sec: %.2f", tokens_per_sec);
      }
    }
  } catch (const std::exception &e) {
    LOGE("[DEBUG] run_model_streaming_on_handle: Exception caught: %s",
         e.what());
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  } catch (...) {
    LOGE("[DEBUG] run_model_streaming_on_handle: Unknown exception caught");
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }

  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode quickAiRunText(CausalLmHandle handle, const char *input,
                         CausalLmTokenCallback callback, void *user_data) {
  if (handle == nullptr || input == nullptr || input[0] == '\0' ||
      callback == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  auto &h = *handle;
  std::lock_guard<std::mutex> lock(h.mtx);
  if (!h.initialized || h.models.empty()) {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }
  ScopedRunRequest run_request(h);

  try {
    const size_t model_index = text_generation_model_index(h);
    if (model_index >= h.models.size() || !h.models[model_index]) {
      return CAUSAL_LM_ERROR_NOT_INITIALIZED;
    }
    ScopedGeneration generation(h);
    h.last_output.clear();
    h.models[model_index]->resetConversationState();
    reset_handle_session_state(h);
    return run_model_streaming_on_handle(h, input, callback, user_data,
                                         model_index);
  } catch (const std::exception &e) {
    LOGE("quickAiRunText: inference failed: %s", e.what());
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  } catch (...) {
    LOGE("quickAiRunText: unknown failure");
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
}

ErrorCode encodeModelHandle(CausalLmHandle handle, const char *text,
                            float **out_embedding, int *out_dim) {
  if (handle == nullptr || text == nullptr || out_embedding == nullptr ||
      out_dim == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  *out_embedding = nullptr;
  *out_dim = 0;

  auto &h = *handle;
  std::lock_guard<std::mutex> lock(h.mtx);

  if (!h.initialized || h.models.empty()) {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  // Embedding models occupy models[0] (single-model embedding handle).
  auto *st = dynamic_cast<causallm::SentenceTransformer *>(h.models[0].get());
  if (st == nullptr) {
    LOGE("encodeModelHandle: models[0] is not a SentenceTransformer");
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }

  try {
    const int dim = st->getEmbeddingDim();
    if (dim <= 0) {
      return CAUSAL_LM_ERROR_INFERENCE_FAILED;
    }

    // WSTR is std::string in this codebase; pass the text directly,
    // The embedding tokenizer consumes ordinary UTF-8 text.
    std::string s(text);

    std::vector<float *> results = st->encode(s);
    if (results.empty() || results[0] == nullptr) {
      for (auto *p : results)
        delete[] p;
      return CAUSAL_LM_ERROR_INFERENCE_FAILED;
    }

    // Copy the batch-0 embedding (first DIM floats) into a caller-owned buffer.
    float *buf = new float[dim];
    std::memcpy(buf, results[0], sizeof(float) * static_cast<size_t>(dim));

    // encode() allocates each pointer with new[]; release them all.
    for (auto *p : results)
      delete[] p;

    *out_embedding = buf;
    *out_dim = dim;
    return CAUSAL_LM_ERROR_NONE;
  } catch (const std::exception &e) {
    LOGE("encodeModelHandle: exception: %s", e.what());
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  } catch (...) {
    LOGE("encodeModelHandle: unknown exception");
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
}

void freeEmbedding(float *embedding) { delete[] embedding; }

ErrorCode unloadModelHandle(CausalLmHandle handle) {
  if (handle == nullptr) {
    return CAUSAL_LM_ERROR_NONE;
  }
  std::lock_guard<std::mutex> lock(handle->mtx);
  clear_handle_models(*handle);
  handle->architectures.clear();
  handle->model_dirs.clear();
  handle->chat_templates.clear();
  handle->grammar_manager.reset();
  handle->dynamic_grammar_schemas.clear();
  handle->last_output.clear();
  handle->initialization_duration_ms.clear();
  handle->initialized = false;
  reset_handle_session_state(*handle);
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode destroyModelHandle(CausalLmHandle handle) {
  if (handle == nullptr) {
    return CAUSAL_LM_ERROR_NONE;
  }
  // Take the mutex to make sure no in-flight call on this handle is still
  // running, then release and delete. Any caller that still holds a pointer
  // to an earlier borrowed output buffer is reading freed memory after this
  // point.
  {
    std::lock_guard<std::mutex> lock(handle->mtx);
    clear_handle_models(*handle);
    handle->architectures.clear();
    handle->model_dirs.clear();
    handle->chat_templates.clear();
    handle->grammar_manager.reset();
    handle->dynamic_grammar_schemas.clear();
    handle->last_output.clear();
    handle->initialization_duration_ms.clear();
    handle->initialized = false;
    reset_handle_session_state(*handle);
  }
  delete handle;
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode cancelModelHandle(CausalLmHandle handle) {
  LOGD("[DEBUG] cancelModelHandle: handle=%p", (void *)handle);

  if (handle == nullptr) {
    LOGE("[DEBUG] cancelModelHandle: handle is nullptr, returning "
         "INVALID_PARAMETER");
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  // Do not take the inference mutex: run() holds it for the full decode. The
  // separate target mutex makes model lifetime stable while requestStop()
  // sets its atomic flag, and unload/destroy clear targets before ownership.
  std::lock_guard<std::mutex> cancellation_lock(handle->cancellation_mtx);
  if (handle->cancellation_targets.empty()) {
    LOGE(
      "[DEBUG] cancelModelHandle: not initialized, returning NOT_INITIALIZED");
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }
  if (!handle->run_active && handle->run_announced) {
    // This specific request has been announced but generation has not started.
    // ScopedGeneration consumes the stop request; ScopedRunRequest discards it
    // on parse, validation, or capability failures.
    handle->cancellation_pending = true;
    LOGD("[DEBUG] cancelModelHandle: cancellation queued before generation");
    return CAUSAL_LM_ERROR_NONE;
  }
  if (!handle->run_active) {
    // Cancellation is request-scoped. An idle cancellation must never affect
    // an unrelated future generation.
    LOGD("[DEBUG] cancelModelHandle: no active request");
    return CAUSAL_LM_ERROR_NONE;
  }

  for (size_t i = 0; i < handle->cancellation_targets.size(); ++i) {
    if (!request_model_stop(handle->cancellation_targets[i])) {
      LOGD("[DEBUG] cancelModelHandle: model[%zu] is not cancellable", i);
      continue;
    }
    LOGD("[DEBUG] cancelModelHandle: requested stop on model[%zu]", i);
  }

  LOGD("[DEBUG] cancelModelHandle: returning NONE (success)");
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode quickAiArmRunCancellation(CausalLmHandle handle) {
  if (handle == nullptr)
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;

  std::lock_guard<std::mutex> cancellation_lock(handle->cancellation_mtx);
  if (handle->cancellation_targets.empty())
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  if (handle->run_announced || handle->run_active)
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;

  handle->run_announced = true;
  handle->cancellation_pending = false;
  return CAUSAL_LM_ERROR_NONE;
}

void quickAiDisarmRunCancellation(CausalLmHandle handle) {
  if (handle == nullptr)
    return;

  std::lock_guard<std::mutex> cancellation_lock(handle->cancellation_mtx);
  if (!handle->run_active) {
    handle->run_announced = false;
    handle->cancellation_pending = false;
  }
}

/*============================================================================
 * Multimodal API Implementation
 *
 * Preconditions: the handle must have been loaded from a multi-model
 * nntr_config.json carrying at least two sub-models. The first sub-model
 * is expected to be the vision encoder and the second the embedding-input LLM.

 * * Single-model and incompatible handles return CAUSAL_LM_ERROR_UNSUPPORTED.

 * *============================================================================*/

#if defined(ENABLE_QNN_MODELS) &&                                              \
  defined(QUICKAI_ENABLE_EXPERIMENTAL_MULTIMODAL_NNTRAINER_API)
/**
 * Model-agnostic multimodal composer. Works through base Transformer virtuals
 * only (no concrete-model casts), so any [vision producer, LLM consumer] pair
 * (e.g. future vJEPA/siglip+LFM) is driven identically.
 *
 *   llm:          embedding CONSUMER (lookupEmbedding / run_with_embeddings)
 *   image_embeds: producer output; ownership taken here (freed before return)
 */
static ErrorCode
execute_multimodal(CausalLmModel &h, causallm::Transformer *llm,
                   causallm::multimodal_pointer image_embeds,
                   const std::string &prompt, CausalLmTokenCallback callback,
                   void *user_data, causallm::XGrammar *grammar = nullptr) {
  std::unique_ptr<void, decltype(&std::free)> image_guard(image_embeds.first,
                                                          &std::free);
  if (llm == nullptr || !llm->supportsTextGeneration() ||
      !llm->supportsEmbeddingInput()) {
    LOGE("[MM] model does not support embedding-based text generation");
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }
  if (image_embeds.first == nullptr || image_embeds.second == 0) {
    LOGE("[MM] vision encoder returned no embeddings");
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
  auto *tok = llm->getTokenizer();
  if (tok == nullptr) {
    LOGE("[MM] llm has no tokenizer");
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }
  std::vector<int> text_ids = tok->Encode(prompt);
  int32_t image_token_id = tok->TokenToId("<|image|>");

  const size_t bpt = llm->embeddingBytesPerToken();
  if (bpt == 0) {
    LOGE("[MM] llm embedding table not loaded (needs uses_embedding=false + "
         "embedding_file_name)");
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }
  if (image_embeds.second % bpt != 0) {
    LOGE("[MM] image_embeds.size=%zu not a multiple of bpt=%zu",
         image_embeds.second, bpt);
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
  const size_t n_image = image_embeds.second / bpt;

  if (image_token_id < 0) {
    LOGE("[MM] tokenizer does not define the <|image|> placeholder");
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }
  const size_t placeholder_count = static_cast<size_t>(
    std::count(text_ids.begin(), text_ids.end(), image_token_id));
  if (placeholder_count != 1) {
    LOGE("[MM] expected exactly one image placeholder, found %zu",
         placeholder_count);
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  const auto it_img =
    std::find(text_ids.begin(), text_ids.end(), image_token_id);
  const size_t img_pos =
    static_cast<size_t>(std::distance(text_ids.begin(), it_img));
  const size_t n_text_kept = text_ids.size() - 1;
  if (n_image > (std::numeric_limits<size_t>::max)() - n_text_kept) {
    LOGE("[MM] combined token count overflow");
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
  const size_t n_total = n_text_kept + n_image;
  if (n_total > (std::numeric_limits<size_t>::max)() / bpt) {
    LOGE("[MM] combined embedding size overflow");
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
  LOGD("[MM] text=%zu image=%zu total=%zu pos=%zu", text_ids.size(), n_image,
       n_total, img_pos);

  std::vector<uint8_t> combined(n_total * bpt);
  uint8_t *dst = combined.data();
  auto copy_text_range = [&](size_t start, size_t end) -> bool {
    for (size_t i = start; i < end; ++i) {
      const void *e = llm->lookupEmbedding(text_ids[i]);
      if (e == nullptr) {
        LOGE("[MM] lookupEmbedding(%d) null", text_ids[i]);
        return false;
      }
      std::memcpy(dst, e, bpt);
      dst += bpt;
    }
    return true;
  };
  if (!copy_text_range(0, img_pos)) {
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
  std::memcpy(dst, image_embeds.first, n_image * bpt);
  dst += n_image * bpt;
  const size_t after_start = img_pos + 1;
  if (!copy_text_range(after_start, text_ids.size())) {
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
  image_guard.reset();
  image_embeds.first = nullptr;

  CallbackStreamer streamer;
  callback_streamer_init(&streamer, callback, user_data);
  // Attach via the cast helper: setStreamer lives on Quick_Dot_AI_QNN /
  // CausalLM, not on the base Transformer the composer drives through.
  if (!set_model_streamer(llm, &streamer.base)) {
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }
  std::unique_ptr<XGrammarLogitsProcessor> grammar_processor;
  struct Detach {
    causallm::Transformer *t;
    bool detach_logits = false;
    causallm::Quick_Dot_AI_QNN *qnn_model = nullptr;
    ~Detach() {
      set_model_streamer(t, nullptr);
      if (detach_logits)
        t->setLogitsProcessor(nullptr);
      if (qnn_model != nullptr)
        qnn_model->resetXGrammar();
    }
  } detach_guard{llm};

  bool qnn_grammar_attached = false;
  if (grammar != nullptr) {
    if (auto *qnn_model = as_qnn_model(llm)) {
      qnn_model->setXGrammar(grammar);
      detach_guard.qnn_model = qnn_model;
      qnn_grammar_attached = true;
    }
  }
  if (grammar != nullptr && !qnn_grammar_attached) {
    grammar_processor = std::make_unique<XGrammarLogitsProcessor>(
      grammar, [llm]() { request_model_stop(llm); });
    llm->setLogitsProcessor(grammar_processor.get());
    detach_guard.detach_logits = true;
  }

  try {
    llm->run_with_embeddings(combined.data(), n_total, text_ids,
                             /*do_sample=*/false, /*log_output=*/h.verbose);
    if (grammar_processor != nullptr && grammar_processor->failed()) {
      LOGE("[MM] grammar rejected a generated token");
      return CAUSAL_LM_ERROR_INFERENCE_FAILED;
    }
    if (detach_guard.qnn_model != nullptr &&
        detach_guard.qnn_model->hasXGrammarFailure()) {
      LOGE("[MM] QNN grammar rejected a generated token");
      return CAUSAL_LM_ERROR_INFERENCE_FAILED;
    }
    get_model_output(llm, h.last_output);
    h.kv_len = llm->getKvLen();
  } catch (const std::exception &e) {
    LOGE("[MM] llm threw: %s", e.what());
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
  return CAUSAL_LM_ERROR_NONE;
}

/**
 * Run the vision encoder (producer) on raw pixels and return its image
 * embeddings, after matching its quant space to the LLM consumer.
 * @return image_embeds (ownership transferred to caller) or {nullptr,0}.
 */
static causallm::multimodal_pointer
run_vision_encoder(CausalLmModel &h, const char *prompt,
                   const float *pixelValues, size_t valueCount,
                   int originalHeight, int originalWidth) {
  causallm::Transformer *vision = h.models[0].get();
  causallm::Transformer *llm = h.models[1].get();

  if (!vision->supportsImageEncoding() || !llm->supportsEmbeddingInput()) {
    LOGE("[MM] model pair does not support multimodal composition");
    return {nullptr, 0};
  }

  auto info = llm->get_embedding_info();
  vision->set_quant_param(info.first, info.second);

  if (valueCount > (std::numeric_limits<size_t>::max)() / sizeof(float)) {
    LOGE("[MM] image tensor byte size overflow");
    return {nullptr, 0};
  }
  const size_t pixel_bytes = valueCount * sizeof(float);
  causallm::multimodal_pointer image_in{const_cast<float *>(pixelValues),
                                        pixel_bytes};
  return vision->run_image(std::string(prompt ? prompt : ""), image_in,
                           originalHeight, originalWidth, /*do_sample=*/false,
                           "", "", h.verbose);
}
#endif // ENABLE_QNN_MODELS &&
       // QUICKAI_ENABLE_EXPERIMENTAL_MULTIMODAL_NNTRAINER_API

#ifdef ENABLE_QNN_MODELS
/**
 * Standalone vision/video encoder run (no LLM). Wraps models[0]->run_image
 * and transfers the malloc'd output buffer to the caller. See header for the
 * ownership contract (free with freeImageEmbedding, not freeEmbedding).
 *
 * Always defined under ENABLE_QNN_MODELS (part of the public ABI). The
 * run_image- dependent body is gated on the experimental multimodal macro;
 * without it the function returns CAUSAL_LM_ERROR_UNSUPPORTED.
 */
ErrorCode encodeImageModelHandle(CausalLmHandle handle,
                                 const float *pixelValues, size_t numFloats,
                                 int height, int width, void **out_embedding,
                                 int *out_bytes) {
  if (handle == nullptr || pixelValues == nullptr || out_embedding == nullptr ||
      out_bytes == nullptr)
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;

  *out_embedding = nullptr;
  *out_bytes = 0;

  auto &h = *handle;
  std::lock_guard<std::mutex> lock(h.mtx);
  if (!h.initialized || h.models.empty()) {
    LOGE("encodeImageModelHandle: handle not initialized or empty");
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

#if defined(QUICKAI_ENABLE_EXPERIMENTAL_MULTIMODAL_NNTRAINER_API)
  causallm::Transformer *vision = h.models[0].get();
  if (!vision->supportsImageEncoding()) {
    LOGE("encodeImageModelHandle: model is not an image encoder");
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }
  // Intentionally do NOT call set_quant_param: standalone encode keeps
  // llm_quant_param_given_ == false so run_image returns the raw quantized
  // embedding via plain memcpy (no LLM consumer required).

  const size_t pixel_bytes = numFloats * sizeof(float);
  causallm::multimodal_pointer image_in{const_cast<float *>(pixelValues),
                                        pixel_bytes};
  try {
    causallm::multimodal_pointer embeds =
      vision->run_image(std::string(""), image_in, height, width,
                        /*do_sample=*/false, "", "", h.verbose);
    if (embeds.first == nullptr || embeds.second == 0) {
      LOGE("encodeImageModelHandle: run_image returned empty output");
      return CAUSAL_LM_ERROR_INFERENCE_FAILED;
    }
    *out_embedding = embeds.first; // ownership transferred to caller
    *out_bytes = static_cast<int>(embeds.second);
  } catch (const std::exception &e) {
    LOGE("encodeImageModelHandle: exception: %s", e.what());
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  } catch (...) {
    LOGE("encodeImageModelHandle: unknown exception");
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
  return CAUSAL_LM_ERROR_NONE;
#else
  // Multimodal vision encoding relies on Transformer::run_image, which the
  // main nntrainer API does not provide. The proprietary vision QNN models
  // that implemented it (e.g. vjepa2-qnn) are excluded from the build. See
  // docs/qnn-model-main-adaptation-todo.ko.md.
  (void)height;
  (void)width;
  (void)numFloats;
  LOGE("encodeImageModelHandle: multimodal vision encoder API is not enabled");
  return CAUSAL_LM_ERROR_UNSUPPORTED;
#endif
}

void freeImageEmbedding(void *embedding) { std::free(embedding); }
#endif // ENABLE_QNN_MODELS

#ifndef ENABLE_QNN_MODELS
ErrorCode encodeImageModelHandle(CausalLmHandle handle,
                                 const float *pixelValues, size_t numFloats,
                                 int height, int width, void **out_embedding,
                                 int *out_bytes) {
  (void)handle;
  (void)pixelValues;
  (void)numFloats;
  (void)height;
  (void)width;
  if (out_embedding)
    *out_embedding = nullptr;
  if (out_bytes)
    *out_bytes = 0;
  return CAUSAL_LM_ERROR_UNSUPPORTED;
}

void freeImageEmbedding(void *embedding) { (void)embedding; }
#endif // !ENABLE_QNN_MODELS

/*============================================================================

 * * OpenAI JSON streaming API implementation

 * *============================================================================*/

static ErrorCode prepare_openai_grammar(
  CausalLmModel &h, const causallm::openai::GrammarSelection &grammar_selection,
  causallm::XGrammar **grammar, std::string &grammar_key) {
  *grammar = nullptr;
  grammar_key.clear();
  if (grammar_selection.kind == causallm::openai::GrammarKind::NONE) {
    return CAUSAL_LM_ERROR_NONE;
  }
  if (!h.grammar_manager || !h.grammar_manager->isInitialized()) {
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }

  json schema;
  if (grammar_selection.kind == causallm::openai::GrammarKind::JSON_OBJECT) {
    schema = {{"type", "object"}};
  } else {
    schema = grammar_selection.schema;
  }
  const std::string schema_text = schema.dump();
  const std::string key_prefix =
    "__openai_" + std::to_string(std::hash<std::string>{}(schema_text));
  grammar_key = key_prefix;
  size_t collision = 0;
  while (true) {
    const auto existing = h.dynamic_grammar_schemas.find(grammar_key);
    if (existing != h.dynamic_grammar_schemas.end() &&
        existing->second == schema_text) {
      break;
    }
    if (existing == h.dynamic_grammar_schemas.end() &&
        !h.grammar_manager->hasTool(grammar_key))
      break;
    grammar_key = key_prefix + "_" + std::to_string(++collision);
  }

  if (!h.grammar_manager->hasTool(grammar_key)) {
    constexpr size_t MAX_DYNAMIC_GRAMMARS = 16;
    if (h.dynamic_grammar_schemas.size() >= MAX_DYNAMIC_GRAMMARS) {
      const auto victim = h.dynamic_grammar_schemas.begin();
      h.grammar_manager->unregisterTool(victim->first);
      h.dynamic_grammar_schemas.erase(victim);
    }
    if (!h.grammar_manager->registerTool(grammar_key, schema_text)) {
      return CAUSAL_LM_ERROR_INFERENCE_FAILED;
    }
  }
  h.dynamic_grammar_schemas[grammar_key] = schema_text;
  h.grammar_manager->resetGrammar(grammar_key);
  *grammar = h.grammar_manager->getGrammar(grammar_key);
  return *grammar != nullptr ? CAUSAL_LM_ERROR_NONE
                             : CAUSAL_LM_ERROR_INFERENCE_FAILED;
}

static ErrorCode
validate_openai_images(const causallm::openai::Request &request,
                       const QuickAiImageTensorV1 *images, size_t image_count) {
  if (request.image_sources.size() != image_count ||
      (image_count > 0 && images == nullptr)) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  for (size_t i = 0; i < image_count; ++i) {
    const auto &image = images[i];
    if (image.struct_size != sizeof(QuickAiImageTensorV1) ||
        image.source == nullptr || image.values == nullptr ||
        image.value_count == 0 || image.patch_count == 0 ||
        image.channels == 0 || image.patch_height == 0 ||
        image.patch_width == 0 || image.original_height == 0 ||
        image.original_width == 0 ||
        image.original_height >
          static_cast<uint32_t>((std::numeric_limits<int>::max)()) ||
        image.original_width >
          static_cast<uint32_t>((std::numeric_limits<int>::max)()) ||
        request.image_sources[i] != image.source ||
        image.layout < QUICK_AI_IMAGE_LAYOUT_MODEL_NATIVE ||
        image.layout > QUICK_AI_IMAGE_LAYOUT_CHW) {
      return CAUSAL_LM_ERROR_INVALID_PARAMETER;
    }

    for (size_t value_index = 0; value_index < image.value_count;
         ++value_index) {
      if (!std::isfinite(image.values[value_index])) {
        return CAUSAL_LM_ERROR_INVALID_PARAMETER;
      }
    }

    if (image.layout != QUICK_AI_IMAGE_LAYOUT_MODEL_NATIVE) {
      size_t expected = image.patch_count;
      for (uint32_t dimension :
           {image.channels, image.patch_height, image.patch_width}) {
        if (expected > (std::numeric_limits<size_t>::max)() / dimension) {
          return CAUSAL_LM_ERROR_INVALID_PARAMETER;
        }
        expected *= dimension;
      }
      if (expected != image.value_count) {
        return CAUSAL_LM_ERROR_INVALID_PARAMETER;
      }
    }
  }
  return CAUSAL_LM_ERROR_NONE;
}

#if defined(ENABLE_QNN_MODELS) &&                                              \
  defined(QUICKAI_ENABLE_EXPERIMENTAL_MULTIMODAL_NNTRAINER_API)
static std::vector<float>
convert_chw_to_hwc(const QuickAiImageTensorV1 &image) {
  std::vector<float> converted(image.value_count);
  const size_t patch_size = static_cast<size_t>(image.channels) *
                            image.patch_height * image.patch_width;
  for (size_t patch = 0; patch < image.patch_count; ++patch) {
    const size_t patch_offset = patch * patch_size;
    for (size_t y = 0; y < image.patch_height; ++y) {
      for (size_t x = 0; x < image.patch_width; ++x) {
        for (size_t channel = 0; channel < image.channels; ++channel) {
          const size_t chw =
            patch_offset +
            (channel * image.patch_height + y) * image.patch_width + x;
          const size_t hwc = patch_offset +
                             (y * image.patch_width + x) * image.channels +
                             channel;
          converted[hwc] = image.values[chw];
        }
      }
    }
  }
  return converted;
}
#endif

/**
 * Render a validated request with the model-provided template, or with the

 * * canonical Gemma4 format used by the bundled Gemma4 configurations. The
 *
 * built-in fallback intentionally does not guess how to serialize tools or
 *
 * tool history; those require a tokenizer-supplied tool-aware template.
 */
static ErrorCode render_openai_prompt(CausalLmModel &h, size_t model_index,
                                      const causallm::openai::Request &request,
                                      std::string &formatted_input) {
  if (model_index < h.chat_templates.size() && h.chat_templates[model_index]) {
    causallm::ChatTemplate::Options template_options;
    template_options.template_name = h.chat_template_name;
    formatted_input =
      h.chat_templates[model_index]->apply(request.original, template_options);
    return CAUSAL_LM_ERROR_NONE;
  }

  if (model_index >= h.architectures.size() ||
      (h.architectures[model_index] != "Gemma4ForCausalLM" &&
       h.architectures[model_index] != "Gemma4_E2B_QNN")) {
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }
  if (!request.tools.empty()) {
    LOGE("Gemma4 built-in chat format cannot serialize tool definitions");
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }

  formatted_input = "<bos>";
  for (size_t i = 0; i < request.messages.size(); ++i) {
    const auto &message = request.messages[i];
    const auto &original_message = request.original["messages"][i];
    if (message.role == "tool" || original_message.contains("tool_calls") ||
        original_message.contains("function_call")) {
      LOGE("Gemma4 built-in chat format cannot serialize tool history");
      return CAUSAL_LM_ERROR_UNSUPPORTED;
    }

    std::string role = message.role;
    if (role == "assistant")
      role = "model";
    else if (role == "developer")
      role = "system";
    formatted_input += "<|turn>" + role + "\n";
    formatted_input +=
      causallm::openai::renderContentWithImagePlaceholders(message.content);
    formatted_input += "<turn|>\n";
  }

  bool add_generation_prompt = request.messages.back().role != "assistant";
  if (request.original.contains("add_generation_prompt")) {
    add_generation_prompt =
      request.original["add_generation_prompt"].get<bool>();
  }
  if (add_generation_prompt)
    formatted_input += "<|turn>model\n";
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode quickAiRunOpenAI(CausalLmHandle handle, const char *json_request,
                           const QuickAiImageTensorV1 *images,
                           size_t image_count, CausalLmTokenCallback callback,
                           void *user_data) {
  if (handle == nullptr || json_request == nullptr || callback == nullptr ||
      (images == nullptr && image_count != 0)) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  auto &h = *handle;
  std::lock_guard<std::mutex> lock(h.mtx);
  if (!h.initialized || h.models.empty()) {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }
  ScopedRunRequest run_request(h);

  try {
    const auto request =
      causallm::openai::parseRequest(std::string(json_request));
    if (!request.unsupported_fields.empty()) {
      LOGE("quickAiRunOpenAI: unsupported request field: %s",
           request.unsupported_fields.front().c_str());
      return CAUSAL_LM_ERROR_UNSUPPORTED;
    }
    const ErrorCode image_validation =
      validate_openai_images(request, images, image_count);
    if (image_validation != CAUSAL_LM_ERROR_NONE) {
      return image_validation;
    }

    const size_t model_index = text_generation_model_index(h);
    if (model_index >= h.models.size() || !h.models[model_index]) {
      return CAUSAL_LM_ERROR_NOT_INITIALIZED;
    }
    if (image_count > 0) {
#if defined(ENABLE_QNN_MODELS) &&                                              \
  defined(QUICKAI_ENABLE_EXPERIMENTAL_MULTIMODAL_NNTRAINER_API)
      if (image_count != 1 || h.models.size() < 2 || model_index != 1 ||
          !h.models[0]->supportsImageEncoding() ||
          !h.models[model_index]->supportsEmbeddingInput())
        return CAUSAL_LM_ERROR_UNSUPPORTED;
#else
      return CAUSAL_LM_ERROR_UNSUPPORTED;
#endif
    }

    std::string formatted_input;
    const ErrorCode render_result =
      render_openai_prompt(h, model_index, request, formatted_input);
    if (render_result != CAUSAL_LM_ERROR_NONE)
      return render_result;

    causallm::XGrammar *grammar = nullptr;
    std::string grammar_key;
    const ErrorCode grammar_result =
      prepare_openai_grammar(h, request.grammar, &grammar, grammar_key);
    if (grammar_result != CAUSAL_LM_ERROR_NONE) {
      return grammar_result;
    }
    struct GrammarReset {
      causallm::XGrammarManager *manager;
      const std::string *key;
      ~GrammarReset() {
        if (manager != nullptr && key != nullptr && !key->empty())
          manager->resetGrammar(*key);
      }
    } grammar_reset{h.grammar_manager.get(), &grammar_key};

    ScopedGeneration generation(h);
    h.last_output.clear();
    h.models[model_index]->resetConversationState();
    reset_handle_session_state(h);

    if (image_count == 0) {
      return run_model_streaming_on_handle(h, formatted_input, callback,
                                           user_data, model_index, grammar);
    }
#if defined(ENABLE_QNN_MODELS) &&                                              \
  defined(QUICKAI_ENABLE_EXPERIMENTAL_MULTIMODAL_NNTRAINER_API)
    const auto &image = images[0];
    std::vector<float> converted;
    const float *values = image.values;
    if (image.layout == QUICK_AI_IMAGE_LAYOUT_CHW) {
      converted = convert_chw_to_hwc(image);
      values = converted.data();
    }
    auto image_embeds =
      run_vision_encoder(h, formatted_input.c_str(), values, image.value_count,
                         static_cast<int>(image.original_height),
                         static_cast<int>(image.original_width));
    if (image_embeds.first == nullptr || image_embeds.second == 0) {
      return CAUSAL_LM_ERROR_INFERENCE_FAILED;
    }
    return execute_multimodal(h, h.models[model_index].get(), image_embeds,
                              formatted_input, callback, user_data, grammar);
#else
    return CAUSAL_LM_ERROR_UNSUPPORTED;
#endif
  } catch (const std::invalid_argument &e) {
    LOGE("quickAiRunOpenAI: invalid request: %s", e.what());
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  } catch (const std::exception &e) {
    LOGE("quickAiRunOpenAI: inference failed: %s", e.what());
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  } catch (...) {
    LOGE("quickAiRunOpenAI: unknown failure");
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
}
