// SPDX-License-Identifier: Apache-2.0
/**
 * @file   model_callbacks.h
 * @brief  Per-architecture callback registry bridging proprietary model TUs
 *         to the public C API.
 * @author jayden0701 <jrock.oh@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#pragma once
#include <functional>
#include <string>
#include <unordered_map>

#include "quick_dot_ai_api.h" // ErrorCode, CausalLmTokenCallback, CausalLmHandle

namespace causallm {
class Transformer;
class XGrammar;
namespace openai {
struct Request;
}
} // namespace causallm

/// Versioned borrowed view of the models loaded in one handle.
struct OpenAIMultimodalModelViewV1 {
  /// Set to sizeof(OpenAIMultimodalModelViewV1).
  uint32_t struct_size;
  /// Borrowed model pointer array for this synchronous call.
  causallm::Transformer *const *models;
  size_t model_count;
  /// Architecture index that owns the registered callback.
  size_t callback_model_index;
  /// Conventional text-generation model index.
  size_t text_model_index;
};

/// OpenAI multimodal callback with full sidecar and xgrammar input.
/// Every pointer is borrowed for the synchronous call.
/// The result is final, including CAUSAL_LM_ERROR_UNSUPPORTED.
/// A non-null grammar must be honored or rejected.
/// Implementations retain no pointers and detach temporary hooks.
using OpenAIMultimodalStreamingCallback = std::function<ErrorCode(
  const OpenAIMultimodalModelViewV1 *model_view,
  const causallm::openai::Request *request, const std::string *formatted_prompt,
  const QuickAiImageTensorV1 *images, size_t image_count,
  causallm::XGrammar *grammar, CausalLmTokenCallback cb, void *user_data)>;

/**
 * @brief Per-architecture callbacks registered by proprietary model TU files.
 * When a proprietary TU is absent (public build), no callbacks are
 * registered for that architecture; callers should fall back to
 * CAUSAL_LM_ERROR_UNSUPPORTED.
 */
struct ModelCallbacks {
  /**
   * Apply architecture-specific chat template to a raw single-turn input.
   * Returns empty string if not registered (caller uses raw input).
   */
  std::function<std::string(const std::string &raw_input)> format_prompt;

  /** True when this architecture requires an HTP/QNN backend. */
  bool requires_htp = false;

  /**
   * Read the current KV-cache length from a loaded transformer.
   * Used for incremental-session tracking.
   * Returns 0 if not registered.
   */
  std::function<int(causallm::Transformer *model)> read_kv_len;

  /**
   * Given the full prompt history (already-formatted), extract the latest user
   * content and rebuild it as the minimal incremental prompt for next turn.
   * Returns empty string if not registered.
   */
  std::function<std::string(const std::string &full_prompt)> incremental_prompt;

  /// Legacy streaming multimodal execution on h.models[0]/[1].
  /// The OpenAI adapter permits one unconstrained RGB image.
  /// Patches must be 512x512; CHW is converted to HWC.
  /// Other tensor contracts require the V2 registry.
  std::function<ErrorCode(CausalLmHandle handle, const float *pixel_values,
                          int num_patches, int orig_h, int orig_w,
                          const std::string &prompt, CausalLmTokenCallback cb,
                          void *user_data)>
    multimodal_streaming;

  /**
   * Blocking multimodal execution; appends generated text to *output.
   * `handle` is CausalLmHandle (= CausalLmModel*).
   */
  std::function<ErrorCode(CausalLmHandle handle, const float *pixel_values,
                          int num_patches, int orig_h, int orig_w,
                          const std::string &prompt, std::string *output)>
    multimodal_blocking;

  /**
   * Enable speculative decoding on an already-loaded model instance.
   * `model` is the handle's primary sub-model (handle->models[0]). The
   * registering TU casts it to its own concrete type to confirm draft-model
   * support before enabling. Returns CAUSAL_LM_ERROR_MODEL_LOAD_FAILED if the
   * model does not support speculative decoding; if no callback is
   * registered for the architecture, the caller treats it the same way
   * (current no-op behavior for unsupported architectures).
   */
  std::function<ErrorCode(causallm::Transformer *model, bool use_sd)>
    configure_speculative_decoding;
};

/**
 * @brief Registry keyed by architecture name string (e.g. "VendorArch_QNN").
 * Proprietary model TUs call register_for() at static-init time.
 * quick_dot_ai_api.cpp calls lookup() at runtime.
 */
class ModelCallbackRegistry {
public:
  static ModelCallbackRegistry &instance();

  /** Register callbacks for one architecture name. */
  void register_for(const std::string &architecture, ModelCallbacks cb);

  /**
   * Look up callbacks for the given architecture.
   * Returns nullptr if not registered (public architecture, or the
   * proprietary TU that would register it is absent from this build).
   */
  const ModelCallbacks *lookup(const std::string &architecture) const;

  /** True if ANY registered architecture has requires_htp = true. */
  bool any_requires_htp() const;

private:
  ModelCallbackRegistry() = default;
  ModelCallbackRegistry(const ModelCallbackRegistry &) = delete;
  ModelCallbackRegistry &operator=(const ModelCallbackRegistry &) = delete;

  std::unordered_map<std::string, ModelCallbacks> by_arch_;
};

/// Separate V2 registry for OpenAI multimodal callbacks.
/// Keeping V2 out of ModelCallbacks preserves the legacy object's size.
/// It also preserves the by-value registration calling contract.
class OpenAIMultimodalCallbackRegistry {
public:
  static OpenAIMultimodalCallbackRegistry &instance();

  void register_for(const std::string &architecture,
                    OpenAIMultimodalStreamingCallback callback);

  const OpenAIMultimodalStreamingCallback *
  lookup(const std::string &architecture) const;

private:
  std::unordered_map<std::string, OpenAIMultimodalStreamingCallback> by_arch_;
};
