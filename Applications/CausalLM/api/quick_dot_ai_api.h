// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    quick_dot_ai_api.h
 * @brief   Self-contained C API for Quick.AI on-device model execution.
 */
#ifndef __QUICK_DOT_AI_API_H__
#define __QUICK_DOT_AI_API_H__

#ifndef WIN_EXPORT
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  CAUSAL_LM_ERROR_NONE = 0,
  CAUSAL_LM_ERROR_INVALID_PARAMETER = 1,
  CAUSAL_LM_ERROR_MODEL_LOAD_FAILED = 2,
  CAUSAL_LM_ERROR_INFERENCE_FAILED = 3,
  CAUSAL_LM_ERROR_NOT_INITIALIZED = 4,
  CAUSAL_LM_ERROR_INFERENCE_NOT_RUN = 5,
  CAUSAL_LM_ERROR_UNSUPPORTED = 6,
  CAUSAL_LM_ERROR_UNKNOWN = 99
} ErrorCode;

typedef enum {
  CAUSAL_LM_BACKEND_CPU = 0,
  CAUSAL_LM_BACKEND_GPU = 1,
  CAUSAL_LM_BACKEND_NPU = 2,
} BackendType;

/**
 * Deprecated enum-based model selection retained for existing load callers.
 * New callers should use loadModelHandleByName() and catalog string IDs.
 */
#ifndef __TRANSFORMER_H__
typedef enum {
  CAUSAL_LM_MODEL_QWEN3_0_6B = 0,
  CAUSAL_LM_MODEL_QWEN3_1_7B_Q40 = 4,
  CAUSAL_LM_MODEL_TINY_BERT = 8,
  CAUSAL_LM_MODEL_FUNCTION_GEMMA = 9,
  CAUSAL_LM_MODEL_GEMMA4_CPU = 11,
  CAUSAL_LM_MODEL_GEMMA4_E2B_QNN = 12,
  CAUSAL_LM_MODEL_VJEPA2_QNN = 13,
  CAUSAL_LM_MODEL_OURO_EMBEDDING = 14,
} ModelType;
#endif

typedef enum {
  CAUSAL_LM_QUANTIZATION_UNKNOWN = 0,
  CAUSAL_LM_QUANTIZATION_W4A32 = 1,
  CAUSAL_LM_QUANTIZATION_W16A16 = 2,
  CAUSAL_LM_QUANTIZATION_W8A16 = 3,
  CAUSAL_LM_QUANTIZATION_W32A32 = 4,
} ModelQuantizationType;

/** Defaults copied into handles created after setOptions() returns. */
typedef struct {
  bool use_chat_template; /**< Deprecated ABI field; ignored by the run APIs. */
  bool debug_mode;        /**< Validate registered model files immediately. */
  bool verbose;           /**< Enable model output/performance logging. */
  const char *chat_template_name; /**< Named tokenizer template or NULL. */
} Config;

typedef struct {
  unsigned int prefill_tokens;
  double prefill_duration_ms;
  unsigned int generation_tokens;
  double generation_duration_ms;
  double total_duration_ms;
  double initialization_duration_ms;
  size_t peak_memory_kb;
} PerformanceMetrics;

/** Opaque owner of one loaded model or compatible model composition. */
typedef struct CausalLmModel *CausalLmHandle;

#ifndef QUICK_AI_TOKEN_CALLBACK_DEFINED
#define QUICK_AI_TOKEN_CALLBACK_DEFINED
/**
 * Token delta callback used by both generation functions.
 *
 * @return zero to continue, non-zero to request cooperative cancellation.
 * @note @p delta is UTF-8 and valid only for the callback invocation. The
 * callback is synchronous and must not re-enter the same handle.
 */
typedef int (*CausalLmTokenCallback)(const char *delta, void *user_data);
#endif

/** Memory layout of one dense patch in an image tensor sidecar. */
typedef enum {
  QUICK_AI_IMAGE_LAYOUT_MODEL_NATIVE = 0,
  QUICK_AI_IMAGE_LAYOUT_HWC = 1,
  QUICK_AI_IMAGE_LAYOUT_CHW = 2,
} QuickAiImageLayout;

/**
 * Versioned sidecar for one OpenAI image_url content part.
 *
 * Set @c struct_size exactly to @c sizeof(QuickAiImageTensorV1). The array
 * accepted by quickAiRunOpenAI() has V1 element stride; a future layout will
 * use a separately named type/entry point rather than extending this struct.
 * @c source must
 * exactly match the corresponding messages[].content[].image_url.url string.
 * Sidecars follow image occurrence order, including repeated URLs. The
 * library never downloads the URL.
 */
typedef struct {
  uint32_t struct_size;
  const char *source;
  const float *values;
  size_t value_count;
  QuickAiImageLayout layout;
  uint32_t patch_count;
  uint32_t channels;
  uint32_t patch_height;
  uint32_t patch_width;
  uint32_t original_height;
  uint32_t original_width;
} QuickAiImageTensorV1;

WIN_EXPORT ErrorCode setOptions(Config config);

/**
 * Load a model into the deprecated process-wide default handle.
 * Prefer loadModelHandleByName() for all new code.
 */
#ifndef __TRANSFORMER_H__
WIN_EXPORT ErrorCode loadModel(BackendType compute, ModelType modeltype,
                               ModelQuantizationType quant_type,
                               const char *model_base_path);
#endif

WIN_EXPORT ErrorCode getPerformanceMetrics(PerformanceMetrics *metrics);
WIN_EXPORT ErrorCode saveQnnKvCache(const char *cache_path);
WIN_EXPORT ErrorCode loadQnnKvCache(const char *cache_path);
WIN_EXPORT ErrorCode resetQnnKvCache(void);

/** Deprecated enum-based handle loader. Prefer loadModelHandleByName(). */
#ifndef __TRANSFORMER_H__
WIN_EXPORT ErrorCode loadModelHandle(BackendType compute, ModelType modeltype,
                                     ModelQuantizationType quant_type,
                                     const char *native_lib_dir,
                                     const char *model_base_path,
                                     CausalLmHandle *out_handle);
#endif

/**
 * Load a catalog model into a new independent handle.
 *
 * A registered model id may describe a text model, a fused multimodal model,
 * or a plugin-owned composite. Callers use the same loader and generation
 * entry points for each topology.
 *
 * @param model_base_path Non-empty directory containing model directories.
 * @param out_handle Receives a new handle on success and NULL on failure.
 */
WIN_EXPORT ErrorCode loadModelHandleByName(BackendType compute,
                                           const char *model_id,
                                           ModelQuantizationType quant_type,
                                           const char *native_lib_dir,
                                           const char *model_base_path,
                                           CausalLmHandle *out_handle);

/** Enable speculative decoding when the loaded model supports it. */
WIN_EXPORT ErrorCode configureSpeculativeDecoding(CausalLmHandle handle,
                                                  bool enabled);

/**
 * Load a compatible [vision encoder, text model] composition.
 *
 * This is a low-level handle-construction helper. Generation still uses
 * quickAiRunOpenAI(); an extension may instead register a fused or composite
 * model behind one catalog id and load it with loadModelHandleByName().
 * Returns CAUSAL_LM_ERROR_UNSUPPORTED when the selected descriptors or runtime
 * model interfaces cannot form the requested composition.
 */
WIN_EXPORT ErrorCode loadMultimodalHandleByName(
  BackendType compute, const char *embedding_model_id, const char *llm_model_id,
  ModelQuantizationType quant_type, const char *native_lib_dir,
  const char *model_base_path, CausalLmHandle *out_handle);

/**
 * Generate from exact, already-formatted UTF-8 text.
 *
 * No JSON parsing, chat template, role marker, grammar, or implicit previous
 * turn is added. The call streams synchronously until completion.
 */
WIN_EXPORT ErrorCode quickAiRunText(CausalLmHandle handle, const char *input,
                                    CausalLmTokenCallback callback,
                                    void *user_data);

/**
 * Generate from an OpenAI Chat Completions-compatible JSON request.
 *
 * The request is validated and rendered by the loaded tokenizer chat
 * template. Explicit response_format constraints and required/named function
 * tools use xgrammar. For text-only requests pass NULL and zero for @p images
 * and @p image_count. Multimodal callers provide one tensor per image_url
 * occurrence in the same order. The loaded descriptor must advertise
 * QDA_CAP_MULTIMODAL; image_count greater than one additionally requires
 * QDA_CAP_MULTI_IMAGE.
 *
 * Image requests first use an architecture callback registered by a fused or
 * plugin-owned composite model. If no such hook is registered, a compatible
 * [vision encoder, embedding-input LLM] handle uses the generic composer.
 * Once a full sidecar/grammar-aware callback is invoked, its return code is
 * authoritative and no fallback path is attempted.
 * That callback also receives the validated request and may apply its own
 * model-specific template when no compatible core template is available.
 * Incompatible capabilities, image counts/layouts, hooks, or model interfaces
 * return CAUSAL_LM_ERROR_UNSUPPORTED instead of dropping image inputs.
 * Extension callbacks share the C++ model ABI and must be rebuilt from the
 * same source revision as the core/API libraries.
 *
 * The loaded handle, not the optional JSON model field, selects the model.
 * Unsupported per-request generation controls are rejected instead of being
 * silently ignored; see api/README.md for the accepted subset.
 */
WIN_EXPORT ErrorCode quickAiRunOpenAI(CausalLmHandle handle,
                                      const char *json_request,
                                      const QuickAiImageTensorV1 *images,
                                      size_t image_count,
                                      CausalLmTokenCallback callback,
                                      void *user_data);

WIN_EXPORT ErrorCode saveQnnKvCacheHandle(CausalLmHandle handle,
                                          const char *cache_path);
WIN_EXPORT ErrorCode loadQnnKvCacheHandle(CausalLmHandle handle,
                                          const char *cache_path);
WIN_EXPORT ErrorCode resetQnnKvCacheHandle(CausalLmHandle handle);

/** Retrieve metrics for the most recent completed run. */
WIN_EXPORT ErrorCode getPerformanceMetricsHandle(CausalLmHandle handle,
                                                 PerformanceMetrics *metrics);

/**
 * Request cancellation from another thread. This is the only handle API that
 * may run concurrently with generation.
 */
WIN_EXPORT ErrorCode cancelModelHandle(CausalLmHandle handle);

/** Release model weights while retaining the empty handle object. */
WIN_EXPORT ErrorCode unloadModelHandle(CausalLmHandle handle);

/**
 * Destroy a handle. NULL is accepted.
 * The caller must prevent every concurrent API entry once destruction begins.
 */
WIN_EXPORT ErrorCode destroyModelHandle(CausalLmHandle handle);

/** Encode text with a sentence-embedding model. */
WIN_EXPORT ErrorCode encodeModelHandle(CausalLmHandle handle, const char *text,
                                       float **out_embedding, int *out_dim);
WIN_EXPORT void freeEmbedding(float *embedding);

/** Encode raw pixels with a standalone vision encoder. */
WIN_EXPORT ErrorCode encodeImageModelHandle(CausalLmHandle handle,
                                            const float *pixel_values,
                                            size_t value_count, int height,
                                            int width, void **out_embedding,
                                            int *out_bytes);
WIN_EXPORT void freeImageEmbedding(void *embedding);

/** Return the registered model catalog as a borrowed JSON array string. */
WIN_EXPORT const char *getModelCatalogJson(void);

#ifdef __cplusplus
}
#endif

#endif /* __QUICK_DOT_AI_API_H__ */
