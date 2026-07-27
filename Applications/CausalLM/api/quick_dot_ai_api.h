// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    quick_dot_ai_api.h
 * @brief   Public C API for Quick.AI model loading and inference
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */
#ifndef QUICK_DOT_AI_API_H_
#define QUICK_DOT_AI_API_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(QUICK_AI_API_BUILD)
#define QUICK_AI_API_EXPORT __declspec(dllexport)
#else
#define QUICK_AI_API_EXPORT __declspec(dllimport)
#endif
#elif defined(__GNUC__) && __GNUC__ >= 4
#define QUICK_AI_API_EXPORT __attribute__((visibility("default")))
#else
#define QUICK_AI_API_EXPORT
#endif

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

/** Values emitted in the catalog JSON @c runtime field. */
typedef uint32_t QuickAiRuntimeKind;
enum {
  QUICK_AI_RUNTIME_NATIVE = 0,
  QUICK_AI_RUNTIME_LITERT = 1,
};

/** Bit values emitted in the catalog JSON @c backend_mask field. */
typedef uint32_t QuickAiBackendMask;
enum {
  QUICK_AI_BACKEND_MASK_CPU = 1u << 0,
  QUICK_AI_BACKEND_MASK_GPU = 1u << 1,
  QUICK_AI_BACKEND_MASK_NPU = 1u << 2,
};

/** Bit values emitted in the catalog JSON @c capabilities field. */
typedef uint32_t QuickAiCapabilityMask;
enum {
  QUICK_AI_CAP_STREAMING = 1u << 0,
  QUICK_AI_CAP_OPENAI_API = 1u << 1,
  QUICK_AI_CAP_MULTIMODAL = 1u << 2,
  QUICK_AI_CAP_TOOL_USE = 1u << 3,
  QUICK_AI_CAP_EMBEDDING = 1u << 4,
  QUICK_AI_CAP_MULTI_IMAGE = 1u << 5,
  QUICK_AI_CAP_VISION_ENCODER = 1u << 6,
  QUICK_AI_CAP_SPECULATIVE = 1u << 7,
};

typedef enum {
  CAUSAL_LM_QUANTIZATION_UNKNOWN = 0,
  CAUSAL_LM_QUANTIZATION_W4A32 = 1,
  CAUSAL_LM_QUANTIZATION_W16A16 = 2,
  CAUSAL_LM_QUANTIZATION_W8A16 = 3,
  CAUSAL_LM_QUANTIZATION_W32A32 = 4,
} ModelQuantizationType;

/**
 * @brief Process-wide defaults copied into subsequently loaded handles.
 *
 * @c use_chat_template is reserved and ignored. Request formatting is selected
 * by the run entry point: quickAiRunText() accepts exact text and
 * quickAiRunOpenAI() applies the loaded tokenizer template.
 */
typedef struct {
  bool use_chat_template; /**< Reserved for compatibility; currently ignored */
  bool debug_mode;        /**< Validate registered model files immediately */
  bool verbose;           /**< Default logging mode for subsequently loaded
                               handles */
  const char *chat_template_name; /**< Default template name for subsequently
                                       loaded handles; NULL selects "default" */
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

/** @brief Opaque handle to one loaded model instance. */
typedef struct CausalLmModel *CausalLmHandle;

/**
 * @brief Callback invoked synchronously for each generated UTF-8 delta.
 *
 * Return zero to continue generation or non-zero to request cancellation.
 * The delta is valid only for the duration of the callback.
 */
#ifndef QUICK_AI_TOKEN_CALLBACK_DEFINED
#define QUICK_AI_TOKEN_CALLBACK_DEFINED
typedef int (*CausalLmTokenCallback)(const char *delta, void *user_data);
#endif

/**
 * @brief Memory layout of one dense patch in an image tensor sidecar.
 *
 * A fixed-width integer is used instead of a C enum so public structure
 * layout remains stable across compilers and shared-library boundaries.
 */
typedef uint32_t QuickAiImageLayout;
enum {
  QUICK_AI_IMAGE_LAYOUT_MODEL_NATIVE = 0,
  QUICK_AI_IMAGE_LAYOUT_HWC = 1,
  QUICK_AI_IMAGE_LAYOUT_CHW = 2,
};

/**
 * @brief Versioned tensor sidecar for one OpenAI image_url content part.
 *
 * Set @c struct_size to @c sizeof(QuickAiImageTensorV1). @c source must
 * exactly match the corresponding image_url.url string in the request.
 * Sidecars must follow image occurrence order; the library does not download
 * URLs.
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

QUICK_AI_API_EXPORT ErrorCode setOptions(Config config);

/**
 * @brief Load a model selected by its catalog identifier.
 *
 * @param compute Backend requested from the model descriptor
 * @param model_id Catalog identifier or alias
 * @param quant_type Compatibility selector; validated, but current file-based
 *        descriptors select the concrete variant without a path suffix
 * @param native_lib_dir Native library directory, or NULL
 * @param model_base_path Base directory containing model files, or NULL
 * @param out_handle Receives a newly allocated handle on success and NULL on
 *        failure
 */
QUICK_AI_API_EXPORT ErrorCode loadModelHandleByName(
  BackendType compute, const char *model_id, ModelQuantizationType quant_type,
  const char *native_lib_dir, const char *model_base_path,
  CausalLmHandle *out_handle);

/**
 * @brief Enable or disable speculative decoding for a loaded handle.
 *
 * Disabling is a no-op on unsupported models. Enabling requires a compatible
 * descriptor and compatible model extension.
 */
QUICK_AI_API_EXPORT ErrorCode
configureSpeculativeDecoding(CausalLmHandle handle, bool use_sd);

/**
 * @brief Generate from exact, already-formatted UTF-8 text.
 *
 * This entry point does not parse JSON, apply a chat template, add role
 * markers, or configure a grammar.
 */
QUICK_AI_API_EXPORT ErrorCode quickAiRunText(CausalLmHandle handle,
                                             const char *input,
                                             CausalLmTokenCallback callback,
                                             void *user_data);

/**
 * @brief Generate from an OpenAI Chat Completions-compatible JSON request.
 *
 * The request is strictly validated and rendered by the loaded tokenizer chat
 * template. The handle, not an optional JSON model field, selects the model.
 * Image requests require a registered ABI-compatible model extension; once
 * invoked, its result is authoritative and the core does not fall back.
 */
QUICK_AI_API_EXPORT ErrorCode
quickAiRunOpenAI(CausalLmHandle handle, const char *json_request,
                 const QuickAiImageTensorV1 *images, size_t image_count,
                 CausalLmTokenCallback callback, void *user_data);

QUICK_AI_API_EXPORT ErrorCode saveQnnKvCacheHandle(CausalLmHandle handle,
                                                   const char *cache_path);
QUICK_AI_API_EXPORT ErrorCode loadQnnKvCacheHandle(CausalLmHandle handle,
                                                   const char *cache_path);
QUICK_AI_API_EXPORT ErrorCode resetQnnKvCacheHandle(CausalLmHandle handle);

QUICK_AI_API_EXPORT ErrorCode
getPerformanceMetricsHandle(CausalLmHandle handle, PerformanceMetrics *metrics);

/**
 * @brief Request cooperative cancellation from another thread.
 *
 * This is the only handle operation intended to run concurrently with a
 * generation call.
 */
QUICK_AI_API_EXPORT ErrorCode cancelModelHandle(CausalLmHandle handle);

/** @brief Release model resources while retaining the empty handle object. */
QUICK_AI_API_EXPORT ErrorCode unloadModelHandle(CausalLmHandle handle);

/**
 * @brief Destroy a handle; a NULL handle is accepted.
 *
 * The caller must prevent every concurrent or queued API call on the handle
 * before destruction begins.
 */
QUICK_AI_API_EXPORT ErrorCode destroyModelHandle(CausalLmHandle handle);

/**
 * @brief Encode text using a loaded sentence-embedding model.
 *
 * On success, @p out_embedding receives a newly allocated float array owned by
 * the caller. Release it with freeEmbedding().
 */
QUICK_AI_API_EXPORT ErrorCode encodeModelHandle(CausalLmHandle handle,
                                                const char *text,
                                                float **out_embedding,
                                                int *out_dim);
QUICK_AI_API_EXPORT void freeEmbedding(float *embedding);

/**
 * @brief Run a standalone vision encoder and copy its native output bytes.
 *
 * On success, @p out_embedding receives a newly allocated buffer owned by the
 * caller. Release it with freeImageEmbedding().
 */
QUICK_AI_API_EXPORT ErrorCode encodeImageModelHandle(
  CausalLmHandle handle, const float *pixel_values, size_t num_floats,
  int height, int width, void **out_embedding, int *out_bytes);
QUICK_AI_API_EXPORT void freeImageEmbedding(void *embedding);

/**
 * @brief Return a JSON array describing all registered model descriptors.
 *
 * The returned NUL-terminated string is owned by the library and remains valid
 * until the next call on the same thread. Copy it before making another call.
 */
QUICK_AI_API_EXPORT const char *getModelCatalogJson(void);

#ifdef __cplusplus
}
#endif

#endif /* QUICK_DOT_AI_API_H_ */
