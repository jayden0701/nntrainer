// SPDX-License-Identifier: Apache-2.0
/**
 * @file   quick_dot_ai_extension_api.h
 * @brief  Versioned C ABI for Quick.AI model extensions
 * @author Quick.AI Team
 * @bug    No known bugs except for NYI items
 *
 * This header intentionally exposes only C-compatible, fixed-width views.
 * Every pointer passed to an extension callback is borrowed for that
 * synchronous callback invocation. Extensions must not retain those pointers.
 *
 * Registration copies all descriptor and architecture strings into
 * host-owned storage. Function pointers and user_data are process-lifetime:
 * unregistering and unloading a successfully registered plugin are not
 * supported.
 *
 * A callback must not allow a C++ exception to cross this C ABI. It must return
 * one of QuickAiExtensionStatusCode. The host normalizes unknown callback
 * results to QUICK_AI_EXTENSION_STATUS_INFERENCE_FAILED.
 */
#ifndef QUICK_DOT_AI_EXTENSION_API_H_
#define QUICK_DOT_AI_EXTENSION_API_H_

#include <stdint.h>

#if defined(_WIN32)
#if defined(QUICK_AI_EXTENSION_API_BUILD)
#define QUICK_AI_EXTENSION_EXPORT __declspec(dllexport)
#else
#define QUICK_AI_EXTENSION_EXPORT __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define QUICK_AI_EXTENSION_EXPORT __attribute__((visibility("default")))
#else
#define QUICK_AI_EXTENSION_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * ABI versions are intentionally independent of the product version.
 * QUICK_AI_EXTENSION_BUILD_TAG must be changed whenever the opaque
 * Transformer ABI or callback semantics change.
 */
#define QUICK_AI_EXTENSION_ABI_MAJOR 1u
#define QUICK_AI_EXTENSION_ABI_MINOR 0u
#define QUICK_AI_EXTENSION_TRANSFORMER_ABI_VERSION 1u
#define QUICK_AI_EXTENSION_BUILD_TAG                                           \
  "nntrainer.quickai.extension-v1.transformer-v1.contract-r1"

typedef int32_t QuickAiExtensionStatus;

typedef enum {
  QUICK_AI_EXTENSION_STATUS_NONE = 0,
  QUICK_AI_EXTENSION_STATUS_INVALID_PARAMETER = 1,
  QUICK_AI_EXTENSION_STATUS_MODEL_LOAD_FAILED = 2,
  QUICK_AI_EXTENSION_STATUS_INFERENCE_FAILED = 3,
  QUICK_AI_EXTENSION_STATUS_NOT_INITIALIZED = 4,
  QUICK_AI_EXTENSION_STATUS_INFERENCE_NOT_RUN = 5,
  QUICK_AI_EXTENSION_STATUS_UNSUPPORTED = 6,
  QUICK_AI_EXTENSION_STATUS_UNKNOWN = 99,
} QuickAiExtensionStatusCode;

/** Borrowed string view. size excludes any trailing NUL byte. */
typedef struct {
  const char *data;
  uint64_t size;
} QuickAiExtensionStringViewV1;

typedef uint32_t QuickAiExtensionFeatureMask;
typedef enum {
  QUICK_AI_EXTENSION_FEATURE_OPENAI_MULTIMODAL = 1u << 0,
  QUICK_AI_EXTENSION_FEATURE_GRAMMAR = 1u << 1,
  QUICK_AI_EXTENSION_FEATURE_MULTI_IMAGE = 1u << 2,
  QUICK_AI_EXTENSION_FEATURE_SPECULATIVE = 1u << 3,
} QuickAiExtensionFeature;

typedef enum {
  QUICK_AI_EXTENSION_RUNTIME_NATIVE = 0,
} QuickAiExtensionRuntime;

typedef enum {
  QUICK_AI_EXTENSION_BACKEND_CPU = 1u << 0,
  QUICK_AI_EXTENSION_BACKEND_GPU = 1u << 1,
  QUICK_AI_EXTENSION_BACKEND_NPU = 1u << 2,
} QuickAiExtensionBackendMask;

/**
 * Values intentionally match CapabilityFlag in model_descriptor.h while this
 * extension header remains self-contained.
 */
typedef enum {
  QUICK_AI_EXTENSION_CAP_STREAMING = 1u << 0,
  QUICK_AI_EXTENSION_CAP_OPENAI_API = 1u << 1,
  QUICK_AI_EXTENSION_CAP_MULTIMODAL = 1u << 2,
  QUICK_AI_EXTENSION_CAP_TOOL_USE = 1u << 3,
  QUICK_AI_EXTENSION_CAP_EMBEDDING = 1u << 4,
  QUICK_AI_EXTENSION_CAP_MULTI_IMAGE = 1u << 5,
  QUICK_AI_EXTENSION_CAP_VISION_ENCODER = 1u << 6,
  QUICK_AI_EXTENSION_CAP_SPECULATIVE = 1u << 7,
} QuickAiExtensionCapabilityMask;

typedef enum {
  QUICK_AI_EXTENSION_IMAGE_LAYOUT_MODEL_NATIVE = 0,
  QUICK_AI_EXTENSION_IMAGE_LAYOUT_HWC = 1,
  QUICK_AI_EXTENSION_IMAGE_LAYOUT_CHW = 2,
} QuickAiExtensionImageLayout;

typedef uint32_t QuickAiExtensionGrammarKind;
typedef enum {
  QUICK_AI_EXTENSION_GRAMMAR_NONE = 0,
  QUICK_AI_EXTENSION_GRAMMAR_JSON_OBJECT = 1,
  QUICK_AI_EXTENSION_GRAMMAR_JSON_SCHEMA = 2,
  QUICK_AI_EXTENSION_GRAMMAR_TOOL_CALL = 3,
} QuickAiExtensionGrammarKindValue;

/**
 * Host-owned view of one validated image sidecar. Entries are supplied in the
 * exact image_url occurrence order from the OpenAI request.
 */
typedef struct {
  uint32_t struct_size;
  uint32_t layout;
  QuickAiExtensionStringViewV1 source;
  const float *values;
  uint64_t value_count;
  uint32_t patch_count;
  uint32_t channels;
  uint32_t patch_height;
  uint32_t patch_width;
  uint32_t original_height;
  uint32_t original_width;
  uint64_t reserved[4];
} QuickAiExtensionImageViewV1;

/** Serialized grammar request. payload is empty only when kind is NONE. */
typedef struct {
  uint32_t struct_size;
  QuickAiExtensionGrammarKind kind;
  QuickAiExtensionStringViewV1 payload;
  uint64_t reserved[4];
} QuickAiExtensionGrammarViewV1;

/**
 * Opaque, borrowed model array. Plugins may cast an entry only when built
 * against the exact build tag and Transformer ABI version accepted by the
 * host. The array and every entry are valid only during the callback.
 */
typedef struct {
  uint32_t struct_size;
  uint32_t reserved0;
  void *const *models;
  uint64_t model_count;
  uint64_t callback_model_index;
  uint64_t text_model_index;
  uint64_t reserved[4];
} QuickAiExtensionModelViewV1;

/**
 * Return non-zero to request generation stop. data is UTF-8, must not contain
 * an embedded NUL, is not required to be NUL-terminated, and is valid only for
 * the call.
 */
typedef int32_t (*QuickAiExtensionTokenCallbackV1)(const char *data,
                                                   uint64_t size,
                                                   void *user_data);

typedef struct {
  uint32_t struct_size;
  uint32_t reserved0;
  QuickAiExtensionStringViewV1 raw_json;
  /** NULL/zero when the core has no compatible chat template. */
  QuickAiExtensionStringViewV1 formatted_prompt;
  const QuickAiExtensionImageViewV1 *images;
  uint64_t image_count;
  QuickAiExtensionGrammarViewV1 grammar;
  QuickAiExtensionModelViewV1 models;
  QuickAiExtensionTokenCallbackV1 token_callback;
  void *token_user_data;
  uint64_t reserved[4];
} QuickAiExtensionOpenAIRequestV1;

/**
 * Synchronous OpenAI multimodal callback. The return value is authoritative,
 * including UNSUPPORTED; the host never falls back after invoking it.
 */
typedef QuickAiExtensionStatus (*QuickAiExtensionRunOpenAIV1)(
  const QuickAiExtensionOpenAIRequestV1 *request, void *user_data);

/** Synchronous speculative-decoding configuration callback. */
typedef QuickAiExtensionStatus (*QuickAiExtensionConfigureSpeculativeV1)(
  const QuickAiExtensionModelViewV1 *models, uint32_t enabled, void *user_data);

/**
 * Descriptor published atomically with its callbacks. All mandatory strings
 * must be non-empty. sd_variant_id may be NULL/zero.
 */
typedef struct {
  uint32_t struct_size;
  uint32_t runtime;
  uint32_t backend_mask;
  uint32_t capabilities;
  QuickAiExtensionStringViewV1 id;
  QuickAiExtensionStringViewV1 family;
  QuickAiExtensionStringViewV1 display_name;
  QuickAiExtensionStringViewV1 config_name;
  QuickAiExtensionStringViewV1 sd_variant_id;
  uint64_t reserved[4];
} QuickAiExtensionModelDescriptorV1;

/**
 * Process-lifetime registration table. architecture is both the model factory
 * architecture and the callback lookup key.
 */
typedef struct {
  uint32_t struct_size;
  uint32_t abi_major;
  uint32_t abi_minor;
  uint32_t transformer_abi_version;
  QuickAiExtensionStringViewV1 build_tag;
  QuickAiExtensionStringViewV1 architecture;
  QuickAiExtensionFeatureMask feature_mask;
  uint32_t reserved0;
  QuickAiExtensionModelDescriptorV1 descriptor;
  QuickAiExtensionRunOpenAIV1 run_openai;
  QuickAiExtensionConfigureSpeculativeV1 configure_speculative;
  void *user_data;
  uint64_t reserved[4];
} QuickAiModelExtensionV1;

typedef struct {
  uint32_t struct_size;
  uint32_t abi_major;
  uint32_t abi_minor;
  uint32_t transformer_abi_version;
  QuickAiExtensionFeatureMask supported_feature_mask;
  uint32_t reserved0;
  QuickAiExtensionStringViewV1 build_tag;
  uint64_t reserved[4];
} QuickAiExtensionHostInfoV1;

/**
 * Query the host contract for diagnostics and an early compatibility check.
 * A plugin must still register the build tag compiled into its own copy of
 * this header; copying the host-reported tag into a registration table is not
 * a compatibility mechanism and cannot make different model ABIs compatible.
 */
QUICK_AI_EXTENSION_EXPORT QuickAiExtensionStatus
quickAiGetExtensionHostInfoV1(QuickAiExtensionHostInfoV1 *out_info);

/**
 * Atomically publish one descriptor and its callbacks. Duplicate model ids or
 * architectures and every ABI/build mismatch are rejected. Successful
 * registrations cannot be replaced or unregistered.
 */
QUICK_AI_EXTENSION_EXPORT QuickAiExtensionStatus
quickAiRegisterModelExtensionV1(const QuickAiModelExtensionV1 *extension);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // QUICK_DOT_AI_EXTENSION_API_H_
