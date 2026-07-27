// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    quickai_jni.cpp
 * @brief   Minimal JNI bridge for the QuickDotAI Android AAR.
 * @author  junbong.yu <junbong.yu@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include <android/log.h>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <jni.h>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "quick_dot_ai_api.h"
#include "quick_dot_ai_api_internal.h"

#define LOG_TAG "quickai_jni"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

static_assert(sizeof(jlong) >= sizeof(std::uintptr_t),
              "A Java long must hold a native model handle");

struct JniCache {
  jclass load_result_class = nullptr;
  jmethodID load_result_constructor = nullptr;
  jclass metrics_result_class = nullptr;
  jmethodID metrics_result_constructor = nullptr;
};

struct FunctionStreamContext {
  JNIEnv *env;
  jobject callback;
  jmethodID invoke;
  jmethodID int_value;
  bool failed = false;
};

JniCache g_cache;

void throw_java_exception(JNIEnv *env, const char *class_name,
                          const char *message) noexcept {
  if (env == nullptr || env->ExceptionCheck()) {
    return;
  }

  jclass exception_class = env->FindClass(class_name);
  if (exception_class == nullptr) {
    return;
  }
  env->ThrowNew(exception_class, message);
  env->DeleteLocalRef(exception_class);
}

void report_out_of_memory(JNIEnv *env, const char *operation) noexcept {
  LOGE("%s failed: out of memory", operation);
  throw_java_exception(env, "java/lang/OutOfMemoryError", operation);
}

void report_cpp_exception(JNIEnv *env, const char *operation,
                          const char *detail = nullptr) noexcept {
  if (detail != nullptr) {
    LOGE("%s failed: %s", operation, detail);
  } else {
    LOGE("%s failed with an unknown C++ exception", operation);
  }
  throw_java_exception(env, "java/lang/RuntimeException", operation);
}

void clear_jni_cache(JNIEnv *env) noexcept {
  if (env != nullptr) {
    if (g_cache.metrics_result_class != nullptr) {
      env->DeleteGlobalRef(g_cache.metrics_result_class);
    }
    if (g_cache.load_result_class != nullptr) {
      env->DeleteGlobalRef(g_cache.load_result_class);
    }
  }
  g_cache = {};
}

void clear_pending_exception(JNIEnv *env, const char *operation) noexcept {
  if (env != nullptr && env->ExceptionCheck()) {
    LOGE("%s left a pending Java exception", operation);
    env->ExceptionClear();
  }
}

CausalLmHandle handle_from_java(jlong value) noexcept {
  return reinterpret_cast<CausalLmHandle>(static_cast<std::uintptr_t>(value));
}

jlong handle_to_java(CausalLmHandle handle) noexcept {
  return static_cast<jlong>(reinterpret_cast<std::uintptr_t>(handle));
}

jclass find_global(JNIEnv *env, const char *name) {
  jclass local = env->FindClass(name);
  if (local == nullptr) {
    LOGE("FindClass failed: %s", name);
    return nullptr;
  }

  auto *global = reinterpret_cast<jclass>(env->NewGlobalRef(local));
  env->DeleteLocalRef(local);
  return global;
}

jstring new_java_string_utf8_impl(JNIEnv *env, const char *utf8) {
  std::vector<jchar> utf16;
  const auto *cursor =
    reinterpret_cast<const unsigned char *>(utf8 != nullptr ? utf8 : "");

  while (*cursor != 0) {
    uint32_t code_point = 0;
    size_t continuation_count = 0;
    uint32_t minimum = 0;
    if (*cursor < 0x80) {
      code_point = *cursor++;
    } else if ((*cursor & 0xE0) == 0xC0) {
      code_point = *cursor++ & 0x1F;
      continuation_count = 1;
      minimum = 0x80;
    } else if ((*cursor & 0xF0) == 0xE0) {
      code_point = *cursor++ & 0x0F;
      continuation_count = 2;
      minimum = 0x800;
    } else if ((*cursor & 0xF8) == 0xF0) {
      code_point = *cursor++ & 0x07;
      continuation_count = 3;
      minimum = 0x10000;
    } else {
      return nullptr;
    }

    for (size_t i = 0; i < continuation_count; ++i) {
      if ((*cursor & 0xC0) != 0x80) {
        return nullptr;
      }
      code_point = (code_point << 6) | (*cursor++ & 0x3F);
    }
    if ((continuation_count > 0 && code_point < minimum) ||
        code_point > 0x10FFFF ||
        (code_point >= 0xD800 && code_point <= 0xDFFF)) {
      return nullptr;
    }

    if (code_point <= 0xFFFF) {
      utf16.push_back(static_cast<jchar>(code_point));
    } else {
      code_point -= 0x10000;
      utf16.push_back(static_cast<jchar>(0xD800 + (code_point >> 10)));
      utf16.push_back(static_cast<jchar>(0xDC00 + (code_point & 0x3FF)));
    }
  }

  if (utf16.size() > static_cast<size_t>(std::numeric_limits<jsize>::max())) {
    return nullptr;
  }

  const jchar empty = 0;
  return env->NewString(utf16.empty() ? &empty : utf16.data(),
                        static_cast<jsize>(utf16.size()));
}

jstring new_java_string_utf8(JNIEnv *env, const char *utf8) noexcept {
  try {
    return new_java_string_utf8_impl(env, utf8);
  } catch (const std::bad_alloc &) {
    report_out_of_memory(env, "UTF-8 to Java string conversion");
  } catch (const std::exception &exception) {
    report_cpp_exception(env, "UTF-8 to Java string conversion",
                         exception.what());
  } catch (...) {
    report_cpp_exception(env, "UTF-8 to Java string conversion");
  }
  return nullptr;
}

bool java_string_to_utf8_impl(JNIEnv *env, jstring input, std::string &utf8) {
  if (input == nullptr) {
    return false;
  }

  const jsize length = env->GetStringLength(input);
  const jchar *chars = env->GetStringChars(input, nullptr);
  if (chars == nullptr) {
    return false;
  }
  struct StringCharsGuard {
    JNIEnv *env;
    jstring input;
    const jchar *chars;
    ~StringCharsGuard() { env->ReleaseStringChars(input, chars); }
  } chars_guard{env, input, chars};

  utf8.clear();
  utf8.reserve(static_cast<size_t>(length) * 3);
  bool valid = true;
  for (jsize i = 0; i < length; ++i) {
    uint32_t code_point = chars[i];
    if (code_point >= 0xD800 && code_point <= 0xDBFF) {
      if (++i >= length || chars[i] < 0xDC00 || chars[i] > 0xDFFF) {
        valid = false;
        break;
      }
      code_point =
        0x10000 + ((code_point - 0xD800) << 10) + (chars[i] - 0xDC00);
    } else if (code_point >= 0xDC00 && code_point <= 0xDFFF) {
      valid = false;
      break;
    }

    if (code_point < 0x80) {
      utf8.push_back(static_cast<char>(code_point));
    } else if (code_point < 0x800) {
      utf8.push_back(static_cast<char>(0xC0 | (code_point >> 6)));
      utf8.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    } else if (code_point < 0x10000) {
      utf8.push_back(static_cast<char>(0xE0 | (code_point >> 12)));
      utf8.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
      utf8.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    } else {
      utf8.push_back(static_cast<char>(0xF0 | (code_point >> 18)));
      utf8.push_back(static_cast<char>(0x80 | ((code_point >> 12) & 0x3F)));
      utf8.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
      utf8.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    }
  }

  return valid;
}

bool java_string_to_utf8(JNIEnv *env, jstring input,
                         std::string &utf8) noexcept {
  try {
    return java_string_to_utf8_impl(env, input, utf8);
  } catch (const std::bad_alloc &) {
    report_out_of_memory(env, "Java string to UTF-8 conversion");
  } catch (const std::exception &exception) {
    report_cpp_exception(env, "Java string to UTF-8 conversion",
                         exception.what());
  } catch (...) {
    report_cpp_exception(env, "Java string to UTF-8 conversion");
  }
  return false;
}

bool contains_nul(const std::string &value) {
  return value.find('\0') != std::string::npos;
}

struct ModelHandleGuard {
  CausalLmHandle handle = nullptr;

  ~ModelHandleGuard() noexcept {
    if (handle == nullptr) {
      return;
    }
    try {
      destroyModelHandle(handle);
    } catch (const std::exception &exception) {
      LOGE("Native model-handle cleanup failed: %s", exception.what());
    } catch (...) {
      LOGE("Native model-handle cleanup failed");
    }
  }

  CausalLmHandle release() noexcept {
    CausalLmHandle result = handle;
    handle = nullptr;
    return result;
  }
};

struct EmbeddingGuard {
  float *embedding = nullptr;

  ~EmbeddingGuard() noexcept {
    if (embedding == nullptr) {
      return;
    }
    try {
      freeEmbedding(embedding);
    } catch (const std::exception &exception) {
      LOGE("Native embedding cleanup failed: %s", exception.what());
    } catch (...) {
      LOGE("Native embedding cleanup failed");
    }
  }
};

template <typename Result, typename Callable>
Result guard_reference_entry(JNIEnv *env, const char *operation, Result neutral,
                             Callable &&callable) noexcept {
  try {
    return callable();
  } catch (const std::bad_alloc &) {
    report_out_of_memory(env, operation);
  } catch (const std::exception &exception) {
    report_cpp_exception(env, operation, exception.what());
  } catch (...) {
    report_cpp_exception(env, operation);
  }
  return neutral;
}

template <typename Callable>
jint guard_error_entry(JNIEnv *env, const char *operation, ErrorCode fallback,
                       Callable &&callable) noexcept {
  try {
    const jint result = callable();
    if (env != nullptr && env->ExceptionCheck()) {
      LOGE("%s left a pending Java exception", operation);
      return static_cast<jint>(fallback);
    }
    return result;
  } catch (const std::bad_alloc &) {
    report_out_of_memory(env, operation);
  } catch (const std::exception &exception) {
    LOGE("%s failed: %s", operation, exception.what());
  } catch (...) {
    LOGE("%s failed with an unknown C++ exception", operation);
  }
  return static_cast<jint>(fallback);
}

template <typename Callable>
void guard_void_entry(JNIEnv *env, const char *operation,
                      Callable &&callable) noexcept {
  try {
    callable();
  } catch (const std::bad_alloc &) {
    report_out_of_memory(env, operation);
  } catch (const std::exception &exception) {
    report_cpp_exception(env, operation, exception.what());
  } catch (...) {
    report_cpp_exception(env, operation);
  }
}

jobject new_load_result(JNIEnv *env, ErrorCode error_code,
                        CausalLmHandle handle) {
  if (g_cache.load_result_class == nullptr ||
      g_cache.load_result_constructor == nullptr) {
    return nullptr;
  }

  const jlong result_handle =
    error_code == CAUSAL_LM_ERROR_NONE ? handle_to_java(handle) : 0;
  return env->NewObject(g_cache.load_result_class,
                        g_cache.load_result_constructor,
                        static_cast<jint>(error_code), result_handle);
}

bool initialize_function_callback(JNIEnv *env, jobject callback,
                                  FunctionStreamContext &context) {
  jclass callback_class = env->GetObjectClass(callback);
  if (callback_class == nullptr) {
    return false;
  }
  context.invoke = env->GetMethodID(callback_class, "invoke",
                                    "(Ljava/lang/Object;)Ljava/lang/Object;");
  env->DeleteLocalRef(callback_class);
  if (context.invoke == nullptr) {
    return false;
  }

  jclass integer_class = env->FindClass("java/lang/Integer");
  if (integer_class != nullptr) {
    context.int_value = env->GetMethodID(integer_class, "intValue", "()I");
    env->DeleteLocalRef(integer_class);
  }

  if (context.invoke == nullptr || context.int_value == nullptr) {
    return false;
  }
  return true;
}

void mark_function_callback_failed(FunctionStreamContext *context,
                                   const char *message) noexcept {
  if (context == nullptr) {
    return;
  }

  context->failed = true;
  LOGE("%s", message);
}

int function_stream_trampoline(const char *delta, void *user_data) noexcept {
  auto *context = static_cast<FunctionStreamContext *>(user_data);
  if (context == nullptr || context->env == nullptr ||
      context->callback == nullptr || context->invoke == nullptr ||
      context->int_value == nullptr || context->failed) {
    return 1;
  }

  try {
    jstring text =
      new_java_string_utf8(context->env, delta != nullptr ? delta : "");
    if (text == nullptr) {
      mark_function_callback_failed(
        context, "Unable to convert model UTF-8 output to a Java string");
      return 1;
    }

    jobject result =
      context->env->CallObjectMethod(context->callback, context->invoke, text);
    context->env->DeleteLocalRef(text);
    if (context->env->ExceptionCheck()) {
      if (result != nullptr) {
        context->env->DeleteLocalRef(result);
      }
      mark_function_callback_failed(context,
                                    "Streaming callback threw an exception");
      return 1;
    }
    if (result == nullptr) {
      mark_function_callback_failed(context,
                                    "Streaming callback returned null");
      return 1;
    }

    const jint cancellation =
      context->env->CallIntMethod(result, context->int_value);
    context->env->DeleteLocalRef(result);
    if (context->env->ExceptionCheck()) {
      mark_function_callback_failed(
        context, "Unable to read streaming callback return value");
      return 1;
    }
    return cancellation;
  } catch (const std::bad_alloc &) {
    report_out_of_memory(context->env, "Streaming callback bridge");
    mark_function_callback_failed(
      context, "Out of memory while delivering streaming callback");
    return 1;
  } catch (const std::exception &exception) {
    report_cpp_exception(context->env, "Streaming callback bridge",
                         exception.what());
    mark_function_callback_failed(context, "Streaming callback bridge failed");
    return 1;
  } catch (...) {
    report_cpp_exception(context->env, "Streaming callback bridge");
    mark_function_callback_failed(context, "Streaming callback bridge failed");
    return 1;
  }
}

template <typename Callable>
jint guard_streaming_entry(JNIEnv *env, const char *operation,
                           Callable &&callable) noexcept {
  return guard_error_entry(env, operation, CAUSAL_LM_ERROR_INFERENCE_FAILED,
                           std::forward<Callable>(callable));
}

} // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm,
                                             void * /*reserved*/) noexcept {
  JNIEnv *env = nullptr;
  try {
    if (vm == nullptr ||
        vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) !=
          JNI_OK ||
        env == nullptr) {
      return JNI_ERR;
    }

    clear_jni_cache(env);
    g_cache.load_result_class =
      find_global(env, "com/example/quickdotai/NativeCausalLm$LoadResult");
    if (g_cache.load_result_class != nullptr) {
      g_cache.load_result_constructor =
        env->GetMethodID(g_cache.load_result_class, "<init>", "(IJ)V");
    }
    if (g_cache.load_result_constructor == nullptr || env->ExceptionCheck()) {
      clear_jni_cache(env);
      clear_pending_exception(env, "JNI_OnLoad");
      return JNI_ERR;
    }

    g_cache.metrics_result_class =
      find_global(env, "com/example/quickdotai/NativeCausalLm$MetricsResult");
    if (g_cache.metrics_result_class != nullptr) {
      g_cache.metrics_result_constructor =
        env->GetMethodID(g_cache.metrics_result_class, "<init>", "(IIDIDDDJ)V");
    }
    if (g_cache.metrics_result_constructor == nullptr ||
        env->ExceptionCheck()) {
      clear_jni_cache(env);
      clear_pending_exception(env, "JNI_OnLoad");
      return JNI_ERR;
    }
    return JNI_VERSION_1_6;
  } catch (const std::bad_alloc &) {
    LOGE("JNI_OnLoad failed: out of memory");
  } catch (const std::exception &exception) {
    LOGE("JNI_OnLoad failed: %s", exception.what());
  } catch (...) {
    LOGE("JNI_OnLoad failed with an unknown C++ exception");
  }

  clear_jni_cache(env);
  clear_pending_exception(env, "JNI_OnLoad");
  return JNI_ERR;
}

extern "C" JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm,
                                               void * /*reserved*/) noexcept {
  JNIEnv *env = nullptr;
  try {
    if (vm != nullptr &&
        vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) ==
          JNI_OK &&
        env != nullptr) {
      clear_jni_cache(env);
      clear_pending_exception(env, "JNI_OnUnload");
      return;
    }
  } catch (const std::exception &exception) {
    LOGE("JNI_OnUnload failed: %s", exception.what());
  } catch (...) {
    LOGE("JNI_OnUnload failed with an unknown C++ exception");
  }
  g_cache = {};
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_example_quickdotai_NativeCausalLm_loadModelHandleByNameNative(
  JNIEnv *env, jobject /*thiz*/, jint backend, jstring model_id_java,
  jint quantization, jstring native_lib_dir_java,
  jstring model_base_path_java) noexcept {
  return guard_reference_entry<jobject>(
    env, "loadModelHandleByNameNative", nullptr, [&]() -> jobject {
      std::string model_id;
      std::string native_lib_dir;
      std::string model_base_path;
      if (!java_string_to_utf8(env, model_id_java, model_id) ||
          (native_lib_dir_java != nullptr &&
           !java_string_to_utf8(env, native_lib_dir_java, native_lib_dir)) ||
          (model_base_path_java != nullptr &&
           !java_string_to_utf8(env, model_base_path_java, model_base_path)) ||
          contains_nul(model_id) || contains_nul(native_lib_dir) ||
          contains_nul(model_base_path)) {
        if (env->ExceptionCheck()) {
          return nullptr;
        }
        return new_load_result(env, CAUSAL_LM_ERROR_INVALID_PARAMETER, nullptr);
      }

      ModelHandleGuard handle_guard;
      const ErrorCode error_code = loadModelHandleByName(
        static_cast<BackendType>(backend), model_id.c_str(),
        static_cast<ModelQuantizationType>(quantization),
        native_lib_dir_java != nullptr ? native_lib_dir.c_str() : nullptr,
        model_base_path_java != nullptr ? model_base_path.c_str() : nullptr,
        &handle_guard.handle);

      jobject result = new_load_result(env, error_code, handle_guard.handle);
      if (result != nullptr && !env->ExceptionCheck() &&
          error_code == CAUSAL_LM_ERROR_NONE) {
        handle_guard.release();
      }
      return result;
    });
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_quickdotai_NativeCausalLm_nativeQueryCatalog(
  JNIEnv *env, jobject /*thiz*/) noexcept {
  return guard_reference_entry<jstring>(
    env, "nativeQueryCatalog", nullptr, [&]() -> jstring {
      const char *catalog = getModelCatalogJson();
      return catalog != nullptr ? new_java_string_utf8(env, catalog) : nullptr;
    });
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_quickdotai_NativeCausalLm_encodeModelHandleNative(
  JNIEnv *env, jobject /*thiz*/, jlong handle_value,
  jstring text_java) noexcept {
  return guard_reference_entry<jfloatArray>(
    env, "encodeModelHandleNative", nullptr, [&]() -> jfloatArray {
      const CausalLmHandle handle = handle_from_java(handle_value);
      std::string text;
      if (handle == nullptr || !java_string_to_utf8(env, text_java, text) ||
          contains_nul(text)) {
        return nullptr;
      }

      EmbeddingGuard embedding_guard;
      int dimension = 0;
      const ErrorCode error_code = encodeModelHandle(
        handle, text.c_str(), &embedding_guard.embedding, &dimension);
      if (error_code != CAUSAL_LM_ERROR_NONE ||
          embedding_guard.embedding == nullptr || dimension <= 0) {
        return nullptr;
      }
      if (static_cast<size_t>(dimension) >
          static_cast<size_t>(std::numeric_limits<jsize>::max())) {
        throw_java_exception(env, "java/lang/IllegalStateException",
                             "Native embedding dimension exceeds JNI limits");
        return nullptr;
      }

      const jsize result_dimension = static_cast<jsize>(dimension);
      jfloatArray result = env->NewFloatArray(result_dimension);
      if (result != nullptr) {
        env->SetFloatArrayRegion(result, 0, result_dimension,
                                 embedding_guard.embedding);
      }
      return result;
    });
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_quickdotai_NativeCausalLm_runTextStreamingNative(
  JNIEnv *env, jobject /*thiz*/, jlong handle_value, jstring input_java,
  jobject callback) noexcept {
  return guard_streaming_entry(env, "runTextStreamingNative", [&]() -> jint {
    if (handle_value == 0 || input_java == nullptr || callback == nullptr) {
      return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
    }

    FunctionStreamContext context{env, callback, nullptr, nullptr};
    if (!initialize_function_callback(env, callback, context)) {
      return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
    }

    std::string input;
    if (!java_string_to_utf8(env, input_java, input) || contains_nul(input)) {
      return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
    }

    const ErrorCode result =
      quickAiRunText(handle_from_java(handle_value), input.c_str(),
                     &function_stream_trampoline, &context);
    return static_cast<jint>(context.failed ? CAUSAL_LM_ERROR_INFERENCE_FAILED
                                            : result);
  });
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_quickdotai_NativeCausalLm_runOpenAIStreamingNative(
  JNIEnv *env, jobject /*thiz*/, jlong handle_value, jstring json_request_java,
  jobjectArray image_sources_java, jobjectArray pixel_values_java,
  jintArray layouts_java, jintArray patch_counts_java, jintArray channels_java,
  jintArray patch_heights_java, jintArray patch_widths_java,
  jintArray original_heights_java, jintArray original_widths_java,
  jobject callback) noexcept {
  return guard_streaming_entry(env, "runOpenAIStreamingNative", [&]() -> jint {
    if (handle_value == 0 || json_request_java == nullptr ||
        image_sources_java == nullptr || pixel_values_java == nullptr ||
        layouts_java == nullptr || patch_counts_java == nullptr ||
        channels_java == nullptr || patch_heights_java == nullptr ||
        patch_widths_java == nullptr || original_heights_java == nullptr ||
        original_widths_java == nullptr || callback == nullptr) {
      return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
    }

    const jsize image_count = env->GetArrayLength(image_sources_java);
    if (env->GetArrayLength(pixel_values_java) != image_count ||
        env->GetArrayLength(layouts_java) != image_count ||
        env->GetArrayLength(patch_counts_java) != image_count ||
        env->GetArrayLength(channels_java) != image_count ||
        env->GetArrayLength(patch_heights_java) != image_count ||
        env->GetArrayLength(patch_widths_java) != image_count ||
        env->GetArrayLength(original_heights_java) != image_count ||
        env->GetArrayLength(original_widths_java) != image_count) {
      return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
    }

    FunctionStreamContext context{env, callback, nullptr, nullptr};
    if (!initialize_function_callback(env, callback, context)) {
      return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
    }

    std::string json_request;
    if (!java_string_to_utf8(env, json_request_java, json_request) ||
        contains_nul(json_request)) {
      return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
    }

    const size_t count = static_cast<size_t>(image_count);
    std::vector<jint> layouts(count);
    std::vector<jint> patch_counts(count);
    std::vector<jint> channels(count);
    std::vector<jint> patch_heights(count);
    std::vector<jint> patch_widths(count);
    std::vector<jint> original_heights(count);
    std::vector<jint> original_widths(count);
    if (image_count > 0) {
      env->GetIntArrayRegion(layouts_java, 0, image_count, layouts.data());
      env->GetIntArrayRegion(patch_counts_java, 0, image_count,
                             patch_counts.data());
      env->GetIntArrayRegion(channels_java, 0, image_count, channels.data());
      env->GetIntArrayRegion(patch_heights_java, 0, image_count,
                             patch_heights.data());
      env->GetIntArrayRegion(patch_widths_java, 0, image_count,
                             patch_widths.data());
      env->GetIntArrayRegion(original_heights_java, 0, image_count,
                             original_heights.data());
      env->GetIntArrayRegion(original_widths_java, 0, image_count,
                             original_widths.data());
      if (env->ExceptionCheck()) {
        return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
      }
    }

    std::vector<std::string> sources(count);
    std::vector<std::vector<float>> pixel_values(count);
    for (jsize i = 0; i < image_count; ++i) {
      const size_t index = static_cast<size_t>(i);
      auto source_java =
        static_cast<jstring>(env->GetObjectArrayElement(image_sources_java, i));
      auto values_java = static_cast<jfloatArray>(
        env->GetObjectArrayElement(pixel_values_java, i));
      if (source_java == nullptr || values_java == nullptr ||
          !java_string_to_utf8(env, source_java, sources[index]) ||
          contains_nul(sources[index])) {
        if (source_java != nullptr) {
          env->DeleteLocalRef(source_java);
        }
        if (values_java != nullptr) {
          env->DeleteLocalRef(values_java);
        }
        return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
      }

      const jsize value_count = env->GetArrayLength(values_java);
      if (value_count <= 0) {
        env->DeleteLocalRef(source_java);
        env->DeleteLocalRef(values_java);
        return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
      }

      pixel_values[index].resize(static_cast<size_t>(value_count));
      env->GetFloatArrayRegion(values_java, 0, value_count,
                               pixel_values[index].data());
      env->DeleteLocalRef(source_java);
      env->DeleteLocalRef(values_java);
      if (env->ExceptionCheck()) {
        return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
      }
    }

    std::vector<QuickAiImageTensorV1> images(count);
    for (size_t i = 0; i < count; ++i) {
      if (layouts[i] < QUICK_AI_IMAGE_LAYOUT_MODEL_NATIVE ||
          layouts[i] > QUICK_AI_IMAGE_LAYOUT_CHW || patch_counts[i] < 0 ||
          channels[i] < 0 || patch_heights[i] < 0 || patch_widths[i] < 0 ||
          original_heights[i] < 0 || original_widths[i] < 0) {
        return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
      }

      QuickAiImageTensorV1 &image = images[i];
      image.struct_size = static_cast<uint32_t>(sizeof(QuickAiImageTensorV1));
      image.source = sources[i].c_str();
      image.values = pixel_values[i].data();
      image.value_count = pixel_values[i].size();
      image.layout = static_cast<QuickAiImageLayout>(layouts[i]);
      image.patch_count = static_cast<uint32_t>(patch_counts[i]);
      image.channels = static_cast<uint32_t>(channels[i]);
      image.patch_height = static_cast<uint32_t>(patch_heights[i]);
      image.patch_width = static_cast<uint32_t>(patch_widths[i]);
      image.original_height = static_cast<uint32_t>(original_heights[i]);
      image.original_width = static_cast<uint32_t>(original_widths[i]);
    }

    const ErrorCode result =
      quickAiRunOpenAI(handle_from_java(handle_value), json_request.c_str(),
                       images.empty() ? nullptr : images.data(), images.size(),
                       &function_stream_trampoline, &context);
    return static_cast<jint>(context.failed ? CAUSAL_LM_ERROR_INFERENCE_FAILED
                                            : result);
  });
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_example_quickdotai_NativeCausalLm_getPerformanceMetricsHandleNative(
  JNIEnv *env, jobject /*thiz*/, jlong handle_value) noexcept {
  return guard_reference_entry<jobject>(
    env, "getPerformanceMetricsHandleNative", nullptr, [&]() -> jobject {
      PerformanceMetrics metrics{};
      const ErrorCode error_code =
        getPerformanceMetricsHandle(handle_from_java(handle_value), &metrics);

      if (g_cache.metrics_result_class == nullptr ||
          g_cache.metrics_result_constructor == nullptr) {
        return nullptr;
      }
      return env->NewObject(
        g_cache.metrics_result_class, g_cache.metrics_result_constructor,
        static_cast<jint>(error_code),
        static_cast<jint>(metrics.prefill_tokens),
        static_cast<jdouble>(metrics.prefill_duration_ms),
        static_cast<jint>(metrics.generation_tokens),
        static_cast<jdouble>(metrics.generation_duration_ms),
        static_cast<jdouble>(metrics.total_duration_ms),
        static_cast<jdouble>(metrics.initialization_duration_ms),
        static_cast<jlong>(metrics.peak_memory_kb));
    });
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_quickdotai_NativeCausalLm_destroyModelHandleNative(
  JNIEnv *env, jobject /*thiz*/, jlong handle_value) noexcept {
  return guard_error_entry(
    env, "destroyModelHandleNative", CAUSAL_LM_ERROR_UNKNOWN, [&]() -> jint {
      return static_cast<jint>(
        destroyModelHandle(handle_from_java(handle_value)));
    });
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_quickdotai_NativeCausalLm_armRunCancellationNative(
  JNIEnv *env, jobject /*thiz*/, jlong handle_value) noexcept {
  return guard_error_entry(
    env, "armRunCancellationNative", CAUSAL_LM_ERROR_UNKNOWN, [&]() -> jint {
      return static_cast<jint>(
        quickAiArmRunCancellation(handle_from_java(handle_value)));
    });
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_quickdotai_NativeCausalLm_disarmRunCancellationNative(
  JNIEnv *env, jobject /*thiz*/, jlong handle_value) noexcept {
  guard_void_entry(env, "disarmRunCancellationNative", [&]() {
    quickAiDisarmRunCancellation(handle_from_java(handle_value));
  });
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_quickdotai_NativeCausalLm_cancelModelHandleNative(
  JNIEnv *env, jobject /*thiz*/, jlong handle_value) noexcept {
  return guard_error_entry(
    env, "cancelModelHandleNative", CAUSAL_LM_ERROR_UNKNOWN, [&]() -> jint {
      return static_cast<jint>(
        cancelModelHandle(handle_from_java(handle_value)));
    });
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_quickdotai_NativeCausalLm_configureSpeculativeDecodingNative(
  JNIEnv *env, jobject /*thiz*/, jlong handle_value,
  jboolean enabled) noexcept {
  return guard_error_entry(
    env, "configureSpeculativeDecodingNative", CAUSAL_LM_ERROR_UNKNOWN,
    [&]() -> jint {
      return static_cast<jint>(configureSpeculativeDecoding(
        handle_from_java(handle_value), enabled == JNI_TRUE));
    });
}
