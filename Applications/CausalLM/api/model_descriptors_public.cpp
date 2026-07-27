// SPDX-License-Identifier: Apache-2.0
/**
 * @file   model_descriptors_public.cpp
 * @brief  Public model descriptor self-registration. Zero proprietary-model
 *         literals — proprietary/extension models register themselves in
 *         their own TUs.
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * config_name values are the canonical keys passed to the by-name loader.
 */
#include "model_descriptor.h"

using namespace quick_dot_ai;

#define B(x) (1u << (unsigned)(x)) /* BackendType: CPU=0, GPU=1, NPU=2 */

namespace {
struct RegisterPublicDescriptors {
  RegisterPublicDescriptors() {
    static const ModelDescriptor kPublic[] = {
      {"qwen3-0.6b", "qwen3-0.6b", "Qwen3 0.6B", QDA_RUNTIME_NATIVE, B(0),
       QDA_CAP_STREAMING | QDA_CAP_OPENAI_API | QDA_CAP_TOOL_USE, "QWEN3-0.6B"},
      {"qwen3-1.7b-q40", "qwen3-1.7b", "Qwen3 1.7B (Q40)", QDA_RUNTIME_NATIVE,
       B(0), QDA_CAP_STREAMING | QDA_CAP_OPENAI_API | QDA_CAP_TOOL_USE,
       "QWEN3-1.7B-Q40"},
#if !defined(_WIN32)
      {"tiny-bert", "tiny-bert", "Tiny BERT", QDA_RUNTIME_NATIVE, B(0),
       QDA_CAP_EMBEDDING, "TINY_BERT"},
#endif
      {"function-gemma", "function-gemma", "Function Gemma", QDA_RUNTIME_NATIVE,
       B(0), QDA_CAP_STREAMING | QDA_CAP_OPENAI_API | QDA_CAP_TOOL_USE,
       "FUNCTION_GEMMA"},
      {"gemma4-cpu", "gemma4", "Gemma4 (CPU)", QDA_RUNTIME_NATIVE, B(0),
       QDA_CAP_STREAMING | QDA_CAP_OPENAI_API, "GEMMA4_CPU"},
#ifdef ENABLE_QNN_MODELS
      {"gemma4-e2b-qnn", "gemma4", "Gemma4 E2B (QNN)", QDA_RUNTIME_NATIVE, B(2),
       QDA_CAP_STREAMING | QDA_CAP_OPENAI_API, "GEMMA4-E2B-QNN"},
#if !defined(_WIN32)
      {"vjepa2-qnn", "vjepa", "V-JEPA 2 (QNN)", QDA_RUNTIME_NATIVE, B(2),
       QDA_CAP_VISION_ENCODER, "VJEPA2-QNN"},
#endif
#endif
    };
    for (const auto &d : kPublic)
      register_model_descriptor(&d);
  }
};

const RegisterPublicDescriptors register_public_descriptors;
} // namespace
