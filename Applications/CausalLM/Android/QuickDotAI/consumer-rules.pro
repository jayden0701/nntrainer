# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# QuickDotAI consumer ProGuard rules. These rules are automatically
# applied to any app that depends on the QuickDotAI AAR.

# Keep all JNI entry points — they are called from native code and
# renaming them would break System.loadLibrary + external symbols.
-keepclasseswithmembernames class com.example.quickdotai.NativeCausalLm {
    native <methods>;
}
-keep class com.example.quickdotai.NativeCausalLm$* { *; }
-keep class com.example.quickdotai.NativeCausalLm { *; }

# Keep the public QuickDotAI surface so consumers that expose these DTOs from
# their own serialized or reflected boundary retain stable names.
-keep interface com.example.quickdotai.QuickDotAI { *; }
-keep interface com.example.quickdotai.StreamSink { *; }
-keep class com.example.quickdotai.BackendResult** { *; }
-keep class com.example.quickdotai.LoadModelRequest { *; }
-keep class com.example.quickdotai.OpenAIRequest { *; }
-keep class com.example.quickdotai.OpenAIImageTensor { *; }
-keep class com.example.quickdotai.OpenAIImageTensorSidecar { *; }
-keep class com.example.quickdotai.ImageTensorLayout { *; }
-keep class com.example.quickdotai.LlavaNextImagePreprocessor { *; }
-keep class com.example.quickdotai.PerformanceMetrics { *; }
-keep class com.example.quickdotai.ModelDescriptor { *; }
-keep class com.example.quickdotai.ModelIds { *; }
-keep class com.example.quickdotai.ModelCatalog { *; }
-keepclassmembers class com.example.quickdotai.RuntimeKind { *; }
-keepclassmembers class com.example.quickdotai.Capability { *; }
-keepclassmembers class com.example.quickdotai.BackendType { *; }
-keepclassmembers class com.example.quickdotai.QuantizationType { *; }
-keepclassmembers class com.example.quickdotai.QuickAiError { *; }
