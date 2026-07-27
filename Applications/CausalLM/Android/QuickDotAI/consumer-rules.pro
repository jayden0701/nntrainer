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
-keep class com.example.quickdotai.LlavaNextImagePreprocessor { *; }
-keep class com.example.quickdotai.PerformanceMetrics { *; }
-keep class com.example.quickdotai.ModelDescriptor { *; }
-keep class com.example.quickdotai.ModelIds { *; }
-keep class com.example.quickdotai.ModelCatalog { *; }

# Public enum class and entry names are part of serialized and reflected
# contracts. Keep those names stable without retaining unused enum classes or
# their implementation methods.
-keepnames enum com.example.quickdotai.RuntimeKind
-keepnames enum com.example.quickdotai.Capability
-keepnames enum com.example.quickdotai.BackendType
-keepnames enum com.example.quickdotai.QuantizationType
-keepnames enum com.example.quickdotai.QuickAiError
-keepnames enum com.example.quickdotai.ImageTensorLayout

-keepclassmembers enum com.example.quickdotai.RuntimeKind {
    public static final <fields>;
}
-keepclassmembers enum com.example.quickdotai.Capability {
    public static final <fields>;
}
-keepclassmembers enum com.example.quickdotai.BackendType {
    public static final <fields>;
}
-keepclassmembers enum com.example.quickdotai.QuantizationType {
    public static final <fields>;
}
-keepclassmembers enum com.example.quickdotai.QuickAiError {
    public static final <fields>;
}
-keepclassmembers enum com.example.quickdotai.ImageTensorLayout {
    public static final <fields>;
}
