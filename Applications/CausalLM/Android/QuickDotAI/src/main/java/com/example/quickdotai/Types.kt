// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    Types.kt
 * @brief   Value types shared by the QuickDotAI interface and its
 *          implementations.
 *
 * The enums mirror the C enums in Applications/CausalLM/api/quick_dot_ai_api.h.
 * Model identifiers are plain Strings (see [ModelIds] in ModelCatalog.kt)
 * so the AAR is not re-compiled whenever the model list changes.
 *
 * Stable wire-facing configuration and metric types carry `@Serializable`.
 * Runtime-only types such as image tensors deliberately do not: callers
 * should keep large float buffers out of JSON.
 */
package com.example.quickdotai

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull

/** Compute backend. Mirrors BackendType in quick_dot_ai_api.h. */
@Serializable
enum class BackendType {
    CPU,
    GPU,
    NPU
}

/** Quantization type. Mirrors ModelQuantizationType in quick_dot_ai_api.h. */
@Serializable
enum class QuantizationType {
    UNKNOWN,
    W4A32,
    W16A16,
    W8A16,
    W32A32
}

/** Error code. Mirrors ErrorCode in quick_dot_ai_api.h. */
@Serializable
enum class QuickAiError(val code: Int) {
    NONE(0),
    INVALID_PARAMETER(1),
    MODEL_LOAD_FAILED(2),
    INFERENCE_FAILED(3),
    NOT_INITIALIZED(4),
    INFERENCE_NOT_RUN(5),
    UNSUPPORTED(6),
    UNKNOWN(99);

    companion object {
        fun fromNativeCode(code: Int): QuickAiError =
            entries.firstOrNull { it.code == code } ?: UNKNOWN
    }
}

/**
 * Memory layout of one image tensor supplied alongside an OpenAI request.
 *
 * [HWC] and [CHW] describe conventional dense image patches. Use
 * [MODEL_NATIVE] only when the tensor has already been transformed into the
 * exact model-specific representation expected by the selected model.
 */
enum class ImageTensorLayout(val nativeValue: Int) {
    MODEL_NATIVE(0),
    HWC(1),
    CHW(2)
}

/**
 * A preprocessed image tensor referenced by an OpenAI `image_url` content
 * part. [source] must exactly match that part's `image_url.url` value at the
 * same occurrence index in [OpenAIImageTensorSidecar.tensors].
 *
 * The number of float values is [pixelValues.size]; it is intentionally not a
 * second caller-controlled field. For [ImageTensorLayout.HWC] and
 * [ImageTensorLayout.CHW], the expected value count is
 * `patchCount * channels * patchHeight * patchWidth`. A model-native tensor
 * still carries the metadata for routing and diagnostics, but its model-
 * specific value count is validated by the native model implementation.
 */
data class OpenAIImageTensor(
    val source: String,
    val pixelValues: FloatArray,
    val layout: ImageTensorLayout,
    val patchCount: Int,
    val channels: Int,
    val patchHeight: Int,
    val patchWidth: Int,
    val originalHeight: Int,
    val originalWidth: Int
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is OpenAIImageTensor) return false
        return source == other.source &&
            pixelValues.contentEquals(other.pixelValues) &&
            layout == other.layout &&
            patchCount == other.patchCount &&
            channels == other.channels &&
            patchHeight == other.patchHeight &&
            patchWidth == other.patchWidth &&
            originalHeight == other.originalHeight &&
            originalWidth == other.originalWidth
    }

    override fun hashCode(): Int {
        var result = source.hashCode()
        result = 31 * result + pixelValues.contentHashCode()
        result = 31 * result + layout.hashCode()
        result = 31 * result + patchCount
        result = 31 * result + channels
        result = 31 * result + patchHeight
        result = 31 * result + patchWidth
        result = 31 * result + originalHeight
        result = 31 * result + originalWidth
        return result
    }
}

/**
 * Versioned collection of preprocessed image tensors carried out-of-band from
 * an OpenAI JSON request. Tensors follow `image_url` occurrence order; repeated
 * URLs require repeated entries. Versioning lets the tensor contract evolve
 * without changing the JSON wire format.
 */
data class OpenAIImageTensorSidecar(
    val version: Int = CURRENT_VERSION,
    val tensors: List<OpenAIImageTensor>
) {
    companion object {
        const val CURRENT_VERSION: Int = 1
    }
}

/**
 * OpenAI-compatible JSON request plus optional preprocessed image tensors.
 *
 * [json] is forwarded unchanged to the selected OpenAI request path. The AAR
 * inspects it for structural validation and, when a sidecar is present, to
 * ensure [OpenAIImageTensor.source] values refer to real `image_url.url`
 * entries. Native image requests require a sidecar; LiteRT-LM instead resolves
 * supported data/file image URLs directly.
 */
data class OpenAIRequest(
    val json: String,
    val imageTensors: OpenAIImageTensorSidecar? = null
) {
    /**
     * Validate JSON shape and, when present, tensor metadata and source links.
     *
     * An image tensor sidecar is backend-specific. A sidecar-less `image_url`
     * is valid here because LiteRT-LM can consume supported URL forms directly.
     */
    fun validate(): BackendResult<Unit> =
        validationToUnit(validateAndCollectImageSources(requireTensorSidecarForImages = false))

    /** Native inference requires one tensor for every `image_url` occurrence. */
    internal fun validateForNative(): BackendResult<Unit> =
        validationToUnit(validateAndCollectImageSources(requireTensorSidecarForImages = true))

    /**
     * Validate only the JSON request shape and collect image URL occurrences.
     *
     * Tensor sidecars are intentionally not inspected here, so adapters can
     * reject unsupported image capabilities without scanning tensor payloads.
     */
    internal fun structuralImageUrlSources(): BackendResult<List<String>> {
        val root = try {
            Json.parseToJsonElement(json)
        } catch (exception: Exception) {
            return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "OpenAI request is not valid JSON: ${exception.message}"
            )
        }

        val requestObject = root as? JsonObject
            ?: return invalidRequest("OpenAI request must be a JSON object")
        val messages = requestObject["messages"]
        if (messages !is JsonArray) {
            return invalidRequest("OpenAI request must contain a messages array")
        }
        if (messages.isEmpty()) {
            return invalidRequest("OpenAI request messages must not be empty")
        }
        return collectImageUrlSources(messages)
    }

    private fun validateAndCollectImageSources(
        requireTensorSidecarForImages: Boolean
    ): BackendResult<List<String>> {
        val referencedSources = when (val result = structuralImageUrlSources()) {
            is BackendResult.Ok -> result.value
            is BackendResult.Err -> return result
        }
        val sidecar = imageTensors
        if (sidecar == null) {
            return if (requireTensorSidecarForImages && referencedSources.isNotEmpty()) {
                invalidRequest(
                    "Native OpenAI image_url content requires a matching image tensor sidecar"
                )
            } else {
                BackendResult.Ok(referencedSources)
            }
        }
        if (sidecar.version != OpenAIImageTensorSidecar.CURRENT_VERSION) {
            return invalidRequest(
                "Unsupported image tensor sidecar version ${sidecar.version}"
            )
        }
        if (sidecar.tensors.size != referencedSources.size) {
            return invalidRequest(
                "OpenAI request has ${referencedSources.size} image_url occurrences " +
                    "but the sidecar has ${sidecar.tensors.size} tensors"
            )
        }
        for ((index, tensor) in sidecar.tensors.withIndex()) {
            val prefix = "image tensor[$index]"
            if (tensor.source.isBlank()) {
                return invalidRequest("$prefix source must not be blank")
            }
            val expectedSource = referencedSources[index]
            if (tensor.source != expectedSource) {
                return invalidRequest(
                    "$prefix source '${tensor.source}' does not match image_url[$index] " +
                        "source '$expectedSource'"
                )
            }
            if (tensor.pixelValues.isEmpty()) {
                return invalidRequest("$prefix pixelValues must not be empty")
            }
            if (tensor.pixelValues.any { !it.isFinite() }) {
                return invalidRequest("$prefix pixelValues must contain only finite values")
            }
            if (tensor.patchCount <= 0 || tensor.channels <= 0 ||
                tensor.patchHeight <= 0 || tensor.patchWidth <= 0 ||
                tensor.originalHeight <= 0 || tensor.originalWidth <= 0
            ) {
                return invalidRequest("$prefix dimensions and patch count must be positive")
            }

            if (tensor.layout != ImageTensorLayout.MODEL_NATIVE) {
                var expected = 1L
                for (dimension in intArrayOf(
                    tensor.patchCount,
                    tensor.channels,
                    tensor.patchHeight,
                    tensor.patchWidth
                )) {
                    if (expected > Int.MAX_VALUE.toLong() / dimension) {
                        return invalidRequest(
                            "$prefix dense tensor value count exceeds the supported range"
                        )
                    }
                    expected *= dimension
                }
                if (expected.toInt() != tensor.pixelValues.size) {
                    return invalidRequest(
                        "$prefix has ${tensor.pixelValues.size} values; expected $expected"
                    )
                }
            }
        }
        return BackendResult.Ok(referencedSources)
    }

    private fun validationToUnit(
        result: BackendResult<List<String>>
    ): BackendResult<Unit> = when (result) {
        is BackendResult.Ok -> BackendResult.Ok(Unit)
        is BackendResult.Err -> result
    }

    private fun collectImageUrlSources(
        messages: JsonArray
    ): BackendResult<List<String>> {
        val result = mutableListOf<String>()
        for ((messageIndex, messageElement) in messages.withIndex()) {
            val message = messageElement as? JsonObject
                ?: return invalidRequest("messages[$messageIndex] must be an object")
            val role = (message["role"] as? JsonPrimitive)
                ?.takeIf { it.isString }
                ?.contentOrNull
                ?.takeIf { it in SUPPORTED_MESSAGE_ROLES }
                ?: return invalidRequest(
                    "messages[$messageIndex].role must be a supported string role"
                )
            val contentElement = message["content"]
            if (contentElement == null || contentElement is JsonNull) {
                if (role == "assistant" &&
                    (message.containsKey("tool_calls") || message.containsKey("function_call"))
                ) {
                    continue
                }
                return invalidRequest("messages[$messageIndex].content is required")
            }
            if (contentElement is JsonPrimitive) {
                if (!contentElement.isString) {
                    return invalidRequest(
                        "messages[$messageIndex].content must be a string or content array"
                    )
                }
                continue
            }
            val content = contentElement as? JsonArray
                ?: return invalidRequest(
                    "messages[$messageIndex].content must be a string or content array"
                )
            if (content.isEmpty()) {
                return invalidRequest("messages[$messageIndex].content must not be empty")
            }
            for ((partIndex, partElement) in content.withIndex()) {
                val part = partElement as? JsonObject
                    ?: return invalidRequest(
                        "messages[$messageIndex].content[$partIndex] must be an object"
                    )
                val typePrimitive = part["type"] as? JsonPrimitive
                val type = typePrimitive?.takeIf { it.isString }?.contentOrNull
                    ?: return invalidRequest(
                        "messages[$messageIndex].content[$partIndex].type must be a string"
                    )
                if (type == "text" || type == "input_text") {
                    val text = part["text"] as? JsonPrimitive
                    if (text?.isString != true) {
                        return invalidRequest(
                            "messages[$messageIndex].content[$partIndex].text must be a string"
                        )
                    }
                    continue
                }
                if (type != "image_url") {
                    continue
                }
                if (role != "user") {
                    return invalidRequest(
                        "messages[$messageIndex].content[$partIndex].image_url requires role user"
                    )
                }
                val imageUrl = part["image_url"]
                val source = when (imageUrl) {
                    is JsonObject -> {
                        val detail = (imageUrl["detail"] as? JsonPrimitive)
                            ?.takeIf { it.isString }
                            ?.contentOrNull
                        if (imageUrl.containsKey("detail") &&
                            detail !in SUPPORTED_IMAGE_DETAILS
                        ) {
                            return invalidRequest(
                                "messages[$messageIndex].content[$partIndex]." +
                                    "image_url.detail must be auto, low, or high"
                            )
                        }
                        (imageUrl["url"] as? JsonPrimitive)
                            ?.takeIf { it.isString }
                            ?.contentOrNull
                    }
                    is JsonPrimitive -> imageUrl.takeIf { it.isString }?.contentOrNull
                    else -> null
                }
                if (source.isNullOrBlank()) {
                    return invalidRequest(
                        "messages[$messageIndex].content[$partIndex].image_url " +
                            "must contain a non-blank string URL"
                    )
                }
                result.add(source)
            }
        }
        return BackendResult.Ok(result)
    }

    private fun <T> invalidRequest(message: String): BackendResult<T> =
        BackendResult.Err(QuickAiError.INVALID_PARAMETER, message)

    private companion object {
        val SUPPORTED_MESSAGE_ROLES = setOf(
            "system",
            "developer",
            "user",
            "assistant",
            "tool",
            "function"
        )
        val SUPPORTED_IMAGE_DETAILS = setOf("auto", "low", "high")
    }
}

/**
 * Descriptor passed to [QuickDotAI.load].
 *
 * [modelPath], [visionBackend], [cacheDir], and [maxNumTokens] configure the
 * LiteRT-LM engine. [nativeLibDir] and [modelBasePath] configure the native
 * engine. [quantization] is native-only. Backend-specific fields that do not
 * apply to the selected engine are ignored as documented below.
 */
@Serializable
data class LoadModelRequest(
    val backend: BackendType = BackendType.GPU,
    @SerialName("model_id") val modelId: String,
    val quantization: QuantizationType = QuantizationType.W4A32,
    @SerialName("model_path") val modelPath: String? = null,

    /**
     * Compute backend for the model's vision encoder when loading a
     * multimodal-capable model. Null loads the engine in text-only mode.
     *
     * Only honored by [LiteRTLm]; [NativeQuickDotAI] ignores it.
     */
    @SerialName("vision_backend") val visionBackend: BackendType? = null,

    /**
     * Writable directory for engine on-disk caches. Null uses the engine
     * default.
     *
     * Only honored by [LiteRTLm]; [NativeQuickDotAI] ignores it.
     */
    @SerialName("cache_dir") val cacheDir: String? = null,

    /**
     * Maximum number of tokens allocated for the KV cache/context window.
     * Null uses the engine default.
     *
     * Only honored by [LiteRTLm]; [NativeQuickDotAI] ignores it.
     */
    @SerialName("max_num_tokens") val maxNumTokens: Int? = null,

    /**
     * Native library directory from ApplicationInfo.nativeLibraryDir.
     *
     * Only honored by [NativeQuickDotAI]; [LiteRTLm] ignores it.
     */
    @SerialName("native_lib_dir") val nativeLibDir: String? = null,

    /**
     * Base directory for native model files. Null lets the C API use its
     * configured fallback path.
     *
     * Only honored by [NativeQuickDotAI]; [LiteRTLm] ignores it.
     */
    @SerialName("model_base_path") val modelBasePath: String? = null,

    /**
     * Optional QNN HTP backend-extension config path retained for compatibility
     * with the current Android build. The by-name native loader derives its
     * active config from [modelBasePath] and currently only diagnoses this
     * field rather than forwarding it.
     */
    @SerialName("htp_backend_config_path")
    val htpBackendConfigPath: String? = null,

    /**
     * Enable the native model's speculative-decoding path. LiteRT-LM returns
     * [QuickAiError.UNSUPPORTED] when this is true. [modelId] must name the
     * descriptor's declared speculative variant.
     */
    @SerialName("use_speculative_decoding")
    val useSpeculativeDecoding: Boolean = false,
) {
    /** Canonical key shared across the stack for one loaded model handle. */
    val modelKey: String
        get() = "$modelId:${quantization.name}:sd=$useSpeculativeDecoding"
}

/**
 * Performance metrics for the most recent run.
 *
 * Not every engine fills every field:
 *
 * - [NativeQuickDotAI] fills prefill/generation counters and peak memory from
 *   the C API.
 * - [LiteRTLm] currently fills initialization and total duration only.
 */
@Serializable
data class PerformanceMetrics(
    @SerialName("prefill_tokens") val prefillTokens: Int = 0,
    @SerialName("prefill_duration_ms") val prefillDurationMs: Double = 0.0,
    @SerialName("generation_tokens") val generationTokens: Int = 0,
    @SerialName("generation_duration_ms") val generationDurationMs: Double = 0.0,
    @SerialName("total_duration_ms") val totalDurationMs: Double = 0.0,
    @SerialName("initialization_duration_ms") val initializationDurationMs: Double = 0.0,
    @SerialName("peak_memory_kb") val peakMemoryKb: Long = 0
)
