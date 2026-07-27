// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    QuickDotAI.kt
 * @brief   Public surface of the QuickDotAI AAR.
 */
package com.example.quickdotai

/** Outcome of a QuickDotAI call. */
sealed class BackendResult<out T> {
    data class Ok<T>(val value: T) : BackendResult<T>()
    data class Err(
        val error: QuickAiError,
        val message: String? = null
    ) : BackendResult<Nothing>()
}

/**
 * Receives generated text while a generation call is running.
 *
 * A call emits zero or more [onDelta]/[onReasoningDelta] events followed by
 * exactly one terminal [onDone] or [onError] event. Callbacks are not marshalled
 * to Android's main thread. Native callbacks run on the invoking thread;
 * backend adapters may use their own callback thread.
 *
 * Callbacks are non-reentrant. While generation is active, do not call
 * [QuickDotAI.load], [QuickDotAI.runText], [QuickDotAI.runOpenAI],
 * [QuickDotAI.metrics], [QuickDotAI.unload], or [QuickDotAI.close] on the same
 * engine from a callback. The native backend holds its handle lock during
 * generation. The only supported cross-thread operation is [QuickDotAI.cancel].
 * Apply the same rule to terminal callbacks even though the Kotlin adapter
 * emits them immediately before the blocking run call returns.
 */
interface StreamSink {
    fun onDelta(text: String)
    fun onReasoningDelta(text: String) {
    }
    fun onDone()
    fun onError(error: QuickAiError, message: String?)
}

/**
 * One loaded on-device model.
 *
 * Lifecycle is [load], generation calls, then [unload] or [close]. Generation
 * is deliberately expressed as only two concepts:
 *
 * - [runText] sends exact text without a chat template or implicit history.
 * - [runOpenAI] sends an OpenAI-compatible JSON object unchanged, with an
 *   optional versioned sidecar for preprocessed image tensors.
 *
 * Both calls stream while synchronously blocking the invoking worker thread.
 * Do not invoke them from Android's main thread. Instances are not generally
 * thread-safe; [cancel] is the only operation intended for cross-thread use.
 * Sink callbacks must not re-enter this engine; see [StreamSink].
 */
interface QuickDotAI {
    /** A short engine identifier such as `native` or `litert-lm`. */
    val kind: String

    /** Identifier of the currently loaded model, or null before load/after unload. */
    val modelId: String?

    /** Load one model. */
    fun load(req: LoadModelRequest): BackendResult<Unit>

    /**
     * Generate from [text] exactly as supplied.
     *
     * No chat template, role marker, or previous-turn KV state may be added by
     * this API. A backend that cannot provide this guarantee returns
     * [QuickAiError.UNSUPPORTED].
     */
    fun runText(text: String, sink: StreamSink): BackendResult<Unit>

    /**
     * Generate from an OpenAI-compatible request.
     *
     * [OpenAIRequest.json] is forwarded unchanged. Native image requests use
     * tensor sidecars correlated by an exact `image_url.url` source match;
     * LiteRT-LM resolves supported sidecar-less data/file image URLs directly.
     */
    fun runOpenAI(
        request: OpenAIRequest,
        sink: StreamSink
    ): BackendResult<Unit>

    /** Unload model weights while keeping the wrapper object closable. */
    fun unload(): BackendResult<Unit>

    /** Metrics for the most recent completed run. */
    fun metrics(): BackendResult<PerformanceMetrics>

    /**
     * Encode text with an embedding model. Generative models return an error.
     * This remains separate from the two generation concepts.
     */
    fun encode(text: String): BackendResult<FloatArray> =
        BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "encode() is not supported by this engine"
        )

    /** Request cancellation of an in-flight generation. */
    fun cancel() {
    }

    /** Release all resources. Idempotent. */
    fun close()
}

/**
 * Create an engine bound to [descriptor].
 *
 * The subsequent [QuickDotAI.load] request must name this descriptor (or its
 * declared speculative-decoding variant) and use one of its backends.
 */
fun createEngine(
    descriptor: ModelDescriptor
): QuickDotAI =
    when (descriptor.runtime) {
        RuntimeKind.LITERT -> LiteRTLm(descriptor)
        RuntimeKind.NATIVE -> NativeQuickDotAI(descriptor)
    }

/** Validate the catalog identity and backend captured by [createEngine]. */
internal fun ModelDescriptor.validateLoadRequest(
    request: LoadModelRequest
): BackendResult<Unit> {
    val hasSpeculativeCapability = Capability.SPECULATIVE in capabilities
    val hasSpeculativeVariant = !sdVariantId.isNullOrBlank()
    if (hasSpeculativeCapability != hasSpeculativeVariant ||
        (hasSpeculativeVariant && sdVariantId == id)
    ) {
        return BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "Model '$id' has an inconsistent speculative-decoding declaration"
        )
    }
    if (request.useSpeculativeDecoding && !hasSpeculativeCapability) {
        return BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "Model '$id' does not support speculative decoding"
        )
    }
    val expectedModelId =
        if (request.useSpeculativeDecoding) sdVariantId.orEmpty() else id
    if (request.modelId != expectedModelId) {
        return BackendResult.Err(
            QuickAiError.INVALID_PARAMETER,
            "Engine for '$id' cannot load modelId '${request.modelId}'; " +
                "expected '$expectedModelId'"
        )
    }
    if (request.backend !in backends) {
        return BackendResult.Err(
            QuickAiError.INVALID_PARAMETER,
            "Backend ${request.backend} is not supported by model '$id'"
        )
    }
    return BackendResult.Ok(Unit)
}
