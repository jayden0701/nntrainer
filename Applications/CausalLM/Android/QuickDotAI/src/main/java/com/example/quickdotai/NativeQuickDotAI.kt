// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    NativeQuickDotAI.kt
 * @brief   Minimal QuickDotAI adapter for the native C API.
 */
package com.example.quickdotai

import android.util.Log
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/** One native CausalLmHandle exposed through the two QuickDotAI run modes. */
internal class NativeQuickDotAI(
    private val descriptor: ModelDescriptor
) : QuickDotAI {
    override val kind: String = "native"

    override var modelId: String? = null
        private set

    @Volatile
    private var handle: Long = 0L
    private var loadedRequest: LoadModelRequest? = null
    private val requestStateLock = Any()
    private val requestInFlight = AtomicBoolean(false)
    private val nativeInvocationActive = AtomicBoolean(false)
    private val cancellationEpoch = AtomicLong(0L)
    private var closePending = false

    override fun load(req: LoadModelRequest): BackendResult<Unit> {
        when (val validation = descriptor.validateLoadRequest(req)) {
            is BackendResult.Ok -> Unit
            is BackendResult.Err -> return validation
        }
        if (handle != 0L) {
            return if (loadedRequest == req) {
                BackendResult.Ok(Unit)
            } else {
                BackendResult.Err(
                    QuickAiError.INVALID_PARAMETER,
                    "A different model configuration is already loaded; unload it first"
                )
            }
        }
        if (req.modelId.isBlank()) {
            return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "modelId must not be blank"
            )
        }
        if (!req.htpBackendConfigPath.isNullOrBlank()) {
            Log.w(
                TAG,
                "htpBackendConfigPath is not forwarded separately by the catalog load path; " +
                    "the native loader resolves QNN configuration from modelBasePath"
            )
        }
        val modelBasePath = req.modelBasePath?.takeIf { it.isNotBlank() }
        if (!NativeCausalLm.ensureLoaded()) {
            return BackendResult.Err(
                QuickAiError.MODEL_LOAD_FAILED,
                "libquickai_jni.so or one of its packaged native dependencies is unavailable"
            )
        }

        return try {
            val loadResult = NativeCausalLm.loadModelHandleByNameNative(
                backend = mapBackend(req.backend),
                modelId = req.modelId,
                quant = mapQuant(req.quantization),
                nativeLibDir = req.nativeLibDir,
                modelBasePath = modelBasePath
            )
            if (loadResult.errorCode != 0 || loadResult.handle == 0L) {
                val error = QuickAiError.fromNativeCode(loadResult.errorCode)
                    .takeUnless { it == QuickAiError.NONE }
                    ?: QuickAiError.MODEL_LOAD_FAILED
                return BackendResult.Err(
                    error,
                    "Unable to load native model '${req.modelId}'"
                )
            }

            val speculativeError = if (req.useSpeculativeDecoding) {
                NativeCausalLm.configureSpeculativeDecodingNative(
                    loadResult.handle,
                    true
                )
            } else {
                0
            }
            if (speculativeError != 0) {
                NativeCausalLm.destroyModelHandleNative(loadResult.handle)
                val error = QuickAiError.fromNativeCode(speculativeError)
                return BackendResult.Err(
                    error,
                    "Unable to configure speculative decoding for '${req.modelId}'"
                )
            }

            handle = loadResult.handle
            loadedRequest = req
            modelId = req.modelId
            BackendResult.Ok(Unit)
        } catch (t: Throwable) {
            Log.e(TAG, "Native model load failed", t)
            BackendResult.Err(
                QuickAiError.MODEL_LOAD_FAILED,
                t.message ?: "Native model load failed"
            )
        }
    }

    override fun runText(text: String, sink: StreamSink): BackendResult<Unit> {
        val runEpoch = beginRequest("runText", sink) ?: return requestInFlightError()
        try {
            if (Capability.VISION_ENCODER in descriptor.capabilities) {
                return failRun(
                    sink,
                    QuickAiError.UNSUPPORTED,
                    "Standalone vision encoder '${descriptor.id}' cannot generate text"
                )
            }
            if (Capability.STREAMING !in descriptor.capabilities) {
                return failRun(
                    sink,
                    QuickAiError.UNSUPPORTED,
                    "Model '${descriptor.id}' does not support text generation"
                )
            }
            if (text.isEmpty()) {
                return failRun(
                    sink,
                    QuickAiError.INVALID_PARAMETER,
                    "runText input must not be empty"
                )
            }
            completeCancelledBeforeNative(runEpoch, sink)?.let { return it }
            return runNativeGeneration("runText", sink, runEpoch) { callback ->
                NativeCausalLm.runTextStreamingNative(handle, text, callback)
            }
        } finally {
            endRequest()
        }
    }

    override fun runOpenAI(
        request: OpenAIRequest,
        sink: StreamSink
    ): BackendResult<Unit> {
        val runEpoch = beginRequest("runOpenAI", sink) ?: return requestInFlightError()
        try {
            if (Capability.OPENAI_API !in descriptor.capabilities) {
                return failRun(
                    sink,
                    QuickAiError.UNSUPPORTED,
                    "Model '${descriptor.id}' does not support the OpenAI request API"
                )
            }
            if (Capability.VISION_ENCODER in descriptor.capabilities) {
                return failRun(
                    sink,
                    QuickAiError.UNSUPPORTED,
                    "Standalone vision encoder '${descriptor.id}' cannot run generation"
                )
            }
            val imageSources = when (val result = request.structuralImageUrlSources()) {
                is BackendResult.Ok -> result.value
                is BackendResult.Err -> {
                    emitError(sink, result.error, result.message)
                    return result
                }
            }
            if (imageSources.isNotEmpty() &&
                Capability.MULTIMODAL !in descriptor.capabilities
            ) {
                return failRun(
                    sink,
                    QuickAiError.UNSUPPORTED,
                    "Model '${descriptor.id}' does not support image sidecars"
                )
            }
            if (imageSources.size > 1 &&
                Capability.MULTI_IMAGE !in descriptor.capabilities
            ) {
                return failRun(
                    sink,
                    QuickAiError.UNSUPPORTED,
                    "Model '${descriptor.id}' does not support multiple images"
                )
            }
            when (val validation = request.validateForNative()) {
                is BackendResult.Ok -> Unit
                is BackendResult.Err -> {
                    emitError(sink, validation.error, validation.message)
                    return validation
                }
            }

            val tensors = request.imageTensors?.tensors.orEmpty()
            completeCancelledBeforeNative(runEpoch, sink)?.let { return it }
            return runNativeGeneration("runOpenAI", sink, runEpoch) { callback ->
                NativeCausalLm.runOpenAIStreamingNative(
                    handle = handle,
                    jsonRequest = request.json,
                    imageSources = Array(tensors.size) { tensors[it].source },
                    pixelValues = Array(tensors.size) { tensors[it].pixelValues },
                    layouts = IntArray(tensors.size) { tensors[it].layout.nativeValue },
                    patchCounts = IntArray(tensors.size) { tensors[it].patchCount },
                    channels = IntArray(tensors.size) { tensors[it].channels },
                    patchHeights = IntArray(tensors.size) { tensors[it].patchHeight },
                    patchWidths = IntArray(tensors.size) { tensors[it].patchWidth },
                    originalHeights = IntArray(tensors.size) { tensors[it].originalHeight },
                    originalWidths = IntArray(tensors.size) { tensors[it].originalWidth },
                    callback = callback
                )
            }
        } finally {
            endRequest()
        }
    }

    private fun beginRequest(operation: String, sink: StreamSink): Long? {
        val epoch = synchronized(requestStateLock) {
            if (!requestInFlight.compareAndSet(false, true)) {
                null
            } else {
                cancellationEpoch.get()
            }
        }
        if (epoch == null) {
            emitError(
                sink,
                QuickAiError.INVALID_PARAMETER,
                "$operation cannot run while another request is active"
            )
        }
        return epoch
    }

    private fun endRequest() {
        val handleToDestroy = synchronized(requestStateLock) {
            nativeInvocationActive.set(false)
            requestInFlight.set(false)
            if (closePending) {
                closePending = false
                val loadedHandle = handle
                clearLoadedState()
                loadedHandle
            } else {
                0L
            }
        }
        destroyDetachedHandle(handleToDestroy)
    }

    private fun requestInFlightError(): BackendResult.Err =
        BackendResult.Err(
            QuickAiError.INVALID_PARAMETER,
            "Another native request is already active"
        )

    private fun completeCancelledBeforeNative(
        runEpoch: Long,
        sink: StreamSink
    ): BackendResult<Unit>? {
        if (cancellationEpoch.get() == runEpoch) return null
        return try {
            sink.onDone()
            BackendResult.Ok(Unit)
        } catch (t: Throwable) {
            BackendResult.Err(
                QuickAiError.INFERENCE_FAILED,
                "Cancellation terminal listener failed: ${t.message}"
            )
        }
    }

    private fun failRun(
        sink: StreamSink,
        error: QuickAiError,
        message: String
    ): BackendResult.Err {
        emitError(sink, error, message)
        return BackendResult.Err(error, message)
    }

    private fun runNativeGeneration(
        operation: String,
        sink: StreamSink,
        runEpoch: Long,
        invocation: (((String) -> Int) -> Int)
    ): BackendResult<Unit> {
        if (handle == 0L) {
            val message = "NativeQuickDotAI has not been loaded yet"
            emitError(sink, QuickAiError.NOT_INITIALIZED, message)
            return BackendResult.Err(QuickAiError.NOT_INITIALIZED, message)
        }

        var armErrorCode = 0
        var armFailure: Throwable? = null
        val cancelledBeforeInvocation = synchronized(requestStateLock) {
            if (cancellationEpoch.get() != runEpoch) {
                true
            } else {
                try {
                    armErrorCode = NativeCausalLm.armRunCancellationNative(handle)
                    if (armErrorCode == 0) {
                        nativeInvocationActive.set(true)
                    }
                } catch (t: Throwable) {
                    armFailure = t
                }
                false
            }
        }
        if (cancelledBeforeInvocation) {
            return completeCancelledBeforeNative(runEpoch, sink)
                ?: BackendResult.Ok(Unit)
        }
        armFailure?.let { failure ->
            val message = "$operation could not arm native cancellation: ${failure.message}"
            Log.e(TAG, message, failure)
            emitError(sink, QuickAiError.INFERENCE_FAILED, message)
            return BackendResult.Err(QuickAiError.INFERENCE_FAILED, message)
        }
        if (armErrorCode != 0) {
            val error = QuickAiError.fromNativeCode(armErrorCode)
            val message = "$operation could not start (errorCode=$armErrorCode)"
            emitError(sink, error, message)
            return BackendResult.Err(error, message)
        }

        var callbackFailure: Throwable? = null
        return try {
            val errorCode = try {
                invocation { delta ->
                    if (callbackFailure != null) {
                        1
                    } else {
                        try {
                            sink.onDelta(delta)
                            0
                        } catch (t: Throwable) {
                            callbackFailure = t
                            1
                        }
                    }
                }
            } finally {
                synchronized(requestStateLock) {
                    try {
                        NativeCausalLm.disarmRunCancellationNative(handle)
                    } catch (t: Throwable) {
                        Log.w(TAG, "Unable to disarm native cancellation", t)
                    } finally {
                        nativeInvocationActive.set(false)
                    }
                }
            }

            val listenerFailure = callbackFailure
            if (listenerFailure != null) {
                val message = "$operation listener failed: ${listenerFailure.message}"
                emitError(sink, QuickAiError.INFERENCE_FAILED, message)
                BackendResult.Err(QuickAiError.INFERENCE_FAILED, message)
            } else if (errorCode != 0) {
                val error = QuickAiError.fromNativeCode(errorCode)
                val message = "$operation failed (errorCode=$errorCode)"
                emitError(sink, error, message)
                BackendResult.Err(error, message)
            } else {
                try {
                    sink.onDone()
                    BackendResult.Ok(Unit)
                } catch (t: Throwable) {
                    BackendResult.Err(
                        QuickAiError.INFERENCE_FAILED,
                        "$operation terminal listener failed: ${t.message}"
                    )
                }
            }
        } catch (t: Throwable) {
            Log.e(TAG, "$operation threw", t)
            emitError(sink, QuickAiError.INFERENCE_FAILED, t.message)
            BackendResult.Err(QuickAiError.INFERENCE_FAILED, t.message)
        }
    }

    override fun metrics(): BackendResult<PerformanceMetrics> {
        if (handle == 0L) {
            return BackendResult.Err(QuickAiError.NOT_INITIALIZED)
        }
        return try {
            val metrics = NativeCausalLm.getPerformanceMetricsHandleNative(handle)
            if (metrics.errorCode != 0) {
                BackendResult.Err(QuickAiError.fromNativeCode(metrics.errorCode))
            } else {
                BackendResult.Ok(
                    PerformanceMetrics(
                        prefillTokens = metrics.prefillTokens,
                        prefillDurationMs = metrics.prefillDurationMs,
                        generationTokens = metrics.generationTokens,
                        generationDurationMs = metrics.generationDurationMs,
                        totalDurationMs = metrics.totalDurationMs,
                        initializationDurationMs = metrics.initializationDurationMs,
                        peakMemoryKb = metrics.peakMemoryKb
                    )
                )
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Native metrics call failed", t)
            BackendResult.Err(QuickAiError.UNKNOWN, t.message)
        }
    }

    override fun encode(text: String): BackendResult<FloatArray> {
        if (handle == 0L) {
            return BackendResult.Err(QuickAiError.NOT_INITIALIZED)
        }
        if (Capability.EMBEDDING !in descriptor.capabilities) {
            return BackendResult.Err(
                QuickAiError.UNSUPPORTED,
                "Model '${descriptor.id}' is not an embedding model"
            )
        }
        return try {
            val embedding = NativeCausalLm.encodeModelHandleNative(handle, text)
            if (embedding == null || embedding.isEmpty()) {
                BackendResult.Err(
                    QuickAiError.INFERENCE_FAILED,
                    "The loaded model did not return an embedding"
                )
            } else {
                BackendResult.Ok(embedding)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Native encode call failed", t)
            BackendResult.Err(QuickAiError.INFERENCE_FAILED, t.message)
        }
    }

    override fun unload(): BackendResult<Unit> {
        return synchronized(requestStateLock) {
            if (requestInFlight.get()) {
                return@synchronized BackendResult.Err(
                    QuickAiError.INVALID_PARAMETER,
                    "Cannot unload while a generation request is active"
                )
            }
            val loadedHandle = handle
            if (loadedHandle == 0L) {
                clearLoadedState()
                return@synchronized BackendResult.Ok(Unit)
            }
            try {
                val errorCode = NativeCausalLm.destroyModelHandleNative(loadedHandle)
                if (errorCode != 0) {
                    BackendResult.Err(QuickAiError.fromNativeCode(errorCode))
                } else {
                    clearLoadedState()
                    BackendResult.Ok(Unit)
                }
            } catch (t: Throwable) {
                Log.w(TAG, "Native handle destruction failed", t)
                BackendResult.Err(QuickAiError.UNKNOWN, t.message)
            }
        }
    }

    override fun cancel() {
        synchronized(requestStateLock) {
            if (!requestInFlight.get()) {
                return
            }
            cancellationEpoch.incrementAndGet()
            requestNativeCancellationLocked()
        }
    }

    override fun close() {
        val handleToDestroy = synchronized(requestStateLock) {
            if (requestInFlight.get()) {
                closePending = true
                cancellationEpoch.incrementAndGet()
                requestNativeCancellationLocked()
                0L
            } else {
                val loadedHandle = handle
                clearLoadedState()
                loadedHandle
            }
        }
        destroyDetachedHandle(handleToDestroy)
    }

    /** requestStateLock must be held while this method executes. */
    private fun requestNativeCancellationLocked() {
        if (!nativeInvocationActive.get()) return
        val loadedHandle = handle
        if (loadedHandle == 0L) return
        try {
            NativeCausalLm.cancelModelHandleNative(loadedHandle)
        } catch (t: Throwable) {
            Log.w(TAG, "Native cancellation failed", t)
        }
    }

    private fun destroyDetachedHandle(loadedHandle: Long) {
        if (loadedHandle == 0L) return
        try {
            NativeCausalLm.destroyModelHandleNative(loadedHandle)
        } catch (t: Throwable) {
            Log.w(TAG, "Native handle destruction failed", t)
        }
    }

    private fun clearLoadedState() {
        handle = 0L
        loadedRequest = null
        modelId = null
    }

    private fun emitError(
        sink: StreamSink,
        error: QuickAiError,
        message: String?
    ) {
        try {
            sink.onError(error, message)
        } catch (t: Throwable) {
            Log.e(TAG, "StreamSink.onError threw", t)
        }
    }

    private fun mapBackend(backend: BackendType): Int = when (backend) {
        BackendType.CPU -> 0
        BackendType.GPU -> 1
        BackendType.NPU -> 2
    }

    private fun mapQuant(quantization: QuantizationType): Int = when (quantization) {
        QuantizationType.UNKNOWN -> 0
        QuantizationType.W4A32 -> 1
        QuantizationType.W16A16 -> 2
        QuantizationType.W8A16 -> 3
        QuantizationType.W32A32 -> 4
    }

    companion object {
        private const val TAG = "NativeQuickDotAI"
    }
}
