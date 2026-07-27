// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    NativeCausalLm.kt
 * @brief   Minimal JNI bridge used by the QuickDotAI native adapter.
 */
package com.example.quickdotai

import android.util.Log

/** JNI implementation detail; applications should use [QuickDotAI]. */
internal object NativeCausalLm {
    @Volatile
    private var loaded: Boolean = false

    @Synchronized
    fun ensureLoaded(): Boolean {
        if (loaded) return true
        return try {
            // QNN acceleration is optional. CPU-only packages do not ship it.
            try {
                System.loadLibrary("qnn_context")
                Log.i(TAG, "Loaded optional QNN backend")
            } catch (t: LinkageError) {
                Log.d(TAG, "Optional qnn_context library is unavailable", t)
            } catch (t: SecurityException) {
                Log.d(TAG, "Optional qnn_context library is not permitted", t)
            }

            // This owner library pulls in the canonical CausalLM and QuickDotAI
            // API libraries before any self-registering model plugin is loaded.
            System.loadLibrary("quickai_jni")

            // Downstream packages may add private model descriptors at runtime.
            try {
                System.loadLibrary("qai_ext_model")
                Log.i(TAG, "Loaded optional model-extension plugin")
            } catch (t: LinkageError) {
                Log.d(TAG, "Optional model-extension plugin is unavailable", t)
            } catch (t: SecurityException) {
                Log.d(TAG, "Optional model-extension plugin is not permitted", t)
            }

            loaded = true
            true
        } catch (t: LinkageError) {
            Log.e(TAG, "Failed to load libquickai_jni.so", t)
            false
        } catch (t: SecurityException) {
            Log.e(TAG, "Loading libquickai_jni.so is not permitted", t)
            false
        }
    }

    /** Native model-load result. A non-zero [errorCode] implies a zero [handle]. */
    data class LoadResult(val errorCode: Int, val handle: Long)

    /** Native performance counters for the most recent completed generation. */
    data class MetricsResult(
        val errorCode: Int,
        val prefillTokens: Int,
        val prefillDurationMs: Double,
        val generationTokens: Int,
        val generationDurationMs: Double,
        val totalDurationMs: Double,
        val initializationDurationMs: Double,
        val peakMemoryKb: Long
    )

    external fun loadModelHandleByNameNative(
        backend: Int,
        modelId: String,
        quant: Int,
        nativeLibDir: String?,
        modelBasePath: String?
    ): LoadResult

    external fun nativeQueryCatalog(): String

    external fun encodeModelHandleNative(handle: Long, text: String): FloatArray?

    external fun runTextStreamingNative(
        handle: Long,
        input: String,
        callback: (String) -> Int
    ): Int

    external fun runOpenAIStreamingNative(
        handle: Long,
        jsonRequest: String,
        imageSources: Array<String>,
        pixelValues: Array<FloatArray>,
        layouts: IntArray,
        patchCounts: IntArray,
        channels: IntArray,
        patchHeights: IntArray,
        patchWidths: IntArray,
        originalHeights: IntArray,
        originalWidths: IntArray,
        callback: (String) -> Int
    ): Int

    external fun getPerformanceMetricsHandleNative(handle: Long): MetricsResult

    external fun destroyModelHandleNative(handle: Long): Int

    external fun armRunCancellationNative(handle: Long): Int

    external fun disarmRunCancellationNative(handle: Long)

    external fun cancelModelHandleNative(handle: Long): Int

    @JvmStatic
    external fun configureSpeculativeDecodingNative(handle: Long, on: Boolean): Int

    private const val TAG = "NativeCausalLm"
}
