// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    LlavaNextImagePreprocessor.kt
 * @brief   Public, model-specific image sidecar helper.
 */
package com.example.quickdotai

import android.graphics.Bitmap
import android.graphics.BitmapFactory

/**
 * Preprocesses one encoded image for the LLaVA-NeXT 512-pixel patch contract.
 *
 * This utility is deliberately model-specific: applications must select a
 * preprocessor that matches the loaded model rather than treating image
 * tensors as a universal format. The returned tensor uses per-patch HWC RGB
 * layout and can be placed directly in [OpenAIImageTensorSidecar].
 */
class LlavaNextImagePreprocessor {
    private val processor = LlavaNextImageProcessor()

    /** Decode [encodedImage] (for example JPEG or PNG) and preprocess it. */
    fun preprocess(
        source: String,
        encodedImage: ByteArray
    ): BackendResult<OpenAIImageTensor> {
        if (encodedImage.isEmpty()) {
            return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "encodedImage must not be empty"
            )
        }
        val bitmap = BitmapFactory.decodeByteArray(encodedImage, 0, encodedImage.size)
            ?: return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "encodedImage is not a supported image"
            )
        return try {
            preprocess(source, bitmap)
        } finally {
            bitmap.recycle()
        }
    }

    /** Preprocess [bitmap] without taking ownership of or recycling it. */
    fun preprocess(
        source: String,
        bitmap: Bitmap
    ): BackendResult<OpenAIImageTensor> {
        if (source.isBlank()) {
            return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "source must not be blank"
            )
        }
        if (bitmap.isRecycled || bitmap.width <= 0 || bitmap.height <= 0) {
            return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "bitmap must be a live image with positive dimensions"
            )
        }

        return try {
            val input = processor.preprocess(bitmap)
            val patchSize = processor.getCropSize()
            val valuesPerPatch = CHANNELS * patchSize * patchSize
            if (input.pixelValues.isEmpty() ||
                input.pixelValues.size % valuesPerPatch != 0
            ) {
                return BackendResult.Err(
                    QuickAiError.INFERENCE_FAILED,
                    "LLaVA-NeXT preprocessor produced an invalid tensor size"
                )
            }
            BackendResult.Ok(
                OpenAIImageTensor(
                    source = source,
                    pixelValues = input.pixelValues,
                    layout = ImageTensorLayout.HWC,
                    patchCount = input.pixelValues.size / valuesPerPatch,
                    channels = CHANNELS,
                    patchHeight = patchSize,
                    patchWidth = patchSize,
                    originalHeight = input.originalSize.first,
                    originalWidth = input.originalSize.second
                )
            )
        } catch (exception: Exception) {
            BackendResult.Err(
                QuickAiError.INFERENCE_FAILED,
                exception.message ?: "LLaVA-NeXT image preprocessing failed"
            )
        }
    }

    private companion object {
        const val CHANNELS: Int = 3
    }
}
