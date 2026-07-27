// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    LlavaNextImageProcessor.kt
 * @brief   Internal LLaVA-NeXT image preprocessing implementation.
 */
package com.example.quickdotai

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import androidx.core.graphics.createBitmap
import kotlin.math.ceil
import kotlin.math.min


/**
 * A Kotlin implementation of the Llava-NeXT image processor for Android.
 *
 * This class transforms a Bitmap into a list of normalized FloatArrays,
 * representing image patches, ready for input into a vision model.
 *
 * @param cropSize The size (height and width) of each square patch. Corresponds to the model's input size.
 * @param imageGridPinpoints A list of possible high-resolution grids to select from.
 * @param imageMean The mean values for normalization (R, G, B).
 * @param imageStd The standard deviation values for normalization (R, G, B).
 */
internal class LlavaNextImageProcessor(
    private val cropSize: Int = 512,
    private val imageGridPinpoints: List<Pair<Int, Int>> = listOf(
        Pair(512,1024),Pair(512,1536),Pair(512,2048),Pair(512,2560),Pair(512,3072),Pair(512,3584),Pair(512,4096),Pair(512,4608),Pair(512,5120),Pair(512,5632),Pair(512,6144),Pair(1024,512),Pair(1024,1024),Pair(1024,1536),Pair(1024,2048),Pair(1024,2560),Pair(1024,3072),Pair(1536,512),Pair(1536,1024),Pair(1536,1536),Pair(1536,2048),Pair(2048,512),Pair(2048,1024),Pair(2048,1536),Pair(2560,512),Pair(2560,1024),Pair(3072,512),Pair(3072,1024),Pair(3584,512),Pair(4096,512),Pair(4608,512),Pair(5120,512),Pair(5632,512),Pair(6144,512)
    ),
    private val imageMean: FloatArray = floatArrayOf(0.5f, 0.5f, 0.5f),
    private val imageStd: FloatArray = floatArrayOf(0.5f, 0.5f, 0.5f),
    private val rescaleFactor: Double = 1.0 / 255.0,
    private val patchMergeType: String = "nopad"
) {

    private fun resizeImage(inputBitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        val width = inputBitmap.width
        val height = inputBitmap.height
        val pixels = IntArray(width * height)
        inputBitmap.getPixels(pixels, 0, width, 0, 0, width, height)
//        val resizedPixels = PillowBicubicResizer.resize(pixels, width, height, targetWidth, targetHeight)
        val resizedPixels = PillowBilinearResizer.resize(pixels, width, height, targetWidth, targetHeight)
        return Bitmap.createBitmap(resizedPixels, targetWidth, targetHeight, Bitmap.Config.ARGB_8888)

//        return inputBitmap.scale(targetWidth, targetHeight, filter = true)
    }

    // Represents the final model input for a single image
    data class ModelInput(val pixelValues: FloatArray, val originalSize: Pair<Int, Int>)

    /**
     * @brief Returns the crop size (patch size) used for image preprocessing.
     */
    fun getCropSize(): Int = cropSize

    /**
     * Preprocesses a single Bitmap image.
     *
     * @param image The input Bitmap.
     * @return A ModelInput object containing a list of float arrays (patches) and original image size.
     */
    fun preprocess(image: Bitmap): ModelInput {
        val originalSize = Pair(image.height, image.width)
        val imagePatches = getImagePatches(image)
        return try {
            val perImagePatchSize = cropSize * cropSize * 3
            val floatValues = FloatArray(imagePatches.size * perImagePatchSize)

            imagePatches.forEachIndexed { index, patch ->
                // All patches, including the base one, have the target crop size.
                normalize(patch, floatValues, index * perImagePatchSize)
            }

            ModelInput(pixelValues = floatValues, originalSize = originalSize)
        } finally {
            imagePatches.forEach { patch ->
                if (!patch.isRecycled) patch.recycle()
            }
        }
    }

    /**
     * Creates image patches based on the LLaVa-NeXT "any-resolution" strategy.
     */
    private fun getImagePatches(image: Bitmap): List<Bitmap> {
        // 1. Create the base, low-res image (resized to cropSize x cropSize)
        val baseImage = resizeImage(image, cropSize, cropSize)
        try {
            // 2. Handle high-resolution patching
            val bestResolution = selectBestResolution(Pair(image.height, image.width))
                ?: throw IllegalStateException("imageGridPinpoints must not be empty")
            val resizedForPatching = resizeForPatching(image, bestResolution)
            val paddedImage = try {
                padToResolution(resizedForPatching, bestResolution)
            } catch (throwable: Throwable) {
                if (!resizedForPatching.isRecycled) {
                    resizedForPatching.recycle()
                }
                throw throwable
            }
            val highResPatches = try {
                divideToPatches(paddedImage, cropSize)
            } catch (throwable: Throwable) {
                if (!paddedImage.isRecycled) {
                    paddedImage.recycle()
                }
                if (resizedForPatching !== paddedImage && !resizedForPatching.isRecycled) {
                    resizedForPatching.recycle()
                }
                throw throwable
            }

            // Bitmap.createBitmap may return its source when a single patch
            // covers the whole image. Leave that object for preprocess() to
            // recycle with the returned patches.
            if (highResPatches.none { it === paddedImage } && !paddedImage.isRecycled) {
                paddedImage.recycle()
            }
            if (resizedForPatching !== paddedImage && !resizedForPatching.isRecycled) {
                resizedForPatching.recycle()
            }
            return listOf(baseImage) + highResPatches
        } catch (throwable: Throwable) {
            if (!baseImage.isRecycled) {
                baseImage.recycle()
            }
            throw throwable
        }
    }

    /**
     * Selects the best grid resolution from `imageGridPinpoints` that fits the image.
     */
    private fun selectBestResolution(originalSize: Pair<Int, Int>): Pair<Int, Int>? {
        if (imageGridPinpoints.isEmpty()) {
            return null
        }

        val (originalHeight, originalWidth) = originalSize
        var bestFit: Pair<Int, Int>? = null
        var maxEffectiveResolution = -1
        var minWastedResolution = Int.MAX_VALUE

        for (resolution in imageGridPinpoints) {
            val (height, width) = resolution

            // Use Double for division to maintain precision
            val scale = min(
                width.toDouble() / originalWidth,
                height.toDouble() / originalHeight
            )
            val downscaledWidth = (originalWidth * scale).toInt()
            val downscaledHeight = (originalHeight * scale).toInt()

            val effectiveResolution = min(
                downscaledWidth * downscaledHeight,
                originalWidth * originalHeight
            )
            val wastedResolution = (width * height) - effectiveResolution

            if (effectiveResolution > maxEffectiveResolution ||
                (effectiveResolution == maxEffectiveResolution && wastedResolution < minWastedResolution)
            ) {
                maxEffectiveResolution = effectiveResolution
                minWastedResolution = wastedResolution
                bestFit = resolution
            }
        }

        return bestFit
    }

    private fun getPatchOutputSize(image: Bitmap, targetResolution: Pair<Int, Int>): Pair<Int, Int>{
        val (targetHeight, targetWidth) = targetResolution
        val (originalHeight, originalWidth) = image.height to image.width

        val scaleW = targetWidth.toFloat() / originalWidth
        val scaleH = targetHeight.toFloat() / originalHeight

        val newWidth: Int
        val newHeight: Int

        if (scaleW < scaleH) {
            newWidth = targetWidth
            newHeight = min(ceil(originalHeight * scaleW).toInt(), targetHeight)
        } else {
            newHeight = targetHeight
            newWidth = min(ceil(originalWidth * scaleH).toInt(), targetWidth)
        }
        return Pair<Int, Int>(newHeight, newWidth);
    }

    /**
     * Resizes an image to fit within a target resolution while maintaining aspect ratio.
     */
    private fun resizeForPatching(image: Bitmap, targetResolution: Pair<Int, Int>): Bitmap {
        val (targetHeight, targetWidth) = targetResolution
        val newWidth: Int
        val newHeight: Int

        if (patchMergeType == "nopad") {
            newHeight = targetHeight
            newWidth = targetWidth
        }
        else { // spatial_unpad
            val (newH, newW) = getPatchOutputSize(image, targetResolution)
            newHeight = newH
            newWidth = newW
        }

        return resizeImage(image, newWidth, newHeight)
    }

    /**
     * Pads a resized image to the exact target resolution by adding black bars.
     */
    private fun padToResolution(image: Bitmap, targetResolution: Pair<Int, Int>): Bitmap {
        val (targetHeight, targetWidth) = targetResolution
        val (imageHeight, imageWidth) = getPatchOutputSize(image, targetResolution)

        if (imageHeight == targetHeight && imageWidth == targetWidth) {
            return image
        }

        val paddedBitmap = createBitmap(targetWidth, targetHeight, image.config ?: Bitmap.Config.ARGB_8888)
        val canvas = Canvas(paddedBitmap)
        canvas.drawColor(Color.BLACK) // Pad with black

        val left = (targetWidth - imageWidth) / 2f
        val top = (targetHeight - imageHeight) / 2f

        canvas.drawBitmap(image, left, top, null)
        return paddedBitmap
    }

    /**
     * Divides an image into a grid of square patches.
     */
    private fun divideToPatches(image: Bitmap, patchSize: Int): List<Bitmap> {
        val patches = mutableListOf<Bitmap>()
        val (height, width) = image.height to image.width

        try {
            for (i in 0 until height step patchSize) {
                for (j in 0 until width step patchSize) {
                    val patch = Bitmap.createBitmap(image, j, i, patchSize, patchSize)
                    patches.add(patch)
                }
            }
        } catch (throwable: Throwable) {
            patches.forEach { patch ->
                if (!patch.isRecycled) {
                    patch.recycle()
                }
            }
            throw throwable
        }
        return patches
    }

    /**
     * Rescales pixel values from [0, 255] to [0, 1] and then normalizes them.
     * The output is a flattened FloatArray in HWC (Height, Width, Channels) format.
     */
    private fun normalize(image: Bitmap, floatValues: FloatArray, offset: Int) {
        val width = image.width
        val height = image.height
        val pixels = IntArray(width * height)
        image.getPixels(pixels, 0, width, 0, 0, width, height)

        // HWC format: RGB...RGB...RGB...

        for (i in 0 until (width * height)) {
            val pixel = pixels[i]
            // Rescale to [0, 1] then normalize
            floatValues[offset + i * 3] =
                ((Color.red(pixel) * rescaleFactor).toFloat() - imageMean[0]) / imageStd[0]
            floatValues[offset + i * 3 + 1] =
                ((Color.green(pixel) * rescaleFactor).toFloat() - imageMean[1]) / imageStd[1]
            floatValues[offset + i * 3 + 2] =
                ((Color.blue(pixel) * rescaleFactor).toFloat() - imageMean[2]) / imageStd[2]
        }
    }

}
