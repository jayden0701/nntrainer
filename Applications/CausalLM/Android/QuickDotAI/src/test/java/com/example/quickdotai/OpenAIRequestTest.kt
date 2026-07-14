// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 */
package com.example.quickdotai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class OpenAIRequestTest {
    @Test
    fun textOnlyRequestIsValid() {
        val request = OpenAIRequest(
            json = """{"messages":[{"role":"user","content":"hello"}]}"""
        )

        assertTrue(request.validate() is BackendResult.Ok)
    }

    @Test
    fun everyMessageMustBeAnObject() {
        val result = OpenAIRequest(json = """{"messages":[42]}""").validate()

        assertTrue(result is BackendResult.Err)
        assertEquals(QuickAiError.INVALID_PARAMETER, (result as BackendResult.Err).error)
    }

    @Test
    fun everyMessageRequiresASupportedRole() {
        val result = OpenAIRequest(
            json = """{"messages":[{"role":"unknown","content":"hello"}]}"""
        ).validate()

        assertTrue(result is BackendResult.Err)
        assertEquals(QuickAiError.INVALID_PARAMETER, (result as BackendResult.Err).error)
    }

    @Test
    fun emptySidecarIsValidForTextOnlyRequest() {
        val request = OpenAIRequest(
            json = """{"messages":[{"role":"user","content":"hello"}]}""",
            imageTensors = OpenAIImageTensorSidecar(tensors = emptyList())
        )

        assertTrue(request.validate() is BackendResult.Ok)
    }

    @Test
    fun tensorSourceAndDenseValueCountAreValidated() {
        val request = OpenAIRequest(
            json = requestJson("quickdotai://image/0"),
            imageTensors = OpenAIImageTensorSidecar(
                tensors = listOf(
                    tensor(
                        source = "quickdotai://image/0",
                        values = FloatArray(12),
                        layout = ImageTensorLayout.CHW
                    )
                )
            )
        )

        assertTrue(request.validate() is BackendResult.Ok)
    }

    @Test
    fun tensorSourceMustMatchImageUrl() {
        val request = OpenAIRequest(
            json = requestJson("quickdotai://image/0"),
            imageTensors = OpenAIImageTensorSidecar(
                tensors = listOf(
                    tensor("quickdotai://different", FloatArray(12))
                )
            )
        )

        val result = request.validate()
        assertTrue(result is BackendResult.Err)
        assertEquals(
            QuickAiError.INVALID_PARAMETER,
            (result as BackendResult.Err).error
        )
    }

    @Test
    fun imageUrlWithoutTensorSidecarIsStructurallyValid() {
        val request = OpenAIRequest(
            json = requestJson("data:image/png;base64,AA==")
        )

        assertTrue(request.validate() is BackendResult.Ok)
        val nativeResult = request.validateForNative()
        assertTrue(nativeResult is BackendResult.Err)
        assertEquals(
            QuickAiError.INVALID_PARAMETER,
            (nativeResult as BackendResult.Err).error
        )
    }

    @Test
    fun imageUrlSourceMustBeANonBlankString() {
        val result = OpenAIRequest(
            json = """
                {
                  "messages": [{
                    "role": "user",
                    "content": [
                      {"type": "image_url", "image_url": {"url": ""}}
                    ]
                  }]
                }
            """.trimIndent()
        ).validate()

        assertTrue(result is BackendResult.Err)
        assertEquals(
            QuickAiError.INVALID_PARAMETER,
            (result as BackendResult.Err).error
        )
    }

    @Test
    fun everyImageUrlRequiresMatchingTensor() {
        val request = OpenAIRequest(
            json = requestJson(
                "quickdotai://image/0",
                "quickdotai://image/1"
            ),
            imageTensors = OpenAIImageTensorSidecar(
                tensors = listOf(
                    tensor("quickdotai://image/0", FloatArray(12))
                )
            )
        )

        assertTrue(request.validate() is BackendResult.Err)
    }

    @Test
    fun duplicateImageUrlOccurrencesRequireOrderedDuplicateTensors() {
        val source = "quickdotai://image/reused"
        val request = OpenAIRequest(
            json = requestJson(source, source),
            imageTensors = OpenAIImageTensorSidecar(
                tensors = listOf(
                    tensor(source, FloatArray(12)),
                    tensor(source, FloatArray(12))
                )
            )
        )

        assertTrue(request.validate() is BackendResult.Ok)
    }

    @Test
    fun denseTensorMustHaveExpectedValueCount() {
        val request = OpenAIRequest(
            json = requestJson("quickdotai://image/0"),
            imageTensors = OpenAIImageTensorSidecar(
                tensors = listOf(
                    tensor("quickdotai://image/0", FloatArray(11))
                )
            )
        )

        assertTrue(request.validate() is BackendResult.Err)
    }

    @Test
    fun modelNativeTensorUsesModelSpecificValueCount() {
        val request = OpenAIRequest(
            json = requestJson("quickdotai://image/0"),
            imageTensors = OpenAIImageTensorSidecar(
                tensors = listOf(
                    tensor(
                        source = "quickdotai://image/0",
                        values = FloatArray(7),
                        layout = ImageTensorLayout.MODEL_NATIVE
                    )
                )
            )
        )

        assertTrue(request.validate() is BackendResult.Ok)
    }

    @Test
    fun tensorValuesMustBeFinite() {
        val values = FloatArray(12)
        values[5] = Float.NaN
        val request = OpenAIRequest(
            json = requestJson("quickdotai://image/0"),
            imageTensors = OpenAIImageTensorSidecar(
                tensors = listOf(
                    tensor("quickdotai://image/0", values)
                )
            )
        )

        assertTrue(request.validate() is BackendResult.Err)
    }

    @Test
    fun nativeUnsupportedCodeIsPreserved() {
        assertEquals(QuickAiError.UNSUPPORTED, QuickAiError.fromNativeCode(6))
    }

    @Test
    fun imageLayoutsMatchTheNativeAbi() {
        assertEquals(0, ImageTensorLayout.MODEL_NATIVE.nativeValue)
        assertEquals(1, ImageTensorLayout.HWC.nativeValue)
        assertEquals(2, ImageTensorLayout.CHW.nativeValue)
    }

    private fun tensor(
        source: String,
        values: FloatArray,
        layout: ImageTensorLayout = ImageTensorLayout.HWC
    ) = OpenAIImageTensor(
        source = source,
        pixelValues = values,
        layout = layout,
        patchCount = 1,
        channels = 3,
        patchHeight = 2,
        patchWidth = 2,
        originalHeight = 10,
        originalWidth = 20
    )

    private fun requestJson(vararg sources: String): String = """
        {
          "messages": [{
            "role": "user",
            "content": [
              ${sources.joinToString(",\n") { source ->
                  """{"type": "image_url", "image_url": {"url": "$source"}}"""
              }},
              {"type": "text", "text": "describe it"}
            ]
          }]
        }
    """.trimIndent()
}
