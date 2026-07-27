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
    fun malformedJsonIsRejected() {
        assertInvalid(OpenAIRequest(json = """{"messages":[""").validate())
    }

    @Test
    fun requestMustBeAnObjectWithNonEmptyMessages() {
        assertInvalid(OpenAIRequest(json = "[]").validate())
        assertInvalid(OpenAIRequest(json = "{}").validate())
        assertInvalid(OpenAIRequest(json = """{"messages":[]}""").validate())
    }

    @Test
    fun everyMessageMustBeAnObject() {
        assertInvalid(OpenAIRequest(json = """{"messages":[42]}""").validate())
    }

    @Test
    fun everyMessageRequiresASupportedRole() {
        assertInvalid(
            OpenAIRequest(
                json = """{"messages":[{"role":"unknown","content":"hello"}]}"""
            ).validate()
        )
    }

    @Test
    fun messageContentMustBeAStringOrNonEmptyArray() {
        assertInvalid(
            OpenAIRequest(
                json = """{"messages":[{"role":"user","content":42}]}"""
            ).validate()
        )
        assertInvalid(
            OpenAIRequest(
                json = """{"messages":[{"role":"user","content":[]}]}"""
            ).validate()
        )
    }

    @Test
    fun textContentPartRequiresAString() {
        assertInvalid(
            OpenAIRequest(
                json = """
                    {
                      "messages": [{
                        "role": "user",
                        "content": [{"type": "text", "text": 42}]
                      }]
                    }
                """.trimIndent()
            ).validate()
        )
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

        assertInvalid(request.validate())
    }

    @Test
    fun imageUrlWithoutTensorSidecarIsStructurallyValid() {
        val request = OpenAIRequest(
            json = requestJson("data:image/png;base64,AA==")
        )

        assertTrue(request.validate() is BackendResult.Ok)
        assertInvalid(request.validateForNative())
    }

    @Test
    fun imageUrlSourceMustBeANonBlankString() {
        assertInvalid(
            OpenAIRequest(
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
        )
    }

    @Test
    fun imageUrlDetailMustBeSupported() {
        assertInvalid(
            OpenAIRequest(
                json = """
                    {
                      "messages": [{
                        "role": "user",
                        "content": [{
                          "type": "image_url",
                          "image_url": {"url": "image://0", "detail": "maximum"}
                        }]
                      }]
                    }
                """.trimIndent()
            ).validate()
        )
    }

    @Test
    fun imageUrlRequiresAUserMessage() {
        assertInvalid(
            OpenAIRequest(
                json = """
                    {
                      "messages": [{
                        "role": "assistant",
                        "content": [{
                          "type": "image_url",
                          "image_url": {"url": "image://0"}
                        }]
                      }]
                    }
                """.trimIndent()
            ).validate()
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

        assertInvalid(request.validate())
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
    fun sidecarVersionMustBeCurrent() {
        val request = OpenAIRequest(
            json = requestJson("quickdotai://image/0"),
            imageTensors = OpenAIImageTensorSidecar(
                version = OpenAIImageTensorSidecar.CURRENT_VERSION + 1,
                tensors = listOf(
                    tensor("quickdotai://image/0", FloatArray(12))
                )
            )
        )

        assertInvalid(request.validate())
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

        assertInvalid(request.validate())
    }

    @Test
    fun denseTensorDimensionsMustBePositiveAndMustNotOverflow() {
        val source = "quickdotai://image/0"
        val nonPositive = requestWithTensor(
            tensor(source, FloatArray(1), patchCount = 0)
        )
        val overflowing = requestWithTensor(
            tensor(
                source,
                FloatArray(1),
                patchCount = Int.MAX_VALUE,
                channels = 2,
                patchHeight = 1,
                patchWidth = 1
            )
        )

        assertInvalid(nonPositive.validate())
        assertInvalid(overflowing.validate())
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
        val nanValues = FloatArray(12)
        nanValues[5] = Float.NaN
        val infiniteValues = FloatArray(12)
        infiniteValues[5] = Float.POSITIVE_INFINITY

        assertInvalid(
            requestWithTensor(
                tensor("quickdotai://image/0", nanValues)
            ).validate()
        )
        assertInvalid(
            requestWithTensor(
                tensor("quickdotai://image/0", infiniteValues)
            ).validate()
        )
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

    private fun assertInvalid(result: BackendResult<Unit>) {
        assertTrue(result is BackendResult.Err)
        assertEquals(
            QuickAiError.INVALID_PARAMETER,
            (result as BackendResult.Err).error
        )
    }

    private fun requestWithTensor(tensor: OpenAIImageTensor): OpenAIRequest =
        OpenAIRequest(
            json = requestJson("quickdotai://image/0"),
            imageTensors = OpenAIImageTensorSidecar(tensors = listOf(tensor))
        )

    private fun tensor(
        source: String,
        values: FloatArray,
        layout: ImageTensorLayout = ImageTensorLayout.HWC,
        patchCount: Int = 1,
        channels: Int = 3,
        patchHeight: Int = 2,
        patchWidth: Int = 2
    ) = OpenAIImageTensor(
        source = source,
        pixelValues = values,
        layout = layout,
        patchCount = patchCount,
        channels = channels,
        patchHeight = patchHeight,
        patchWidth = patchWidth,
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
