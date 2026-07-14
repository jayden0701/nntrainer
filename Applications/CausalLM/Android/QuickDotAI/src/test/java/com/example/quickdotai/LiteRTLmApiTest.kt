// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 */
package com.example.quickdotai

import java.io.File
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LiteRTLmApiTest {
    @Test
    fun blankModelIdIsRejectedBeforeEngineCreation() {
        val result = newLiteRt().load(
            LoadModelRequest(modelId = "", modelPath = "missing.litertlm")
        )

        assertTrue(result is BackendResult.Err)
        assertEquals(QuickAiError.INVALID_PARAMETER, (result as BackendResult.Err).error)
    }

    @Test
    fun exactTextIsExplicitlyUnsupported() {
        val sink = RecordingSink()

        val result = newLiteRt().runText("exact input", sink)

        assertTrue(result is BackendResult.Err)
        assertEquals(QuickAiError.UNSUPPORTED, (result as BackendResult.Err).error)
        assertEquals(listOf(QuickAiError.UNSUPPORTED), sink.errors)
    }

    @Test
    fun unsupportedOpenAIOptionsAreNotSilentlyDropped() {
        val sink = RecordingSink()
        val request = OpenAIRequest(
            json = """
                {
                  "messages": [{"role": "user", "content": "hello"}],
                  "temperature": 0.2
                }
            """.trimIndent()
        )

        val result = newLiteRt().runOpenAI(request, sink)

        assertTrue(result is BackendResult.Err)
        assertEquals(QuickAiError.UNSUPPORTED, (result as BackendResult.Err).error)
        assertEquals(listOf(QuickAiError.UNSUPPORTED), sink.errors)
    }

    @Test
    fun developerRoleIsNotSilentlyMappedToSystem() {
        val result = newLiteRt().parseOpenAIRequest(
            """
                {
                  "messages": [
                    {"role": "developer", "content": "follow policy"},
                    {"role": "user", "content": "hello"}
                  ]
                }
            """.trimIndent()
        )

        assertTrue(result is BackendResult.Err)
        assertEquals(QuickAiError.UNSUPPORTED, (result as BackendResult.Err).error)
    }

    @Test
    fun dataImageAndTextPartsPreserveOccurrenceOrder() {
        val source = "data:image/png;base64,AQID"
        val result = newLiteRt().parseOpenAIRequest(
            """
                {
                  "messages": [{
                    "role": "user",
                    "content": [
                      {"type": "text", "text": "before"},
                      {"type": "image_url", "image_url": {"url": "$source"}},
                      {"type": "input_text", "text": "after"}
                    ]
                  }]
                }
            """.trimIndent()
        )

        assertTrue(result is BackendResult.Ok)
        val contents = (result as BackendResult.Ok).value.single().contents
        assertEquals("before", (contents[0] as LiteRTLm.ParsedContent.Text).text)
        val image = contents[1] as LiteRTLm.ParsedContent.ImageBytes
        assertEquals(source, image.source)
        assertArrayEquals(byteArrayOf(1, 2, 3), image.bytes)
        assertEquals("after", (contents[2] as LiteRTLm.ParsedContent.Text).text)
    }

    @Test
    fun remoteImageUrlIsExplicitlyUnsupported() {
        val result = newLiteRt().parseOpenAIRequest(
            """
                {
                  "messages": [{
                    "role": "user",
                    "content": [{
                      "type": "image_url",
                      "image_url": {"url": "https://example.com/image.png"}
                    }]
                  }]
                }
            """.trimIndent()
        )

        assertTrue(result is BackendResult.Err)
        assertEquals(QuickAiError.UNSUPPORTED, (result as BackendResult.Err).error)
    }

    @Test
    fun imageContentRequiresAUserMessage() {
        val result = newLiteRt().parseOpenAIRequest(
            """
                {
                  "messages": [
                    {"role": "system", "content": [{
                      "type": "image_url",
                      "image_url": {"url": "data:image/png;base64,AQID"}
                    }]},
                    {"role": "user", "content": "hello"}
                  ]
                }
            """.trimIndent()
        )

        assertTrue(result is BackendResult.Err)
        assertEquals(QuickAiError.INVALID_PARAMETER, (result as BackendResult.Err).error)
    }

    @Test
    fun preprocessedImageSidecarIsExplicitlyUnsupported() {
        val source = "quickdotai://image/0"
        val sink = RecordingSink()
        val result = newLiteRt().runOpenAI(
            OpenAIRequest(
                json = """
                    {"messages":[{"role":"user","content":[
                      {"type":"image_url","image_url":{"url":"$source"}}
                    ]}]}
                """.trimIndent(),
                imageTensors = OpenAIImageTensorSidecar(
                    tensors = listOf(
                        OpenAIImageTensor(
                            source = source,
                            pixelValues = FloatArray(3),
                            layout = ImageTensorLayout.HWC,
                            patchCount = 1,
                            channels = 3,
                            patchHeight = 1,
                            patchWidth = 1,
                            originalHeight = 1,
                            originalWidth = 1
                        )
                    )
                )
            ),
            sink
        )

        assertTrue(result is BackendResult.Err)
        assertEquals(QuickAiError.UNSUPPORTED, (result as BackendResult.Err).error)
        assertEquals(listOf(QuickAiError.UNSUPPORTED), sink.errors)
    }

    @Test
    fun readableFileUrlMapsToAnAbsoluteImageFile() {
        val file = File.createTempFile("quickdotai-image-", ".png")
        try {
            file.writeBytes(byteArrayOf(1, 2, 3))
            val source = file.toURI().toString()
            val result = newLiteRt().parseOpenAIRequest(
                """
                    {
                      "messages": [{
                        "role": "user",
                        "content": [{
                          "type": "image_url",
                          "image_url": {"url": "$source"}
                        }]
                      }]
                    }
                """.trimIndent()
            )

            assertTrue(result is BackendResult.Ok)
            val content = (result as BackendResult.Ok)
                .value.single().contents.single()
                as LiteRTLm.ParsedContent.ImageFile
            assertEquals(source, content.source)
            assertEquals(file.canonicalPath, content.absolutePath)
        } finally {
            file.delete()
        }
    }

    private class RecordingSink : StreamSink {
        val errors = mutableListOf<QuickAiError>()

        override fun onDelta(text: String) = Unit

        override fun onDone() = Unit

        override fun onError(error: QuickAiError, message: String?) {
            errors.add(error)
        }
    }

    private fun newLiteRt(): LiteRTLm = LiteRTLm(LITE_RT_DESCRIPTOR)

    private companion object {
        val LITE_RT_DESCRIPTOR = ModelDescriptor(
            id = ModelIds.GEMMA4,
            family = "gemma4",
            displayName = "Gemma4 (LiteRT)",
            runtime = RuntimeKind.LITERT,
            backends = setOf(BackendType.CPU, BackendType.GPU),
            capabilities = setOf(
                Capability.STREAMING,
                Capability.OPENAI_API,
                Capability.MULTIMODAL
            )
        )
    }
}
