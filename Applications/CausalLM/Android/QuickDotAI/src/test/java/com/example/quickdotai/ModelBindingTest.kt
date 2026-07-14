// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 */
package com.example.quickdotai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelBindingTest {
    @Test
    fun engineRejectsARequestForAnotherCatalogModel() {
        val engine = createEngine(LITE_RT_DESCRIPTOR)

        val result = engine.load(
            LoadModelRequest(
                backend = BackendType.GPU,
                modelId = "another-model",
                modelPath = "missing.litertlm"
            )
        )

        assertTrue(result is BackendResult.Err)
        assertEquals(QuickAiError.INVALID_PARAMETER, (result as BackendResult.Err).error)
        assertTrue(result.message.orEmpty().contains(LITE_RT_DESCRIPTOR.id))
    }

    @Test
    fun engineRejectsABackendOutsideTheDescriptor() {
        val descriptor = LITE_RT_DESCRIPTOR.copy(backends = setOf(BackendType.CPU))
        val engine = createEngine(descriptor)

        val result = engine.load(
            LoadModelRequest(
                backend = BackendType.GPU,
                modelId = descriptor.id,
                modelPath = "missing.litertlm"
            )
        )

        assertTrue(result is BackendResult.Err)
        assertEquals(QuickAiError.INVALID_PARAMETER, (result as BackendResult.Err).error)
    }

    @Test
    fun standaloneVisionEncoderModifiersAreNotSelectableForGeneration() {
        val descriptor = ModelDescriptor(
            id = ModelIds.VJEPA_QNN,
            family = "vjepa",
            displayName = "V-JEPA 2 (QNN)",
            runtime = RuntimeKind.NATIVE,
            backends = setOf(BackendType.NPU),
            capabilities = setOf(
                Capability.VISION_ENCODER,
                Capability.MULTIMODAL,
                Capability.TOOL_USE,
                Capability.MULTI_IMAGE
            )
        )

        assertFalse(ModelCatalog.isSelectable(descriptor))
    }

    @Test
    fun multimodalEncoderWithGenerationCapabilitiesIsSelectable() {
        val descriptor = ModelDescriptor(
            id = ModelIds.VJEPA_LFM2,
            family = "vjepa",
            displayName = "V-JEPA 2 + LFM2",
            runtime = RuntimeKind.NATIVE,
            backends = setOf(BackendType.CPU),
            capabilities = setOf(
                Capability.VISION_ENCODER,
                Capability.MULTIMODAL,
                Capability.OPENAI_API,
                Capability.MULTI_IMAGE
            )
        )

        assertTrue(ModelCatalog.isSelectable(descriptor))
    }

    @Test
    fun nativeImageSidecarsRequireMultimodalCapability() {
        val descriptor = NATIVE_DESCRIPTOR.copy(
            capabilities = setOf(Capability.OPENAI_API, Capability.MULTI_IMAGE)
        )

        val result = validateNativeOpenAIImageCapabilities(descriptor, imageCount = 1)

        assertTrue(result is BackendResult.Err)
        assertEquals(QuickAiError.UNSUPPORTED, (result as BackendResult.Err).error)
    }

    @Test
    fun multipleNativeImageSidecarsRequireMultiImageCapability() {
        val descriptor = NATIVE_DESCRIPTOR.copy(
            capabilities = setOf(Capability.OPENAI_API, Capability.MULTIMODAL)
        )

        val result = validateNativeOpenAIImageCapabilities(descriptor, imageCount = 2)

        assertTrue(result is BackendResult.Err)
        assertEquals(QuickAiError.UNSUPPORTED, (result as BackendResult.Err).error)
    }

    @Test
    fun multiImageNativeModelAcceptsMultipleImageSidecars() {
        val descriptor = NATIVE_DESCRIPTOR.copy(
            capabilities = setOf(
                Capability.OPENAI_API,
                Capability.MULTIMODAL,
                Capability.MULTI_IMAGE
            )
        )

        val result = validateNativeOpenAIImageCapabilities(descriptor, imageCount = 2)

        assertTrue(result is BackendResult.Ok)
    }

    @Test
    fun textOnlyNativeRequestDoesNotRequireImageCapabilities() {
        val result = validateNativeOpenAIImageCapabilities(
            NATIVE_DESCRIPTOR.copy(capabilities = setOf(Capability.OPENAI_API)),
            imageCount = 0
        )

        assertTrue(result is BackendResult.Ok)
    }

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

        val NATIVE_DESCRIPTOR = ModelDescriptor(
            id = "native-test-model",
            family = "native-test",
            displayName = "Native Test Model",
            runtime = RuntimeKind.NATIVE,
            backends = setOf(BackendType.NPU),
            capabilities = setOf(Capability.OPENAI_API)
        )
    }
}
