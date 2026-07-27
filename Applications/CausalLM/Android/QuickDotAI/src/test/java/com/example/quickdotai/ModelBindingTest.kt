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
    fun speculativeDecodingRequiresADeclaredVariant() {
        val result = LITE_RT_DESCRIPTOR.validateLoadRequest(
            LoadModelRequest(
                backend = BackendType.GPU,
                modelId = LITE_RT_DESCRIPTOR.id,
                useSpeculativeDecoding = true
            )
        )

        assertTrue(result is BackendResult.Err)
        assertEquals(QuickAiError.UNSUPPORTED, (result as BackendResult.Err).error)
    }

    @Test
    fun speculativeDecodingBindsTheDeclaredVariant() {
        val descriptor = LITE_RT_DESCRIPTOR.copy(sdVariantId = "gemma4-sd")

        val valid = descriptor.validateLoadRequest(
            LoadModelRequest(
                backend = BackendType.GPU,
                modelId = "gemma4-sd",
                useSpeculativeDecoding = true
            )
        )
        val invalid = descriptor.validateLoadRequest(
            LoadModelRequest(
                backend = BackendType.GPU,
                modelId = descriptor.id,
                useSpeculativeDecoding = true
            )
        )

        assertTrue(valid is BackendResult.Ok)
        assertTrue(invalid is BackendResult.Err)
        assertEquals(QuickAiError.INVALID_PARAMETER, (invalid as BackendResult.Err).error)
    }

    @Test
    fun standaloneVisionEncoderIsNotSelectableForGeneration() {
        val descriptor = ModelDescriptor(
            id = ModelIds.VJEPA_QNN,
            family = "vjepa",
            displayName = "V-JEPA 2 (QNN)",
            runtime = RuntimeKind.NATIVE,
            backends = setOf(BackendType.NPU),
            capabilities = setOf(Capability.VISION_ENCODER, Capability.MULTI_IMAGE)
        )

        assertFalse(ModelCatalog.isSelectable(descriptor))
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
    }
}
