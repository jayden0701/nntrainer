// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    LiteRTLm.kt
 * @brief   Minimal QuickDotAI adapter for LiteRT-LM.
 */
package com.example.quickdotai

import android.util.Log
import com.google.ai.edge.litertlm.Backend as LlmBackend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import java.io.File
import java.net.URI
import java.util.Base64
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull

/**
 * LiteRT-LM-backed engine.
 *
 * LiteRT-LM's Conversation API cannot guarantee exact untemplated input, so
 * [runText] is explicitly unsupported. [runOpenAI] implements the strict
 * message subset that LiteRT-LM can represent, including ordered text and
 * locally resolvable image content. Callers must provide
 * [LoadModelRequest.modelPath]; this class never guesses a shared storage
 * location.
 */
internal class LiteRTLm(
    private val descriptor: ModelDescriptor
) : QuickDotAI {
    override val kind: String = "litert-lm"

    override var modelId: String? = null
        private set

    private var engine: Engine? = null
    private var loadedRequest: LoadModelRequest? = null

    @Volatile
    private var currentConversation: Conversation? = null

    private val conversationLock = Any()
    private val cancellationEpoch = AtomicLong(0L)
    private var initializationDurationMs: Double = 0.0
    private var lastRunDurationMs: Double = 0.0

    override fun load(req: LoadModelRequest): BackendResult<Unit> {
        when (val validation = descriptor.validateLoadRequest(req)) {
            is BackendResult.Ok -> Unit
            is BackendResult.Err -> return validation
        }
        val existingEngine = engine
        if (existingEngine != null) {
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

        val modelPath = req.modelPath?.takeIf { it.isNotBlank() }
            ?: return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "modelPath is required when loading a LiteRT-LM model"
            )
        val modelFile = File(modelPath)
        if (!modelFile.isFile || !modelFile.canRead()) {
            return BackendResult.Err(
                QuickAiError.MODEL_LOAD_FAILED,
                "model file is not readable at $modelPath"
            )
        }
        if (req.backend == BackendType.NPU || req.visionBackend == BackendType.NPU) {
            return BackendResult.Err(
                QuickAiError.UNSUPPORTED,
                "The bundled LiteRT-LM adapter supports CPU and GPU backends only"
            )
        }
        if (req.useSpeculativeDecoding) {
            return BackendResult.Err(
                QuickAiError.UNSUPPORTED,
                "Speculative decoding is not supported by the LiteRT-LM adapter"
            )
        }
        if (req.maxNumTokens != null && req.maxNumTokens <= 0) {
            return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "maxNumTokens must be positive when provided"
            )
        }

        val config = EngineConfig(
            modelPath = modelPath,
            backend = mapBackend(req.backend),
            visionBackend = req.visionBackend?.let(::mapBackend),
            cacheDir = req.cacheDir,
            maxNumTokens = req.maxNumTokens
        )

        var createdEngine: Engine? = null
        return try {
            val startNs = System.nanoTime()
            createdEngine = Engine(config)
            createdEngine.initialize()
            initializationDurationMs = elapsedMs(startNs)
            engine = createdEngine
            loadedRequest = req
            modelId = req.modelId
            BackendResult.Ok(Unit)
        } catch (t: Throwable) {
            Log.e(TAG, "LiteRT-LM engine load failed", t)
            try {
                createdEngine?.close()
            } catch (closeError: Throwable) {
                Log.w(TAG, "Partially initialized LiteRT-LM engine failed to close", closeError)
            }
            BackendResult.Err(
                QuickAiError.MODEL_LOAD_FAILED,
                t.message ?: "LiteRT-LM engine initialization failed"
            )
        }
    }

    override fun runText(text: String, sink: StreamSink): BackendResult<Unit> {
        val message =
            "LiteRT-LM Conversation cannot guarantee exact raw-text input without a chat template"
        emitError(sink, QuickAiError.UNSUPPORTED, message)
        return BackendResult.Err(QuickAiError.UNSUPPORTED, message)
    }

    override fun runOpenAI(
        request: OpenAIRequest,
        sink: StreamSink
    ): BackendResult<Unit> {
        val runCancellationEpoch = cancellationEpoch.get()
        when (val validation = request.validate()) {
            is BackendResult.Ok -> Unit
            is BackendResult.Err -> {
                emitError(sink, validation.error, validation.message)
                return validation
            }
        }
        if (!request.imageTensors?.tensors.isNullOrEmpty()) {
            val message =
                "LiteRT-LM does not accept preprocessed OpenAI image tensor sidecars"
            emitError(sink, QuickAiError.UNSUPPORTED, message)
            return BackendResult.Err(QuickAiError.UNSUPPORTED, message)
        }

        val parsed = when (val result = parseOpenAIRequest(request.json)) {
            is BackendResult.Ok -> result.value
            is BackendResult.Err -> {
                emitError(sink, result.error, result.message)
                return result
            }
        }
        val loadedEngine = engine
        if (loadedEngine == null) {
            val message = "LiteRTLm has not been loaded yet"
            emitError(sink, QuickAiError.NOT_INITIALIZED, message)
            return BackendResult.Err(QuickAiError.NOT_INITIALIZED, message)
        }

        val initialMessages: List<Message>
        val finalUserContents: Contents
        try {
            initialMessages = parsed.dropLast(1).map(::toLiteRtMessage)
            finalUserContents = toLiteRtContents(parsed.last().contents)
        } catch (t: Throwable) {
            val message = t.message ?: "Unable to convert OpenAI messages"
            emitError(sink, QuickAiError.INVALID_PARAMETER, message)
            return BackendResult.Err(QuickAiError.INVALID_PARAMETER, message)
        }
        val conversation = try {
            if (initialMessages.isEmpty()) {
                loadedEngine.createConversation()
            } else {
                loadedEngine.createConversation(
                    ConversationConfig(initialMessages = initialMessages)
                )
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Unable to create LiteRT-LM conversation", t)
            val message = t.message ?: "Unable to create LiteRT-LM conversation"
            emitError(sink, QuickAiError.INFERENCE_FAILED, message)
            return BackendResult.Err(QuickAiError.INFERENCE_FAILED, message)
        }

        val cancelledBeforePublish = synchronized(conversationLock) {
            currentConversation = conversation
            cancellationEpoch.get() != runCancellationEpoch
        }
        if (cancelledBeforePublish) {
            try {
                conversation.cancelProcess()
            } catch (t: Throwable) {
                Log.w(TAG, "Early LiteRT-LM cancellation failed", t)
            }
        }
        return try {
            streamFinalUserMessage(
                conversation,
                finalUserContents,
                sink,
                runCancellationEpoch
            )
        } finally {
            synchronized(conversationLock) {
                if (currentConversation === conversation) {
                    currentConversation = null
                }
            }
            try {
                conversation.close()
            } catch (t: Throwable) {
                Log.w(TAG, "LiteRT-LM conversation.close() threw", t)
            }
        }
    }

    private fun streamFinalUserMessage(
        conversation: Conversation,
        contents: Contents,
        sink: StreamSink,
        runCancellationEpoch: Long
    ): BackendResult<Unit> {
        val terminal = LiteRtStreamTerminal(sink) { error, message ->
            emitError(sink, error, message)
        }
        val accumulated = StringBuilder()
        val reasoningAccumulated = StringBuilder()
        val startNs = System.nanoTime()

        val callback = object : MessageCallback {
            override fun onMessage(message: Message) {
                terminal.tryDeliver(
                    delivery = {
                        if (!isRunCancelled(runCancellationEpoch)) {
                            val fullText = message.toString()
                            val delta = cumulativeDelta(accumulated, fullText)
                            if (delta.isNotEmpty()) sink.onDelta(delta)

                            val reasoning = message.channels[THOUGHT_CHANNEL_NAME].orEmpty()
                            val reasoningDelta =
                                cumulativeDelta(reasoningAccumulated, reasoning)
                            if (reasoningDelta.isNotEmpty()) {
                                sink.onReasoningDelta(reasoningDelta)
                            }
                        }
                    },
                    errorFrom = { throwable ->
                        BackendResult.Err(
                            QuickAiError.INFERENCE_FAILED,
                            "StreamSink callback failed: ${throwable.message}"
                        )
                    },
                    afterFailure = {
                        cancellationEpoch.incrementAndGet()
                        try {
                            conversation.cancelProcess()
                        } catch (cancelError: Throwable) {
                            Log.w(
                                TAG,
                                "LiteRT-LM callback cancellation failed",
                                cancelError
                            )
                        }
                    }
                )
            }

            override fun onDone() {
                lastRunDurationMs = elapsedMs(startNs)
                terminal.tryCompleteDone()
            }

            override fun onError(throwable: Throwable) {
                lastRunDurationMs = elapsedMs(startNs)
                if (isRunCancelled(runCancellationEpoch)) {
                    terminal.tryCompleteDone()
                    return
                }
                val error = BackendResult.Err(
                    QuickAiError.INFERENCE_FAILED,
                    throwable.message ?: "LiteRT-LM streaming failed"
                )
                terminal.tryCompleteError(error)
            }
        }

        return try {
            conversation.sendMessageAsync(contents, callback)
            if (isRunCancelled(runCancellationEpoch)) {
                try {
                    conversation.cancelProcess()
                } catch (cancelError: Throwable) {
                    Log.w(TAG, "Early LiteRT-LM generation failed to cancel", cancelError)
                }
            }
            val timeoutNanos = TimeUnit.MINUTES.toNanos(STREAM_TIMEOUT_MINUTES)
            while (!terminal.await(CANCELLATION_POLL_MILLIS, TimeUnit.MILLISECONDS)) {
                if (isRunCancelled(runCancellationEpoch)) {
                    lastRunDurationMs = elapsedMs(startNs)
                    if (!terminal.tryCompleteDone()) {
                        terminal.awaitCompletion()
                    }
                    break
                }
                if (System.nanoTime() - startNs >= timeoutNanos) {
                    lastRunDurationMs = elapsedMs(startNs)
                    val error = BackendResult.Err(
                        QuickAiError.INFERENCE_FAILED,
                        "LiteRT-LM streaming timed out"
                    )
                    val ownsTimeout = terminal.tryCompleteError(error) {
                        cancellationEpoch.incrementAndGet()
                        try {
                            conversation.cancelProcess()
                        } catch (cancelError: Throwable) {
                            Log.w(
                                TAG,
                                "Timed-out LiteRT-LM generation failed to cancel",
                                cancelError
                            )
                        }
                    }
                    if (!ownsTimeout) {
                        terminal.awaitCompletion()
                    }
                    break
                }
            }
            terminal.result()
        } catch (t: Throwable) {
            Log.e(TAG, "LiteRT-LM streaming call failed", t)
            val error = BackendResult.Err(
                QuickAiError.INFERENCE_FAILED,
                t.message ?: "LiteRT-LM streaming failed"
            )
            val ownsFailure = terminal.tryCompleteError(error) {
                cancellationEpoch.incrementAndGet()
            }
            if (!ownsFailure) {
                terminal.awaitCompletion()
            }
            terminal.result()
        }
    }

    private fun isRunCancelled(runCancellationEpoch: Long): Boolean =
        cancellationEpoch.get() != runCancellationEpoch

    private fun cumulativeDelta(accumulated: StringBuilder, fullText: String): String {
        val previous = accumulated.toString()
        val delta = if (fullText.startsWith(previous)) {
            fullText.substring(previous.length)
        } else {
            fullText
        }
        if (delta.isNotEmpty()) accumulated.append(delta)
        return delta
    }

    /** Parse only the OpenAI subset LiteRT-LM can represent faithfully. */
    internal fun parseOpenAIRequest(
        jsonRequest: String
    ): BackendResult<List<ParsedMessage>> {
        val root = try {
            Json.parseToJsonElement(jsonRequest) as JsonObject
        } catch (t: Throwable) {
            return BackendResult.Err(QuickAiError.INVALID_PARAMETER, t.message)
        }

        val unsupportedFields = root.keys - setOf("messages", "model")
        if (unsupportedFields.isNotEmpty()) {
            return BackendResult.Err(
                QuickAiError.UNSUPPORTED,
                "LiteRT-LM OpenAI adapter does not support: " +
                    unsupportedFields.sorted().joinToString()
            )
        }
        val model = root["model"]
        if (model != null && (model !is JsonPrimitive || !model.isString)) {
            return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "OpenAI model must be a string when provided"
            )
        }
        val messageArray = root["messages"] as? JsonArray
            ?: return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "OpenAI request must contain a messages array"
            )
        if (messageArray.isEmpty()) {
            return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "OpenAI messages array is empty"
            )
        }

        val messages = mutableListOf<ParsedMessage>()
        for ((index, element) in messageArray.withIndex()) {
            val message = element as? JsonObject
                ?: return BackendResult.Err(
                    QuickAiError.INVALID_PARAMETER,
                    "messages[$index] must be an object"
                )
            val unsupportedMessageFields = message.keys - setOf("role", "content")
            if (unsupportedMessageFields.isNotEmpty()) {
                return BackendResult.Err(
                    QuickAiError.UNSUPPORTED,
                    "messages[$index] contains unsupported fields: " +
                        unsupportedMessageFields.sorted().joinToString()
                )
            }
            val roleName = (message["role"] as? JsonPrimitive)
                ?.takeIf { it.isString }
                ?.contentOrNull
                ?: return BackendResult.Err(
                    QuickAiError.INVALID_PARAMETER,
                    "messages[$index].role must be a string"
                )
            val role = when (roleName) {
                "system" -> ParsedRole.SYSTEM
                "user" -> ParsedRole.USER
                "assistant" -> ParsedRole.ASSISTANT
                else -> return BackendResult.Err(
                    QuickAiError.UNSUPPORTED,
                    "LiteRT-LM OpenAI adapter does not support role '$roleName'"
                )
            }
            val contents = when (val content = message["content"]) {
                is JsonPrimitive -> {
                    if (!content.isString) {
                        return BackendResult.Err(
                            QuickAiError.INVALID_PARAMETER,
                            "messages[$index].content must be a string"
                        )
                    }
                    listOf(ParsedContent.Text(content.content))
                }
                is JsonArray -> parseContentParts(index, role, content).let { result ->
                    when (result) {
                        is BackendResult.Ok -> result.value
                        is BackendResult.Err -> return result
                    }
                }
                else -> return BackendResult.Err(
                    QuickAiError.INVALID_PARAMETER,
                    "messages[$index].content must be text or a content array"
                )
            }
            messages.add(ParsedMessage(role, contents))
        }

        if (messages.last().role != ParsedRole.USER) {
            return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "The final OpenAI message must have role 'user' for LiteRT-LM"
            )
        }
        return BackendResult.Ok(messages)
    }

    private fun parseContentParts(
        messageIndex: Int,
        role: ParsedRole,
        content: JsonArray
    ): BackendResult<List<ParsedContent>> {
        if (content.isEmpty()) {
            return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "messages[$messageIndex].content must not be empty"
            )
        }
        val result = mutableListOf<ParsedContent>()
        for ((partIndex, partElement) in content.withIndex()) {
            val part = partElement as? JsonObject
                ?: return BackendResult.Err(
                    QuickAiError.INVALID_PARAMETER,
                    "messages[$messageIndex].content[$partIndex] must be an object"
                )
            val path = "messages[$messageIndex].content[$partIndex]"
            val type = (part["type"] as? JsonPrimitive)
                ?.takeIf { it.isString }
                ?.contentOrNull
                ?: return BackendResult.Err(
                    QuickAiError.INVALID_PARAMETER,
                    "$path.type must be a string"
                )
            when (type) {
                "text", "input_text" -> {
                    val extraFields = part.keys - setOf("type", "text")
                    if (extraFields.isNotEmpty()) {
                        return unsupportedFields(path, extraFields)
                    }
                    val value = (part["text"] as? JsonPrimitive)
                        ?.takeIf { it.isString }
                        ?.contentOrNull
                        ?: return BackendResult.Err(
                            QuickAiError.INVALID_PARAMETER,
                            "$path.text must be a string"
                        )
                    result.add(ParsedContent.Text(value))
                }
                "image_url" -> {
                    if (role != ParsedRole.USER) {
                        return BackendResult.Err(
                            QuickAiError.INVALID_PARAMETER,
                            "$path.image_url requires role user"
                        )
                    }
                    val extraFields = part.keys - setOf("type", "image_url")
                    if (extraFields.isNotEmpty()) {
                        return unsupportedFields(path, extraFields)
                    }
                    when (val image = parseImageUrl(part["image_url"], "$path.image_url")) {
                        is BackendResult.Ok -> result.add(image.value)
                        is BackendResult.Err -> return image
                    }
                }
                else -> return BackendResult.Err(
                    QuickAiError.UNSUPPORTED,
                    "LiteRT-LM OpenAI adapter does not support content type '$type'"
                )
            }
        }
        return BackendResult.Ok(result)
    }

    private fun parseImageUrl(
        imageUrl: JsonElement?,
        path: String
    ): BackendResult<ParsedContent> {
        val source = when (imageUrl) {
            is JsonPrimitive -> imageUrl.takeIf { it.isString }?.contentOrNull
                ?: return BackendResult.Err(
                    QuickAiError.INVALID_PARAMETER,
                    "$path must be a string or object"
                )
            is JsonObject -> {
                val extraFields = imageUrl.keys - setOf("url", "detail")
                if (extraFields.isNotEmpty()) {
                    return unsupportedFields(path, extraFields)
                }
                val detail = imageUrl["detail"]
                if (detail != null) {
                    val value = (detail as? JsonPrimitive)
                        ?.takeIf { it.isString }
                        ?.contentOrNull
                        ?: return BackendResult.Err(
                            QuickAiError.INVALID_PARAMETER,
                            "$path.detail must be a string"
                        )
                    if (value != "auto") {
                        return BackendResult.Err(
                            QuickAiError.UNSUPPORTED,
                            "LiteRT-LM cannot honor image detail '$value'"
                        )
                    }
                }
                (imageUrl["url"] as? JsonPrimitive)
                    ?.takeIf { it.isString }
                    ?.contentOrNull
                    ?: return BackendResult.Err(
                        QuickAiError.INVALID_PARAMETER,
                        "$path.url must be a string"
                    )
            }
            else -> return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "$path must be a string or object"
            )
        }
        if (source.isBlank()) {
            return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "$path URL must not be blank"
            )
        }

        return when (source.substringBefore(':', missingDelimiterValue = "").lowercase()) {
            "data" -> parseDataImage(source, path)
            "file" -> parseFileImage(source, path)
            else -> BackendResult.Err(
                QuickAiError.UNSUPPORTED,
                "LiteRT-LM only resolves data:image/...;base64 and file:// image URLs"
            )
        }
    }

    private fun parseDataImage(
        source: String,
        path: String
    ): BackendResult<ParsedContent> {
        val commaIndex = source.indexOf(',')
        if (commaIndex <= "data:".length || commaIndex == source.lastIndex) {
            return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "$path contains a malformed data URL"
            )
        }
        val metadata = source.substring("data:".length, commaIndex).split(';')
        if (!metadata.first().startsWith("image/", ignoreCase = true)) {
            return BackendResult.Err(
                QuickAiError.UNSUPPORTED,
                "$path data URL must contain an image media type"
            )
        }
        if (metadata.drop(1).none { it.equals("base64", ignoreCase = true) }) {
            return BackendResult.Err(
                QuickAiError.UNSUPPORTED,
                "$path data image must use base64 encoding"
            )
        }
        val bytes = try {
            Base64.getDecoder().decode(source.substring(commaIndex + 1))
        } catch (t: IllegalArgumentException) {
            return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "$path contains invalid base64 image data"
            )
        }
        if (bytes.isEmpty()) {
            return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "$path decoded image must not be empty"
            )
        }
        return BackendResult.Ok(ParsedContent.ImageBytes(source, bytes))
    }

    private fun parseFileImage(
        source: String,
        path: String
    ): BackendResult<ParsedContent> {
        val file = try {
            val uri = URI(source)
            if (!uri.isAbsolute || uri.isOpaque || uri.rawQuery != null ||
                uri.rawFragment != null
            ) {
                return BackendResult.Err(
                    QuickAiError.INVALID_PARAMETER,
                    "$path must be an absolute file:// URL without query or fragment"
                )
            }
            File(uri).canonicalFile
        } catch (t: Exception) {
            return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "$path contains an invalid file URL: ${t.message}"
            )
        }
        if (!file.isFile || !file.canRead()) {
            return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "$path does not identify a readable file"
            )
        }
        return BackendResult.Ok(ParsedContent.ImageFile(source, file.absolutePath))
    }

    private fun unsupportedFields(
        path: String,
        fields: Set<String>
    ): BackendResult.Err = BackendResult.Err(
        QuickAiError.UNSUPPORTED,
        "$path contains unsupported fields: ${fields.sorted().joinToString()}"
    )

    private fun toLiteRtMessage(message: ParsedMessage): Message {
        val contents = toLiteRtContents(message.contents)
        return when (message.role) {
            ParsedRole.SYSTEM -> Message.system(contents)
            ParsedRole.USER -> Message.user(contents)
            ParsedRole.ASSISTANT -> Message.model(contents = contents)
        }
    }

    private fun toLiteRtContents(contents: List<ParsedContent>): Contents =
        Contents.of(contents.map { content ->
            when (content) {
                is ParsedContent.Text -> Content.Text(content.text)
                is ParsedContent.ImageBytes -> Content.ImageBytes(content.bytes)
                is ParsedContent.ImageFile -> Content.ImageFile(content.absolutePath)
            }
        })

    override fun unload(): BackendResult<Unit> {
        cancel()
        closeEngine()
        return BackendResult.Ok(Unit)
    }

    override fun metrics(): BackendResult<PerformanceMetrics> {
        if (engine == null) {
            return BackendResult.Err(
                QuickAiError.NOT_INITIALIZED,
                "LiteRTLm has not been loaded yet"
            )
        }
        return BackendResult.Ok(
            PerformanceMetrics(
                initializationDurationMs = initializationDurationMs,
                totalDurationMs = lastRunDurationMs
            )
        )
    }

    override fun cancel() {
        val conversation = synchronized(conversationLock) {
            cancellationEpoch.incrementAndGet()
            currentConversation
        }
        try {
            conversation?.cancelProcess()
        } catch (t: Throwable) {
            Log.w(TAG, "LiteRT-LM cancellation failed", t)
        }
    }

    override fun close() {
        cancel()
        closeEngine()
    }

    private fun closeEngine() {
        val conversation = synchronized(conversationLock) {
            val active = currentConversation
            currentConversation = null
            active
        }
        try {
            conversation?.close()
        } catch (t: Throwable) {
            Log.w(TAG, "Active LiteRT-LM conversation failed to close", t)
        }
        try {
            engine?.close()
        } catch (t: Throwable) {
            Log.w(TAG, "LiteRT-LM engine.close() threw", t)
        }
        engine = null
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

    private fun mapBackend(backend: BackendType): LlmBackend = when (backend) {
        BackendType.CPU -> LlmBackend.CPU()
        BackendType.GPU -> LlmBackend.GPU()
        BackendType.NPU -> error("NPU must be rejected before backend mapping")
    }

    private fun elapsedMs(startNs: Long): Double =
        (System.nanoTime() - startNs) / 1_000_000.0

    internal enum class ParsedRole { SYSTEM, USER, ASSISTANT }

    internal sealed interface ParsedContent {
        data class Text(val text: String) : ParsedContent
        data class ImageBytes(
            val source: String,
            val bytes: ByteArray
        ) : ParsedContent
        data class ImageFile(
            val source: String,
            val absolutePath: String
        ) : ParsedContent
    }

    internal data class ParsedMessage(
        val role: ParsedRole,
        val contents: List<ParsedContent>
    )

    companion object {
        private const val TAG = "LiteRTLm"
        private const val THOUGHT_CHANNEL_NAME = "thought"
        private const val STREAM_TIMEOUT_MINUTES = 5L
        private const val CANCELLATION_POLL_MILLIS = 50L
    }
}

/**
 * Arbitrates the single terminal event for one LiteRT-LM streaming request.
 *
 * A producer must claim terminal ownership before it records a result or calls
 * the sink. Losers wait for the owner to finish before reading [result], so a
 * returned result cannot disagree with the terminal event that won the race.
 */
internal class LiteRtStreamTerminal(
    private val sink: StreamSink,
    private val emitError: (QuickAiError, String?) -> Unit
) {
    private val claimed = AtomicBoolean(false)
    private val completed = CountDownLatch(1)
    private val terminalError = AtomicReference<BackendResult.Err?>(null)
    private val deliveryLock = Any()

    fun tryDeliver(
        delivery: () -> Unit,
        errorFrom: (Throwable) -> BackendResult.Err,
        afterFailure: () -> Unit = {}
    ): Boolean {
        var deliveryError: BackendResult.Err? = null
        synchronized(deliveryLock) {
            if (claimed.get()) return false
            try {
                delivery()
            } catch (throwable: Throwable) {
                claimed.set(true)
                deliveryError = try {
                    errorFrom(throwable)
                } catch (errorFactoryFailure: Throwable) {
                    BackendResult.Err(
                        QuickAiError.INFERENCE_FAILED,
                        "Stream failure handler failed: ${errorFactoryFailure.message}"
                    )
                }
                terminalError.set(deliveryError)
            }
        }
        val error = deliveryError
        if (error != null) {
            try {
                try {
                    emitError(error.error, error.message)
                } catch (errorSinkFailure: Throwable) {
                    terminalError.set(
                        BackendResult.Err(
                            QuickAiError.INFERENCE_FAILED,
                            "StreamSink.onError failed: ${errorSinkFailure.message}"
                        )
                    )
                }
                afterFailure()
            } finally {
                completed.countDown()
            }
        }
        return error == null
    }

    fun tryCompleteDone(): Boolean {
        val ownsTerminal = synchronized(deliveryLock) {
            claimed.compareAndSet(false, true)
        }
        if (!ownsTerminal) return false
        try {
            sink.onDone()
        } catch (throwable: Throwable) {
            terminalError.set(
                BackendResult.Err(
                    QuickAiError.INFERENCE_FAILED,
                    "StreamSink.onDone failed: ${throwable.message}"
                )
            )
        } finally {
            completed.countDown()
        }
        return true
    }

    fun tryCompleteError(
        error: BackendResult.Err,
        afterEmit: () -> Unit = {}
    ): Boolean {
        val ownsTerminal = synchronized(deliveryLock) {
            if (!claimed.compareAndSet(false, true)) return@synchronized false
            terminalError.set(error)
            true
        }
        if (!ownsTerminal) return false
        try {
            try {
                emitError(error.error, error.message)
            } catch (errorSinkFailure: Throwable) {
                terminalError.set(
                    BackendResult.Err(
                        QuickAiError.INFERENCE_FAILED,
                        "StreamSink.onError failed: ${errorSinkFailure.message}"
                    )
                )
            }
            afterEmit()
        } finally {
            completed.countDown()
        }
        return true
    }

    fun await(timeout: Long, unit: TimeUnit): Boolean =
        completed.await(timeout, unit)

    fun awaitCompletion() {
        var interrupted = false
        while (completed.count != 0L) {
            try {
                completed.await()
            } catch (_: InterruptedException) {
                interrupted = true
            }
        }
        if (interrupted) Thread.currentThread().interrupt()
    }

    fun result(): BackendResult<Unit> =
        terminalError.get() ?: BackendResult.Ok(Unit)
}
