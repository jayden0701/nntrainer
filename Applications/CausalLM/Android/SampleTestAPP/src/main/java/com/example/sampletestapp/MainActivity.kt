// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    MainActivity.kt
 * @brief   Standalone sample that drives the :QuickDotAI AAR directly —
 *          no QuickAIService, no REST, no remote process.
 *
 * Engine wiring:
 *
 *  1. Creates the catalog-selected [QuickDotAI] implementation and keeps all
 *     calls touching it on a single worker thread (the interface is not
 *     internally thread-safe).
 *  2. Calls [QuickDotAI.load] once per chosen (model, quant) pair. For
 *     GEMMA4 it auto-populates [LoadModelRequest.visionBackend] so the
 *     multimodal path is armed from load time.
 *  3. Uses [QuickDotAI.runText] for exact raw prompts and
 *     [QuickDotAI.runOpenAI] for OpenAI-compatible chat/tool/multimodal
 *     requests. Native models receive preprocessed tensor sidecars; LiteRT-LM
 *     receives OpenAI data-image URLs that it maps to image bytes.
 *
 * UI (M3 Expressive redesign — see Applications/QuickAI/QuickDotAI/QuickDotAI.html
 * design bundle): the screen is rebuilt as a tabbed Material 3 interface
 * with a custom top bar, hero status pill, collapsible Model section
 * with chip-group backend / quantization pickers, and a terminal-styled
 * output panel shared across the Run / Chat / OpenAI tabs. A light/dark
 * toggle in the top bar swaps the full M3 token set at runtime.
 */
package com.example.sampletestapp

import android.content.res.ColorStateList
import android.graphics.Color
import android.graphics.PorterDuff
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.RippleDrawable
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.text.Editable
import android.text.InputType
import android.text.TextWatcher
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.PopupMenu
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.widget.NestedScrollView
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import com.example.quickdotai.BackendResult
import com.example.quickdotai.BackendType
import com.example.quickdotai.Capability
import com.example.quickdotai.LoadModelRequest
import com.example.quickdotai.ModelCatalog
import com.example.quickdotai.ModelDescriptor
import com.example.quickdotai.ModelIds
import com.example.quickdotai.LlavaNextImagePreprocessor
import com.example.quickdotai.OpenAIImageTensor
import com.example.quickdotai.OpenAIImageTensorSidecar
import com.example.quickdotai.OpenAIRequest
import com.example.quickdotai.RuntimeKind
import com.example.quickdotai.createEngine
import com.example.quickdotai.PerformanceMetrics
import com.example.quickdotai.QuantizationType
import com.example.quickdotai.QuickAiError
import com.example.quickdotai.QuickDotAI
import com.example.quickdotai.StreamSink
import java.io.File
import java.util.Base64
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull

/* ────────────────────────────────────────────────────────────────────────
 * M3 Expressive token set — mirrors the LIGHT / DARK objects defined in
 * the QuickDotAI.html design bundle. Holding both palettes in a single
 * data class lets us swap them atomically when the user taps the dark-
 * mode toggle in the top bar.
 * ──────────────────────────────────────────────────────────────────────── */
private data class M3Tokens(
    val bg: Int, val surface: Int, val surfaceDim: Int,
    val surfaceContainer: Int, val surfaceContainerHigh: Int,
    val outline: Int, val outlineVariant: Int,
    val onSurface: Int, val onSurfaceVar: Int,
    val primary: Int, val onPrimary: Int,
    val primaryContainer: Int, val onPrimaryContainer: Int,
    val secondary: Int, val secondaryContainer: Int,
    val tertiary: Int, val tertiaryContainer: Int,
    val error: Int, val errorContainer: Int,
    val success: Int, val successContainer: Int,
    val codeBg: Int, val codeFg: Int,
)

private val LIGHT = M3Tokens(
    bg = 0xFFFBF8FF.toInt(),
    surface = 0xFFFFFFFF.toInt(),
    surfaceDim = 0xFFF2EEF8.toInt(),
    surfaceContainer = 0xFFF4EFFA.toInt(),
    surfaceContainerHigh = 0xFFEDE7F6.toInt(),
    outline = 0xFFCAC4D0.toInt(),
    outlineVariant = 0xFFE7E0EC.toInt(),
    onSurface = 0xFF1C1B1F.toInt(),
    onSurfaceVar = 0xFF49454F.toInt(),
    primary = 0xFF5B3EBE.toInt(),
    onPrimary = 0xFFFFFFFF.toInt(),
    primaryContainer = 0xFFE9DDFF.toInt(),
    onPrimaryContainer = 0xFF21005D.toInt(),
    secondary = 0xFF625B71.toInt(),
    secondaryContainer = 0xFFE8DEF8.toInt(),
    tertiary = 0xFF7D5260.toInt(),
    tertiaryContainer = 0xFFFFD8E4.toInt(),
    error = 0xFFB3261E.toInt(),
    errorContainer = 0xFFF9DEDC.toInt(),
    success = 0xFF146C2E.toInt(),
    successContainer = 0xFFD5F5DF.toInt(),
    codeBg = 0xFF0F0B1E.toInt(),
    codeFg = 0xFFEDE7F6.toInt(),
)

private val DARK = M3Tokens(
    bg = 0xFF121019.toInt(),
    surface = 0xFF1B1823.toInt(),
    surfaceDim = 0xFF100E17.toInt(),
    surfaceContainer = 0xFF211E2B.toInt(),
    surfaceContainerHigh = 0xFF2B2834.toInt(),
    outline = 0xFF4A4458.toInt(),
    outlineVariant = 0xFF2D2A37.toInt(),
    onSurface = 0xFFE6E0E9.toInt(),
    onSurfaceVar = 0xFFCAC4D0.toInt(),
    primary = 0xFFCFBCFF.toInt(),
    onPrimary = 0xFF371E73.toInt(),
    primaryContainer = 0xFF4A3A8C.toInt(),
    onPrimaryContainer = 0xFFE9DDFF.toInt(),
    secondary = 0xFFCCC2DC.toInt(),
    secondaryContainer = 0xFF4A4458.toInt(),
    tertiary = 0xFFEFB8C8.toInt(),
    tertiaryContainer = 0xFF633B48.toInt(),
    error = 0xFFF2B8B5.toInt(),
    errorContainer = 0xFF601410.toInt(),
    success = 0xFF6EDB88.toInt(),
    successContainer = 0xFF124F24.toInt(),
    codeBg = 0xFF06040F.toInt(),
    codeFg = 0xFFCFBCFF.toInt(),
)

private data class OpenAIPreviewMessage(val role: String, val text: String)

private data class LocalChatMessage(
    val role: String,
    val text: String,
    val imageSources: List<String> = emptyList(),
    val imageTensors: List<OpenAIImageTensor> = emptyList(),
)

private data class PreparedOpenAIImages(
    val sources: List<String>,
    val tensors: List<OpenAIImageTensor>,
)

private data class ActiveChatSnapshot(
    val generation: Long,
    val key: String,
    val history: List<LocalChatMessage>,
    val nextImageIndex: Int,
)

private data class ChatUiState(
    val active: Boolean,
    val assistantTurns: Int,
)

class MainActivity : AppCompatActivity() {

    /* ───── Engine plumbing ───── */

    private val engineExecutor: Executor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "SampleTestAPP-Engine").apply { isDaemon = true }
    }

    private val mainHandler by lazy { android.os.Handler(mainLooper) }

    @Volatile private var engine: QuickDotAI? = null
    @Volatile private var loadedKey: String? = null
    @Volatile private var selectedImageBytesList: List<ByteArray> = emptyList()
    private val imagePreprocessor by lazy { LlavaNextImagePreprocessor() }
    private val chatStateLock = Any()
    private val chatHistory = mutableListOf<LocalChatMessage>()
    private var chatGeneration = 0L
    private var nextChatImageIndex = 0

    /** Convenience: first selected image (or null) — for single-image code paths. */
    private val selectedImageBytes: ByteArray?
        get() = selectedImageBytesList.firstOrNull()

    /* ───── UI state (preserved across light/dark theme rebuilds) ───── */

    private var darkMode = false
    private var selectedTab: String = "openai"          // chat | openai | metrics
    private var modelExpanded = true
    private var samplingExpanded = false

    private var selFamily: String = ModelIds.GEMMA4
    private var selRuntime: RuntimeKind = RuntimeKind.NATIVE
    private var selBackend: BackendType = BackendType.NPU
    private var selSdEnabled: Boolean = false
    private val selDescriptor: ModelDescriptor?
        get() = ModelCatalog.resolve(selFamily, selRuntime, selBackend)
    private var selectedQuant: QuantizationType = QuantizationType.W4A32

    private var chatSelFamily: String = ModelIds.GEMMA4
    private var chatSelRuntime: RuntimeKind = RuntimeKind.NATIVE
    private var chatSelBackend: BackendType = BackendType.NPU
    private var chatSelSdEnabled: Boolean = false
    private val chatSelDescriptor: ModelDescriptor?
        get() = ModelCatalog.resolve(chatSelFamily, chatSelRuntime, chatSelBackend)
    private var chatSelectedQuant: QuantizationType = QuantizationType.W4A32

    private var modelBasePathText: String = "/sdcard/Download/aistudio-mobile/models/"
    private var modelPathText: String = ""
    private var promptText: String = "What is rainbow?"

    private var systemPromptText: String = ""
    private var chatPromptText: String = "I bought a red car"
    private var openAiJsonText: String = """{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "system", "content": "Answer in one short sentence."},
    {"role": "user", "content": "Write a short joke about saving RAM."}
  ]
}"""

    private var statusText: String = "Idle."
    private var outputText: String = ""
    @Volatile private var streaming: Boolean = false
    private var loadStatus: String = "idle"             // idle | loading | loaded
    private var loadedLabel: String = ""
    private var chatActive: Boolean = false
    private var activeChatKey: String? = null
    private var lastMetrics: PerformanceMetrics? = null

    private var mainScrollY = 0
    private var outputScrollY = 0

    /* ───── UI refs (re-wired on every rebuildUi() call) ───── */

    private lateinit var rootHost: FrameLayout
    private lateinit var mainScrollView: NestedScrollView
    private lateinit var outputScrollView: NestedScrollView
    private lateinit var statusView: TextView
    private lateinit var outputView: TextView
    private lateinit var modelBasePathField: EditText
    private lateinit var modelPathField: EditText
    private lateinit var promptField: EditText
    private lateinit var imageStatusView: TextView
    private lateinit var chatSystemPromptField: EditText
    private lateinit var chatPromptField: EditText
    private lateinit var openAIMessagesField: EditText
    private lateinit var chatModelBasePathField: EditText
    private lateinit var chatStatusView: TextView

    /**
     * @brief ActivityResult launcher for the Android system photo picker
     * (single image). Uses [ActivityResultContracts.PickVisualMedia] which
     * does NOT require any runtime permissions.
     */
    private val imagePickerLauncher = registerForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri: Uri? ->
        if (uri == null) {
            setStatus("Image pick cancelled.")
            return@registerForActivityResult
        }
        selectedImageBytesList = emptyList()
        readImageBytesAsync(listOf(uri))
    }

    /**
     * @brief ActivityResult launcher for multi-image photo picker.
     * Uses [ActivityResultContracts.PickMultipleVisualMedia] which allows
     * selecting multiple images for V-JEPA multi-image inference.
     */
    private val multiImagePickerLauncher = registerForActivityResult(
        ActivityResultContracts.PickMultipleVisualMedia()
    ) { uris: List<Uri> ->
        if (uris.isEmpty()) {
            setStatus("Image pick cancelled.")
            return@registerForActivityResult
        }
        selectedImageBytesList = emptyList()
        readImageBytesAsync(uris)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        checkAllFilesAccess()
        // Seed the path field so a single Load tap works without typing.
        modelPathText = defaultModelPathFor(selDescriptor) ?: ""
        rebuildUi()
    }

    /**
     * @brief Tear down and re-inflate the entire view tree using the
     * current [darkMode] palette. EditText contents are preserved by
     * round-tripping through the `*Text` state vars, which are kept in
     * sync via [TextWatcher]s installed in the field builders below.
     */
    private fun rebuildUi(resetModelPath: Boolean = false) {
        val tokens = if (darkMode) DARK else LIGHT
        // Snapshot any in-flight EditText contents into state vars so the
        // theme rebuild does not lose typed input.
        if (::promptField.isInitialized) promptText = promptField.text.toString()
        if (::modelBasePathField.isInitialized) modelBasePathText = modelBasePathField.text.toString()
        if (::chatModelBasePathField.isInitialized) modelBasePathText = chatModelBasePathField.text.toString()
        if (::chatSystemPromptField.isInitialized) systemPromptText = chatSystemPromptField.text.toString()
        if (::chatPromptField.isInitialized) chatPromptText = chatPromptField.text.toString()
        if (::openAIMessagesField.isInitialized) openAiJsonText = openAIMessagesField.text.toString()

        // Save scroll positions before rebuilding
        if (::mainScrollView.isInitialized) mainScrollY = mainScrollView.scrollY
        if (::outputScrollView.isInitialized) outputScrollY = outputScrollView.scrollY

        rootHost = FrameLayout(this).apply {
            layoutParams = ViewGroup.LayoutParams(MATCH_PARENT, MATCH_PARENT)
            setBackgroundColor(tokens.bg)
            fitsSystemWindows = true
        }
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT)
        }
        rootHost.addView(column)

        column.addView(buildTopBar(tokens))
        column.addView(buildHeroCard(tokens))
        column.addView(buildTabBar(tokens))

        // Main scrolling content area — wraps the model section, the
        // active tab body, and (for non-metrics tabs) the shared output
        // panel.
        mainScrollView = NestedScrollView(this).apply {
            isFillViewport = false
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, 0, 1f)
            overScrollMode = View.OVER_SCROLL_NEVER
        }
        val scrollColumn = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), 0, dp(12), dp(120))
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        mainScrollView.addView(scrollColumn)
        column.addView(mainScrollView)

        if (selectedTab != "chat") {
            scrollColumn.addView(buildModelSection(tokens))
            spacer(scrollColumn, 10)
        }

        scrollColumn.addView(when (selectedTab) {
            "chat"    -> buildChatTab(tokens)
            "openai"  -> buildOpenAiTab(tokens)
            "metrics" -> buildMetricsTab(tokens)
            else      -> buildOpenAiTab(tokens)
        })

        if (selectedTab != "metrics") {
            spacer(scrollColumn, 10)
            scrollColumn.addView(buildOutputPanel(tokens))
        }

        setContentView(rootHost)

        // Restore scroll positions after UI is built
        mainScrollView.post { mainScrollView.scrollTo(0, mainScrollY) }
        if (::outputScrollView.isInitialized) {
            outputScrollView.post { outputScrollView.scrollTo(0, outputScrollY) }
        }
    }

    /* ════════════════════════════════════════════════════════════════
     * Component builders
     * ════════════════════════════════════════════════════════════════ */

    private fun buildTopBar(t: M3Tokens): View {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(16), dp(14), dp(16), dp(10))
        }
        // Brand mark — gradient-filled rounded square with a "✦" glyph,
        // approximating the hero icon in the design.
        val brand = TextView(this).apply {
            text = "✦"
            setTextColor(Color.WHITE)
            textSize = 18f
            typeface = Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
            background = gradient(t.primary, t.tertiary, 12)
            layoutParams = LinearLayout.LayoutParams(dp(40), dp(40))
        }
        row.addView(brand)

        spacerH(row, 12)

        val titleColumn = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
        }
        val title = TextView(this).apply {
            text = "QuickDotAI"
            setTextColor(t.onSurface)
            textSize = 18f
            typeface = Typeface.DEFAULT_BOLD
        }
        val sub = TextView(this).apply {
            val tail = if (loadStatus == "loaded") loadedLabel else "no model"
            val tailColor = if (loadStatus == "loaded") t.success else t.onSurfaceVar
            val full = "In-process AAR  ·  $tail"
            text = full
            textSize = 11f
            // Approximate the React design's two-tone subtitle by using
            // the success color when a model is loaded, otherwise neutral.
            setTextColor(tailColor)
        }
        titleColumn.addView(title)
        titleColumn.addView(sub)
        row.addView(titleColumn)

        // Light/dark toggle button.
        val toggle = TextView(this).apply {
            text = if (darkMode) "☀" else "☾"
            textSize = 16f
            gravity = Gravity.CENTER
            setTextColor(t.onSurfaceVar)
            background = solid(t.surfaceContainer, 20)
            layoutParams = LinearLayout.LayoutParams(dp(40), dp(40))
            setOnClickListener {
                darkMode = !darkMode
                rebuildUi()
            }
        }
        row.addView(toggle)
        return row
    }

    private fun buildHeroCard(t: M3Tokens): View {
        val tone = statusTone()
        val (bgColor, dotColor, fgColor) = when (tone) {
            "error"    -> Triple(t.errorContainer,   t.error,   t.error)
            "success"  -> Triple(t.successContainer, t.success, t.success)
            "progress" -> Triple(t.primaryContainer, t.primary, t.onPrimaryContainer)
            else       -> Triple(t.surfaceContainer, t.outline, t.onSurfaceVar)
        }
        val outer = FrameLayout(this).apply {
            setPadding(dp(12), 0, dp(12), dp(12))
        }
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(14), dp(12), dp(14), dp(12))
            background = solid(bgColor, 20)
            layoutParams = FrameLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        // Status indicator dot.
        val dot = View(this).apply {
            background = circle(dotColor)
            layoutParams = LinearLayout.LayoutParams(dp(10), dp(10))
        }
        card.addView(dot)
        spacerH(card, 10)

        statusView = TextView(this).apply {
            text = statusText
            setTextColor(fgColor)
            textSize = 13f
            typeface = Typeface.MONOSPACE
            ellipsize = android.text.TextUtils.TruncateAt.END
            maxLines = 1
            layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
        }
        card.addView(statusView)

        if (streaming) {
            val stop = TextView(this).apply {
                text = "■  Stop"
                setTextColor(Color.WHITE)
                textSize = 12f
                typeface = Typeface.DEFAULT_BOLD
                background = solid(t.error, 100)
                setPadding(dp(12), dp(6), dp(12), dp(6))
                setOnClickListener { onCancelClicked() }
            }
            card.addView(stop)
        }
        outer.addView(card)
        return outer
    }

    private fun buildTabBar(t: M3Tokens): View {
        val outer = FrameLayout(this).apply {
            setPadding(dp(12), 0, dp(12), dp(10))
        }
        val pill = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            background = solid(t.surfaceContainer, 100)
            setPadding(dp(4), dp(4), dp(4), dp(4))
            layoutParams = FrameLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        val tabs = listOf("chat" to "⌬ Chat",
                          "openai" to "{ } OpenAI", "metrics" to "▤ Metrics")
        for ((idx, entry) in tabs.withIndex()) {
            val (key, label) = entry
            val active = key == selectedTab
            val tab = TextView(this).apply {
                text = label
                gravity = Gravity.CENTER
                textSize = 13f
                typeface = if (active) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
                setTextColor(if (active) t.onPrimary else t.onSurfaceVar)
                background = if (active) solid(t.primary, 100) else null
                setPadding(dp(6), dp(10), dp(6), dp(10))
                layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f).also {
                    if (idx > 0) it.leftMargin = dp(4)
                }
                setOnClickListener {
                    selectedTab = key
                    rebuildUi()
                }
            }
            pill.addView(tab)
        }
        outer.addView(pill)
        return outer
    }

    private fun buildModelSection(t: M3Tokens): View {
        val subtitle = if (loadStatus == "loaded")
            "$loadedLabel  ·  ${selBackend.name}"
        else
            "${selDescriptor?.displayName ?: selFamily}  ·  ${selRuntime.name}  ·  ${selBackend.name}"
        val statusDotColor = when (loadStatus) {
            "loaded"  -> t.success
            "loading" -> t.primary
            else      -> t.outline
        }
        val card = collapsibleCard(
            t = t,
            iconGlyph = "▦",
            iconBg = t.primaryContainer,
            iconFg = t.onPrimaryContainer,
            title = "Model",
            subtitle = subtitle,
            rightAdornment = View(this).apply {
                background = circle(statusDotColor)
                layoutParams = LinearLayout.LayoutParams(dp(10), dp(10)).also {
                    it.rightMargin = dp(4)
                }
            },
            expanded = modelExpanded,
            onToggle = {
                modelExpanded = !modelExpanded
                rebuildUi()
            }
        ) { body ->
            // MODEL select — 3-axis cascading
            body.addView(labelView(t, "FAMILY"))
            body.addView(dropdownField(t, ModelCatalog.selectableFamilies(), selFamily) { picked ->
                selFamily = picked
                selRuntime = ModelCatalog.runtimesFor(selFamily).firstOrNull() ?: selRuntime
                selBackend = ModelCatalog.backendsFor(selFamily, selRuntime).firstOrNull() ?: selBackend
                selSdEnabled = false
                modelPathText = defaultModelPathFor(selDescriptor) ?: ""
                selDescriptor?.id?.let { openAiJsonText = defaultOpenAIExampleForId(it) }
                rebuildUi(resetModelPath = true)
            })
            spacer(body, 12)

            body.addView(labelView(t, "RUNTIME"))
            body.addView(chipRow(t, ModelCatalog.runtimesFor(selFamily).map { it.name }, selRuntime.name) { picked ->
                selRuntime = RuntimeKind.valueOf(picked)
                selBackend = ModelCatalog.backendsFor(selFamily, selRuntime).firstOrNull() ?: selBackend
                selSdEnabled = false
                modelPathText = defaultModelPathFor(selDescriptor) ?: ""
                selDescriptor?.id?.let { openAiJsonText = defaultOpenAIExampleForId(it) }
                rebuildUi(resetModelPath = true)
            })
            spacer(body, 12)

            body.addView(labelView(t, "BACKEND"))
            body.addView(chipRow(t, ModelCatalog.backendsFor(selFamily, selRuntime).map { it.name }, selBackend.name) { picked ->
                selBackend = BackendType.valueOf(picked)
                selSdEnabled = false
                modelPathText = defaultModelPathFor(selDescriptor) ?: ""
                selDescriptor?.id?.let { openAiJsonText = defaultOpenAIExampleForId(it) }
                rebuildUi(resetModelPath = true)
            })
            spacer(body, 12)

            // MODEL BASE PATH — editable root directory for model files.
            body.addView(labelView(t, "MODEL BASE PATH"))
            modelBasePathField = roundedEditText(t, modelBasePathText, mono = true,
                onTextChange = { modelBasePathText = it })
            body.addView(modelBasePathField)
            spacer(body, 12)

            // MODEL NAME — read-only display of default folder name + error if missing.
            body.addView(labelView(t, "MODEL NAME"))
            body.addView(modelNameView(t, selDescriptor))
            spacer(body, 12)

            // Speculative Decoding switch
            spacer(body, 12)
            body.addView(labelView(t, "SPECULATIVE DECODING"))
            val sdSwitch = androidx.appcompat.widget.SwitchCompat(this).apply {
                isChecked = selSdEnabled
                isEnabled = isSdCapableModel()
                alpha = if (isEnabled) 1.0f else 0.38f
                setOnCheckedChangeListener { _, isChecked ->
                    selSdEnabled = isChecked
                }
            }
            body.addView(sdSwitch)
            spacer(body, 12)

            // Load / Unload action row.
            val actions = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
            val loadLabel = if (loadStatus == "loaded") "↻  Reload model" else "↓  Load model"
            actions.addView(filledButton(t, loadLabel, fill = "horizontal") { onLoadClicked() })
            if (loadStatus == "loaded") {
                spacerH(actions, 8)
                actions.addView(tonalButton(t, "✕  Unload", danger = true) { onUnloadClicked() })
            }
            body.addView(actions)
        }
        return card
    }

    private fun buildRunTab(t: M3Tokens): View {
        val card = roundedCard(t, t.surfaceContainer)
        val header = sectionHeader(t, "⚡", t.secondaryContainer, t.onSurface,
            "One-shot run", "Raw prompt · streaming output")
        card.addView(header)
        spacer(card, 14)

        // PROMPT.
        card.addView(labelView(t, "PROMPT"))
        promptField = roundedEditText(t, promptText, multiline = true, mono = true, rows = 5,
            onTextChange = { promptText = it })
        card.addView(promptField)
        spacer(card, 14)

        // IMAGE INPUT.
        val imgLabelRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        imgLabelRow.addView(labelView(t, "IMAGE INPUT").also {
            it.layoutParams = LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT)
        })
        if (selDescriptor?.let { isMultimodal(it) } != true) {
            spacerH(imgLabelRow, 6)
            val badge = TextView(this).apply {
                text = "GEMMA4 only"
                setTextColor(t.tertiary)
                textSize = 10f
                typeface = Typeface.DEFAULT_BOLD
                background = solid(t.tertiaryContainer, 4)
                setPadding(dp(6), dp(1), dp(6), dp(1))
            }
            imgLabelRow.addView(badge)
        }
        card.addView(imgLabelRow)
        spacer(card, 6)

        if (selectedImageBytes != null) {
            val attached = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                background = solid(t.surfaceContainerHigh, 12)
                setPadding(dp(12), dp(12), dp(12), dp(12))
            }
            val thumb = TextView(this).apply {
                text = "🖼"
                setTextColor(t.onSurface)
                textSize = 22f
                gravity = Gravity.CENTER
                background = gradient(
                    blendAlpha(t.primary, 0x55),
                    blendAlpha(t.tertiary, 0x55), 8
                )
                layoutParams = LinearLayout.LayoutParams(dp(48), dp(48))
            }
            attached.addView(thumb)
            spacerH(attached, 10)
            val info = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
            }
            info.addView(TextView(this).apply {
                text = "Selected image"
                setTextColor(t.onSurface)
                textSize = 13f
                typeface = Typeface.DEFAULT_BOLD
            })
            val totalBytes = selectedImageBytesList.sumOf { it.size }
            val modeLabel = if (selectedImageBytesList.size > 1) "  ·  multi-image mode" else ""
            info.addView(TextView(this).apply {
                text = "${totalBytes} bytes  ·  raw bytes ready$modeLabel"
                setTextColor(t.onSurfaceVar)
                textSize = 11f
                typeface = Typeface.MONOSPACE
            })
            attached.addView(info)
            val close = TextView(this).apply {
                text = "✕"
                setTextColor(t.onSurfaceVar)
                gravity = Gravity.CENTER
                textSize = 14f
                layoutParams = LinearLayout.LayoutParams(dp(32), dp(32))
                setOnClickListener { onClearImageClicked(); rebuildUi() }
            }
            attached.addView(close)
            card.addView(attached)
            // Keep the legacy imageStatusView reference happy — it is
            // touched by onClearImageClicked / readImageBytesAsync.
            imageStatusView = TextView(this).apply { visibility = View.GONE }
        } else {
            val pickLabel = if (isMultiImageModel(selDescriptor))
                "+  Pick images for V-JEPA" else "+  Pick image for multimodal run"
            val dropzone = TextView(this).apply {
                text = pickLabel
                setTextColor(t.onSurfaceVar)
                textSize = 13f
                typeface = Typeface.DEFAULT_BOLD
                gravity = Gravity.CENTER
                background = dashedBg(t)
                setPadding(dp(14), dp(14), dp(14), dp(14))
                setOnClickListener { onPickImageClicked() }
            }
            card.addView(dropzone)
            imageStatusView = TextView(this).apply { visibility = View.GONE }
        }
        spacer(card, 14)

        // RUN button.
        val runLabel = when {
            streaming -> "■  Stop streaming"
            selectedImageBytes != null -> "▶  Run multimodal (streaming)"
            else -> "▶  Run (streaming)"
        }
        val runBtn = filledButton(t, runLabel, fill = "vertical",
            danger = streaming) {
            if (streaming) onCancelClicked() else onRunClicked()
        }
        card.addView(runBtn)
        return card
    }

    private fun buildChatTab(t: M3Tokens): View {
        val container = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }

        // ── Model Selection Card ──
        val modelCard = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = strokedSolid(t.surface, 24, t.outlineVariant, 1)
            setPadding(dp(14), dp(14), dp(14), dp(14))
        }
        val chatUiState = snapshotChatUiState(loadedKey)
        val active = chatUiState.active

        // Top row: status icon + session controls.
        val modelHeaderRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val modelIcon = TextView(this).apply {
            text = "▦"
            gravity = Gravity.CENTER
            textSize = 18f
            setTextColor(if (active) t.success else t.onSurfaceVar)
            background = solid(if (active) t.successContainer else t.surfaceContainerHigh, 10)
            layoutParams = LinearLayout.LayoutParams(dp(38), dp(38))
        }
        modelHeaderRow.addView(modelIcon)
        spacerH(modelHeaderRow, 12)
        modelHeaderRow.addView(TextView(this).apply {
            text = chatSelDescriptor?.let { badgeLabel(it) } ?: chatSelFamily
            setTextColor(t.onSurface)
            textSize = 13f
            typeface = Typeface.MONOSPACE
            layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
        })

        if (active) {
            chatStatusView = TextView(this).apply {
                val turnCount = chatUiState.assistantTurns
                text = "$turnCount turn${if (turnCount == 1) "" else "s"}"
                setTextColor(t.success)
                textSize = 12f
                typeface = Typeface.MONOSPACE
            }
            spacerH(modelHeaderRow, 8)
            modelHeaderRow.addView(chatStatusView)
            spacerH(modelHeaderRow, 6)
            modelHeaderRow.addView(tonalButton(t, "✕ Close", danger = true) { onChatCloseClicked() })
        } else {
            spacerH(modelHeaderRow, 8)
            modelHeaderRow.addView(filledButton(t, "+ Open") { onChatOpenClicked() })
        }
        modelCard.addView(modelHeaderRow)
        spacer(modelCard, 12)

        // 3-axis cascading selection.
        modelCard.addView(labelView(t, "FAMILY"))
        modelCard.addView(dropdownField(t, ModelCatalog.selectableFamilies(), chatSelFamily) { picked ->
            chatSelFamily = picked
            chatSelRuntime = ModelCatalog.runtimesFor(chatSelFamily).firstOrNull() ?: chatSelRuntime
            chatSelBackend = ModelCatalog.backendsFor(chatSelFamily, chatSelRuntime).firstOrNull() ?: chatSelBackend
            chatSelSdEnabled = false
            clearChatSessionState()
            rebuildUi()
        })
        spacer(modelCard, 12)

        modelCard.addView(labelView(t, "RUNTIME"))
        modelCard.addView(chipRow(t, ModelCatalog.runtimesFor(chatSelFamily).map { it.name }, chatSelRuntime.name) { picked ->
            chatSelRuntime = RuntimeKind.valueOf(picked)
            chatSelBackend = ModelCatalog.backendsFor(chatSelFamily, chatSelRuntime).firstOrNull() ?: chatSelBackend
            chatSelSdEnabled = false
            clearChatSessionState()
            rebuildUi()
        })
        spacer(modelCard, 12)

        modelCard.addView(labelView(t, "BACKEND"))
        modelCard.addView(chipRow(t, ModelCatalog.backendsFor(chatSelFamily, chatSelRuntime).map { it.name }, chatSelBackend.name) { picked ->
            chatSelBackend = BackendType.valueOf(picked)
            chatSelSdEnabled = false
            clearChatSessionState()
            rebuildUi()
        })
        spacer(modelCard, 12)

        // Speculative Decoding switch
        spacer(modelCard, 12)
        modelCard.addView(labelView(t, "SPECULATIVE DECODING"))
        val chatSdSwitch = androidx.appcompat.widget.SwitchCompat(this).apply {
            isChecked = chatSelSdEnabled
            isEnabled = isChatSdCapableModel()
            alpha = if (isEnabled) 1.0f else 0.38f
            setOnCheckedChangeListener { _, isChecked ->
                chatSelSdEnabled = isChecked
            }
        }
        modelCard.addView(chatSdSwitch)

        // MODEL BASE PATH — editable root directory for model files.
        spacer(modelCard, 12)
        modelCard.addView(labelView(t, "MODEL BASE PATH"))
        chatModelBasePathField = roundedEditText(t, modelBasePathText, mono = true,
            onTextChange = { modelBasePathText = it })
        modelCard.addView(chatModelBasePathField)
        spacer(modelCard, 12)

        // MODEL NAME — read-only display of default folder name + error if missing.
        modelCard.addView(labelView(t, "MODEL NAME"))
        modelCard.addView(modelNameView(t, chatSelDescriptor))
        container.addView(modelCard)
        spacer(container, 10)

        val chatImageCard = roundedCard(t, t.surfaceContainer)
        val imgSubtitle = if (isMultiImageModel(chatSelDescriptor))
            "Select multiple images for V-JEPA" else "Attach one image to the next chat message"
        chatImageCard.addView(sectionHeader(t, "[ ]", t.secondaryContainer, t.onSurface,
            "Image input", imgSubtitle))
        spacer(chatImageCard, 12)

        val chatImgLabelRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        chatImgLabelRow.addView(labelView(t, "IMAGE INPUT").also {
            it.layoutParams = LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT)
        })
        if (chatSelDescriptor?.let { isMultimodal(it) } != true) {
            spacerH(chatImgLabelRow, 6)
            val badge = TextView(this).apply {
                text = "Vision model only"
                setTextColor(t.tertiary)
                textSize = 10f
                typeface = Typeface.DEFAULT_BOLD
                background = solid(t.tertiaryContainer, 4)
                setPadding(dp(6), dp(1), dp(6), dp(1))
            }
            chatImgLabelRow.addView(badge)
        }
        chatImageCard.addView(chatImgLabelRow)
        spacer(chatImageCard, 6)

        if (selectedImageBytesList.isNotEmpty()) {
            val attached = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                background = solid(t.surfaceContainerHigh, 12)
                setPadding(dp(12), dp(12), dp(12), dp(12))
            }
            val thumb = TextView(this).apply {
                text = if (selectedImageBytesList.size > 1) "[${selectedImageBytesList.size}]" else "[ ]"
                setTextColor(t.onSurface)
                textSize = if (selectedImageBytesList.size > 1) 16f else 22f
                gravity = Gravity.CENTER
                background = gradient(
                    blendAlpha(t.primary, 0x55),
                    blendAlpha(t.tertiary, 0x55), 8
                )
                layoutParams = LinearLayout.LayoutParams(dp(48), dp(48))
            }
            attached.addView(thumb)
            spacerH(attached, 10)
            val info = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
            }
            val totalBytes = selectedImageBytesList.sumOf { it.size }
            info.addView(TextView(this).apply {
                text = if (selectedImageBytesList.size == 1) "Selected image" else "${selectedImageBytesList.size} images selected"
                setTextColor(t.onSurface)
                textSize = 13f
                typeface = Typeface.DEFAULT_BOLD
            })
            info.addView(TextView(this).apply {
                val modeLabel = if (selectedImageBytesList.size > 1) "  ·  multi-image mode" else ""
                text = "${totalBytes} bytes  raw bytes ready$modeLabel"
                setTextColor(t.onSurfaceVar)
                textSize = 11f
                typeface = Typeface.MONOSPACE
            })
            attached.addView(info)
            val close = TextView(this).apply {
                text = "x"
                setTextColor(t.onSurfaceVar)
                gravity = Gravity.CENTER
                textSize = 14f
                layoutParams = LinearLayout.LayoutParams(dp(32), dp(32))
                setOnClickListener { onClearImageClicked(); rebuildUi() }
            }
            attached.addView(close)
            chatImageCard.addView(attached)
            imageStatusView = TextView(this).apply { visibility = View.GONE }
        } else {
            val pickLabel = if (isMultiImageModel(chatSelDescriptor))
                "+  Pick images for V-JEPA" else "+  Pick image for multimodal chat"
            val dropzone = TextView(this).apply {
                text = pickLabel
                setTextColor(t.onSurfaceVar)
                textSize = 13f
                typeface = Typeface.DEFAULT_BOLD
                gravity = Gravity.CENTER
                background = dashedBg(t)
                setPadding(dp(14), dp(14), dp(14), dp(14))
                setOnClickListener { onPickImageClicked() }
            }
            chatImageCard.addView(dropzone)
            imageStatusView = TextView(this).apply { visibility = View.GONE }
        }
        container.addView(chatImageCard)
        spacer(container, 10)

        // ── Collapsible request config ──
        val configCard = collapsibleCard(
            t = t,
            iconGlyph = "⚙",
            iconBg = t.tertiaryContainer,
            iconFg = t.tertiary,
            title = "Chat request config",
            subtitle = "System prompt",
            rightAdornment = null,
            expanded = samplingExpanded,
            onToggle = {
                samplingExpanded = !samplingExpanded
                rebuildUi()
            }
        ) { body ->
            body.addView(labelView(t, "SYSTEM PROMPT"))
            chatSystemPromptField = roundedEditText(t, systemPromptText,
                multiline = true, rows = 2,
                placeholder = "You are a helpful assistant.",
                onTextChange = { systemPromptText = it })
            body.addView(chatSystemPromptField)
        }
        container.addView(configCard)
        spacer(container, 10)

        // ── Composer ──
        val composer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = solid(t.surfaceContainer, 20)
            setPadding(dp(14), dp(14), dp(14), dp(14))
        }
        composer.addView(labelView(t, "CHAT MESSAGE"))
        spacer(composer, 6)
        val composerRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = solid(t.surfaceContainerHigh, 24)
            setPadding(dp(6), dp(6), dp(6), dp(6))
        }
        chatPromptField = EditText(this).apply {
            setText(chatPromptText)
            hint = "Type a chat message…"
            setHintTextColor(t.onSurfaceVar)
            setTextColor(t.onSurface)
            textSize = 14f
            background = null
            minLines = 2
            maxLines = 6
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            setPadding(dp(14), dp(10), dp(14), dp(10))
            layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
            addTextChangedListener(simpleWatcher { chatPromptText = it })
        }
        composerRow.addView(chatPromptField)

        val canSend = snapshotChatUiState(loadedKey).active && !streaming
        val sendFab = TextView(this).apply {
            text = "▶"
            gravity = Gravity.CENTER
            textSize = 18f
            setTextColor(if (canSend) t.onPrimary else t.onSurfaceVar)
            background = solid(if (canSend) t.primary else t.outlineVariant, 22)
            layoutParams = LinearLayout.LayoutParams(dp(44), dp(44)).also {
                it.bottomMargin = dp(2); it.rightMargin = dp(2)
            }
            isEnabled = canSend
            setOnClickListener { onChatRunStreamingClicked() }
        }
        composerRow.addView(sendFab)
        composer.addView(composerRow)
        container.addView(composer)
        return container
    }

    private fun buildOpenAiTab(t: M3Tokens): View {
        val card = roundedCard(t, t.surfaceContainer)
        card.addView(sectionHeader(t, "{ }", t.secondaryContainer, t.onSurface,
            "OpenAI request",
            "Full request object · forwarded without dropping fields"))
        spacer(card, 12)

        // Parsed preview.
        var parseErr: String? = null
        val parsed: List<OpenAIPreviewMessage>? = try {
            parseOpenAIPreview(openAiJsonText)
        } catch (e: Throwable) { parseErr = e.message; null }

        if (parsed != null) {
            val previewWrap = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                background = solid(t.surfaceContainerHigh, 16)
                setPadding(dp(10), dp(10), dp(10), dp(10))
            }
            for (msg in parsed) {
                val row = LinearLayout(this).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.TOP
                    setPadding(0, dp(3), 0, dp(3))
                }
                val (badgeBg, badgeFg, badgeLabel) = when (msg.role.lowercase()) {
                    "system", "developer" ->
                        Triple(t.tertiaryContainer, t.tertiary, msg.role.uppercase())
                    "user" -> Triple(t.primaryContainer, t.onPrimaryContainer, "USER")
                    "assistant" -> Triple(t.secondaryContainer, t.onSurface, "ASSISTANT")
                    else -> Triple(t.surfaceContainer, t.onSurfaceVar, msg.role.uppercase())
                }
                val badge = TextView(this).apply {
                    text = badgeLabel
                    setTextColor(badgeFg)
                    textSize = 10f
                    typeface = Typeface.MONOSPACE
                    gravity = Gravity.CENTER
                    background = solid(badgeBg, 100)
                    setPadding(dp(8), dp(2), dp(8), dp(2))
                    layoutParams = LinearLayout.LayoutParams(dp(80), WRAP_CONTENT)
                }
                row.addView(badge)
                spacerH(row, 8)
                val content = TextView(this).apply {
                    text = msg.text
                    setTextColor(t.onSurface)
                    textSize = 13f
                    layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
                }
                row.addView(content)
                previewWrap.addView(row)
            }
            card.addView(previewWrap)
            spacer(card, 12)
        }
        if (parseErr != null) {
            val errBox = TextView(this).apply {
                text = "ⓘ  $parseErr"
                setTextColor(t.error)
                textSize = 12f
                typeface = Typeface.MONOSPACE
                background = solid(t.errorContainer, 10)
                setPadding(dp(12), dp(8), dp(12), dp(8))
            }
            card.addView(errBox)
            spacer(card, 12)
        }

        val openAiImageCard = roundedCard(t, t.surfaceContainerHigh)
        openAiImageCard.addView(sectionHeader(t, "[ ]", t.secondaryContainer, t.onSurface,
            "Image input", "Attach the selected image to the last user message"))
        spacer(openAiImageCard, 12)

        val imageLabelRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        imageLabelRow.addView(labelView(t, "IMAGE INPUT").also {
            it.layoutParams = LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT)
        })
        if (selDescriptor?.let { isMultimodal(it) } != true) {
            spacerH(imageLabelRow, 6)
            imageLabelRow.addView(TextView(this).apply {
                text = "Vision model only"
                setTextColor(t.tertiary)
                textSize = 10f
                typeface = Typeface.DEFAULT_BOLD
                background = solid(t.tertiaryContainer, 4)
                setPadding(dp(6), dp(1), dp(6), dp(1))
            })
        }
        openAiImageCard.addView(imageLabelRow)
        spacer(openAiImageCard, 6)

        if (selectedImageBytesList.isNotEmpty()) {
            val attached = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                background = solid(t.surfaceContainer, 12)
                setPadding(dp(12), dp(12), dp(12), dp(12))
            }
            attached.addView(TextView(this).apply {
                text = if (selectedImageBytesList.size > 1) "[${selectedImageBytesList.size}]" else "[ ]"
                setTextColor(t.onSurface)
                textSize = if (selectedImageBytesList.size > 1) 16f else 22f
                gravity = Gravity.CENTER
                background = gradient(
                    blendAlpha(t.primary, 0x55),
                    blendAlpha(t.tertiary, 0x55), 8
                )
                layoutParams = LinearLayout.LayoutParams(dp(48), dp(48))
            })
            spacerH(attached, 10)
            val info = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
            }
            val totalBytes = selectedImageBytesList.sumOf { it.size }
            info.addView(TextView(this).apply {
                text = if (selectedImageBytesList.size == 1) "Selected image" else "${selectedImageBytesList.size} images selected"
                setTextColor(t.onSurface)
                textSize = 13f
                typeface = Typeface.DEFAULT_BOLD
            })
            info.addView(TextView(this).apply {
                val modeLabel = if (selectedImageBytesList.size > 1) "  ·  multi-image mode" else ""
                text = "${totalBytes} bytes  raw bytes ready$modeLabel"
                setTextColor(t.onSurfaceVar)
                textSize = 11f
                typeface = Typeface.MONOSPACE
            })
            attached.addView(info)
            attached.addView(TextView(this).apply {
                text = "x"
                setTextColor(t.onSurfaceVar)
                gravity = Gravity.CENTER
                textSize = 14f
                layoutParams = LinearLayout.LayoutParams(dp(32), dp(32))
                setOnClickListener { onClearImageClicked(); rebuildUi() }
            })
            openAiImageCard.addView(attached)
            imageStatusView = TextView(this).apply { visibility = View.GONE }
        } else {
            val pickLabel = if (isMultiImageModel(selDescriptor))
                "+  Pick images for V-JEPA" else "+  Pick image for OpenAI multimodal"
            openAiImageCard.addView(TextView(this).apply {
                text = pickLabel
                setTextColor(t.onSurfaceVar)
                textSize = 13f
                typeface = Typeface.DEFAULT_BOLD
                gravity = Gravity.CENTER
                background = dashedBg(t)
                setPadding(dp(14), dp(14), dp(14), dp(14))
                setOnClickListener { onPickImageClicked() }
            })
            imageStatusView = TextView(this).apply { visibility = View.GONE }
        }
        card.addView(openAiImageCard)
        spacer(card, 12)

        card.addView(labelView(t, "OPENAI REQUEST JSON"))
        openAIMessagesField = roundedEditText(t, openAiJsonText, multiline = true, mono = true, rows = 8,
            onTextChange = {
                val same = it == openAiJsonText
                openAiJsonText = it
                // Re-render the preview so role pills track edits live.
                if (!same) rebuildUi()
            })
        card.addView(openAIMessagesField)
        spacer(card, 12)

        val runBtn = filledButton(t, "▶  Run OpenAI request", fill = "vertical") {
            onOpenAIMessagesRunClicked()
        }.apply { isEnabled = !streaming && parseErr == null }
        card.addView(runBtn)
        return card
    }

    private fun buildMetricsTab(t: M3Tokens): View {
        val column = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        val m = lastMetrics
        if (m == null) {
            val empty = roundedCard(t, t.surfaceContainer).apply {
                gravity = Gravity.CENTER
                setPadding(dp(20), dp(32), dp(20), dp(32))
            }
            val icon = TextView(this).apply {
                text = "▤"
                gravity = Gravity.CENTER
                textSize = 28f
                setTextColor(t.onSurfaceVar)
                background = solid(t.surfaceContainerHigh, 28)
                layoutParams = LinearLayout.LayoutParams(dp(56), dp(56)).also {
                    it.bottomMargin = dp(12); it.gravity = Gravity.CENTER_HORIZONTAL
                }
            }
            empty.addView(icon)
            empty.addView(TextView(this).apply {
                text = "No metrics yet"
                setTextColor(t.onSurface)
                textSize = 15f
                typeface = Typeface.DEFAULT_BOLD
                gravity = Gravity.CENTER
            })
            empty.addView(TextView(this).apply {
                text = "Run a prompt and tap Fetch metrics in the Run tab to populate counters."
                setTextColor(t.onSurfaceVar)
                textSize = 13f
                gravity = Gravity.CENTER
                setPadding(0, dp(4), 0, 0)
            })
            spacer(empty, 12)
            empty.addView(filledButton(t, "Fetch metrics", fill = "vertical") { onMetricsClicked() })
            column.addView(empty)
            return column
        }

        val tps = if (m.generationDurationMs > 0)
            String.format("%.1f", m.generationTokens / (m.generationDurationMs / 1000.0))
        else "—"
        val ttft = String.format("%.0f", m.prefillDurationMs)

        // Big stat tile: tokens/sec.
        column.addView(metricTile(t, "TOKENS PER SECOND", tps, "tok/s", big = true))
        spacer(column, 10)

        val grid = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        grid.addView(metricTile(t, "TTFT", ttft, "ms", weight = 1f))
        spacerH(grid, 10)
        grid.addView(metricTile(t, "TOTAL", String.format("%.2f", m.totalDurationMs / 1000.0), "s", weight = 1f))
        column.addView(grid)
        spacer(column, 10)

        val grid2 = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        grid2.addView(metricTile(t, "PREFILL TOKENS", m.prefillTokens.toString(), "tok", weight = 1f))
        spacerH(grid2, 10)
        grid2.addView(metricTile(t, "GEN TOKENS", m.generationTokens.toString(), "tok", weight = 1f))
        column.addView(grid2)
        spacer(column, 10)

        val grid3 = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        grid3.addView(metricTile(t, "INIT", String.format("%.0f", m.initializationDurationMs), "ms", weight = 1f))
        spacerH(grid3, 10)
        grid3.addView(metricTile(t, "PEAK MEMORY",
            String.format("%.1f", m.peakMemoryKb / 1024.0), "MB", weight = 1f))
        column.addView(grid3)
        spacer(column, 10)

        // Prefill ▸ Gen bar.
        val barCard = roundedCard(t, t.surfaceContainer)
        barCard.addView(TextView(this).apply {
            text = "PREFILL ▸ GENERATION"
            setTextColor(t.onSurfaceVar)
            textSize = 11f
            typeface = Typeface.DEFAULT_BOLD
            setPadding(0, 0, 0, dp(10))
        })
        val total = (m.totalDurationMs).coerceAtLeast(1.0)
        val prefillFrac = (m.prefillDurationMs / total).coerceIn(0.0, 1.0).toFloat()
        val bar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            background = solid(t.surfaceContainerHigh, 4)
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, dp(8))
        }
        bar.addView(View(this).apply {
            background = solid(t.tertiary, 0)
            layoutParams = LinearLayout.LayoutParams(0, MATCH_PARENT, prefillFrac)
        })
        bar.addView(View(this).apply {
            background = solid(t.primary, 0)
            layoutParams = LinearLayout.LayoutParams(0, MATCH_PARENT, 1f - prefillFrac)
        })
        barCard.addView(bar)
        spacer(barCard, 8)
        val barLegend = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        barLegend.addView(TextView(this).apply {
            text = "● prefill ${ttft}ms"
            setTextColor(t.tertiary)
            textSize = 11f
            typeface = Typeface.MONOSPACE
            layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
        })
        barLegend.addView(TextView(this).apply {
            text = "● gen ${String.format("%.0f", m.generationDurationMs)}ms"
            setTextColor(t.primary)
            textSize = 11f
            typeface = Typeface.MONOSPACE
            gravity = Gravity.RIGHT
            layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
        })
        barCard.addView(barLegend)
        column.addView(barCard)
        spacer(column, 10)
        column.addView(tonalButton(t, "↻  Refresh metrics") { onMetricsClicked() })
        return column
    }

    private fun buildOutputPanel(t: M3Tokens): View {
        val wrap = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = solid(t.codeBg, 20)
        }
        // Title bar with macOS-style traffic-light dots.
        val titleBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(14), dp(10), dp(14), dp(10))
        }
        val dotRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        for (color in intArrayOf(0xFFFF5F57.toInt(), 0xFFFEBC2E.toInt(), 0xFF28C840.toInt())) {
            val d = View(this).apply {
                background = circle(color)
                layoutParams = LinearLayout.LayoutParams(dp(10), dp(10)).also {
                    it.rightMargin = dp(4)
                }
            }
            dotRow.addView(d)
        }
        titleBar.addView(dotRow)
        titleBar.addView(TextView(this).apply {
            text = "output  ·  stream.kt"
            setTextColor(0x80FFFFFF.toInt())
            textSize = 11f
            typeface = Typeface.MONOSPACE
            gravity = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
        })
        val copy = TextView(this).apply {
            text = "📋"
            setTextColor(0x80FFFFFF.toInt())
            gravity = Gravity.CENTER
            textSize = 12f
            layoutParams = LinearLayout.LayoutParams(dp(24), dp(24))
            setOnClickListener {
                if (outputText.isNotEmpty()) {
                    val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                    val clip = ClipData.newPlainText("output", outputText)
                    clipboard.setPrimaryClip(clip)
                }
            }
        }
        titleBar.addView(copy)
        spacerH(titleBar, 8)
        val clear = TextView(this).apply {
            text = "🗑"
            setTextColor(0x80FFFFFF.toInt())
            gravity = Gravity.CENTER
            textSize = 12f
            layoutParams = LinearLayout.LayoutParams(dp(24), dp(24))
            setOnClickListener {
                outputText = ""
                outputView.text = ""
            }
        }
        titleBar.addView(clear)
        wrap.addView(titleBar)
        wrap.addView(View(this).apply {
            setBackgroundColor(0x10FFFFFF)
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, 1)
        })

        outputScrollView = NestedScrollView(this).apply {
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, dp(220))
        }
        outputView = TextView(this).apply {
            text = if (outputText.isEmpty())
                "// streaming output appears here…" else outputText
            setTextColor(if (outputText.isEmpty()) 0x4DFFFFFF else 0xFFEDE7F6.toInt())
            textSize = 13f
            typeface = Typeface.MONOSPACE
            setPadding(dp(16), dp(12), dp(16), dp(12))
            layoutParams = FrameLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        outputScrollView.addView(outputView)
        wrap.addView(outputScrollView)
        return wrap
    }

    /* ════════════════════════════════════════════════════════════════
     * Reusable UI primitives (drawables, buttons, fields, chips, …)
     * ════════════════════════════════════════════════════════════════ */

    private fun roundedCard(t: M3Tokens, color: Int): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = solid(color, 20)
            setPadding(dp(16), dp(16), dp(16), dp(16))
        }
    }

    private fun collapsibleCard(
        t: M3Tokens,
        iconGlyph: String, iconBg: Int, iconFg: Int,
        title: String, subtitle: String,
        rightAdornment: View?,
        expanded: Boolean,
        onToggle: () -> Unit,
        body: (LinearLayout) -> Unit,
    ): View {
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = strokedSolid(t.surface, 24, t.outlineVariant, 1)
            setPadding(dp(12), dp(12), dp(12), dp(12))
        }
        // Header row.
        val header = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(4), dp(6), dp(4), dp(6))
            isClickable = true
            setOnClickListener { onToggle() }
        }
        val iconBox = TextView(this).apply {
            text = iconGlyph
            gravity = Gravity.CENTER
            textSize = 16f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(iconFg)
            background = solid(iconBg, 10)
            layoutParams = LinearLayout.LayoutParams(dp(36), dp(36))
        }
        header.addView(iconBox)
        spacerH(header, 12)
        val titleCol = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
        }
        titleCol.addView(TextView(this).apply {
            text = title
            setTextColor(t.onSurface)
            textSize = 15f
            typeface = Typeface.DEFAULT_BOLD
        })
        titleCol.addView(TextView(this).apply {
            text = subtitle
            setTextColor(t.onSurfaceVar)
            textSize = 12f
        })
        header.addView(titleCol)
        if (rightAdornment != null) header.addView(rightAdornment)
        header.addView(TextView(this).apply {
            text = if (expanded) "▲" else "▼"
            setTextColor(t.onSurfaceVar)
            textSize = 12f
            setPadding(dp(6), 0, 0, 0)
        })
        card.addView(header)
        if (expanded) {
            val bodyContainer = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(dp(4), dp(12), dp(4), dp(4))
            }
            body(bodyContainer)
            card.addView(bodyContainer)
        }
        return card
    }

    private fun sectionHeader(
        t: M3Tokens, glyph: String, iconBg: Int, iconFg: Int,
        title: String, subtitle: String
    ): View {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val icon = TextView(this).apply {
            text = glyph
            gravity = Gravity.CENTER
            textSize = 16f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(iconFg)
            background = solid(iconBg, 10)
            layoutParams = LinearLayout.LayoutParams(dp(32), dp(32))
        }
        row.addView(icon)
        spacerH(row, 8)
        val col = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        col.addView(TextView(this).apply {
            text = title
            setTextColor(t.onSurface)
            textSize = 15f
            typeface = Typeface.DEFAULT_BOLD
        })
        col.addView(TextView(this).apply {
            text = subtitle
            setTextColor(t.onSurfaceVar)
            textSize = 12f
        })
        row.addView(col)
        return row
    }

    private fun labelView(t: M3Tokens, text: String): TextView = TextView(this).apply {
        this.text = text
        setTextColor(t.onSurfaceVar)
        textSize = 12f
        typeface = Typeface.DEFAULT_BOLD
        setPadding(dp(4), 0, 0, dp(6))
    }

    private fun roundedEditText(
        t: M3Tokens, value: String,
        multiline: Boolean = false,
        mono: Boolean = false,
        numeric: Boolean = false,
        rows: Int = 1,
        placeholder: String? = null,
        onTextChange: (String) -> Unit,
    ): EditText {
        val baseBg = strokedSolid(t.surfaceContainer, 12, Color.TRANSPARENT, 0)
        val focusBg = strokedSolid(t.surfaceContainer, 12, t.primary, 2)
        val field = EditText(this).apply {
            setText(value)
            if (placeholder != null) {
                hint = placeholder
                setHintTextColor(t.onSurfaceVar)
            }
            setTextColor(t.onSurface)
            textSize = 14f
            background = baseBg
            setPadding(dp(14), dp(12), dp(14), dp(12))
            if (mono) typeface = Typeface.MONOSPACE
            if (multiline) {
                inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
                minLines = rows
                maxLines = rows + 4
                setHorizontallyScrolling(false)
                gravity = Gravity.TOP
            }
            if (numeric) {
                inputType = InputType.TYPE_CLASS_NUMBER or
                    InputType.TYPE_NUMBER_FLAG_DECIMAL or
                    InputType.TYPE_NUMBER_FLAG_SIGNED
            }
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
            setOnFocusChangeListener { _, hasFocus ->
                background = if (hasFocus) focusBg else baseBg
                setPadding(dp(14), dp(12), dp(14), dp(12))
            }
            addTextChangedListener(simpleWatcher(onTextChange))
        }
        return field
    }

    private fun labeledColumn(t: M3Tokens, label: String, field: View, weight: Float): View {
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, weight)
        }
        col.addView(labelView(t, label))
        col.addView(field)
        return col
    }

    private fun chipRow(t: M3Tokens, options: List<String>, selected: String,
                        onPick: (String) -> Unit): View {
        val scroll = HorizontalScrollView(this).apply {
            isHorizontalScrollBarEnabled = false
            overScrollMode = View.OVER_SCROLL_NEVER
        }
        val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        for ((i, opt) in options.withIndex()) {
            val active = opt == selected
            val chip = TextView(this).apply {
                text = if (active) "✓ $opt" else opt
                setTextColor(if (active) t.onSurface else t.onSurfaceVar)
                textSize = 13f
                typeface = if (active) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
                background = if (active)
                    solid(t.secondaryContainer, 8)
                else
                    strokedSolid(Color.TRANSPARENT, 8, t.outline, 1)
                setPadding(dp(12), dp(6), dp(12), dp(6))
                gravity = Gravity.CENTER
                layoutParams = LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).also {
                    if (i > 0) it.leftMargin = dp(6)
                }
                setOnClickListener { onPick(opt) }
            }
            row.addView(chip)
        }
        scroll.addView(row)
        return scroll
    }

    /**
     * @brief Read-only view showing the default model name (folder name)
     * derived from the model descriptor. Displays an error message if the
     * expected folder does not exist under MODEL BASE PATH.
     */
    private fun modelNameView(t: M3Tokens, descriptor: ModelDescriptor?): View {
        val wrapper = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        val relPath = descriptor?.let { modelPathById[it.id] } ?: descriptor?.id ?: "—"
        val folderName = relPath.substringBefore('/')
        val fullPath = "${modelBasePathText.trimEnd('/')}/$folderName"
        val folderExists = File(fullPath).exists()

        val nameView = TextView(this).apply {
            text = folderName
            setTextColor(t.onSurface)
            textSize = 14f
            typeface = Typeface.DEFAULT_BOLD
            background = strokedSolid(t.surfaceContainer, 8, t.outline, 1)
            setPadding(dp(14), dp(12), dp(14), dp(12))
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        wrapper.addView(nameView)

        if (!folderExists) {
            wrapper.addView(TextView(this).apply {
                text = "⚠ Folder not found: $fullPath"
                setTextColor(t.error)
                textSize = 11f
                typeface = Typeface.MONOSPACE
                setPadding(dp(4), dp(4), 0, 0)
                layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
            })
        }
        return wrapper
    }

    /**
     * @brief Dropdown field that lists subdirectories inside [basePath]
     * and updates [modelPathText] with the full path when a model is picked.
     * Falls back to the current [currentPath] as the displayed value if
     * the directory is not readable or has no subdirectories.
     */
    private fun modelNameDropdown(t: M3Tokens, currentPath: String, basePath: String): View {
        // Scan the base path for subdirectories
        val baseDir = File(basePath.trimEnd('/'))
        val subDirs: List<String> = if (baseDir.exists() && baseDir.isDirectory) {
            baseDir.listFiles()
                ?.filter { it.isDirectory }
                ?.map { it.name }
                ?.sorted()
                ?: emptyList()
        } else {
            emptyList()
        }

        // Determine display name: if currentPath starts with basePath, show the relative name
        val displayPath = currentPath.trimEnd('/')
        val displayValue = if (displayPath.startsWith(basePath.trimEnd('/')) && displayPath.length > basePath.trimEnd('/').length) {
            displayPath.substring(basePath.trimEnd('/').length + 1)
        } else {
            displayPath.substringAfterLast('/')
        }

        val field = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = strokedSolid(t.surfaceContainer, 8, t.outline, 1)
            setPadding(dp(12), dp(10), dp(12), dp(10))
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        val valueView = TextView(this).apply {
            text = displayValue.ifEmpty { "— tap to select —" }
            setTextColor(if (displayValue.isNotEmpty()) t.onSurface else t.onSurfaceVar)
            textSize = 14f
            typeface = Typeface.DEFAULT_BOLD
            layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
        }
        val arrow = TextView(this).apply {
            text = "▾"
            setTextColor(t.onSurfaceVar)
            textSize = 14f
        }
        field.addView(valueView)
        field.addView(arrow)

        field.setOnClickListener { anchor ->
            // Build options: subdirectories + a "Custom…" option
            val options = if (subDirs.isNotEmpty()) subDirs + "Custom…" else listOf("Custom…")
            val menu = PopupMenu(this, anchor)
            options.forEachIndexed { i, opt ->
                menu.menu.add(0, i, i, opt).apply {
                    isCheckable = true
                    isChecked = opt == displayValue
                }
            }
            menu.setOnMenuItemClickListener { item ->
                val picked = options[item.itemId]
                if (picked == "Custom…") {
                    // Switch to a free-form EditText for custom model name entry
                    modelPathField = roundedEditText(t, modelPathText, mono = true,
                        onTextChange = { modelPathText = it })
                    // Replace the dropdown with the edit text in the parent
                    val parent = field.parent as? ViewGroup
                    val index = parent?.indexOfChild(field) ?: -1
                    if (parent != null && index >= 0) {
                        parent.removeView(field)
                        parent.addView(modelPathField, index)
                    }
                } else {
                    // Build the full path: basePath/picked
                    val newPath = "${basePath.trimEnd('/')}/$picked"
                    modelPathText = newPath
                    valueView.text = picked
                    setStatus("Model name: $picked")
                }
                true
            }
            menu.show()
        }
        return field
    }

    private fun dropdownField(t: M3Tokens, options: List<String>, selected: String,
                             onPick: (String) -> Unit): View {
        val enabled = options.isNotEmpty()
        val field = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = strokedSolid(
                if (enabled) t.surfaceContainer else Color.TRANSPARENT, 8, t.outline, 1)
            setPadding(dp(12), dp(10), dp(12), dp(10))
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        val valueView = TextView(this).apply {
            text = if (enabled) selected else "—"
            setTextColor(if (enabled) t.onSurface else t.onSurfaceVar)
            textSize = 14f
            typeface = Typeface.DEFAULT_BOLD
            layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
        }
        val arrow = TextView(this).apply {
            text = "▾"
            setTextColor(t.onSurfaceVar)
            textSize = 14f
        }
        field.addView(valueView)
        field.addView(arrow)
        if (enabled) {
            field.setOnClickListener { anchor ->
                val menu = PopupMenu(this, anchor)
                options.forEachIndexed { i, opt ->
                    menu.menu.add(0, i, i, opt).apply {
                        isCheckable = true
                        isChecked = opt == selected
                    }
                }
                menu.setOnMenuItemClickListener { item ->
                    onPick(options[item.itemId])
                    true
                }
                menu.show()
            }
        }
        return field
    }

    private fun filledButton(t: M3Tokens, label: String, fill: String? = null,
                             danger: Boolean = false, onClick: () -> Unit): Button {
        return Button(this).apply {
            text = label
            isAllCaps = false
            setTextColor(if (danger) Color.WHITE else t.onPrimary)
            textSize = 14f
            typeface = Typeface.DEFAULT_BOLD
            stateListAnimator = null
            background = solid(if (danger) t.error else t.primary, 100)
            setPadding(dp(20), dp(12), dp(20), dp(12))
            layoutParams = when (fill) {
                "vertical"   -> LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
                "horizontal" -> LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
                else         -> LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT)
            }
            setOnClickListener { onClick() }
        }
    }

    private fun tonalButton(t: M3Tokens, label: String, danger: Boolean = false,
                            onClick: () -> Unit): Button {
        return Button(this).apply {
            text = label
            isAllCaps = false
            setTextColor(if (danger) t.error else t.onSurface)
            textSize = 13f
            typeface = Typeface.DEFAULT_BOLD
            stateListAnimator = null
            background = solid(if (danger) t.errorContainer else t.secondaryContainer, 100)
            setPadding(dp(14), dp(8), dp(14), dp(8))
            layoutParams = LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT)
            setOnClickListener { onClick() }
        }
    }

    private fun outlinedButton(t: M3Tokens, label: String,
                               onClick: () -> Unit): Button {
        return Button(this).apply {
            text = label
            isAllCaps = false
            setTextColor(t.primary)
            textSize = 13f
            typeface = Typeface.DEFAULT_BOLD
            stateListAnimator = null
            background = strokedSolid(Color.TRANSPARENT, 100, t.outline, 1)
            setPadding(dp(14), dp(8), dp(14), dp(8))
            layoutParams = LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT)
            setOnClickListener { onClick() }
        }
    }

    private fun metricTile(t: M3Tokens, label: String, value: String, unit: String,
                           big: Boolean = false, weight: Float = 0f): View {
        val tile = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = solid(if (big) t.primaryContainer else t.surfaceContainer, 20)
            setPadding(dp(if (big) 20 else 14),
                       dp(if (big) 20 else 14),
                       dp(if (big) 20 else 14),
                       dp(if (big) 20 else 14))
            layoutParams = if (weight > 0)
                LinearLayout.LayoutParams(0, WRAP_CONTENT, weight)
            else
                LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        val fg = if (big) t.onPrimaryContainer else t.onSurface
        tile.addView(TextView(this).apply {
            text = label
            setTextColor(fg and 0x00FFFFFF or (0xB3 shl 24))
            textSize = 11f
            typeface = Typeface.DEFAULT_BOLD
        })
        val valueRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.BOTTOM
            setPadding(0, dp(4), 0, 0)
        }
        valueRow.addView(TextView(this).apply {
            text = value
            setTextColor(fg)
            textSize = if (big) 38f else 24f
            typeface = Typeface.MONOSPACE
        })
        valueRow.addView(TextView(this).apply {
            text = " $unit"
            setTextColor(fg and 0x00FFFFFF or (0x99 shl 24))
            textSize = if (big) 14f else 11f
            typeface = Typeface.DEFAULT_BOLD
            setPadding(dp(4), 0, 0, dp(if (big) 6 else 3))
        })
        tile.addView(valueRow)
        return tile
    }

    /* ───── Drawable / dimension helpers ───── */

    private fun solid(color: Int, radiusDp: Int): GradientDrawable = GradientDrawable().apply {
        setColor(color)
        cornerRadius = dpf(radiusDp)
    }

    private fun strokedSolid(fill: Int, radiusDp: Int, strokeColor: Int, strokeDp: Int): GradientDrawable =
        GradientDrawable().apply {
            setColor(fill)
            cornerRadius = dpf(radiusDp)
            if (strokeDp > 0) setStroke(dp(strokeDp), strokeColor)
        }

    private fun circle(color: Int): GradientDrawable = GradientDrawable().apply {
        shape = GradientDrawable.OVAL
        setColor(color)
    }

    private fun gradient(c1: Int, c2: Int, radiusDp: Int): GradientDrawable =
        GradientDrawable(GradientDrawable.Orientation.TL_BR, intArrayOf(c1, c2)).apply {
            cornerRadius = dpf(radiusDp)
        }

    private fun dashedBg(t: M3Tokens): GradientDrawable = GradientDrawable().apply {
        setColor(Color.TRANSPARENT)
        cornerRadius = dpf(12)
        setStroke(dp(1), t.outline, dpf(6), dpf(4))
    }

    private fun blendAlpha(color: Int, alpha: Int): Int =
        (color and 0x00FFFFFF) or (alpha shl 24)

    private fun dp(v: Int): Int = TypedValue.applyDimension(
        TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), resources.displayMetrics
    ).toInt()

    private fun dpf(v: Int): Float = TypedValue.applyDimension(
        TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), resources.displayMetrics
    )

    private fun spacer(parent: LinearLayout, h: Int) {
        parent.addView(View(this).apply {
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, dp(h))
        })
    }

    private fun spacerH(parent: LinearLayout, w: Int) {
        parent.addView(View(this).apply {
            layoutParams = LinearLayout.LayoutParams(dp(w), MATCH_PARENT)
        })
    }

    private fun simpleWatcher(onChange: (String) -> Unit): TextWatcher = object : TextWatcher {
        override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
        override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
        override fun afterTextChanged(s: Editable?) { onChange(s?.toString() ?: "") }
    }

    private fun statusTone(): String {
        val s = statusText.lowercase()
        return when {
            "fail" in s || "error" in s || "empty" in s || "cancel" in s -> "error"
            "done" in s || "loaded" in s || "opened" in s || "ready" in s -> "success"
            "load" in s || "running" in s || "open" in s || "stream" in s -> "progress"
            else -> "neutral"
        }
    }

    /** Pretty label including the design's "MULTIMODAL/TEXT/QNN" tag. */
    private fun badgeLabel(d: ModelDescriptor): String = when {
        Capability.MULTIMODAL in d.capabilities -> "[MULTIMODAL]  ${d.displayName}"
        Capability.EMBEDDING in d.capabilities  -> "[EMBEDDING]   ${d.displayName}"
        d.backends == setOf(BackendType.NPU)    -> "[QNN]         ${d.displayName}"
        else -> "[TEXT]        ${d.displayName}"
    }

    private fun isMultimodal(d: ModelDescriptor) = Capability.MULTIMODAL in d.capabilities
    private fun usesOpenAiApi(d: ModelDescriptor) = Capability.OPENAI_API in d.capabilities

    private fun visionBackendFor(d: ModelDescriptor, backend: BackendType): BackendType? =
        if (isMultimodal(d)) backend else null

    /* ════════════════════════════════════════════════════════════════
     * Engine handlers (logic preserved from the original sample)
     * ════════════════════════════════════════════════════════════════ */

    override fun onDestroy() {
        // Fire-and-forget close on the engine thread so we don't leak the
        // native model handle / LiteRT-LM Engine on config changes.
        // engine.close() internally closes any active chat session.
        val e = engine
        engine = null
        loadedKey = null
        if (e != null) {
            engineExecutor.execute {
                try { e.close() } catch (_: Throwable) { /* best effort */ }
            }
        }
        super.onDestroy()
    }

    private fun onLoadClicked() {
        val req = buildLoadRequest()
        val requestKey = modelRequestKey(req)
        loadStatus = "loading"
        setStatus("Loading $requestKey…  (vision=${req.visionBackend?.name ?: "off"})")
        outputText = ""
        mainHandler.post { rebuildUi() }
        engineExecutor.execute { loadModelInternal(req) }
    }

    private fun onCancelClicked() {
        engine?.cancel()
        streaming = false
        setStatus("Cancel requested.")
        mainHandler.post { rebuildUi() }
    }

    private fun isSdCapableModel(): Boolean =
        selDescriptor?.runtime == RuntimeKind.NATIVE && selDescriptor?.capabilities?.contains(Capability.SPECULATIVE) == true

    private fun isChatSdCapableModel(): Boolean =
        chatSelDescriptor?.runtime == RuntimeKind.NATIVE && chatSelDescriptor?.capabilities?.contains(Capability.SPECULATIVE) == true

    /**
     * @brief Build a [LoadModelRequest] from the current UI state. Must
     * be called on the main thread.
     */
    private fun buildLoadRequest(): LoadModelRequest {
        val d = selDescriptor
        val backend = selBackend
        val quant = selectedQuant
        val modelPath = modelPathText.trim().ifEmpty { null }
        val nativeLibDir = applicationContext.applicationInfo.nativeLibraryDir
        val basePath = (if (::modelBasePathField.isInitialized) modelBasePathField.text.toString()
                        else modelBasePathText).trim().ifEmpty { modelBasePathText }
        val effectiveModelId = (if (selSdEnabled) (d?.sdVariantId ?: d?.id) else d?.id)
                               ?: selFamily
        return LoadModelRequest(
            backend = backend,
            modelId = effectiveModelId,
            quantization = quant,
            modelPath = modelPath,
            visionBackend = d?.let { visionBackendFor(it, backend) },
            nativeLibDir = nativeLibDir,
            modelBasePath = basePath,
            useSpeculativeDecoding = selSdEnabled,
        )
    }

    private fun buildChatLoadRequest(): LoadModelRequest {
        val d = chatSelDescriptor
        val backend = chatSelBackend
        val quant = chatSelectedQuant
        val modelPath = defaultModelPathFor(d)
        val nativeLibDir = applicationContext.applicationInfo.nativeLibraryDir
        val basePath = (if (::chatModelBasePathField.isInitialized) chatModelBasePathField.text.toString()
                        else modelBasePathText).trim().ifEmpty { modelBasePathText }
        val effectiveModelId = (if (chatSelSdEnabled) (d?.sdVariantId ?: d?.id) else d?.id)
                               ?: chatSelFamily
        return LoadModelRequest(
            backend = backend,
            modelId = effectiveModelId,
            quantization = quant,
            modelPath = modelPath,
            visionBackend = d?.let { visionBackendFor(it, backend) },
            nativeLibDir = nativeLibDir,
            modelBasePath = basePath,
            useSpeculativeDecoding = chatSelSdEnabled,
        )
    }

    private fun modelRequestKey(req: LoadModelRequest): String = buildString {
        append(req.modelId)
        append(':')
        append(req.backend.name)
        append(':')
        append(req.quantization.name)
        append(":vision=")
        append(req.visionBackend?.name ?: "off")
        append(":sd=")
        append(req.useSpeculativeDecoding)
        append(":config=")
        append(Integer.toHexString(req.hashCode()))
    }

    /**
     * @brief Core model loading logic. Must be called from [engineExecutor].
     * Returns the loaded [QuickDotAI] engine, or null on failure.
     */
    private fun loadModelInternal(req: LoadModelRequest): QuickDotAI? {
        val requestKey = modelRequestKey(req)
        if (loadedKey != null && loadedKey != requestKey) {
            try { engine?.close() } catch (_: Throwable) { /* best effort */ }
            engine = null
            loadedKey = null
            clearChatSessionState()
        }
        if (engine != null && loadedKey == requestKey) {
            clearChatIfBoundToDifferentKey(requestKey)
            loadStatus = "loaded"
            loadedLabel = requestKey
            setStatus("Already loaded: $requestKey")
            mainHandler.post { rebuildUi() }
            return engine
        }

        val descriptor = ModelCatalog.byId(req.modelId)
            ?: ModelCatalog.all().firstOrNull { it.sdVariantId == req.modelId }
        if (descriptor == null) {
            loadStatus = "idle"
            setStatus("Load failed: unknown model id '${req.modelId}'.")
            mainHandler.post { rebuildUi() }
            return null
        }
        val newEngine = createEngine(descriptor)
        return when (val r = newEngine.load(req)) {
            is BackendResult.Ok -> {
                engine = newEngine
                loadedKey = requestKey
                clearChatSessionState()
                loadStatus = "loaded"
                loadedLabel = requestKey
                setStatus("Loaded $requestKey (${newEngine.kind}, model=${newEngine.modelId ?: "?"})")
                mainHandler.post { rebuildUi() }
                newEngine
            }
            is BackendResult.Err -> {
                try { newEngine.close() } catch (_: Throwable) { /* best effort */ }
                loadStatus = "idle"
                clearChatSessionState()
                setStatus("Load failed: [${r.error.name}] ${r.message ?: ""}")
                mainHandler.post { rebuildUi() }
                null
            }
        }
    }

    private fun onRunClicked() {
        val prompt = promptField.text.toString()
        if (prompt.isBlank()) {
            setStatus("Prompt is empty.")
            return
        }
        val imgBytesList = selectedImageBytesList
        if (imgBytesList.isNotEmpty() &&
            (selDescriptor?.let { isMultimodal(it) && usesOpenAiApi(it) } != true)
        ) {
            setStatus("Selected model does not support OpenAI image input.")
            return
        }
        outputText = ""
        mainHandler.post { outputView.text = "" }
        streaming = true
        val imgStatus = if (imgBytesList.isEmpty()) ""
            else if (imgBytesList.size == 1) "Running multimodal (${imgBytesList[0].size}B image)…"
            else "Running multimodal (${imgBytesList.size} images)…"
        setStatus(if (imgStatus.isNotEmpty()) imgStatus else "Running…")
        mainHandler.post { rebuildUi() }

        engineExecutor.execute {
            val e = engine
            if (e == null) {
                streaming = false
                setStatus("No model loaded — tap Load first.")
                mainHandler.post { rebuildUi() }
                return@execute
            }
            val sink = object : StreamSink {
                override fun onDelta(text: String) {
                    outputText += text
                    mainHandler.post { outputView.append(text) }
                }
                override fun onDone() {
                    streaming = false
                    setStatus("Done.")
                    mainHandler.post { rebuildUi() }
                }
                override fun onError(error: QuickAiError, message: String?) {
                    streaming = false
                    setStatus("Run failed: [${error.name}] ${message ?: ""}")
                    mainHandler.post { rebuildUi() }
                }
            }
            try {
                val result = if (imgBytesList.isEmpty()) {
                    e.runText(prompt, sink)
                } else {
                    e.runOpenAI(
                        createSingleTurnOpenAIRequest(prompt, imgBytesList, e.kind),
                        sink
                    )
                }
                when (result) {
                    is BackendResult.Ok -> {
                        when (val metrics = e.metrics()) {
                            is BackendResult.Ok -> lastMetrics = metrics.value
                            is BackendResult.Err -> Unit
                        }
                    }
                    is BackendResult.Err -> {
                        streaming = false
                        setStatus("Run failed: [${result.error.name}] ${result.message ?: ""}")
                    }
                }
                mainHandler.post { rebuildUi() }
            } catch (t: Throwable) {
                streaming = false
                setStatus("Run threw: ${t.message}")
                mainHandler.post { rebuildUi() }
            }
        }
    }

    private fun onMetricsClicked() {
        engineExecutor.execute {
            val e = engine
            if (e == null) {
                setStatus("No model loaded.")
                return@execute
            }
            when (val r = e.metrics()) {
                is BackendResult.Ok -> {
                    lastMetrics = r.value
                    setStatus("Metrics fetched.")
                    mainHandler.post { rebuildUi() }
                }
                is BackendResult.Err ->
                    setStatus("Metrics failed: [${r.error.name}] ${r.message ?: ""}")
            }
        }
    }

    private fun onUnloadClicked() {
        val e = engine
        e?.cancel()  // Immediately cancel any in-flight inference (thread-safe)

        engineExecutor.execute {
            val e = engine
            if (e == null) {
                clearChatSessionState()
                setStatus("Nothing to unload.")
                return@execute
            }
            when (val r = e.unload()) {
                is BackendResult.Ok -> {
                    try { e.close() } catch (_: Throwable) { /* best effort */ }
                    if (engine === e) engine = null
                    loadedKey = null
                    loadStatus = "idle"
                    loadedLabel = ""
                    clearChatSessionState()
                    setStatus("Unloaded.")
                    mainHandler.post { rebuildUi() }
                }
                is BackendResult.Err ->
                    setStatus("Unload failed: [${r.error.name}] ${r.message ?: ""}")
            }
        }
    }

    /* ───── Local OpenAI chat handlers ───── */

    private fun onChatOpenClicked() {
        if (chatSelDescriptor?.let { usesOpenAiApi(it) } == false) {
            setStatus("Selected chat model does not support the OpenAI request API.")
            return
        }
        val req = buildChatLoadRequest()
        val systemPrompt = systemPromptText.trim().ifEmpty { null }

        setStatus("Starting local OpenAI chat…")
        engineExecutor.execute {
            val e = loadModelInternal(req)
            if (e == null) {
                setStatus("Cannot start chat — model load failed.")
                return@execute
            }
            val chatKey = loadedKey
            if (chatKey == null) {
                setStatus("Cannot start chat — loaded model identity is unavailable.")
                return@execute
            }
            startChatSession(chatKey, systemPrompt)
            setStatus("Chat ready. History will be sent through runOpenAI().")
            mainHandler.post { rebuildUi() }
        }
    }

    private fun onChatRunStreamingClicked() {
        val prompt = chatPromptField.text.toString()
        if (prompt.isBlank()) { setStatus("Chat message is empty."); return }
        if (!isChatActiveFor(loadedKey)) {
            setStatus("No active chat — tap Open first.")
            return
        }
        val imgBytesList = selectedImageBytesList
        if (imgBytesList.isNotEmpty() && chatSelDescriptor?.let { isMultimodal(it) } != true) {
            setStatus("Selected chat model does not support image input.")
            return
        }
        outputText = ""
        outputView.text = ""
        streaming = true
        val imgStatus = if (imgBytesList.isEmpty()) ""
            else if (imgBytesList.size == 1) "Chat multimodal streaming..."
            else "Chat multimodal streaming (${imgBytesList.size} images)..."
        setStatus(if (imgStatus.isNotEmpty()) imgStatus else "Chat streaming...")
        mainHandler.post { rebuildUi() }

        engineExecutor.execute {
            val e = engine
            if (e == null) {
                streaming = false
                setStatus("No model loaded - tap Open first.")
                mainHandler.post { rebuildUi() }
                return@execute
            }
            val chatSnapshot = snapshotActiveChat(loadedKey)
            if (chatSnapshot == null) {
                streaming = false
                setStatus("No active chat - tap Open first.")
                mainHandler.post { rebuildUi() }
                return@execute
            }
            val currentImages = try {
                prepareOpenAIImages(
                    bytesList = imgBytesList,
                    firstSourceIndex = chatSnapshot.nextImageIndex,
                    engineKind = e.kind,
                )
            } catch (t: Throwable) {
                streaming = false
                setStatus("Image preprocessing failed: ${t.message}")
                mainHandler.post { rebuildUi() }
                return@execute
            }
            val userMessage = LocalChatMessage(
                role = "user",
                text = prompt,
                imageSources = currentImages.sources,
                imageTensors = currentImages.tensors,
            )
            val generated = StringBuilder()
            val sink = object : StreamSink {
                override fun onDelta(text: String) {
                    generated.append(text)
                    outputText += text
                    mainHandler.post { outputView.append(text) }
                }
                override fun onDone() = Unit
                override fun onError(error: QuickAiError, message: String?) {
                    streaming = false
                    setStatus("Chat error: [${error.name}] ${message ?: ""}")
                    mainHandler.post { rebuildUi() }
                }
            }
            try {
                val request = buildChatOpenAIRequest(
                    history = chatSnapshot.history,
                    currentUser = userMessage,
                )
                when (val r = e.runOpenAI(request, sink)) {
                    is BackendResult.Ok -> {
                        val committed = commitChatTurn(
                            snapshot = chatSnapshot,
                            userMessage = userMessage,
                            assistantText = generated.toString(),
                            imageCount = currentImages.sources.size,
                        )
                        if (!committed) {
                            streaming = false
                            setStatus("Chat result discarded because the chat was closed.")
                            mainHandler.post { rebuildUi() }
                            return@execute
                        }
                        if (selectedImageBytesList === imgBytesList) {
                            selectedImageBytesList = emptyList()
                        }
                        streaming = false
                        val duration = when (val metrics = e.metrics()) {
                            is BackendResult.Ok -> {
                                lastMetrics = metrics.value
                                metrics.value.totalDurationMs.toLong().toString()
                            }
                            is BackendResult.Err -> "?"
                        }
                        setStatus("Chat done. ($duration ms)")
                        mainHandler.post { rebuildUi() }
                    }
                    is BackendResult.Err -> {
                        streaming = false
                        setStatus("Chat failed: [${r.error.name}] ${r.message ?: ""}")
                        mainHandler.post { rebuildUi() }
                    }
                }
            } catch (t: Throwable) {
                streaming = false
                setStatus("Chat threw: ${t.message}")
                mainHandler.post { rebuildUi() }
            }
        }
    }

    private fun onChatCloseClicked() {
        if (streaming) engine?.cancel()
        clearChatSessionState()
        streaming = false
        setStatus("Chat closed and local history cleared.")
        mainHandler.post { rebuildUi() }
    }

    private fun buildChatOpenAIRequest(
        history: List<LocalChatMessage>,
        currentUser: LocalChatMessage,
    ): OpenAIRequest {
        val messages = history + currentUser
        val requestFields = linkedMapOf<String, JsonElement>(
            "messages" to JsonArray(messages.map(::localChatMessageToJson)),
        )
        val tensors = messages.flatMap { it.imageTensors }
        return OpenAIRequest(
            json = JsonObject(requestFields).toString(),
            imageTensors = tensors.takeIf { it.isNotEmpty() }?.let {
                OpenAIImageTensorSidecar(tensors = it)
            },
        )
    }

    private fun localChatMessageToJson(message: LocalChatMessage): JsonObject {
        val content: JsonElement = if (message.imageSources.isEmpty()) {
            JsonPrimitive(message.text)
        } else {
            JsonArray(
                message.imageSources.map(::imageUrlPart) +
                    textPart(message.text)
            )
        }
        return JsonObject(
            linkedMapOf(
                "role" to JsonPrimitive(message.role),
                "content" to content,
            )
        )
    }

    /* ───── OpenAI-style messages handlers ───── */

    private fun parseOpenAIRequestObject(jsonString: String): JsonObject {
        val root = Json.parseToJsonElement(jsonString) as? JsonObject
            ?: throw IllegalArgumentException("OpenAI request must be a JSON object.")
        if (root["messages"] !is JsonArray) {
            throw IllegalArgumentException("OpenAI request must contain a messages array.")
        }
        return root
    }

    private fun parseOpenAIPreview(jsonString: String): List<OpenAIPreviewMessage> {
        val messages = parseOpenAIRequestObject(jsonString)["messages"] as JsonArray
        return messages.mapIndexed { index, element ->
            val message = element as? JsonObject
                ?: throw IllegalArgumentException("messages[$index] must be an object.")
            val role = (message["role"] as? JsonPrimitive)?.contentOrNull
                ?: throw IllegalArgumentException("messages[$index].role must be a string.")
            OpenAIPreviewMessage(role, previewOpenAIContent(message["content"]))
        }
    }

    private fun previewOpenAIContent(content: JsonElement?): String = when (content) {
        null -> ""
        is JsonPrimitive -> content.contentOrNull ?: content.toString()
        is JsonArray -> content.joinToString(" ") { element ->
            val part = element as? JsonObject ?: return@joinToString element.toString()
            when ((part["type"] as? JsonPrimitive)?.contentOrNull?.lowercase()) {
                "text", "input_text" ->
                    (part["text"] as? JsonPrimitive)?.contentOrNull.orEmpty()
                "image_url", "input_image" -> "[image]"
                else -> "[content part]"
            }
        }
        else -> content.toString()
    }

    private fun createSingleTurnOpenAIRequest(
        prompt: String,
        imageBytes: List<ByteArray>,
        engineKind: String,
    ): OpenAIRequest {
        val images = prepareOpenAIImages(imageBytes, engineKind = engineKind)
        val message = localChatMessageToJson(
            LocalChatMessage(
                role = "user",
                text = prompt,
                imageSources = images.sources,
                imageTensors = images.tensors,
            )
        )
        return OpenAIRequest(
            json = JsonObject(
                mapOf("messages" to JsonArray(listOf(message)))
            ).toString(),
            imageTensors = images.tensors.takeIf { it.isNotEmpty() }?.let {
                OpenAIImageTensorSidecar(tensors = it)
            },
        )
    }

    private fun createOpenAIRequest(
        originalJson: String,
        requestObject: JsonObject,
        imageBytes: List<ByteArray>,
        engineKind: String,
    ): OpenAIRequest {
        if (imageBytes.isEmpty()) return OpenAIRequest(json = originalJson)

        val images = prepareOpenAIImages(imageBytes, engineKind = engineKind)
        val withImages = bindImagesToOpenAIRequest(
            requestObject = requestObject,
            sources = images.sources,
        )
        return OpenAIRequest(
            json = withImages.toString(),
            imageTensors = images.tensors.takeIf { it.isNotEmpty() }?.let {
                OpenAIImageTensorSidecar(tensors = it)
            },
        )
    }

    private fun prepareOpenAIImages(
        bytesList: List<ByteArray>,
        firstSourceIndex: Int = 0,
        engineKind: String,
    ): PreparedOpenAIImages {
        if (engineKind == "litert-lm") {
            val sources = bytesList.mapIndexed { index, bytes ->
                require(bytes.isNotEmpty()) { "image[$index] is empty." }
                dataImageSource(bytes)
            }
            return PreparedOpenAIImages(sources = sources, tensors = emptyList())
        }

        val tensors = preprocessImages(bytesList, firstSourceIndex)
        return PreparedOpenAIImages(
            sources = tensors.map { it.source },
            tensors = tensors,
        )
    }

    private fun preprocessImages(
        bytesList: List<ByteArray>,
        firstSourceIndex: Int = 0,
    ): List<OpenAIImageTensor> {
        if (bytesList.isEmpty()) return emptyList()
        return bytesList.mapIndexed { index, bytes ->
            require(bytes.isNotEmpty()) { "image[$index] is empty." }
            val source = mediaSource(firstSourceIndex + index)
            when (val result = imagePreprocessor.preprocess(source, bytes)) {
                is BackendResult.Ok -> result.value
                is BackendResult.Err -> throw IllegalArgumentException(
                    "image[$index] preprocessing failed: " +
                        "[${result.error.name}] ${result.message.orEmpty()}"
                )
            }
        }
    }

    private fun bindImagesToOpenAIRequest(
        requestObject: JsonObject,
        sources: List<String>,
    ): JsonObject {
        val messages = requestObject["messages"] as JsonArray
        val existingCount = countImageUrlParts(messages)
        val updatedMessages = when {
            existingCount == 0 -> attachImagesToLastUser(messages, sources)
            existingCount == sources.size -> replaceImageUrlSources(messages, sources)
            else -> throw IllegalArgumentException(
                "Request contains $existingCount image_url parts, but ${sources.size} images are selected."
            )
        }
        return JsonObject(requestObject.toMutableMap().apply {
            this["messages"] = updatedMessages
        })
    }

    private fun countImageUrlParts(messages: JsonArray): Int = messages.sumOf { messageElement ->
        val content = (messageElement as? JsonObject)?.get("content") as? JsonArray
            ?: return@sumOf 0
        content.count { partElement ->
            val part = partElement as? JsonObject
            (part?.get("type") as? JsonPrimitive)?.contentOrNull == "image_url"
        }
    }

    private fun attachImagesToLastUser(
        messages: JsonArray,
        sources: List<String>,
    ): JsonArray {
        val lastUserIndex = messages.indexOfLast { element ->
            val message = element as? JsonObject
            (message?.get("role") as? JsonPrimitive)?.contentOrNull == "user"
        }
        require(lastUserIndex >= 0) {
            "Select-image attachment requires at least one user message."
        }
        val result = messages.toMutableList()
        val user = result[lastUserIndex] as JsonObject
        val existingContent = when (val content = user["content"]) {
            null -> emptyList()
            is JsonArray -> content.toList()
            is JsonPrimitive -> content.contentOrNull?.let { listOf(textPart(it)) }.orEmpty()
            else -> throw IllegalArgumentException(
                "The last user message content must be a string or content array."
            )
        }
        val content = JsonArray(sources.map(::imageUrlPart) + existingContent)
        result[lastUserIndex] = JsonObject(user.toMutableMap().apply {
            this["content"] = content
        })
        return JsonArray(result)
    }

    private fun replaceImageUrlSources(
        messages: JsonArray,
        sources: List<String>,
    ): JsonArray {
        var sourceIndex = 0
        return JsonArray(messages.map messageLoop@ { messageElement ->
            val message = messageElement as? JsonObject ?: return@messageLoop messageElement
            val content = message["content"] as? JsonArray ?: return@messageLoop messageElement
            val updatedContent = JsonArray(content.map partLoop@ { partElement ->
                val part = partElement as? JsonObject ?: return@partLoop partElement
                val type = (part["type"] as? JsonPrimitive)?.contentOrNull
                if (type != "image_url") return@partLoop partElement
                val source = sources[sourceIndex++]
                val oldImageUrl = part["image_url"] as? JsonObject
                val imageUrlFields = oldImageUrl?.toMutableMap() ?: mutableMapOf()
                imageUrlFields["url"] = JsonPrimitive(source)
                val newImageUrl = JsonObject(imageUrlFields)
                JsonObject(part.toMutableMap().apply {
                    this["image_url"] = newImageUrl
                })
            })
            JsonObject(message.toMutableMap().apply {
                this["content"] = updatedContent
            })
        })
    }

    private fun imageUrlPart(source: String): JsonObject = JsonObject(
        linkedMapOf(
            "type" to JsonPrimitive("image_url"),
            "image_url" to JsonObject(mapOf("url" to JsonPrimitive(source))),
        )
    )

    private fun textPart(text: String): JsonObject = JsonObject(
        linkedMapOf(
            "type" to JsonPrimitive("text"),
            "text" to JsonPrimitive(text),
        )
    )

    private fun mediaSource(index: Int): String = "quickai-media://image-$index"

    private fun dataImageSource(bytes: ByteArray): String {
        val mediaType = when {
            bytes.size >= 8 && bytes.copyOfRange(0, 8).contentEquals(
                byteArrayOf(
                    0x89.toByte(), 0x50, 0x4E, 0x47,
                    0x0D, 0x0A, 0x1A, 0x0A
                )
            ) -> "image/png"
            bytes.size >= 3 && bytes[0] == 0xFF.toByte() &&
                bytes[1] == 0xD8.toByte() && bytes[2] == 0xFF.toByte() -> "image/jpeg"
            bytes.size >= 12 && bytes.copyOfRange(0, 4).decodeToString() == "RIFF" &&
                bytes.copyOfRange(8, 12).decodeToString() == "WEBP" -> "image/webp"
            bytes.size >= 6 && bytes.copyOfRange(0, 6).decodeToString()
                .let { it == "GIF87a" || it == "GIF89a" } -> "image/gif"
            else -> "image/unknown"
        }
        return "data:$mediaType;base64,${Base64.getEncoder().encodeToString(bytes)}"
    }

    private fun onOpenAIMessagesRunClicked() {
        val jsonText = openAIMessagesField.text.toString()
        if (jsonText.isBlank()) { setStatus("OpenAI request JSON is empty."); return }
        val requestObject = try {
            parseOpenAIRequestObject(jsonText)
        } catch (t: IllegalArgumentException) {
            setStatus(t.message ?: "Invalid OpenAI request JSON.")
            return
        }
        if (selDescriptor?.let { usesOpenAiApi(it) } == false) {
            setStatus("Selected model does not support the OpenAI request API.")
            return
        }
        val imgBytesList = selectedImageBytesList
        if (imgBytesList.isNotEmpty() && selDescriptor?.let { isMultimodal(it) } != true) {
            setStatus("Selected model does not support OpenAI image input.")
            return
        }
        outputText = ""
        outputView.text = ""
        streaming = true
        val imgStatus = if (imgBytesList.isEmpty()) ""
            else if (imgBytesList.size == 1) "Running OpenAI multimodal (streaming)..."
            else "Running OpenAI multimodal (${imgBytesList.size} images, streaming)..."
        setStatus(if (imgStatus.isNotEmpty()) imgStatus else "Running OpenAI JSON (streaming)...")
        mainHandler.post { rebuildUi() }

        val req = buildLoadRequest()
        engineExecutor.execute {
            val e = loadModelInternal(req)
            if (e == null) {
                streaming = false; setStatus("Model load failed.")
                mainHandler.post { rebuildUi() }; return@execute
            }
            val sink = object : StreamSink {
                override fun onDelta(text: String) {
                    outputText += text
                    mainHandler.post { outputView.append(text) }
                }
                override fun onDone() = Unit
                override fun onError(error: QuickAiError, message: String?) {
                    streaming = false; setStatus("OpenAI error: [${error.name}] ${message ?: ""}")
                    mainHandler.post { rebuildUi() }
                }
            }
            try {
                val request = createOpenAIRequest(
                    originalJson = jsonText,
                    requestObject = requestObject,
                    imageBytes = imgBytesList,
                    engineKind = e.kind,
                )
                when (val r = e.runOpenAI(request, sink)) {
                    is BackendResult.Ok -> {
                        streaming = false
                        if (imgBytesList.isNotEmpty() && selectedImageBytesList === imgBytesList) {
                            selectedImageBytesList = emptyList()
                        }
                        when (val metrics = e.metrics()) {
                            is BackendResult.Ok -> lastMetrics = metrics.value
                            is BackendResult.Err -> Unit
                        }
                        setStatus("OpenAI request done.")
                        mainHandler.post { rebuildUi() }
                    }
                    is BackendResult.Err -> {
                        streaming = false
                        setStatus("OpenAI request failed: [${r.error.name}] ${r.message ?: ""}")
                        mainHandler.post { rebuildUi() }
                    }
                }
            } catch (t: Throwable) {
                streaming = false; setStatus("Threw: ${t.message}")
                mainHandler.post { rebuildUi() }
            }
        }
    }

    /** Whether the currently selected model supports multi-image (V-JEPA). */
    private fun isMultiImageModel(d: ModelDescriptor?): Boolean =
        d != null && Capability.MULTI_IMAGE in d.capabilities

    /* ───── Image picker handlers ───── */

    private fun onPickImageClicked() {
        // Use multi-image picker for V-JEPA models, single for others
        if (isMultiImageModel(selDescriptor) || isMultiImageModel(chatSelDescriptor)) {
            multiImagePickerLauncher.launch(
                PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
            )
        } else {
            imagePickerLauncher.launch(
                PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
            )
        }
        setStatus("Opening photo picker…")
    }

    private fun onClearImageClicked() {
        selectedImageBytesList = emptyList()
        if (::imageStatusView.isInitialized) {
            mainHandler.post { imageStatusView.text = "Image: none" }
        }
        setStatus("Image cleared.")
    }

    private fun readImageBytesAsync(uris: List<Uri>) {
        val total = uris.size
        setStatus(if (total == 1) "Reading image…" else "Reading $total images…")
        Thread({
            try {
                val bytesRead = mutableListOf<ByteArray>()
                for ((i, uri) in uris.withIndex()) {
                    val bytes = contentResolver.openInputStream(uri)?.use { it.readBytes() }
                    if (bytes == null || bytes.isEmpty()) {
                        setStatus("Image ${i + 1} read failed or empty."); return@Thread
                    }
                    bytesRead.add(bytes)
                }
                selectedImageBytesList = bytesRead.toList()
                val totalBytes = bytesRead.sumOf { it.size }
                val statusMsg = if (bytesRead.size == 1) {
                    "Image loaded ($totalBytes bytes). Ready for Run or Send."
                } else {
                    "${bytesRead.size} images loaded ($totalBytes bytes total). Multi-image mode."
                }
                setStatus(statusMsg)
                mainHandler.post { rebuildUi() }
            } catch (t: Throwable) {
                setStatus("Failed to read image: ${t.message}")
            }
        }, "SampleTestAPP-ImageRead").apply { isDaemon = true }.start()
    }

    /* ───── Misc helpers ───── */

    private val exampleByModelId: Map<String, String> = mapOf(
        ModelIds.GEMMA4 to """{
  "messages": [
    {"role": "system", "content": "You are a concise vision assistant."},
    {"role": "user", "content": "Describe the selected image."}
  ]
}""",
        ModelIds.FUNCTION_GEMMA to """{
  "messages": [
    {"role": "system", "content": "You can call tools when they are useful."},
    {"role": "user", "content": "01012345678 번호로 상담 예약 확인 문자를 보내줘."}
  ],
  "tools": [{"type": "function", "function": {"name": "send_sms", "description": "Send a text message to a phone number.", "parameters": {"type": "object", "properties": {"phone_number": {"type": "string"}, "message": {"type": "string"}}, "required": ["phone_number", "message"]}}}]
}""",
        ModelIds.TINY_BERT to """{
  "messages": [{"role": "user", "content": "Explain what text embeddings are in one sentence."}]
}""",
    )
    // Shared template for models exposing the OpenAI request API.
    private val openAiApiExample = """{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant. Answer briefly."},
    {"role": "user", "content": "Summarize why on-device language models are useful."}
  ]
}"""
    private val defaultToolExample = """{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short checklist for testing an Android API wrapper."}
  ]
}"""

    private fun defaultOpenAIExampleForId(modelId: String): String {
        exampleByModelId[modelId]?.let { return it }
        val d = ModelCatalog.byId(modelId)
        return when {
            d != null && Capability.OPENAI_API in d.capabilities -> openAiApiExample
            else -> defaultToolExample
        }
    }

    private fun snapshotChatUiState(expectedKey: String?): ChatUiState =
        synchronized(chatStateLock) {
            ChatUiState(
                active = chatActive && activeChatKey == expectedKey,
                assistantTurns = chatHistory.count { it.role == "assistant" },
            )
        }

    private fun isChatActiveFor(expectedKey: String?): Boolean =
        synchronized(chatStateLock) {
            chatActive && activeChatKey == expectedKey
        }

    private fun snapshotActiveChat(expectedKey: String?): ActiveChatSnapshot? =
        synchronized(chatStateLock) {
            if (!chatActive || expectedKey == null || activeChatKey != expectedKey) {
                return@synchronized null
            }
            ActiveChatSnapshot(
                generation = chatGeneration,
                key = expectedKey,
                history = chatHistory.toList(),
                nextImageIndex = nextChatImageIndex,
            )
        }

    private fun startChatSession(key: String, systemPrompt: String?) =
        synchronized(chatStateLock) {
            clearChatSessionStateLocked()
            if (systemPrompt != null) {
                chatHistory += LocalChatMessage(role = "system", text = systemPrompt)
            }
            chatActive = true
            activeChatKey = key
        }

    private fun commitChatTurn(
        snapshot: ActiveChatSnapshot,
        userMessage: LocalChatMessage,
        assistantText: String,
        imageCount: Int,
    ): Boolean = synchronized(chatStateLock) {
        if (!chatActive ||
            activeChatKey != snapshot.key ||
            chatGeneration != snapshot.generation
        ) {
            return@synchronized false
        }
        chatHistory += userMessage
        chatHistory += LocalChatMessage(
            role = "assistant",
            text = assistantText,
        )
        nextChatImageIndex += imageCount
        true
    }

    private fun clearChatIfBoundToDifferentKey(expectedKey: String) =
        synchronized(chatStateLock) {
            if (activeChatKey != null && activeChatKey != expectedKey) {
                clearChatSessionStateLocked()
            }
        }

    private fun clearChatSessionState() = synchronized(chatStateLock) {
        clearChatSessionStateLocked()
    }

    private fun clearChatSessionStateLocked() {
        chatGeneration += 1
        chatActive = false
        activeChatKey = null
        chatHistory.clear()
        nextChatImageIndex = 0
    }

    private fun setStatus(text: String) {
        statusText = text
        mainHandler.post {
            if (::statusView.isInitialized) statusView.text = text
        }
    }

    /**
     * @brief Builds the configured on-device path for a catalog model,
     * rooted in the model base directory selected by the sample.
     */
    private fun defaultModelPathFor(d: ModelDescriptor?): String? {
        if (d == null) return null
        val base = modelBasePathText.trimEnd('/')
        return modelPathById[d.id]?.let { "$base/$it" }
            ?: "$base/${d.id}"
    }

    private val modelPathById: Map<String, String> = mapOf(
        ModelIds.GEMMA4         to "gemma-4-E2B-it/gemma-4-E2B-it.litertlm",
        ModelIds.QWEN3_0_6B     to "qwen3-0.6b",
        ModelIds.QWEN3_1_7B_Q40 to "qwen3-1.7b-q40-arm",
        ModelIds.TINY_BERT      to "tiny-bert",
        ModelIds.FUNCTION_GEMMA to "function_gemma",
        ModelIds.GEMMA4_CPU     to "gemma4_cpu",
        ModelIds.GEMMA4_E2B_QNN to "gemma-4-e2b-qnn",
        ModelIds.VJEPA_QNN      to "vjepa2-qnn",
    )

    private fun checkAllFilesAccess() {
        if (Environment.isExternalStorageManager()) return
        val intent = Intent(
            Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
            Uri.parse("package:${packageName}")
        )
        startActivity(intent)
    }

    private fun quantizationSuffix(quant: QuantizationType): String = when (quant) {
        QuantizationType.W4A32 -> "-w4a32"
        QuantizationType.W16A16 -> "-w16a16"
        QuantizationType.W8A16 -> "-w8a16"
        QuantizationType.W32A32 -> "-w32a32"
        QuantizationType.UNKNOWN -> "-w4a32"
    }
}
