# QuickDotAI AAR

QuickDotAI provides one Kotlin API over two on-device runtimes:

- `NativeQuickDotAI` for NNTrainer CPU and optional QNN models.
- `LiteRTLm` for LiteRT-LM `.litertlm` models.

The package is `com.example.quickdotai`. The current Android build supports
`arm64-v8a`.

## Build

From `Applications/CausalLM`:

```bash
export ANDROID_NDK=/path/to/android-ndk
./build_android.sh --app
```

This builds the CPU native runtime by default, stages its public shared
libraries, builds `libquickai_jni.so`, and assembles both the AAR and
SampleTestAPP. It does not install anything unless `--install` is present.

Build the QNN variant explicitly:

```bash
export QNN_SDK_ROOT=/path/to/qnn-sdk
./build_android.sh --qnn --app
```

The AAR is written under `Android/QuickDotAI/build/outputs/aar/`. A direct
Gradle build must pass
`-PnntrainerNdkPath=/absolute/path/to/android-ndk`. It also requires
`prebuilt_libs/` to have been populated with the native QuickDotAI libraries,
normally by `build_android.sh`.

The public build does not provide a proprietary `libqai_ext_model.so` or
`htp_backend_ext_config.json`. A downstream package may add the optional plugin
after `libquickai_jni.so`, but the plugin must match the exact extension ABI,
build tag, and Transformer ABI and must stay loaded for the process lifetime.

For a source-tree dependency:

```kotlin
dependencies {
    implementation(project(":QuickDotAI"))
}
```

## Public API

```kotlin
interface QuickDotAI {
    val kind: String
    val modelId: String?

    fun load(req: LoadModelRequest): BackendResult<Unit>
    fun runText(text: String, sink: StreamSink): BackendResult<Unit>
    fun runOpenAI(
        request: OpenAIRequest,
        sink: StreamSink
    ): BackendResult<Unit>
    fun unload(): BackendResult<Unit>
    fun metrics(): BackendResult<PerformanceMetrics>
    fun encode(text: String): BackendResult<FloatArray>
    fun cancel()
    fun close()
}
```

`runText` and `runOpenAI` block the invoking worker thread while streaming to
`StreamSink`. `cancel` is the only operation intended to be called from another
thread. See [../AsyncAndStreaming.md](../AsyncAndStreaming.md).

There is no hidden chat session or image cache. A `runOpenAI` call supplies the
complete `messages` history and all request images every time.

## Select and load an engine

Use a catalog descriptor to bind the engine's runtime and capabilities:

```kotlin
val descriptor = requireNotNull(ModelCatalog.byId(ModelIds.QWEN3_0_6B))
val engine = createEngine(descriptor)

val loaded = engine.load(
    LoadModelRequest(
        modelId = descriptor.id,
        backend = BackendType.CPU,
        quantization = QuantizationType.W4A32,
        nativeLibDir = applicationInfo.nativeLibraryDir,
        modelBasePath =
            applicationContext.filesDir.resolve("Quick.AI/models").absolutePath
    )
)
```

`load` validates the requested ID and backend against the descriptor.
Speculative decoding additionally requires a descriptor with `SPECULATIVE` and
requires the caller to pass its declared `sdVariantId` as `modelId` together
with `useSpeculativeDecoding = true`; variant selection is not automatic.
For the native backend, `quantization` is a validated compatibility field;
current file-based catalog entries select the actual model variant and do not
derive a quantization-suffixed directory from it.

Picker helpers expose only descriptors satisfying:

```text
(STREAMING or OPENAI_API) and not VISION_ENCODER
```

In a QNN-enabled build, the standalone `vjepa2-qnn` model therefore remains
catalog-visible for image encoding but is excluded from generation selection.

## Text and OpenAI requests

`runText` is for exact, already-formatted model input:

```kotlin
engine.runText("<already formatted prompt>", sink)
```

It adds no template or history. LiteRT-LM deliberately returns
`QuickAiError.UNSUPPORTED` for this method because it cannot guarantee those
semantics.

`runOpenAI` accepts one complete OpenAI Chat Completions-style JSON object:

```kotlin
val request = OpenAIRequest(
    json = """
        {
          "messages": [
            {"role":"system","content":"Answer briefly."},
            {"role":"user","content":"What is NNTrainer?"}
          ]
        }
    """.trimIndent()
)
engine.runOpenAI(request, sink)
```

The loaded descriptor selects the model. The native callback API rejects
`stream: false`, an explicitly false `response_format.json_schema.strict`, a
`strict` field on functions, and unsupported sampling or length controls
instead of silently ignoring them.

## Native images

Native `image_url` content requires a preprocessed
`OpenAIImageTensorSidecar`. There must be one tensor per image occurrence, in
the same order, and each tensor's `source` must exactly match the corresponding
JSON URL:

```kotlin
val source = "quickai://request/image/0"
val request = OpenAIRequest(
    json = """
        {
          "messages": [{
            "role":"user",
            "content":[
              {"type":"image_url","image_url":{"url":"$source"}},
              {"type":"text","text":"Describe the image."}
            ]
          }]
        }
    """.trimIndent(),
    imageTensors = OpenAIImageTensorSidecar(
        tensors = listOf(
            OpenAIImageTensor(
                source = source,
                pixelValues = preprocessedPixels,
                layout = ImageTensorLayout.MODEL_NATIVE,
                patchCount = patchCount,
                channels = channels,
                patchHeight = patchHeight,
                patchWidth = patchWidth,
                originalHeight = imageHeight,
                originalWidth = imageWidth
            )
        )
    )
)
engine.runOpenAI(request, sink)
```

The model needs `MULTIMODAL`; more than one image also needs `MULTI_IMAGE`.
Native image execution exists only through a compatible versioned C extension.
There is no generic composer or legacy fallback, and an invoked extension's
status is final.

## LiteRT-LM

The built-in `gemma4` LiteRT descriptor is GPU-only:

```kotlin
val descriptor = requireNotNull(ModelCatalog.byId(ModelIds.GEMMA4))
val engine = createEngine(descriptor)
engine.load(
    LoadModelRequest(
        modelId = descriptor.id,
        backend = BackendType.GPU,
        modelPath =
            applicationContext.filesDir.resolve("models/gemma4.litertlm").absolutePath,
        visionBackend = BackendType.GPU
    )
)
```

Copy or materialize the selected model into that app-owned path first. If the
model is selected through Android's Storage Access Framework, do not assume a
raw shared-storage path is directly readable by the runtime.

LiteRT-LM uses `runOpenAI`, not `runText`. It does not accept a preprocessed
sidecar. An image URL must be either:

```json
{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
```

or an absolute file URL for an app-owned file. Construct it from the actual
file rather than assuming shared-storage access:

```kotlin
val imageUrl =
    applicationContext.filesDir.resolve("images/photo.png").toURI().toString()
```

Place that value in the request's `image_url.url` field.

The current descriptor does not advertise `MULTI_IMAGE`, so a request with
more than one image is rejected. Image requests also require a non-null
`visionBackend` at load time.

## Metrics, embeddings, and cleanup

`metrics()` reports the most recent completed run. Native fills token timing
and peak-memory counters; LiteRT-LM currently reports initialization and total
duration.

`encode()` is available only for a descriptor with `EMBEDDING`. Generative
models return `UNSUPPORTED`. The standalone native V-JEPA image encoder is
available through the native C image-encoding API rather than Kotlin
`encode(text)`.

Use `unload()` to release loaded model state while retaining the wrapper, or
`close()` to release everything. `close()` is idempotent.

## QNN backend configuration

QNN runtime libraries are included only in a `--qnn` build. The native loader
uses `QUICK_DOT_AI_QNN_BACKEND_EXT_CONFIG_PATH` when set. Otherwise it looks
for `htp_backend_ext_config.json` under `modelBasePath`; when that path ends in
`/models`, it looks under the parent directory instead. The public package does
not supply this proprietary JSON, so the downstream packager must place it at
the resolved path. The setting is process-wide: an existing environment value
or the path derived by the first QNN load is reused by later loads.
