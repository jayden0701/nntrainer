# QuickDotAI Android AAR

`QuickDotAI` is the Android-facing API for running one loaded on-device model.
The AAR contains a native nntrainer/QNN adapter and a LiteRT-LM adapter.
Only `arm64-v8a` native prebuilts are currently packaged.

## Dependency packaging

QuickDotAI is a thin AAR: it packages QuickDotAI classes and native `.so`
files, but it does not merge dependency bytecode into the archive. A Gradle
project dependency resolves the declarations in `QuickDotAI/build.gradle.kts`.
If the library is later published with Maven metadata, consumers get the same
transitive dependency resolution from that metadata.

Copying only a raw `.aar` loses all Gradle dependency metadata. Raw-AAR
consumers must therefore declare the current dependencies themselves:

```kotlin
implementation(files("libs/QuickDotAI.aar"))
implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
implementation("com.google.ai.edge.litertlm:litertlm-android:0.10.0")
implementation("androidx.core:core-ktx:1.12.0")
```

`kotlinx-serialization-json` is part of the public compile contract;
LiteRT-LM and AndroidX Core are runtime implementation dependencies. Keep the
versions synchronized with `Android/gradle/libs.versions.toml` and
`QuickDotAI/build.gradle.kts` when updating the AAR.

## Public generation API

Generation has two input concepts (the interface excerpt omits lifecycle
metadata and the separate embedding helper):

```kotlin
interface QuickDotAI {
    fun load(request: LoadModelRequest): BackendResult<Unit>

    fun runText(
        text: String,
        sink: StreamSink,
    ): BackendResult<Unit>

    fun runOpenAI(
        request: OpenAIRequest,
        sink: StreamSink,
    ): BackendResult<Unit>

    fun cancel()
    fun unload(): BackendResult<Unit>
    fun close()
}
```

`runText()` sends the exact input text without adding a chat template or
implicit history. A backend that cannot guarantee exact raw-text semantics
returns `UNSUPPORTED`. Empty text is rejected as `INVALID_PARAMETER`.

`runOpenAI()` forwards an OpenAI-compatible JSON object unchanged. This is the
path for chat templates, tools, functions, response constraints, and
multimodal requests.

The native adapter accepts the subset documented by the
[C API reference](../../api/README.md): messages, function tools, structured
response formats, and compatibility metadata. The loaded engine—not the JSON
`model` field—selects the model, and callback delivery is always streaming.
Unsupported sampling/length controls are reported as `UNSUPPORTED` rather
than being ignored. `store: true` and `parallel_tool_calls: true` are likewise
unsupported by the local single-response runner.

LiteRT-LM currently represents a narrower subset: top-level `messages` and an
optional string `model`, with `system`, `user`, and `assistant` text or locally
resolvable image content. `developer` (whose priority cannot be preserved by
the LiteRT conversation API), tools, and structured-output controls return
`UNSUPPORTED` on that backend. The final message must be `user`, image content
is accepted only in `user` messages, and `image_url.detail` currently supports
only `auto`.

Both methods synchronously block the invoking worker thread while streaming
deltas. Do not call them on Android's main thread. `StreamSink` callbacks are
not marshalled to the main thread by the AAR. Callbacks must not re-enter the
same engine through `load`, generation, `metrics`, `unload`, or `close`: the
native handle remains locked until generation returns. Only `cancel()` is
supported from another thread. Apply this rule to terminal callbacks too.

`QuickDotAI.modelId` identifies the successfully loaded model. It is `null`
before `load()` succeeds and again after `unload()` or `close()`; it does not
claim to report a runtime-inspected model architecture.

## Native multimodal tensor sidecar

The native backend carries preprocessed image tensors in a versioned sidecar
rather than placing large float arrays in JSON. Each tensor's `source` must
exactly match an `image_url.url` in the JSON request.

This is a descriptor-capability-gated path. A descriptor must advertise
`MULTIMODAL` to accept any sidecar and `MULTI_IMAGE` to accept more than one
sidecar in the same request. `runOpenAI()` remains the only public structured
generation call: it first uses an architecture-specific fused/composite hook
registered by an optional model plugin, then falls back to the generic
`[vision encoder, embedding-input LLM]` composer when the loaded handle exposes
that topology. A full V2 hook receives every sidecar plus the active grammar;
it also receives the validated OpenAI request, all loaded sub-model pointers,
and an optional core-formatted prompt. This lets a fused plugin apply its own
model-specific chat template when no compatible core template exists. Once
invoked, the hook's result is authoritative. The legacy single-image callback
is used only for an unconstrained RGB 512x512-patch tensor because its older
signature cannot express arbitrary metadata. A missing private plugin is
normal for the public package and does not make native image input globally or
intentionally unsupported. An
individual request returns `UNSUPPORTED` when its descriptor, image count,
plugin hook, or generic model interfaces are incompatible.

The following example uses the model-specific LLaVA-NeXT preprocessor. Replace
it with the preprocessor or `MODEL_NATIVE` tensor contract declared for the
selected plugin model; its pixels are not a universal native image format.

```kotlin
val source = "quickdotai://image/0"
val imageTensor = when (
    val result = LlavaNextImagePreprocessor().preprocess(
        source = source,
        encodedImage = encodedImage,
    )
) {
    is BackendResult.Ok -> result.value
    is BackendResult.Err -> error(result.message ?: result.error.name)
}

val json = """
    {
      "messages": [{
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "$source"}},
          {"type": "text", "text": "Describe this image."}
        ]
      }]
    }
""".trimIndent()

val request = OpenAIRequest(
    json = json,
    imageTensors = OpenAIImageTensorSidecar(
        tensors = listOf(imageTensor),
    )
)

engine.runOpenAI(request, sink)
```

For `HWC` and `CHW`, validation requires
`pixelValues.size == patchCount * channels * patchHeight * patchWidth`.
`MODEL_NATIVE` is for a model-specific preprocessed representation whose value
count is validated by that native model. Dimensions, patch count, source
order, sidecar version, and JSON source references are validated before JNI is
entered. Sidecar tensors follow `image_url` occurrence order. Repeating the
same URL twice therefore requires two sidecar entries with the same source.
Multiple entries are accepted when the loaded descriptor advertises
`MULTI_IMAGE`; otherwise the native adapter fails before inference rather than
silently dropping all but the first image.

`LlavaNextImagePreprocessor` returns the HWC RGB, 512-pixel patch representation
expected by a LLaVA-NeXT path. It accepts encoded JPEG/PNG bytes or a `Bitmap`;
it is intentionally model-specific rather than a universal image format.

### Optional native model plugin

A downstream AAR or host app may package `libqai_ext_model.so` in its
`jniLibs/arm64-v8a` directory. QuickDotAI loads it after the core native
libraries so its static registration can add catalog descriptors, model
factory entries, and architecture callbacks. The plugin may expose either a
self-contained fused multimodal model or a composite model backed by multiple
sub-models; callers still select one catalog descriptor and call
`runOpenAI()`.

The plugin boundary uses C++ virtual interfaces and callback registry types.
`libcausallm.so`, `libquick_dot_ai_api.so`, `libquickai_jni.so`, and
`libqai_ext_model.so` must therefore be rebuilt from the same source revision;
dropping an older plugin binary next to newer core libraries is unsupported.
The public AAR does not bundle a private plugin by default.

This contract has been checked statically in the source and JNI/AAR packaging
paths. Fused/composite plugin execution, multi-image streaming, and Android
device lifecycle behavior still require verification with a same-revision
plugin on a physical device.

## LiteRT-LM multimodal image URLs

LiteRT-LM consumes sidecar-less OpenAI image content directly. It preserves the
order of text and images in both initial history and the final user message.
Supported sources are:

- `data:image/...;base64,...`, mapped to LiteRT-LM `Content.ImageBytes`;
- absolute, readable `file://` URLs, mapped to `Content.ImageFile`.

For example, put an encoded image in the JSON and omit `imageTensors`:

```kotlin
val source = "data:image/jpeg;base64," +
    java.util.Base64.getEncoder().encodeToString(encodedImage)
val request = OpenAIRequest(
    json = """
        {"messages":[{"role":"user","content":[
          {"type":"image_url","image_url":{"url":"$source"}},
          {"type":"text","text":"Describe this image."}
        ]}]}
    """.trimIndent(),
)
engine.runOpenAI(request, sink)
```

LiteRT-LM rejects preprocessed tensor sidecars because they are native-model
representations. The AAR never fetches `http://` or `https://` URLs and cannot
resolve `content://` or custom schemes; those return `UNSUPPORTED`. The host
must resolve such media into bytes or an app-readable file first.

The AAR declares no storage permission. Use app-owned storage or Android's
Storage Access Framework for model and image files.

## Lifecycle and threading

- Call `load()` before generation.
- Create an implementation with `createEngine(descriptor)`; concrete backend
  classes are internal implementation details.
- Supply `modelPath` for LiteRT-LM or `modelBasePath` for the native backend;
  the AAR never guesses a shared-storage model location.
- Serialize ordinary calls for one engine on a worker thread.
- `cancel()` is the cross-thread cancellation entry point.
- Call `unload()` or `close()` when finished; `close()` is idempotent.
- Native and LiteRT capabilities differ. Unsupported behavior is reported
  explicitly rather than silently dropping OpenAI fields or changing raw text.

The legacy handle/messages/multimodal/chat-session `run*` variants are not part
of the AAR's public source API. Applications should keep full conversation
history in the OpenAI JSON request until a backend-owned session abstraction
with consistent KV-cache semantics is introduced.
