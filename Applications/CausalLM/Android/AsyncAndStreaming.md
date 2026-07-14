# Android Native Async and Streaming

Quick.AI generation is synchronous at the native C/JNI boundary and
asynchronous only when the host app schedules it on a worker. A native call
blocks that worker while delivering token deltas; callers must not invoke it on
Android's main thread.

## Current path

```text
QuickDotAI.kt
  -> NativeQuickDotAI.kt
    -> NativeCausalLm.kt
      -> quickai_jni.cpp
        -> quick_dot_ai_api.h / libquick_dot_ai_api.so
```

The planned REST/foreground-service layer is outside this contract.

## Two generation calls

The public Kotlin API and native C API have matching generation concepts:

| Kotlin | JNI | C API | Input contract |
|---|---|---|---|
| `runText` | `runTextStreamingNative` | `quickAiRunText` | Exact, already-formatted UTF-8 text |
| `runOpenAI` | `runOpenAIStreamingNative` | `quickAiRunOpenAI` | OpenAI JSON plus an optional native image tensor sidecar |

The native declarations are:

```c
ErrorCode quickAiRunText(
    CausalLmHandle handle,
    const char *input,
    CausalLmTokenCallback callback,
    void *user_data);

ErrorCode quickAiRunOpenAI(
    CausalLmHandle handle,
    const char *json_request,
    const QuickAiImageTensorV1 *images,
    size_t image_count,
    CausalLmTokenCallback callback,
    void *user_data);
```

`quickAiRunText` does not apply a chat template, roles, JSON parsing, or a
grammar. `quickAiRunOpenAI` parses the request, applies the model's chat
template, and attaches xgrammar for `json_object`, `json_schema`, or a named /
required tool choice. A tool grammar constrains the normalized complete
`{"name": ..., "arguments": {...}}` envelope rather than the raw parameters
object alone. JSON input is therefore not merely templated text;
structured-output constraints are enforced during decoding when requested.

`tool_choice: "auto"` intentionally does not force a tool grammar. A separate
`response_format`, if present, still constrains the response.

The loaded native handle is authoritative even if the JSON contains `model`,
and this embedded callback API always streams regardless of the JSON `stream`
value. Unsupported sampling/length controls, `store: true`, and
`parallel_tool_calls: true` return `UNSUPPORTED` instead of being silently
discarded. LiteRT-LM accepts only `messages` and optional `model` at the top
level; its adapter reports richer native-only controls as unsupported.

## Callback contract

Both C functions invoke:

```c
typedef int (*CausalLmTokenCallback)(const char *delta, void *user_data);
```

Returning `0` continues generation. Returning non-zero requests cooperative
cancellation at the next token boundary. JNI forwards each delta on the same
thread that entered the native method, so it can use the current `JNIEnv *` and
does not attach a new JVM thread.

Kotlin adapts that callback to:

```kotlin
interface StreamSink {
    fun onDelta(text: String)
    fun onReasoningDelta(text: String) {}
    fun onDone()
    fun onError(error: QuickAiError, message: String?)
}
```

Each `runText` or `runOpenAI` call emits zero or more delta events followed by
exactly one terminal event: `onDone()` on success or `onError(...)` on failure.
Callbacks are not marshalled to the main thread. UI code must post updates to
the main dispatcher itself.

Callbacks are non-reentrant. The native generation path keeps its handle lock
until the run returns, so a callback must not call `load`, either generation
method, `metrics`, `unload`, or `close` on the same engine. Cross-thread
`cancel()` is the only supported exception. Kotlin emits terminal callbacks
before returning from the blocking method; callers should apply the same rule
to `onDone` and `onError`.

## Native image tensor sidecar

`OpenAIRequest.json` retains ordered `image_url` parts. The AAR converts each
`OpenAIImageTensor` into a `QuickAiImageTensorV1` sidecar entry rather than
placing float arrays in JSON.

For every native call:

- text-only requests pass `images == NULL` and `image_count == 0`;
- every `image_url` part has exactly one sidecar entry in the same order;
- `source` exactly matches the corresponding `image_url.url`;
- `struct_size` identifies the current C structure version;
- `values`, `value_count`, patch count, and original dimensions are valid for
  the duration of the blocking call;
- dense HWC/CHW entries have
  `value_count == patch_count * channels * patch_height * patch_width`;
- model-native entries follow the selected model's private tensor layout.

The native library never fetches the image URL. A same-revision extension
callback receives the complete sidecar and can implement fused and multi-image
models. When no full callback is registered, a compatible
`[vision encoder, embedding-input LLM]` handle can use the generic single-image
composer. The shape-limited legacy callback accepts only one unconstrained RGB
512x512-patch image (CHW is converted to HWC). Unsupported combinations return
`CAUSAL_LM_ERROR_UNSUPPORTED`.

## LiteRT-LM image sources

LiteRT-LM does not enter JNI and does not accept preprocessed tensor sidecars.
Its strict OpenAI adapter preserves ordered `text` / `input_text` and
`image_url` parts. A `data:image/...;base64` URL becomes
`Content.ImageBytes`; an absolute readable `file://` URL becomes
`Content.ImageFile`. This conversion applies to initial conversation messages
and the final user `Contents` sent for streaming.

The adapter performs no network or content-provider I/O. `http(s)://`,
`content://`, and custom schemes return `UNSUPPORTED`; malformed data URLs or
unreadable files return `INVALID_PARAMETER`. A valid native tensor sidecar is
also `UNSUPPORTED` on LiteRT-LM rather than being silently ignored.

## Scheduling and cancellation

A host application should:

1. load an engine;
2. call `runText` or `runOpenAI` from one worker thread;
3. consume deltas without blocking the callback; and
4. marshal UI work to the main thread.

`QuickDotAI.cancel()` may be called from another thread. It forwards to
`cancelModelHandle`, whose stop request is cooperative. Other operations on one
`QuickDotAI` instance are not generally thread-safe and should remain on the
owning worker.

## Failure mapping

Native `ErrorCode` values are mapped with `QuickAiError.fromNativeCode`:

- malformed JSON or sidecar mismatch -> `INVALID_PARAMETER`;
- missing loaded model -> `NOT_INITIALIZED`;
- unavailable chat template, grammar, or multimodal path -> `UNSUPPORTED`;
- generation failure -> `INFERENCE_FAILED`.

The Kotlin adapter converts the return value into the single terminal
`StreamSink` event described above.

## Related docs

- [QuickDotAI AAR API](QuickDotAI/README.md)
- [Android Architecture](Architecture.md)
- [C API Reference](../api/README.md)
