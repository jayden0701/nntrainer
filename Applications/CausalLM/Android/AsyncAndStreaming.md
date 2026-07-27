# QuickDotAI asynchronous and streaming behavior

`runText` and `runOpenAI` are synchronous, streaming calls. They block the
invoking thread until generation succeeds, fails, or is cancelled. Run them on
a worker thread, not Android's main thread.

## Event contract

Generated output is delivered to:

```kotlin
interface StreamSink {
    fun onDelta(text: String)
    fun onReasoningDelta(text: String) {}
    fun onDone()
    fun onError(error: QuickAiError, message: String?)
}
```

One run emits zero or more delta events followed by exactly one terminal event:

- `onDone()` for success or cooperative cancellation;
- `onError(...)` for failure.

Callbacks are not marshalled to the main thread. Native callbacks execute on
the thread that entered the JNI call; another backend may use its own callback
thread. A UI consumer must dispatch updates explicitly.

The blocking method also returns `BackendResult<Unit>`. The terminal callback
and return value describe the same run; do not start a second request from a
terminal callback.

## Reentrancy and ownership

Callbacks are non-reentrant. While a run is active, a callback must not call
`load`, `runText`, `runOpenAI`, `metrics`, `encode`, `unload`, or `close` on the
same engine. The native backend holds its handle lock during generation.

`cancel` is the only operation intended for another thread. It requests
cooperative cancellation; a backend may stop at its next token or polling
boundary. Calling it does not make other engine operations concurrently safe.

Use one worker/queue per engine:

```kotlin
withContext(Dispatchers.IO) {
    engine.runOpenAI(request, sink)
}
```

Call `close()` after the worker has stopped using the engine. `close()` is
idempotent.

## Request state

QuickDotAI does not retain a chat session or image store. Each `runOpenAI`
request must include the full conversation history in its `messages` array.
Images are request-scoped:

- native requests provide ordered `OpenAIImageTensorSidecar` tensors whose
  sources exactly match the JSON `image_url` occurrences;
- LiteRT-LM resolves a supported data URL or absolute file URL directly and
  does not accept tensor sidecars.

Native extension dispatch remains inside the same blocking run. If an image
extension callback is invoked, its status is authoritative and the core never
falls back to another multimodal path. A nonzero token-callback result is
forwarded as a stop request to the active extension models.

## Native C boundary

The equivalent public C calls are:

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

They invoke the callback synchronously. Returning nonzero from
`CausalLmTokenCallback` requests cancellation. JNI uses private arm/disarm
coordination so a Kotlin `cancel()` issued from another thread reaches the
currently active native run.

See [QuickDotAI/README.md](QuickDotAI/README.md) for request examples and
[../api/README.md](../api/README.md) for the C API contract.
