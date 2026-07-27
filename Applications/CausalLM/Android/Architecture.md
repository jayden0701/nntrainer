# QuickDotAI Android architecture

The Android tree contains two Gradle modules:

```text
Android/
  QuickDotAI/       reusable AAR
  SampleTestAPP/    direct AAR consumer
```

There is no service, socket, or hidden session layer. An application owns a
`QuickDotAI` instance and calls it directly.

## Public layer

Package `com.example.quickdotai` exposes:

- `QuickDotAI.kt`: lifecycle, generation, cancellation, and streaming
  contracts.
- `Types.kt`: load requests, OpenAI image sidecars, errors, and metrics.
- `ModelCatalog.kt`: native and LiteRT model descriptors and selection helpers.

The public engine lifecycle is:

```text
createEngine(descriptor)
  -> load(request)
  -> runText(...) or runOpenAI(...)
  -> metrics() / encode(...) as supported
  -> unload() or close()
```

There is no chat session retained between calls. A backend may create a
request-local conversation object while `runOpenAI` is active, but every
request contains the complete message history and all images needed for that
run.

## Native runtime

The native implementation is arranged as:

```text
NativeQuickDotAI
  -> NativeCausalLm JNI declarations
  -> libquickai_jni.so
  -> libquick_dot_ai_api.so
  -> libcausallm.so and NNTrainer
```

`NativeCausalLm` first attempts to load the optional `libqnn_context.so`, then
loads the owner library `libquickai_jni.so`. After that it attempts to load an
optional downstream `libqai_ext_model.so`, whose initializer may register
model descriptors and callbacks through `quick_dot_ai_extension_api.h`.
The extension API does not register a C++ model constructor. A plugin that
introduces a new architecture must also arrange for that architecture to be
registered with the shared CausalLM `Factory`.

The public build does not create or stage `libqai_ext_model.so`. A downstream
plugin must match the host's exact extension ABI, build tag, and Transformer
ABI, and must remain loaded for the process lifetime.

Text-only OpenAI requests run in the public core. Native image requests require
preprocessed tensor sidecars and a registered versioned extension. Once an
extension callback is invoked, its result is final; there is no generic or
legacy multimodal fallback.

## LiteRT-LM runtime

`LiteRTLm` wraps a `.litertlm` model through LiteRT-LM. The built-in `gemma4`
descriptor is GPU-only and supports streaming OpenAI requests. `runText` is
unsupported because LiteRT-LM cannot guarantee the exact-text contract.

For an image request:

- `LoadModelRequest.visionBackend` must be configured.
- the OpenAI request uses `data:image/...;base64,...` or an absolute `file://`
  URL.
- preprocessed `OpenAIImageTensorSidecar` data is unsupported.
- the current descriptor lacks `MULTI_IMAGE`, so it accepts at most one image.

## Catalog and engine selection

`ModelCatalog` obtains native descriptors from `getModelCatalogJson()` and
adds LiteRT-only descriptors in Kotlin. Model IDs are strings and each
descriptor supplies its runtime, supported backends, capabilities, and
optional speculative-decoding variant.

A descriptor is available to the generation picker exactly when:

```text
(STREAMING or OPENAI_API) and not VISION_ENCODER
```

In a QNN-enabled build, the standalone `vjepa2-qnn` `VISION_ENCODER` remains
visible through `ModelCatalog.all()` and `byId()`, but is excluded from the
generative family/runtime/backend picker. `createEngine(descriptor)` selects
`NativeQuickDotAI` or `LiteRTLm` from `descriptor.runtime`; `load` then verifies
the model ID, backend, and speculative variant against that descriptor.

## Threading

Generation blocks its invoking worker thread while delivering stream events.
A `QuickDotAI` instance is not generally thread-safe:

- serialize `load`, generation, `metrics`, `encode`, `unload`, and `close`;
- do not call those methods from a `StreamSink` callback;
- use `cancel` as the only cross-thread operation;
- marshal callbacks to Android's main thread before updating UI.

See [AsyncAndStreaming.md](AsyncAndStreaming.md) for the detailed event
contract.

## Packaging

From `Applications/CausalLM`, a CPU package is the default:

```bash
export ANDROID_NDK=/path/to/android-ndk
./build_android.sh --app
```

QNN is opt-in:

```bash
export QNN_SDK_ROOT=/path/to/qnn-sdk
./build_android.sh --qnn --app
```

The script stages public native dependencies into
`Android/QuickDotAI/prebuilt_libs/`; Gradle builds `libquickai_jni.so` and the
AAR. It does not stage a proprietary model plugin or
`htp_backend_ext_config.json`.

For QNN, the native loader first honors
`QUICK_DOT_AI_QNN_BACKEND_EXT_CONFIG_PATH`. Otherwise it looks for
`htp_backend_ext_config.json` under `modelBasePath`, or under the parent when
`modelBasePath` ends in `/models`. This is process-wide: the environment or
first QNN load determines the path reused by later loads.

See [QuickDotAI/README.md](QuickDotAI/README.md) for application integration.
