# Android Architecture 📱

This document describes the current Android state of Quick.AI and separates it
from the planned REST/foreground-service layer that older documents described
as if it already existed.

## ✅ Current Gradle Modules

The Android build currently includes:

```text
Android/
├── QuickDotAI/       # AAR module
└── SampleTestAPP/    # Direct sample app using the AAR
```

`Android/settings.gradle.kts` includes only `:QuickDotAI` and
`:SampleTestAPP`.

## 🧱 QuickDotAI AAR

`QuickDotAI` exposes the public Kotlin API in
`com.example.quickdotai`.

Key files:

| File | Role |
|---|---|
| `QuickDotAI.kt` | Public `runText` / `runOpenAI` interface and `BackendResult` / `StreamSink` contracts |
| `Types.kt` | OpenAI request and image-sidecar types, load options, errors, metrics |
| `NativeQuickDotAI.kt` | Kotlin wrapper around one native `CausalLmHandle` |
| `NativeCausalLm.kt` | Low-level JNI declarations |
| `LiteRTLm.kt` | LiteRT-LM engine wrapper for the `gemma4` (`ModelIds.GEMMA4`) model |
| `LlavaNextImagePreprocessor.kt` | Public model-specific LLaVA-NeXT image-sidecar helper |
| `LlavaNextImageProcessor.kt` | Internal any-resolution patching implementation |
| `PilloBilinearResizer.kt` | Pillow-compatible bilinear resize used by the image processors |
| `src/main/cpp/quickai_jni.cpp` | JNI bridge to `quick_dot_ai_api.h` |
| `src/main/cpp/CMakeLists.txt` | Builds `libquickai_jni.so` and links `libquick_dot_ai_api.so` |

## 🔌 Native Path

`NativeQuickDotAI` owns one native handle:

```text
NativeQuickDotAI
  └── NativeCausalLm.ensureLoaded()
      ├── System.loadLibrary("qnn_context") (optional)
      └── System.loadLibrary("quickai_jni")
            └── links/calls libquick_dot_ai_api.so
```

The native API surface is declared in `api/quick_dot_ai_api.h`. Generation is
reduced to two preferred handle-based calls:

- `loadModelHandleByName` (dispatched from Kotlin via the
  `loadModelHandleByNameNative` JNI declaration in `NativeCausalLm.kt`)
- `quickAiRunText` for an exact, already-formatted prompt
- `quickAiRunOpenAI` for text, structured output, tools, and optional image
  tensor sidecars in one OpenAI-compatible request
- `cancelModelHandle`
- `destroyModelHandle`

The matching Android paths are:

```text
QuickDotAI.runText(text, sink)
  -> NativeCausalLm.runTextStreamingNative
  -> quickAiRunText

QuickDotAI.runOpenAI(OpenAIRequest(json, imageTensors), sink)
  -> NativeCausalLm.runOpenAIStreamingNative
  -> quickAiRunOpenAI
```

`runText` never applies a chat template or implicit conversation state. The
caller owns every byte of the model prompt. `runOpenAI` validates the JSON,
applies the loaded model's chat template, and enables xgrammar for
`response_format` (`json_object` / `json_schema`) or an explicitly required or
named tool/function. An `auto` tool choice remains unconstrained because the
model may produce a normal assistant response.

For a constrained tool call, xgrammar targets the normalized full envelope
`{"name": ..., "arguments": {...}}`, not only the tool's raw parameter schema.

### Multimodal OpenAI requests

On the native path, images remain identified by ordered `image_url` content
parts in the JSON, but preprocessed float tensors are passed out-of-band.
`OpenAIImageTensor.source`
must exactly equal its `image_url.url`, and the tensor list must have the same
count and order as those parts. This keeps large float arrays out of JSON and
does not imply that the native library fetches URLs.

Dense HWC/CHW tensors must satisfy
`pixelValues.size == patchCount * channels * patchHeight * patchWidth`.
`MODEL_NATIVE` is reserved for a tensor already transformed for the selected
vision model. The sidecar is versioned by `OpenAIImageTensorSidecar`; the
current version is `1`. The current native fused path supports one image and a
compatible vision-encoder/LLM handle.

The LiteRT-LM path uses the same `image_url` parts without a tensor sidecar.
It maps `data:image/...;base64` sources to `Content.ImageBytes` and validated
`file://` sources to `Content.ImageFile`, preserving mixed content order across
initial messages and the final user message. Network and custom URL schemes
are not fetched and return `UNSUPPORTED`.

## ModelCatalog

Model selection in the AAR is driven by the `ModelCatalog` singleton. Models
are identified by string ids rather than an enum.

### Seeding

`ModelCatalog` is seeded on first access by calling `nativeQueryCatalog()`
through JNI, which delegates to `getModelCatalogJson()` in
`libquick_dot_ai_api.so`. Hardcoded LiteRT descriptors (e.g., `gemma4`) are
merged in at the Kotlin layer.

### Key types

| Type | Role |
|---|---|
| `enum class RuntimeKind { NATIVE, LITERT }` | Selects the engine path |
| `enum class Capability` | Per-model feature flags in `ModelCatalog.kt` |
| `data class ModelDescriptor(id, family, displayName, runtime, backends, capabilities)` | Descriptor from the catalog |
| `object ModelIds` | String constants for well-known model ids |
| `object ModelCatalog` | Singleton: `all()`, `families()`, `selectable()`, `selectableFamilies()`, `runtimesFor(family)`, `backendsFor(family, rt)`, `resolve(family, rt, backend)`, `byId(id)` |

### 3-axis cascading UI

`SampleTestAPP` presents a 3-axis cascading UI:

1. **Family** — populated from `ModelCatalog.selectableFamilies()`
2. **Runtime chip row** — populated from `ModelCatalog.runtimesFor(selectedFamily)`
3. **Backend chip row** — populated from `ModelCatalog.backendsFor(selectedFamily, selectedRuntime)`

The app lists only **selectable** (generative) models. Embedding-only models
such as `tiny-bert` and standalone vision encoders such as `vjepa2-qnn` are
filtered out by `selectableFamilies()`. They remain in the AAR catalog for
capability discovery and native model pairing.

The resolved descriptor is obtained via `ModelCatalog.resolve(family, runtime, backend)`
and passed directly to `createEngine()`.

### Engine factory

```kotlin
createEngine(descriptor: ModelDescriptor): QuickDotAI
```

`createEngine` dispatches to `NativeQuickDotAI` (for `RuntimeKind.NATIVE`) or
`LiteRTLm` (for `RuntimeKind.LITERT`) and binds the resulting engine to the
descriptor. `load()` rejects a different model id or a backend outside that
catalog entry. A declared speculative-decoding variant is the only alternate
id accepted when speculative decoding is requested.

### LoadModelRequest

`LoadModelRequest.modelId` is a `String` catalog id. An already-loaded engine
accepts only an identical full `LoadModelRequest`; otherwise it must be
unloaded first. The descriptor passed to `createEngine()` and the load request
therefore cannot drift apart. The JNI call dispatched on native load is
`loadModelHandleByNameNative`.

## 🌗 LiteRT Runtime Path

`LiteRTLm` is selected for the `gemma4` (`ModelIds.GEMMA4`) model and takes a
`.litertlm` file path through `LoadModelRequest.modelPath`. Its public OpenAI
adapter supports ordered text and locally resolvable image content, and rejects
fields or URL schemes it cannot represent faithfully.

## 🧵 Threading Model

A `QuickDotAI` instance is not internally thread-safe. Host apps should drive a
loaded engine from one worker thread. `SampleTestAPP` follows this pattern with
a background dispatcher.

Streaming callbacks are delivered to the caller-provided `StreamSink`.
Apps that update UI must marshal callbacks to the main thread.
Callbacks must not re-enter the same engine while generation is active because
the native path holds its handle lock. Cross-thread `cancel()` is the sole
exception; the same conservative rule applies to terminal callbacks.

## 🧪 SampleTestAPP

`SampleTestAPP` is the current runnable Android sample. It links the
`:QuickDotAI` module directly; it does not start a REST service and does not
communicate over sockets.

## 🗺️ Planned Service Layer

The following pieces are design targets, not current Gradle modules:

| Planned component | Status |
|---|---|
| `LauncherApp` foreground-service bootstrap UI | Planned |
| `QuickAIService` remote foreground service | Planned |
| NanoHTTPD loopback REST server | Planned |
| `RequestDispatcher`, `ModelRegistry`, `ModelWorker` | Planned |
| Standalone REST client app | Planned |

When implemented, the service layer should wrap the same `QuickDotAI` AAR
contract rather than inventing a separate model API.

## 📦 Packaging

`apk-build-install.sh` performs the current full Android workflow:

1. Build native libraries with `./build.sh --platform=android --enable-qnn --clean`.
2. Install/copy native shared libraries through `apk_install_android.sh`.
3. Copy `.so` files into `Android/QuickDotAI/prebuilt_libs/`.
4. Run Gradle install for `:SampleTestAPP`.

Set `NDK_ROOT` inside `apk-build-install.sh` before using it on a new machine.

## 📎 Related Docs

- [QuickDotAI AAR API](QuickDotAI/README.md)
- [Android Native Async & Streaming](AsyncAndStreaming.md)
- [Main README](../README.md)
