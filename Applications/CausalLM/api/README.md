# QuickDotAI C API

`libquick_dot_ai_api` is the public, handle-based C interface for CausalLM.
Application calls are declared in `quick_dot_ai_api.h`. Native model plugins
use the separate, versioned C ABI in `quick_dot_ai_extension_api.h`.

With a normal Meson install, both headers are installed under:

```text
<includedir>/nntrainer/causallm/
  quick_dot_ai_api.h
  quick_dot_ai_extension_api.h
```

Link applications against `libquick_dot_ai_api`; its `libcausallm` and
NNTrainer dependencies must also be visible to the platform loader.
`quick_dot_ai_api_internal.h` contains JNI-only cancellation coordination and
is not an installed application header.

## Build and validation

From the repository root on x86 Linux:

```bash
git submodule update --init --recursive
meson setup build-causallm-check \
  -Denable-transformer=true \
  -Denable-api=true \
  -Denable-test=true
ninja -C build-causallm-check \
  quick_dot_ai_test \
  unittest_openai_request \
  unittest_xgrammar_manager
meson test -C build-causallm-check --print-errorlogs \
  unittest_openai_request \
  unittest_xgrammar_manager
```

The Android standalone build is documented in the
[parent README](../README.md#android-build).

## Handle lifecycle

A normal application flow is:

1. Read `getModelCatalogJson()` and select a descriptor and supported backend.
2. Call `loadModelHandleByName()`.
3. For a descriptor with `SPECULATIVE`, optionally call
   `configureSpeculativeDecoding()`.
4. Run `quickAiRunText()` or `quickAiRunOpenAI()`.
5. Read `getPerformanceMetricsHandle()` and optionally use the handle-based QNN
   KV-cache functions.
6. Call `destroyModelHandle()` to release the handle. If weights must be
   released earlier, call `unloadModelHandle()` first, but still call
   `destroyModelHandle()` afterward to free the retained empty handle object.

`cancelModelHandle()` is the operation intended to interrupt a generation from
another thread. Encoder descriptors use `encodeModelHandle()` or
`encodeImageModelHandle()` and their matching free functions instead of a
generation entry point.

`quant_type` remains in the native load ABI and is validated. Current
file-based descriptors select their concrete model variant through the catalog
ID and configuration; the loader does not append a quantization suffix to the
model directory.

The maintained surface is:

- `setOptions`
- `loadModelHandleByName`
- `quickAiRunText`, `quickAiRunOpenAI`
- `configureSpeculativeDecoding`
- `saveQnnKvCacheHandle`, `loadQnnKvCacheHandle`, `resetQnnKvCacheHandle`
- `getPerformanceMetricsHandle`
- `cancelModelHandle`, `unloadModelHandle`, `destroyModelHandle`
- `encodeModelHandle`, `freeEmbedding`
- `encodeImageModelHandle`, `freeImageEmbedding`
- `getModelCatalogJson`

Refer to `quick_dot_ai_api.h` for pointer ownership, structure sizes, and exact
error codes.

`getModelCatalogJson()` returns records with string `id`, `family`, and
`display_name` fields; numeric `runtime`, `backend_mask`, and `capabilities`
fields; and an optional `sd_variant_id`. Interpret those numeric fields with
the installed header's `QUICK_AI_RUNTIME_*`, `QUICK_AI_BACKEND_MASK_*`, and
`QUICK_AI_CAP_*` constants. The returned buffer is thread-local library
storage: copy it before the next catalog call on the same thread.

## Generation contracts

Both generation calls block the invoking thread and synchronously deliver UTF-8
deltas to `CausalLmTokenCallback`. Returning nonzero requests cooperative
cancellation. Accumulate deltas in the caller when a complete string is
needed.

`quickAiRunText()` sends non-empty, already-formatted text exactly as supplied.
It adds no chat template, role, JSON interpretation, grammar, or conversation
history.

`quickAiRunOpenAI()` accepts one complete OpenAI Chat Completions-style JSON
object. The request must carry the full conversation history for that run; the
C API has no hidden chat session. The loaded handle, not the optional JSON
`model` field, selects the model.

The native callback API rejects requests it cannot honor instead of silently
changing their meaning. In particular, `stream: false`, an explicitly false
`response_format.json_schema.strict`, a `strict` field on a function, and
unsupported sampling or length controls return
`CAUSAL_LM_ERROR_UNSUPPORTED`.

## Images and model capabilities

Image requests passed to `quickAiRunOpenAI()` require one
`QuickAiImageTensorV1` sidecar for each `image_url` occurrence, in occurrence
order. Each sidecar `source` must exactly match the corresponding JSON URL.
The descriptor must advertise `MULTIMODAL`; multiple images also require
`MULTI_IMAGE`.

Native images run only through a successfully registered versioned model
extension. There is no generic vision-to-LLM composer or legacy multimodal
fallback. After the extension callback is invoked, its result is authoritative,
including `UNSUPPORTED`.

In a QNN-enabled build, `vjepa2-qnn` is a standalone `VISION_ENCODER`. It can
use `encodeImageModelHandle()` but is not a generation model.

## Versioned model extensions

`quick_dot_ai_extension_api.h` is a C/POD boundary. A plugin:

1. Obtains host metadata with `quickAiGetExtensionHostInfoV1()`.
2. Registers a `QuickAiModelExtensionV1` with
   `quickAiRegisterModelExtensionV1()`.
3. Supplies compile-time ABI major/minor, build tag, Transformer ABI version,
   fixed-width descriptor data, and plain function pointers.

Registration requires the exact ABI contract, build tag, and Transformer ABI
expected by the host. Registration copies descriptor strings, but callback
function pointers and `user_data` remain plugin-owned. A successfully
registered plugin must remain loaded for the rest of the process; unregister
and plugin unload are unsupported.

Extension callback arguments are borrowed only for the synchronous invocation,
and C++ exceptions must never cross the ABI. Packages should build the host and
plugin from the same revision. The extension registration API publishes a
descriptor and callbacks; a plugin that introduces a new C++ architecture must
also register its constructor with the shared CausalLM `Factory`. The public
Android build does not provide a proprietary plugin.
