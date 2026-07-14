# Quick.AI C API

This directory contains the C API used by the CausalLM application and the
QuickDotAI Android AAR. The public declarations are in
`quick_dot_ai_api.h`; applications link `libquick_dot_ai_api.so`, which uses
`libcausallm.so` and `libnntrainer.so`.

## Generation API

New callers should express generation with two functions:

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

Both functions stream token deltas through `callback` and block the invoking
thread until generation finishes or is cancelled. Returning non-zero from the
callback requests cooperative cancellation.

### Exact text: `quickAiRunText`

`quickAiRunText` passes the supplied UTF-8 bytes to the selected text model as
an already-formatted prompt. It does not parse JSON, apply a chat template, add
roles, reuse an implicit conversation prompt, or enable a grammar.

Use it when the caller already owns the complete model prompt:

```c
static int on_token(const char *delta, void *user_data) {
  (void)user_data;
  fputs(delta, stdout);
  return 0;
}

ErrorCode error = quickAiRunText(handle, already_formatted_prompt,
                                 on_token, NULL);
```

### OpenAI request: `quickAiRunOpenAI`

`quickAiRunOpenAI` accepts one OpenAI Chat Completions-style JSON object. It:

1. validates `messages` and their ordered `text` / `image_url` content parts;
2. correlates image occurrences with sidecars and checks the loaded
   descriptor's `MULTIMODAL` / `MULTI_IMAGE` capabilities;
3. renders the request with the loaded model's chat template;
4. applies xgrammar when structured output is explicitly required;
5. dispatches image requests to a plugin-provided fused/composite hook or to
   the generic `[vision encoder, embedding-input LLM]` composer; and
6. streams generated deltas through the callback.

Text-only requests pass `NULL, 0` for the image sidecar:

```c
const char *request =
  "{\"messages\":[{\"role\":\"user\","
  "\"content\":\"Explain on-device inference briefly.\"}]}";

ErrorCode error = quickAiRunOpenAI(handle, request, NULL, 0,
                                   on_token, NULL);
```

The native runner implements an explicit Chat Completions-compatible subset:

- `messages` supports `system`, `developer`, `user`, `assistant`, and `tool`
  roles. Content is a string or ordered `text` / `image_url` parts; images are
  accepted only in `user` messages.
- `tools` + `tool_choice` and the legacy `functions` + `function_call` pair are
  supported for function tools. Modern and legacy controls cannot be mixed.
- `response_format`, `model`, `stream`, `user`, `metadata`, and
  `parallel_tool_calls: false` are accepted. `add_generation_prompt` is also
  accepted as a local chat-template extension.
- `model` is informational: the loaded handle remains authoritative. The C API
  always streams through its callback regardless of the JSON `stream` value.
  `store: false` is accepted, while `store: true` and
  `parallel_tool_calls: true` return `CAUSAL_LM_ERROR_UNSUPPORTED` because the
  local runner neither stores completions nor emits multiple tool calls.

Sampling and length controls such as `temperature`, `top_p`, `seed`,
`max_tokens`, and `max_completion_tokens` are not silently ignored. Until the
per-run model controls can honor them, their presence returns
`CAUSAL_LM_ERROR_UNSUPPORTED`.

The request can constrain decoding with either response formats or tools:

- `response_format.type = "json_object"` enables JSON-object decoding.
- `response_format.type = "json_schema"` compiles the supplied
  `json_schema.schema`.
- a named tool/function choice, or `tool_choice = "required"`, compiles a
  normalized `{"name": ..., "arguments": {...}}` tool-call envelope whose
  `arguments` field uses the selected function parameter schema.
- `tool_choice = "auto"` does not force a grammar because the model may choose
  an ordinary assistant response. An independent `response_format` still
  applies.

For example:

```json
{
  "messages": [
    {"role": "user", "content": "Return a short status."}
  ],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "status",
      "schema": {
        "type": "object",
        "properties": {"status": {"type": "string"}},
        "required": ["status"],
        "additionalProperties": false
      }
    }
  }
}
```

A model directory normally provides a usable chat template for this API. The
bundled Gemma4 architectures also have a text-only fallback format; tool
definitions and tool history still require a tokenizer-supplied tool-aware
template. Structured output additionally requires xgrammar to be initialized
for the model.

## Multimodal sidecar

OpenAI JSON identifies images while preprocessed float tensors travel
out-of-band in `QuickAiImageTensorV1`. The library does not download or decode
the URL.

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image."},
      {
        "type": "image_url",
        "image_url": {"url": "quickai://image/0", "detail": "high"}
      }
    ]
  }]
}
```

The sidecar contract is:

- provide exactly one tensor for every `image_url` part;
- preserve message/content order in the sidecar array;
- set `source` to the exact corresponding `image_url.url` string;
- set `struct_size = sizeof(QuickAiImageTensorV1)`;
- keep `source` and `values` alive for the blocking call;
- set non-zero `value_count`, `patch_count`, `channels`, `patch_height`,
  `patch_width`, `original_height`, and `original_width`;
- for `QUICK_AI_IMAGE_LAYOUT_HWC` and `QUICK_AI_IMAGE_LAYOUT_CHW`, ensure
  `value_count == patch_count * channels * patch_height * patch_width`;
- use `QUICK_AI_IMAGE_LAYOUT_MODEL_NATIVE` only for a representation already
  prepared for the selected vision model.

When present, `image_url.detail` must be `auto`, `low`, or `high`. It is
descriptive at this layer: tensor preprocessing has already happened before
the call, so the library does not resize the supplied sidecar again.

Image sidecars are accepted only when the loaded catalog descriptor advertises
`QDA_CAP_MULTIMODAL`. More than one sidecar additionally requires
`QDA_CAP_MULTI_IMAGE`; otherwise the request returns
`CAUSAL_LM_ERROR_UNSUPPORTED` before inference. A multi-image caller passes one
array element per `image_url` occurrence and sets `image_count` to the complete
array length.

`quickAiRunOpenAI` chooses the execution topology behind the same public entry
point. It first looks in `OpenAIMultimodalCallbackRegistry` for the loaded
architecture, which lets `libqai_ext_model.so` drive a self-contained fused
model or another plugin-owned composite. If no V2 hook is registered, the
adapter can use the shape-limited legacy callback or a compatible handle laid
out as `[vision encoder, embedding-input LLM]`. This dispatch is based on
descriptor capabilities and runtime model interfaces, not on whether the
build is QNN or whether the model is present in the public source tree.
Missing capabilities, an unsupported image count/layout, or an incompatible
execution path return `CAUSAL_LM_ERROR_UNSUPPORTED`.

The sidecar metadata is preserved for a fused plugin hook. The generic pair
path may canonicalize CHW input for its vision producer; HWC and model-native
representations remain model contracts. A V2 plugin hook must honor the
supplied layout/count and any non-null grammar, or explicitly return
`CAUSAL_LM_ERROR_UNSUPPORTED` rather than silently ignore them. Once a V2 hook
is invoked, its result is final; the dispatcher does not try another model path
after externally visible token or model-state changes. The callback arguments
are borrowed for the synchronous call and must not be retained.

The V2 callback receives the validated `causallm::openai::Request`, a versioned
view of every loaded sub-model, and a nullable core-formatted prompt. A null
formatted prompt means that the core has no compatible chat template; a fused
model may then render the validated messages with its own model-specific
template. The callback model/text indices identify the registration owner and
the conventional generation model without exposing the private
`CausalLmModel` layout. If the callback uses the core-formatted prompt, that
template must represent image occurrences in the form expected by the model.

The legacy `ModelCallbacks::multimodal_streaming` adapter is retained for
existing extension models. Its signature cannot describe general tensor
metadata or xgrammar, so it is invoked only for one unconstrained image made of
dense RGB 512x512 patches. HWC is passed directly and CHW is canonicalized to
HWC; model-native, multi-image, constrained, and other-shaped requests bypass
the legacy hook and may use the generic composer when compatible.

```c
QuickAiImageTensorV1 image = {0};
image.struct_size = sizeof(image);
image.source = "quickai://image/0";
image.values = pixels;
image.value_count = pixel_count;
image.layout = QUICK_AI_IMAGE_LAYOUT_CHW;
image.patch_count = patch_count;
image.channels = 3;
image.patch_height = patch_height;
image.patch_width = patch_width;
image.original_height = original_height;
image.original_width = original_width;

ErrorCode error = quickAiRunOpenAI(handle, request, &image, 1,
                                   on_token, NULL);
```

### Optional fused/composite plugin

The public source distribution need not contain every multimodal model. A
downstream `libqai_ext_model.so` may register a catalog descriptor, its model
factory entry, and an architecture callback at load time. New fused or
multi-image implementations register an `OpenAIMultimodalStreamingCallback`
with `OpenAIMultimodalCallbackRegistry`; the older `ModelCallbacks` table
remains size-stable for its existing by-value registration ABI. The descriptor
can represent either one fused model or an already-defined composite; callers
use `loadModelHandleByName()` and `quickAiRunOpenAI()` in both cases. The low-level
`loadMultimodalHandleByName()` helper remains available for constructing a
compatible `[vision, LLM]` pair, but it does not add another generation API.

This plugin boundary shares C++ virtual interfaces and callback registry types.
The plugin, `libcausallm.so`, and `libquick_dot_ai_api.so` must be rebuilt from
the same source revision. Compatibility with an older binary plugin is not
promised even when its exported filename is unchanged.

The fused/composite and multi-image contracts have been reviewed with static
source/header checks. Actual plugin execution, streaming, cancellation, and
Android device behavior have not been validated on this host.

## Loading and lifecycle

Prefer catalog string IDs so applications do not duplicate the model list:

```c
Config config = {0};
config.verbose = false;
setOptions(config);

CausalLmHandle handle = NULL;
ErrorCode error = loadModelHandleByName(
  CAUSAL_LM_BACKEND_CPU,
  "qwen3-0.6b",
  CAUSAL_LM_QUANTIZATION_W4A32,
  NULL,                 /* native library directory */
  model_base_path,
  &handle);

if (error == CAUSAL_LM_ERROR_NONE) {
  error = quickAiRunOpenAI(handle, request, NULL, 0, on_token, NULL);
}

destroyModelHandle(handle);
```

QNN deployments expect `htp_backend_ext_config.json` beside the model
collection root. When `model_base_path` ends in `/models`, the loader uses its
parent directory; otherwise it uses `model_base_path` itself. An existing
`QUICK_DOT_AI_QNN_BACKEND_EXT_CONFIG_PATH` environment variable overrides this
default.

Each handle owns its model state and serializes its own operations. Different
handles may be driven from different worker threads. Use
`cancelModelHandle(handle)` from another thread to request cancellation, and
always call `destroyModelHandle` when finished.

## Common errors

- `CAUSAL_LM_ERROR_INVALID_PARAMETER`: null arguments, malformed OpenAI JSON,
  unsupported content parts, or a mismatched/invalid image sidecar.
- `CAUSAL_LM_ERROR_NOT_INITIALIZED`: the handle has no loaded model.
- `CAUSAL_LM_ERROR_UNSUPPORTED`: the selected model lacks a required chat
  template, grammar setup, or multimodal path.
- `CAUSAL_LM_ERROR_INFERENCE_FAILED`: model, grammar compilation, or generation
  failed after validation.

The former message-array, tool-specific, raw-prompt, and standalone multimodal
generation functions were removed before release. Generation has exactly the
two entry points above; callers that need a complete blocking response should
accumulate callback deltas.
