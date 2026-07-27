# CausalLM inference with NNTrainer

CausalLM runs transformer inference on NNTrainer. The directory provides:

- `nntr_causallm`, a path-based command-line application.
- `libcausallm`, the model, tokenizer, and grammar runtime.
- `libquick_dot_ai_api`, a handle-based C API.
- the Android `QuickDotAI` AAR, with native and LiteRT-LM backends.

CPU is the default backend. Android QNN support is opt-in.

## Supported models

- Llama
- Qwen3 (0.6B, 1.7B, 4B, 8B, 14B, 32B)
  [[link](https://huggingface.co/Qwen/Qwen3-4B)]
- Qwen3-MoE (30B-A3B)
  [[link](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)]
- GPT-OSS (MoE: 20B, 120B)
  [[link](https://huggingface.co/openai/gpt-oss-20b)]
- Custom models implemented with NNTrainer layers

See [models/README.md](models/README.md) for model-specific configuration and
conversion notes.

## Performance

Measured on a **Galaxy S26 Ultra (SM-S948U)**, CPU backend, with **Qwen3-0.6B**
quantized to **Q4_0** FC weights + **Q6_K** embedding / LM head. `prefill` is
prompt-encode throughput, and `decode` is autoregressive generation throughput
for 32 new tokens. Flash (GEMM) attention engages for prompts with at least 32
tokens.

| Activation | Threads | Prompt | Prefill (tok/s) | Decode (tok/s) |
| --- | --- | --- | --- | --- |
| **Q4_0-FP16** | 8 | 437 | **755** | **80** |
| Q4_0-FP16 | 8 | 1003 | 596 | 60 |
| Q4_0-FP16 | 4 | 1003 | 423 | 52 |
| Q4_0-FP32 | 8 | 437 | 329 | 77 |
| Q4_0-FP32 | 8 | 1003 | 299 | 50 |
| Q4_0-FP32 | 4 | 1003 | 206 | 45 |

`Q4_0-FP16` (FP16 activation) is the recommended device configuration: it
provides about twice the prefill throughput of the FP32-activation path on the
FP16 build and is token-coherent with it. Prefill throughput drops as the
prompt grows because attention is O(n²). A very short prompt, such as the
18-token default `sample_input`, reports a much lower number (about 240 tok/s)
because it is below the flash-attention threshold and dominated by fixed setup
cost. Peak RSS is approximately 0.94 GB.

## QuickDotAI API

The public API is a handle-based C interface. Load a catalog descriptor with
`loadModelHandleByName()`, then use one of the two generation contracts:

- `quickAiRunText()` accepts non-empty, already-formatted model input exactly
  as supplied. It does not add a chat template, roles, or conversation history.
- `quickAiRunOpenAI()` accepts one complete OpenAI Chat Completions-style JSON
  object. It renders messages with the model's Hugging Face chat template and
  the request must include the complete history for that run.

Both calls stream synchronous UTF-8 deltas through a callback. The API has no
hidden chat session. Native image requests use the separate, versioned model
extension ABI and have no generic or legacy multimodal fallback.

See [api/README.md](api/README.md) for lifecycle, ownership, capability, image,
extension ABI, and exact error-code contracts.

## Model directory

A native model directory normally contains:

- `config.json`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `nntr_config.json`
- the weight file named by `nntr_config.json`

When `tokenizer_config.json` contains a Hugging Face `chat_template`,
OpenAI-style requests are rendered with that template. `quickAiRunText`, by
contrast, accepts already-formatted text exactly as supplied.

### Chat template renderer

CausalLM includes a small Jinja2-compatible renderer for the constructs
commonly used by Hugging Face chat templates:

| Feature | Example |
| --- | --- |
| For loops | `{% for message in messages %}...{% endfor %}` |
| Conditionals | `{% if %}...{% elif %}...{% else %}...{% endif %}` |
| Output expressions | `{{ bos_token }}` |
| Variable assignment | `{% set offset = 1 %}` |
| Dictionary and array access | `message['role']`, `messages[0]` |
| String concatenation | `'<\|im_start\|>' + message['role']` |
| Comparison operators | `==`, `!=`, `>`, `<`, `>=`, `<=` |
| Boolean operators | `and`, `or`, `not` |
| Loop variables | `loop.first`, `loop.last`, `loop.index`, `loop.index0` |
| Filters | `\| trim`, `\| length`, `\| tojson` |
| String methods | `.strip()`, `.startswith()`, `.upper()`, `.split()` |
| Containment test | `'keyword' in message['content']` |
| Namespace | `namespace()` for cross-scope variable mutation |
| Whitespace control | `{%- -%}`, `{{- -}}` |

The path-based `nntr_causallm` CLI treats an explicit command-line prompt as
exact input. When the prompt is omitted, it applies the model template to
`nntr_config.json`'s `chat_input`. The C API makes the distinction explicit:
`quickAiRunOpenAI()` performs message rendering, while `quickAiRunText()` is
the exact-input entry point.

## Linux build

Initialize the nested dependencies, configure the transformer application, and
build:

```bash
git submodule update --init --recursive
meson setup build-causallm \
  -Denable-transformer=true \
  -Denable-api=true
ninja -C build-causallm
```

Run a native model:

```bash
export NNTR_NUM_THREADS=4
./build-causallm/Applications/CausalLM/nntr_causallm \
  /path/to/model "Hello"
```

The C API library and model-dependent client are
`build-causallm/Applications/CausalLM/libquick_dot_ai_api.so` and
`quick_dot_ai_test`. See [api/README.md](api/README.md) for the API contract.

### x86 Linux validation

The request parser and xgrammar tests are registered under these exact Meson
names:

```bash
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

`quick_dot_ai_test` is a model-dependent client rather than a registered unit
test. Run it separately with a compatible model installation when validating
inference.

## Windows build

Windows CausalLM builds need a `tokenizers_c.lib` that matches the local MSVC
toolchain. The repository keeps the Linux static library in
`Applications/CausalLM/lib/`; Windows builds generate the matching library from
source instead of carrying a checked-in binary.

### Prerequisites

- Visual Studio Build Tools with the MSVC C++ toolchain
- Meson and Ninja
- Rust (`cargo`) from https://rustup.rs/

### Build the tokenizer library

Meson builds the default `tokenizers_c.lib` automatically when it is missing.
The helper can also be run directly to pre-build or refresh the library:

```powershell
powershell -ExecutionPolicy Bypass `
  -File Applications\CausalLM\build_tokenizer_windows.ps1 `
  -BuildDir build-causallm-win
```

The build writes the default Meson input under the build directory:

```text
build-causallm-win\tokenizers_c_win\target\release\tokenizers_c.lib
```

For Windows cross builds, Meson passes the matching Rust target triple and
writes the library under `target\<triple>\release\`.

If you already have a compatible `tokenizers_c.lib`, pass it explicitly during
Meson setup:

```powershell
meson setup build-causallm-win `
  -Dplatform=windows `
  -Denable-transformer=true `
  -Dcausallm-tokenizer-lib=C:\path\to\tokenizers_c.lib
```

When using a DLL import library instead of a static library, make sure the
matching `tokenizers_c.dll` is available on `PATH` at runtime.

### Build and run

```powershell
meson setup build-causallm-win `
  -Dplatform=windows `
  -Denable-transformer=true `
  -Denable-test=false
ninja -C build-causallm-win nntr_causallm
$build = Resolve-Path build-causallm-win
$dllDirs = Get-ChildItem $build -Filter *.dll -Recurse |
  ForEach-Object { Split-Path -Parent $_.FullName } |
  Sort-Object -Unique
$env:PATH = (($dllDirs + @($build, "$build\Applications\CausalLM")) -join ";") +
  ";" + $env:PATH
$env:NNTR_NUM_THREADS = "4"
.\build-causallm-win\Applications\CausalLM\nntr_causallm.exe `
  C:\path\to\model "Hello from Windows"
```

## Android build

`build_android.sh` is the single Android entry point. It builds the public CPU
native artifacts by default and does not invoke Gradle or modify a device:

### Prerequisites

- Android NDK (`ANDROID_NDK` or `NDK_ROOT`)
- Meson, Ninja, CMake, and Rust for tokenizers-cpp
- ADB only when using `--install`
- QNN SDK (`QNN_SDK_ROOT`) only when using `--qnn`

```bash
export ANDROID_NDK=/path/to/android-ndk
cd Applications/CausalLM
./build_android.sh
```

CPU and QNN native outputs are placed in `builddir_app/cpu/` and
`builddir_app/qnn/`. To additionally build the standalone app and AAR without
installing them, run `./build_android.sh --app`. That mode stages
`libcausallm.so`, `libquick_dot_ai_api.so`, and their runtime dependencies;
Gradle adds `libquickai_jni.so` and writes the debug AAR under
`Android/QuickDotAI/build/outputs/aar/`.

The most useful options are:

| Option | Behavior |
| --- | --- |
| `--app` | Add the QuickDotAI AAR and sample APK to the canonical native build. |
| `--install` | Push native libraries and tools; with `--app`, also install the built APK. |
| `--qnn` | Build the QNN variant and include its runtime libraries; requires `QNN_SDK_ROOT`. CPU is the default. |
| `--cache` | Reuse a compatible nntrainer Android engine build, or build it when absent. |
| `--clean` | Recreate the selected CPU or QNN CausalLM build directory. |
| `--nntr-threads=N` | Set the positive NNTrainer compute-thread count. |

`--app`, `--install`, and QNN selection are independent. For example,
`--install` pushes only native artifacts, while `--app --install` also installs
the APK. Set `ANDROID_SERIAL` when more than one device is connected.

To build the QNN variant:

```bash
export QNN_SDK_ROOT=/path/to/qnn-sdk
./build_android.sh --qnn --app
```

QNN deployment supports HTP V75, V79, and V81 SDK runtimes and requires at
least one complete matching Stub/Skel pair.

The public build stages neither a proprietary `libqai_ext_model.so` model
extension nor `htp_backend_ext_config.json`. A downstream package may add the
plugin after `libquickai_jni.so` is loaded, but the plugin must use the exact
extension ABI, build tag, and Transformer ABI from the same source revision and
must remain loaded for the process lifetime.

For a QNN model, the native loader uses
`QUICK_DOT_AI_QNN_BACKEND_EXT_CONFIG_PATH` when that environment variable is
set. Otherwise it appends `htp_backend_ext_config.json` to
`modelBasePath`; if that path ends in `/models`, it uses the parent directory.
This setting is process-wide: the environment or first QNN load determines the
path reused by later loads. The packager is responsible for putting the JSON at
that resolved location.

See [Android/QuickDotAI/README.md](Android/QuickDotAI/README.md) for AAR usage.

## Quantizing models

`nntr_quantize` converts FP32 CausalLM weights to lower-precision data types,
reducing model size for efficient on-device inference.

### Supported quantization types

| Data type | Description |
| --- | --- |
| `FP32` | 32-bit floating point (default for embedding and LM head) |
| `FP16` | 16-bit floating point |
| `Q4_0` | 4-bit quantization (default for fully connected layers) |
| `Q4_K` | 4-bit K-quant quantization |
| `Q6_K` | 6-bit K-quant quantization |

> **Q4_0 platform dependency:** `Q4_0` uses architecture-specific binary
> layouts. Use `--isa ARM` when producing an ARM artifact on x86, or quantize
> on the target architecture. An x86-layout output is not directly compatible
> with ARM and vice versa.

### Prerequisites

The source model directory must contain:

- `config.json`
- `generation_config.json`
- `nntr_config.json`
- the FP32 `.bin` or `.safetensors` weight file referenced by
  `model_file_name` in `nntr_config.json`

The quantization utility is built with the CausalLM application:

```bash
ninja -C build-causallm nntr_quantize
```

### Usage

```text
nntr_quantize <model_path> [options]
```

| Option | Description | Default |
| --- | --- | --- |
| `--output`, `-o <path>` | Output directory | Same as `<model_path>` |
| `--fc_dtype <type>` | Target dtype for fully connected layers | `Q4_0` |
| `--embd_dtype <type>` | Target dtype for the embedding layer | `FP32` |
| `--lmhead_dtype <type>` | Target dtype for the LM head | Same as `embd_dtype` |
| `--output_bin <name>` | Output weight filename | Auto-generated |
| `--output_format <bin\|safetensors>` | Output weight container format | `bin` |
| `--config <path>` | Target `nntr_config.json` carrying dtype settings | None |
| `--isa <x86\|ARM\|DEFAULT>` | Target ISA for quantization | `DEFAULT` |

The input container (`.bin` or `.safetensors`) is auto-detected from
`model_file_name`, so all four input/output container combinations are
supported.

### Examples

```bash
# Q4_0 fully connected layers; embedding and LM head stay FP32.
nntr_quantize /path/to/qwen3-4b

# Q4_0 fully connected layers and Q6_K embedding/LM head.
nntr_quantize /path/to/qwen3-4b \
  --fc_dtype Q4_0 \
  --embd_dtype Q6_K

# Produce an ARM Q4_0 layout while running the tool on x86.
nntr_quantize /path/to/qwen3-4b \
  --fc_dtype Q4_0 \
  --embd_dtype Q6_K \
  --isa ARM

# Write to a separate, self-contained model directory.
nntr_quantize /path/to/qwen3-4b -o /output/qwen3-4b-q4

# Use a preconfigured target nntr_config.json.
nntr_quantize /path/to/qwen3-4b \
  --config /path/to/target/nntr_config.json

# Store quantized weights in a safetensors container.
nntr_quantize /path/to/qwen3-4b \
  --fc_dtype Q4_0 \
  --output_format safetensors
```

### Output

The utility produces:

1. A quantized `.bin` or `.safetensors` weight file.
2. A new `nntr_config_quantized.json`, or `nntr_config.json` when the output
   directory differs from the source.

If output stays in the source directory, activate the generated configuration
before running the model:

```bash
mv /path/to/model/nntr_config_quantized.json \
  /path/to/model/nntr_config.json
nntr_causallm /path/to/model
```

With `-o`, the tool copies `config.json`, `generation_config.json`, and the
tokenizer files, so the output directory is self-contained:

```bash
nntr_causallm /output/qwen3-4b-q4
```

## Quantized safetensors format

NNTrainer can store quantized weights (`Q4_0`, `Q4_K`, and `Q6_K`) in the
[safetensors](https://github.com/huggingface/safetensors) container in addition
to raw `.bin`. The quantized payload is byte-for-byte identical; only the
container differs.

### Data flow

```text
                      ┌─────────────────────────────────┐
   FP32 weights ─────▶│          nntr_quantize          │
 (.bin / .safetensors)│  GgmlQuantizer (Q4_0/Q4_K/Q6_K)│
                      └───────────────┬─────────────────┘
                                      │ --output_format
                          ┌───────────┴────────────┐
                          ▼                        ▼
                  quantized .bin          quantized .safetensors
                                            (self-describing header)
                                                    │
                  ┌─────────────────────────────────┤
                  ▼                                 ▼
       nntr_safetensors_info              nntr_causallm runtime
       (header-only inspection)       1. parse header and offsets
                                      2. mmap the data section
                                      3. point tensors at quantized blocks
```

At load time, the runtime parses only the small JSON header to obtain tensor
offsets, then memory-maps the data section. The large payload is not read
twice.

### Header layout

A safetensors file consists of
`[8-byte header length][JSON header][packed tensor data]`. Quantized tensors
are opaque byte blobs, while extension fields preserve the native NNTrainer
type and logical pre-quantization shape:

```json
{
  "__metadata__": {
    "format": "nntrainer",
    "nntr_format": "nntr-safetensors-v1",
    "nntr_q4_0_isa": "arm"
  },
  "layer0_wq:weight": {
    "dtype": "U8",
    "shape": [2359296],
    "nntr_dtype": "Q4_0",
    "nntr_shape": [1, 1, 1024, 4096],
    "data_offsets": [0, 2359296]
  },
  "output_norm:weight": {
    "dtype": "F32",
    "shape": [1, 1, 1, 1024],
    "data_offsets": [2359296, 2363392]
  }
}
```

| Field | Meaning |
| --- | --- |
| `dtype` | Standard safetensors dtype; `U8` for a block-quantized tensor |
| `shape` | Standard shape; raw byte length for a quantized tensor |
| `nntr_dtype` | Native type (`Q4_0`, `Q4_K`, or `Q6_K`); absent for FP32/FP16 |
| `nntr_shape` | Logical pre-quantization `[N, C, H, W]` shape; absent for FP32/FP16 |
| `data_offsets` | Half-open `[start, end)` range within the data section |

FP32 and FP16 tensors use their standard `dtype` and `shape` without extension
fields, so unquantized files remain standard safetensors files.

`Q4_0` is repacked into an ISA-specific layout (`q4_0x8` on x86 and `q4_0x4` on
ARM) that cannot be inferred from the tensor bytes alone. A file containing a
`Q4_0` tensor therefore records `nntr_q4_0_isa` (`x86` or `arm`) under
`__metadata__`. The value follows `--isa`, with `DEFAULT` resolving to the
build platform. A file cross-quantized on x86 with `--isa ARM` is consequently
tagged `arm` and can be identified before loading on the wrong architecture.

### Inspecting a file

`nntr_safetensors_info` reads only the header and prints metadata and a
per-tensor table:

```bash
./build-causallm/Applications/CausalLM/nntr_safetensors_info \
  /path/to/model.safetensors
```

```text
file: model.safetensors
header bytes: 24960

metadata:
  format = nntrainer
  nntr_format = nntr-safetensors-v1
  nntr_q4_0_isa = arm

tensors: 2
  name                 dtype     bytes         shape
  layer0_wq:weight     Q4_0      2359296       [1,1,1024,4096]
  output_norm:weight   F32       4096          [1,1,1,1024]
```

This makes a quantized `.safetensors` file self-describing: each weight's
quantization type is visible without an accompanying `nntr_config.json`.
