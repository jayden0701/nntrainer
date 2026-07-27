// SPDX-License-Identifier: Apache-2.0
/**
 * @file    test_api.cpp
 * @brief   Command-line smoke application for the public Quick.AI C API
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "quick_dot_ai_api.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::string lowercase(std::string value) {
  std::transform(
    value.begin(), value.end(), value.begin(),
    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool parse_bool(const char *value) {
  const std::string normalized = lowercase(value != nullptr ? value : "");
  return normalized == "1" || normalized == "true";
}

std::string json_quote(std::string_view value) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(value.size() + 2);
  result.push_back('"');
  for (unsigned char c : value) {
    switch (c) {
    case '"':
      result += "\\\"";
      break;
    case '\\':
      result += "\\\\";
      break;
    case '\b':
      result += "\\b";
      break;
    case '\f':
      result += "\\f";
      break;
    case '\n':
      result += "\\n";
      break;
    case '\r':
      result += "\\r";
      break;
    case '\t':
      result += "\\t";
      break;
    default:
      if (c < 0x20) {
        result += "\\u00";
        result.push_back(hex[c >> 4]);
        result.push_back(hex[c & 0x0f]);
      } else {
        result.push_back(static_cast<char>(c));
      }
      break;
    }
  }
  result.push_back('"');
  return result;
}

void print_usage(const char *program) {
  std::cerr
    << "Usage: " << program
    << " <model> [prompt] [chat_template] [quant] [verbose]"
       " [model_base_path]\n"
    << "  model: qwen3-0.6b | qwen3-1.7b-q40 | tiny-bert | "
       "function-gemma |\n"
    << "         gemma4-cpu | gemma4-e2b-qnn | vjepa2-qnn | "
       "<catalog id>\n"
    << "         Generic catalog ids default to the CPU backend.\n"
    << "  chat_template: true uses quickAiRunOpenAI; false uses exact "
       "quickAiRunText\n"
    << "  quant: W4A32 | W16A16 | W8A16 | W32A32 (compatibility field; "
       "catalog/config selects files)\n"
    << "Environment:\n"
    << "  QDA_API=openai|text overrides chat_template API selection\n"
    << "  QDA_STREAM=1 prints callback deltas live\n"
    << "  QDA_TOOL=1 runs the migrated tool/schema request on known "
       "tool-capable models\n"
    << "  STRESS_CYCLES=N controls load/unload cycles before the final load\n"
    << "  QUICKAI_MODEL_BASE_PATH supplies model_base_path\n";
}

bool parse_quantization(const char *argument,
                        ModelQuantizationType &quantization,
                        std::string &display_name) {
  display_name = argument != nullptr ? argument : "W4A32";
  std::transform(
    display_name.begin(), display_name.end(), display_name.begin(),
    [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  if (display_name == "W4A32") {
    quantization = CAUSAL_LM_QUANTIZATION_W4A32;
  } else if (display_name == "W16A16") {
    quantization = CAUSAL_LM_QUANTIZATION_W16A16;
  } else if (display_name == "W8A16") {
    quantization = CAUSAL_LM_QUANTIZATION_W8A16;
  } else if (display_name == "W32A32") {
    quantization = CAUSAL_LM_QUANTIZATION_W32A32;
  } else {
    return false;
  }
  return true;
}

struct ModelSelection {
  std::string catalog_id;
  BackendType backend = CAUSAL_LM_BACKEND_CPU;
  bool embedding = false;
  bool vision = false;
  bool tool_use = false;
};

ModelSelection select_model(const char *argument) {
  ModelSelection selection;
  const std::string original = argument != nullptr ? argument : "";
  const std::string name = lowercase(original);

  if (name == "qwen3_0.6b") {
    selection.catalog_id = "qwen3-0.6b";
  } else if (name == "qwen3_1.7b_q40") {
    selection.catalog_id = "qwen3-1.7b-q40";
  } else if (name == "tiny_bert" || name == "tiny-bert") {
    selection.catalog_id = "tiny-bert";
    selection.embedding = true;
  } else if (name == "function_gemma" || name == "function-gemma") {
    selection.catalog_id = "function-gemma";
  } else if (name == "gemma4_cpu" || name == "gemma4-cpu") {
    selection.catalog_id = "gemma4-cpu";
  } else if (name == "gemma4_e2b_qnn" || name == "gemma4-e2b-qnn") {
    selection.catalog_id = "gemma4-e2b-qnn";
    selection.backend = CAUSAL_LM_BACKEND_NPU;
  } else if (name == "vjepa" || name == "vjepa2_qnn" || name == "vjepa2-qnn") {
    selection.catalog_id = "vjepa2-qnn";
    selection.backend = CAUSAL_LM_BACKEND_NPU;
    selection.vision = true;
  } else {
    selection.catalog_id = original;
  }
  selection.tool_use = selection.catalog_id == "qwen3-0.6b" ||
                       selection.catalog_id == "qwen3-1.7b-q40" ||
                       selection.catalog_id == "function-gemma";
  return selection;
}

ErrorCode load_handle(const ModelSelection &model,
                      ModelQuantizationType quantization,
                      const char *model_base_path, CausalLmHandle *handle) {
  return loadModelHandleByName(model.backend, model.catalog_id.c_str(),
                               quantization, nullptr, model_base_path, handle);
}

bool run_embedding_smoke(CausalLmHandle handle, const char *text) {
  float *embedding = nullptr;
  int dimension = 0;
  const ErrorCode result =
    encodeModelHandle(handle, text, &embedding, &dimension);
  if (result != CAUSAL_LM_ERROR_NONE || embedding == nullptr ||
      dimension <= 0) {
    std::cerr << "encodeModelHandle failed: " << result << '\n';
    freeEmbedding(embedding);
    return false;
  }

  double squared_norm = 0.0;
  bool finite = true;
  for (int i = 0; i < dimension; ++i) {
    finite = finite && std::isfinite(embedding[i]);
    squared_norm += static_cast<double>(embedding[i]) * embedding[i];
  }
  const double norm = std::sqrt(squared_norm);
  std::cout << "Embedding dimension: " << dimension << "\nL2 norm: " << norm
            << "\nFirst values:";
  for (int i = 0; i < std::min(dimension, 8); ++i)
    std::cout << ' ' << embedding[i];
  std::cout << '\n';
  freeEmbedding(embedding);
  return finite && norm > 0.0;
}

bool run_vision_smoke(CausalLmHandle handle) {
  constexpr size_t float_count = 1ULL * 24 * 3 * 256 * 256;
  std::vector<float> pixels(float_count);
  const char *fill = std::getenv("VJEPA2_PIXEL_FILL");
  const bool zero_fill = lowercase(fill != nullptr ? fill : "") == "zero";
  if (!zero_fill) {
    for (size_t i = 0; i < pixels.size(); ++i)
      pixels[i] = static_cast<float>(i % 255) / 255.0f;
  }

  int runs = 1;
  if (const char *value = std::getenv("VJEPA2_RUNS")) {
    runs = std::max(1, std::atoi(value));
  }

  void *embedding = nullptr;
  int output_bytes = 0;
  double total_ms = 0.0;
  for (int i = 0; i < runs; ++i) {
    freeImageEmbedding(embedding);
    embedding = nullptr;
    const auto start = std::chrono::steady_clock::now();
    const ErrorCode result =
      encodeImageModelHandle(handle, pixels.data(), pixels.size(), 256, 256,
                             &embedding, &output_bytes);
    const auto end = std::chrono::steady_clock::now();
    if (result != CAUSAL_LM_ERROR_NONE || embedding == nullptr ||
        output_bytes <= 0) {
      std::cerr << "encodeImageModelHandle failed: " << result << '\n';
      freeImageEmbedding(embedding);
      return false;
    }
    total_ms += std::chrono::duration<double, std::milli>(end - start).count();
  }

  std::cout << "Vision output: " << output_bytes
            << " bytes, average latency: " << total_ms / runs << " ms\n";
  const auto *values = static_cast<const uint16_t *>(embedding);
  const int value_count =
    std::min(output_bytes / static_cast<int>(sizeof(uint16_t)), 8);
  std::cout << "First values:";
  for (int i = 0; i < value_count; ++i)
    std::cout << ' ' << values[i];
  std::cout << '\n';
  freeImageEmbedding(embedding);
  return true;
}

struct OutputCollector {
  bool print_deltas = false;
  std::string output;
};

int collect_delta(const char *delta, void *user_data) {
  if (delta == nullptr || user_data == nullptr)
    return 1;
  auto &collector = *static_cast<OutputCollector *>(user_data);
  collector.output.append(delta);
  if (collector.print_deltas)
    std::cout << delta << std::flush;
  return 0;
}

std::string make_chat_request(std::string_view prompt) {
  return "{\"messages\":[{\"role\":\"user\",\"content\":" + json_quote(prompt) +
         "}],\"stream\":true}";
}

std::string make_tool_request(std::string_view prompt) {
  static constexpr std::string_view schema =
    R"({"type":"object","properties":{"query":{"type":"string","description":"Search query in the most effective language for the requested results"},"count":{"type":"integer","minimum":1,"maximum":10,"description":"Number of results to return"}},"required":["query"],"additionalProperties":false})";
  return "{\"messages\":[{\"role\":\"user\",\"content\":" + json_quote(prompt) +
         "}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":"
         "\"web_search\",\"description\":\"Search the web\","
         "\"parameters\":" +
         std::string(schema) +
         "}}],\"response_format\":{\"type\":\"json_schema\",\"json_schema\":"
         "{\"name\":\"web_search\",\"strict\":true,\"schema\":" +
         std::string(schema) + "}},\"stream\":true}";
}

void print_metrics(CausalLmHandle handle) {
  PerformanceMetrics metrics{};
  const ErrorCode result = getPerformanceMetricsHandle(handle, &metrics);
  if (result != CAUSAL_LM_ERROR_NONE) {
    std::cout << "Performance metrics unavailable: " << result << '\n';
    return;
  }

  const double prefill_rate =
    metrics.prefill_duration_ms > 0.0
      ? metrics.prefill_tokens * 1000.0 / metrics.prefill_duration_ms
      : 0.0;
  const double generation_rate =
    metrics.generation_duration_ms > 0.0
      ? metrics.generation_tokens * 1000.0 / metrics.generation_duration_ms
      : 0.0;
  std::cout << std::fixed << std::setprecision(2)
            << "Initialization: " << metrics.initialization_duration_ms
            << " ms\nPrefill: " << metrics.prefill_tokens << " tokens, "
            << prefill_rate
            << " tokens/s\nGeneration: " << metrics.generation_tokens
            << " tokens, " << generation_rate
            << " tokens/s\nTotal: " << metrics.total_duration_ms
            << " ms\nPeak memory: " << metrics.peak_memory_kb << " KB\n";
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  const char *prompt = argc >= 3 ? argv[2] : "Hello, how are you?";
  bool use_openai = argc < 4 || parse_bool(argv[3]);
  if (const char *api = std::getenv("QDA_API")) {
    const std::string choice = lowercase(api);
    if (choice == "openai") {
      use_openai = true;
    } else if (choice == "text") {
      use_openai = false;
    } else {
      std::cerr << "QDA_API must be either 'openai' or 'text'\n";
      return 1;
    }
  }

  ModelQuantizationType quantization = CAUSAL_LM_QUANTIZATION_W4A32;
  std::string quantization_name;
  if (!parse_quantization(argc >= 5 ? argv[4] : nullptr, quantization,
                          quantization_name)) {
    std::cerr << "Unsupported quantization: " << quantization_name << '\n';
    return 1;
  }
  const bool verbose = argc < 6 || parse_bool(argv[5]);

  std::string base_path_storage;
  if (argc >= 7) {
    base_path_storage = argv[6];
  } else if (const char *base_path = std::getenv("QUICKAI_MODEL_BASE_PATH")) {
    base_path_storage = base_path;
  }
  const char *model_base_path =
    base_path_storage.empty() ? nullptr : base_path_storage.c_str();

  const ModelSelection model = select_model(argv[1]);
  if (model.catalog_id.empty()) {
    std::cerr << "Model catalog id must not be empty\n";
    return 1;
  }
  const char *tool_value = std::getenv("QDA_TOOL");
  const bool tool_mode = tool_value != nullptr && parse_bool(tool_value);
  if (tool_mode && (!use_openai || !model.tool_use)) {
    std::cerr << "QDA_TOOL requires the OpenAI API and a known catalog model "
                 "with tool-use capability\n";
    return 1;
  }

  Config config{};
  config.debug_mode = verbose;
  config.verbose = verbose;
  config.chat_template_name = nullptr;
  ErrorCode result = setOptions(config);
  if (result != CAUSAL_LM_ERROR_NONE) {
    std::cerr << "setOptions failed: " << result << '\n';
    return 1;
  }

  int stress_cycles = 1;
  if (const char *cycles = std::getenv("STRESS_CYCLES"))
    stress_cycles = std::max(0, std::atoi(cycles));

  std::cout << "Model: " << model.catalog_id
            << "\nBackend: " << static_cast<int>(model.backend)
            << "\nQuantization: " << quantization_name
            << "\nAPI: " << (use_openai ? "OpenAI" : "exact text")
            << "\nModel base path: "
            << (model_base_path != nullptr ? model_base_path : "(default)")
            << '\n';

  for (int i = 0; i < stress_cycles; ++i) {
    CausalLmHandle cycle_handle = nullptr;
    result = load_handle(model, quantization, model_base_path, &cycle_handle);
    if (result != CAUSAL_LM_ERROR_NONE) {
      std::cerr << "Stress load " << i + 1 << " failed: " << result
                << "\nCatalog: " << getModelCatalogJson() << '\n';
      return 1;
    }
    result = unloadModelHandle(cycle_handle);
    const ErrorCode destroy_result = destroyModelHandle(cycle_handle);
    if (result != CAUSAL_LM_ERROR_NONE ||
        destroy_result != CAUSAL_LM_ERROR_NONE) {
      std::cerr << "Stress unload/destroy " << i + 1 << " failed: " << result
                << '/' << destroy_result << '\n';
      return 1;
    }
  }

  CausalLmHandle handle = nullptr;
  result = load_handle(model, quantization, model_base_path, &handle);
  if (result != CAUSAL_LM_ERROR_NONE) {
    std::cerr << "Final load failed: " << result
              << "\nCatalog: " << getModelCatalogJson() << '\n';
    return 1;
  }

  bool success = true;
  if (model.embedding) {
    success = run_embedding_smoke(handle, prompt);
  } else if (model.vision) {
    success = run_vision_smoke(handle);
    const ErrorCode destroy_result = destroyModelHandle(handle);
    if (destroy_result != CAUSAL_LM_ERROR_NONE) {
      std::cerr << "destroyModelHandle failed: " << destroy_result << '\n';
      success = false;
    }

    // The QNN context singleton currently has a fragile process-exit
    // destructor after V-JEPA inference. Flush the completed smoke result and
    // bypass global teardown; the OS reclaims process resources.
    std::cout.flush();
    std::cerr.flush();
    std::fflush(nullptr);
    std::_Exit(success ? 0 : 1);
  } else {
    const char *stream_value = std::getenv("QDA_STREAM");
    const bool stream_output =
      stream_value != nullptr && parse_bool(stream_value);
    OutputCollector collector{stream_output, {}};
    if (use_openai) {
      const std::string request =
        tool_mode ? make_tool_request(prompt) : make_chat_request(prompt);
      result = quickAiRunOpenAI(handle, request.c_str(), nullptr, 0,
                                collect_delta, &collector);
    } else {
      result = quickAiRunText(handle, prompt, collect_delta, &collector);
    }
    if (stream_output)
      std::cout << '\n';
    else
      std::cout << "Output:\n" << collector.output << '\n';
    if (result != CAUSAL_LM_ERROR_NONE) {
      std::cerr << "Inference failed: " << result << '\n';
      success = false;
    } else {
      print_metrics(handle);
    }
  }

  const ErrorCode destroy_result = destroyModelHandle(handle);
  if (destroy_result != CAUSAL_LM_ERROR_NONE) {
    std::cerr << "destroyModelHandle failed: " << destroy_result << '\n';
    success = false;
  }
  return success ? 0 : 1;
}
