// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd.
 *
 * @file   bench_ggml_kernels.cpp
 * @date   30 March 2026
 * @brief  Minimal x86 microbenchmark harness for selected nntrainer ggml
 * kernels
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nntr_ggml_impl.h>
#include <nntr_ggml_impl_common.h>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string output_path;
  int warmup = 8;
  int iterations = 40;
  int inner_loops = 16;
};

enum class KernelKind {
  GEMM_Q4_0_Q8_0,
  GEMV_Q4_0_Q8_0,
  GEMM_Q4_K_Q8_K,
  GEMV_Q4_K_Q8_K,
};

struct BenchCase {
  const char *id;
  const char *kernel_name;
  const char *kind_name;
  KernelKind kind;
  int k;
  int nr;
  int nc;
};

struct Stats {
  double median_ns;
  double min_ns;
  double max_ns;
};

struct Result {
  BenchCase bench_case;
  Stats stats;
  double checksum;
};

constexpr BenchCase kDefaultCases[] = {
  {"q4_0_gemm_nr16_nc256_k4096", "nntr_gemm_q4_0_8x8_q8_0", "gemm",
   KernelKind::GEMM_Q4_0_Q8_0, 4096, 16, 256},
  {"q4_0_gemm_nr32_nc1024_k4096", "nntr_gemm_q4_0_8x8_q8_0", "gemm",
   KernelKind::GEMM_Q4_0_Q8_0, 4096, 32, 1024},
  {"q4_0_gemv_nr1_nc256_k4096", "nntr_gemv_q4_0_8x8_q8_0", "gemv",
   KernelKind::GEMV_Q4_0_Q8_0, 4096, 1, 256},
  {"q4_0_gemv_nr1_nc1024_k4096", "nntr_gemv_q4_0_8x8_q8_0", "gemv",
   KernelKind::GEMV_Q4_0_Q8_0, 4096, 1, 1024},
  {"q4_K_gemm_nr16_nc256_k4096", "nntr_gemm_q4_K_8x8_q8_K", "gemm",
   KernelKind::GEMM_Q4_K_Q8_K, 4096, 16, 256},
  {"q4_K_gemm_nr32_nc1024_k4096", "nntr_gemm_q4_K_8x8_q8_K", "gemm",
   KernelKind::GEMM_Q4_K_Q8_K, 4096, 32, 1024},
  {"q4_K_gemv_nr1_nc256_k4096", "nntr_gemv_q4_K_8x8_q8_K", "gemv",
   KernelKind::GEMV_Q4_K_Q8_K, 4096, 1, 256},
  {"q4_K_gemv_nr1_nc1024_k4096", "nntr_gemv_q4_K_8x8_q8_K", "gemv",
   KernelKind::GEMV_Q4_K_Q8_K, 4096, 1, 1024},
};

void print_usage(const char *argv0) {
  std::cout << "Usage: " << argv0
            << " [--output PATH] [--warmup N] [--iterations N]"
               " [--inner-loops N]\n";
}

Options parse_options(int argc, char **argv) {
  Options options;

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto require_value = [&](const char *flag_name) -> const char * {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + flag_name);
      }
      return argv[++i];
    };

    if (arg == "--output") {
      options.output_path = require_value("--output");
    } else if (arg == "--warmup") {
      options.warmup = std::stoi(require_value("--warmup"));
    } else if (arg == "--iterations") {
      options.iterations = std::stoi(require_value("--iterations"));
    } else if (arg == "--inner-loops") {
      options.inner_loops = std::stoi(require_value("--inner-loops"));
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (options.warmup < 0) {
    throw std::runtime_error("--warmup must be >= 0");
  }
  if (options.iterations <= 0) {
    throw std::runtime_error("--iterations must be > 0");
  }
  if (options.inner_loops <= 0) {
    throw std::runtime_error("--inner-loops must be > 0");
  }

  return options;
}

uint32_t mix32(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  return value;
}

float make_input_value(size_t index, uint32_t seed) {
  const uint32_t mixed = mix32(static_cast<uint32_t>(index) + seed);
  const int32_t centered = static_cast<int32_t>(mixed % 4096U) - 2048;
  return static_cast<float>(centered) / 511.0f;
}

std::vector<float> make_tensor(size_t rows, size_t cols, uint32_t seed) {
  std::vector<float> data(rows * cols);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = make_input_value(i, seed);
  }
  return data;
}

std::vector<char> quantize_weights_q4_0(const std::vector<float> &weights,
                                        int nc, int k) {
  const size_t data_size =
    sizeof(block_q4_0) * static_cast<size_t>(nc) * static_cast<size_t>(k) /
    QK4_0;
  std::vector<char> quantized(data_size);
  std::vector<char> repacked(data_size);

  nntr_quantize_q4_0(weights.data(), quantized.data(), nc, k, nullptr);
  nntr_repack_q4_0_to_q4_0_8_bl(repacked.data(), 8, quantized.data(),
                                quantized.size(), nc, k);

  return repacked;
}

std::vector<char> quantize_weights_q4_K(const std::vector<float> &weights,
                                        int nc, int k) {
  const size_t data_size =
    sizeof(block_q4_K) * static_cast<size_t>(nc) * static_cast<size_t>(k) /
    QK_K;
  std::vector<char> quantized(data_size);
  std::vector<char> repacked(data_size);

  nntr_quantize_q4_K(weights.data(), quantized.data(), nc, k, nullptr);
  nntr_repack_q4_K_to_q4_K_8_bl(repacked.data(), 8, quantized.data(),
                                quantized.size(), nc, k);

  return repacked;
}

std::vector<char> quantize_gemm_activations_q8_0(const std::vector<float> &src,
                                                 int nr, int k) {
  const size_t blocks_per_row = static_cast<size_t>(k) / QK8_0;
  const size_t bytes_per_4_rows = sizeof(block_q8_0x4) * blocks_per_row;
  std::vector<char> packed(bytes_per_4_rows * static_cast<size_t>(nr) / 4U);

  for (int row = 0; row < nr; row += 4) {
    nntr_quantize_mat_q8_0_4x8(src.data() + static_cast<size_t>(row) * k,
                               packed.data() +
                                 (static_cast<size_t>(row) / 4U) *
                                   bytes_per_4_rows,
                               k);
  }

  return packed;
}

std::vector<char> quantize_gemv_activation_q8_0(const std::vector<float> &src,
                                                int k) {
  const size_t data_size = sizeof(block_q8_0) * static_cast<size_t>(k) / QK8_0;
  std::vector<char> packed(data_size);
  nntr_quantize_row_q8_0(src.data(), packed.data(), k);
  return packed;
}

std::vector<char> quantize_gemm_activations_q8_K(const std::vector<float> &src,
                                                 int nr, int k) {
  const size_t blocks_per_row = static_cast<size_t>(k) / QK_K;
  const size_t bytes_per_4_rows = sizeof(block_q8_Kx4) * blocks_per_row;
  std::vector<char> packed(bytes_per_4_rows * static_cast<size_t>(nr) / 4U);

  for (int row = 0; row < nr; row += 4) {
    nntr_quantize_mat_q8_K_4x8(src.data() + static_cast<size_t>(row) * k,
                               packed.data() +
                                 (static_cast<size_t>(row) / 4U) *
                                   bytes_per_4_rows,
                               k);
  }

  return packed;
}

std::vector<char> quantize_gemv_activation_q8_K(const std::vector<float> &src,
                                                int k) {
  const size_t data_size = sizeof(block_q8_K) * static_cast<size_t>(k) / QK_K;
  std::vector<char> packed(data_size);
  nntr_quantize_row_q8_K(src.data(), packed.data(), k);
  return packed;
}

template <typename Fn> Stats benchmark_kernel(const Options &options, Fn &&fn) {
  for (int sample = 0; sample < options.warmup; ++sample) {
    for (int loop = 0; loop < options.inner_loops; ++loop) {
      fn();
    }
  }

  std::vector<double> samples;
  samples.reserve(options.iterations);

  for (int sample = 0; sample < options.iterations; ++sample) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto start = Clock::now();
    for (int loop = 0; loop < options.inner_loops; ++loop) {
      fn();
    }
    const auto end = Clock::now();
    std::atomic_signal_fence(std::memory_order_seq_cst);

    const auto elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
        .count();
    samples.push_back(static_cast<double>(elapsed_ns) / options.inner_loops);
  }

  std::sort(samples.begin(), samples.end());

  Stats stats{};
  stats.min_ns = samples.front();
  stats.max_ns = samples.back();

  const size_t middle = samples.size() / 2U;
  if (samples.size() % 2U == 0U) {
    stats.median_ns = (samples[middle - 1U] + samples[middle]) * 0.5;
  } else {
    stats.median_ns = samples[middle];
  }

  return stats;
}

double checksum_output(const std::vector<float> &output) {
  return std::accumulate(output.begin(), output.end(), 0.0);
}

void validate_case(const BenchCase &bench_case) {
  if (bench_case.k <= 0 || bench_case.nr <= 0 || bench_case.nc <= 0) {
    throw std::runtime_error("all benchmark dimensions must be positive");
  }
  if (bench_case.nc % 8 != 0) {
    throw std::runtime_error("nc must be divisible by 8");
  }

  switch (bench_case.kind) {
  case KernelKind::GEMM_Q4_0_Q8_0:
  case KernelKind::GEMV_Q4_0_Q8_0:
    if (bench_case.k % QK8_0 != 0) {
      throw std::runtime_error("q4_0/q8_0 kernels require k divisible by 32");
    }
    break;
  case KernelKind::GEMM_Q4_K_Q8_K:
  case KernelKind::GEMV_Q4_K_Q8_K:
    if (bench_case.k % QK_K != 0) {
      throw std::runtime_error("q4_K/q8_K kernels require k divisible by 256");
    }
    break;
  }

  if ((bench_case.kind == KernelKind::GEMM_Q4_0_Q8_0 ||
       bench_case.kind == KernelKind::GEMM_Q4_K_Q8_K) &&
      (bench_case.nr % 4 != 0)) {
    throw std::runtime_error("gemm kernels require nr divisible by 4");
  }
}

Result run_case(const BenchCase &bench_case, const Options &options) {
  validate_case(bench_case);

  const auto weights =
    make_tensor(static_cast<size_t>(bench_case.nc), bench_case.k, 0x1234U);
  const auto activations =
    make_tensor(static_cast<size_t>(bench_case.nr), bench_case.k, 0x5678U);

  std::vector<float> output(static_cast<size_t>(bench_case.nr) *
                            static_cast<size_t>(bench_case.nc));

  Stats stats{};

  switch (bench_case.kind) {
  case KernelKind::GEMM_Q4_0_Q8_0: {
    const auto packed_weights =
      quantize_weights_q4_0(weights, bench_case.nc, bench_case.k);
    const auto packed_activations =
      quantize_gemm_activations_q8_0(activations, bench_case.nr, bench_case.k);

    stats = benchmark_kernel(options, [&]() {
      nntr_gemm_q4_0_8x8_q8_0(bench_case.k, output.data(), bench_case.nc,
                              packed_weights.data(), packed_activations.data(),
                              bench_case.nr, bench_case.nc);
    });
    break;
  }
  case KernelKind::GEMV_Q4_0_Q8_0: {
    const auto packed_weights =
      quantize_weights_q4_0(weights, bench_case.nc, bench_case.k);
    const auto packed_activation =
      quantize_gemv_activation_q8_0(activations, bench_case.k);

    stats = benchmark_kernel(options, [&]() {
      nntr_gemv_q4_0_8x8_q8_0(bench_case.k, output.data(), bench_case.nc,
                              packed_weights.data(), packed_activation.data(),
                              bench_case.nr, bench_case.nc);
    });
    break;
  }
  case KernelKind::GEMM_Q4_K_Q8_K: {
    const auto packed_weights =
      quantize_weights_q4_K(weights, bench_case.nc, bench_case.k);
    const auto packed_activations =
      quantize_gemm_activations_q8_K(activations, bench_case.nr, bench_case.k);

    stats = benchmark_kernel(options, [&]() {
      nntr_gemm_q4_K_8x8_q8_K(bench_case.k, output.data(), bench_case.nc,
                              packed_weights.data(), packed_activations.data(),
                              bench_case.nr, bench_case.nc);
    });
    break;
  }
  case KernelKind::GEMV_Q4_K_Q8_K: {
    const auto packed_weights =
      quantize_weights_q4_K(weights, bench_case.nc, bench_case.k);
    const auto packed_activation =
      quantize_gemv_activation_q8_K(activations, bench_case.k);

    stats = benchmark_kernel(options, [&]() {
      nntr_gemv_q4_K_8x8_q8_K(bench_case.k, output.data(), bench_case.nc,
                              packed_weights.data(), packed_activation.data(),
                              bench_case.nr, bench_case.nc);
    });
    break;
  }
  }

  Result result{bench_case, stats, checksum_output(output)};
  return result;
}

std::string json_escape(const std::string &value) {
  std::ostringstream escaped;
  for (const char ch : value) {
    switch (ch) {
    case '\\':
      escaped << "\\\\";
      break;
    case '"':
      escaped << "\\\"";
      break;
    case '\n':
      escaped << "\\n";
      break;
    default:
      escaped << ch;
      break;
    }
  }
  return escaped.str();
}

std::string detect_arch() {
#if defined(__x86_64__) || defined(_M_X64)
  return "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
  return "x86";
#else
  return "non-x86";
#endif
}

std::string make_summary_json(const std::vector<Result> &results,
                              const Options &options) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(2);
  os << "{\n";
  os << "  \"schema_version\": 1,\n";
  os << "  \"single_threaded\": true,\n";
  os << "  \"arch\": \"" << json_escape(detect_arch()) << "\",\n";
  os << "  \"unit\": \"ns\",\n";
  os << "  \"warmup\": " << options.warmup << ",\n";
  os << "  \"iterations\": " << options.iterations << ",\n";
  os << "  \"inner_loops\": " << options.inner_loops << ",\n";
  os << "  \"benchmarks\": [\n";

  for (size_t i = 0; i < results.size(); ++i) {
    const auto &result = results[i];
    os << "    {\n";
    os << "      \"id\": \"" << json_escape(result.bench_case.id) << "\",\n";
    os << "      \"kernel\": \""
       << json_escape(result.bench_case.kernel_name) << "\",\n";
    os << "      \"kind\": \"" << json_escape(result.bench_case.kind_name)
       << "\",\n";
    os << "      \"shape\": {\n";
    os << "        \"nr\": " << result.bench_case.nr << ",\n";
    os << "        \"nc\": " << result.bench_case.nc << ",\n";
    os << "        \"k\": " << result.bench_case.k << ",\n";
    os << "        \"bs\": " << result.bench_case.nc << "\n";
    os << "      },\n";
    os << "      \"median\": " << result.stats.median_ns << ",\n";
    os << "      \"min\": " << result.stats.min_ns << ",\n";
    os << "      \"max\": " << result.stats.max_ns << ",\n";
    os << "      \"checksum\": " << result.checksum << "\n";
    os << "    }";
    if (i + 1U != results.size()) {
      os << ",";
    }
    os << "\n";
  }

  os << "  ]\n";
  os << "}\n";
  return os.str();
}

void write_summary(const std::string &path, const std::string &summary_json) {
  std::filesystem::path output_path(path);
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }

  std::ofstream output(path, std::ios::out | std::ios::trunc);
  if (!output.is_open()) {
    throw std::runtime_error("failed to open output file: " + path);
  }
  output << summary_json;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);

    nntr_ggml_init();

    std::vector<Result> results;
    results.reserve(std::size(kDefaultCases));

    for (const auto &bench_case : kDefaultCases) {
      results.push_back(run_case(bench_case, options));
    }

    const std::string summary_json = make_summary_json(results, options);
    std::cout << summary_json;

    if (!options.output_path.empty()) {
      write_summary(options.output_path, summary_json);
    }
  } catch (const std::exception &error) {
    std::cerr << "bench_ggml_kernels: " << error.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
