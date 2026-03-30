#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

BUILD_DIR="${REPO_ROOT}/build-x86-perf"
SUMMARY_PATH="${REPO_ROOT}/perf/out/summary.json"
COMPARE_PATH="${REPO_ROOT}/perf/out/compare.json"
BASELINE_PATH=""
BENCH_ARGS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --build-dir PATH     Meson build directory (default: ${BUILD_DIR})
  --summary PATH       Benchmark summary output path (default: ${SUMMARY_PATH})
  --baseline PATH      Baseline summary JSON to compare against
  --warmup N           Warmup samples passed to bench_ggml_kernels
  --iterations N       Measured samples passed to bench_ggml_kernels
  --inner-loops N      Inner loop count passed to bench_ggml_kernels
  --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --summary)
      SUMMARY_PATH="$2"
      shift 2
      ;;
    --baseline)
      BASELINE_PATH="$2"
      shift 2
      ;;
    --warmup|--iterations|--inner-loops)
      BENCH_ARGS+=("$1" "$2")
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

ARCH=$(uname -m)
case "${ARCH}" in
  x86_64|amd64|i386|i686)
    ;;
  *)
    echo "bench_x86.sh requires an x86 host, got: ${ARCH}" >&2
    exit 1
    ;;
esac

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

MESON_ARGS=(
  --buildtype=release
  -Denable-app=false
  -Denable-capi=disabled
  -Denable-transformer=true
  -Denable-tflite-backbone=false
  -Denable-tflite-interpreter=false
  -Denable-nnstreamer-backbone=false
  -Denable-nnstreamer-tensor-filter=disabled
  -Denable-nnstreamer-tensor-trainer=disabled
  -Dnntr-num-threads=1
  -Domp-num-threads=1
)

mkdir -p "$(dirname "${SUMMARY_PATH}")"

if [[ -d "${BUILD_DIR}" ]]; then
  meson setup --reconfigure "${BUILD_DIR}" "${MESON_ARGS[@]}"
else
  meson setup "${BUILD_DIR}" "${MESON_ARGS[@]}"
fi

meson compile -C "${BUILD_DIR}"
meson test -C "${BUILD_DIR}" --suite unittests --print-errorlogs
"${BUILD_DIR}/perf/bench_ggml_kernels" --output "${SUMMARY_PATH}" "${BENCH_ARGS[@]}"

COMPARE_ARGS=(
  --current "${SUMMARY_PATH}"
  --history "${REPO_ROOT}/perf/perf_history.jsonl"
  --output "${COMPARE_PATH}"
)

if [[ -n "${BASELINE_PATH}" ]]; then
  COMPARE_ARGS+=(--baseline "${BASELINE_PATH}")
fi

python3 "${REPO_ROOT}/tools/compare_perf.py" "${COMPARE_ARGS[@]}"

echo "summary: ${SUMMARY_PATH}"
if [[ -f "${COMPARE_PATH}" ]]; then
  echo "comparison: ${COMPARE_PATH}"
fi
