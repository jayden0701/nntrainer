#!/bin/bash

# Run script for T5Gemma2 model
# This script builds (if needed) and runs the T5Gemma2 multimodal model

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default model path
MODEL_PATH="${NNTRAINER_ROOT}/Applications/CausalLM/res/t5gemma2"

# Default input prompt (with image placeholder)
DEFAULT_INPUT="<start_of_image> ./test_image.jpg Describe this image."

# Parse arguments
INPUT_PROMPT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --input)
            INPUT_PROMPT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model-path PATH    Path to T5Gemma2 model directory (default: res/t5gemma2)"
            echo "  --input PROMPT       Input prompt (can include <start_of_image> /path/to/image.jpg)"
            echo "  --help              Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --model-path /path/to/t5gemma2 --input \"<start_of_image> ./cat.jpg What is in this image?\""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Use default input if not provided
if [ -z "$INPUT_PROMPT" ]; then
    INPUT_PROMPT="$DEFAULT_INPUT"
    echo "Using default input prompt"
fi

echo "=========================================="
echo "T5Gemma2 Runner"
echo "=========================================="
echo "Model Path: $MODEL_PATH"
echo "Input Prompt: $INPUT_PROMPT"
echo "=========================================="

# Check if model directory exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found: $MODEL_PATH"
    echo "Please ensure the model files are in the specified directory."
    echo "Required files:"
    echo "  - config.json"
    echo "  - generation_config.json"
    echo "  - nntr_config.json"
    echo "  - tokenizer.json"
    echo "  - [model weights binfile]"
    exit 1
fi

# Check for required config files
for config_file in config.json generation_config.json nntr_config.json; do
    if [ ! -f "$MODEL_PATH/$config_file" ]; then
        echo "Error: Required config file not found: $MODEL_PATH/$config_file"
        exit 1
    fi
done

# Find the executable
EXECUTABLE="$NNTRAINER_ROOT/build/Applications/CausalLM/nntr_causallm"

if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable not found: $EXECUTABLE"
    echo ""
    echo "Please build the CausalLM application first:"
    echo "  cd $NNTRAINER_ROOT"
    echo "  meson setup build -Denable-transformer=true"
    echo "  ninja -C build"
    exit 1
fi

# Run the model
echo ""
echo "Running T5Gemma2..."
echo ""
"$EXECUTABLE" "$MODEL_PATH" "$INPUT_PROMPT"

echo ""
echo "=========================================="
echo "Execution completed"
echo "=========================================="