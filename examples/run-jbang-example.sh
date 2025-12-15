#!/bin/bash
# Example script to run llama-tornado with JBang
#
# This demonstrates how to use the JBang CLI for quick experimentation
# with llama-tornado models.

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Llama-Tornado JBang CLI Example                     ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if JBang is installed
if ! command -v jbang &> /dev/null; then
    echo "❌ JBang is not installed!"
    echo "Please install JBang first: https://www.jbang.dev/download/"
    exit 1
fi

echo -e "${GREEN}✓${NC} JBang is installed"

# Check if model file is provided
if [ -z "$1" ]; then
    echo ""
    echo "Usage: $0 <path-to-model.gguf> [prompt]"
    echo ""
    echo "Examples:"
    echo "  $0 beehive-llama-3.2-1b-instruct-fp16.gguf"
    echo "  $0 beehive-llama-3.2-1b-instruct-fp16.gguf \"Tell me a joke\""
    echo ""
    exit 1
fi

MODEL_PATH="$1"
PROMPT="${2:-What is the capital of France?}"

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model file not found: $MODEL_PATH"
    echo ""
    echo "Please download a model first. See:"
    echo "https://huggingface.co/collections/beehive-lab/llama3-gpullama3java"
    exit 1
fi

echo -e "${GREEN}✓${NC} Model file found: $MODEL_PATH"
echo ""

# Run with JBang
echo "Running inference with prompt: \"$PROMPT\""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd "$(dirname "$0")/.." || exit

jbang LlamaTornadoCli.java \
    --model "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --temperature 0.7 \
    --max-tokens 256

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}Done!${NC}"
echo ""
echo "Try interactive mode:"
echo "  jbang LlamaTornadoCli.java --model $MODEL_PATH --interactive"
