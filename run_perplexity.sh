#!/bin/bash
# Compute perplexity for sanity check results
# Run on GPU machine where the model can be loaded

set -e

PRESET="${1:-sanity}"
INPUT="${PRESET}/generations.csv"
MODEL="Qwen/Qwen3-32B"

if [ ! -f "$INPUT" ]; then
    echo "Error: $INPUT not found"
    exit 1
fi

echo "Computing perplexity for $INPUT with $MODEL"
python compute_perplexity.py --input "$INPUT" --model "$MODEL"
