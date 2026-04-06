#!/usr/bin/env bash
# Usage:
#   ./run_experiment.sh <preset>          # single GPU
#   ./run_experiment.sh <preset> 0 1      # 2 GPUs in parallel
set -euo pipefail

PRESET="${1:-sanity}"

# ── Single GPU (default) ────────────────────────────────────────────────────
if [ $# -lt 3 ]; then
    python run_generation.py --preset "$PRESET"
    exit 0
fi

GPU0="$2"
GPU1="$3"

# ── 2-GPU parallel ──────────────────────────────────────────────────────────
case "$PRESET" in
    sanity)   MID=3  ;;   # 5 prompts  → [0:3]  [3:5]
    thorough) MID=8  ;;   # 15 prompts → [0:8]  [8:15]
    full)     MID=25 ;;   # 50 prompts → [0:25] [25:50]
    focused)  MID=25 ;;   # 50 prompts → [0:25] [25:50]
    *) echo "Unknown preset: $PRESET"; exit 1 ;;
esac

mkdir -p _tmp/gpu0 _tmp/gpu1

CUDA_VISIBLE_DEVICES="$GPU0" python run_generation.py \
    --preset "$PRESET" --prompt-slice "0:$MID" --output-dir _tmp/gpu0 \
    > _tmp/gpu0.log 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES="$GPU1" python run_generation.py \
    --preset "$PRESET" --prompt-slice "$MID:" --output-dir _tmp/gpu1 \
    > _tmp/gpu1.log 2>&1 &
PID1=$!

wait $PID0 || { echo "GPU $GPU0 failed — see _tmp/gpu0.log"; exit 1; }
wait $PID1 || { echo "GPU $GPU1 failed — see _tmp/gpu1.log"; exit 1; }

python merge_results.py --preset "$PRESET" --gpus "$GPU0" "$GPU1"
rm -rf _tmp
