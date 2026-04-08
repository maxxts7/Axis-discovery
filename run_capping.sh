#!/usr/bin/env bash
# Run the jailbreak capping experiment.
#
# Usage:
#   ./run_capping.sh <preset>               # single GPU
#   ./run_capping.sh <preset> 0 1           # 2 GPUs in parallel
#   ./run_capping.sh <preset> 0 1 2 3       # 4 GPUs in parallel

set -euo pipefail

PRESET="${1:-full}"
shift
GPUS=("$@")
N_GPUS=${#GPUS[@]}

# ── Single GPU ───────────────────────────────────────────────────────────────
if [ "$N_GPUS" -eq 0 ]; then
    python run_capping.py --preset "$PRESET"
    exit 0
fi

# ── Multi-GPU parallel ───────────────────────────────────────────────────────
case "$PRESET" in
    sanity)        N_TOTAL=5   ;;
    light)         N_TOTAL=20  ;;
    full)          N_TOTAL=100 ;;
    paper)         N_TOTAL=100 ;;
    cal_inv)       N_TOTAL=20  ;;
    cross_sanity)  N_TOTAL=5   ;;
    cross_axis)    N_TOTAL=20  ;;
    cross_full)    N_TOTAL=100 ;;
    *) echo "Unknown preset: $PRESET"; exit 1 ;;
esac

CHUNK=$(( N_TOTAL / N_GPUS ))
TMP="_parallel_tmp"
PIDS=()

# Pre-download the axis file so GPU processes don't race
echo "Pre-downloading axis vectors..."
python -c "from capping_experiment import download_axis; download_axis('Qwen/Qwen3-32B')"

echo "Splitting $N_TOTAL behaviors across $N_GPUS GPUs (chunk=$CHUNK)..."

for i in "${!GPUS[@]}"; do
    GPU="${GPUS[$i]}"
    START=$(( i * CHUNK ))
    if [ "$i" -eq $(( N_GPUS - 1 )) ]; then
        SLICE="$START:"
    else
        SLICE="$START:$(( START + CHUNK ))"
    fi

    DIR="$TMP/gpu$i"
    mkdir -p "$DIR"

    echo "  GPU $GPU  slice $SLICE  → $DIR"
    if [ "$i" -eq 0 ]; then
        # Show first GPU's output live; also save to log via tee
        CUDA_VISIBLE_DEVICES="$GPU" python run_capping.py \
            --preset "$PRESET" --prompt-slice "$SLICE" --output-dir "$DIR" \
            2>&1 | tee "$TMP/gpu${i}.log" &
    else
        TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
        CUDA_VISIBLE_DEVICES="$GPU" python run_capping.py \
            --preset "$PRESET" --prompt-slice "$SLICE" --output-dir "$DIR" \
            > "$TMP/gpu${i}.log" 2>&1 &
    fi
    PIDS+=($!)
done

# Wait for all shards
echo "Waiting for ${N_GPUS} GPUs... (logs in $TMP/gpu*.log)"
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  GPU ${GPUS[$i]} done."
    else
        echo "  GPU ${GPUS[$i]} FAILED — see $TMP/gpu${i}.log"
        FAILED=1
    fi
done

[ "$FAILED" -eq 1 ] && exit 1

echo "Merging results..."
python merge_results.py --preset "cap_$PRESET" --gpus "${GPUS[@]}"

rm -rf "$TMP"
echo "Done."
