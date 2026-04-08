#!/usr/bin/env bash
# Run the jailbreak capping experiment.
#
# Usage:
#   ./run_capping.sh <preset>               # single GPU
#   ./run_capping.sh <preset> 0 1           # 2 GPUs in parallel
#   ./run_capping.sh <preset> 0 1 2 3       # 4 GPUs in parallel
#
# Presets:
#   sanity       5 behaviors, α=0.25 only
#   light        20 behaviors, α ∈ {0.1, 0.25, 0.5}
#   full         100 behaviors, α ∈ {0.1, 0.25, 0.5, 0.75}
#   paper        100 behaviors, α=0.25 only
#   cal_inv      20 behaviors, jbb_cal_raw_inv axis only (polarity-fix test)
#   cross_sanity cross-axis capping (5 prompts, sanity check)
#   cross_axis   cross-axis capping (20 prompts)
#   cross_full   cross-axis capping (100 prompts, full run)

set -euo pipefail

PRESET="${1:-full}"
shift                        # remaining args are GPU ids
GPUS=("$@")                  # e.g. (0 1) or (0 1 2 3)
N_GPUS=${#GPUS[@]}

# ── Single GPU ───────────────────────────────────────────────────────────────
if [ "$N_GPUS" -eq 0 ]; then
    python run_capping.py --preset "$PRESET"
    exit 0
fi

# ── Multi-GPU parallel ───────────────────────────────────────────────────────
# Divide behaviors evenly; last shard takes any remainder.
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

echo "Splitting $N_TOTAL behaviors across $N_GPUS GPUs (chunk=$CHUNK)..."

for i in "${!GPUS[@]}"; do
    GPU="${GPUS[$i]}"
    START=$(( i * CHUNK ))
    if [ "$i" -eq $(( N_GPUS - 1 )) ]; then
        END=""            # last shard takes remainder
        SLICE="$START:"
    else
        END=$(( START + CHUNK ))
        SLICE="$START:$END"
    fi

    DIR="$TMP/gpu$i"
    mkdir -p "$DIR"

    echo "  GPU $GPU  slice $SLICE  → $DIR"
    CUDA_VISIBLE_DEVICES="$GPU" TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
        python run_capping.py \
        --preset "$PRESET" --prompt-slice "$SLICE" --output-dir "$DIR" \
        > "$TMP/gpu${i}.log" 2>&1 &
    PIDS+=($!)
done

# Wait for all shards with progress monitoring
FAILED=0
DONE=()
for i in "${!GPUS[@]}"; do DONE[$i]=0; done

while true; do
    # Check how many are still running
    RUNNING=0
    for i in "${!PIDS[@]}"; do
        if [ "${DONE[$i]}" -eq 0 ] && kill -0 "${PIDS[$i]}" 2>/dev/null; then
            RUNNING=$((RUNNING + 1))
        elif [ "${DONE[$i]}" -eq 0 ]; then
            # Just finished — check exit status
            wait "${PIDS[$i]}" 2>/dev/null
            EXIT_CODE=$?
            DONE[$i]=1
            if [ "$EXIT_CODE" -ne 0 ]; then
                echo "  GPU ${GPUS[$i]} FAILED (exit $EXIT_CODE) — see $TMP/gpu${i}.log"
                FAILED=1
            else
                echo "  GPU ${GPUS[$i]} done."
            fi
        fi
    done

    # All finished?
    [ "$RUNNING" -eq 0 ] && break

    # Print progress from each GPU's log
    STATUS=""
    for i in "${!GPUS[@]}"; do
        GPU="${GPUS[$i]}"
        LOG="$TMP/gpu${i}.log"
        if [ "${DONE[$i]}" -eq 1 ]; then
            STATUS="${STATUS}GPU${GPU}:done  "
        elif [ -f "$LOG" ]; then
            # Extract latest "Prompt N done" progress (|| true to avoid set -e exit)
            PROGRESS=$(grep -oE 'Prompt [0-9]+ done.*ETA [^ ]+' "$LOG" 2>/dev/null | tail -1) || true
            if [ -z "$PROGRESS" ]; then
                # Check for earlier phases (threshold computation, axis building)
                PHASE=$(grep -E 'Computing|Building|Loading|Running' "$LOG" 2>/dev/null | tail -1 | sed 's/.*] //') || true
                STATUS="${STATUS}GPU${GPU}:${PHASE:-starting}  "
            else
                STATUS="${STATUS}GPU${GPU}:${PROGRESS}  "
            fi
        else
            STATUS="${STATUS}GPU${GPU}:waiting  "
        fi
    done
    printf "\r\033[K%s" "$STATUS"

    sleep 5
done
echo ""

[ "$FAILED" -eq 1 ] && exit 1

echo "Merging results..."
python merge_results.py --preset "cap_$PRESET" --gpus "${GPUS[@]}"

rm -rf "$TMP"
echo "Done."
