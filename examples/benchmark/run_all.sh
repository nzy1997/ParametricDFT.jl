#!/usr/bin/env bash
# ============================================================================
# Run benchmark suite: smoke → moderate → heavy, across all datasets
# ============================================================================
# Usage:
#   CUDA_VISIBLE_DEVICES=0 nohup bash examples/benchmark/run_all.sh > benchmark_run.log 2>&1 &
#
# Results are preserved under examples/benchmark/results/<preset>/<dataset>/
# after each dataset completes.
#
# Datasets:
#   - quickdraw: auto-downloaded on first run
#   - div2k: requires manual download to examples/benchmark/data/DIV2K_train_HR/
#   - atd12k: requires manual download to examples/benchmark/data/ATD-12K/
#
# Missing datasets are skipped with a warning.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

RESULTS_BASE="$SCRIPT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULTS_BASE"

# Dataset run scripts and their data directories (for skip detection)
declare -A DATASETS
DATASETS[quickdraw]="run_quickdraw.jl"
DATASETS[div2k]="run_div2k.jl"
DATASETS[atd12k]="run_atd12k.jl"

# Order matters — quickdraw first (auto-downloads), then manual datasets
DATASET_ORDER=(quickdraw div2k atd12k)

echo "============================================================"
echo "Benchmark Suite — started at $(date)"
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
echo "Datasets: ${DATASET_ORDER[*]}"
echo "============================================================"

for PRESET in smoke moderate heavy; do
    echo ""
    echo "============================================================"
    echo "PRESET: $PRESET — started at $(date)"
    echo "============================================================"

    for DATASET in "${DATASET_ORDER[@]}"; do
        SCRIPT="${DATASETS[$DATASET]}"

        echo ""
        echo "--- $PRESET / $DATASET — started at $(date) ---"

        # Clean results dir so each preset starts fresh
        rm -rf "$SCRIPT_DIR/results/$DATASET"

        # Run benchmark — if it fails (e.g., missing data), warn and continue
        if julia --project=examples/benchmark "examples/benchmark/$SCRIPT" "$PRESET" 2>&1 | tee "$RESULTS_BASE/${PRESET}_${DATASET}_${TIMESTAMP}.log"; then
            # Preserve results under preset-specific directory
            PRESET_DIR="$RESULTS_BASE/$PRESET"
            mkdir -p "$PRESET_DIR"
            if [ -d "$SCRIPT_DIR/results/$DATASET" ]; then
                cp -r "$SCRIPT_DIR/results/$DATASET" "$PRESET_DIR/"
            fi
            echo "--- $PRESET / $DATASET — completed at $(date) ---"
        else
            echo "WARNING: $PRESET / $DATASET failed (missing data?). Skipping."
        fi
    done

    echo ""
    echo "PRESET: $PRESET — completed at $(date)"
    echo "============================================================"
done

echo ""
echo "============================================================"
echo "All benchmarks complete — finished at $(date)"
echo "Results in: $RESULTS_BASE/{smoke,moderate,heavy}/{quickdraw,div2k,atd12k}/"
echo "============================================================"
