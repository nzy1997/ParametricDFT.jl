#!/usr/bin/env bash
# ============================================================================
# Run benchmark suite: smoke → moderate → heavy
# ============================================================================
# Usage:
#   nohup bash examples/benchmark/run_all.sh > benchmark_run.log 2>&1 &
#
# Results are preserved under examples/benchmark/results/<preset>/<dataset>/
# after each preset completes.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

RESULTS_BASE="$SCRIPT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULTS_BASE"

echo "============================================================"
echo "Benchmark Suite — started at $(date)"
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
echo "============================================================"

for PRESET in smoke moderate heavy; do
    echo ""
    echo "============================================================"
    echo "PRESET: $PRESET — started at $(date)"
    echo "============================================================"

    # Clean results dir so each preset starts fresh
    rm -rf "$SCRIPT_DIR/results/quickdraw"

    # Run Quick Draw benchmark
    julia --project=examples/benchmark examples/benchmark/run_quickdraw.jl "$PRESET" 2>&1 | tee "$RESULTS_BASE/${PRESET}_quickdraw_${TIMESTAMP}.log"

    # Preserve results under preset-specific directory
    PRESET_DIR="$RESULTS_BASE/$PRESET"
    mkdir -p "$PRESET_DIR"
    if [ -d "$SCRIPT_DIR/results/quickdraw" ]; then
        cp -r "$SCRIPT_DIR/results/quickdraw" "$PRESET_DIR/"
    fi

    echo ""
    echo "PRESET: $PRESET — completed at $(date)"
    echo "Results preserved in: $PRESET_DIR"
    echo "============================================================"
done

echo ""
echo "============================================================"
echo "All benchmarks complete — finished at $(date)"
echo "Results in: $RESULTS_BASE/{smoke,moderate,heavy}/"
echo "============================================================"
