#!/bin/bash
# Script to run img_process_dataset.jl in the background
# Usage: ./run_training.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

echo "Starting training script..."
echo "Log file: $LOG_FILE"
echo "To monitor: tail -f $LOG_FILE"
echo "To check if running: ps aux | grep julia"

cd "$SCRIPT_DIR"
nohup julia --project=examples examples/img_process_dataset.jl > "$LOG_FILE" 2>&1 &

PID=$!
echo "Process started with PID: $PID"
echo "PID saved to: ${LOG_DIR}/training_${TIMESTAMP}.pid"
echo $PID > "${LOG_DIR}/training_${TIMESTAMP}.pid"

echo ""
echo "To stop the process: kill $PID"
echo "Or: kill \$(cat ${LOG_DIR}/training_${TIMESTAMP}.pid)"

