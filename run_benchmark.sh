#!/bin/bash
# Run all MedCalc benchmark experiments (L0-L3+)

# Load API key from .env file
source .env
export PYTHONPATH=.

OUTPUT_DIR="benchmark_results_250_260101"
INSTANCES=250
LOG_FILE="$OUTPUT_DIR/full_run.log"

# Create output directory
mkdir -p $OUTPUT_DIR

# Start logging
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "Running MedCalc Benchmark - All Experiments (L0-L3+)"
echo "Output: $OUTPUT_DIR"
echo "Instances: $INSTANCES"
echo "Log file: $LOG_FILE"
echo "Started: $(date)"
echo "============================================================"

# Run all experiments
python -m benchmark.cli run \
    -e baseline \
    -e l1_ptool \
    -e l1o_official \
    -e l2_distilled \
    -e l2o_official \
    -e l3_react \
    -e l3_audit \
    -e l4_pipeline \
    --debug -n $INSTANCES \
    -o $OUTPUT_DIR \
    --log-responses

echo "============================================================"
echo "Benchmark complete!"
echo "Finished: $(date)"
echo "Results: $OUTPUT_DIR/report.html"
echo "Full log: $LOG_FILE"
echo "============================================================"
