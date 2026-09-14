#!/usr/bin/env bash
# Reproduction Runner for Financial Services & Banking Support (@ChaseSupport / Banking77 / Hiver Domain)
# Executes data setup, evaluation harness, cost-matrix routing score, and Cohen's Kappa judge validation.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

if [ -f "./venv/bin/python" ]; then
    PYTHON_BIN="./venv/bin/python"
elif [ -f "/opt/homebrew/bin/python3.10" ]; then
    PYTHON_BIN="/opt/homebrew/bin/python3.10"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

export PYTHONPATH="."

echo "========================================================================="
echo "   REPRODUCIBILITY PIPELINE: FINANCIAL AI SUPPORT (@ChaseSupport / Banking77)"
echo "   Aligned with Hiver Shared Inbox & Customer Support Platform           "
echo "========================================================================="
echo "Starting pipeline execution..."

# Step 1: Process Dataset Subset
echo "[Step 1/4] Checking & generating processed dataset subset..."
$PYTHON_BIN src/data_processor.py

# Step 2: Build Golden Evaluation Set
echo "[Step 2/4] Generating 200-sample Golden Evaluation Set..."
$PYTHON_BIN src/build_golden_set.py

# Step 3: Run Full Benchmark Evaluation (Baselines vs Proposed Agent)
echo "[Step 3/4] Running Benchmark Evaluation Harness..."
$PYTHON_BIN src/eval_harness.py

# Step 4: Run Cohen's Kappa Judge Agreement Validation
echo "[Step 4/4] Executing Cohen's Kappa Judge Reliability Validation..."
$PYTHON_BIN src/cohen_kappa.py

echo "Pipeline execution completed successfully!"
