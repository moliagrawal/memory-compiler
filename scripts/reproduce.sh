#!/usr/bin/env bash
# End-to-end reproduction script for Memory Compiler
set -e
export PYTHONPATH=.

VENV="${HOME}/venvs/mc"
PYTHON="${VENV}/bin/python"

if [ ! -f "${PYTHON}" ]; then
    PYTHON="python3"
fi

echo "=========================================================="
echo "Memory Compiler - Reproduction Pipeline"
echo "=========================================================="

echo "[1/5] Executing test suite..."
${PYTHON} -m pytest tests/ -v

echo "[2/5] Preparing dataset and video-level splits..."
bash scripts/download_data.sh

echo "[3/5] Extracting video frame features..."
${PYTHON} scripts/extract_features.py --config configs/data/charades_sta.yaml

echo "[4/5] Executing baseline evaluations (B0, B1, B2)..."
${PYTHON} scripts/run_baselines.py --split val

echo "[5/5] Generating all 5 quantitative result tables..."
${PYTHON} scripts/make_tables.py --runs_dir runs/ --output_dir results/

echo "=========================================================="
echo "Reproduction complete! Tables available in results/ and RESULTS.md"
echo "=========================================================="
