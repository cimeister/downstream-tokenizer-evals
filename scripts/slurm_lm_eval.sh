#!/bin/bash
#SBATCH --job-name=lm-eval
#SBATCH --account=a139
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/lm_eval_%j.out
#SBATCH --error=logs/lm_eval_%j.err
#SBATCH --uenv=pytorch/v2.8.0:v1
#SBATCH --view=default

# ============================================================
# Run lm-eval-harness tasks on a trained model checkpoint.
# Usage: sbatch scripts/slurm_lm_eval.sh <run_name> <tasks>
#   tasks: "blimp", "math", "code", or "all"
# ============================================================

set -euo pipefail

RUN_NAME="${1:?Usage: sbatch slurm_lm_eval.sh <run_name> <tasks>}"
TASKS="${2:-all}"

REPO_DIR="${HOME}/tokenizer-lm"
VENV_DIR="${HOME}/tokenizer_lm_exps_env"
CHECKPOINT_BASE="/capstor/store/cscs/swissai/a139/cmeister/tokenizer-lm/checkpoints"

export PATH="${VENV_DIR}/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/nanochat:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL=1

PYTHON="${VENV_DIR}/bin/python"

# Get tokenizer path from checkpoint metadata
CKPT_DIR="${CHECKPOINT_BASE}/${RUN_NAME}"
META_FILE=$(ls "${CKPT_DIR}"/meta_*.json 2>/dev/null | sort | tail -1)
TOKENIZER=$(${PYTHON} -c "import json; print(json.load(open('${META_FILE}'))['tokenizer_path'])")

mkdir -p "${REPO_DIR}/results/lm_eval"
OUTPUT="${REPO_DIR}/results/lm_eval/${RUN_NAME}_${TASKS}.json"

echo "=== LM Eval ==="
echo "Job: ${SLURM_JOB_ID}"
echo "Run: ${RUN_NAME}"
echo "Tasks: ${TASKS}"
echo "Tokenizer: ${TOKENIZER}"
echo "Output: ${OUTPUT}"

${PYTHON} -u ${REPO_DIR}/eval_runner.py \
    --checkpoint-dir "${CKPT_DIR}" \
    --tokenizer "${TOKENIZER}" \
    --output "${OUTPUT}" \
    --tasks ${TASKS} \
    --batch-size 16 \
    --device cuda
