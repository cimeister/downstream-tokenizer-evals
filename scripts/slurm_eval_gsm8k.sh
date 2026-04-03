#!/bin/bash
#SBATCH --job-name=eval-gsm
#SBATCH --account=a139
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_gsm_%j.out
#SBATCH --error=logs/eval_gsm_%j.err
#SBATCH --uenv=pytorch/v2.8.0:v1
#SBATCH --view=default

# GSM8K only (8-shot CoT, ~1319 problems, ~6h estimated)
set -euo pipefail
RUN_NAME="${1:?Usage: sbatch slurm_eval_gsm8k.sh <run_name>}"
REPO_DIR="${HOME}/tokenizer-lm"
VENV_DIR="${HOME}/tokenizer_lm_exps_env"
CKPT_DIR="/capstor/store/cscs/swissai/a139/cmeister/tokenizer-lm/checkpoints/${RUN_NAME}"
export PATH="${VENV_DIR}/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/nanochat:${PYTHONPATH:-}"
PYTHON="${VENV_DIR}/bin/python"
META_FILE=$(ls "${CKPT_DIR}"/meta_*.json 2>/dev/null | sort | tail -1)
TOKENIZER=$(${PYTHON} -c "import json; print(json.load(open('${META_FILE}'))['tokenizer_path'])")
mkdir -p "${REPO_DIR}/results/lm_eval"
OUTPUT="${REPO_DIR}/results/lm_eval/${RUN_NAME}_gsm8k.json"
echo "=== GSM8K === Job: ${SLURM_JOB_ID} | Run: ${RUN_NAME}"
${PYTHON} -u ${REPO_DIR}/eval_runner.py \
    --checkpoint-dir "${CKPT_DIR}" --tokenizer "${TOKENIZER}" \
    --output "${OUTPUT}" --tasks gsm8k --batch-size 1 --device cuda
