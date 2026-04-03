#!/bin/bash
#SBATCH --job-name=eval-cgen
#SBATCH --account=a139
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_cgen_%j.out
#SBATCH --error=logs/eval_cgen_%j.err
#SBATCH --uenv=pytorch/v2.8.0:v1
#SBATCH --view=default

# HumanEval + MBPP (164 + 500 = 664 problems, ~3h estimated)
set -euo pipefail
RUN_NAME="${1:?Usage: sbatch slurm_eval_code_gen.sh <run_name>}"
REPO_DIR="${HOME}/tokenizer-lm"
VENV_DIR="${HOME}/tokenizer_lm_exps_env"
CKPT_DIR="/capstor/store/cscs/swissai/a139/cmeister/tokenizer-lm/checkpoints/${RUN_NAME}"
export PATH="${VENV_DIR}/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/nanochat:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL=1
PYTHON="${VENV_DIR}/bin/python"
META_FILE=$(ls "${CKPT_DIR}"/meta_*.json 2>/dev/null | sort | tail -1)
TOKENIZER=$(${PYTHON} -c "import json; print(json.load(open('${META_FILE}'))['tokenizer_path'])")
mkdir -p "${REPO_DIR}/results/lm_eval"
OUTPUT="${REPO_DIR}/results/lm_eval/${RUN_NAME}_code_gen.json"
echo "=== Code Gen === Job: ${SLURM_JOB_ID} | Run: ${RUN_NAME}"
${PYTHON} -u ${REPO_DIR}/eval_runner.py \
    --checkpoint-dir "${CKPT_DIR}" --tokenizer "${TOKENIZER}" \
    --output "${OUTPUT}" --tasks code --batch-size 1 --device cuda
