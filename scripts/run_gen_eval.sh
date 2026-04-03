#!/bin/bash
# Run Phase B generation evals (limit=100) on specified GPUs.
# Automatically detects which models still need evaluation, skipping any
# that are already done or currently being evaluated on another node.
#
# Usage:
#   bash scripts/run_gen_eval.sh 0 1 2 3     # Use GPUs 0,1,2,3
#   bash scripts/run_gen_eval.sh 0 2          # Use GPUs 0,2 only
#   bash scripts/run_gen_eval.sh 0            # Single GPU
#
# Run with nohup to persist after logout:
#   nohup bash scripts/run_gen_eval.sh 0 1 2 3 > logs/gen_eval_$(hostname).log 2>&1 &

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: bash scripts/run_gen_eval.sh <gpu_id> [gpu_id ...]"
    echo "Example: bash scripts/run_gen_eval.sh 0 1 2 3"
    exit 1
fi

GPUS=("$@")
NUM_GPUS=${#GPUS[@]}

REPO_DIR="${HOME}/tokenizer-lm"
VENV_DIR="${HOME}/tokenizer_lm_exps_env"
CKPT_BASE="/capstor/store/cscs/swissai/a139/cmeister/tokenizer-lm/checkpoints"
RESULTS_DIR="${REPO_DIR}/results/lm_eval"
PYTHON="${VENV_DIR}/bin/python"
LIMIT=100

export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/nanochat:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL=1

cd "${REPO_DIR}"

# All completed 1B models
ALL_MODELS=(
    full-128k-apertus
    full-128k-claude-balanced-bpe
    full-128k-claude-balanced-nfc-bpe
    full-128k-claude-balanced-unigram
    full-128k-claude-english-bpe
    full-128k-gpt4o-balanced-bpe
    full-128k-gpt4o-balanced-nfc-bpe
    full-128k-gpt4o-balanced-unigram
    full-128k-gpt4o-code-bpe
    full-128k-gpt4o-english-bpe
    full-128k-llama3
    full-128k-pabpe
    full-128k-punct-balanced-bpe
    full-128k-punct-english-bpe
    full-128k-rightalign-balanced-bpe
    full-128k-rightalign-balanced-nfc-bpe
    full-128k-rightalign-balanced-unigram
    full-128k-superbpe
)

# Determine which models still need evaluation.
# A model is skipped if:
#   1. Its result file exists (done), OR
#   2. Its lock file exists (being evaluated on another node)
LOCK_DIR="${RESULTS_DIR}/.locks"
mkdir -p "${LOCK_DIR}"

REMAINING=()
for model in "${ALL_MODELS[@]}"; do
    result="${RESULTS_DIR}/${model}_gen_limit${LIMIT}.json"
    lock="${LOCK_DIR}/${model}.lock"

    if [ -f "${result}" ]; then
        continue  # already done
    fi

    # Try to acquire lock (atomic via mkdir)
    if mkdir "${lock}" 2>/dev/null; then
        # Write our hostname and PID into the lock for debugging
        echo "$(hostname):$$:$(date -Iseconds)" > "${lock}/info"
        REMAINING+=("${model}")
    else
        # Lock exists — check if the holder is still alive
        if [ -f "${lock}/info" ]; then
            lock_host=$(cut -d: -f1 "${lock}/info")
            if [ "${lock_host}" = "$(hostname)" ]; then
                # Same host — check if PID is alive
                lock_pid=$(cut -d: -f2 "${lock}/info")
                if ! kill -0 "${lock_pid}" 2>/dev/null; then
                    # Stale lock from dead process on this host — reclaim
                    rm -rf "${lock}"
                    mkdir "${lock}" 2>/dev/null && {
                        echo "$(hostname):$$:$(date -Iseconds)" > "${lock}/info"
                        REMAINING+=("${model}")
                    }
                    continue
                fi
            fi
        fi
        # Lock held by another node or active process — skip
    fi

    # Stop once we have enough models for our GPUs
    if [ ${#REMAINING[@]} -ge ${NUM_GPUS} ]; then
        break
    fi
done

if [ ${#REMAINING[@]} -eq 0 ]; then
    echo "No models remaining to evaluate. All done or locked by other nodes."
    exit 0
fi

echo "=== Gen Eval (limit=${LIMIT}) on $(hostname) ==="
echo "GPUs: ${GPUS[*]}"
echo "Models to evaluate (${#REMAINING[@]}):"
for m in "${REMAINING[@]}"; do echo "  ${m}"; done
echo "Started: $(date)"
echo ""

# Cleanup locks on exit (for models that didn't finish)
cleanup() {
    for model in "${REMAINING[@]}"; do
        result="${RESULTS_DIR}/${model}_gen_limit${LIMIT}.json"
        lock="${LOCK_DIR}/${model}.lock"
        if [ ! -f "${result}" ] && [ -d "${lock}" ]; then
            # Only remove our own locks
            if [ -f "${lock}/info" ]; then
                lock_host=$(cut -d: -f1 "${lock}/info")
                lock_pid=$(cut -d: -f2 "${lock}/info")
                if [ "${lock_host}" = "$(hostname)" ] && [ "${lock_pid}" = "$$" ]; then
                    rm -rf "${lock}"
                fi
            fi
        fi
    done
}
trap cleanup EXIT

eval_model() {
    local run_name="$1"
    local gpu="$2"
    local ckpt_dir="${CKPT_BASE}/${run_name}"
    local meta_file=$(ls "${ckpt_dir}"/meta_*.json 2>/dev/null | sort | tail -1)
    local tokenizer=$(${PYTHON} -c "import json; print(json.load(open('${meta_file}'))['tokenizer_path'])")
    local output="${RESULTS_DIR}/${run_name}_gen_limit${LIMIT}.json"
    local lock="${LOCK_DIR}/${run_name}.lock"

    echo "[GPU${gpu}] START ${run_name} ($(date +%H:%M:%S))"
    if ${PYTHON} -u ${REPO_DIR}/eval_runner.py \
        --checkpoint-dir "${ckpt_dir}" \
        --tokenizer "${tokenizer}" \
        --output "${output}" \
        --tasks gsm8k mgsm code \
        --limit ${LIMIT} \
        --batch-size 1 \
        --device "cuda:${gpu}"; then
        echo "[GPU${gpu}] DONE ${run_name} ($(date +%H:%M:%S))"
        # Remove lock after successful completion (result file serves as the record)
        rm -rf "${lock}"
    else
        echo "[GPU${gpu}] FAILED ${run_name} ($(date +%H:%M:%S))"
        rm -rf "${lock}"
    fi
    echo ""
}

# Launch one model per GPU in parallel
for i in "${!REMAINING[@]}"; do
    gpu=${GPUS[$i]}
    model=${REMAINING[$i]}
    eval_model "${model}" "${gpu}" &
done

wait
echo ""
echo "=== Finished: $(date) ==="
