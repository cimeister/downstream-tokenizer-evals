#!/bin/bash
# Launch pilot-scale tokenizer comparison experiments SEQUENTIALLY.
#
# Compares diverse tokenizers within two vocab-size buckets:
#   ~50K:  GPT-2, Pythia, StarCoder2, SmolLM2 (4 distinct tokenizers)
#   ~128K: Apertus/Tekken, LLaMA-3 (2 distinct tokenizers)
#
# All experiments use the same data mixture and architecture (within each bucket).
# Only the tokenizer varies.
#
# Usage: bash scripts/run_pilot_experiments.sh [bucket]
#   bucket: "50k", "128k", or "all" (default: all)

set -euo pipefail

REPO_DIR="${HOME}/tokenizer-lm"
DATA_DIR="/capstor/scratch/cscs/${USER}/data/tokenizer-lm-mix-pilot"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/nanochat:${PYTHONPATH:-}"

BUCKET="${1:-all}"

if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: Data mixture not found at ${DATA_DIR}"
    echo "Run first: python data/prepare.py mix ..."
    exit 1
fi

mkdir -p "${REPO_DIR}/logs"

run_experiment() {
    local config="$1"
    local tok_name="$2"
    local run_name="$3"

    echo ""
    echo "============================================================"
    echo "  Experiment: ${run_name}"
    echo "  Tokenizer:  ${tok_name}"
    echo "  Config:     ${config}"
    echo "============================================================"

    torchrun --nproc_per_node=4 \
        ${REPO_DIR}/train.py \
        --config ${config} \
        --tokenizer "${tok_name}" \
        --data-dir "${DATA_DIR}" \
        --run-name "${run_name}" \
        2>&1 | tee ${REPO_DIR}/logs/${run_name}.log

    echo "  Completed: ${run_name}"
    echo ""
}

if [ "${BUCKET}" = "50k" ] || [ "${BUCKET}" = "all" ]; then
    echo "=== 50K vocab bucket ==="
    echo "Tokenizers: GPT-2 (50257), Pythia (50277), StarCoder2 (49152), SmolLM2 (49152)"
    echo ""

    run_experiment configs/pilot_50k.yaml "openai-community/gpt2" "pilot-50k-gpt2"
    run_experiment configs/pilot_50k.yaml "EleutherAI/pythia-1b" "pilot-50k-pythia"
    run_experiment configs/pilot_50k.yaml "bigcode/starcoder2-3b" "pilot-50k-starcoder2"
    run_experiment configs/pilot_50k.yaml "HuggingFaceTB/SmolLM2-1.7B" "pilot-50k-smollm2"
fi

if [ "${BUCKET}" = "128k" ] || [ "${BUCKET}" = "all" ]; then
    echo "=== 128K vocab bucket ==="
    echo "Tokenizers: Apertus (131072), LLaMA-3 (128256)"
    echo ""

    run_experiment configs/pilot_128k.yaml "swiss-ai/Apertus-70B-2509" "pilot-128k-apertus"
    run_experiment configs/pilot_128k.yaml "NousResearch/Meta-Llama-3-8B" "pilot-128k-llama3"
fi

echo ""
echo "=== All experiments complete ==="
echo "Results logged to W&B project: ${WANDB_PROJECT:-tokenizer-lm}"
echo "Checkpoints at: /capstor/scratch/cscs/${USER}/tokenizer-lm/checkpoints/"
