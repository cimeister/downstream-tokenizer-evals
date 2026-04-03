#!/bin/bash
# Create the training data mixture from datasets available on Clariden.
#
# Usage: bash scripts/create_mixture.sh [total_docs]
#   total_docs: number of documents in the mixture (default: 5000000)
#
# Domain split: 35% English, 30% multilingual (30 languages), 15% math, 15% code
# Multilingual weights are proportional to estimated character counts in
# the source data (filtered FineWeb2, quality_33, robot-filtered).
#
# Output: /capstor/scratch/cscs/$USER/data/tokenizer-lm-mix

set -euo pipefail

TOTAL_DOCS="${1:-5000000}"
REPO_DIR="${HOME}/tokenizer-lm"
VENV_DIR="${HOME}/tokenizer_lm_exps_env"
OUTPUT="/capstor/store/cscs/swissai/a139/cmeister/tokenizer-lm/data"

source "${VENV_DIR}/bin/activate"

# --- Data roots ---
FINEWEB_EDU="/capstor/store/cscs/swissai/infra01/datasets/HuggingFaceFW/fineweb-edu/data"
FINEWEB2="/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/fineweb-2_0_1-quality_33-filterrobots/data/output"
FINEMATH="/capstor/store/cscs/swissai/infra01/datasets/HuggingFaceTB/finemath/finemath-4plus"
STARCODERDATA="/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/starcoderdata/thresholds"

echo "=== Creating training mixture ==="
echo "Total docs: ${TOTAL_DOCS}"
echo "Output: ${OUTPUT}"
echo ""

# Verify sources exist
for src in "$FINEWEB_EDU" "$FINEWEB2" "$FINEMATH" "$STARCODERDATA"; do
    if [ ! -d "$src" ]; then
        echo "ERROR: Source not found: $src"
        exit 1
    fi
done

export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/nanochat:${PYTHONPATH:-}"

# Weights: 35% English, 30% multilingual (char-proportional), 15% math, 15% code
# Multilingual weights derived from estimated character counts in source data.
# See scripts/train_custom_tokenizers.py for the full derivation.
python ${REPO_DIR}/data/prepare.py mix \
    --sources \
        "${FINEWEB_EDU}:0.35:text" \
        "${FINEWEB2}/rus_Cyrl:0.10104:text" \
        "${FINEWEB2}/spa_Latn:0.02324:text" \
        "${FINEWEB2}/deu_Latn:0.02257:text" \
        "${FINEWEB2}/fra_Latn:0.01957:text" \
        "${FINEWEB2}/cmn_Hani:0.01819:text" \
        "${FINEWEB2}/jpn_Jpan:0.01202:text" \
        "${FINEWEB2}/ita_Latn:0.01135:text" \
        "${FINEWEB2}/por_Latn:0.01131:text" \
        "${FINEWEB2}/tur_Latn:0.00894:text" \
        "${FINEWEB2}/ind_Latn:0.00867:text" \
        "${FINEWEB2}/pol_Latn:0.00836:text" \
        "${FINEWEB2}/ukr_Cyrl:0.00644:text" \
        "${FINEWEB2}/nld_Latn:0.00617:text" \
        "${FINEWEB2}/ron_Latn:0.00550:text" \
        "${FINEWEB2}/arb_Arab:0.00514:text" \
        "${FINEWEB2}/hun_Latn:0.00483:text" \
        "${FINEWEB2}/vie_Latn:0.00411:text" \
        "${FINEWEB2}/ces_Latn:0.00407:text" \
        "${FINEWEB2}/ell_Grek:0.00335:text" \
        "${FINEWEB2}/fin_Latn:0.00223:text" \
        "${FINEWEB2}/tha_Thai:0.00219:text" \
        "${FINEWEB2}/slk_Latn:0.00188:text" \
        "${FINEWEB2}/bul_Cyrl:0.00183:text" \
        "${FINEWEB2}/hrv_Latn:0.00161:text" \
        "${FINEWEB2}/kor_Hang:0.00156:text" \
        "${FINEWEB2}/cat_Latn:0.00098:text" \
        "${FINEWEB2}/hin_Deva:0.00094:text" \
        "${FINEWEB2}/heb_Hebr:0.00089:text" \
        "${FINEWEB2}/ben_Beng:0.00067:text" \
        "${FINEWEB2}/tam_Taml:0.00036:text" \
        "${FINEMATH}:0.15:text" \
        "${STARCODERDATA}/python/threshold_0:0.10:content" \
        "${STARCODERDATA}/javascript/threshold_0:0.05:content" \
    --output "${OUTPUT}" \
    --total-docs "${TOTAL_DOCS}" \
    --val-docs 10000

echo ""
echo "=== Mixture created ==="
echo "To train: pass --data-dir ${OUTPUT} to train.py"
