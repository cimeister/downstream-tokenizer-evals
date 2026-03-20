# tokenizer-lm

A controlled experimental framework for measuring how **tokenization affects language model quality**. Train identical architectures with different tokenizers on the same data, compare on downstream benchmarks. All models trained from scratch.

## Motivation

Tokenization is a foundational design choice in language model training, yet its impact on downstream performance is poorly understood in isolation. Most tokenizer comparisons are confounded by differences in model architecture, training data, or compute budget. This repo isolates the tokenizer as the **sole experimental variable** by holding everything else constant.

Key experimental properties:
- **Data is tokenizer-independent**: raw text stored as parquet files, tokenized on-the-fly during training
- **Same architecture** for all tokenizers within a vocab-size bucket
- **Same training data** (identical documents in identical order) across all runs
- **Primary metric is BPB** (bits-per-byte): a tokenizer-independent measure of model quality
- **Per-vocab-size configs**: tokenizers are compared within the same vocab size (64K, 128K, 256K), never across

## Design choices

### Architecture

Built on [nanochat](https://github.com/karpathy/nanochat) (git submodule) with all its architectural innovations. These interact with tokenization in known ways, documented here for transparency:

| Feature | Tokenization interaction |
|---------|------------------------|
| **RoPE** (rotary positional embeddings) | None — position-based, not token-based |
| **QK-Norm** (RMSNorm on Q,K) | None |
| **ReLU²** activation | None |
| **Muon optimizer** (Polar Express) | Used for transformer matrices only; embeddings use AdamW. Optimizer behavior is identical across tokenizers within the same vocab size |
| **Value embeddings** (ResFormer-style) | Per-token-ID lookup tables. Dimension capped at `ve_dim=128` with learned projection to prevent embedding-dominated parameter budgets. Same total VE params for same vocab size |
| **Sliding window attention** (`SSSL` pattern) | Window size is in tokens. A more efficient tokenizer covers more bytes per window — this is a real effect, not an artifact |
| **Smear gate** (adjacent token mixing) | Token boundaries differ by tokenizer, so "adjacent" means different things. Effect is small (gate is learned and initialized near zero) |
| **Logit soft-capping** (`15·tanh(logits/15)`) | Applied uniformly regardless of vocab size |
| **Parameterless RMSNorm** | None |
| **Norm after embedding** | None |

### Training budget

Token budget is computed as `target_param_data_ratio × (transformer_matrices + lm_head)` with ratio 10.5 (Chinchilla-like). This means:
- Transformer matrix params are identical across tokenizers within the same vocab size
- `lm_head` params scale with vocab size, so the token budget scales proportionally
- Since comparisons are within the same vocab size, all runs get the same budget

Batch size is auto-computed via Power Lines scaling (`B_opt ∝ D^0.383`). LR and weight decay are scaled accordingly. All schedules (warmup, warmdown, momentum) are identical across runs.

### Evaluation metrics

- **BPB (bits-per-byte)**: Primary metric for both training validation and FLORES-200 multilingual evaluation. Computed as `sum(nats) / (ln(2) × sum(bytes))`, weighting each token's loss by its UTF-8 byte length. Special tokens are excluded. This is inherently tokenizer-independent.
- **Downstream tasks** (GSM8K, MGSM, HumanEval, MBPP, BLiMP): These have natural tokenizer sensitivity — measuring that sensitivity is part of the experimental goal.

### What is NOT controlled

These effects are inherent to tokenizer comparison and are part of what is being measured:
- **Packing efficiency**: Different tokenizers produce different token counts for the same text. A more efficient tokenizer sees more unique text per training step (~35% of tokens are cropped in the BOS-aligned best-fit packing).
- **Effective context length**: A 2048-token sliding window covers different amounts of text depending on tokenizer fertility.
- **Token boundary effects**: Smear gate and attention patterns operate on token boundaries, which differ by tokenizer.

## Data

Raw text in parquet files, sourced from datasets already available on CSCS Clariden. Data preparation is a one-time operation shared across all tokenizer experiments.

### Sources

| Domain | Dataset | Path on Clariden | Weight |
|--------|---------|-----------------|--------|
| English web | FineWeb-Edu | `.../HuggingFaceFW/fineweb-edu/data` | 45% |
| Multilingual | Filtered FineWeb2 (top 33% quality) | `.../swiss-ai/fineweb-2_0_1-quality_33-filterrobots/data/output/{lang}` | 20% |
| Math | FineMath-4plus (score ≥ 4) | `.../HuggingFaceTB/finemath/finemath-4plus` | 15% |
| Code | StarCoderData (highest quality tier) | `.../swiss-ai/starcoderdata/thresholds/{lang}/threshold_0` | 15% |

Multilingual split across: French, German, Spanish, Russian, Japanese, Chinese, Arabic, Hindi, Korean, Thai, Vietnamese.

### Creating the mixture

```bash
bash scripts/create_mixture.sh          # 5M docs default
bash scripts/create_mixture.sh 10000000 # 10M docs
```

Or use `data/prepare.py` directly for custom mixtures:
```bash
python data/prepare.py mix \
    --sources /path/to/source1:0.5:text /path/to/source2:0.3:content \
    --output /capstor/scratch/cscs/$USER/data/my-mix \
    --total-docs 5000000
```

Sources are `path:weight[:text_field]`. Parquet files are discovered recursively. The `text_field` defaults to `text` but can be overridden (e.g., `content` for StarCoderData).

## Configs

Per-vocab-size configs with properly balanced architectures. Compare tokenizers **only within the same vocab-size bucket**.

### Pilot scale (single node, 4× GH200 GPUs)

| Config | Vocab | n_embd | n_layer | Transformer params | Total params |
|--------|-------|--------|---------|-------------------|-------------|
| `pilot_50k.yaml` | ~50K | 1024 | 16 | ~300M | ~400M |
| `pilot_64k.yaml` | ~64K | 1024 | 16 | ~300M | ~500M |
| `pilot_128k.yaml` | ~128K | 1024 | 16 | ~300M | ~700M |
| `pilot_256k.yaml` | ~256K | 1280 | 16 | ~470M | ~1.4B |

### Full scale (4 nodes, 16× GH200 GPUs)

| Config | Vocab | n_embd | n_layer | Transformer params | Total params |
|--------|-------|--------|---------|-------------------|-------------|
| `full_64k.yaml` | ~64K | 2048 | 24 | ~1.2B | ~1.6B |
| `full_128k.yaml` | ~128K | 2048 | 24 | ~1.2B | ~1.9B |
| `full_256k.yaml` | ~256K | 2048 | 24 | ~1.2B | ~2.5B |

Token budgets are auto-computed. All configs use `ve_dim=128` to cap value embedding parameters.

## Setup on CSCS Clariden

```bash
ssh clariden.alps.cscs.ch

git clone --recurse-submodules <repo-url> tokenizer-lm
cd tokenizer-lm

# Python environment
uenv start pytorch/v2.8.0:v1 --view=default
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install datasets pyarrow wandb lm-eval evaluate sentencepiece tiktoken rustbpe

export PYTHONPATH="$HOME/tokenizer-lm:$HOME/tokenizer-lm/nanochat"
export WANDB_API_KEY="your-key"
export WANDB_PROJECT="tokenizer-lm"
```

## Running experiments

### 1. Prepare data (once)

```bash
bash scripts/create_mixture.sh
```

### 2. Train

```bash
# Interactive (on a node with GPUs)
torchrun --nproc_per_node=4 train.py \
    --config configs/pilot_128k.yaml \
    --tokenizer swiss-ai/Apertus-70B-2509 \
    --data-dir /capstor/scratch/cscs/$USER/data/tokenizer-lm-mix-pilot \
    --run-name pilot-128k-apertus

# Via SLURM
sbatch scripts/slurm_pilot.sh configs/pilot_128k.yaml \
    swiss-ai/Apertus-70B-2509 \
    /capstor/scratch/cscs/$USER/data/tokenizer-lm-mix-pilot \
    pilot-128k-apertus
```

### 3. Evaluate

```bash
sbatch scripts/slurm_eval.sh \
    /capstor/scratch/cscs/$USER/tokenizer-lm/checkpoints/pilot-128k-apertus \
    swiss-ai/Apertus-70B-2509 \
    all
```

### 4. Compare

Results are saved as structured JSON. The primary comparison metric is **val BPB** (logged to W&B as `val/bpb`). FLORES-200 evaluation reports per-language and per-language-family BPB.

### Batch experiments

```bash
bash scripts/run_pilot_experiments.sh        # all buckets
bash scripts/run_pilot_experiments.sh 50k    # 50K bucket only
bash scripts/run_pilot_experiments.sh 128k   # 128K bucket only
```

## Evaluation benchmarks

| Group | Tasks | What it measures |
|-------|-------|-----------------|
| Math | GSM8K (8-shot CoT), MGSM (11 languages) | Math reasoning, cross-lingual math × tokenization |
| Code | HumanEval, MBPP | Code generation quality |
| Multilingual | FLORES-200 BPB (per-language, per-family) | **Tokenization fairness across languages** |
| Linguistic | BLiMP (67 subtasks) | Grammatical knowledge |

## Estimated MFU

Target: **60–70% on GH200 GPUs** at full scale.

- GH200 uses the H100 Hopper die: 989 TFLOPS bf16 peak
- `torch.compile` + Flash Attention 3 + ReLU² + bf16 throughout
- Pilot scale (~300M transformer): ~25-30% MFU (model too small to saturate GPU)
- Full scale (~1.2B transformer): ~60-70% MFU expected

## Project structure

```
tokenizer-lm/
├── nanochat/                 # Git submodule (model, optimizer, dataloader)
├── configs/
│   ├── pilot_{50k,64k,128k,256k}.yaml
│   └── full_{64k,128k,256k}.yaml
├── scripts/
│   ├── slurm_pilot.sh        # SLURM: single-node training
│   ├── slurm_full.sh         # SLURM: multi-node training
│   ├── slurm_eval.sh         # SLURM: evaluation
│   ├── create_mixture.sh     # Build data mixture from Clariden datasets
│   └── run_pilot_experiments.sh  # Launch tokenizer comparison experiments
├── train.py                  # Training entry point
├── evaluate.py               # LM Eval Harness + FLORES-200 BPB
├── data/
│   └── prepare.py            # Build data mixtures from parquet sources
├── tokenizer_lm/
│   ├── __init__.py
│   └── tokenizer.py          # Tokenizer wrapper (robust BOS detection, token_bytes)
├── environment.yml
├── README.md
└── EXPERIMENTS.md            # Lab notebook template
```
