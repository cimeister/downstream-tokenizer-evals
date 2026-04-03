# Model Training Plan

## Overview

Train 16 language models at two scales (pilot and full), each with a different tokenizer but identical architecture, data, and hyperparameters. The sole experimental variable is the tokenizer.

- **Pilot scale** (completed): nanochat d16 (~596M total params), single-node, for fast iteration and FLORES/BLiMP evaluation.
- **Full scale**: nanochat d24 (~1.27B total params), 4-node, for generative benchmarks (GSM8K, MGSM, HumanEval, MBPP) in addition to FLORES/BLiMP.

## Tokenizers (16 total)

### Off-the-shelf (2)

| Name | Vocab | Pretokenizer | Training data |
|------|-------|-------------|---------------|
| Apertus | 131,072 | GPT-4o-style regex | Multilingual web + code |
| LLaMA-3 | 128,256 | Tiktoken BPE | Massive multilingual |

### Custom BPE (11)

All trained at vocab size 128,256 on ~10GB of data from the same sources as LM training. Sampling is character-proportional, deterministic (seed=42), and reads directly from parquet with no intermediate files.

| Name | Pretokenizer | Training data | NFC |
|------|-------------|---------------|-----|
| `punct-balanced-bpe` | Punctuation + ByteLevel | Balanced multilingual | No |
| `punct-english-bpe` | Punctuation + ByteLevel | English-only | No |
| `gpt4o-balanced-bpe` | GPT-4o regex | Balanced multilingual | No |
| `gpt4o-balanced-nfc-bpe` | GPT-4o regex | Balanced multilingual | Yes |
| `gpt4o-english-bpe` | GPT-4o regex | English-only | No |
| `gpt4o-code-bpe` | GPT-4o regex | Code-heavy | No |
| `claude-balanced-bpe` | Claude regex | Balanced multilingual | No |
| `claude-balanced-nfc-bpe` | Claude regex | Balanced multilingual | Yes |
| `claude-english-bpe` | Claude regex | English-only | No |
| `rightalign-balanced-bpe` | Right-aligned digits | Balanced multilingual | No |
| `rightalign-balanced-nfc-bpe` | Right-aligned digits | Balanced multilingual | Yes |

### Custom UnigramLM (3)

| Name | Pretokenizer | Training data |
|------|-------------|---------------|
| `gpt4o-balanced-unigram` | GPT-4o regex | Balanced multilingual |
| `claude-balanced-unigram` | Claude regex | Balanced multilingual |
| `rightalign-balanced-unigram` | Right-aligned digits | Balanced multilingual |

---

## Model architecture

Built on nanochat. All 16 models use identical architecture — only the tokenizer-dependent components (wte, lm_head, value embeddings) change size.

### Pilot scale (d16, config: `pilot_128k.yaml`)

| Parameter | Value |
|-----------|-------|
| `n_layer` | 16 |
| `n_embd` | 1024 |
| `n_head` | 8 (head_dim=128) |
| `n_kv_head` | 8 (MHA) |
| `sequence_len` | 2048 |
| `window_pattern` | `SSSL` (3 short-window layers + 1 full-context, tiled) |
| `ve_dim` | 128 (value embedding dimension, projected up to kv_dim=1024) |
| Activation | ReLU² |
| Positional encoding | RoPE (base=100K) |
| Normalization | Parameterless RMSNorm |
| Logit softcapping | 15·tanh(logits/15) |

#### Pilot parameter counts (128,256 vocab)

| Component | Parameters | Notes |
|-----------|-----------|-------|
| Transformer matrices | 202,375,936 | Identical across tokenizers |
| wte (input embedding) | 131,334,144 | Scales with vocab |
| lm_head (output projection) | 131,334,144 | Scales with vocab |
| Value embeddings (8 layers × 128 dim) | 131,334,144 | Scales with vocab |
| Scalars (lambdas, gates) | 58 | Identical |
| **Total** | **596,378,426** | |

For Apertus (131,072 vocab): total is ~599M (wte/lm_head/VE each ~2.9M larger).

### Full scale (d24, config: `full_128k.yaml`)

Uses nanochat's default d24 configuration (`depth × aspect_ratio = 24 × 64 = 1536` width). This is nanochat's canonical reference depth, with validated hyperparameters across the d12–d26 range.

| Parameter | Value |
|-----------|-------|
| `n_layer` | 24 |
| `n_embd` | 1536 |
| `n_head` | 12 (head_dim=128) |
| `n_kv_head` | 12 (MHA) |
| `sequence_len` | 2048 |
| `window_pattern` | `SSSL` |
| `ve_dim` | 128 (projected up to kv_dim=1536) |
| Activation | ReLU² |
| Positional encoding | RoPE (base=100K) |
| Normalization | Parameterless RMSNorm |
| Logit softcapping | 15·tanh(logits/15) |

#### Full-scale parameter counts (128,256 vocab)

| Component | Parameters | Notes |
|-----------|-----------|-------|
| Transformer matrices | 681,838,272 | Identical across tokenizers |
| wte (input embedding) | 197,001,216 | Scales with vocab |
| lm_head (output projection) | 197,001,216 | Scales with vocab |
| Value embeddings (12 layers × 128 dim) | 197,001,216 | Scales with vocab |
| Scalars (lambdas, gates) | 74 | Identical |
| **Total** | **1,272,841,994** | |

#### Why ve_dim=128

nanochat's default (`ve_dim=0`) uses full `kv_dim = n_kv_head × head_dim`. At d24 with 128K vocab, this would add 12 × 128,320 × 1,536 = 2.36B VE parameters — larger than the entire rest of the model. Capping at `ve_dim=128` with a learned projection (Linear(128, 1536) per VE layer) gives 197M VE parameters instead, keeping VE proportional to embeddings.

---

## Training data

### LM training mixture

Created with:
```bash
bash scripts/create_mixture.sh 5000000
```

Which runs:
```bash
python data/prepare.py mix \
    --sources <34 source:weight:text_field entries> \
    --output /capstor/scratch/cscs/$USER/data/tokenizer-lm-mix \
    --total-docs 5000000 --val-docs 10000
```

See `scripts/create_mixture.sh` for the full command with all source paths and weights.

**Result**: 5,000,000 documents, 24.97 GB of text (UTF-8 bytes), 101 parquet shards. Last shard is validation (10K docs, 100 docs/row-group for multi-GPU compatibility). Stored at `/capstor/store/cscs/swissai/a139/cmeister/tokenizer-lm/data/`.

**Mixture metadata** (per-source byte counts, doc counts, and actual byte fractions): [`/capstor/store/cscs/swissai/a139/cmeister/tokenizer-lm/data/metadata.json`](/capstor/store/cscs/swissai/a139/cmeister/tokenizer-lm/data/metadata.json).

**Tokenizer training data configs** (specifying per-language GB targets for the three compositions used to train custom tokenizers): [`configs/data/english.json`](configs/data/english.json), [`configs/data/balanced.json`](configs/data/balanced.json), [`configs/data/code.json`](configs/data/code.json). See [TOKENIZER_TRAINING.md](TOKENIZER_TRAINING.md) for full details on tokenizer training data compositions.

All tokenizers train on the same documents in the same order (deterministic via seed=42, parquet shard ordering, and DDP rank sharding). Tokenization happens on-the-fly via nanochat's streaming dataloader.

### Sampling method

Documents are sampled via **weighted random source selection**: each draw picks a source with probability proportional to its weight, then reads the next document from that source's parquet iterator. Weights are specified as document-sampling probabilities, derived from estimated character counts in the source data (see below). Because document lengths vary across sources, the **actual byte proportions diverge from the sampling weights**.

### Domain split — actual bytes

Measured from the output mixture (text UTF-8 bytes, not parquet file sizes):

| Domain | Sampling weight | Actual text bytes | Actual % |
|--------|----------------|------------------|----------|
| English web (FineWeb-Edu) | 36.8% | 9.225 GB | 36.9% |
| Multilingual (30 languages) | 31.6% | 7.768 GB | 31.1% |
| Math (FineMath-4plus) | 15.8% | 4.009 GB | 16.1% |
| Code — Python (StarCoder threshold_0) | 10.5% | 2.494 GB | 10.0% |
| Code — JavaScript (StarCoder threshold_0) | 5.3% | 0.889 GB | 3.6% |
| **Total** | | **24.97 GB** | |

Note: JavaScript gets 3.6% actual bytes despite 5.3% sampling weight because JS snippets are shorter than average.

### Multilingual language breakdown — actual bytes

Sampling weights are proportional to estimated character counts in the source data (filtered FineWeb2). Character counts were approximated by: avg chars/doc (from first parquet file) × rows per file × number of files.

Because document lengths vary by language, actual byte proportions diverge from sampling weights. Russian is the most affected: its documents average ~8.5K chars (vs ~5K overall), so it receives 18.1% of bytes despite a 10.6% sampling weight.

| Language | Code | Script | Sampling weight | Actual text bytes | Actual % | Docs |
|----------|------|--------|----------------|------------------|----------|------|
| Russian | `rus_Cyrl` | Cyrillic | 10.6% | 4.521 GB | 18.1% | 531,387 |
| Spanish | `spa_Latn` | Latin | 2.4% | 0.423 GB | 1.7% | 122,231 |
| German | `deu_Latn` | Latin | 2.4% | 0.353 GB | 1.4% | 118,340 |
| French | `fra_Latn` | Latin | 2.1% | 0.364 GB | 1.5% | 103,496 |
| Chinese | `cmn_Hani` | CJK | 1.9% | 0.326 GB | 1.3% | 95,805 |
| Japanese | `jpn_Jpan` | CJK | 1.3% | 0.204 GB | 0.8% | 63,316 |
| Italian | `ita_Latn` | Latin | 1.2% | 0.165 GB | 0.7% | 59,676 |
| Portuguese | `por_Latn` | Latin | 1.2% | 0.189 GB | 0.8% | 59,460 |
| Turkish | `tur_Latn` | Latin | 0.9% | 0.152 GB | 0.6% | 47,074 |
| Indonesian | `ind_Latn` | Latin | 0.9% | 0.236 GB | 0.9% | 45,970 |
| Polish | `pol_Latn` | Latin | 0.9% | 0.107 GB | 0.4% | 43,902 |
| Ukrainian | `ukr_Cyrl` | Cyrillic | 0.7% | 0.211 GB | 0.8% | 34,088 |
| Dutch | `nld_Latn` | Latin | 0.6% | 0.071 GB | 0.3% | 32,404 |
| Romanian | `ron_Latn` | Latin | 0.6% | 0.097 GB | 0.4% | 29,203 |
| Arabic | `arb_Arab` | Arabic | 0.5% | 0.168 GB | 0.7% | 27,292 |
| Hungarian | `hun_Latn` | Latin | 0.5% | 0.099 GB | 0.4% | 25,066 |
| Vietnamese | `vie_Latn` | Latin | 0.4% | 0.109 GB | 0.4% | 21,306 |
| Czech | `ces_Latn` | Latin | 0.4% | 0.077 GB | 0.3% | 21,696 |
| Greek | `ell_Grek` | Greek | 0.4% | 0.108 GB | 0.4% | 17,672 |
| Thai | `tha_Thai` | Thai | 0.2% | 0.087 GB | 0.3% | 11,639 |
| Finnish | `fin_Latn` | Latin | 0.2% | 0.034 GB | 0.1% | 11,721 |
| Slovak | `slk_Latn` | Latin | 0.2% | 0.028 GB | 0.1% | 10,015 |
| Bulgarian | `bul_Cyrl` | Cyrillic | 0.2% | 0.052 GB | 0.2% | 9,513 |
| Korean | `kor_Hang` | Hangul | 0.2% | 0.029 GB | 0.1% | 8,263 |
| Croatian | `hrv_Latn` | Latin | 0.2% | 0.026 GB | 0.1% | 8,395 |
| Catalan | `cat_Latn` | Latin | 0.1% | 0.014 GB | 0.1% | 5,087 |
| Hindi | `hin_Deva` | Devanagari | 0.1% | 0.032 GB | 0.1% | 4,928 |
| Hebrew | `heb_Hebr` | Hebrew | 0.1% | 0.025 GB | 0.1% | 4,665 |
| Bengali | `ben_Beng` | Bengali | 0.1% | 0.024 GB | 0.1% | 3,525 |
| Tamil | `tam_Taml` | Tamil | 0.0% | 0.017 GB | 0.1% | 1,831 |

8 scripts, 11+ language families. The data scarcity gradient spans Russian (18.1% of bytes, 531K docs) to Tamil (0.1% of bytes, 1.8K docs) — a ~180× range in byte budget and ~290× range in document count.

### Epoch count

The 24.97 GB mixture yields roughly 6–8B tokens depending on tokenizer. nanochat's BOS-bestfit dataloader discards ~34% of tokens as crop waste (fitting documents into fixed-length sequences). At the full-scale budget of ~9.2B tokens, this means ~1.5 epochs over the data — acceptable for a study focused on tokenizer comparison rather than data diversity.

---

## Training hyperparameters

### Budget

Training budget is `target_param_data_ratio × (transformer_matrices + lm_head)` with ratio 10.5 (Chinchilla-like). Only transformer matrices and lm_head count toward the budget — wte and VE scale with vocab and don't represent model capacity in the Chinchilla sense.

#### Pilot scale

| Vocab | Scaling params | Target tokens | Batch size | Steps | Actual tokens |
|-------|---------------|---------------|-----------|-------|--------------|
| 128,256 | 333,710,080 | 3,503,955,840 | 1,048,576 | 3,341 | 3,503,292,416 |
| 131,072 (Apertus) | 336,593,664 | 3,534,233,472 | 1,048,576 | 3,370 | 3,533,701,120 |

#### Full scale

| Vocab | Scaling params | Target tokens | Batch size | Steps | Actual tokens |
|-------|---------------|---------------|-----------|-------|--------------|
| 128,256 | 878,839,488 | 9,227,814,624 | 1,048,576 | 8,800 | 9,227,468,800 |
| 131,072 (Apertus) | 881,723,072 | 9,258,092,256 | 1,048,576 | 8,829 | 9,257,635,584 |

### Optimizer

nanochat's combined MuonAdamW: Muon for transformer weight matrices, AdamW for everything else. All base LR values are nanochat defaults, scaled by batch size and model width.

#### LR scaling

Two scaling factors are applied to the base LRs from the config:

- **Batch scale**: `(batch_size / B_REF)^0.5` where `B_REF = 524,288` (d12 reference). At `B = 1,048,576`: scale = `√2 ≈ 1.4142`. Applied to all parameter groups.
- **Width scale (µP)**: `(n_embd / 768)^{width_lr_exponent}` with `width_lr_exponent = -1.0`. Applied to AdamW groups only (embeddings, lm_head, scalars). Muon is width-invariant due to spectral normalization.

#### Weight decay scaling (T_epoch framework)

Following the T_epoch framework ([Lewkowycz et al., 2024](https://arxiv.org/abs/2405.13698)), weight decay is scaled to maintain constant `T_epoch = B/(η·λ·D)`:

```
λ = λ_ref × √(B/B_ref) × (D_ref/D)
```

Where `λ_ref = 0.28` (calibrated at d12), `D_ref` is the d12 compute-optimal token budget, and `D` is the current model's token budget. This ensures weight decay decreases for larger models trained on more tokens. Additionally, a cosine schedule decays WD from the scaled value to zero over training.

**Note on pilot runs**: The pilot runs used a buggy WD scaling formula where `D_ref` was incorrectly set to the current model's token budget (making the `D_ref/D` ratio ≈ 1.0). This resulted in WD = 0.396 instead of the correct 0.131. Since all 16 pilot tokenizers saw the same (incorrect) WD, relative comparisons remain valid, but absolute performance was suboptimal. The bug was fixed before full-scale runs.

#### Pilot scale effective hyperparameters (d16, n_embd=1024, B=1,048,576)

Width scale: `(1024/768)^{-1} = 0.750`

| Parameter group | Optimizer | Base LR | Effective LR | Betas | Weight decay |
|----------------|-----------|---------|-------------|-------|-------------|
| Transformer matrices | Muon | 0.02 | 0.02828 | momentum (see schedule) | 0.396 → 0 (cosine)* |
| Input embeddings (wte) | AdamW | 0.30 | 0.3182 | (0.8, 0.995) | 0.001 |
| Output projection (lm_head) | AdamW | 0.008 | 0.00849 | (0.8, 0.96) | 0.01 |
| Value embeddings | AdamW | 0.15 | 0.1591 | (0.8, 0.995) | 0.01 |
| Scalars (resid_lambdas) | AdamW | 0.005 | 0.00707 | (0.8, 0.95) | 0.05 |
| Scalars (x0_lambdas) | AdamW | 0.50 | 0.707 | (0.96, 0.95) | 0.0 |
| Scalars (smear, backout) | AdamW | 0.20 | 0.20 | (0.8, 0.95) | 0.0 |

*Pilot runs used WD=0.396 due to the bug described above. The correct value would have been 0.131.

#### Full scale effective hyperparameters (d24, n_embd=1536, B=1,048,576)

Width scale: `(1536/768)^{-1} = 0.500`

| Parameter group | Optimizer | Base LR | Effective LR | Betas | Weight decay |
|----------------|-----------|---------|-------------|-------|-------------|
| Transformer matrices | Muon | 0.02 | 0.02828 | momentum (see schedule) | 0.0496 → 0 (cosine) |
| Input embeddings (wte) | AdamW | 0.30 | 0.2121 | (0.8, 0.995) | 0.001 |
| Output projection (lm_head) | AdamW | 0.008 | 0.00566 | (0.8, 0.96) | 0.01 |
| Value embeddings | AdamW | 0.15 | 0.1061 | (0.8, 0.995) | 0.01 |
| Scalars (resid_lambdas) | AdamW | 0.005 | 0.00707 | (0.8, 0.95) | 0.05 |
| Scalars (x0_lambdas) | AdamW | 0.50 | 0.707 | (0.96, 0.95) | 0.0 |
| Scalars (smear, backout) | AdamW | 0.20 | 0.20 | (0.8, 0.95) | 0.0 |

Effective LR formula: `base_lr × batch_scale × width_scale` (AdamW groups) or `base_lr × batch_scale` (Muon).

### LR schedule

Linear warmup → constant plateau → linear warmdown.

LR warmup and Muon momentum warmup run on different schedules: LR warms over `warmup_steps` (40 steps), while Muon momentum warms over a hardcoded 400 steps (0.85 → 0.97).

#### Pilot scale

| Phase | Steps | LR multiplier | Muon momentum |
|-------|-------|--------------|---------------|
| LR warmup | 0 → 40 | 0 → 1.0 (linear) | |
| Momentum warmup | 0 → 400 | | 0.85 → 0.97 (linear) |
| Constant | 400 → 1,169 | 1.0 | 0.97 |
| Warmdown | 1,169 → 3,341 | 1.0 → 0.05 (linear) | 0.97 → 0.90 |

#### Full scale

| Phase | Steps | LR multiplier | Muon momentum |
|-------|-------|--------------|---------------|
| LR warmup | 0 → 40 | 0 → 1.0 (linear) | |
| Momentum warmup | 0 → 400 | | 0.85 → 0.97 (linear) |
| Constant | 400 → 3,080 | 1.0 | 0.97 |
| Warmdown | 3,080 → 8,800 | 1.0 → 0.05 (linear) | 0.97 → 0.90 |

Warmdown ratio = 0.65 (65% of training is in warmdown). Final LR = 5% of peak.

### Validation

| Parameter | Pilot | Full |
|-----------|-------|------|
| Eval frequency | Every 250 steps | Every 500 steps |
| Save frequency | Every end-of-training | Every 500 steps (keep last 3) |
| Eval tokens | 41,943,040 (80 × 524,288) | 41,943,040 |
| Metric | BPB (bits-per-byte) on validation shard | BPB |

### FLORES-200 evaluation

Runs automatically at end of training (rank 0 only). Computes per-language BPB on FLORES-200 devtest split (200 samples per language). Logs:
- Mean, std, min, max, CV of BPB across languages
- Per-language-family mean BPB
- Per-language BPB
- All metrics to W&B and `flores_results.json` in checkpoint directory

### Post-training evaluation (full scale only)

Run via `eval_runner.py` after training completes. Tasks:

| Benchmark | Type | Metric | Notes |
|-----------|------|--------|-------|
| GSM8K | Math (8-shot CoT) | exact_match | Generation-based |
| MGSM | Multilingual math (11 languages) | exact_match | Uses `jbross-ibm-research/mgsm` (parquet) |
| HumanEval | Code generation | pass@1 | Greedy, 0-shot |
| MBPP | Code generation | pass@1 | Greedy |
| BLiMP | Linguistic acceptability | accuracy | 66 subtasks, loglikelihood |
| FLORES-200 | Multilingual perplexity | BPB | 254 languages |

Per-sample logs are saved (`log_samples=True`) for downstream analysis (e.g., per-problem digit alignment for the right-aligned tokenizer study).

---

## Infrastructure

### Pilot scale

| Parameter | Value |
|-----------|-------|
| Hardware | CSCS Clariden, GH200 120GB nodes |
| GPUs per run | 4 (single node) |
| SLURM account | a139, partition `normal` |
| Wall time | 4 hours per run |
| Python | 3.11.5 (venv at `~/tokenizer_lm_exps_env`) |
| uenv | `pytorch/v2.8.0:v1` (for Slingshot/NCCL network stack) |
| NCCL backend | AWS Libfabric (Slingshot) |
| torch.compile | Enabled (dynamic=False) |
| Precision | bf16 throughout |
| DDP | No — Muon handles gradient communication via DistMuonAdamW |
| Expected throughput | ~700K tok/s, ~30% MFU |
| Expected wall time | ~2h training + ~15min FLORES eval per run |
| Random seed | 42 |

### Full scale

| Parameter | Value |
|-----------|-------|
| Hardware | CSCS Clariden, GH200 120GB nodes |
| Nodes per run | 4 (16 GPUs) |
| SLURM account | a139, partition `normal` |
| Wall time | 12 hours per run |
| Python | 3.11.5 (venv at `~/tokenizer_lm_exps_env`) |
| uenv | `pytorch/v2.8.0:v1` |
| NCCL backend | AWS Libfabric (Slingshot) |
| torch.compile | Enabled (dynamic=False) |
| Precision | bf16 throughout |
| DDP | No — DistMuonAdamW |
| Expected throughput | ~600–700K tok/s (estimated, 4-node scaling) |
| Expected wall time | ~4–5h training + ~30min FLORES eval per run |
| Random seed | 42 |

### SLURM invocation

```bash
# Pilot
sbatch scripts/slurm_pilot.sh <config> <tokenizer_path> <data_dir> <run_name>

# Full scale
sbatch scripts/slurm_full.sh <config> <tokenizer_path> <data_dir> <run_name>
```

The SLURM scripts use the absolute path to the venv Python (`${HOME}/tokenizer_lm_exps_env/bin/python`) via `srun` with `env PYTHONPATH=...` to bypass the uenv's Python 3.12 while retaining the uenv's NCCL/Slingshot libraries.

### Checkpointing

Checkpoints saved to `/capstor/scratch/cscs/$USER/tokenizer-lm/checkpoints/<run_name>/` during training. Completed checkpoints have been archived to `/capstor/store/cscs/swissai/a139/cmeister/tokenizer-lm/checkpoints/`. The checkpoint directory uses the `--run-name` (unique per experiment), not the config `name` field.

Files per checkpoint:
- `model_<step>.pt` — model state dict
- `meta_<step>.json` — metadata (tokenizer path, config, training state)
- `optim_<step>_rank<N>.pt` — optimizer state per rank
- `flores_results.json` — FLORES-200 per-language BPB (written at end of training)

### W&B logging

Project: `tokenizer-lm-experiments`, entity: `cmeister7-47`.

Metrics logged:
- `val/bpb` — every 250 (pilot) or 500 (full) steps
- `train/loss`, `train/mfu`, `train/tok_per_sec`, `train/bytes_consumed` — every 100 steps
- `flores/mean_bpb`, `flores/std_bpb`, `flores/cv_bpb` — at end of training
- `flores/family/<name>` — per-language-family BPB
- `flores/lang/<code>` — per-language BPB

---

## Run matrix

### Pilot scale (completed)

| # | Run name | Tokenizer | Vocab | Source |
|---|----------|-----------|-------|--------|
| 1 | `pilot-128k-apertus` | swiss-ai/Apertus-70B-2509 | 131,072 | Off-the-shelf |
| 2 | `pilot-128k-llama3` | NousResearch/Meta-Llama-3-8B | 128,256 | Off-the-shelf |
| 3 | `pilot-128k-punct-balanced-bpe` | custom | 128,256 | Punct pretok, balanced |
| 4 | `pilot-128k-punct-english-bpe` | custom | 128,256 | Punct pretok, English |
| 5 | `pilot-128k-gpt4o-balanced-bpe` | custom | 128,256 | GPT-4o pretok, balanced |
| 6 | `pilot-128k-gpt4o-balanced-nfc-bpe` | custom | 128,256 | GPT-4o + NFC, balanced |
| 7 | `pilot-128k-gpt4o-english-bpe` | custom | 128,256 | GPT-4o pretok, English |
| 8 | `pilot-128k-gpt4o-code-bpe` | custom | 128,256 | GPT-4o pretok, code |
| 9 | `pilot-128k-claude-balanced-bpe` | custom | 128,256 | Claude pretok, balanced |
| 10 | `pilot-128k-claude-balanced-nfc-bpe` | custom | 128,256 | Claude + NFC, balanced |
| 11 | `pilot-128k-claude-english-bpe` | custom | 128,256 | Claude pretok, English |
| 12 | `pilot-128k-rightalign-balanced-bpe` | custom | 128,256 | Right-aligned digits, balanced |
| 13 | `pilot-128k-rightalign-balanced-nfc-bpe` | custom | 128,256 | Right-aligned + NFC, balanced |
| 14 | `pilot-128k-gpt4o-balanced-unigram` | custom | 128,256 | GPT-4o UnigramLM, balanced |
| 15 | `pilot-128k-claude-balanced-unigram` | custom | 128,256 | Claude UnigramLM, balanced |
| 16 | `pilot-128k-rightalign-balanced-unigram` | custom | 128,256 | Right-aligned UnigramLM, balanced |

### Full scale

Same 16 tokenizers, with run names prefixed `full-128k-` instead of `pilot-128k-`.

### Controlled comparisons

| Comparison | Runs | What varies |
|------------|------|-------------|
| Pretokenizer effect | 3, 5, 9, 12 | Pretok strategy (balanced BPE, no NFC) |
| Training data effect | 5, 7, 8 | Data composition (GPT-4o BPE) |
| NFC normalization | 5↔6, 9↔10, 12↔13 | NFC on/off (same pretok + data) |
| BPE vs UnigramLM | 5↔14, 9↔15, 12↔16 | Algorithm (same pretok + data) |
| Right vs left digit align | 5↔12 | Digit grouping strategy |
| Claude vs GPT-4o pretok | 5↔9 | Case handling + contractions |
| Off-the-shelf vs custom | 1,2 vs 5,9,12 | Tokenizer provenance |

---

## Math+code experiments (sanity checks)

Additional experiments to test whether continued training or domain-specific finetuning affects cross-tokenizer comparisons.

### Math+code training data

A 50/50 (by bytes) mixture of high-quality math and code data. Created with `scripts/create_mathcode_mixture.sh`.

**Sources:**

| Domain | Dataset | Text bytes | Docs | Text column |
|--------|---------|-----------|------|-------------|
| Math | megamath-web-pro (LLM360 MegaMath, robots.txt filtered) | ~50 GB | ~14M | `text` |
| Code | stackv2-edu (7 languages, educational code) | ~50 GB | ~13M | `code` |

**Code language breakdown** (proportional to original dataset sizes):

| Language | Target GB | Target docs |
|----------|----------|-------------|
| JavaScript | 12.7 | 3.4M |
| Python | 11.0 | 2.3M |
| Java | 11.0 | 3.4M |
| C++ | 7.1 | 1.4M |
| TypeScript | 3.5 | 1.3M |
| Go | 3.4 | 0.8M |
| Rust | 1.3 | 0.2M |

**Source paths:**
- megamath-web-pro: `/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/megamath-filterrobots/data/output/megamath-web-pro`
- stackv2-edu: `/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/code/stackv2-edu/{python,javascript,java,c++,typescript,go,rust}`

**Output:** `/capstor/scratch/cscs/cmeister747/data/tokenizer-lm-mathcode/` (~27M docs, ~100 GB text)

Languages excluded from stackv2-edu: C, C#, PHP, Ruby, Shell, SQL, Swift, Markdown (less relevant to downstream code benchmarks).

The data uses weighted random document sampling (same mechanism as `prepare.py mix`), so any prefix of the output maintains the 50/50 ratio approximately. This allows the same dataset to be used for both the from-scratch model (20B tokens) and finetuning (5B tokens).

### Experiment matrix

#### 1. From-scratch math+code (config: `full_128k_mathcode_scratch.yaml`)

| Run name | Tokenizer | Data | Tokens | Steps |
|----------|-----------|------|--------|-------|
| `full-128k-mathcode-scratch` | gpt4o-balanced-bpe | math+code | 20B | ~19,073 |

Same d24 architecture and hyperparameters as main experiments. `target_param_data_ratio` disabled; fixed 20B token budget.

#### 2. Finetune on math+code (config: `full_128k_mathcode_finetune.yaml`)

Model weights initialized from the final checkpoint of each model's main training run (via `--init-from`). Optimizer and LR schedule start fresh. 5B tokens of math+code data.

| Run name | Init from | Data | Tokens | Steps |
|----------|-----------|------|--------|-------|
| `full-128k-gpt4o-balanced-bpe-mathcode-ft` | full-128k-gpt4o-balanced-bpe | math+code | 5B | ~4,768 |
| `full-128k-rightalign-balanced-bpe-mathcode-ft` | full-128k-rightalign-balanced-bpe | math+code | 5B | ~4,768 |
| `full-128k-claude-balanced-nfc-bpe-mathcode-ft` | full-128k-claude-balanced-nfc-bpe | math+code | 5B | ~4,768 |
| `full-128k-llama3-mathcode-ft` | full-128k-llama3 | math+code | 5B | ~4,768 |
| `full-128k-gpt4o-code-bpe-mathcode-ft` | full-128k-gpt4o-code-bpe | math+code | 5B | ~4,768 |

#### 3. Continue training on standard mix (config: `full_128k_continue.yaml`)

Same as finetune setup, but using the original training data mixture for 10B additional tokens.

| Run name | Init from | Data | Tokens | Steps |
|----------|-----------|------|--------|-------|
| `full-128k-gpt4o-balanced-bpe-continue` | full-128k-gpt4o-balanced-bpe | standard mix | 10B | ~9,536 |
| `full-128k-rightalign-balanced-bpe-continue` | full-128k-rightalign-balanced-bpe | standard mix | 10B | ~9,536 |
| `full-128k-claude-balanced-nfc-bpe-continue` | full-128k-claude-balanced-nfc-bpe | standard mix | 10B | ~9,536 |
| `full-128k-llama3-continue` | full-128k-llama3 | standard mix | 10B | ~9,536 |
| `full-128k-gpt4o-code-bpe-continue` | full-128k-gpt4o-code-bpe | standard mix | 10B | ~9,536 |

### Infrastructure

All experiments use single-node 4×GH200 (same as main full-scale training). Estimated wall times:
- From-scratch (20B): ~10h (1 SLURM job)
- Finetune (5B): ~2.5h each
- Continue (10B): ~5h each

---

## Total compute

### Pilot scale (completed)

| Item | Hours |
|------|-------|
| 16 runs × ~2.25h × 4 GPUs | ~144 GPU-hours |
| FLORES eval (included in above) | ~4 GPU-hours |
| **Total** | **~148 GH200-hours** |

### Full scale (estimated)

| Item | Hours |
|------|-------|
| 16 runs × ~4.5h × 16 GPUs | ~1,152 GPU-hours |
| FLORES eval (included in above) | ~8 GPU-hours |
| Post-training eval (GSM8K, HumanEval, etc.) | ~32 GPU-hours |
| **Total** | **~1,192 GH200-hours** |

With 64 nodes available in parallel (16 per run × 4 concurrent runs): ~18 hours wall time.
