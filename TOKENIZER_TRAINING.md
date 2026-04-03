# Tokenizer Training Settings

## Global parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Vocab size** | 128,256 | Matches LLaMA-3's vocab size for direct comparison |
| **Training data** | 10GB (10B bytes) per composition | ~2M documents, sampled proportionally from source parquets |
| **Special tokens** | `<s>`, `</s>`, `<unk>`, `<pad>` | `<s>` serves as BOS token (id=0) for nanochat's dataloader |
| **Random seed** | 42 | Same seed for all tokenizers using the same composition — ensures identical training data |
| **Max chars per doc** | 90,000 | Documents >90K chars split at whitespace boundaries (UnigramLM has 100K char limit) |
| **Data reading** | Direct from parquet via `train_from_iterator()` | No intermediate text files; streams from source |
| **Min doc length** | 20 chars | Documents shorter than 20 chars are filtered out |

## BPE-specific settings

| Parameter | Value |
|-----------|-------|
| **Algorithm** | HuggingFace `BpeTrainer` |
| **min_frequency** | 2 |
| **max_token_length** | 64 |

## UnigramLM-specific settings

| Parameter | Value |
|-----------|-------|
| **Algorithm** | HuggingFace `UnigramTrainer` |
| **unk_token** | `<unk>` |

## Pretokenization strategies

All use **ByteLevel encoding** as the final step (converts raw bytes to single-byte tokens before BPE/Unigram merging). The strategies differ in what happens *before* ByteLevel:

### P1 — Punctuation + ByteLevel (`punct-*`)
```
Sequence([Punctuation(behavior="Isolated"), ByteLevel(use_regex=True)])
```
Minimal structure: isolates punctuation, ByteLevel handles word splitting via its built-in regex.

### P2 — GPT-4o regex + ByteLevel (`gpt4o-*`)
```
Sequence([Split(regex=REGEX_GPT4O, behavior="Isolated"), ByteLevel(use_regex=False)])
```
Production GPT-4o (o200k_base) regex:
- CamelCase splitting at lowercase→uppercase transitions
- `\p{N}{1,3}` digit groups (left-aligned, max 3 digits)
- English contraction suffixes (`'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`)
- Dot attaches forward to following letter
- Whitespace handling with indent-aware backtracking

### P3 — Claude regex + ByteLevel (`claude-*`)
```
Sequence([Split(regex=REGEX_CLAUDE, behavior="Isolated"), ByteLevel(use_regex=False)])
```
Claude-style pretokenizer:
- Case-level splits: `\p{Lu}+` and `\p{Ll}+` as separate runs (splits within words at case boundaries: "camelCase" → ["camel", "C", "ase"])
- `\p{N}{1,3}` digit groups
- No contraction handling (no `'s`, `'t`, etc.)
- CJK/emoji grouped with punctuation/symbols in `[^\s\p{L}\p{N}]+`
- Whitespace split by type (space, tab, newline, carriage return as separate runs)

### P4 — Right-aligned digits + ByteLevel (`rightalign-*`)
```
Sequence([Split(regex=REGEX_GPT4O_RIGHTALIGN, behavior="Isolated"), ByteLevel(use_regex=False)])
```
Same as GPT-4o but with place-value-aligned digit grouping:
- `\p{N}{1,3}(?=(?:\p{N}{3})*(?:\P{N}|$))` instead of `\p{N}{1,3}`
- "123456" → ["123", "456"] (right-aligned) instead of ["123", "456"] or ["12", "345", "6"]
- Aligns digit groups to ones/thousands/millions place values

## Normalizer

| Setting | Effect |
|---------|--------|
| **None** (default) | No normalization — raw Unicode as-is |
| **NFC** | Unicode NFC normalization — composes decomposed characters (e.g., `é` as single codepoint instead of `e` + combining acute). May help low-resource languages with inconsistent Unicode representations |

## Training data compositions

All read directly from source parquet files on Clariden. Sampling is **weighted random**: each document draw picks a source with probability proportional to its weight, then reads the next document from that source's parquet iterator.

Data config files for each composition are at [`configs/data/`](configs/data/) in JSON format compatible with `train_tokenizer.py`: [`english.json`](configs/data/english.json), [`balanced.json`](configs/data/balanced.json), [`code.json`](configs/data/code.json).

**Note**: The tokenizer training data compositions (D1/D2/D3) are distinct from the LM training mixture. The LM mixture uses fixed proportions across all models (see [MODEL_TRAINING.md](MODEL_TRAINING.md#lm-training-mixture)), while the tokenizer compositions vary by design to study the effect of training data on tokenizer quality.

### Source paths on Clariden

| Short name | Full path |
|-----------|-----------|
| `fineweb_edu` | `/capstor/store/cscs/swissai/infra01/datasets/HuggingFaceFW/fineweb-edu/data` |
| `fineweb2/{lang}` | `/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/fineweb-2_0_1-quality_33-filterrobots/data/output/{lang}` |
| `finemath` | `/capstor/store/cscs/swissai/infra01/datasets/HuggingFaceTB/finemath/finemath-4plus` |
| `starcoderdata/{lang}` | `/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/starcoderdata/thresholds/{lang}/threshold_0` |

### D1 — English only (10GB)

Used by: `gpt4o-english-bpe`, `punct-english-bpe`, `claude-english-bpe`

Data config: [`configs/data/english.json`](configs/data/english.json)

| Source | Path | GB | Text column |
|--------|------|----|-------------|
| FineWeb-Edu | `fineweb_edu` | 10.0 | `text` |

### D2 — Balanced multilingual (10GB)

Used by: 8 BPE + 3 UnigramLM = 11 tokenizers.

Data config: [`configs/data/balanced.json`](configs/data/balanced.json)

35% English, 30% multilingual (30 languages, character-proportional), 15% math, 15% code.

Multilingual weights are proportional to estimated character counts in the source data. Character counts were approximated by sampling average characters per document from the first parquet file of each language, then multiplying by total rows × number of files.

| Source | Text GB | Text column |
|--------|---------|-------------|
| FineWeb-Edu (English) | 3.500 | `text` |
| Russian (`rus_Cyrl`) | 1.010 | `text` |
| Spanish (`spa_Latn`) | 0.232 | `text` |
| German (`deu_Latn`) | 0.226 | `text` |
| French (`fra_Latn`) | 0.196 | `text` |
| Chinese (`cmn_Hani`) | 0.182 | `text` |
| Japanese (`jpn_Jpan`) | 0.120 | `text` |
| Italian (`ita_Latn`) | 0.114 | `text` |
| Portuguese (`por_Latn`) | 0.113 | `text` |
| Turkish (`tur_Latn`) | 0.089 | `text` |
| Indonesian (`ind_Latn`) | 0.087 | `text` |
| Polish (`pol_Latn`) | 0.084 | `text` |
| Ukrainian (`ukr_Cyrl`) | 0.064 | `text` |
| Dutch (`nld_Latn`) | 0.062 | `text` |
| Romanian (`ron_Latn`) | 0.055 | `text` |
| Arabic (`arb_Arab`) | 0.051 | `text` |
| Hungarian (`hun_Latn`) | 0.048 | `text` |
| Vietnamese (`vie_Latn`) | 0.041 | `text` |
| Czech (`ces_Latn`) | 0.041 | `text` |
| Greek (`ell_Grek`) | 0.034 | `text` |
| Finnish (`fin_Latn`) | 0.022 | `text` |
| Thai (`tha_Thai`) | 0.022 | `text` |
| Slovak (`slk_Latn`) | 0.019 | `text` |
| Bulgarian (`bul_Cyrl`) | 0.018 | `text` |
| Croatian (`hrv_Latn`) | 0.016 | `text` |
| Korean (`kor_Hang`) | 0.016 | `text` |
| Catalan (`cat_Latn`) | 0.010 | `text` |
| Hindi (`hin_Deva`) | 0.009 | `text` |
| Hebrew (`heb_Hebr`) | 0.009 | `text` |
| Bengali (`ben_Beng`) | 0.007 | `text` |
| Tamil (`tam_Taml`) | 0.004 | `text` |
| FineMath-4plus | 1.500 | `text` |
| StarCoder Python (threshold_0) | 1.000 | `content` |
| StarCoder JavaScript (threshold_0) | 0.500 | `content` |

### D3 — Code-heavy (10GB)

Used by: `gpt4o-code-bpe`

Data config: [`configs/data/code.json`](configs/data/code.json)

| Source | Text GB | Text column |
|--------|---------|-------------|
| FineWeb-Edu | 5.0 | `text` |
| StarCoder Python | 2.0 | `content` |
| StarCoder JavaScript | 1.0 | `content` |
| StarCoder Java | 1.0 | `content` |
| StarCoder C++ | 1.0 | `content` |

## Full experiment matrix (14 custom tokenizers)

| Name | Pretok | Algorithm | Data | NFC | Purpose |
|------|--------|-----------|------|-----|---------|
| `punct-balanced-bpe` | P1 | BPE | balanced | No | Minimal pretok baseline |
| `punct-english-bpe` | P1 | BPE | english | No | English-only baseline |
| `gpt4o-balanced-bpe` | P2 | BPE | balanced | No | Production pretok, multilingual data |
| `gpt4o-balanced-nfc-bpe` | P2 | BPE | balanced | Yes | NFC normalization effect |
| `gpt4o-english-bpe` | P2 | BPE | english | No | Data composition effect |
| `gpt4o-code-bpe` | P2 | BPE | code | No | Code-optimized data |
| `claude-balanced-bpe` | P3 | BPE | balanced | No | Claude pretok effect |
| `claude-balanced-nfc-bpe` | P3 | BPE | balanced | Yes | Claude + NFC |
| `claude-english-bpe` | P3 | BPE | english | No | Claude + English-only |
| `rightalign-balanced-bpe` | P4 | BPE | balanced | No | Place-value digit alignment |
| `rightalign-balanced-nfc-bpe` | P4 | BPE | balanced | Yes | Right-align + NFC |
| `gpt4o-balanced-unigram` | P2 | Unigram | balanced | No | Algorithm comparison |
| `claude-balanced-unigram` | P3 | Unigram | balanced | No | Algorithm comparison |
| `rightalign-balanced-unigram` | P4 | Unigram | balanced | No | Algorithm comparison |

## Controlled comparisons

- **Pretok effect** (balanced BPE, no NFC): P1 vs P2 vs P3 vs P4
- **Training data effect** (GPT-4o BPE): english vs balanced vs code
- **NFC effect** (balanced BPE): gpt4o ± NFC, claude ± NFC, rightalign ± NFC
- **Algorithm effect** (balanced, no NFC): BPE vs UnigramLM for P2, P3, P4
- **Right-aligned vs left-aligned digits**: gpt4o-balanced vs rightalign-balanced
- **Claude vs GPT-4o pretok**: claude-balanced vs gpt4o-balanced (same data, different case handling + contractions)

## Off-the-shelf tokenizers (included in experiments)

| Tokenizer | Vocab | Source |
|-----------|-------|--------|
| Apertus | 131,072 | `swiss-ai/Apertus-70B-2509` |
| LLaMA-3 | 128,256 | `NousResearch/Meta-Llama-3-8B` |

## SLURM job settings

| Parameter | Value |
|-----------|-------|
| CPUs | 32 |
| Memory | 200 GB |
| GPUs | 0 (CPU only) |
| Wall time | 12 hours |
| Output dir | `/capstor/store/cscs/swissai/a139/cmeister/tokenizer-lm/tokenizers/` |

### Trained tokenizer locations

- **Custom tokenizers (14)**: `/capstor/store/cscs/swissai/a139/cmeister/tokenizer-lm/tokenizers/<tokenizer_name>/`
- **SuperBPE** (with special tokens added): `~/superbpe/gpt4o_regex_full_data_with_special/`
- **PA-BPE** (with special tokens added): `~/pa_tokenizers_branch/pabpe-128k-nfc-gpt4-reg_with_special/`
- **Off-the-shelf**: loaded directly from HuggingFace (`swiss-ai/Apertus-70B-2509`, `NousResearch/Meta-Llama-3-8B`)
