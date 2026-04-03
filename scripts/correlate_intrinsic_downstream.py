#!/usr/bin/env python3
"""Correlate intrinsic tokenizer metrics with downstream LM performance.

Reads intrinsic eval results and downstream results (FLORES BPB, BLiMP
accuracy) and computes correlations treating each (tokenizer, language)
pair as an individual data point.

Usage:
    python scripts/correlate_intrinsic_downstream.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import false_discovery_control

# ─── Paths ───────────────────────────────────────────────────────────

LM_ROOT = Path(__file__).resolve().parent.parent
INTRINSIC_EVALS_ROOT = Path.home() / "tokenizer-intrinsic-evals"

INTRINSIC_RESULTS_DEFAULT = INTRINSIC_EVALS_ROOT / "results" / "downstream_lm_flores" / "analysis_results.json"
FLORES_DIR = LM_ROOT / "results" / "flores"
BLIMP_DIR = LM_ROOT / "results" / "lm_eval"
LANG_CONFIG = INTRINSIC_EVALS_ROOT / "configs" / "downstream_lm_lang_config.json"
OUTPUT_DIR = LM_ROOT / "results" / "correlations"

# ─── Constants ───────────────────────────────────────────────────────

# Intrinsic display name -> downstream run slug (without "pilot-128k-" prefix)
NAME_MAP = {
    "Apertus": "apertus",
    "LLaMA-3": "llama3",
    "BPE Punct (balanced)": "punct-balanced-bpe",
    "BPE Punct (english)": "punct-english-bpe",
    "BPE GPT-4o (balanced)": "gpt4o-balanced-bpe",
    "BPE GPT-4o NFC (balanced)": "gpt4o-balanced-nfc-bpe",
    "BPE GPT-4o (english)": "gpt4o-english-bpe",
    "BPE GPT-4o (code)": "gpt4o-code-bpe",
    "BPE Claude (balanced)": "claude-balanced-bpe",
    "BPE Claude NFC (balanced)": "claude-balanced-nfc-bpe",
    "BPE Claude (english)": "claude-english-bpe",
    "BPE RightAlign (balanced)": "rightalign-balanced-bpe",
    "BPE RightAlign NFC (balanced)": "rightalign-balanced-nfc-bpe",
    "Unigram GPT-4o (balanced)": "gpt4o-balanced-unigram",
    "Unigram Claude (balanced)": "claude-balanced-unigram",
    "Unigram RightAlign (balanced)": "rightalign-balanced-unigram",
    "SuperBPE (balanced)": "superbpe",
    "PA-BPE NFC (balanced)": "pabpe",
}

LANG_CODE_MAP = {"cmn_Hani": "cmn_Hans"}

TRAINING_DATA_WEIGHTS = {
    "eng_Latn": 0.369, "rus_Cyrl": 0.181, "spa_Latn": 0.017, "deu_Latn": 0.014,
    "fra_Latn": 0.015, "cmn_Hani": 0.013, "jpn_Jpan": 0.008, "ita_Latn": 0.007,
    "por_Latn": 0.008, "tur_Latn": 0.006, "ind_Latn": 0.009, "pol_Latn": 0.004,
    "ukr_Cyrl": 0.008, "nld_Latn": 0.003, "ron_Latn": 0.004, "arb_Arab": 0.007,
    "hun_Latn": 0.004, "vie_Latn": 0.004, "ces_Latn": 0.003, "ell_Grek": 0.004,
    "tha_Thai": 0.003, "fin_Latn": 0.001, "slk_Latn": 0.001, "bul_Cyrl": 0.002,
    "kor_Hang": 0.001, "hrv_Latn": 0.001, "cat_Latn": 0.001, "hin_Deva": 0.001,
    "heb_Hebr": 0.001, "ben_Beng": 0.001, "tam_Taml": 0.001,
}

# Per-language intrinsic metrics: (json_key, per_lang_subkey, global_subkey)
# per_lang_subkey=None means the per-language value is a scalar
INTRINSIC_METRICS = {
    "compression_rate":     ("compression_rate",       None,             "compression_rate"),
    "fertility":            ("fertility",              "mean",           "mean"),
    "unigram_entropy":      ("unigram_distribution_metrics", "unigram_entropy", "unigram_entropy"),
    "bigram_entropy":       ("bigram_entropy",         "bigram_entropy", "bigram_entropy"),
    "trigram_entropy":      ("trigram_entropy",         "trigram_entropy", None),  # global uses custom key
    "vocab_utilization":    ("vocabulary_utilization",  "utilization",    "utilization"),
    "utf8_char_split":      ("utf8_char_split",         "split_rate",    "split_rate"),
}
# Renyi has a unique structure: pt["renyi_X.X"]["overall"] for global,
# pt["renyi_X.X"][lang] for per-language. Handled separately.
RENYI_KEY = "renyi_efficiency"
RENYI_ALPHAS = {"renyi_2.0": "renyi_2.0"}  # alpha=2.0 emphasizes frequent tokens

# Global-only metrics (per-language values are not the same quantity, or not per-language)
GLOBAL_ONLY_METRICS = {
    "gini_coefficient":     ("tokenizer_fairness_gini",      "gini_coefficient"),
    "ast_full_alignment":   ("ast_boundary_alignment",       "full_alignment_rate"),
    "ident_fragmentation":  ("identifier_fragmentation",     "fragmentation_rate"),
}

# Math-specific metrics (from built-in math data, 1 "language" only — global-only for correlation)
MATH_INTRINSIC_METRICS = {
    "digit_boundary_f1":    "three_digit_boundary_alignment",
    "operator_isolation":   "operator_isolation_rate",
}

CONTROLLED_COMPARISONS = {
    "NFC: GPT-4o": ("BPE GPT-4o (balanced)", "BPE GPT-4o NFC (balanced)"),
    "NFC: Claude": ("BPE Claude (balanced)", "BPE Claude NFC (balanced)"),
    "NFC: RightAlign": ("BPE RightAlign (balanced)", "BPE RightAlign NFC (balanced)"),
    "Algo: GPT-4o BPE vs Unigram": ("BPE GPT-4o (balanced)", "Unigram GPT-4o (balanced)"),
    "Algo: Claude BPE vs Unigram": ("BPE Claude (balanced)", "Unigram Claude (balanced)"),
    "Algo: RightAlign BPE vs Unigram": ("BPE RightAlign (balanced)", "Unigram RightAlign (balanced)"),
    "Pretok: GPT-4o vs Claude": ("BPE GPT-4o (balanced)", "BPE Claude (balanced)"),
    "Pretok: GPT-4o vs Punct": ("BPE GPT-4o (balanced)", "BPE Punct (balanced)"),
    "Pretok: GPT-4o vs RightAlign": ("BPE GPT-4o (balanced)", "BPE RightAlign (balanced)"),
    "Data: GPT-4o balanced vs english": ("BPE GPT-4o (balanced)", "BPE GPT-4o (english)"),
    "Data: GPT-4o balanced vs code": ("BPE GPT-4o (balanced)", "BPE GPT-4o (code)"),
}


# ─── Data Loading ────────────────────────────────────────────────────

def _extract_value(obj, subkey):
    if subkey is None:
        return float(obj) if not isinstance(obj, dict) else None
    if isinstance(obj, dict):
        return float(obj[subkey])
    return None


def load_intrinsic(path=INTRINSIC_RESULTS_DEFAULT):
    """Load intrinsic metrics. Returns (global_df, per_lang_dict).

    global_df: tokenizer × metric (includes global-only metrics like gini)
    per_lang_dict: metric_name → DataFrame of tokenizer × language
    """
    with open(path) as f:
        raw = json.load(f)

    tokenizers = sorted(NAME_MAP.keys())

    def _get_global_value(pt, global_subkey, metric_name):
        """Extract global value from a per-tokenizer entry, handling layout variants."""
        # Standard layout: pt["global"][subkey]
        if "global" in pt and isinstance(pt["global"], dict) and global_subkey in pt["global"]:
            return float(pt["global"][global_subkey])
        # Flat layout: pt[subkey] directly (e.g., renyi_efficiency)
        if global_subkey is not None and global_subkey in pt and not isinstance(pt[global_subkey], dict):
            return float(pt[global_subkey])
        # Non-standard key: pt["global_<metric_name>"] (e.g., trigram_entropy)
        if f"global_{metric_name}" in pt:
            val = pt[f"global_{metric_name}"]
            return float(val) if val is not None else None
        return None

    global_data = {}
    for metric_name, (json_key, _, global_subkey) in INTRINSIC_METRICS.items():
        if json_key not in raw:
            continue
        vals = {}
        for tok in tokenizers:
            pt = raw[json_key]["per_tokenizer"][tok]
            vals[tok] = _get_global_value(pt, global_subkey, metric_name)
        global_data[metric_name] = vals

    for metric_name, (json_key, global_subkey) in GLOBAL_ONLY_METRICS.items():
        if json_key not in raw:
            continue
        vals = {}
        for tok in tokenizers:
            pt = raw[json_key]["per_tokenizer"][tok]
            vals[tok] = _get_global_value(pt, global_subkey, metric_name)
        global_data[metric_name] = vals

    # Renyi efficiency: two possible structures:
    # (a) New: pt["global"]["renyi_2.0"], pt["per_language"][lang]["renyi_2.0"]
    # (b) Old: pt["renyi_2.0"]["overall"], pt["renyi_2.0"][lang]
    per_lang_renyi = {}
    if RENYI_KEY in raw:
        for metric_name, alpha_key in RENYI_ALPHAS.items():
            vals = {}
            pl_renyi = {}
            for tok in tokenizers:
                pt = raw[RENYI_KEY]["per_tokenizer"][tok]
                # Try new structure first: global dict with alpha keys
                g = pt.get("global", {})
                if isinstance(g, dict) and alpha_key in g:
                    vals[tok] = g[alpha_key]
                    pl = pt.get("per_language", {})
                    pl_renyi[tok] = {lang: v.get(alpha_key) for lang, v in pl.items()
                                     if isinstance(v, dict) and alpha_key in v}
                else:
                    # Old structure: flat pt["renyi_2.0"]["overall"]
                    renyi_data = pt.get(alpha_key, {})
                    if isinstance(renyi_data, dict):
                        vals[tok] = renyi_data.get("overall")
                        pl_renyi[tok] = {k: v for k, v in renyi_data.items() if k != "overall"}
                    else:
                        vals[tok] = float(renyi_data) if renyi_data is not None else None
                        pl_renyi[tok] = {}
            global_data[metric_name] = vals
            per_lang_renyi[metric_name] = pl_renyi

    # Math-specific metrics: extract from per_language (single entry) or by_digit_length
    for metric_name, json_key in MATH_INTRINSIC_METRICS.items():
        if json_key not in raw:
            continue
        vals = {}
        for tok in tokenizers:
            pt = raw[json_key]["per_tokenizer"][tok]
            pl = pt.get("per_language", {})
            if pl:
                # Math metrics have 1 "language" entry from built-in data
                first_lang = list(pl.values())[0]
                if isinstance(first_lang, dict):
                    if "mean_f1" in first_lang:
                        vals[tok] = first_lang["mean_f1"]
                    elif "f1" in first_lang:
                        vals[tok] = first_lang["f1"]
                    elif "isolation_rate" in first_lang:
                        vals[tok] = first_lang["isolation_rate"]
                    elif "overall" in first_lang and isinstance(first_lang["overall"], dict):
                        vals[tok] = first_lang["overall"].get("f1") or first_lang["overall"].get("mean_f1")
                    else:
                        vals[tok] = None
                else:
                    vals[tok] = float(first_lang) if first_lang is not None else None
            else:
                vals[tok] = None
        global_data[metric_name] = vals

    global_df = pd.DataFrame(global_data, index=tokenizers)

    per_lang = {}
    for metric_name, (json_key, pl_subkey, _) in INTRINSIC_METRICS.items():
        data = {}
        for tok in tokenizers:
            pl = raw[json_key]["per_tokenizer"][tok].get("per_language", {})
            data[tok] = {lang: _extract_value(val, pl_subkey) for lang, val in pl.items()}
        per_lang[metric_name] = pd.DataFrame(data).T

    # Add renyi per-language
    for metric_name, pl_data in per_lang_renyi.items():
        if pl_data:
            per_lang[metric_name] = pd.DataFrame(pl_data).T

    return global_df, per_lang


def load_flores(prefix):
    """Load FLORES BPB. Handles both end-of-training and eval_runner formats."""
    agg_data = {}
    per_lang_data = {}

    for display_name, slug in NAME_MAP.items():
        path = FLORES_DIR / f"{prefix}-{slug}.json"
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)

        # End-of-training format: top-level summary_all, per_language
        if "summary_all" in d:
            agg_data[display_name] = {
                "flores_all_bpb": d["summary_all"]["mean_bpb"],
                "flores_train_bpb": d["summary_train_langs"]["mean_bpb"],
            }
            per_lang_data[display_name] = d["per_language"]
        # eval_runner format: scores.flores200.per_language[lang] = {"bpb": ..., "num_samples": ...}
        elif "scores" in d and "flores200" in d["scores"]:
            fl = d["scores"]["flores200"]
            pl = fl.get("per_language", {})
            bpbs = {}
            for lang, v in pl.items():
                bpbs[lang] = v["bpb"] if isinstance(v, dict) else v
            per_lang_data[display_name] = bpbs
            if bpbs:
                all_bpbs = list(bpbs.values())
                agg_data[display_name] = {
                    "flores_all_bpb": sum(all_bpbs) / len(all_bpbs),
                    "flores_train_bpb": None,  # not available in this format
                }

    return pd.DataFrame(agg_data).T, pd.DataFrame(per_lang_data).T


def load_blimp_and_code_bpb(prefix):
    """Load BLiMP accuracy, val_bpb, and code_bpb."""
    data = {}
    for display_name, slug in NAME_MAP.items():
        for pattern in [
            f"{prefix}-{slug}_blimp_code_bpb.json",
            f"{prefix}-{slug}_blimp.json",
        ]:
            path = BLIMP_DIR / pattern
            if path.exists():
                with open(path) as f:
                    d = json.load(f)
                row = {
                    "blimp_acc": d["scores"]["blimp"]["acc,none"],
                    "val_bpb": d["metadata"]["val_bpb"],
                }
                # Extract code_bpb if present
                code = d["scores"].get("code_bpb", {}).get("summary", {})
                if "mean_bpb" in code:
                    row["code_bpb"] = code["mean_bpb"]
                data[display_name] = row
                break
        # Also try standalone code_bpb files
        if display_name in data and "code_bpb" not in data[display_name]:
            for cp in [f"{prefix}-{slug}_codebpb.json"]:
                cpath = BLIMP_DIR / cp
                if cpath.exists():
                    with open(cpath) as f:
                        cd = json.load(f)
                    code = cd["scores"].get("code_bpb", {}).get("summary", {})
                    if "mean_bpb" in code:
                        data[display_name]["code_bpb"] = code["mean_bpb"]
    return pd.DataFrame(data).T


def load_gen_results(prefix, use_limit=None):
    """Load GSM8K, HumanEval, MBPP scores.

    Args:
        prefix: 'pilot-128k' or 'full-128k'
        use_limit: if set (e.g., 100), load from *_gen_limit{N}.json files
    """
    data = {}
    for display_name, slug in NAME_MAP.items():
        row = {}
        # When use_limit is set, ONLY load gen_limit files to avoid mixing
        # evaluation protocols (different sample sizes)
        if use_limit:
            patterns = [f"{prefix}-{slug}_gen_limit{use_limit}.json"]
        else:
            patterns = [
                f"{prefix}-{slug}_math_code.json",
                f"{prefix}-{slug}_gsm8k.json",
                f"{prefix}-{slug}_code_gen.json",
            ]
        for pattern in patterns:
            path = BLIMP_DIR / pattern
            if path.exists():
                with open(path) as f:
                    d = json.load(f)
                scores = d.get("scores", {})
                if "gsm8k_cot" in scores and "gsm8k" not in row:
                    row["gsm8k"] = scores["gsm8k_cot"].get("exact_match,strict-match")
                if "humaneval" in scores and "humaneval" not in row:
                    row["humaneval"] = scores["humaneval"].get("pass@1,create_test")
                if "mbpp" in scores and "mbpp" not in row:
                    row["mbpp"] = scores["mbpp"].get("pass_at_1,none")
        if row:
            data[display_name] = row
    return pd.DataFrame(data).T


def load_language_metadata():
    """Load language config with script_family groups."""
    with open(LANG_CONFIG) as f:
        cfg = json.load(f)
    langs = cfg.get("languages", {})
    groups = cfg.get("analysis_groups", {})
    script_map = {}
    for family, codes in groups.get("script_family", {}).items():
        for code in codes:
            script_map[code] = family
    resource_map = {}
    for level, codes in groups.get("resource_level", {}).items():
        for code in codes:
            resource_map[code] = level

    rows = {}
    for code, info in langs.items():
        name = info["name"] if isinstance(info, dict) else code
        rows[code] = {
            "name": name,
            "script_family": script_map.get(code, "Other"),
            "resource_level": resource_map.get(code, "unknown"),
            "training_weight": TRAINING_DATA_WEIGHTS.get(code, 0.0),
        }
    return pd.DataFrame(rows).T


def build_long_form(intrinsic_per_lang, flores_per_lang, lang_meta):
    """Build a long-form DataFrame with one row per (tokenizer, language) pair.

    Columns: tokenizer, language, flores_bpb, script_family, training_weight,
             compression_rate, fertility, bigram_entropy, renyi_efficiency, vocab_utilization
    """
    # Get union of languages across all per-language metrics
    all_langs = set()
    for df in intrinsic_per_lang.values():
        all_langs.update(df.columns)

    rows = []
    for tok in sorted(NAME_MAP.keys()):
        if tok not in flores_per_lang.index:
            continue
        for lang in sorted(all_langs):
            flores_lang = LANG_CODE_MAP.get(lang, lang)
            if flores_lang not in flores_per_lang.columns:
                continue
            bpb = flores_per_lang.loc[tok, flores_lang]
            if pd.isna(bpb):
                continue
            row = {"tokenizer": tok, "language": lang, "flores_bpb": bpb}
            for metric_name, df in intrinsic_per_lang.items():
                val = df.loc[tok, lang] if lang in df.columns else np.nan
                row[metric_name] = val
            if lang in lang_meta.index:
                row["script_family"] = lang_meta.loc[lang, "script_family"]
                row["training_weight"] = lang_meta.loc[lang, "training_weight"]
            else:
                row["script_family"] = "Other"
                row["training_weight"] = 0.0
            rows.append(row)
    return pd.DataFrame(rows)


# ─── Analysis ────────────────────────────────────────────────────────

def fdr_adjust(p_values):
    """Apply BH-FDR correction using scipy. NaN p-values are returned as NaN."""
    p = np.asarray(p_values, dtype=float)
    if len(p) == 0:
        return p
    valid = ~np.isnan(p)
    if not valid.any():
        return p
    result = np.full_like(p, np.nan)
    result[valid] = false_discovery_control(p[valid], method="bh")
    return result


def compute_aggregate_correlations(intrinsic_global, downstream):
    """Analysis 1: Cross-tokenizer aggregate correlations.

    Each data point is one tokenizer (n=15-16).
    """
    results = []
    for im in intrinsic_global.columns:
        for dm in downstream.columns:
            combined = pd.concat([intrinsic_global[im], downstream[dm]], axis=1).dropna()
            n = len(combined)
            if n < 5:
                continue
            x, y = combined.iloc[:, 0].values, combined.iloc[:, 1].values
            sr, sp = stats.spearmanr(x, y)
            pr, pp = stats.pearsonr(x, y)
            results.append({
                "intrinsic": im, "downstream": dm, "n": n,
                "spearman_rho": sr, "spearman_p": sp,
                "pearson_r": pr, "pearson_p": pp,
            })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df["spearman_p_adj"] = fdr_adjust(df["spearman_p"].values)
        df["pearson_p_adj"] = fdr_adjust(df["pearson_p"].values)
    return df


def compute_per_language_cross_tokenizer(intrinsic_per_lang, flores_per_lang):
    """Analysis 2: Per-language cross-tokenizer correlations.

    For each language L: ρ(intrinsic[T,L], flores_bpb[T,L]) across tokenizers.
    Each data point within a language is one tokenizer (n=15).
    """
    results = []
    for metric_name, intrinsic_df in intrinsic_per_lang.items():
        for lang in intrinsic_df.columns:
            flores_lang = LANG_CODE_MAP.get(lang, lang)
            if flores_lang not in flores_per_lang.columns:
                continue
            x = intrinsic_df[lang]
            y = flores_per_lang[flores_lang]
            combined = pd.concat([x, y], axis=1).dropna()
            n = len(combined)
            if n < 5:
                continue
            rho, p = stats.spearmanr(combined.iloc[:, 0], combined.iloc[:, 1])
            results.append({
                "metric": metric_name, "language": lang, "n": n,
                "rho": rho, "p": p,
                "training_weight": TRAINING_DATA_WEIGHTS.get(lang, 0.0),
            })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df["p_adj"] = fdr_adjust(df["p"].values)
    return df


def compute_mixed_effects(long_df):
    """Analysis 3: Mixed-effects regression of FLORES BPB on intrinsic metrics.

    Fits: BPB(T, L) ~ intrinsic(T, L) + (1 | language)

    The random intercept for language accounts for the fact that different
    languages have different baseline BPB regardless of tokenizer. The
    fixed-effect coefficient on the intrinsic metric tests whether, within
    a language, tokenizers with better intrinsic scores achieve lower BPB.
    This correctly handles the clustering of observations by language.
    """
    from statsmodels.regression.mixed_linear_model import MixedLM

    per_lang_metrics = set(INTRINSIC_METRICS.keys()) | set(RENYI_ALPHAS.keys())
    intrinsic_cols = [c for c in long_df.columns if c in per_lang_metrics]
    results = []

    for metric in intrinsic_cols:
        df = long_df[["tokenizer", "language", metric, "flores_bpb"]].dropna()
        if len(df) < 10 or df["language"].nunique() < 3:
            continue

        # Standardize X only; beta is in BPB units per SD of intrinsic metric
        x_std = df[metric].std()
        if x_std == 0:
            continue
        df["x_scaled"] = (df[metric] - df[metric].mean()) / x_std

        try:
            model = MixedLM.from_formula(
                "flores_bpb ~ x_scaled",
                groups="language",
                data=df,
            )
            fit = model.fit(reml=True)

            coef = fit.fe_params["x_scaled"]
            pval = fit.pvalues["x_scaled"]
            se = fit.bse["x_scaled"]
            n_obs = int(fit.nobs)
            n_groups = int(fit.model.n_groups)

            results.append({
                "metric": metric,
                "coefficient": coef,
                "std_error": se,
                "p_value": pval,
                "n_obs": n_obs,
                "n_languages": n_groups,
                "n_tokenizers": df["tokenizer"].nunique(),
            })
        except Exception as e:
            print(f"  Warning: MixedLM failed for {metric}: {e}")

    df = pd.DataFrame(results)
    if len(df) > 0:
        df["p_adj"] = fdr_adjust(df["p_value"].values)
    return df


def compute_controlled_comparisons(intrinsic_global, downstream, intrinsic_per_lang, flores_per_lang):
    """Analysis 4: Controlled pairwise comparisons with FDR correction."""
    results = {}
    all_wilcoxon_p = []  # collect for joint FDR correction

    for name, (tok_a, tok_b) in CONTROLLED_COMPARISONS.items():
        entry = {"tokenizers": [tok_a, tok_b]}

        for df, label in [(intrinsic_global, "intrinsic"), (downstream, "downstream")]:
            vals = {}
            for col in df.columns:
                va = df.loc[tok_a, col] if tok_a in df.index else None
                vb = df.loc[tok_b, col] if tok_b in df.index else None
                if va is not None and vb is not None and not (pd.isna(va) or pd.isna(vb)):
                    vals[col] = {"A": float(va), "B": float(vb), "delta": float(vb - va)}
            entry[f"{label}_values"] = vals

        if tok_a in flores_per_lang.index and tok_b in flores_per_lang.index:
            bpb_a = flores_per_lang.loc[tok_a]
            bpb_b = flores_per_lang.loc[tok_b]
            common = bpb_a.dropna().index.intersection(bpb_b.dropna().index)
            intrinsic_common = [LANG_CODE_MAP.get(l, l) for l in intrinsic_per_lang["compression_rate"].columns]
            flores_common = [l for l in intrinsic_common if l in common]

            deltas_bpb = bpb_b[flores_common] - bpb_a[flores_common]
            deltas_clean = deltas_bpb.dropna()
            if len(deltas_clean) >= 5:
                w_stat, w_p = stats.wilcoxon(deltas_clean)
                entry["flores_bpb_delta"] = {
                    "mean": float(deltas_clean.mean()),
                    "median": float(deltas_clean.median()),
                    "wilcoxon_stat": float(w_stat),
                    "wilcoxon_p": float(w_p),
                    "n_languages": int(len(deltas_clean)),
                    "direction": "B better" if deltas_clean.mean() < 0 else "A better",
                }
                all_wilcoxon_p.append((name, w_p))

        results[name] = entry

    # Apply FDR correction across all Wilcoxon tests
    if all_wilcoxon_p:
        names_ordered, raw_p = zip(*all_wilcoxon_p)
        adj_p = fdr_adjust(np.array(raw_p))
        for comp_name, p_adj in zip(names_ordered, adj_p):
            results[comp_name]["flores_bpb_delta"]["wilcoxon_p_adj"] = float(p_adj)

    return results


# ─── Output ──────────────────────────────────────────────────────────

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


def print_aggregate_table(df):
    if df.empty:
        print("  No results")
        return

    downstream_metrics = df["downstream"].unique()
    intrinsic_metrics = df["intrinsic"].unique()

    header = f"{'Intrinsic':>25s}"
    for dm in downstream_metrics:
        header += f" | {dm:>18s}"
    print(header)
    print("-" * len(header))

    for im in intrinsic_metrics:
        row = f"{im:>25s}"
        for dm in downstream_metrics:
            match = df[(df["intrinsic"] == im) & (df["downstream"] == dm)]
            if match.empty:
                row += f" | {'N/A':>18s}"
            else:
                r = match.iloc[0]
                stars = sig_stars(r["spearman_p_adj"])
                row += f" | {r['spearman_rho']:>7.3f}{stars:<3s} (n={int(r['n']):>2d})"
            row += " "
        print(row)


def print_per_language_summary(df, lang_meta):
    if df.empty:
        print("  No results")
        return

    for metric in df["metric"].unique():
        mdf = df[df["metric"] == metric].copy()
        print(f"\n  {metric}:")
        mdf = mdf.merge(lang_meta[["script_family", "name"]], left_on="language", right_index=True, how="left")

        for family in sorted(mdf["script_family"].unique()):
            fdf = mdf[mdf["script_family"] == family]
            median_rho = fdf["rho"].median()
            n_sig = (fdf["p_adj"] < 0.05).sum()
            langs = ", ".join(fdf["language"].values)
            print(f"    {family:>15s}: median ρ={median_rho:>7.3f}  ({n_sig}/{len(fdf)} sig.)  [{langs}]")

        median_rho = mdf["rho"].median()
        n_sig = (mdf["p_adj"] < 0.05).sum()
        print(f"    {'OVERALL':>15s}: median ρ={median_rho:>7.3f}  ({n_sig}/{len(mdf)} sig.)")


def print_mixed_effects_summary(df):
    if df.empty:
        print("  No results")
        return

    print(f"  {'Metric':>20s} {'β':>7s} {'SE':>7s} {'p_adj':>10s} {'n_obs':>6s} {'langs':>5s}")
    print(f"  {'-'*60}")
    for _, r in df.iterrows():
        stars = sig_stars(r["p_adj"])
        print(f"  {r['metric']:>20s} {r['coefficient']:>+6.3f}{stars:<1s} "
              f"{r['std_error']:>7.3f} {r['p_adj']:>10.4f} "
              f"{int(r['n_obs']):>6d} {int(r['n_languages']):>5d}")


# ─── LaTeX Output ────────────────────────────────────────────────────

# Display names for metrics in LaTeX tables
METRIC_DISPLAY = {
    "compression_rate": "Compression rate",
    "fertility": "Fertility",
    "unigram_entropy": "Unigram entropy",
    "bigram_entropy": "Bigram entropy",
    "trigram_entropy": "Trigram entropy",
    "renyi_2.0": r"R\'enyi eff.\ ($\alpha$=2)",
    "vocab_utilization": "Vocab utilization",
    "utf8_char_split": "UTF-8 char split",
    "gini_coefficient": "Gini coefficient",
    "ast_full_alignment": "AST alignment",
    "ident_fragmentation": "Ident.\ fragmentation",
    "digit_boundary_f1": "Digit boundary F1",
    "operator_isolation": "Operator isolation",
}

DOWNSTREAM_DISPLAY = {
    "flores_all_bpb": "FLORES",
    "val_bpb": "Val BPB",
    "code_bpb": "Code BPB",
    "blimp_acc": "BLiMP",
    "gsm8k": "GSM8K",
    "humaneval": "HumanEval",
    "mbpp": "MBPP",
}


def latex_sig(p):
    if p is None or np.isnan(p):
        return ""
    if p < 0.001: return "^{***}"
    if p < 0.01: return "^{**}"
    if p < 0.05: return "^{*}"
    return ""


def compute_metric_ordering(agg_corr):
    """Order intrinsic metrics by mean |ρ| across all downstream metrics."""
    if agg_corr.empty:
        return []
    ordering = {}
    for im in agg_corr["intrinsic"].unique():
        sub = agg_corr[agg_corr["intrinsic"] == im]
        rhos = sub["spearman_rho"].dropna().abs()
        ordering[im] = rhos.mean() if len(rhos) > 0 else 0
    return sorted(ordering.keys(), key=lambda x: ordering[x], reverse=True)


def generate_latex_mixed_effects_table(me_df, per_lang_corr):
    """Table 1: Mixed-effects + median within-language ρ for FLORES BPB."""
    if me_df.empty:
        print("% Mixed-effects table: no data")
        return

    # Compute median per-language Spearman ρ for each metric
    median_rho = {}
    if not per_lang_corr.empty:
        for metric in per_lang_corr["metric"].unique():
            rhos = per_lang_corr[per_lang_corr["metric"] == metric]["rho"]
            median_rho[metric] = rhos.median()

    # Order by |coefficient|
    me_df = me_df.copy()
    me_df["abs_coef"] = me_df["coefficient"].abs()
    me_df = me_df.sort_values("abs_coef", ascending=False)

    n_obs = int(me_df.iloc[0]["n_obs"]) if len(me_df) > 0 else 0
    n_langs = int(me_df.iloc[0]["n_languages"]) if len(me_df) > 0 else 0

    print("% Table: Mixed-effects + median within-language rho")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Relationship between intrinsic tokenizer metrics and FLORES BPB.")
    print(f"$\\beta$: standardized coefficient from mixed-effects regression")
    print(f"(BPB $\\sim$ metric + (1\\,|\\,language), $n={n_obs}$, {n_langs} languages).")
    print(f"$\\tilde{{\\rho}}$: median within-language Spearman correlation across {n_langs} languages.")
    print("\\textbf{Bold}: $p_{{\\text{{adj}}}} < 0.05$ (BH-FDR). $^{{*}}$\\,$p<0.05$, $^{{**}}$\\,$p<0.01$, $^{{***}}$\\,$p<0.001$.}")
    print("\\label{tab:mixed-effects}")
    print("\\small")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("\\textbf{Intrinsic Metric} & $\\beta$ & $\\tilde{\\rho}$ \\\\")
    print("\\midrule")

    for _, r in me_df.iterrows():
        metric = r["metric"]
        name = METRIC_DISPLAY.get(metric, metric)
        sig = latex_sig(r["p_adj"])
        is_sig = r["p_adj"] < 0.05

        # β with significance star, ± SE in scriptsize (not bold)
        beta_val = f"{r['coefficient']:+.3f}"
        if is_sig:
            beta_val = f"\\textbf{{{beta_val}}}"
        se_str = r"{\scriptsize" + f"$\\pm${r['std_error']:.3f}" + "}"
        beta_str = f"${beta_val}{sig}${se_str}"

        # Median ρ
        rho = median_rho.get(metric)
        if rho is not None:
            rho_str = f"{rho:+.3f}"
            if is_sig:
                rho_str = f"\\textbf{{{rho_str}}}"
            rho_str = f"${rho_str}$"
        else:
            rho_str = "--"

        print(f"{name} & {beta_str} & {rho_str} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def generate_latex_aggregate_table(agg_corr, metric_order):
    """Table 2: Aggregate cross-tokenizer correlations across all downstream metrics."""
    if agg_corr.empty:
        print("% Aggregate table: no data")
        return

    downstream_cols = [dm for dm in ["val_bpb", "flores_all_bpb", "code_bpb",
                                      "blimp_acc", "gsm8k", "humaneval", "mbpp"]
                       if dm in agg_corr["downstream"].unique()]

    # Get n from first valid entry
    first_valid = agg_corr.dropna(subset=["n"]).iloc[0] if len(agg_corr) > 0 else None
    n_str = f"$n={int(first_valid['n'])}$" if first_valid is not None else "$n$"

    print("% Table: Aggregate cross-tokenizer correlations")
    print("\\begin{table*}[t]")
    print("\\centering")
    print(f"\\caption{{Aggregate Spearman $\\rho$ between intrinsic metrics and downstream")
    print(f"performance ({n_str} tokenizers).")
    print("\\textbf{Bold}: $p_{\\text{adj}} < 0.05$ (BH-FDR). $^{*}$\\,$p<0.05$, $^{**}$\\,$p<0.01$, $^{***}$\\,$p<0.001$.}")
    print("\\label{tab:aggregate-correlations}")
    print("\\small")
    ncols = len(downstream_cols)
    print(f"\\begin{{tabular}}{{l{'c' * ncols}}}")
    print("\\toprule")
    header = "\\textbf{Intrinsic Metric}"
    for dm in downstream_cols:
        header += f" & \\textbf{{{DOWNSTREAM_DISPLAY.get(dm, dm)}}}"
    header += " \\\\"
    print(header)
    print("\\midrule")

    for im in metric_order:
        name = METRIC_DISPLAY.get(im, im)
        row = name
        for dm in downstream_cols:
            match = agg_corr[(agg_corr["intrinsic"] == im) & (agg_corr["downstream"] == dm)]
            if match.empty:
                row += " & --"
            else:
                r = match.iloc[0]
                sig = latex_sig(r["spearman_p_adj"])
                is_sig = r["spearman_p_adj"] < 0.05 if not np.isnan(r["spearman_p_adj"]) else False
                val_str = f"{r['spearman_rho']:.2f}"
                if is_sig:
                    row += f" & $\\textbf{{{val_str}}}{sig}$"
                else:
                    row += f" & ${val_str}$"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print(f"\\end{{tabular}}")
    print("\\end{table*}")


# Comparison groups with explicit A→B labels for LaTeX table
# Each entry: (group_name, [(comp_key, A_label, B_label), ...])
COMPARISON_GROUPS_LABELED = [
    ("NFC normalization", "Effect of adding NFC", [
        ("NFC: GPT-4o", "GPT-4o", "+ NFC"),
        ("NFC: Claude", "Claude", "+ NFC"),
        ("NFC: RightAlign", "RightAlign", "+ NFC"),
    ]),
    ("Algorithm", "Effect of BPE $\\to$ UnigramLM", [
        ("Algo: GPT-4o BPE vs Unigram", "BPE GPT-4o", "Unigram"),
        ("Algo: Claude BPE vs Unigram", "BPE Claude", "Unigram"),
        ("Algo: RightAlign BPE vs Unigram", "BPE RightAlign", "Unigram"),
    ]),
    ("Pretokenizer", "Baseline: GPT-4o regex", [
        ("Pretok: GPT-4o vs Claude", "GPT-4o", "Claude"),
        ("Pretok: GPT-4o vs Punct", "GPT-4o", "Punct"),
        ("Pretok: GPT-4o vs RightAlign", "GPT-4o", "RightAlign"),
    ]),
    ("Training data", "Baseline: balanced multilingual", [
        ("Data: GPT-4o balanced vs english", "Balanced", "English"),
        ("Data: GPT-4o balanced vs code", "Balanced", "Code"),
    ]),
]


def generate_latex_compact_comparison(controlled):
    """Compact pairwise comparison table with explicit A/B labeling."""
    if not controlled:
        print("% Compact comparison table: no data")
        return

    print("% Table: Compact pairwise tokenizer comparisons")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Controlled pairwise comparisons. $\\Delta$ = B $-$ A for each metric.")
    print("FLORES $\\Delta$ tested via Wilcoxon signed-rank across 31 languages.")
    print("\\textbf{Bold}: $p_{\\text{adj}} < 0.05$ (BH-FDR). BPB metrics: negative $\\Delta$ means B is better.}")
    print("\\label{tab:pairwise-compact}")
    print("\\small")
    print("\\begin{tabular}{llllcccc}")
    print("\\toprule")
    print("\\textbf{Factor} & \\textbf{A} & \\textbf{B} & $\\Delta$\\textbf{Val} & $\\Delta$\\textbf{FLORES} & $\\Delta$\\textbf{BLiMP} & $\\Delta$\\textbf{Code} \\\\")
    print("\\midrule")

    first_group = True
    for group_name, group_desc, comparisons in COMPARISON_GROUPS_LABELED:
        if not first_group:
            print("\\midrule")
        first_group = False

        for j, (comp_key, a_label, b_label) in enumerate(comparisons):
            if comp_key not in controlled:
                continue
            data = controlled[comp_key]
            dv = data.get("downstream_values", {})

            def get_delta(metric):
                v = dv.get(metric, {})
                return v.get("delta") if v else None

            val_d = get_delta("val_bpb")
            blimp_d = get_delta("blimp_acc")
            code_d = get_delta("code_bpb")

            flores_info = data.get("flores_bpb_delta", {})
            flores_d = flores_info.get("mean")
            flores_p = flores_info.get("wilcoxon_p_adj", flores_info.get("wilcoxon_p", 1.0))
            flores_sig = flores_p < 0.05 if flores_p is not None else False

            def fmt_delta(v, bold=False):
                if v is None: return "--"
                s = f"{v:+.4f}" if abs(v) < 0.1 else f"{v:+.3f}"
                return f"\\textbf{{{s}}}" if bold else s

            # Show group name only on first row
            grp = f"\\multirow{{{len(comparisons)}}}{{*}}{{{group_name}}}" if j == 0 else ""

            print(f"{grp} & {a_label} & {b_label} & "
                  f"${fmt_delta(val_d)}$ & ${fmt_delta(flores_d, flores_sig)}$ & "
                  f"${fmt_delta(blimp_d)}$ & ${fmt_delta(code_d)}$ \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def generate_latex_heatmap_comparison(controlled, downstream):
    """Heatmap-style table: tokenizer × metric with winner highlighted."""
    if not controlled or downstream.empty:
        print("% Heatmap comparison table: no data")
        return

    # Collect all tokenizers involved in comparisons
    tok_set = set()
    for data in controlled.values():
        tok_set.update(data["tokenizers"])

    # Get metrics for these tokenizers
    metrics = ["val_bpb", "flores_all_bpb", "code_bpb", "blimp_acc"]
    available_metrics = [m for m in metrics if m in downstream.columns]
    toks = sorted(tok_set & set(downstream.index))

    if not toks or not available_metrics:
        print("% Heatmap: insufficient data")
        return

    # For each metric, find the best value among compared tokenizers
    # Lower is better for BPB metrics, higher for acc
    higher_is_better = {"blimp_acc": True}

    print("% Table: Heatmap-style tokenizer comparison")
    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\caption{Downstream metrics for tokenizers in controlled comparisons.")
    print("\\textbf{Bold}: best within each controlled pair.}")
    print("\\label{tab:heatmap-comparison}")
    print("\\small")
    ncols = len(available_metrics)
    print(f"\\begin{{tabular}}{{l{'c' * ncols}}}")
    print("\\toprule")
    header = "\\textbf{Tokenizer}"
    for m in available_metrics:
        header += f" & \\textbf{{{DOWNSTREAM_DISPLAY.get(m, m)}}}"
    header += " \\\\"
    print(header)
    print("\\midrule")

    # Build pair winners for bolding
    pair_winners = {}  # (tok, metric) -> True if best in its pair
    for comp_name, (tok_a, tok_b) in CONTROLLED_COMPARISONS.items():
        if tok_a not in downstream.index or tok_b not in downstream.index:
            continue
        for m in available_metrics:
            va = downstream.loc[tok_a, m] if tok_a in downstream.index else None
            vb = downstream.loc[tok_b, m] if tok_b in downstream.index else None
            if va is None or vb is None or pd.isna(va) or pd.isna(vb):
                continue
            hib = higher_is_better.get(m, False)
            if (hib and va >= vb) or (not hib and va <= vb):
                pair_winners[(tok_a, m)] = True
            else:
                pair_winners[(tok_b, m)] = True

    # Short display names for tokenizers
    tok_short = {t: t.replace("BPE ", "").replace("(balanced)", "(bal.)").replace("(english)", "(eng.)")
                 for t in toks}

    for tok in toks:
        name = tok_short.get(tok, tok)
        row = name
        for m in available_metrics:
            val = downstream.loc[tok, m] if tok in downstream.index and m in downstream.columns else None
            if val is None or pd.isna(val):
                row += " & --"
            else:
                dec = 4 if "bpb" in m.lower() else 3
                s = f"{val:.{dec}f}"
                if (tok, m) in pair_winners:
                    s = f"\\textbf{{{s}}}"
                row += f" & ${s}$"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print(f"\\end{{tabular}}")
    print("\\end{table*}")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Correlate intrinsic tokenizer metrics with downstream LM performance.")
    parser.add_argument("--scale", choices=["pilot", "full"], default="full",
                        help="Model scale: 'pilot' (300M) or 'full' (1.27B)")
    parser.add_argument("--gen-limit", type=int, default=None,
                        help="Use gen_limit{N} result files for generation benchmarks (e.g., --gen-limit 100)")
    parser.add_argument("--latex", action="store_true",
                        help="Output LaTeX tables instead of markdown")
    parser.add_argument("--intrinsic-results", type=str, default=None,
                        help="Path to intrinsic evaluation results JSON (default: tokenizer-intrinsic-evals/results/downstream_lm_flores/analysis_results.json)")
    args = parser.parse_args()

    prefix = "pilot-128k" if args.scale == "pilot" else "full-128k"
    intrinsic_path = Path(args.intrinsic_results) if args.intrinsic_results else INTRINSIC_RESULTS_DEFAULT
    print(f"Loading data (scale={args.scale}, prefix={prefix}"
          f"{f', gen-limit={args.gen_limit}' if args.gen_limit else ''}"
          f", intrinsic={intrinsic_path.parent.name}/{intrinsic_path.name})...")
    intrinsic_global, intrinsic_per_lang = load_intrinsic(intrinsic_path)
    flores_agg, flores_per_lang = load_flores(prefix)
    blimp_code = load_blimp_and_code_bpb(prefix)
    gen = load_gen_results(prefix, use_limit=args.gen_limit)
    lang_meta = load_language_metadata()
    downstream = pd.concat([flores_agg, blimp_code, gen], axis=1)

    # Build long-form DataFrame: one row per (tokenizer, language) pair
    long_df = build_long_form(intrinsic_per_lang, flores_per_lang, lang_meta)

    n_flores = flores_agg["flores_all_bpb"].dropna().shape[0]
    print(f"Tokenizers: {len(intrinsic_global)} intrinsic, {n_flores} FLORES, {len(blimp_code)} BLiMP")
    print(f"Intrinsic metrics: {list(intrinsic_global.columns)}")
    print(f"Long-form data: {len(long_df)} (tokenizer, language) pairs "
          f"({long_df['tokenizer'].nunique()} tokenizers × {long_df['language'].nunique()} languages)")

    # Analysis 1: Aggregate cross-tokenizer correlations
    agg_corr = compute_aggregate_correlations(intrinsic_global, downstream)

    # Analysis 2: Per-language cross-tokenizer correlations
    per_lang_corr = compute_per_language_cross_tokenizer(intrinsic_per_lang, flores_per_lang)

    # Analysis 3: Mixed-effects regression (BPB ~ intrinsic + (1|language))
    mixed_effects = compute_mixed_effects(long_df)

    # Analysis 4: Controlled pairwise comparisons
    controlled = compute_controlled_comparisons(intrinsic_global, downstream, intrinsic_per_lang, flores_per_lang)

    # Compute metric ordering by mean |ρ| across downstream metrics
    metric_order = compute_metric_ordering(agg_corr)

    if args.latex:
        # LaTeX output
        generate_latex_mixed_effects_table(mixed_effects, per_lang_corr)
        print()
        generate_latex_aggregate_table(agg_corr, metric_order)
        print()
        generate_latex_compact_comparison(controlled)
        print()
        generate_latex_heatmap_comparison(controlled, downstream)
    else:
        # Markdown output
        print(f"\n{'='*70}")
        print("ANALYSIS 1: Aggregate cross-tokenizer correlations (Spearman ρ)")
        print(f"  Each data point = one tokenizer")
        print(f"{'='*70}")
        print_aggregate_table(agg_corr)

        print(f"\n{'='*70}")
        print("ANALYSIS 2: Per-language cross-tokenizer correlations")
        print(f"  For each language: ρ(intrinsic[T,L], flores_bpb[T,L]) across tokenizers")
        print(f"{'='*70}")
        print_per_language_summary(per_lang_corr, lang_meta)

        print(f"\n{'='*70}")
        print("ANALYSIS 3: Mixed-effects regression (BPB ~ intrinsic + (1|language))")
        print(f"  Random intercept per language accounts for baseline BPB differences.")
        print(f"{'='*70}")
        print_mixed_effects_summary(mixed_effects)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    agg_corr.to_csv(OUTPUT_DIR / "aggregate_correlations.csv", index=False)
    per_lang_corr.to_csv(OUTPUT_DIR / "per_language_correlations.csv", index=False)
    mixed_effects.to_csv(OUTPUT_DIR / "mixed_effects.csv", index=False)
    long_df.to_csv(OUTPUT_DIR / "long_form_data.csv", index=False)

    full_results = {
        "aggregate_correlations": agg_corr.to_dict(orient="records"),
        "per_language_correlations": per_lang_corr.to_dict(orient="records"),
        "mixed_effects": mixed_effects.to_dict(orient="records"),
    }
    with open(OUTPUT_DIR / "full_results.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_DIR}", file=__import__('sys').stderr)


if __name__ == "__main__":
    main()
