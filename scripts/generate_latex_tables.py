#!/usr/bin/env python3
"""Generate LaTeX tables for the paper from evaluation results.

Usage:
    python scripts/generate_latex_tables.py [--gen-limit N]

Options:
    --gen-limit N   Use gen_limitN result files for generation benchmarks
"""

import argparse
import json
import os
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CKPT_BASE = Path("/capstor/scratch/cscs/cmeister747/tokenizer-lm/checkpoints")

# Display name -> run slug
TOKENIZER_ORDER = [
    ("Apertus", "apertus"),
    ("LLaMA-3", "llama3"),
    ("BPE Punct (bal.)", "punct-balanced-bpe"),
    ("BPE Punct (eng.)", "punct-english-bpe"),
    ("BPE GPT-4o (bal.)", "gpt4o-balanced-bpe"),
    ("BPE GPT-4o NFC", "gpt4o-balanced-nfc-bpe"),
    ("BPE GPT-4o (eng.)", "gpt4o-english-bpe"),
    ("BPE GPT-4o (code)", "gpt4o-code-bpe"),
    ("BPE Claude (bal.)", "claude-balanced-bpe"),
    ("BPE Claude NFC", "claude-balanced-nfc-bpe"),
    ("BPE Claude (eng.)", "claude-english-bpe"),
    ("BPE RightAlign (bal.)", "rightalign-balanced-bpe"),
    ("BPE RightAlign NFC", "rightalign-balanced-nfc-bpe"),
    ("Unigram GPT-4o", "gpt4o-balanced-unigram"),
    ("Unigram Claude", "claude-balanced-unigram"),
    ("Unigram RightAlign", "rightalign-balanced-unigram"),
    ("SuperBPE", "superbpe"),
    ("PA-BPE NFC", "pabpe"),
]


def load_val_bpb(prefix):
    """Load val_bpb from checkpoint metadata."""
    results = {}
    for display, slug in TOKENIZER_ORDER:
        ckpt_dir = CKPT_BASE / f"{prefix}-{slug}"
        metas = sorted(ckpt_dir.glob("meta_*.json"))
        if not metas:
            continue
        with open(metas[-1]) as f:
            d = json.load(f)
        results[slug] = d.get("val_bpb")
    return results


def load_blimp(prefix):
    """Load BLiMP accuracy from result files."""
    results = {}
    for display, slug in TOKENIZER_ORDER:
        for pattern in [f"{prefix}-{slug}_blimp_code_bpb.json", f"{prefix}-{slug}_blimp.json"]:
            path = RESULTS_DIR / "lm_eval" / pattern
            if path.exists():
                with open(path) as f:
                    d = json.load(f)
                results[slug] = d["scores"]["blimp"]["acc,none"]
                break
    return results


def load_code_bpb(prefix):
    """Load code BPB from result files."""
    results = {}
    for display, slug in TOKENIZER_ORDER:
        for pattern in [f"{prefix}-{slug}_blimp_code_bpb.json", f"{prefix}-{slug}_codebpb.json"]:
            path = RESULTS_DIR / "lm_eval" / pattern
            if path.exists():
                with open(path) as f:
                    d = json.load(f)
                code = d["scores"].get("code_bpb", {})
                if "summary" in code:
                    results[slug] = code["summary"]["mean_bpb"]
                break
    return results


def load_flores(prefix):
    """Load FLORES mean BPB from result files."""
    results = {}
    for display, slug in TOKENIZER_ORDER:
        path = RESULTS_DIR / "flores" / f"{prefix}-{slug}.json"
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            if "summary_all" in d:
                results[slug] = d["summary_all"]["mean_bpb"]
            elif "scores" in d and "flores200" in d["scores"]:
                fl = d["scores"]["flores200"]
                if "per_language" in fl:
                    bpbs = [v["bpb"] for v in fl["per_language"].values()]
                    results[slug] = sum(bpbs) / len(bpbs) if bpbs else None
    return results


def load_gen_results(prefix, gen_limit=None):
    """Load GSM8K, MGSM, HumanEval, MBPP from result files."""
    gsm8k, humaneval, mbpp = {}, {}, {}
    for display, slug in TOKENIZER_ORDER:
        patterns = []
        if gen_limit:
            patterns.append(f"{prefix}-{slug}_gen_limit{gen_limit}.json")
        patterns.extend([
            f"{prefix}-{slug}_math_code.json",
            f"{prefix}-{slug}_gsm8k.json",
            f"{prefix}-{slug}_code_gen.json",
        ])
        for pattern in patterns:
            path = RESULTS_DIR / "lm_eval" / pattern
            if path.exists():
                with open(path) as f:
                    d = json.load(f)
                scores = d.get("scores", {})
                if "gsm8k_cot" in scores:
                    gsm8k[slug] = scores["gsm8k_cot"].get("exact_match,strict-match")
                if "humaneval" in scores:
                    humaneval[slug] = scores["humaneval"].get("pass@1,create_test")
                if "mbpp" in scores:
                    mbpp[slug] = scores["mbpp"].get("pass_at_1,none")
    return gsm8k, humaneval, mbpp


def fmt(val, decimals=3):
    if val is None:
        return "--"
    return f"{val:.{decimals}f}"


def generate_main_table(gen_limit=None):
    """Generate the main results table for 1B models."""
    print("% Table: Main 1B results")
    print("\\begin{table*}[t]")
    print("\\centering")
    limit_note = f" (generation metrics on {gen_limit} samples)" if gen_limit else ""
    print(f"\\caption{{1.27B model evaluation results{limit_note}. $\\downarrow$: lower is better. $\\uparrow$: higher is better.}}")
    print("\\label{tab:main-results}")
    print("\\small")
    print("\\begin{tabular}{l ccc c ccc}")
    print("\\toprule")
    print(" & \\multicolumn{3}{c}{\\textbf{Perplexity $\\downarrow$}} & \\textbf{Ling.} & \\multicolumn{3}{c}{\\textbf{Generation $\\uparrow$}} \\\\")
    print("\\cmidrule(lr){2-4} \\cmidrule(lr){5-5} \\cmidrule(lr){6-8}")
    print("\\textbf{Tokenizer} & \\textbf{Val} & \\textbf{FLORES} & \\textbf{Code} & \\textbf{BLiMP} & \\textbf{HumanEval} & \\textbf{MBPP} & \\textbf{GSM8K} \\\\")
    print("\\midrule")

    val_bpb = load_val_bpb("full-128k")
    blimp = load_blimp("full-128k")
    code_bpb = load_code_bpb("full-128k")
    flores = load_flores("full-128k")
    gsm8k, humaneval, mbpp = load_gen_results("full-128k", gen_limit=gen_limit)

    for display, slug in TOKENIZER_ORDER:
        vb = fmt(val_bpb.get(slug), 4)
        fl = fmt(flores.get(slug), 3)
        cb = fmt(code_bpb.get(slug), 3)
        bl = fmt(blimp.get(slug), 3)
        he = fmt(humaneval.get(slug), 3)
        mb = fmt(mbpp.get(slug), 3)
        gs = fmt(gsm8k.get(slug), 3)
        print(f"{display} & {vb} & {fl} & {cb} & {bl} & {he} & {mb} & {gs} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table*}")


def generate_pilot_table():
    """Generate the pilot results table for 300M models."""
    print("\n% Table: Pilot 300M results")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Pilot-scale (300M) evaluation results. All metrics: lower is better.}")
    print("\\label{tab:pilot-results}")
    print("\\small")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("\\textbf{Tokenizer} & \\textbf{Val BPB} & \\textbf{FLORES} & \\textbf{Code BPB} \\\\")
    print("\\midrule")

    val_bpb = load_val_bpb("pilot-128k")
    code_bpb = load_code_bpb("pilot-128k")
    flores = load_flores("pilot-128k")

    for display, slug in TOKENIZER_ORDER:
        if slug not in val_bpb:
            continue
        vb = fmt(val_bpb.get(slug), 4)
        fl = fmt(flores.get(slug), 3)
        cb = fmt(code_bpb.get(slug), 3)
        print(f"{display} & {vb} & {fl} & {cb} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def generate_tokenizer_grid():
    """Generate the tokenizer configuration grid table."""
    print("\n% Table: Tokenizer grid")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Tokenizer configurations. Pretok: P=Punctuation, G=GPT-4o, C=Claude, R=Right-aligned. Data: B=Balanced, E=English, Co=Code.}")
    print("\\label{tab:tokenizer-grid}")
    print("\\small")
    print("\\begin{tabular}{llcccc}")
    print("\\toprule")
    print("\\textbf{Tokenizer} & \\textbf{Algorithm} & \\textbf{Pretok} & \\textbf{Data} & \\textbf{NFC} & \\textbf{Source} \\\\")
    print("\\midrule")

    grid = [
        ("Apertus", "BPE", "G", "--", "--", "Off-shelf"),
        ("LLaMA-3", "BPE", "--", "--", "--", "Off-shelf"),
        ("BPE Punct (bal.)", "BPE", "P", "B", "\\xmark", "Custom"),
        ("BPE Punct (eng.)", "BPE", "P", "E", "\\xmark", "Custom"),
        ("BPE GPT-4o (bal.)", "BPE", "G", "B", "\\xmark", "Custom"),
        ("BPE GPT-4o NFC", "BPE", "G", "B", "\\cmark", "Custom"),
        ("BPE GPT-4o (eng.)", "BPE", "G", "E", "\\xmark", "Custom"),
        ("BPE GPT-4o (code)", "BPE", "G", "Co", "\\xmark", "Custom"),
        ("BPE Claude (bal.)", "BPE", "C", "B", "\\xmark", "Custom"),
        ("BPE Claude NFC", "BPE", "C", "B", "\\cmark", "Custom"),
        ("BPE Claude (eng.)", "BPE", "C", "E", "\\xmark", "Custom"),
        ("BPE RightAlign (bal.)", "BPE", "R", "B", "\\xmark", "Custom"),
        ("BPE RightAlign NFC", "BPE", "R", "B", "\\cmark", "Custom"),
        ("Unigram GPT-4o", "Unigram", "G", "B", "\\xmark", "Custom"),
        ("Unigram Claude", "Unigram", "C", "B", "\\xmark", "Custom"),
        ("Unigram RightAlign", "Unigram", "R", "B", "\\xmark", "Custom"),
        ("SuperBPE", "SuperBPE", "G", "B", "\\xmark", "Custom"),
        ("PA-BPE NFC", "PA-BPE", "G", "B", "\\cmark", "Custom"),
    ]

    for name, algo, pretok, data, nfc, source in grid:
        print(f"{name} & {algo} & {pretok} & {data} & {nfc} & {source} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def generate_lang_breakdown():
    """Generate multilingual training data breakdown table from MODEL_TRAINING.md."""
    # Data from MODEL_TRAINING.md, sorted by actual byte share
    langs = [
        ("Russian", "rus_Cyrl", "Cyrillic", "10.6", "4.521", "18.1", "531,387"),
        ("Spanish", "spa_Latn", "Latin", "2.4", "0.423", "1.7", "122,231"),
        ("German", "deu_Latn", "Latin", "2.4", "0.353", "1.4", "118,340"),
        ("French", "fra_Latn", "Latin", "2.1", "0.364", "1.5", "103,496"),
        ("Chinese", "cmn_Hani", "CJK", "1.9", "0.326", "1.3", "95,805"),
        ("Japanese", "jpn_Jpan", "CJK", "1.3", "0.204", "0.8", "63,316"),
        ("Italian", "ita_Latn", "Latin", "1.2", "0.165", "0.7", "59,676"),
        ("Portuguese", "por_Latn", "Latin", "1.2", "0.189", "0.8", "59,460"),
        ("Turkish", "tur_Latn", "Latin", "0.9", "0.152", "0.6", "47,074"),
        ("Indonesian", "ind_Latn", "Latin", "0.9", "0.236", "0.9", "45,970"),
        ("Polish", "pol_Latn", "Latin", "0.9", "0.107", "0.4", "43,902"),
        ("Ukrainian", "ukr_Cyrl", "Cyrillic", "0.7", "0.211", "0.8", "34,088"),
        ("Dutch", "nld_Latn", "Latin", "0.6", "0.071", "0.3", "32,404"),
        ("Romanian", "ron_Latn", "Latin", "0.6", "0.097", "0.4", "29,203"),
        ("Arabic", "arb_Arab", "Arabic", "0.5", "0.168", "0.7", "27,292"),
        ("Hungarian", "hun_Latn", "Latin", "0.5", "0.099", "0.4", "25,066"),
        ("Vietnamese", "vie_Latn", "Latin", "0.4", "0.109", "0.4", "21,306"),
        ("Czech", "ces_Latn", "Latin", "0.4", "0.077", "0.3", "21,696"),
        ("Greek", "ell_Grek", "Greek", "0.4", "0.108", "0.4", "17,672"),
        ("Thai", "tha_Thai", "Thai", "0.2", "0.087", "0.3", "11,639"),
        ("Finnish", "fin_Latn", "Latin", "0.2", "0.034", "0.1", "11,721"),
        ("Slovak", "slk_Latn", "Latin", "0.2", "0.028", "0.1", "10,015"),
        ("Bulgarian", "bul_Cyrl", "Cyrillic", "0.2", "0.052", "0.2", "9,513"),
        ("Korean", "kor_Hang", "Hangul", "0.2", "0.029", "0.1", "8,263"),
        ("Croatian", "hrv_Latn", "Latin", "0.2", "0.026", "0.1", "8,395"),
        ("Catalan", "cat_Latn", "Latin", "0.1", "0.014", "0.1", "5,087"),
        ("Hindi", "hin_Deva", "Devanagari", "0.1", "0.032", "0.1", "4,928"),
        ("Hebrew", "heb_Hebr", "Hebrew", "0.1", "0.025", "0.1", "4,665"),
        ("Bengali", "ben_Beng", "Bengali", "0.1", "0.024", "0.1", "3,525"),
        ("Tamil", "tam_Taml", "Tamil", "0.0", "0.017", "0.1", "1,831"),
    ]

    print("% Table: Multilingual training data by language")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Multilingual training data by language, sorted by byte share.")
    print("11 writing systems, 11+ language families. Data scarcity spans $\\sim$180$\\times$")
    print("from Russian to Tamil.}")
    print("\\label{tab:lang-breakdown}")
    print("\\scriptsize")
    print("\\begin{tabular}{llcrrc}")
    print("\\toprule")
    print("\\textbf{Language} & \\textbf{Code} & \\textbf{Script} & \\textbf{GB} & \\textbf{\\%} & \\textbf{Docs} \\\\")
    print("\\midrule")

    for name, code, script, weight, gb, pct, docs in langs:
        print(f"{name} & \\texttt{{{code}}} & {script} & {gb} & {pct} & {docs} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-limit", type=int, default=None,
                        help="Use gen_limit{N} files for generation benchmarks")
    parser.add_argument("--table", choices=["all", "grid", "main", "pilot", "langs"],
                        default="all", help="Which table to generate")
    args = parser.parse_args()

    if args.table in ("all", "grid"):
        generate_tokenizer_grid()
        print()
    if args.table in ("all", "main"):
        generate_main_table(gen_limit=args.gen_limit)
        print()
    if args.table in ("all", "pilot"):
        generate_pilot_table()
        print()
    if args.table in ("all", "langs"):
        generate_lang_breakdown()
        print()
    generate_pilot_table()
