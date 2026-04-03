"""
Training script for tokenizer-lm.

Built on nanochat's GPT model, Muon optimizer, and streaming dataloader.
Adds YAML config support and swappable tokenizer interface.

Usage:
    # Single GPU:
    python train.py --config configs/pilot_125m.yaml --tokenizer /path/to/tokenizer

    # Multi-GPU (single node):
    torchrun --nproc_per_node=4 train.py --config configs/pilot_125m.yaml --tokenizer /path/to/tokenizer

    # Multi-node (via SLURM):
    sbatch scripts/slurm_pilot.sh /path/to/tokenizer
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import time
import math
import argparse
from dataclasses import asdict

import wandb
import yaml
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanochat.common import (
    compute_init, compute_cleanup, print0, DummyWandb,
    get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON,
    is_ddp_initialized,
)
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
import nanochat.dataset

from tokenizer_lm.tokenizer import load_tokenizer, generate_token_bytes


def _prune_checkpoints(checkpoint_dir, max_keep, rank):
    """Remove old checkpoints, keeping only the most recent max_keep."""
    import glob
    model_files = sorted(glob.glob(os.path.join(checkpoint_dir, "model_*.pt")))
    if len(model_files) <= max_keep:
        return
    to_remove = model_files[:-max_keep]
    for model_path in to_remove:
        step_str = os.path.basename(model_path).split("_")[1].split(".")[0]
        # Remove model, meta, and optimizer files for this step
        for pattern in [f"model_{step_str}.pt", f"meta_{step_str}.json", f"optim_{step_str}_rank*.pt"]:
            for f in glob.glob(os.path.join(checkpoint_dir, pattern)):
                os.remove(f)
        if rank == 0:
            print0(f"Pruned checkpoint step {int(step_str)}")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train a language model with swappable tokenizers")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to HuggingFace tokenizer (directory or hub name)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory (default: from config)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="W&B run name (default: auto from config)")
    parser.add_argument("--resume-from-step", type=int, default=-1,
                        help="Resume training from this step (-1 = disabled)")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Initialize model weights from checkpoint dir:step (e.g., /path/to/ckpt:8800). "
                             "Loads only model weights; optimizer, step counter, and LR schedule start fresh.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: from config, or 42)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    optim_cfg = cfg.get("optimizer", {})
    log_cfg = cfg.get("logging", {})
    wandb_cfg = cfg.get("wandb", {})

    # --- Seed (must be set before compute_init, which also sets seeds) ---
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # --- Compute init ---
    # Note: compute_init also calls torch.manual_seed(42) internally.
    # We re-apply our seed after it returns.
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    master_process = ddp_rank == 0
    synchronize = torch.cuda.synchronize

    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
    print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

    # --- W&B ---
    run_name = args.run_name or wandb_cfg.get("run_name", "dummy")
    use_dummy = run_name == "dummy" or not master_process or not wandb_cfg.get("enabled", True)
    wandb_run = DummyWandb() if use_dummy else wandb.init(
        project=os.environ.get("WANDB_PROJECT", wandb_cfg.get("project", "tokenizer-lm")),
        entity=os.environ.get("WANDB_ENTITY", wandb_cfg.get("entity", None)),
        name=run_name,
        config=cfg,
    )

    # --- Tokenizer ---
    tokenizer, vocab_size, bos_id = load_tokenizer(args.tokenizer)
    token_bytes = generate_token_bytes(tokenizer, device=device)
    print0(f"Tokenizer: {args.tokenizer}")
    print0(f"  Vocab size: {vocab_size}, BOS: {tokenizer.get_bos_token_name()} (id={bos_id})")

    # --- Data directory ---
    data_dir = args.data_dir or cfg.get("data", {}).get("data_dir")
    if data_dir is None:
        raise ValueError("No data directory specified. Use --data-dir or set data.data_dir in config.")
    # Point nanochat's dataset module at our data
    nanochat.dataset.DATA_DIR = data_dir
    print0(f"Data directory: {data_dir}")

    # --- Model ---
    head_dim = model_cfg.get("head_dim", 128)
    n_embd = model_cfg["n_embd"]
    # Nudge n_embd to nearest multiple of head_dim
    n_embd = ((n_embd + head_dim - 1) // head_dim) * head_dim
    n_head = n_embd // head_dim

    config = GPTConfig(
        sequence_len=model_cfg.get("sequence_len", 2048),
        vocab_size=vocab_size,
        n_layer=model_cfg["n_layer"],
        n_head=n_head,
        n_kv_head=model_cfg.get("n_kv_head", n_head),
        n_embd=n_embd,
        window_pattern=model_cfg.get("window_pattern", "SSSL"),
        ve_dim=model_cfg.get("ve_dim", 0),
    )
    config_kwargs = asdict(config)
    print0(f"Model config:\n{json.dumps(config_kwargs, indent=2)}")

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()

    # --- Checkpoint resume ---
    output_dir = os.path.expandvars(cfg.get("output_dir", "out"))
    run_tag = args.run_name or cfg.get("name", "run")
    checkpoint_dir = os.path.join(output_dir, run_tag)
    resuming = args.resume_from_step != -1

    if resuming:
        print0(f"Resuming from step {args.resume_from_step}")
        model_data, optimizer_data, meta_data = load_checkpoint(
            checkpoint_dir, args.resume_from_step, device,
            load_optimizer=True, rank=ddp_rank,
        )
        model.load_state_dict(model_data, strict=True, assign=True)
        del model_data
    elif args.init_from:
        # Load model weights only (no optimizer, no step counter)
        init_parts = args.init_from.rsplit(":", 1)
        if len(init_parts) != 2:
            raise ValueError(f"--init-from must be 'checkpoint_dir:step', got '{args.init_from}'")
        init_dir, init_step = init_parts[0], int(init_parts[1])
        print0(f"Initializing model weights from {init_dir} step {init_step}")
        model_data, _, _ = load_checkpoint(
            init_dir, init_step, device, load_optimizer=False,
        )
        model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
        model.load_state_dict(model_data, strict=True, assign=True)
        del model_data
        print0(f"Model weights loaded (optimizer and schedule start fresh)")

    # --- Compile ---
    orig_model = model
    if train_cfg.get("compile", True):
        model = torch.compile(model, dynamic=False)
        print0("Model compiled with torch.compile")

    # --- Parameter counts ---
    param_counts = model.num_scaling_params()
    print0("Parameter counts:")
    for key, value in param_counts.items():
        print0(f"  {key:24s}: {value:,}")
    num_flops_per_token = model.estimate_flops()
    print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    # --- Scaling laws for training horizon ---
    num_scaling_params = param_counts['transformer_matrices'] + param_counts['lm_head']
    target_ratio = train_cfg.get("target_param_data_ratio", -1)
    seq_len = config.sequence_len

    # Batch size
    device_batch_size = train_cfg.get("device_batch_size", 32)
    tokens_per_fwdbwd = device_batch_size * seq_len
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size

    target_tokens = int(target_ratio * num_scaling_params) if target_ratio > 0 else int(train_cfg.get("max_tokens", 5e9))

    # d12 reference model (nanochat default: 768 dim, 32K vocab, aspect_ratio=64)
    # Used for batch size auto-computation and weight decay scaling
    B_REF = 2**19  # 524288 tokens (d12 optimal batch size)
    d12_scaling_params = 12 * (768**2) * 12 + 768 * 32768
    D_REF = target_ratio * d12_scaling_params if target_ratio > 0 else d12_scaling_params * 10.5

    total_batch_size = train_cfg.get("total_batch_size", -1)
    if total_batch_size == -1:
        # Auto-compute via Power Lines scaling (B_opt ∝ D^0.383)
        batch_ratio = target_tokens / D_REF
        predicted = B_REF * batch_ratio ** 0.383
        total_batch_size = 2 ** round(math.log2(predicted))
        print0(f"Auto-computed batch size: {total_batch_size:,} tokens (ratio={batch_ratio:.2f})")

    assert total_batch_size % world_tokens_per_fwdbwd == 0, (
        f"total_batch_size={total_batch_size} must be divisible by "
        f"world_tokens_per_fwdbwd={world_tokens_per_fwdbwd}"
    )
    grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd

    # Training iterations
    num_iterations = train_cfg.get("num_iterations", -1)
    if num_iterations == -1:
        num_iterations = target_tokens // total_batch_size
    total_tokens = total_batch_size * num_iterations

    print0(f"Training: {num_iterations:,} steps, {total_tokens:,} tokens")
    print0(f"Batch: {total_batch_size:,} tokens, {grad_accum_steps} accum steps")
    print0(f"Data:Param ratio: {total_tokens / num_scaling_params:.2f}")

    # --- LR scaling ---
    batch_lr_scale = (total_batch_size / (2**19)) ** 0.5
    if batch_lr_scale != 1.0:
        print0(f"Batch LR scale: {batch_lr_scale:.4f}")

    # Weight decay scaling (T_epoch framework, https://arxiv.org/abs/2405.13698)
    # λ = λ_ref · √(B/B_ref) · (D_ref/D) where D_ref is d12 reference token budget
    base_wd = optim_cfg.get("weight_decay", 0.28)
    weight_decay_scaled = base_wd * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
    print0(f"Weight decay: {base_wd:.4f} → {weight_decay_scaled:.6f} (D_REF={int(D_REF):,}, D={target_tokens:,})")

    # --- Optimizer ---
    # width_lr_exponent: -0.5 = nanochat default (∝1/√width), -1.0 = µP (∝1/width)
    # µP makes hyperparameters transfer across model widths by construction,
    # reducing sensitivity to model size and making the comparison more robust.
    width_lr_exponent = optim_cfg.get("width_lr_exponent", -1.0)
    optimizer = model.setup_optimizer(
        unembedding_lr=optim_cfg.get("unembedding_lr", 0.008) * batch_lr_scale,
        embedding_lr=optim_cfg.get("embedding_lr", 0.3) * batch_lr_scale,
        scalar_lr=optim_cfg.get("scalar_lr", 0.5) * batch_lr_scale,
        matrix_lr=optim_cfg.get("matrix_lr", 0.02) * batch_lr_scale,
        weight_decay=weight_decay_scaled,
        width_lr_exponent=width_lr_exponent,
    )

    if resuming:
        optimizer.load_state_dict(optimizer_data)
        del optimizer_data

    # --- Dataloader ---
    dataloader_resume = None if not resuming else meta_data.get("dataloader_state_dict")
    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, device_batch_size, seq_len, split="train",
        device=device, resume_state_dict=dataloader_resume,
    )
    build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, device_batch_size, seq_len, split="val", device=device,
    )
    x, y, dataloader_state_dict = next(train_loader)

    # --- LR / momentum / WD schedules ---
    warmup_steps = train_cfg.get("warmup_steps", 40)
    warmdown_ratio = train_cfg.get("warmdown_ratio", 0.65)
    final_lr_frac = train_cfg.get("final_lr_frac", 0.05)

    def get_lr_multiplier(it):
        warmdown_iters = round(warmdown_ratio * num_iterations)
        if it < warmup_steps:
            return (it + 1) / warmup_steps
        elif it <= num_iterations - warmdown_iters:
            return 1.0
        else:
            progress = (num_iterations - it) / warmdown_iters
            return progress * 1.0 + (1 - progress) * final_lr_frac

    def get_muon_momentum(it):
        warmdown_iters = round(warmdown_ratio * num_iterations)
        warmdown_start = num_iterations - warmdown_iters
        if it < 400:
            frac = it / 400
            return (1 - frac) * 0.85 + frac * 0.97
        elif it >= warmdown_start:
            progress = (it - warmdown_start) / warmdown_iters
            return 0.97 * (1 - progress) + 0.90 * progress
        else:
            return 0.97

    def get_weight_decay(it):
        return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

    # --- Training loop ---
    step = 0 if not resuming else meta_data["step"]
    val_bpb = None
    min_val_bpb = float("inf")
    smooth_train_loss = 0.0
    total_training_time = 0.0
    total_bytes_consumed = 0  # for cross-tokenizer normalization
    avg_bytes_per_token = token_bytes.float().mean().item()  # cache for byte tracking

    if resuming:
        loop_state = meta_data.get("loop_state", {})
        min_val_bpb = loop_state.get("min_val_bpb", float("inf"))
        smooth_train_loss = loop_state.get("smooth_train_loss", 0.0)
        total_training_time = loop_state.get("total_training_time", 0.0)
        total_bytes_consumed = loop_state.get("total_bytes_consumed", 0)

    eval_every = log_cfg.get("eval_every", 250)
    save_every = log_cfg.get("save_every", -1)

    print0(f"\n{'='*60}")
    print0(f"Starting training: {num_iterations} steps")
    print0(f"{'='*60}\n")

    while True:
        last_step = step == num_iterations

        # --- Validation ---
        if eval_every > 0 and (last_step or step % eval_every == 0):
            model.eval()
            val_loader = build_val_loader()
            eval_steps = train_cfg.get("eval_tokens", 80 * 524288) // (device_batch_size * seq_len * ddp_world_size)
            eval_steps = max(1, eval_steps)  # ensure at least 1 step
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
            print0(f"Step {step:05d} | val bpb: {val_bpb:.6f}")
            if val_bpb < min_val_bpb:
                min_val_bpb = val_bpb
            wandb_run.log({"step": step, "val/bpb": val_bpb, "total_training_time": total_training_time})
            model.train()

        # --- Checkpointing ---
        if last_step or (step > 0 and step != args.resume_from_step and save_every > 0 and step % save_every == 0):
            save_checkpoint(
                checkpoint_dir, step,
                orig_model.state_dict(),
                optimizer.state_dict(),
                {
                    "step": step,
                    "val_bpb": val_bpb,
                    "model_config": config_kwargs,
                    "tokenizer_path": args.tokenizer,
                    "training_config": cfg,
                    "total_batch_size": total_batch_size,
                    "device_batch_size": device_batch_size,
                    "max_seq_len": seq_len,
                    "dataloader_state_dict": dataloader_state_dict,
                    "loop_state": {
                        "min_val_bpb": min_val_bpb,
                        "smooth_train_loss": smooth_train_loss,
                        "total_training_time": total_training_time,
                        "total_bytes_consumed": total_bytes_consumed,
                    },
                },
                rank=ddp_rank,
            )
            # Prune old checkpoints, keeping only the most recent max_keep.
            # Only rank 0 prunes (other ranks' optimizer files are cleaned by glob).
            # Barrier ensures all ranks have finished writing before any deletion.
            max_keep = log_cfg.get("max_checkpoints", 0)
            if max_keep > 0 and not last_step:
                synchronize()
                if ddp_rank == 0:
                    _prune_checkpoints(checkpoint_dir, max_keep, ddp_rank)
                synchronize()

        if last_step:
            break

        # --- Training step ---
        synchronize()
        t0 = time.time()
        for micro_step in range(grad_accum_steps):
            loss = model(x, y)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            x, y, dataloader_state_dict = next(train_loader)

        # Update LR, momentum, weight decay
        lrm = get_lr_multiplier(step)
        muon_momentum = get_muon_momentum(step)
        muon_wd = get_weight_decay(step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group['kind'] == 'muon':
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_wd

        optimizer.step()
        model.zero_grad(set_to_none=True)

        train_loss_f = train_loss.item()
        synchronize()
        t1 = time.time()
        dt = t1 - t0

        # --- Logging ---
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased = smooth_train_loss / (1 - ema_beta ** (step + 1))

        tok_per_sec = int(total_batch_size / dt)
        flops_per_sec = num_flops_per_token * total_batch_size / dt
        mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
        total_bytes_consumed += int(total_batch_size * avg_bytes_per_token)

        if step > 10:
            total_training_time += dt

        pct = 100 * step / num_iterations
        print0(
            f"step {step:05d}/{num_iterations:05d} ({pct:.1f}%) | "
            f"loss: {debiased:.4f} | lrm: {lrm:.2f} | "
            f"dt: {dt*1000:.0f}ms | tok/s: {tok_per_sec:,} | "
            f"mfu: {mfu:.1f}% | bytes: {total_bytes_consumed/1e9:.2f}B"
        )

        if step % 100 == 0:
            wandb_run.log({
                "step": step,
                "total_training_time": total_training_time,
                "train/loss": debiased,
                "train/lrm": lrm,
                "train/tok_per_sec": tok_per_sec,
                "train/mfu": mfu,
                "train/bytes_consumed": total_bytes_consumed,
                "train/dt": dt,
            })

        # GC management (from nanochat)
        first_step = (step == 0) or (resuming and step == args.resume_from_step)
        step += 1
        if first_step:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif step % 5000 == 0:
            gc.collect()

    # --- Cleanup ---
    print0(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024 / 1024:.0f} MiB")
    print0(f"Total training time: {total_training_time/60:.1f}m")
    if val_bpb is not None:
        print0(f"Min val bpb: {min_val_bpb:.6f}")
    print0(f"Total bytes consumed: {total_bytes_consumed/1e9:.2f}B")

    # --- FLORES-200 evaluation (rank 0 only) ---
    if master_process and train_cfg.get("flores_eval", True):
        print0(f"\n{'='*60}")
        print0("Running FLORES-200 BPB evaluation...")
        print0(f"{'='*60}\n")
        try:
            from eval_runner import compute_flores_perplexity, LANGUAGE_FAMILIES
            orig_model.eval()
            flores_results = compute_flores_perplexity(
                orig_model, tokenizer, device=device, max_samples=200,
            )

            # Per-language BPB
            lang_bpbs = {lang: data["bpb"]
                         for lang, data in flores_results.get("per_language", {}).items()}

            if lang_bpbs:
                bpb_values = list(lang_bpbs.values())
                mean_bpb = sum(bpb_values) / len(bpb_values)
                variance = sum((b - mean_bpb) ** 2 for b in bpb_values) / len(bpb_values)
                std_bpb = variance ** 0.5
                min_bpb = min(bpb_values)
                max_bpb = max(bpb_values)

                print0(f"\nFLORES-200 BPB summary ({len(lang_bpbs)} languages):")
                print0(f"  Mean:     {mean_bpb:.4f}")
                print0(f"  Std:      {std_bpb:.4f}")
                print0(f"  Min:      {min_bpb:.4f}")
                print0(f"  Max:      {max_bpb:.4f}")
                print0(f"  Range:    {max_bpb - min_bpb:.4f}")
                print0(f"  CV:       {std_bpb / mean_bpb:.4f}")

                # Per-family summary
                print0(f"\nPer-family BPB:")
                for family, fdata in sorted(
                    flores_results.get("per_family", {}).items(),
                    key=lambda x: x[1]["mean_bpb"],
                ):
                    print0(f"  {family:<20} {fdata['mean_bpb']:.4f} "
                           f"({fdata['num_languages']} langs)")

                # Top 10 best and worst languages
                sorted_langs = sorted(lang_bpbs.items(), key=lambda x: x[1])
                print0(f"\nBest 10 languages:")
                for lang, bpb in sorted_langs[:10]:
                    print0(f"  {lang:<12} {bpb:.4f}")
                print0(f"\nWorst 10 languages:")
                for lang, bpb in sorted_langs[-10:]:
                    print0(f"  {lang:<12} {bpb:.4f}")

                # Log to W&B
                wandb_log = {
                    "flores/mean_bpb": mean_bpb,
                    "flores/std_bpb": std_bpb,
                    "flores/min_bpb": min_bpb,
                    "flores/max_bpb": max_bpb,
                    "flores/cv_bpb": std_bpb / mean_bpb,
                    "flores/num_languages": len(lang_bpbs),
                }
                # Log per-family
                for family, fdata in flores_results.get("per_family", {}).items():
                    safe_family = family.replace("-", "_").replace(" ", "_")
                    wandb_log[f"flores/family/{safe_family}"] = fdata["mean_bpb"]
                # Log per-language (top-level languages only to avoid clutter)
                for lang, bpb in lang_bpbs.items():
                    wandb_log[f"flores/lang/{lang}"] = bpb
                wandb_run.log(wandb_log)

                # Save results to checkpoint dir
                flores_path = os.path.join(checkpoint_dir, "flores_results.json")
                import json as json_mod
                with open(flores_path, "w") as f:
                    json_mod.dump({
                        "summary": {
                            "mean_bpb": mean_bpb,
                            "std_bpb": std_bpb,
                            "min_bpb": min_bpb,
                            "max_bpb": max_bpb,
                            "cv_bpb": std_bpb / mean_bpb,
                            "num_languages": len(lang_bpbs),
                        },
                        "per_language": lang_bpbs,
                        "per_family": flores_results.get("per_family", {}),
                    }, f, indent=2)
                print0(f"\nFLORES results saved to {flores_path}")
        except Exception as e:
            print0(f"WARNING: FLORES evaluation failed: {e}")
            import traceback
            traceback.print_exc()

    wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
