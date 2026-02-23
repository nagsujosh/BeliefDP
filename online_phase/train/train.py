"""Main training script for online phase inference model.

Usage:
    python -m online_phase.train.train --config online_phase/configs/train_structured.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from online_phase.utils.io import load_yaml, merge_configs, set_seed, resolve_path
from online_phase.data.labels import generate_labels, save_labels, load_cached_labels
from online_phase.data.dataset import PhaseDataset, create_splits, compute_normalization_stats
from online_phase.models.phase_model_structured import PhaseModelStructured
from online_phase.losses.multitask_weighting_structured import MultiTaskLossStructured
from online_phase.train.callbacks import CheckpointManager, MetricsLogger


def build_config(config_path: str) -> dict:
    """Load and merge train config with model config."""
    config = load_yaml(config_path)
    model_config_path = config.get("model_config")
    if model_config_path:
        model_config = load_yaml(resolve_path(model_config_path))
        config = merge_configs(config, model_config)
    return config


def get_or_generate_labels(config: dict) -> dict:
    """Load cached labels or generate from scratch."""
    cache_dir = resolve_path(config.get("label_cache_dir", "online_phase/cache/labels"))

    if cache_dir.exists() and any(cache_dir.glob("demo_*.npz")):
        print(f"Loading cached labels from {cache_dir}")
        return load_cached_labels(cache_dir)

    print("Generating labels from phase_segmentation.json...")
    seg_path = resolve_path(config["segmentation_json"])
    emb_dir = resolve_path(config["embeddings_dir"])
    labels = generate_labels(
        seg_json_path=seg_path,
        embeddings_dir=emb_dir,
        sigma_b=config.get("sigma_b", 4.0),
        c_min=config.get("c_min", 0.3),
        c_max=config.get("c_max", 1.0),
        anchor_sigma=config.get("anchor_sigma", 6.0),
        camera_key=config.get("camera_keys", ["agentview_rgb"])[0],
        include_tail_boundary=config.get("include_tail_boundary", False),
        tail_sigma=config.get("tail_sigma", 2.0),
    )
    print(f"Generated labels for {len(labels)} demos. Caching to {cache_dir}")
    save_labels(labels, cache_dir)
    return labels


def train_epoch(
    model: torch.nn.Module,
    criterion: MultiTaskLossStructured,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    log_interval: int = 10,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    criterion.train()

    running = {}
    n_batches = 0
    
    # DEBUG: Track HMM log_normalizer magnitude for structured models
    log_norm_sum = 0.0
    log_norm_count = 0

    for i, batch in enumerate(loader):
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(batch["embedding_window"], batch["attention_mask"])
        total_loss, details = criterion(outputs, batch)
        total_loss.backward()
        
        # DEBUG: Accumulate log_normalizer stats for structured models
        if "log_normalizers" in outputs:
            mask = batch["attention_mask"]
            valid_log_norms = outputs["log_normalizers"][mask.bool()]
            if len(valid_log_norms) > 0:
                log_norm_sum += valid_log_norms.sum().item()
                log_norm_count += len(valid_log_norms)

        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(criterion.parameters()),
            max_norm=grad_clip,
        )
        
        # Compute global gradient norm (before clipping would be better, but after is still useful)
        total_norm = 0.0
        for p in list(model.parameters()) + list(criterion.parameters()):
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        optimizer.step()

        # Accumulate metrics
        for k, v in details.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            running[k] = running.get(k, 0) + v
        running["grad_norm"] = running.get("grad_norm", 0) + total_norm
        n_batches += 1

        if log_interval > 0 and (i + 1) % log_interval == 0:
            avg_loss = running.get("total", 0) / n_batches
            pct = 100 * (i + 1) / len(loader)
            print(f"    [{pct:5.1f}%] batch {i+1}/{len(loader)} loss={avg_loss:.4f}    ", end="\r")

    # Average
    metrics = {k: v / max(n_batches, 1) for k, v in running.items()}
    
    # DEBUG: Add mean log_normalizer to metrics
    if log_norm_count > 0:
        metrics["debug_mean_log_norm"] = log_norm_sum / log_norm_count
    
    return metrics


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Validate and compute metrics.

    For structured models, reports both emission argmax and HMM belief
    argmax accuracy so we can measure the temporal filtering benefit.
    """
    model.eval()
    criterion.eval()

    running = {}
    n_batches = 0
    # Emission (raw) phase accuracy
    raw_correct = 0
    total_frames = 0
    progress_abs_err = 0.0
    boundary_tp = 0
    boundary_fp = 0
    boundary_fn = 0
    # HMM belief phase accuracy (structured only)
    belief_correct = 0
    is_structured = hasattr(model, "hmm")
    # Emission confidence tracking
    emission_confidence_sum = 0.0
    emission_confidence_count = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch["embedding_window"], batch["attention_mask"])
        _, details = criterion(outputs, batch)

        for k, v in details.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            running[k] = running.get(k, 0) + v
        n_batches += 1

        mask = batch["attention_mask"][:, -1]  # (B,)

        # Emission phase accuracy (last frame in window)
        z_logits_last = outputs["z_logits"][:, -1]  # (B, num_phases)
        z_pred = z_logits_last.argmax(dim=-1)  # (B,)
        z_true = batch["z"][:, -1]
        raw_correct += ((z_pred == z_true) * mask).sum().item()
        total_frames += mask.sum().item()
        
        # Emission confidence: mean max softmax probability (measures model certainty)
        z_probs = torch.softmax(z_logits_last, dim=-1)  # (B, num_phases)
        max_probs = z_probs.max(dim=-1).values  # (B,)
        emission_confidence_sum += (max_probs * mask).sum().item()
        emission_confidence_count += mask.sum().item()

        # HMM belief accuracy (structured model)
        if is_structured and "beliefs" in outputs:
            belief_pred = outputs["beliefs"][:, -1].argmax(dim=-1)  # (B,)
            belief_correct += ((belief_pred == z_true) * mask).sum().item()

        # Progress MAE (last frame)
        p_pred = outputs["progress"][:, -1]
        p_true = batch["p"][:, -1]
        progress_abs_err += ((p_pred - p_true).abs() * mask).sum().item()

        # Boundary detection (last frame, threshold=0.5)
        b_pred = (outputs["boundary"][:, -1] > 0.5).float()
        b_true = (batch["b"][:, -1] > 0.5).float()
        boundary_tp += (b_pred * b_true * mask).sum().item()
        boundary_fp += (b_pred * (1 - b_true) * mask).sum().item()
        boundary_fn += ((1 - b_pred) * b_true * mask).sum().item()

    metrics = {k: v / max(n_batches, 1) for k, v in running.items()}
    metrics["phase_acc"] = raw_correct / max(total_frames, 1)
    metrics["progress_mae"] = progress_abs_err / max(total_frames, 1)
    metrics["emission_confidence"] = emission_confidence_sum / max(emission_confidence_count, 1)

    if is_structured:
        metrics["belief_acc"] = belief_correct / max(total_frames, 1)
        improvement = metrics["belief_acc"] - metrics["phase_acc"]
        metrics["hmm_improvement"] = improvement

    bnd_precision = boundary_tp / max(boundary_tp + boundary_fp, 1)
    bnd_recall = boundary_tp / max(boundary_tp + boundary_fn, 1)
    metrics["boundary_precision"] = bnd_precision
    metrics["boundary_recall"] = bnd_recall
    metrics["boundary_f1"] = (
        2 * bnd_precision * bnd_recall / max(bnd_precision + bnd_recall, 1e-8)
    )
    return metrics


def main(config_path: str) -> None:
    config = build_config(config_path)
    set_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Labels ---
    labels = get_or_generate_labels(config)
    all_demo_ids = sorted(labels.keys(), key=lambda s: int(s.split("_")[-1]))
    print(f"Total demos: {len(all_demo_ids)}")

    # --- Splits ---
    train_ids, val_ids, test_ids = create_splits(
        all_demo_ids,
        train_frac=config.get("train_frac", 0.75),
        val_frac=config.get("val_frac", 0.15),
        test_frac=config.get("test_frac", 0.10),
        seed=config.get("split_seed", 42),
    )
    print(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

    # --- Normalization ---
    emb_dir = resolve_path(config["embeddings_dir"])
    camera_key = config.get("camera_keys", ["agentview_rgb"])[0]
    norm_stats = None
    mean, std = None, None

    if config.get("normalize_embeddings", True):
        print("Computing embedding normalization stats from training set...")
        mean, std = compute_normalization_stats(emb_dir, train_ids, camera_key)
        norm_stats = {"mean": mean, "std": std}
        print(f"  mean norm: {np.linalg.norm(mean):.2f}, std mean: {std.mean():.4f}")

    # Auto-detect input_dim from actual embedding files (handles multi-view fused)
    sample_emb_path = emb_dir / f"{train_ids[0]}_{camera_key}.npy"
    if sample_emb_path.exists():
        actual_dim = np.load(sample_emb_path).shape[1]
        if actual_dim != config.get("input_dim", 2048):
            print(f"  Auto-detected input_dim={actual_dim} (config had {config.get('input_dim', 2048)})")
            config["input_dim"] = actual_dim

    # --- Datasets ---
    window_size = config.get("window_size", 48)
    stride = config.get("stride", 1)
    aug_config = config.get("augmentation") if config.get("augmentation") else None
    per_demo_normalize = config.get("per_demo_normalize", False)

    train_dataset = PhaseDataset(
        emb_dir, labels, window_size=window_size, stride=stride,
        demo_ids=train_ids, camera_key=camera_key,
        mean=mean, std=std, augmentation=aug_config,
        per_demo_normalize=per_demo_normalize,
    )
    val_dataset = PhaseDataset(
        emb_dir, labels, window_size=window_size, stride=stride,
        demo_ids=val_ids, camera_key=camera_key,
        mean=mean, std=std, augmentation=None,  # No augmentation for val
        per_demo_normalize=per_demo_normalize,
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 4)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # --- Model + Loss + Optimizer ---
    requested_model_type = config.get("model_type")
    if requested_model_type not in (None, "structured"):
        raise ValueError(
            f"Unsupported model_type='{requested_model_type}'. "
            "This repository is structured-only; use PhaseModelStructured."
        )

    print("\n>>> Using STRUCTURED model (Transformer + Neural HMM)")
    model = PhaseModelStructured(config).to(device)
    criterion = MultiTaskLossStructured(config).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=config.get("lr", 3e-4),
        weight_decay=config.get("weight_decay", 0.01),
    )

    epochs = config.get("epochs", 50)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- Callbacks ---
    output_dir = resolve_path(config.get("output_dir", "online_phase/runs/structured"))
    ckpt_manager = CheckpointManager(output_dir, config)
    logger = MetricsLogger(output_dir)

    # Save test split info for later evaluation
    from online_phase.utils.io import save_json
    save_json(
        {"train": train_ids, "val": val_ids, "test": test_ids},
        output_dir / "splits.json",
    )

    # --- Training Loop ---
    early_stop_patience = config.get("early_stop_patience", 10)
    epochs_without_improvement = 0

    # Print training config summary
    print(f"\n{'='*60}")
    print(f"  Training Configuration Summary")
    print(f"{'='*60}")
    print("  Model type:  structured")
    print(f"  Model:       {config.get('backbone_type', 'transformer')} | "
          f"d={config.get('d_model', 256)} | "
          f"layers={config.get('num_layers', 4)} | "
          f"ff={config.get('dim_feedforward', 512)}")
    print(f"  Params:      {param_count:,}")
    print(f"  Dropout:     {config.get('dropout', 0.1)} | "
          f"weight_decay={config.get('weight_decay', 0.01)}")
    print(f"  Data:        {len(train_dataset)} train / {len(val_dataset)} val samples | "
          f"stride={stride} | window={window_size}")
    print(f"  Augment:     noise={aug_config.get('noise_std', 0) if aug_config else 0} | "
          f"frame_mask={aug_config.get('frame_mask_prob', 0) if aug_config else 0}")
    print(f"  Normalize:   global={config.get('normalize_embeddings', True)} | "
          f"per_demo={per_demo_normalize}")
    print(f"  Schedule:    {epochs} epochs | lr={config.get('lr', 3e-4)} | "
          f"early_stop={early_stop_patience}")
    print(f"  Label smooth: {config.get('label_smoothing_max', 0.1)} | "
          f"boundary_pos_weight={config.get('boundary_pos_weight', 6.0)}")
    print(f"  HMM:         eta=[{config.get('hmm_eta_min', 0.02)}, "
          f"{config.get('hmm_eta_max', 0.40)}] | "
          f"rho={config.get('hmm_rho', 0.001)} | "
          f"w_hmm={config.get('w_hmm', 0.1)} | "
          f"warmup={config.get('hmm_warmup_epochs', 5)} epochs")
    print(f"{'='*60}")
    print(f"\n  * = new best checkpoint saved")
    for epoch in range(1, epochs + 1):
        logger.start_epoch()

        # Update epoch-dependent weights (HMM warm-up)
        if hasattr(criterion, 'set_epoch'):
            criterion.set_epoch(epoch - 1)  # 0-indexed

        train_metrics = train_epoch(
            model, criterion, train_loader, optimizer, device,
            grad_clip=config.get("grad_clip_norm", 1.0),
            log_interval=config.get("log_interval", 10),
        )

        val_metrics = validate(model, criterion, val_loader, device)
        scheduler.step()

        # Checkpoint
        extra = {"lr": scheduler.get_last_lr()[0]}
        val_loss = val_metrics.get("total", float("inf"))
        saved = ckpt_manager.save(
            model, criterion, optimizer, epoch,
            val_loss=val_loss,
            norm_stats=norm_stats,
        )
        if saved:
            extra["saved_best"] = saved
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        logger.log_epoch(epoch, train_metrics, val_metrics, extra)

        # Early stopping
        if epochs_without_improvement >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
            break

    print(f"\nTraining complete. Best val loss: {ckpt_manager.best_val_loss:.4f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train online phase inference model")
    parser.add_argument(
        "--config", type=str,
        default="online_phase/configs/train_structured.yaml",
        help="Path to training config YAML",
    )
    args = parser.parse_args()
    main(args.config)
