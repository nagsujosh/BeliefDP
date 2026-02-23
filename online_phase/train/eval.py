"""Offline evaluation script for trained structured phase inference model.

Usage:
    python -m online_phase.train.eval --ckpt online_phase/runs/structured/checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from online_phase.utils.io import load_json, save_json, resolve_path, set_seed
from online_phase.data.labels import generate_labels, load_cached_labels
from online_phase.models.phase_model_structured import PhaseModelStructured
from online_phase.train.callbacks import CheckpointManager
from online_phase.filter.postproc import detect_boundary_events
from online_phase.utils.metrics import (
    phase_accuracy, phase_macro_f1, phase_per_class_f1,
    progress_mae, progress_rmse, boundary_f1,
    count_illegal_transitions, count_phase_jitter,
)


def evaluate_dataset(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    K: int,
) -> dict:
    """Evaluate model on a dataset split (frame-level, no temporal filtering)."""
    model.eval()

    all_z_pred = []
    all_z_true = []
    all_p_pred = []
    all_p_true = []
    all_b_pred = []
    all_b_true = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["embedding_window"], batch["attention_mask"])

            # Last frame predictions
            z_pred = outputs["z_logits"][:, -1].argmax(dim=-1).cpu().numpy()
            z_true = batch["z"][:, -1].cpu().numpy()
            p_pred = outputs["progress"][:, -1].cpu().numpy()
            p_true = batch["p"][:, -1].cpu().numpy()
            b_pred = outputs["boundary"][:, -1].cpu().numpy()
            b_true = batch["b"][:, -1].cpu().numpy()
            mask = batch["attention_mask"][:, -1].cpu().numpy()

            # Only include valid frames
            valid = mask > 0.5
            all_z_pred.extend(z_pred[valid])
            all_z_true.extend(z_true[valid])
            all_p_pred.extend(p_pred[valid])
            all_p_true.extend(p_true[valid])
            all_b_pred.extend(b_pred[valid])
            all_b_true.extend(b_true[valid])

    z_pred = np.array(all_z_pred)
    z_true = np.array(all_z_true)
    p_pred = np.array(all_p_pred)
    p_true = np.array(all_p_true)

    results = {
        "phase_accuracy": phase_accuracy(z_pred, z_true),
        "phase_macro_f1": phase_macro_f1(z_pred, z_true),
        "phase_per_class_f1": phase_per_class_f1(z_pred, z_true, K),
        "progress_mae": progress_mae(p_pred, p_true),
        "progress_rmse": progress_rmse(p_pred, p_true),
        "num_frames": len(z_pred),
    }
    return results


def evaluate_demo(
    model: torch.nn.Module,
    embeddings: np.ndarray,
    labels: dict[str, np.ndarray],
    window_size: int,
    device: torch.device,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    boundary_threshold: float = 0.75,
    per_demo_normalize: bool = False,
) -> dict:
    """Evaluate a single demo using raw emissions and structured beliefs."""
    model.eval()

    T = len(embeddings)
    emb = embeddings.astype(np.float32)

    # Per-demo normalization: subtract demo mean to fix domain shift
    if per_demo_normalize:
        demo_mean = emb.mean(axis=0)
        emb = emb - demo_mean

    if mean is not None and std is not None:
        emb = (emb - mean) / std

    # Raw model predictions (emission argmax, no filtering)
    z_raw_preds = []
    # Structured beliefs (HMM output)
    z_filt_preds = []
    p_preds = []
    b_preds = []

    with torch.no_grad():
        for t in range(T):
            # Build window
            start = max(0, t - window_size + 1)
            window = emb[start: t + 1]
            pad_len = window_size - len(window)
            if pad_len > 0:
                window = np.vstack([np.zeros((pad_len, emb.shape[1]), dtype=np.float32), window])
            mask = np.zeros(window_size, dtype=np.float32)
            mask[pad_len:] = 1.0

            # Forward pass
            x = torch.from_numpy(window).unsqueeze(0).to(device)
            m = torch.from_numpy(mask).unsqueeze(0).to(device)
            out = model(x, m)

            z_logits = out["z_logits"][0, -1].cpu().numpy()
            b_prob = out["boundary"][0, -1].item()
            p_val = out["progress"][0, -1].item()

            # Raw emission argmax prediction
            z_raw_preds.append(int(np.argmax(z_logits)))

            # Structured belief argmax prediction
            if "beliefs" not in out:
                raise RuntimeError("Structured model output is missing 'beliefs'.")
            belief = out["beliefs"][0, -1].cpu().numpy()
            z_filt_preds.append(int(np.argmax(belief)))

            p_preds.append(p_val)
            b_preds.append(b_prob)

    z_raw = np.array(z_raw_preds)
    z_filt = np.array(z_filt_preds)
    z_true = labels["z"]
    p_pred = np.array(p_preds)
    p_true = labels["p"]

    # Boundary events (use configurable threshold)
    b_pred_events = detect_boundary_events(
        np.array(b_preds), threshold=boundary_threshold, refractory=10
    )
    boundaries = labels.get("boundaries", [])
    if len(boundaries) == 0:
        z_diff = np.diff(z_true)
        boundaries = list(np.where(z_diff != 0)[0] + 1)

    bnd_metrics = boundary_f1(b_pred_events, boundaries, tolerance=5)

    return {
        # Raw model metrics (argmax of emission logits)
        "raw_phase_accuracy": phase_accuracy(z_raw, z_true),
        "raw_phase_macro_f1": phase_macro_f1(z_raw, z_true),
        "raw_illegal_transitions": count_illegal_transitions(z_raw),
        "raw_jitter": count_phase_jitter(z_raw),
        # Structured belief metrics
        "filt_phase_accuracy": phase_accuracy(z_filt, z_true),
        "filt_phase_macro_f1": phase_macro_f1(z_filt, z_true),
        "filt_illegal_transitions": count_illegal_transitions(z_filt),
        "filt_jitter": count_phase_jitter(z_filt),
        # Shared metrics (progress + boundary heads)
        "progress_mae": progress_mae(p_pred, p_true),
        "progress_rmse": progress_rmse(p_pred, p_true),
        "boundary_precision": bnd_metrics["precision"],
        "boundary_recall": bnd_metrics["recall"],
        "boundary_f1": bnd_metrics["f1"],
        # Arrays for further analysis
        "z_raw": z_raw,
        "z_filt": z_filt,
        "p_pred": p_pred,
        "b_pred": np.array(b_preds),
    }


def main(ckpt_path: str) -> None:
    ckpt = CheckpointManager.load(ckpt_path)
    config = ckpt["config"]
    set_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    requested_model_type = config.get("model_type")
    if requested_model_type not in (None, "structured"):
        raise ValueError(
            f"Unsupported model_type='{requested_model_type}'. "
            "This repository is structured-only; use PhaseModelStructured."
        )

    print(">>> Evaluating STRUCTURED model (Transformer + Neural HMM)")
    model = PhaseModelStructured(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    K = config.get("num_phases", 11)
    window_size = config.get("window_size", 48)

    # Load normalization stats
    norm_stats = ckpt.get("norm_stats")
    mean = norm_stats["mean"] if norm_stats else None
    std = norm_stats["std"] if norm_stats else None

    # Load labels
    labels = get_labels(config)

    # Load splits
    output_dir = resolve_path(config.get("output_dir", "online_phase/runs/structured"))
    splits = load_json(output_dir / "splits.json")
    test_ids = splits["test"]
    print(f"Evaluating on {len(test_ids)} test demos")

    boundary_threshold = config.get("eval_boundary_threshold", 0.75)
    per_demo_normalize = config.get("per_demo_normalize", False)

    # Evaluate each test demo with structured belief output
    emb_dir = resolve_path(config["embeddings_dir"])
    camera_key = config.get("camera_keys", ["agentview_rgb"])[0]
    all_results = {}

    for demo_id in test_ids:
        emb_path = emb_dir / f"{demo_id}_{camera_key}.npy"
        if not emb_path.exists() or demo_id not in labels:
            continue

        embeddings = np.load(emb_path)

        result = evaluate_demo(
            model, embeddings, labels[demo_id],
            window_size, device, mean, std,
            boundary_threshold=boundary_threshold,
            per_demo_normalize=per_demo_normalize,
        )
        # Remove large arrays for summary
        summary = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
        all_results[demo_id] = summary
        print(
            f"  {demo_id}: "
            f"raw_acc={summary['raw_phase_accuracy']:.3f} "
            f"filt_acc={summary['filt_phase_accuracy']:.3f} "
            f"bnd_f1={summary['boundary_f1']:.3f} "
            f"raw_jitter={summary['raw_jitter']:.1f} "
            f"filt_jitter={summary['filt_jitter']:.1f}"
        )

    # Aggregate
    if all_results:
        agg = {}
        keys = [
            # Raw model metrics
            "raw_phase_accuracy", "raw_phase_macro_f1",
            "raw_illegal_transitions", "raw_jitter",
            # Structured belief metrics
            "filt_phase_accuracy", "filt_phase_macro_f1",
            "filt_illegal_transitions", "filt_jitter",
            # Shared metrics
            "progress_mae", "progress_rmse",
            "boundary_precision", "boundary_recall", "boundary_f1",
        ]
        for k in keys:
            vals = [r[k] for r in all_results.values()]
            agg[k] = {"mean": np.mean(vals), "std": np.std(vals)}

        print("\n=== Aggregate Test Results ===")
        print("  --- Raw Model (argmax) ---")
        for k in ["raw_phase_accuracy", "raw_phase_macro_f1",
                   "raw_illegal_transitions", "raw_jitter"]:
            print(f"    {k}: {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}")
        print("  --- Structured Belief ---")
        for k in ["filt_phase_accuracy", "filt_phase_macro_f1",
                   "filt_illegal_transitions", "filt_jitter"]:
            print(f"    {k}: {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}")
        print("  --- Shared (progress + boundary) ---")
        for k in ["progress_mae", "progress_rmse",
                   "boundary_precision", "boundary_recall", "boundary_f1"]:
            print(f"    {k}: {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}")

        # Structured belief improvement summary
        raw_acc = agg["raw_phase_accuracy"]["mean"]
        filt_acc = agg["filt_phase_accuracy"]["mean"]
        raw_jit = agg["raw_jitter"]["mean"]
        filt_jit = agg["filt_jitter"]["mean"]
        print(f"\n  HMM belief effect: acc {raw_acc:.3f}→{filt_acc:.3f} "
              f"({filt_acc - raw_acc:+.3f}), "
              f"jitter {raw_jit:.2f}→{filt_jit:.2f} "
              f"({filt_jit - raw_jit:+.2f})")

        save_json(
            {"per_demo": all_results, "aggregate": agg},
            output_dir / "eval_results.json",
        )
        print(f"\nResults saved to {output_dir / 'eval_results.json'}")


def get_labels(config: dict) -> dict:
    cache_dir = resolve_path(config.get("label_cache_dir", "online_phase/cache/labels"))
    if cache_dir.exists() and any(cache_dir.glob("demo_*.npz")):
        return load_cached_labels(cache_dir)
    seg_path = resolve_path(config["segmentation_json"])
    emb_dir = resolve_path(config["embeddings_dir"])
    return generate_labels(
        seg_json_path=seg_path, embeddings_dir=emb_dir,
        sigma_b=config.get("sigma_b", 4.0),
        camera_key=config.get("camera_keys", ["agentview_rgb"])[0],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained phase inference model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt file")
    args = parser.parse_args()
    main(args.ckpt)
