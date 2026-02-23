"""Simulate online streaming inference through a demo.

Usage:
    python -m online_phase.infer.stream_demo --ckpt path.pt --demo demo_0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from online_phase.utils.io import resolve_path, set_seed
from online_phase.data.labels import generate_labels
from online_phase.models.phase_model_structured import PhaseModelStructured
from online_phase.train.callbacks import CheckpointManager
from online_phase.filter.postproc import detect_boundary_events
from online_phase.utils.metrics import (
    phase_accuracy, phase_macro_f1, boundary_f1,
    progress_mae, count_illegal_transitions, count_phase_jitter,
)
from online_phase.utils.viz import plot_streaming_summary


def stream_demo(
    model: torch.nn.Module,
    embeddings: np.ndarray,
    window_size: int,
    device: torch.device,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> dict:
    """Stream through a demo frame-by-frame using structured beliefs."""
    model.eval()

    T, D = embeddings.shape
    emb = embeddings.astype(np.float32)
    if mean is not None and std is not None:
        emb = (emb - mean) / std

    phase_preds = []
    progress_preds = []
    boundary_preds = []
    belief_history = []

    with torch.no_grad():
        for t in range(T):
            # Build window [t-W+1, ..., t]
            start = max(0, t - window_size + 1)
            window = emb[start: t + 1]
            pad_len = window_size - len(window)
            if pad_len > 0:
                window = np.vstack([
                    np.zeros((pad_len, D), dtype=np.float32),
                    window,
                ])
            mask = np.zeros(window_size, dtype=np.float32)
            mask[pad_len:] = 1.0

            # Model forward
            x = torch.from_numpy(window).unsqueeze(0).to(device)
            m = torch.from_numpy(mask).unsqueeze(0).to(device)
            out = model(x, m)

            b_prob = out["boundary"][0, -1].item()
            p_val = out["progress"][0, -1].item()
            if "beliefs" not in out:
                raise RuntimeError("Structured model output is missing 'beliefs'.")
            belief = out["beliefs"][0, -1].cpu().numpy()

            phase_preds.append(int(np.argmax(belief)))
            progress_preds.append(p_val)
            boundary_preds.append(b_prob)
            belief_history.append(belief.copy())

    return {
        "z_pred": np.array(phase_preds),
        "p_pred": np.array(progress_preds),
        "b_pred": np.array(boundary_preds),
        "belief_history": np.array(belief_history),
    }


def main(ckpt_path: str, demo_name: str) -> None:
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

    print("Loading structured model (PhaseModelStructured)")
    model = PhaseModelStructured(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    K = config.get("num_phases", 11)
    window_size = config.get("window_size", 48)

    # Normalization stats
    norm_stats = ckpt.get("norm_stats")
    mean = norm_stats["mean"] if norm_stats else None
    std = norm_stats["std"] if norm_stats else None

    # Load embeddings
    emb_dir = resolve_path(config["embeddings_dir"])
    camera_key = config.get("camera_keys", ["agentview_rgb"])[0]
    emb_path = emb_dir / f"{demo_name}_{camera_key}.npy"
    if not emb_path.exists():
        print(f"Error: embeddings not found at {emb_path}")
        return
    embeddings = np.load(emb_path)
    print(f"Demo {demo_name}: {len(embeddings)} frames")

    # Load labels
    cache_dir = resolve_path(config.get("label_cache_dir", "online_phase/cache/labels"))
    if cache_dir.exists() and (cache_dir / f"{demo_name}.npz").exists():
        data = np.load(cache_dir / f"{demo_name}.npz")
        labels = {k: data[k] for k in data.files}
    else:
        all_labels = generate_labels(
            resolve_path(config["segmentation_json"]),
            emb_dir, camera_key=camera_key,
        )
        labels = all_labels.get(demo_name, {})

    # Stream
    print("Streaming through demo...")
    results = stream_demo(model, embeddings, window_size, device, mean, std)

    z_pred = results["z_pred"]
    p_pred = results["p_pred"]
    b_pred = results["b_pred"]
    belief_history = results["belief_history"]

    # Metrics
    if "z" in labels:
        z_true = labels["z"]
        p_true = labels["p"]
        b_true = labels["b"]

        # Derive true boundary frames
        z_diff = np.diff(z_true)
        true_boundaries = list(np.where(z_diff != 0)[0] + 1)

        # Detected boundary events
        b_events = detect_boundary_events(b_pred, threshold=0.6, refractory=10)
        bnd = boundary_f1(b_events, true_boundaries, tolerance=5)

        print(f"\n=== Results for {demo_name} ===")
        print(f"  Phase accuracy:     {phase_accuracy(z_pred, z_true):.3f}")
        print(f"  Phase macro F1:     {phase_macro_f1(z_pred, z_true):.3f}")
        print(f"  Progress MAE:       {progress_mae(p_pred, p_true):.4f}")
        print(f"  Boundary F1:        {bnd['f1']:.3f} (P={bnd['precision']:.3f}, R={bnd['recall']:.3f})")
        print(f"  Illegal transitions: {count_illegal_transitions(z_pred)}")
        print(f"  Jitter:             {count_phase_jitter(z_pred):.1f} changes/100 frames")

        # Visualization
        output_dir = resolve_path(config.get("output_dir", "online_phase/runs/structured"))
        viz_path = output_dir / "viz" / f"{demo_name}_stream.png"
        plot_streaming_summary(
            z_true, z_pred, p_true, p_pred, b_true, b_pred,
            belief_history, K, demo_name=demo_name,
            save_path=viz_path, true_boundaries=true_boundaries,
        )
    else:
        print("No ground-truth labels available for this demo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream demo with online phase inference")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--demo", type=str, required=True, help="Demo name (e.g. demo_0)")
    args = parser.parse_args()
    main(args.ckpt, args.demo)
