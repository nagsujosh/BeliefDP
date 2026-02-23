"""Label generation from offline phase segmentation outputs.

Converts phase_segmentation.json + cached R3M embeddings into per-frame
training labels: z_t (phase), p_t (progress), b_t (boundary), c_t (confidence).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Import compute_progress from the existing pipeline (avoid duplication).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
from phase_segmentation import compute_progress  # noqa: E402

from online_phase.utils.io import load_json


def generate_phase_labels(boundaries: list[int], T: int) -> np.ndarray:
    """Generate per-frame phase index from boundary list.

    Boundaries define the END of each phase (K entries, last = T-1).
    Phase 0: [0, boundary[0]], Phase i: [boundary[i-1]+1, boundary[i]].

    Returns:
        (T,) int64 array with values in {0, ..., K-1}.
    """
    z = np.zeros(T, dtype=np.int64)
    prev = 0
    for phase_idx, bnd in enumerate(boundaries):
        z[prev: bnd + 1] = phase_idx
        prev = bnd + 1
    return z


def generate_progress_labels(
    embeddings: np.ndarray,
    method: str = "cumulative",
) -> np.ndarray:
    """Generate monotone progress labels over the full demo.

    Args:
        embeddings: (T, D) array of R3M embeddings.
        method: "cumulative" (default, recommended) uses cumulative embedding
                change for motion-based progress. "goal_distance" (legacy) uses
                distance-to-goal normalized by start-to-goal distance.

    Returns:
        (T,) float32 array in [0, 1], monotonically non-decreasing.
    """
    T = len(embeddings)
    progress = compute_progress(embeddings, 0, T - 1, method=method).astype(np.float32)
    # Enforce exact endpoint: p[T-1] = 1.0 (avoid floating drift)
    if T > 0:
        progress[-1] = 1.0
    return progress


def generate_boundary_targets(
    boundaries: list[int],
    T: int,
    sigma_b: float = 4.0,
    include_tail: bool = False,
    tail_sigma: float = 2.0,
) -> np.ndarray:
    """Generate soft Gaussian boundary targets.

    By default excludes the terminal boundary (T-1) since every demo ends there.
    Optionally includes a small tail boundary near the end with a tighter Gaussian.

    Args:
        boundaries: List of K boundary frame indices (last = T-1).
        T: Total number of frames.
        sigma_b: Gaussian std in frames for interior boundaries.
        include_tail: If True, add a small Gaussian at the terminal boundary.
        tail_sigma: Gaussian std for the tail boundary (tighter than interior).

    Returns:
        (T,) float32 array in [0, 1].
    """
    t = np.arange(T, dtype=np.float32)
    # Exclude terminal boundary for main targets
    interior_boundaries = [b for b in boundaries if b < T - 1]

    if len(interior_boundaries) == 0 and not include_tail:
        return np.zeros(T, dtype=np.float32)

    b_target = np.zeros(T, dtype=np.float32)

    if len(interior_boundaries) > 0:
        # Vectorized: (T, num_boundaries) distance matrix
        bnd_arr = np.array(interior_boundaries, dtype=np.float32)
        dists = t[:, None] - bnd_arr[None, :]  # (T, B)
        gaussians = np.exp(-0.5 * (dists / sigma_b) ** 2)  # (T, B)
        b_target = gaussians.max(axis=1)  # (T,)

    # Optional tail boundary at T-1
    if include_tail:
        tail_gaussian = np.exp(-0.5 * ((t - (T - 1)) / tail_sigma) ** 2)
        b_target = np.maximum(b_target, tail_gaussian)

    return b_target


def generate_confidence_weights(
    anchor_indices: list[int],
    T: int,
    c_min: float = 0.3,
    c_max: float = 1.0,
    anchor_sigma: float = 6.0,
) -> np.ndarray:
    """Generate per-frame confidence weights based on anchor proximity.

    Frames near data-driven anchors (high-prominence peaks) get higher
    confidence; frames near interpolated boundaries get lower confidence.

    Returns:
        (T,) float32 array in [c_min, c_max].
    """
    t = np.arange(T, dtype=np.float32)

    if len(anchor_indices) == 0:
        return np.full(T, c_min, dtype=np.float32)

    anc_arr = np.array(anchor_indices, dtype=np.float32)
    dists = t[:, None] - anc_arr[None, :]  # (T, A)
    influence = np.exp(-0.5 * (dists / anchor_sigma) ** 2).max(axis=1)  # (T,)

    c = c_min + (c_max - c_min) * influence
    return np.clip(c, c_min, c_max).astype(np.float32)


def generate_labels(
    seg_json_path: str | Path,
    embeddings_dir: str | Path,
    sigma_b: float = 4.0,
    c_min: float = 0.3,
    c_max: float = 1.0,
    anchor_sigma: float = 6.0,
    camera_key: str = "agentview_rgb",  # Auto-resolves wrist_rgb -> eye_in_hand_rgb
    include_tail_boundary: bool = False,
    tail_sigma: float = 2.0,
) -> dict[str, dict[str, np.ndarray]]:
    """Generate all per-frame labels for every demo.

    Args:
        seg_json_path: Path to phase_segmentation.json.
        embeddings_dir: Directory containing demo_N_{camera_key}.npy files.
        sigma_b: Gaussian std for boundary targets.
        c_min: Minimum confidence weight.
        c_max: Maximum confidence weight.
        anchor_sigma: Gaussian radius for anchor influence.
        camera_key: Camera key for embedding filenames.

    Returns:
        Dict mapping demo_name -> {"z": (T,), "p": (T,), "b": (T,), "c": (T,)}.
    """
    seg_json_path = Path(seg_json_path)
    embeddings_dir = Path(embeddings_dir)

    seg_data = load_json(seg_json_path)
    demos = seg_data["demos"]

    all_labels = {}
    for demo_name, demo_info in demos.items():
        emb_path = embeddings_dir / f"{demo_name}_{camera_key}.npy"
        if not emb_path.exists():
            print(f"  [WARN] Missing embeddings for {demo_name}, skipping")
            continue

        embeddings = np.load(emb_path)
        T = len(embeddings)
        boundaries = demo_info["phase_boundaries"]
        anchor_indices = demo_info.get("anchor_indices", [])

        z = generate_phase_labels(boundaries, T)
        p = generate_progress_labels(embeddings)
        b = generate_boundary_targets(
            boundaries, T, sigma_b=sigma_b,
            include_tail=include_tail_boundary, tail_sigma=tail_sigma,
        )
        c = generate_confidence_weights(
            anchor_indices, T, c_min=c_min, c_max=c_max, anchor_sigma=anchor_sigma
        )

        all_labels[demo_name] = {"z": z, "p": p, "b": b, "c": c}

    return all_labels


def save_labels(labels: dict, output_dir: str | Path) -> None:
    """Cache labels as .npz files for fast loading."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for demo_name, arrays in labels.items():
        np.savez(output_dir / f"{demo_name}.npz", **arrays)


def load_cached_labels(cache_dir: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Load cached labels from .npz files."""
    cache_dir = Path(cache_dir)
    labels = {}
    for npz_path in sorted(cache_dir.glob("demo_*.npz")):
        demo_name = npz_path.stem
        data = np.load(npz_path)
        labels[demo_name] = {k: data[k] for k in data.files}
    return labels
