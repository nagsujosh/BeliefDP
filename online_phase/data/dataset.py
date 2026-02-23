"""PyTorch Dataset for sliding-window phase inference training.

Provides windowed embeddings + per-frame labels with padding for early frames,
global embedding normalization, and temporal augmentation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.camera_utils import normalize_camera_keys


def create_splits(
    demo_ids: list[str],
    train_frac: float = 0.75,
    val_frac: float = 0.15,
    test_frac: float = 0.10,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Split demo IDs into train/val/test sets.

    Returns:
        (train_ids, val_ids, test_ids)
    """
    rng = np.random.RandomState(seed)
    ids = sorted(demo_ids, key=lambda s: int(s.split("_")[-1]))
    perm = rng.permutation(len(ids))

    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = perm[:n_train]
    val_idx = perm[n_train: n_train + n_val]
    test_idx = perm[n_train + n_val:]

    return (
        [ids[i] for i in sorted(train_idx)],
        [ids[i] for i in sorted(val_idx)],
        [ids[i] for i in sorted(test_idx)],
    )


def compute_normalization_stats(
    embeddings_dir: Path,
    demo_ids: list[str],
    camera_key: str = "agentview_rgb",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute global mean and std from training demos' embeddings.

    Returns:
        (mean, std) each of shape (D,).
    """
    all_embs = []
    for demo_id in demo_ids:
        path = embeddings_dir / f"{demo_id}_{camera_key}.npy"
        if path.exists():
            all_embs.append(np.load(path))

    stacked = np.concatenate(all_embs, axis=0)  # (total_frames, D)
    mean = stacked.mean(axis=0).astype(np.float32)
    # Use GLOBAL scalar std (not per-dimension) because R3M embeddings are
    # very sparse with many near-zero-variance dimensions. Per-dimension
    # normalization would produce huge values and NaN losses.
    global_std = float(stacked.std())
    std = np.full_like(mean, global_std)
    std = np.maximum(std, 1e-4)
    return mean, std


class PhaseDataset(Dataset):
    """Sliding-window dataset for online phase inference training.

    Each sample is a window of W embedding frames ending at frame t,
    with the corresponding labels (z_t, p_t, b_t, c_t) for ALL frames
    in the window (full-sequence training).

    Args:
        embeddings_dir: Directory with demo_N_{camera_key}.npy files.
        labels: Dict mapping demo_name -> {"z": (T,), "p": (T,), "b": (T,), "c": (T,)}.
        window_size: Number of frames per window.
        stride: Step between consecutive windows.
        demo_ids: Which demos to include (None = all in labels).
        camera_key: Camera key for embedding filenames.
        mean: Global embedding mean for normalization (D,). None = no normalization.
        std: Global embedding std for normalization (D,). None = no normalization.
        augmentation: Dict with 'noise_std' and 'frame_mask_prob'. None = no augmentation.
        per_demo_normalize: If True, subtract per-demo mean before global normalization.
            This fixes domain shift where some demos have systematically different
            embedding means (e.g. different lighting, camera angle, style).
    """

    def __init__(
        self,
        embeddings_dir: str | Path,
        labels: dict[str, dict[str, np.ndarray]],
        window_size: int = 48,
        stride: int = 1,
        demo_ids: list[str] | None = None,
        camera_key: str = "agentview_rgb",
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        augmentation: dict | None = None,
        per_demo_normalize: bool = False,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.window_size = window_size
        self.augmentation = augmentation
        self.mean = mean
        self.std = std
        self.per_demo_normalize = per_demo_normalize

        if demo_ids is None:
            demo_ids = sorted(labels.keys(), key=lambda s: int(s.split("_")[-1]))

        # Load all embeddings into memory and build index
        self._embeddings: dict[str, np.ndarray] = {}
        self._labels: dict[str, dict[str, np.ndarray]] = {}
        self._index: list[tuple[str, int]] = []  # (demo_id, frame_idx)

        for demo_id in demo_ids:
            if demo_id not in labels:
                continue
            emb_path = self.embeddings_dir / f"{demo_id}_{camera_key}.npy"
            if not emb_path.exists():
                continue

            emb = np.load(emb_path).astype(np.float32)

            # Per-demo normalization: subtract demo mean to fix domain shift
            # e'_t = (e_t - μ_demo) / σ_global
            if self.per_demo_normalize:
                demo_mean = emb.mean(axis=0)
                emb = emb - demo_mean

            # Global normalization
            if self.mean is not None and self.std is not None:
                emb = (emb - self.mean) / self.std

            self._embeddings[demo_id] = emb
            self._labels[demo_id] = labels[demo_id]
            T = len(emb)

            # Create index entries for every valid target frame
            for t in range(0, T, stride):
                self._index.append((demo_id, t))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        demo_id, t = self._index[idx]
        emb = self._embeddings[demo_id]
        lab = self._labels[demo_id]
        T, D = emb.shape
        W = self.window_size

        # Extract window [t-W+1, ..., t] with left-padding
        start = t - W + 1
        if start >= 0:
            window = emb[start: t + 1].copy()
            z_window = lab["z"][start: t + 1].copy()
            p_window = lab["p"][start: t + 1].copy()
            b_window = lab["b"][start: t + 1].copy()
            c_window = lab["c"][start: t + 1].copy()
            mask = np.ones(W, dtype=np.float32)
        else:
            pad_len = -start
            window = np.zeros((W, D), dtype=np.float32)
            window[pad_len:] = emb[: t + 1]

            z_window = np.zeros(W, dtype=np.int64)
            z_window[pad_len:] = lab["z"][: t + 1]

            p_window = np.zeros(W, dtype=np.float32)
            p_window[pad_len:] = lab["p"][: t + 1]

            b_window = np.zeros(W, dtype=np.float32)
            b_window[pad_len:] = lab["b"][: t + 1]

            c_window = np.zeros(W, dtype=np.float32)
            c_window[pad_len:] = lab["c"][: t + 1]

            mask = np.zeros(W, dtype=np.float32)
            mask[pad_len:] = 1.0

        # Temporal augmentation (training only)
        if self.augmentation is not None:
            noise_std = self.augmentation.get("noise_std", 0.0)
            mask_prob = self.augmentation.get("frame_mask_prob", 0.0)

            if noise_std > 0:
                window = window + np.random.randn(*window.shape).astype(np.float32) * noise_std

            if mask_prob > 0:
                frame_mask = np.random.rand(W) > mask_prob
                window = window * frame_mask[:, None].astype(np.float32)

        return {
            "embedding_window": torch.from_numpy(window),       # (W, D)
            "z": torch.from_numpy(z_window),                    # (W,) int64
            "p": torch.from_numpy(p_window),                    # (W,) float32
            "b": torch.from_numpy(b_window),                    # (W,) float32
            "c": torch.from_numpy(c_window),                    # (W,) float32
            "attention_mask": torch.from_numpy(mask),            # (W,) float32
        }
