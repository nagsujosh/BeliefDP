"""Visualization utilities for phase inference results."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Qualitative colormap for phases
PHASE_COLORS = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())


def plot_phase_timeline(
    z_true: np.ndarray,
    z_pred: np.ndarray,
    K: int,
    ax: plt.Axes | None = None,
    title: str = "Phase Timeline",
) -> plt.Axes:
    """Plot ground-truth vs predicted phase colored bands."""
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 2))

    T = len(z_true)
    colors = PHASE_COLORS[:K]

    # Ground truth (top band)
    for t in range(T):
        ax.axvspan(t, t + 1, ymin=0.5, ymax=1.0, color=colors[z_true[t]], alpha=0.7)
    # Predicted (bottom band)
    for t in range(T):
        ax.axvspan(t, t + 1, ymin=0.0, ymax=0.5, color=colors[z_pred[t]], alpha=0.7)

    ax.set_xlim(0, T)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.75])
    ax.set_yticklabels(["Predicted", "Ground Truth"])
    ax.set_xlabel("Frame")
    ax.set_title(title)
    return ax


def plot_belief_heatmap(
    belief_history: np.ndarray,
    K: int,
    ax: plt.Axes | None = None,
    title: str = "Belief Evolution",
) -> plt.Axes:
    """Plot belief vector evolution as a heatmap.

    Args:
        belief_history: (T, K) array of belief vectors.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 3))

    im = ax.imshow(
        belief_history.T,
        aspect="auto",
        origin="lower",
        cmap="hot",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Phase")
    ax.set_yticks(range(K))
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Belief")
    return ax


def plot_progress(
    p_true: np.ndarray,
    p_pred: np.ndarray,
    ax: plt.Axes | None = None,
    title: str = "Progress",
) -> plt.Axes:
    """Overlay true vs predicted progress curves."""
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 2))

    T = len(p_true)
    ax.plot(range(T), p_true, "b-", alpha=0.7, label="True", linewidth=1.5)
    ax.plot(range(T), p_pred, "r--", alpha=0.7, label="Predicted", linewidth=1.5)
    ax.set_xlim(0, T)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Progress")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return ax


def plot_boundary(
    b_true: np.ndarray,
    b_pred: np.ndarray,
    true_boundaries: list[int] | None = None,
    threshold: float = 0.6,
    ax: plt.Axes | None = None,
    title: str = "Boundary Probability",
) -> plt.Axes:
    """Plot boundary target and predicted probability."""
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 2))

    T = len(b_true)
    ax.fill_between(range(T), b_true, alpha=0.3, color="blue", label="Target")
    ax.plot(range(T), b_pred, "r-", alpha=0.8, label="Predicted", linewidth=1)
    ax.axhline(y=threshold, color="gray", linestyle=":", alpha=0.5, label=f"Threshold={threshold}")

    if true_boundaries:
        for tb in true_boundaries:
            ax.axvline(x=tb, color="green", alpha=0.4, linestyle="--", linewidth=0.8)

    ax.set_xlim(0, T)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Boundary Prob")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_streaming_summary(
    z_true: np.ndarray,
    z_pred: np.ndarray,
    p_true: np.ndarray,
    p_pred: np.ndarray,
    b_true: np.ndarray,
    b_pred: np.ndarray,
    belief_history: np.ndarray,
    K: int,
    demo_name: str = "",
    save_path: str | Path | None = None,
    true_boundaries: list[int] | None = None,
) -> plt.Figure:
    """Create a multi-panel summary figure for a streaming demo."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f"Online Phase Inference — {demo_name}", fontsize=14)

    plot_phase_timeline(z_true, z_pred, K, ax=axes[0], title="Phase Timeline (top=GT, bottom=Pred)")
    plot_belief_heatmap(belief_history, K, ax=axes[1], title="Belief Evolution B(t)")
    plot_progress(p_true, p_pred, ax=axes[2], title="Progress (blue=GT, red=Pred)")
    plot_boundary(b_true, b_pred, true_boundaries=true_boundaries, ax=axes[3], title="Boundary Prob")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved visualization to {save_path}")

    return fig
