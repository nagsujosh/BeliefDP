"""Evaluation metrics for phase inference."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def phase_accuracy(z_pred: np.ndarray, z_true: np.ndarray) -> float:
    """Per-frame phase classification accuracy."""
    return accuracy_score(z_true, z_pred)


def phase_macro_f1(z_pred: np.ndarray, z_true: np.ndarray) -> float:
    """Macro-averaged F1 score across all phase classes."""
    return f1_score(z_true, z_pred, average="macro", zero_division=0)


def phase_per_class_f1(z_pred: np.ndarray, z_true: np.ndarray, K: int) -> dict[int, float]:
    """Per-class F1 scores."""
    scores = f1_score(z_true, z_pred, labels=list(range(K)), average=None, zero_division=0)
    return {i: float(s) for i, s in enumerate(scores)}


def phase_confusion_matrix(z_pred: np.ndarray, z_true: np.ndarray, K: int) -> np.ndarray:
    """Confusion matrix (K x K)."""
    return confusion_matrix(z_true, z_pred, labels=list(range(K)))


def progress_mae(p_pred: np.ndarray, p_true: np.ndarray) -> float:
    """Mean absolute error for progress."""
    return float(np.mean(np.abs(p_pred - p_true)))


def progress_rmse(p_pred: np.ndarray, p_true: np.ndarray) -> float:
    """Root mean squared error for progress."""
    return float(np.sqrt(np.mean((p_pred - p_true) ** 2)))


def boundary_f1(
    b_pred_events: list[int],
    b_true_events: list[int],
    tolerance: int = 5,
) -> dict[str, float]:
    """Boundary detection F1 with temporal tolerance.

    A predicted boundary matches a true boundary if they are within
    ±tolerance frames. Each true/pred boundary can match at most once.

    Args:
        b_pred_events: Predicted boundary frame indices.
        b_true_events: True boundary frame indices.
        tolerance: Matching tolerance in frames.

    Returns:
        {"precision", "recall", "f1"}
    """
    if len(b_pred_events) == 0 and len(b_true_events) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if len(b_pred_events) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if len(b_true_events) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred = sorted(b_pred_events)
    true = sorted(b_true_events)
    matched_true = set()
    tp = 0

    for p in pred:
        best_dist = float("inf")
        best_idx = -1
        for i, t in enumerate(true):
            if i in matched_true:
                continue
            dist = abs(p - t)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx >= 0:
            tp += 1
            matched_true.add(best_idx)

    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def estimate_phase_duration_stats(
    labels: dict[str, dict],
    K: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-phase duration statistics from training demo labels.

    Computes the mean and std of phase durations (in frames) for each
    phase across all training demos.

    Args:
        labels: Dict of {demo_id: {"z": phase_labels_array, ...}}.
        K: Number of phases.

    Returns:
        means: (K,) mean duration per phase in frames.
        stds: (K,) std duration per phase in frames.
    """
    phase_durations = [[] for _ in range(K)]

    for demo_id, demo_labels in labels.items():
        z = demo_labels["z"]
        if len(z) == 0:
            continue

        # Count consecutive frames in each phase
        current_phase = z[0]
        current_count = 1

        for t in range(1, len(z)):
            if z[t] == current_phase:
                current_count += 1
            else:
                if 0 <= current_phase < K:
                    phase_durations[current_phase].append(current_count)
                current_phase = z[t]
                current_count = 1

        # Don't forget the last segment
        if 0 <= current_phase < K:
            phase_durations[current_phase].append(current_count)

    means = np.zeros(K)
    stds = np.zeros(K)
    for k in range(K):
        if len(phase_durations[k]) > 0:
            means[k] = np.mean(phase_durations[k])
            stds[k] = np.std(phase_durations[k]) if len(phase_durations[k]) > 1 else means[k] * 0.3
        else:
            means[k] = 30.0  # reasonable default
            stds[k] = 10.0

    return means, stds


def count_illegal_transitions(z_pred: np.ndarray) -> int:
    """Count backward phase transitions (z_t < z_{t-1})."""
    diffs = np.diff(z_pred)
    return int(np.sum(diffs < 0))


def count_phase_jitter(z_pred: np.ndarray, window: int = 100) -> float:
    """Count phase changes per `window` frames (jitter metric)."""
    changes = np.sum(np.diff(z_pred) != 0)
    n_windows = max(len(z_pred) / window, 1)
    return float(changes / n_windows)
