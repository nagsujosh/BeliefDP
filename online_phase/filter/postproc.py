"""Post-processing utilities for belief filter output."""
from __future__ import annotations

import numpy as np


def detect_boundary_events(
    boundary_probs: np.ndarray,
    threshold: float = 0.6,
    refractory: int = 10,
) -> list[int]:
    """Detect boundary events from a sequence of boundary probabilities.

    Args:
        boundary_probs: (T,) array of boundary probabilities.
        threshold: Minimum probability to trigger an event.
        refractory: Minimum frames between events.

    Returns:
        List of frame indices where boundary events fire.
    """
    events = []
    last_event = -refractory  # Allow first event immediately

    for t, prob in enumerate(boundary_probs):
        if prob > threshold and (t - last_event) >= refractory:
            events.append(t)
            last_event = t

    return events


def smooth_progress(
    progress: np.ndarray,
    alpha: float = 0.9,
) -> np.ndarray:
    """EMA smoothing of progress predictions.

    Args:
        progress: (T,) raw progress predictions.
        alpha: Smoothing factor (higher = more weight on current).

    Returns:
        (T,) smoothed progress, still monotonically non-decreasing.
    """
    smoothed = np.zeros_like(progress)
    smoothed[0] = progress[0]
    for t in range(1, len(progress)):
        smoothed[t] = alpha * progress[t] + (1 - alpha) * smoothed[t - 1]

    # Enforce monotonicity on smoothed output
    smoothed = np.maximum.accumulate(smoothed)
    return smoothed


def enforce_left_to_right(phases: np.ndarray) -> np.ndarray:
    """Post-hoc enforcement of left-to-right phase ordering.

    Removes any backward phase transitions by holding the previous phase.

    Args:
        phases: (T,) integer phase predictions.

    Returns:
        (T,) corrected phases (monotonically non-decreasing).
    """
    corrected = phases.copy()
    for t in range(1, len(corrected)):
        if corrected[t] < corrected[t - 1]:
            corrected[t] = corrected[t - 1]
    return corrected
