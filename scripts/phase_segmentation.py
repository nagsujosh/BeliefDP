"""Principled phase segmentation across demonstrations.

Approach (anchor + change-point + progress):

1. TRANSITION SIGNAL: For each demo, compute the embedding transition signal
   (frame-to-frame change magnitude, Savitzky-Golay smoothed).

2. AUTO-DETECT PHASE COUNT (two-pass):
   Pass 1: Detect ALL peaks across all demos, pool their prominences.
   Pass 2: Find a global prominence threshold via Otsu's method (parameter-free
   bimodal separation). Peaks above threshold = "real" task transitions.
   The median surviving peak count across demos = auto-detected K.

3. ANCHOR DETECTION: For each demo, peaks above the global threshold are
   the anchors — reliable, high-confidence task boundaries.

4. PROGRESS-BASED INTERIOR SEGMENTATION: If needed (when anchor count < K
   for a demo), place interior boundaries via progress quantiles between
   anchor pairs. This ensures every demo gets exactly K phases.

5. LEFT-TO-RIGHT ENFORCEMENT: Phase IDs are strictly increasing in time.

6. UNCERTAINTY: Per-boundary confidence from transition signal magnitude,
   per-phase temporal variance across demos.

Improved scoring (v2):

- Multi-cue composite score s_t combining:
  (a) Baseline magnitude change ||phi_t - phi_{t-1}||
  (b) Contextual change: ||mean(phi[t-k:t]) - mean(phi[t:t+k])||
  (c) Velocity + acceleration of embeddings
  (d) Progress slope |dp/dt| (if progress available)
  (e) Boundary probability b_t (heuristic or learned)

- Boundary refinement (optional second-pass):
  Train a small 1D-conv boundary classifier on pseudo-labels from initial
  anchors, then rerun segmentation with predicted b_t.

- Progress-driven alignment: align phases using cumulative progress
  instead of normalized time.

Usage:
    # Auto-detect number of phases (default, baseline scoring)
    python scripts/phase_segmentation.py \
        --metadata output/subgoal_metadata.json \
        --embeddings-dir output/embeddings/ \
        --output-dir output/ \
        --visualize

    # Improved multi-cue scoring
    python scripts/phase_segmentation.py \
        --metadata output/subgoal_metadata.json \
        --embeddings-dir output/embeddings/ \
        --output-dir output/ \
        --scoring composite \
        --visualize

    # Composite scoring + boundary refinement
    python scripts/phase_segmentation.py \
        --metadata output/subgoal_metadata.json \
        --embeddings-dir output/embeddings/ \
        --output-dir output/ \
        --scoring composite \
        --boundary-refine \
        --visualize

    # Progress-driven alignment
    python scripts/phase_segmentation.py \
        --metadata output/subgoal_metadata.json \
        --embeddings-dir output/embeddings/ \
        --output-dir output/ \
        --align-mode progress \
        --visualize
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    # Many SciPy wheels pinned in research environments are incompatible with NumPy 2.x.
    # Skip SciPy import proactively in that case to avoid noisy import tracebacks.
    _np_major = int(str(np.__version__).split(".")[0])
    if _np_major >= 2:
        raise ImportError("SciPy skipped because NumPy >= 2")

    from scipy.signal import savgol_filter as _savgol_filter
    from scipy.signal import find_peaks as _find_peaks

    savgol_filter = _savgol_filter
    find_peaks = _find_peaks
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

    def savgol_filter(x: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
        """Fallback smoother when SciPy is unavailable."""
        x = np.asarray(x)
        if window_length <= 1:
            return x.copy()
        if window_length % 2 == 0:
            window_length += 1
        kernel = np.ones(window_length, dtype=np.float64) / float(window_length)
        return np.convolve(x, kernel, mode="same")

    def find_peaks(
        x: np.ndarray,
        distance: int = 1,
        prominence: float = 0.0,
    ) -> tuple[np.ndarray, dict]:
        """Fallback peak finder with approximate prominence."""
        x = np.asarray(x)
        if x.size < 3:
            return np.array([], dtype=int), {"prominences": np.array([], dtype=float)}

        local_max = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
        if local_max.size == 0:
            return np.array([], dtype=int), {"prominences": np.array([], dtype=float)}

        # Approximate prominence from local neighborhoods.
        approx_prom = []
        half = max(1, int(distance))
        n = len(x)
        for p in local_max:
            left = max(0, p - half)
            right = min(n, p + half + 1)
            left_min = x[left : p + 1].min()
            right_min = x[p:right].min()
            prom = float(x[p] - max(left_min, right_min))
            approx_prom.append(prom)
        approx_prom = np.asarray(approx_prom, dtype=float)

        keep = approx_prom >= float(prominence)
        peaks = local_max[keep]
        prominences = approx_prom[keep]

        if peaks.size == 0:
            return np.array([], dtype=int), {"prominences": np.array([], dtype=float)}

        # Enforce minimum distance by greedy score selection.
        if distance > 1 and peaks.size > 1:
            order = np.argsort(x[peaks])[::-1]
            selected: list[int] = []
            for idx in order:
                p = int(peaks[idx])
                if all(abs(p - q) >= distance for q in selected):
                    selected.append(p)
            peaks = np.array(sorted(selected), dtype=int)
            prom_map = {int(p): float(prom) for p, prom in zip(local_max[keep], prominences)}
            prominences = np.array([prom_map[int(p)] for p in peaks], dtype=float)

        return peaks, {"prominences": prominences}

    print(
        "WARNING: SciPy unavailable/incompatible. "
        "Using NumPy fallback smoothing and peak detection."
    )


# ======================================================================= #
#  Step 1: Change-point detection per demo
# ======================================================================= #

def compute_transition_signal(
    embeddings: np.ndarray,
    metric: str = "l2",
    smooth_window: int = 11,
    smooth_polyorder: int = 2,
) -> np.ndarray:
    """Compute frame-to-frame transition magnitude in embedding space.

    Args:
        embeddings: (T, D) array.
        metric: "l2" for Euclidean distance, "cosine" for cosine distance.
        smooth_window: Savitzky-Golay window for smoothing. Must be odd.
        smooth_polyorder: Polynomial order for smoothing.

    Returns:
        (T,) array of transition magnitudes. First frame is 0.
    """
    if metric == "l2":
        diffs = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
    elif metric == "cosine":
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / np.maximum(norms, 1e-8)
        cos_sim = np.sum(normed[:-1] * normed[1:], axis=1)
        diffs = 1.0 - cos_sim
    else:
        raise ValueError(f"Unknown metric: {metric}")

    signal = np.concatenate([[0.0], diffs])

    # Smooth to reduce noise
    if smooth_window > 0 and len(signal) > smooth_window:
        signal = savgol_filter(signal, smooth_window, smooth_polyorder)
        signal = np.maximum(signal, 0)  # smoothing can create negatives

    return signal


# ======================================================================= #
#  Step 1b: Multi-cue composite scoring (v2)
# ======================================================================= #

def compute_contextual_change(
    embeddings: np.ndarray,
    k: int = 10,
    metric: str = "l2",
) -> np.ndarray:
    """Contextual change detection (CompILE-like).

    For each frame t, compare mean embedding in [t-k, t) vs [t, t+k).
    Large differences indicate context switches.

    Args:
        embeddings: (T, D) array.
        k: Half-window size (frames on each side).
        metric: "l2" or "cosine".

    Returns:
        (T,) array of contextual change magnitudes.
    """
    T, D = embeddings.shape
    ctx_change = np.zeros(T)

    for t in range(T):
        past_start = max(0, t - k)
        future_end = min(T, t + k)

        past_window = embeddings[past_start:t] if t > past_start else embeddings[t:t+1]
        future_window = embeddings[t:future_end] if future_end > t else embeddings[t:t+1]

        past_mean = past_window.mean(axis=0)
        future_mean = future_window.mean(axis=0)

        if metric == "l2":
            ctx_change[t] = np.linalg.norm(past_mean - future_mean)
        elif metric == "cosine":
            norm_p = np.linalg.norm(past_mean)
            norm_f = np.linalg.norm(future_mean)
            if norm_p > 1e-8 and norm_f > 1e-8:
                cos_sim = np.dot(past_mean, future_mean) / (norm_p * norm_f)
                ctx_change[t] = 1.0 - cos_sim
            else:
                ctx_change[t] = 0.0
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return ctx_change


def compute_velocity_acceleration(
    embeddings: np.ndarray,
    smooth_window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute embedding velocity and acceleration magnitudes.

    Velocity: ||phi_t - phi_{t-1}|| (first-order diff).
    Acceleration: ||v_t - v_{t-1}|| (second-order diff).

    Args:
        embeddings: (T, D) array.
        smooth_window: Smoothing window for velocity before computing acceleration.

    Returns:
        velocity: (T,) array of velocity magnitudes.
        acceleration: (T,) array of acceleration magnitudes.
    """
    T = len(embeddings)

    # Velocity: magnitude of first differences
    if T < 2:
        return np.zeros(T), np.zeros(T)

    v = np.diff(embeddings, axis=0)  # (T-1, D)
    vel_mag = np.linalg.norm(v, axis=1)  # (T-1,)
    velocity = np.concatenate([[0.0], vel_mag])

    # Acceleration: magnitude of second differences
    if T < 3:
        return velocity, np.zeros(T)

    # Use the raw velocity vectors for acceleration (not just magnitudes)
    a = np.diff(v, axis=0)  # (T-2, D)
    acc_mag = np.linalg.norm(a, axis=1)  # (T-2,)
    acceleration = np.concatenate([[0.0, 0.0], acc_mag])

    return velocity, acceleration


def compute_progress_slope(
    embeddings: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """Compute progress slope |dp/dt| using cumulative embedding change.

    Uses cumulative frame-to-frame distance as a monotone progress proxy,
    then computes the local rate of change via central differences.
    This is robust to goal-embedding variation across demos.

    Args:
        embeddings: (T, D) array.
        window: Window for finite-difference progress estimate.

    Returns:
        (T,) array of absolute progress slope.
    """
    T = len(embeddings)
    if T < 2:
        return np.zeros(T)

    # Cumulative embedding change as progress proxy
    diffs = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
    total = diffs.sum()
    if total < 1e-8:
        return np.zeros(T)
    progress = np.concatenate([[0.0], np.cumsum(diffs)]) / total

    # Progress slope: |dp/dt| via central differences
    slope = np.zeros(T)
    for t in range(T):
        t_past = max(0, t - window // 2)
        t_future = min(T - 1, t + window // 2)
        dt = t_future - t_past
        if dt > 0:
            slope[t] = abs(progress[t_future] - progress[t_past]) / dt
        else:
            slope[t] = 0.0

    return slope


def heuristic_boundary_prob(signal: np.ndarray) -> np.ndarray:
    """Compute heuristic boundary probability from a signal using sigmoid.

    Normalizes signal to zero-mean unit-variance, then applies sigmoid.
    Peaks in the signal map to high boundary probability.

    Args:
        signal: (T,) array (e.g., transition signal or composite score).

    Returns:
        (T,) array of boundary probabilities in (0, 1).
    """
    mu = signal.mean()
    sigma = signal.std()
    if sigma < 1e-8:
        return np.full_like(signal, 0.5)
    z = (signal - mu) / sigma
    return 1.0 / (1.0 + np.exp(-z))


def compute_composite_score(
    embeddings: np.ndarray,
    w_mag: float = 1.0,
    w_ctx: float = 0.5,
    w_vel: float = 0.3,
    w_acc: float = 0.2,
    w_prog: float = 0.3,
    w_bnd: float = 0.0,
    ctx_k: int = 10,
    metric: str = "l2",
    smooth_window: int = 11,
    smooth_polyorder: int = 2,
    boundary_probs: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Compute multi-cue composite segmentation score.

    s_t = w_mag * mag_t + w_ctx * ctx_t + w_vel * vel_t
          + w_acc * acc_t + w_prog * prog_slope_t + w_bnd * b_t

    Each component is normalized to [0, 1] before weighting.

    Args:
        embeddings: (T, D) array.
        w_mag: Weight for magnitude change (baseline).
        w_ctx: Weight for contextual change.
        w_vel: Weight for velocity.
        w_acc: Weight for acceleration.
        w_prog: Weight for progress slope.
        w_bnd: Weight for boundary probability.
        ctx_k: Context window half-size.
        metric: Distance metric ("l2" or "cosine").
        smooth_window: Savitzky-Golay window for final smoothing.
        smooth_polyorder: Polynomial order for smoothing.
        boundary_probs: (T,) optional external boundary probabilities.

    Returns:
        s_t: (T,) composite score.
        components: Dict mapping component name to (T,) raw array (for debug).
    """
    T = len(embeddings)

    def _norm01(x: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1]."""
        xmin, xmax = x.min(), x.max()
        if xmax - xmin < 1e-10:
            return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin)

    # (a) Baseline magnitude change
    mag = compute_transition_signal(embeddings, metric=metric, smooth_window=0)
    mag_norm = _norm01(mag)

    # (b) Contextual change
    ctx = compute_contextual_change(embeddings, k=ctx_k, metric=metric)
    ctx_norm = _norm01(ctx)

    # (c) Velocity + (d) Acceleration
    vel, acc = compute_velocity_acceleration(embeddings)
    vel_norm = _norm01(vel)
    acc_norm = _norm01(acc)

    # (e) Progress slope
    prog_slope = compute_progress_slope(embeddings, window=ctx_k)
    prog_norm = _norm01(prog_slope)

    # (f) Boundary prob
    if boundary_probs is not None:
        bnd = boundary_probs
    else:
        bnd = np.zeros(T)
    bnd_norm = _norm01(bnd) if bnd.max() > 0 else bnd

    # Combine
    s_t = (
        w_mag * mag_norm
        + w_ctx * ctx_norm
        + w_vel * vel_norm
        + w_acc * acc_norm
        + w_prog * prog_norm
        + w_bnd * bnd_norm
    )

    # Smooth composite score
    if smooth_window > 0 and len(s_t) > smooth_window:
        s_t = savgol_filter(s_t, smooth_window, smooth_polyorder)
        s_t = np.maximum(s_t, 0)

    components = {
        "magnitude": mag,
        "contextual": ctx,
        "velocity": vel,
        "acceleration": acc,
        "progress_slope": prog_slope,
        "boundary_prob": bnd,
        "magnitude_norm": mag_norm,
        "contextual_norm": ctx_norm,
        "velocity_norm": vel_norm,
        "acceleration_norm": acc_norm,
        "progress_slope_norm": prog_norm,
        "boundary_prob_norm": bnd_norm,
    }

    return s_t, components


# ======================================================================= #
#  Boundary refinement: lightweight boundary classifier
# ======================================================================= #

def train_boundary_classifier(
    all_embeddings: dict[str, np.ndarray],
    all_anchors: dict[str, list[int]],
    window_size: int = 15,
    epochs: int = 20,
    lr: float = 1e-3,
    label_sigma: float = 3.0,
) -> object:
    """Train a lightweight 1D-conv boundary classifier on pseudo-labels.

    Uses anchor frames from initial segmentation as positive pseudo-labels
    with soft Gaussian targets (sigma = label_sigma frames).

    Args:
        all_embeddings: {demo_name: (T, D) array} per demo.
        all_anchors: {demo_name: list of anchor frame indices} per demo.
        window_size: 1D conv receptive field.
        epochs: Training epochs (should be quick).
        lr: Learning rate.
        label_sigma: Gaussian sigma for soft boundary labels around anchors.

    Returns:
        Trained model (callable: embeddings -> boundary_probs).
        Returns None if torch is unavailable.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("  [boundary-refine] PyTorch not available, skipping.")
        return None

    D = next(iter(all_embeddings.values())).shape[1]

    # Build a small 1D-conv model
    class BoundaryConvNet(nn.Module):
        def __init__(self, input_dim, window):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(input_dim, 64, kernel_size=window, padding=window // 2),
                nn.ReLU(),
                nn.Conv1d(64, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(32, 1, kernel_size=1),
            )

        def forward(self, x):
            # x: (B, T, D) -> transpose to (B, D, T)
            x = x.transpose(1, 2)
            out = self.net(x)  # (B, 1, T)
            return out.squeeze(1)  # (B, T)

    model = BoundaryConvNet(D, window_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.0))

    # Prepare training data
    X_list = []
    Y_list = []
    for demo_name, emb in all_embeddings.items():
        anchors = all_anchors.get(demo_name, [])
        T = len(emb)

        # Soft labels: Gaussian around anchors
        y = np.zeros(T, dtype=np.float32)
        for a in anchors:
            for t in range(T):
                y[t] = max(y[t], np.exp(-0.5 * ((t - a) / label_sigma) ** 2))

        X_list.append(torch.from_numpy(emb.astype(np.float32)))
        Y_list.append(torch.from_numpy(y))

    # Train (all demos fit in memory for a lightweight model)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in zip(X_list, Y_list):
            x_batch = x.unsqueeze(0)  # (1, T, D)
            y_batch = y.unsqueeze(0)  # (1, T)

            optimizer.zero_grad()
            logits = model(x_batch)  # (1, T)
            # Trim to match if conv changed length
            min_len = min(logits.shape[1], y_batch.shape[1])
            loss = criterion(logits[:, :min_len], y_batch[:, :min_len])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg = total_loss / len(X_list)
            print(f"    [boundary-refine] epoch {epoch+1}/{epochs}, loss={avg:.4f}")

    model.eval()

    def predict_boundary_probs(embeddings: np.ndarray) -> np.ndarray:
        """Predict boundary probabilities for a single demo."""
        with torch.no_grad():
            x = torch.from_numpy(embeddings.astype(np.float32)).unsqueeze(0)
            logits = model(x)
            probs = torch.sigmoid(logits).squeeze(0).numpy()
            # Ensure length matches
            T = len(embeddings)
            if len(probs) < T:
                probs = np.pad(probs, (0, T - len(probs)))
            elif len(probs) > T:
                probs = probs[:T]
            return probs

    return predict_boundary_probs


def detect_anchors(
    transition_signal: np.ndarray,
    min_distance: int = 30,
    prominence_percentile: float = 70.0,
    prominence_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect anchor boundaries as peaks in the transition signal.

    Args:
        transition_signal: (T,) array from compute_transition_signal.
        min_distance: Minimum frames between anchors.
        prominence_percentile: Only keep peaks with prominence above this
            percentile of all peak prominences. Ignored if prominence_threshold
            is provided.
        prominence_threshold: Absolute prominence threshold. When provided,
            overrides prominence_percentile (used by auto-detection).

    Returns:
        anchor_indices: Frame indices of detected anchors.
        anchor_prominences: Peak prominence (sharpness) for each anchor.
    """
    # Find all peaks
    peaks, properties = find_peaks(
        transition_signal,
        distance=min_distance,
        prominence=0,  # get all, filter later
    )

    if len(peaks) == 0:
        return np.array([], dtype=int), np.array([])

    prominences = properties["prominences"]

    # Filter by threshold (absolute or percentile)
    if prominence_threshold is not None:
        mask = prominences >= prominence_threshold
    else:
        threshold = np.percentile(prominences, prominence_percentile)
        mask = prominences >= threshold

    anchor_indices = peaks[mask]
    anchor_prominences = prominences[mask]

    return anchor_indices, anchor_prominences


# ======================================================================= #
#  Step 2: Auto-detect number of phases
# ======================================================================= #

def find_prominence_threshold(prominences: np.ndarray) -> tuple[float, str]:
    """Find a natural threshold separating signal peaks from noise peaks.

    Uses two methods and picks the one that gives a more reasonable result:

    1. Largest gap in sorted prominences: Sort descending, find the largest
       consecutive gap. Peaks above the gap are "real" transitions. This is
       analogous to the eigengap heuristic in spectral clustering.

    2. Otsu's method on log-prominences: Log-transform to handle the heavy
       right-skew typical of prominence distributions, then apply Otsu's
       bimodal separation.

    The method that yields a threshold keeping 20-80% of peaks is preferred
    (avoids degenerate cases where almost all or almost none survive).

    Args:
        prominences: Pooled peak prominences from all demos.

    Returns:
        (threshold, method_used)
    """
    if len(prominences) < 3:
        return float(np.median(prominences)) if len(prominences) > 0 else 0.0, "fallback"

    sorted_desc = np.sort(prominences)[::-1]  # descending
    n = len(sorted_desc)

    # Method 1: Largest gap
    gaps = np.diff(sorted_desc)  # negative values (descending), gaps = magnitude of drop
    gaps = -gaps  # now positive: larger = bigger separation
    gap_idx = np.argmax(gaps)
    # Threshold is between sorted_desc[gap_idx] and sorted_desc[gap_idx + 1]
    threshold_gap = (sorted_desc[gap_idx] + sorted_desc[gap_idx + 1]) / 2
    surviving_gap = np.sum(prominences >= threshold_gap)
    frac_gap = surviving_gap / n

    # Method 2: Otsu on log-prominences
    log_proms = np.log(prominences + 1e-10)  # avoid log(0)
    log_sorted = np.sort(log_proms)

    # Otsu in log space
    best_sigma = -1
    best_log_threshold = log_sorted[0]
    for i in range(1, len(log_sorted)):
        cls0 = log_sorted[:i]
        cls1 = log_sorted[i:]
        w0, w1 = len(cls0) / n, len(cls1) / n
        if w0 == 0 or w1 == 0:
            continue
        sigma = w0 * w1 * (cls0.mean() - cls1.mean()) ** 2
        if sigma > best_sigma:
            best_sigma = sigma
            best_log_threshold = (log_sorted[i - 1] + log_sorted[i]) / 2

    threshold_otsu = np.exp(best_log_threshold)
    surviving_otsu = np.sum(prominences >= threshold_otsu)
    frac_otsu = surviving_otsu / n

    # Pick the method that gives a reasonable fraction (20-80%)
    # If both are reasonable, prefer largest gap (simpler, more interpretable)
    gap_reasonable = 0.10 <= frac_gap <= 0.85
    otsu_reasonable = 0.10 <= frac_otsu <= 0.85

    if gap_reasonable and otsu_reasonable:
        # Both reasonable — pick the one closer to keeping ~40% of peaks
        # (task-level transitions are typically ~30-50% of all detected peaks)
        target = 0.40
        if abs(frac_gap - target) <= abs(frac_otsu - target):
            return threshold_gap, "largest_gap"
        else:
            return threshold_otsu, "otsu_log"
    elif gap_reasonable:
        return threshold_gap, "largest_gap"
    elif otsu_reasonable:
        return threshold_otsu, "otsu_log"
    else:
        # Neither reasonable — use percentile fallback (keep top 40%)
        threshold_pct = np.percentile(prominences, 60)  # top 40%
        return threshold_pct, "percentile_fallback"


def auto_detect_num_phases(
    metadata: dict,
    embeddings_dir: Path,
    min_anchor_distance: int = 30,
    smooth_window: int = 11,
) -> tuple[int, float, dict]:
    """Auto-detect the optimal number of phases from data.

    Two-pass approach:
    1. Compute transition signals for all demos, find ALL peaks, pool prominences.
    2. Find a natural prominence threshold (largest gap or Otsu on log-prominences).
    3. Count surviving peaks per demo, take median as K.

    The threshold separates "real task transitions" (high prominence) from
    "noise / micro-adjustments" (low prominence). This is analogous to the
    eigengap heuristic in spectral clustering.

    Args:
        metadata: Per-demo metadata from extract_subgoals.py.
        embeddings_dir: Directory with cached .npy embeddings.
        min_anchor_distance: Min frames between peaks.
        smooth_window: Savitzky-Golay window.

    Returns:
        (num_phases, prominence_threshold, detection_info)
        detection_info contains per-demo peak counts and the full distribution.
    """
    all_prominences = []
    per_demo_peaks = {}

    print("Auto-detecting phase count...")
    print("  Pass 1: Collecting peak prominences across all demos...")

    for demo_name in sorted(metadata.keys(), key=lambda s: int(s.split("_")[-1])):
        info = metadata[demo_name]
        camera = info["camera"]
        emb_path = embeddings_dir / f"{demo_name}_{camera}.npy"

        if not emb_path.exists():
            continue

        embeddings = np.load(emb_path)
        signal = compute_transition_signal(embeddings, smooth_window=smooth_window)

        # Find ALL peaks (no prominence filtering)
        peaks, properties = find_peaks(
            signal, distance=min_anchor_distance, prominence=0
        )

        if len(peaks) > 0:
            proms = properties["prominences"]
            all_prominences.extend(proms.tolist())
            per_demo_peaks[demo_name] = {
                "peaks": peaks.tolist(),
                "prominences": proms.tolist(),
            }

    all_prominences = np.array(all_prominences)
    print(f"  Collected {len(all_prominences)} peaks across {len(per_demo_peaks)} demos")
    print(f"  Prominence range: [{all_prominences.min():.4f}, {all_prominences.max():.4f}]")
    print(f"  Prominence median: {np.median(all_prominences):.4f}")

    # Pass 2: Find threshold
    threshold, method = find_prominence_threshold(all_prominences)
    print(f"  Pass 2: Threshold = {threshold:.4f} (method: {method})")

    # Count surviving peaks per demo
    surviving_counts = []
    for demo_name, info in per_demo_peaks.items():
        proms = np.array(info["prominences"])
        count = int(np.sum(proms >= threshold))
        surviving_counts.append(count)

    surviving_counts = np.array(surviving_counts)
    median_k = int(np.median(surviving_counts))

    # num_phases = surviving peaks + 1 (for the terminal frame)
    num_phases = median_k + 1

    # Safety: ensure at least 3 phases (start, middle, end)
    if num_phases < 3:
        print(f"  WARNING: Auto-detected only {num_phases} phases, forcing minimum of 3")
        num_phases = 3

    print(f"  Surviving peaks per demo: min={surviving_counts.min()}, "
          f"median={np.median(surviving_counts):.0f}, max={surviving_counts.max()}")
    print(f"  Auto-detected num_phases = {num_phases} "
          f"({num_phases - 1} transitions + terminal frame)")

    detection_info = {
        "threshold": threshold,
        "threshold_method": method,
        "all_prominences_count": len(all_prominences),
        "prominence_min": float(all_prominences.min()),
        "prominence_max": float(all_prominences.max()),
        "prominence_median": float(np.median(all_prominences)),
        "peaks_above_threshold": int(np.sum(all_prominences >= threshold)),
        "peaks_below_threshold": int(np.sum(all_prominences < threshold)),
        "surviving_counts_per_demo": {
            demo: int(np.sum(np.array(info["prominences"]) >= threshold))
            for demo, info in per_demo_peaks.items()
        },
        "surviving_min": int(surviving_counts.min()),
        "surviving_median": float(np.median(surviving_counts)),
        "surviving_max": int(surviving_counts.max()),
    }

    return num_phases, threshold, detection_info


# ======================================================================= #
#  Step 3: Progress variable within chunks
# ======================================================================= #

def compute_progress(
    embeddings: np.ndarray,
    start_idx: int,
    end_idx: int,
    method: str = "cumulative",
) -> np.ndarray:
    """Compute monotone progress variable within a chunk.

    Two methods:
    - "cumulative" (default, recommended): Cumulative embedding change
      p_t = sum_{i=1}^{t} ||e_i - e_{i-1}|| / sum_{i=1}^{T} ||e_i - e_{i-1}||
      Aligns demos by actual motion progression, robust to goal-embedding variation.
    - "goal_distance" (legacy): 1 - ||e_t - e_goal|| / ||e_start - e_goal||
      Fails when goal embedding varies across demos or R3M isn't goal-aware.

    Both are monotonically relaxed and clamped to [0, 1].

    Args:
        embeddings: (T, D) full trajectory embeddings.
        start_idx: Start frame of this chunk.
        end_idx: End frame of this chunk (inclusive).
        method: "cumulative" or "goal_distance".

    Returns:
        (chunk_length,) array of progress values in [0, 1].
    """
    chunk = embeddings[start_idx:end_idx + 1]
    n = len(chunk)

    if n <= 1:
        return np.ones(n)

    if method == "cumulative":
        # Cumulative embedding change — motion-based progress
        diffs = np.linalg.norm(np.diff(chunk, axis=0), axis=1)  # (n-1,)
        total = diffs.sum()
        if total < 1e-8:
            return np.linspace(0, 1, n)
        cumulative = np.concatenate([[0.0], np.cumsum(diffs)])
        progress = cumulative / total  # [0, 1], already monotone
    elif method == "goal_distance":
        # Legacy: goal-distance based
        goal_emb = chunk[-1]
        start_emb = chunk[0]
        start_to_goal = np.linalg.norm(start_emb - goal_emb)
        if start_to_goal < 1e-8:
            return np.ones(n)
        distances = np.linalg.norm(chunk - goal_emb, axis=1)
        progress = 1.0 - distances / start_to_goal
        progress = np.clip(progress, 0, 1)
        # Monotonic relaxation
        progress = np.maximum.accumulate(progress)
    else:
        raise ValueError(f"Unknown progress method: {method}")

    # Enforce exact endpoints
    progress[0] = 0.0
    progress[-1] = 1.0

    return progress


def segment_chunk_by_progress(
    embeddings: np.ndarray,
    start_idx: int,
    end_idx: int,
    num_interior: int,
) -> list[int]:
    """Place interior boundaries at fixed progress quantiles within a chunk.

    Args:
        embeddings: Full trajectory embeddings.
        start_idx: Chunk start.
        end_idx: Chunk end (inclusive).
        num_interior: Number of interior boundaries to place.

    Returns:
        List of frame indices for interior boundaries (absolute indices).
    """
    if num_interior <= 0 or end_idx - start_idx < 2:
        return []

    progress = compute_progress(embeddings, start_idx, end_idx)
    quantiles = np.linspace(0, 1, num_interior + 2)[1:-1]  # exclude 0 and 1

    boundaries = []
    for q in quantiles:
        # Find the first frame where progress >= quantile
        candidates = np.where(progress >= q)[0]
        if len(candidates) > 0:
            boundaries.append(int(start_idx + candidates[0]))
        else:
            # Fallback: linear interpolation
            boundaries.append(int(start_idx + q * (end_idx - start_idx)))

    return boundaries


# ======================================================================= #
#  Step 4: Full segmentation pipeline per demo
# ======================================================================= #

def segment_demo(
    embeddings: np.ndarray,
    num_phases: int,
    min_anchor_distance: int = 30,
    prominence_percentile: float = 70.0,
    prominence_threshold: float | None = None,
    smooth_window: int = 11,
    scoring: str = "baseline",
    composite_weights: dict | None = None,
    ctx_k: int = 10,
    metric: str = "l2",
    boundary_probs: np.ndarray | None = None,
) -> dict:
    """Full segmentation pipeline for a single demo.

    1. Compute transition signal (baseline or composite)
    2. Detect anchors (major task boundaries)
    3. Allocate interior phases proportional to chunk duration
    4. Place interior boundaries via progress quantiles

    Args:
        scoring: "baseline" for original magnitude-only, "composite" for
                 multi-cue scoring.
        composite_weights: Dict with keys w_mag, w_ctx, w_vel, w_acc,
                          w_prog, w_bnd. Defaults provided if None.
        ctx_k: Context window half-size for contextual change.
        metric: Distance metric for embedding comparisons.
        boundary_probs: Optional (T,) external boundary probabilities.

    Returns dict with phase_boundaries, anchors, transition_signal, etc.
    """
    T = len(embeddings)

    # Step 1: Transition signal (baseline or composite)
    if scoring == "composite":
        cw = composite_weights or {}
        composite_score, score_components = compute_composite_score(
            embeddings,
            w_mag=cw.get("w_mag", 1.0),
            w_ctx=cw.get("w_ctx", 0.5),
            w_vel=cw.get("w_vel", 0.3),
            w_acc=cw.get("w_acc", 0.2),
            w_prog=cw.get("w_prog", 0.3),
            w_bnd=cw.get("w_bnd", 0.3 if boundary_probs is not None else 0.0),
            ctx_k=ctx_k,
            metric=metric,
            smooth_window=smooth_window,
            boundary_probs=boundary_probs,
        )
        transition_signal = composite_score
    else:
        transition_signal = compute_transition_signal(
            embeddings, metric=metric, smooth_window=smooth_window
        )
        score_components = None

    # Step 2: Detect anchors
    anchor_indices, anchor_prominences = detect_anchors(
        transition_signal,
        min_distance=min_anchor_distance,
        prominence_percentile=prominence_percentile,
        prominence_threshold=prominence_threshold,
    )

    # Always include start and end as chunk boundaries
    chunk_bounds = np.unique(np.concatenate([[0], anchor_indices, [T - 1]]))
    chunk_bounds = np.sort(chunk_bounds).astype(int)

    # Step 3: Allocate phases to chunks proportional to duration
    num_chunks = len(chunk_bounds) - 1
    chunk_durations = np.diff(chunk_bounds).astype(float)
    total_duration = chunk_durations.sum()

    # We need (num_phases - 1) interior boundaries total
    # (num_phases boundaries including terminal frame = num_phases segments)
    # Anchors already provide some boundaries, need more from interior
    num_anchor_boundaries = len(anchor_indices)
    num_interior_needed = max(0, num_phases - 1 - num_anchor_boundaries)

    if num_interior_needed > 0:
        # Allocate interior points to chunks proportional to duration
        raw_allocation = (chunk_durations / total_duration) * num_interior_needed
        allocation = np.round(raw_allocation).astype(int)

        # Fix rounding to match exactly
        diff = num_interior_needed - allocation.sum()
        if diff > 0:
            # Add to largest chunks
            for _ in range(diff):
                allocation[np.argmax(chunk_durations / (allocation + 1))] += 1
        elif diff < 0:
            # Remove from smallest chunks (that have allocation > 0)
            for _ in range(-diff):
                candidates = np.where(allocation > 0)[0]
                if len(candidates) == 0:
                    break
                allocation[candidates[np.argmin(chunk_durations[candidates])]] -= 1
    else:
        allocation = np.zeros(num_chunks, dtype=int)

    # Step 4: Place interior boundaries via progress
    all_boundaries = []
    for i in range(num_chunks):
        start = int(chunk_bounds[i])
        end = int(chunk_bounds[i + 1])
        interior = segment_chunk_by_progress(
            embeddings, start, end, allocation[i]
        )
        all_boundaries.extend(interior)

    # Combine: anchors + interior boundaries
    all_boundaries = sorted(set(
        list(anchor_indices) + all_boundaries
    ))

    # Ensure we don't include frame 0 or last frame as a "boundary"
    # (they're implicit chunk starts/ends)
    all_boundaries = [b for b in all_boundaries if 0 < b < T - 1]

    # Enforce minimum spacing: if two boundaries are closer than
    # min_anchor_distance // 2, keep the one with higher transition signal
    min_spacing = min_anchor_distance // 2
    if min_spacing > 1:
        filtered = []
        for b in all_boundaries:
            if not filtered or b - filtered[-1] >= min_spacing:
                filtered.append(b)
            else:
                # Keep whichever has stronger transition signal
                if transition_signal[b] > transition_signal[filtered[-1]]:
                    filtered[-1] = b
        all_boundaries = filtered

    # Always end with the last frame
    all_boundaries.append(T - 1)

    # Trim or pad to exactly num_phases boundaries
    # (boundaries define the END of each phase)
    if len(all_boundaries) > num_phases:
        # Keep the most prominent ones + ensure last frame stays
        scored = []
        for b in all_boundaries[:-1]:  # don't score last frame
            # Score = transition signal value (change magnitude at this frame)
            scored.append((transition_signal[b], b))
        scored.sort(reverse=True)
        kept = sorted([b for _, b in scored[:num_phases - 1]])
        all_boundaries = kept + [T - 1]
    elif len(all_boundaries) < num_phases:
        # Add boundaries by splitting largest gaps
        while len(all_boundaries) < num_phases:
            bounds_with_start = [0] + all_boundaries
            gaps = np.diff(bounds_with_start)
            largest = np.argmax(gaps)
            mid = (bounds_with_start[largest] + bounds_with_start[largest + 1]) // 2
            all_boundaries.append(mid)
            all_boundaries = sorted(all_boundaries)

    # Compute confidence per boundary
    boundary_confidence = []
    max_transition = transition_signal.max() if transition_signal.max() > 0 else 1.0
    for b in all_boundaries:
        conf = float(transition_signal[b] / max_transition)
        boundary_confidence.append(conf)

    result = {
        "phase_boundaries": all_boundaries,
        "boundary_confidence": boundary_confidence,
        "anchor_indices": anchor_indices.tolist(),
        "anchor_prominences": anchor_prominences.tolist(),
        "num_anchors": len(anchor_indices),
        "transition_signal": transition_signal,  # for visualization
    }
    if score_components is not None:
        result["score_components"] = score_components
    return result


# ======================================================================= #
#  Step 5: Cross-demo alignment via normalized time
# ======================================================================= #

def compute_demo_progress(
    embeddings: np.ndarray,
    transition_signal: np.ndarray,
) -> np.ndarray:
    """Compute cumulative progress proxy for a demo.

    Uses cumulative normalized transition signal as a monotone progress
    estimate: progress_t = cumsum(s_t) / sum(s_t).

    Args:
        embeddings: (T, D) array.
        transition_signal: (T,) transition/composite score.

    Returns:
        (T,) progress values in [0, 1].
    """
    total = transition_signal.sum()
    if total < 1e-10:
        return np.linspace(0, 1, len(transition_signal))
    cumulative = np.cumsum(transition_signal) / total
    return cumulative


def align_across_demos(
    all_results: dict[str, dict],
    metadata: dict,
    num_phases: int,
    align_mode: str = "progress",
    all_embeddings: dict[str, np.ndarray] | None = None,
    realign_boundaries: bool = True,
) -> dict:
    """Compute global phase statistics and optionally re-align boundaries.

    When realign_boundaries=True and align_mode="progress", this performs
    real cross-demo alignment:
    1. Compute cumulative progress for each demo
    2. Map boundaries to progress coordinates
    3. Compute median boundary position in progress space across demos
    4. Re-map median progress boundaries back to frame indices per demo
    This ensures phase K means the same semantic region across demos.

    Args:
        align_mode: "time" or "progress" (recommended).
        all_embeddings: Required if align_mode == "progress".
        realign_boundaries: If True, re-align boundaries to cross-demo
                           median progress positions. Modifies all_results in-place.
    """
    phase_times = [[] for _ in range(num_phases)]
    phase_confidences = [[] for _ in range(num_phases)]
    demo_progress_maps = {}  # {demo_name: (T,) progress array}

    for demo_name, result in all_results.items():
        num_frames = metadata[demo_name]["num_frames"]

        if align_mode == "progress" and all_embeddings is not None:
            progress = compute_demo_progress(
                all_embeddings[demo_name],
                result["transition_signal"],
            )
            demo_progress_maps[demo_name] = progress
            for i, (boundary, conf) in enumerate(
                zip(result["phase_boundaries"], result["boundary_confidence"])
            ):
                phase_times[i].append(float(progress[boundary]))
                phase_confidences[i].append(conf)
        else:
            for i, (boundary, conf) in enumerate(
                zip(result["phase_boundaries"], result["boundary_confidence"])
            ):
                norm_time = boundary / (num_frames - 1)
                phase_times[i].append(norm_time)
                phase_confidences[i].append(conf)

    # Compute per-phase statistics
    phase_stats = {}
    median_positions = []
    for i in range(num_phases):
        times = np.array(phase_times[i])
        confs = np.array(phase_confidences[i])
        phase_stats[i] = {
            "mean_time": float(times.mean()),
            "std_time": float(times.std()),
            "variance_time": float(times.var()),
            "median_time": float(np.median(times)),
            "mean_confidence": float(confs.mean()),
            "min_confidence": float(confs.min()),
            "count": len(times),
        }
        median_positions.append(float(np.median(times)))

    # Re-align boundaries to cross-demo median positions
    if realign_boundaries and align_mode == "progress" and demo_progress_maps:
        print("  Re-aligning boundaries to cross-demo median progress positions...")
        for demo_name, result in all_results.items():
            progress = demo_progress_maps[demo_name]
            T = len(progress)
            new_boundaries = []
            for i, target_prog in enumerate(median_positions):
                if i == num_phases - 1:
                    # Terminal boundary is always last frame
                    new_boundaries.append(T - 1)
                else:
                    # Find frame closest to target progress
                    frame_idx = int(np.argmin(np.abs(progress - target_prog)))
                    # Clamp to valid range
                    frame_idx = max(1, min(frame_idx, T - 2))
                    new_boundaries.append(frame_idx)

            # Enforce strict ordering and minimum spacing
            for j in range(1, len(new_boundaries) - 1):
                if new_boundaries[j] <= new_boundaries[j - 1]:
                    new_boundaries[j] = new_boundaries[j - 1] + 1

            # Update result in-place
            result["phase_boundaries"] = new_boundaries
            # Recompute confidence from transition signal at new positions
            max_sig = result["transition_signal"].max()
            if max_sig > 0:
                result["boundary_confidence"] = [
                    float(result["transition_signal"][min(b, T - 1)] / max_sig)
                    for b in new_boundaries
                ]

    return phase_stats


# ======================================================================= #
#  Visualization
# ======================================================================= #

def visualize_results(
    all_results: dict[str, dict],
    metadata: dict,
    phase_stats: dict,
    num_phases: int,
    output_dir: Path,
    detection_info: dict | None = None,
) -> None:
    """Generate visualizations of the phase segmentation."""
    import matplotlib.pyplot as plt

    demo_names = sorted(all_results.keys(), key=lambda s: int(s.split("_")[-1]))
    cmap = plt.cm.viridis

    # Plot 1: Phase timeline across demos
    fig, ax = plt.subplots(figsize=(14, max(6, len(demo_names) * 0.12)))
    for y_idx, demo_name in enumerate(demo_names):
        result = all_results[demo_name]
        num_frames = metadata[demo_name]["num_frames"]
        boundaries = result["phase_boundaries"]
        confidences = result["boundary_confidence"]
        anchors = result["anchor_indices"]

        for i, (b, c) in enumerate(zip(boundaries, confidences)):
            nt = b / (num_frames - 1)
            color = cmap(i / max(num_phases - 1, 1))
            size = 20 + 60 * c  # larger = more confident
            ax.scatter(nt, y_idx, c=[color], s=size, zorder=3, alpha=0.8)

        # Mark anchors with red edges
        for a in anchors:
            nt = a / (num_frames - 1)
            ax.scatter(nt, y_idx, facecolors="none", edgecolors="red",
                       s=80, linewidths=1.5, zorder=4)

        # Connect with line
        times = sorted([b / (num_frames - 1) for b in boundaries])
        ax.plot(times, [y_idx] * len(times), c="gray", linewidth=0.4, alpha=0.4)

    title = f"Phase Timeline ({num_phases} phases"
    if detection_info:
        title += ", auto-detected"
    title += ", red circles = anchors)"
    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Demo")
    ax.set_yticks(range(len(demo_names)))
    ax.set_yticklabels(demo_names, fontsize=5)
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / "phase_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Phase consistency (box plot of normalized times)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    phase_time_lists = []
    for i in range(num_phases):
        times = []
        for demo_name in demo_names:
            result = all_results[demo_name]
            nf = metadata[demo_name]["num_frames"]
            times.append(result["phase_boundaries"][i] / (nf - 1))
        phase_time_lists.append(times)
    bp = ax.boxplot(phase_time_lists,
                    tick_labels=[f"P{i}" for i in range(num_phases)],
                    patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(i / max(num_phases - 1, 1)))
    ax.set_xlabel("Phase")
    ax.set_ylabel("Normalized time")
    ax.set_title("Phase Temporal Consistency")

    # Plot 3: Confidence distribution per phase
    ax = axes[1]
    conf_lists = []
    for i in range(num_phases):
        confs = [all_results[d]["boundary_confidence"][i] for d in demo_names]
        conf_lists.append(confs)
    bp2 = ax.boxplot(conf_lists,
                     tick_labels=[f"P{i}" for i in range(num_phases)],
                     patch_artist=True)
    for i, patch in enumerate(bp2["boxes"]):
        patch.set_facecolor(cmap(i / max(num_phases - 1, 1)))
    ax.set_xlabel("Phase")
    ax.set_ylabel("Confidence (transition sharpness)")
    ax.set_title("Phase Boundary Confidence")

    plt.tight_layout()
    plt.savefig(output_dir / "phase_consistency.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 4: Example transition signal for first 3 demos
    fig, axes = plt.subplots(min(3, len(demo_names)), 1,
                             figsize=(12, 3 * min(3, len(demo_names))),
                             squeeze=False)
    for i, demo_name in enumerate(demo_names[:3]):
        ax = axes[i, 0]
        result = all_results[demo_name]
        sig = result["transition_signal"]
        ax.plot(sig, color="steelblue", linewidth=1, label="Transition signal")

        # Mark anchors
        for a in result["anchor_indices"]:
            ax.axvline(a, color="red", linestyle="--", linewidth=1, alpha=0.7)

        # Mark all phase boundaries
        for b in result["phase_boundaries"]:
            ax.axvline(b, color="green", linestyle=":", linewidth=0.8, alpha=0.6)

        # Show Otsu threshold if available
        if detection_info:
            ax.axhline(detection_info["threshold"], color="orange",
                       linestyle="-.", linewidth=0.8, alpha=0.6,
                       label=f"Otsu threshold ({detection_info['threshold']:.3f})")

        ax.set_title(f"{demo_name} — {result['num_anchors']} anchors, "
                     f"{num_phases} phases")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Transition magnitude")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "transition_signals.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 5: Prominence distribution with Otsu threshold (if auto-detected)
    if detection_info:
        fig, ax = plt.subplots(figsize=(10, 4))
        # Collect all prominences for histogram
        all_proms = []
        for demo_name in demo_names:
            result = all_results[demo_name]
            # Use anchor prominences (these are the surviving ones)
            # But we want ALL prominences for the histogram — reconstruct from detection_info
            all_proms.extend(result.get("anchor_prominences", []))

        # We stored the threshold and stats but not all raw prominences.
        # Plot what we have: the anchor prominences from surviving peaks.
        # For a better plot, we'll re-read from detection_info surviving counts.
        ax.hist(all_proms, bins=30, color="steelblue", alpha=0.7, edgecolor="white",
                label="Anchor prominences (above threshold)")
        ax.axvline(detection_info["threshold"], color="red", linestyle="--",
                   linewidth=2, label=f"Otsu threshold = {detection_info['threshold']:.4f}")
        ax.set_xlabel("Peak Prominence")
        ax.set_ylabel("Count")
        ax.set_title(f"Prominence Distribution — Auto-detected K={num_phases}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "prominence_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved visualizations to {output_dir}/")


# ======================================================================= #
#  Main
# ======================================================================= #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase segmentation with anchor + progress approach.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--embeddings-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--num-phases", type=int, default=None,
                        help="Number of phases per demo. If not set, auto-detected "
                             "from data via Otsu's method on peak prominences.")
    parser.add_argument("--min-anchor-distance", type=int, default=30,
                        help="Min frames between anchor boundaries")
    parser.add_argument("--prominence-percentile", type=float, default=70.0,
                        help="Percentile threshold for anchor prominence "
                             "(only used when --num-phases is manually set)")
    parser.add_argument("--smooth-window", type=int, default=11,
                        help="Savitzky-Golay smoothing window (odd number)")
    parser.add_argument("--visualize", action="store_true")

    # --- Improved scoring (v2) ---
    parser.add_argument("--scoring", type=str, default="composite",
                        choices=["baseline", "composite"],
                        help="Scoring method: 'composite' (multi-cue, default) or "
                             "'baseline' (magnitude only, legacy)")
    parser.add_argument("--metric", type=str, default="l2",
                        choices=["l2", "cosine"],
                        help="Distance metric for embedding comparisons")
    parser.add_argument("--ctx-k", type=int, default=10,
                        help="Context window half-size for contextual change")
    parser.add_argument("--w-mag", type=float, default=1.0,
                        help="Weight for magnitude change (baseline cue)")
    parser.add_argument("--w-ctx", type=float, default=0.5,
                        help="Weight for contextual change")
    parser.add_argument("--w-vel", type=float, default=0.3,
                        help="Weight for velocity")
    parser.add_argument("--w-acc", type=float, default=0.2,
                        help="Weight for acceleration")
    parser.add_argument("--w-prog", type=float, default=0.3,
                        help="Weight for progress slope")
    parser.add_argument("--w-bnd", type=float, default=0.3,
                        help="Weight for boundary probability (used with --boundary-refine)")

    # --- Boundary refinement ---
    parser.add_argument("--boundary-refine", action="store_true",
                        help="Enable two-pass boundary refinement: train a small "
                             "boundary classifier on initial anchors, then rerun "
                             "segmentation with predicted b_t")
    parser.add_argument("--boundary-refine-epochs", type=int, default=20,
                        help="Training epochs for boundary classifier")
    parser.add_argument("--boundary-refine-sigma", type=float, default=3.0,
                        help="Gaussian sigma for soft boundary pseudo-labels")

    # --- Progress method ---
    parser.add_argument("--progress-method", type=str, default="cumulative",
                        choices=["cumulative", "goal_distance"],
                        help="Progress computation method: 'cumulative' (motion-based, "
                             "default) or 'goal_distance' (legacy, unreliable)")

    # --- Alignment mode ---
    parser.add_argument("--align-mode", type=str, default="progress",
                        choices=["time", "progress"],
                        help="Alignment mode: 'progress' (cumulative progress, default) "
                             "or 'time' (normalized time, legacy)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = Path(args.embeddings_dir)

    with open(args.metadata) as f:
        metadata = json.load(f)

    print(f"Loaded {len(metadata)} demos")

    # Auto-detect or use manual K
    auto_detected = args.num_phases is None
    detection_info = None
    prom_threshold = None

    if auto_detected:
        num_phases, prom_threshold, detection_info = auto_detect_num_phases(
            metadata, embeddings_dir,
            min_anchor_distance=args.min_anchor_distance,
            smooth_window=args.smooth_window,
        )
    else:
        num_phases = args.num_phases

    scoring_label = args.scoring.upper()
    print(f"\nParams: {num_phases} phases {'(auto-detected)' if auto_detected else '(manual)'}, "
          f"min_anchor_dist={args.min_anchor_distance}, "
          f"smooth={args.smooth_window}, scoring={scoring_label}")
    if prom_threshold is not None:
        print(f"  Global prominence threshold: {prom_threshold:.4f}")
    else:
        print(f"  Prominence percentile: {args.prominence_percentile}")

    if args.scoring == "composite":
        print(f"  Composite weights: mag={args.w_mag}, ctx={args.w_ctx}, "
              f"vel={args.w_vel}, acc={args.w_acc}, prog={args.w_prog}, "
              f"bnd={args.w_bnd}")
        print(f"  Context window k={args.ctx_k}, metric={args.metric}")

    composite_weights = {
        "w_mag": args.w_mag, "w_ctx": args.w_ctx,
        "w_vel": args.w_vel, "w_acc": args.w_acc,
        "w_prog": args.w_prog, "w_bnd": args.w_bnd,
    }

    # --- Load all embeddings ---
    all_embeddings = {}
    demo_names_sorted = sorted(metadata.keys(), key=lambda s: int(s.split("_")[-1]))

    for demo_name in demo_names_sorted:
        info = metadata[demo_name]
        camera = info["camera"]
        emb_path = embeddings_dir / f"{demo_name}_{camera}.npy"
        if emb_path.exists():
            all_embeddings[demo_name] = np.load(emb_path)

    # ===================================================================
    # Pass 1: Initial segmentation
    # ===================================================================
    all_results = {}
    boundary_probs_per_demo = {}  # for boundary refinement

    for demo_name in demo_names_sorted:
        if demo_name not in all_embeddings:
            print(f"  {demo_name}: embeddings not found, skipping")
            continue

        embeddings = all_embeddings[demo_name]

        result = segment_demo(
            embeddings,
            num_phases=num_phases,
            min_anchor_distance=args.min_anchor_distance,
            prominence_percentile=args.prominence_percentile,
            prominence_threshold=prom_threshold,
            smooth_window=args.smooth_window,
            scoring=args.scoring,
            composite_weights=composite_weights,
            ctx_k=args.ctx_k,
            metric=args.metric,
        )

        all_results[demo_name] = result
        print(f"  {demo_name}: {result['num_anchors']} anchors -> "
              f"{len(result['phase_boundaries'])} phases at "
              f"{result['phase_boundaries']}")

    # ===================================================================
    # Pass 2 (optional): Boundary refinement
    # ===================================================================
    if args.boundary_refine and args.scoring == "composite":
        print("\n--- Boundary Refinement (Pass 2) ---")
        print("  Training boundary classifier on initial anchors...")

        all_anchors = {
            d: r["anchor_indices"] for d, r in all_results.items()
        }

        predict_fn = train_boundary_classifier(
            all_embeddings,
            all_anchors,
            epochs=args.boundary_refine_epochs,
            label_sigma=args.boundary_refine_sigma,
        )

        if predict_fn is not None:
            print("  Rerunning segmentation with learned boundary probs...")
            for demo_name in demo_names_sorted:
                if demo_name not in all_embeddings:
                    continue

                embeddings = all_embeddings[demo_name]
                b_probs = predict_fn(embeddings)
                boundary_probs_per_demo[demo_name] = b_probs

                result = segment_demo(
                    embeddings,
                    num_phases=num_phases,
                    min_anchor_distance=args.min_anchor_distance,
                    prominence_percentile=args.prominence_percentile,
                    prominence_threshold=prom_threshold,
                    smooth_window=args.smooth_window,
                    scoring="composite",
                    composite_weights=composite_weights,
                    ctx_k=args.ctx_k,
                    metric=args.metric,
                    boundary_probs=b_probs,
                )

                all_results[demo_name] = result
                print(f"  {demo_name}: {result['num_anchors']} anchors -> "
                      f"{len(result['phase_boundaries'])} phases (refined)")
        else:
            print("  Boundary refinement skipped (PyTorch unavailable)")

    elif args.boundary_refine and args.scoring != "composite":
        print("\nWARNING: --boundary-refine requires --scoring composite. "
              "Ignoring refinement.")

    # ===================================================================
    # Cross-demo alignment statistics
    # ===================================================================
    print(f"\nComputing cross-demo alignment (mode={args.align_mode})...")
    phase_stats = align_across_demos(
        all_results, metadata, num_phases,
        align_mode=args.align_mode,
        all_embeddings=all_embeddings,
        realign_boundaries=(args.align_mode == "progress"),
    )

    coord_label = "progress" if args.align_mode == "progress" else "time"
    print(f"\n{'Phase':>6} {'Mean '+coord_label:>12} {'Std '+coord_label:>12} "
          f"{'Mean Conf':>10} {'Min Conf':>10}")
    print("-" * 56)
    for i in range(num_phases):
        ps = phase_stats[i]
        print(f"  P{i:<4} {ps['mean_time']:>12.3f} {ps['std_time']:>12.3f} "
              f"{ps['mean_confidence']:>10.3f} {ps['min_confidence']:>10.3f}")

    # ===================================================================
    # Save diagnostic arrays (s_t, components) for debugging
    # ===================================================================
    diag_dir = output_dir / "segmentation_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    for demo_name, result in all_results.items():
        # Save transition signal / composite score
        np.save(diag_dir / f"{demo_name}_score.npy", result["transition_signal"])

        # Save per-component breakdown (first 3 demos or all if composite)
        if result.get("score_components"):
            comp = result["score_components"]
            for cname, carray in comp.items():
                np.save(diag_dir / f"{demo_name}_{cname}.npy", carray)

        # Save boundary probs if from refinement
        if demo_name in boundary_probs_per_demo:
            np.save(
                diag_dir / f"{demo_name}_boundary_probs.npy",
                boundary_probs_per_demo[demo_name],
            )

    print(f"\nSaved diagnostic arrays to {diag_dir}/")

    # ===================================================================
    # Segmentation quality summary
    # ===================================================================
    anchor_counts = [r["num_anchors"] for r in all_results.values()]
    mean_prominence = np.mean([
        np.mean(r["anchor_prominences"]) if r["anchor_prominences"] else 0
        for r in all_results.values()
    ])
    print(f"\nSegmentation Quality Summary:")
    print(f"  Anchors per demo: min={min(anchor_counts)}, "
          f"mean={np.mean(anchor_counts):.1f}, max={max(anchor_counts)}")
    print(f"  Mean anchor prominence: {mean_prominence:.4f}")
    mean_conf = np.mean([ps["mean_confidence"] for ps in phase_stats.values()])
    mean_std = np.mean([ps["std_time"] for ps in phase_stats.values()])
    print(f"  Cross-demo alignment: mean_conf={mean_conf:.3f}, "
          f"mean_std_{coord_label}={mean_std:.3f}")

    # ===================================================================
    # Save results (backward-compatible JSON format)
    # ===================================================================
    config = {
        "num_phases": num_phases,
        "auto_detected": auto_detected,
        "min_anchor_distance": args.min_anchor_distance,
        "smooth_window": args.smooth_window,
        "scoring": args.scoring,
        "metric": args.metric,
        "align_mode": args.align_mode,
    }
    if auto_detected:
        config["prominence_threshold"] = round(prom_threshold, 6)
        config["detection_method"] = detection_info.get("threshold_method", "auto")
    else:
        config["prominence_percentile"] = args.prominence_percentile
    if args.scoring == "composite":
        config["composite_weights"] = composite_weights
        config["ctx_k"] = args.ctx_k
        config["boundary_refine"] = args.boundary_refine

    output_data = {
        "config": config,
        "phase_stats": {str(k): v for k, v in phase_stats.items()},
        "demos": {},
    }
    for demo_name, result in all_results.items():
        output_data["demos"][demo_name] = {
            "num_frames": int(metadata[demo_name]["num_frames"]),
            "phase_boundaries": [int(b) for b in result["phase_boundaries"]],
            "boundary_confidence": [round(float(c), 4) for c in result["boundary_confidence"]],
            "anchor_indices": [int(a) for a in result["anchor_indices"]],
            "num_anchors": int(result["num_anchors"]),
        }

    out_path = output_dir / "phase_segmentation.json"
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {out_path}")

    if args.visualize:
        print("\nGenerating visualizations...")
        visualize_results(
            all_results, metadata, phase_stats, num_phases, output_dir,
            detection_info=detection_info,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
