"""Tests for label generation correctness."""
from __future__ import annotations

import numpy as np
import pytest

from online_phase.data.labels import (
    generate_phase_labels,
    generate_progress_labels,
    generate_confidence_weights,
)


class TestPhaseLabels:
    """Test phase index assignment from boundaries."""

    def test_basic_assignment(self):
        """Phase labels cover all frames with correct boundaries."""
        T = 100
        boundaries = [20, 50, 80, 99]
        z = generate_phase_labels(boundaries, T)

        assert z.shape == (T,)
        assert z.dtype == np.int64
        assert set(z.tolist()) == {0, 1, 2, 3}

        # Check boundary alignment
        assert z[0] == 0
        assert z[20] == 0
        assert z[21] == 1
        assert z[50] == 1
        assert z[51] == 2
        assert z[80] == 2
        assert z[81] == 3
        assert z[99] == 3

    def test_all_frames_covered(self):
        """Every frame gets exactly one phase label."""
        T = 500
        boundaries = [50, 120, 200, 300, 400, 499]
        z = generate_phase_labels(boundaries, T)

        assert z.shape == (T,)
        assert len(set(z.tolist())) == 6

        # No gaps: every frame from 0 to T-1 is assigned
        for t in range(T):
            assert 0 <= z[t] < 6

    def test_single_phase(self):
        """Single phase = all zeros."""
        T = 100
        boundaries = [99]
        z = generate_phase_labels(boundaries, T)
        assert np.all(z == 0)

    def test_k_equals_11(self):
        """Match actual dataset: K=11 phases."""
        T = 421
        # Simulate boundaries like actual data
        boundaries = [33, 65, 117, 147, 178, 209, 266, 312, 342, 372, 420]
        z = generate_phase_labels(boundaries, T)

        assert z.shape == (T,)
        assert set(z.tolist()) == set(range(11))
        assert z[0] == 0
        assert z[420] == 10


class TestProgressLabels:
    """Test progress label properties."""

    def test_monotonic(self):
        """Progress should be monotonically non-decreasing."""
        # Create simple embeddings moving from start to goal
        T = 100
        embeddings = np.linspace(0, 1, T).reshape(T, 1).repeat(2048, axis=1)
        p = generate_progress_labels(embeddings.astype(np.float32))

        assert p.shape == (T,)
        # Monotonic: p[t+1] >= p[t]
        diffs = np.diff(p)
        assert np.all(diffs >= -1e-6), f"Progress not monotonic: min diff = {diffs.min()}"

    def test_range(self):
        """Progress should be in [0, 1]."""
        T = 100
        embeddings = np.random.randn(T, 2048).astype(np.float32)
        p = generate_progress_labels(embeddings)

        assert p.min() >= 0.0
        assert p.max() <= 1.0

    def test_endpoints(self):
        """Progress at end should be 1.0."""
        T = 50
        embeddings = np.linspace(0, 10, T).reshape(T, 1).repeat(2048, axis=1)
        p = generate_progress_labels(embeddings.astype(np.float32))

        assert abs(p[-1] - 1.0) < 1e-5, f"Progress at end = {p[-1]}, expected 1.0"


class TestConfidenceWeights:
    """Test confidence weight computation."""

    def test_range(self):
        """Confidence should be in [c_min, c_max]."""
        T = 200
        anchors = [30, 80, 150]
        c = generate_confidence_weights(anchors, T, c_min=0.3, c_max=1.0)

        assert c.shape == (T,)
        assert c.min() >= 0.3 - 1e-6
        assert c.max() <= 1.0 + 1e-6

    def test_high_near_anchors(self):
        """Confidence should peak at anchor locations."""
        T = 200
        anchors = [50, 100, 150]
        c = generate_confidence_weights(anchors, T, c_min=0.3, c_max=1.0, anchor_sigma=6.0)

        for a in anchors:
            # At anchor: should be close to c_max
            assert c[a] > 0.95, f"Confidence at anchor {a} = {c[a]}"

    def test_low_far_from_anchors(self):
        """Confidence far from any anchor should be near c_min."""
        T = 500
        anchors = [50, 450]
        c = generate_confidence_weights(anchors, T, c_min=0.3, c_max=1.0, anchor_sigma=6.0)

        # Frame 250 is far from both anchors
        assert c[250] < 0.35, f"Confidence far from anchors = {c[250]}"

    def test_no_anchors(self):
        """With no anchors, all frames get c_min."""
        T = 100
        c = generate_confidence_weights([], T, c_min=0.3)
        assert np.allclose(c, 0.3)
