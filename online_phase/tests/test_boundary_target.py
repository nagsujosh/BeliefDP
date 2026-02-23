"""Tests for boundary soft-target generation."""
from __future__ import annotations

import numpy as np
import pytest

from online_phase.data.labels import generate_boundary_targets


class TestBoundaryTargets:
    """Test Gaussian soft boundary target generation."""

    def test_peak_at_boundary(self):
        """b_t should be ~1.0 at boundary frames."""
        T = 200
        boundaries = [50, 100, 150, 199]
        b = generate_boundary_targets(boundaries, T, sigma_b=4.0)

        # Interior boundaries (not terminal) should peak at 1.0
        assert abs(b[50] - 1.0) < 1e-5
        assert abs(b[100] - 1.0) < 1e-5
        assert abs(b[150] - 1.0) < 1e-5

    def test_decay_away_from_boundary(self):
        """b_t should decay to near 0 at ±3σ from boundaries."""
        T = 200
        sigma_b = 4.0
        boundaries = [100, 199]
        b = generate_boundary_targets(boundaries, T, sigma_b=sigma_b)

        # At ±3σ ≈ 12 frames, Gaussian ≈ exp(-4.5) ≈ 0.011
        assert b[100] > 0.99
        assert b[100 - 12] < 0.02
        assert b[100 + 12] < 0.02

    def test_no_peak_at_terminal(self):
        """Terminal boundary (T-1) should NOT produce a peak in b_t."""
        T = 200
        boundaries = [50, 199]  # 199 is terminal
        b = generate_boundary_targets(boundaries, T, sigma_b=4.0)

        # Frame 50 should peak
        assert b[50] > 0.99

        # Terminal frame should NOT peak (it's excluded)
        # Frame 199 only gets influence from boundary at 50 (which is far away)
        assert b[199] < 0.01, f"Terminal boundary peak = {b[199]}"

    def test_shape_and_range(self):
        """Output shape should match T, values in [0, 1]."""
        T = 500
        boundaries = [100, 200, 300, 400, 499]
        b = generate_boundary_targets(boundaries, T, sigma_b=4.0)

        assert b.shape == (T,)
        assert b.dtype == np.float32
        assert b.min() >= 0.0
        assert b.max() <= 1.0

    def test_multiple_nearby_boundaries(self):
        """When boundaries are close, soft targets overlap but max is still ≤1."""
        T = 100
        boundaries = [30, 35, 99]  # Two nearby boundaries
        b = generate_boundary_targets(boundaries, T, sigma_b=4.0)

        # Max should not exceed 1.0 (we take max, not sum)
        assert b.max() <= 1.0 + 1e-6

        # Both boundaries should create peaks
        assert b[30] > 0.99
        assert b[35] > 0.99

    def test_no_boundaries(self):
        """No interior boundaries → all zeros."""
        T = 100
        boundaries = [99]  # Only terminal
        b = generate_boundary_targets(boundaries, T, sigma_b=4.0)
        assert np.allclose(b, 0.0)

    def test_sigma_effect(self):
        """Larger sigma → wider peaks."""
        T = 200
        boundaries = [100, 199]

        b_narrow = generate_boundary_targets(boundaries, T, sigma_b=2.0)
        b_wide = generate_boundary_targets(boundaries, T, sigma_b=8.0)

        # 20 frames from boundary: wide should have higher value
        assert b_wide[120] > b_narrow[120]
