"""Differentiable Neural HMM inference layer for structured phase tracking.

Implements a left-to-right Hidden Markov Model where:
- Transition probabilities are modulated by boundary predictions (η_t)
- Emission likelihoods come from a neural emission head
- Forward algorithm is fully differentiable for end-to-end training
- Belief vectors α_t ∈ Δ^{K-1} are computed per timestep

Mathematical formulation:
    Transition: A_t(k → k+1) = η_t,  A_t(k → k) = 1 - η_t - ρ,  A_t(k → k-1) = ρ
    where η_t = η_min + (η_max - η_min) · b̂_t
    
    Forward:  α̃_t = A_t^T @ α_{t-1}        (predict)
              α_t ∝ α̃_t ⊙ p(o_t | z_t)     (update)
    
    All operations use log-space for numerical stability.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DurationPrior:
    """Duration prior for HSMM-style self-transition modulation.

    Modifies the self-transition probability based on how long the model
    has stayed in the current phase. After spending d frames in phase k,
    the self-transition log-probability is adjusted by log D_k(d), where
    D_k is a duration model (Gaussian or log-normal).

    This encourages phase durations to match training statistics and
    reduces jitter from premature or delayed transitions.

    Args:
        K: Number of phases.
        duration_model: "gaussian" or "lognormal".
        duration_strength: Scaling factor for the duration prior (0 = off).
        min_std: Minimum standard deviation (frames) to prevent collapse.
    """

    def __init__(
        self,
        K: int,
        duration_model: str = "gaussian",
        duration_strength: float = 1.0,
        min_std: float = 5.0,
    ):
        self.K = K
        self.duration_model = duration_model
        self.duration_strength = duration_strength
        self.min_std = min_std

        # Per-phase duration stats: (mean, std) in frames
        # These should be set from training data via set_duration_stats()
        self.phase_means = None
        self.phase_stds = None

    def set_duration_stats(
        self, means: list[float] | np.ndarray, stds: list[float] | np.ndarray
    ) -> None:
        """Set per-phase duration statistics from aligned training demos.

        Args:
            means: (K,) mean duration in frames per phase.
            stds: (K,) std duration in frames per phase.
        """
        import numpy as np
        self.phase_means = np.array(means, dtype=np.float64)
        self.phase_stds = np.maximum(np.array(stds, dtype=np.float64), self.min_std)

    def log_duration_factor(self, phase: int, elapsed: int) -> float:
        """Compute log D_k(d): log-probability of duration d in phase k.

        Returns a value that should be ADDED to the log self-transition
        probability. Negative values discourage staying; positive values
        encourage staying.

        Args:
            phase: Current phase index.
            elapsed: Frames spent in current phase.

        Returns:
            Log-scale adjustment to self-transition probability.
        """
        if self.phase_means is None or self.duration_strength == 0:
            return 0.0

        import numpy as np

        mu = self.phase_means[phase]
        sigma = self.phase_stds[phase]
        d = float(elapsed)

        if self.duration_model == "gaussian":
            # Gaussian: log p(d) ∝ -0.5 * ((d - μ) / σ)^2
            z = (d - mu) / sigma
            log_p = -0.5 * z * z
        elif self.duration_model == "lognormal":
            # Log-normal: log p(d) ∝ -0.5 * ((ln(d) - ln(μ)) / σ_ln)^2
            if d < 1:
                d = 1.0
            log_d = np.log(d)
            log_mu = np.log(max(mu, 1.0))
            sigma_ln = sigma / max(mu, 1.0)  # approximate
            sigma_ln = max(sigma_ln, 0.1)
            z = (log_d - log_mu) / sigma_ln
            log_p = -0.5 * z * z
        else:
            return 0.0

        # Scale and clamp for stability
        result = self.duration_strength * log_p
        return float(np.clip(result, -5.0, 2.0))


class NeuralHMM(nn.Module):
    """Differentiable Neural HMM with boundary-modulated transitions.

    The transition matrix is time-varying: the advance probability η_t
    is a linear interpolation between η_min and η_max, controlled by
    the boundary probability b̂_t from the boundary head.

    All forward computations use log-space to avoid numerical underflow
    over long sequences.

    Optionally supports a duration prior (HSMM-like) that modulates
    self-transition probability based on elapsed time in current phase.

    Args:
        K: Number of discrete phases (states).
        eta_min: Base self-transition probability complement (minimum advance prob).
        eta_max: Maximum advance probability (when boundary prob = 1).
        rho: Backward transition probability (very small, for robustness).
        log_space: If True, perform forward algorithm in log-space (recommended).
        use_duration_prior: If True, enable duration-aware transitions.
        duration_model: "gaussian" or "lognormal" (used if use_duration_prior).
        duration_strength: Scaling factor for duration prior.
        min_std: Minimum phase duration std (frames).
    """

    def __init__(
        self,
        K: int = 11,
        eta_min: float = 0.02,
        eta_max: float = 0.35,
        rho: float = 0.001,
        log_space: bool = True,
        use_duration_prior: bool = False,
        duration_model: str = "gaussian",
        duration_strength: float = 1.0,
        min_std: float = 5.0,
    ):
        super().__init__()
        self.K = K
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.rho = rho
        self.log_space = log_space

        # Duration prior (HSMM-like)
        self.use_duration_prior = use_duration_prior
        self.duration_prior = None
        if use_duration_prior:
            self.duration_prior = DurationPrior(
                K=K,
                duration_model=duration_model,
                duration_strength=duration_strength,
                min_std=min_std,
            )

    def _build_transition_matrix(
        self, boundary_probs: torch.Tensor
    ) -> torch.Tensor:
        """Build time-varying transition matrices from boundary probabilities.

        The advance probability η_t interpolates between η_min and η_max
        based on the boundary signal:
            η_t = η_min + (η_max - η_min) · b̂_t

        Transition matrix A_t is K×K where:
            A_t[k, k]   = 1 - η_t - ρ     (stay)
            A_t[k, k+1] = η_t              (advance)
            A_t[k, k-1] = ρ                (backtrack)
        Edge states are clamped appropriately.

        Args:
            boundary_probs: (B, T) boundary probabilities in [0, 1].

        Returns:
            (B, T, K, K) transition matrices A_t[i, j] = P(z_t=j | z_{t-1}=i).
        """
        B, T = boundary_probs.shape
        K = self.K
        device = boundary_probs.device

        # η_t = η_min + (η_max - η_min) · b̂_t, shape (B, T)
        eta = self.eta_min + (self.eta_max - self.eta_min) * boundary_probs
        eta = eta.clamp(0.001, 0.95)  # Safety clamp
        rho = self.rho

        # Build transition matrices: (B, T, K, K)
        # Start with zeros
        A = torch.zeros(B, T, K, K, device=device, dtype=boundary_probs.dtype)

        # Expand eta for broadcasting: (B, T, 1)
        eta_exp = eta.unsqueeze(-1)

        # Interior states (1 to K-2): full left-to-right structure
        for k in range(K):
            if k == 0:
                # First state: no backtrack
                A[:, :, k, k] = 1.0 - eta
                if K > 1:
                    A[:, :, k, k + 1] = eta
            elif k == K - 1:
                # Last state: absorbing with small backtrack
                A[:, :, k, k] = 1.0 - rho
                A[:, :, k, k - 1] = rho
            else:
                # Interior: stay + advance + backtrack
                stay = (1.0 - eta - rho).clamp(min=0.01)
                A[:, :, k, k] = stay
                A[:, :, k, k + 1] = eta
                A[:, :, k, k - 1] = rho

        # Normalize rows to sum to 1 (handles edge clamping effects)
        A = A / A.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return A

    def forward(
        self,
        emission_log_probs: torch.Tensor,
        boundary_probs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run differentiable forward algorithm.

        Computes belief distributions α_t for all timesteps using the
        forward algorithm with boundary-modulated transitions.

        All computation is done in log-space for numerical stability:
            log α̃_t = logsumexp_k(log A_t[k, :] + log α_{t-1}[k])
            log α_t  = log α̃_t + log p(o_t | z_t) - log Z_t

        Args:
            emission_log_probs: (B, T, K) log p(o_t | z_t = k) from emission head.
            boundary_probs: (B, T) boundary probabilities b̂_t ∈ [0, 1].
            mask: (B, T) attention mask, 1.0 for real frames, 0.0 for padding.
                  Padded frames preserve the previous belief without update.

        Returns:
            Dict with:
                beliefs: (B, T, K) posterior beliefs α_t (probability simplex).
                log_beliefs: (B, T, K) log α_t (for loss computation).
                log_normalizers: (B, T) log Z_t per timestep (for HMM NLL loss).
                transition_matrices: (B, T, K, K) A_t for analysis.
        """
        B, T, K = emission_log_probs.shape
        device = emission_log_probs.device
        dtype = emission_log_probs.dtype

        # Build transition matrices: (B, T, K, K)
        A = self._build_transition_matrix(boundary_probs)

        # Initialize α_0 as one-hot on phase 0 (start of task)
        # In log-space: log(1) = 0 for phase 0, log(0) = -inf for others
        log_alpha = torch.full(
            (B, K), -1e6, device=device, dtype=dtype
        )
        log_alpha[:, 0] = 0.0  # Start in phase 0

        # Duration tracking for duration prior (per-batch, per-phase elapsed counts)
        # Shape: (B, K) — elapsed frames in each phase
        if self.use_duration_prior and self.duration_prior is not None:
            elapsed = torch.zeros(B, K, device=device, dtype=torch.long)
            # Initialize: start in phase 0
            elapsed[:, 0] = 1

        # Storage for outputs
        all_log_beliefs = []
        all_log_normalizers = []

        for t in range(T):
            if t == 0:
                # First timestep: just apply emission to initial state
                log_alpha_pred = log_alpha
            else:
                # Apply duration prior to transition matrix if enabled
                A_t = A[:, t]  # (B, K, K)

                if self.use_duration_prior and self.duration_prior is not None:
                    # Modulate self-transition based on elapsed duration
                    # For the MAP phase (argmax of current belief), adjust A[k,k]
                    current_phase = torch.argmax(log_alpha, dim=-1)  # (B,)
                    A_t = A_t.clone()
                    for b_idx in range(B):
                        k = current_phase[b_idx].item()
                        d = elapsed[b_idx, k].item()
                        log_dur = self.duration_prior.log_duration_factor(k, d)
                        if log_dur != 0.0:
                            # Modulate self-transition in log-space
                            old_stay = A_t[b_idx, k, k].item()
                            new_stay = old_stay * np.exp(np.clip(log_dur, -3, 1))
                            new_stay = max(min(new_stay, 0.99), 0.01)
                            A_t[b_idx, k, k] = new_stay
                            # Renormalize row
                            row_sum = A_t[b_idx, k].sum()
                            if row_sum > 0:
                                A_t[b_idx, k] = A_t[b_idx, k] / row_sum
                else:
                    A_t = A[:, t]

                # Predict step: α̃_t = A_t^T @ α_{t-1}
                # In log-space: log α̃_t[j] = logsumexp_k(log A_t[k,j] + log α_{t-1}[k])
                log_A = torch.log(A_t.clamp(min=1e-10))  # (B, K, K)
                # log_A[b, k, j] + log_alpha[b, k] → logsumexp over k → log_alpha_pred[b, j]
                log_alpha_pred = torch.logsumexp(
                    log_A + log_alpha.unsqueeze(-1),  # (B, K, 1) + (B, K, K) → (B, K, K)
                    dim=-2,  # sum over source states k
                )  # (B, K)

            # Update step: log α_t = log α̃_t + log p(o_t | z_t) - log Z_t
            log_joint = log_alpha_pred + emission_log_probs[:, t]  # (B, K)

            # Normalize
            log_Z = torch.logsumexp(log_joint, dim=-1, keepdim=True)  # (B, 1)
            log_alpha_new = log_joint - log_Z

            # Handle padding: if frame is padded, keep previous belief
            if mask is not None:
                m = mask[:, t].unsqueeze(-1)  # (B, 1)
                log_alpha = m * log_alpha_new + (1.0 - m) * log_alpha
                log_Z_out = m.squeeze(-1) * log_Z.squeeze(-1)
            else:
                log_alpha = log_alpha_new
                log_Z_out = log_Z.squeeze(-1)

            all_log_beliefs.append(log_alpha)
            all_log_normalizers.append(log_Z_out)

            # Update elapsed duration counters for duration prior
            if self.use_duration_prior and self.duration_prior is not None:
                current_phase = torch.argmax(log_alpha, dim=-1)  # (B,)
                for b_idx in range(B):
                    k = current_phase[b_idx].item()
                    # Increment elapsed for current phase, reset others
                    new_elapsed = torch.zeros(K, device=device, dtype=torch.long)
                    new_elapsed[k] = elapsed[b_idx, k] + 1
                    elapsed[b_idx] = new_elapsed

        # Stack: (B, T, K) and (B, T)
        log_beliefs = torch.stack(all_log_beliefs, dim=1)
        log_normalizers = torch.stack(all_log_normalizers, dim=1)
        beliefs = torch.exp(log_beliefs)

        return {
            "beliefs": beliefs,              # (B, T, K) probability simplex
            "log_beliefs": log_beliefs,      # (B, T, K) for CE loss
            "log_normalizers": log_normalizers,  # (B, T) for NLL loss
            "transition_matrices": A,        # (B, T, K, K) for analysis
        }
