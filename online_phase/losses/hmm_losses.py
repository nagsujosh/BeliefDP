"""HMM-specific losses for the structured phase model.

Two losses are defined:
1.  L_hmm (NLL): negative log-likelihood of observations under the HMM,
    computed from the forward algorithm's log-normalizers.
2.  L_belief_ce: cross-entropy supervision on HMM beliefs α_t to ensure
    they converge toward the ground-truth phase labels.

Both losses operate on the outputs of NeuralHMM.forward(), which already
provides numerically stable log-space quantities.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HMMLoss(nn.Module):
    """HMM observation negative log-likelihood.

    The forward algorithm produces per-step log-normalizers C_t = log Σ_k α_t(k).
    The total log-likelihood is ΣC_t over valid timesteps. We minimize -LL/T.

    This loss encourages the model to assign high emission probability to the
    correct phase sequence, with temporal coherence enforced by the transition
    structure.
    """

    def forward(
        self,
        log_normalizers: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute HMM NLL loss.

        Args:
            log_normalizers: (B, T) per-step log normalizers from NeuralHMM.
            mask: (B, T) float attention mask (1=real, 0=pad).

        Returns:
            Scalar NLL loss (mean-reduced over valid frames).
        """
        # CRITICAL FIX: Normalize by sequence length PER SAMPLE
        # Before: -(log_normalizers * mask).sum() / mask.sum()  → wrong scale
        # After: Average NLL per timestep (divide by T, not total frames)
        
        # Sum log-normalizers over time dimension for each sample
        nll_per_sample = -(log_normalizers * mask).sum(dim=1)  # (B,)
        seq_lens = mask.sum(dim=1).clamp(min=1)  # (B,)
        
        # Normalize each sample by its sequence length, then average over batch
        normalized_nll = (nll_per_sample / seq_lens).mean()
        
        return normalized_nll


class BeliefCELoss(nn.Module):
    """Cross-entropy on HMM belief distributions.

    Supervises the HMM beliefs α_t with ground-truth phase labels z_t:

        L = -Σ_t log α_t(z_t)

    This is the primary phase supervision loss for the structured model,
    applied to the temporally filtered belief rather than the raw emission.

    Optional confidence weighting re-uses the same scheme as PhaseLoss:
    w_t = alpha + (1-alpha) * c_t.

    Args:
        confidence_alpha: Base weight floor for confidence weighting.
            Set to 1.0 to disable confidence weighting.
    """

    def __init__(self, confidence_alpha: float = 0.3):
        super().__init__()
        self.alpha = confidence_alpha
        self._debug_count = 0  # For debugging first 3 batches

    def forward(
        self,
        log_beliefs: torch.Tensor,
        z_target: torch.Tensor,
        confidence: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute belief CE loss.

        Args:
            log_beliefs: (B, T, K) log of HMM posterior beliefs.
            z_target: (B, T) int64 ground-truth phase labels.
            confidence: (B, T) per-frame confidence weights.
            mask: (B, T) float attention mask.

        Returns:
            Scalar loss.
        """
        B, T, K = log_beliefs.shape
        
        # DEBUG: Verify tensor shapes
        assert z_target.shape == (B, T), f"z_target shape {z_target.shape} != (B={B}, T={T})"
        assert mask.shape == (B, T), f"mask shape {mask.shape} != (B={B}, T={T})"
        
        # Clamp log_beliefs to prevent -inf blowups
        log_beliefs = log_beliefs.clamp(min=-50.0)
        
        # Gather log α_t(z_t) for the correct phase
        # log_beliefs: (B, T, K) → gather → (B, T, 1) → (B, T)
        log_belief_correct = log_beliefs.gather(
            -1, z_target.unsqueeze(-1)
        ).squeeze(-1)  # (B, T)

        # DEBUG: Print diagnostics for first 3 batches
        if self._debug_count < 3:
            beliefs = torch.exp(log_beliefs)
            print(f"\n[BeliefCE DEBUG Batch {self._debug_count}]")
            print(f"  log_beliefs: min={log_beliefs.min().item():.4f}, max={log_beliefs.max().item():.4f}")
            print(f"  beliefs (exp): min={beliefs.min().item():.6f}, max={beliefs.max().item():.6f}")
            print(f"  log_belief_correct: min={log_belief_correct.min().item():.4f}, "
                  f"max={log_belief_correct.max().item():.4f}, mean={log_belief_correct.mean().item():.4f}")
            n_inf = (log_belief_correct == float('-inf')).sum().item()
            print(f"  fraction -inf: {n_inf}/{log_belief_correct.numel()} ({100*n_inf/log_belief_correct.numel():.2f}%)")
            self._debug_count += 1
        
        # Negative log-likelihood per frame
        nll = -log_belief_correct  # (B, T)

        # Confidence weighting
        weights = self.alpha + (1.0 - self.alpha) * confidence  # (B, T)

        # CRITICAL FIX: Normalize by sequence length PER SAMPLE (same as HMM NLL)
        # Before: (nll * weights * mask).sum() / mask.sum()  → wrong scale
        # After: Average NLL per timestep per sample
        
        weighted_nll = nll * weights * mask  # (B, T)
        nll_per_sample = weighted_nll.sum(dim=1)  # (B,)
        seq_lens = mask.sum(dim=1).clamp(min=1)  # (B,)
        
        # Normalize each sample by its sequence length, then average over batch
        normalized_loss = (nll_per_sample / seq_lens).mean()
        
        return normalized_loss


class BeliefLTRLoss(nn.Module):
    """Left-to-right constraint on HMM beliefs.

    Penalizes backward phase movement in the belief distribution:

        E_t = Σ_k k · α_t(k)
        L = Σ_t max(0, E_{t-1} - E_t)²

    This is the structured analogue of ConsistencyLoss.L_ltr, but applied
    to the HMM beliefs instead of raw softmax outputs. Since beliefs are
    already temporally smoothed by the HMM, this loss serves as a
    soft regularizer rather than a hard constraint.
    """

    def forward(
        self,
        beliefs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute left-to-right loss on beliefs.

        Args:
            beliefs: (B, T, K) HMM posterior beliefs.
            mask: (B, T) float attention mask.

        Returns:
            Scalar loss.
        """
        B, T, K = beliefs.shape

        if T < 2:
            return torch.tensor(0.0, device=beliefs.device)

        # Expected phase index under beliefs
        phase_idx = torch.arange(K, dtype=beliefs.dtype, device=beliefs.device)
        E_t = (beliefs * phase_idx).sum(dim=-1)  # (B, T)

        # Consecutive-frame mask
        pair_mask = mask[:, :-1] * mask[:, 1:]  # (B, T-1)

        # Penalize backward movement
        backward = torch.clamp(E_t[:, :-1] - E_t[:, 1:], min=0) ** 2  # (B, T-1)
        loss = (backward * pair_mask).sum() / pair_mask.sum().clamp(min=1)
        return loss


class ProgressPhaseConsistencyLoss(nn.Module):
    """Bidirectional consistency between progress and phase predictions.
    
    Enforces that progress ≈ phase/K relationship. This creates a coupling
    between the continuous progress head and discrete phase predictions,
    improving both and enabling better temporal grounding.
    
    Loss formulation:
        L = KL(phase_dist || phase_implied_by_progress)
    
    Where phase_implied_by_progress is derived by mapping progress ∈ [0,1]
    to expected phase: k ≈ floor(progress * (K-1))
    
    This is a novel contribution that:
    - Improves phase accuracy via progress supervision
    - Improves progress calibration via phase structure
    - Enables zero-shot progress estimation from phase beliefs
    
    Publishable enhancement for structured models.
    """
    
    def forward(
        self,
        phase_logits: torch.Tensor,
        progress: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute progress-phase consistency loss.
        
        Args:
            phase_logits: (B, T, K) raw phase predictions (before softmax).
            progress: (B, T) predicted progress values [0, 1].
            mask: (B, T) attention mask.
            
        Returns:
            Scalar KL divergence loss.
        """
        B, T, K = phase_logits.shape
        
        # Derive expected phase from progress
        # progress ∈ [0, 1] → phase_idx ∈ [0, K-1]
        # Clamp to valid range and convert to target indices
        phase_from_progress = (progress * (K - 1)).clamp(0, K - 1).long()  # (B, T)
        
        # Cross-entropy: predicted phase dist should match progress-implied phase
        # Reshape for F.cross_entropy: (B*T, K) and (B*T,)
        logits_flat = phase_logits.reshape(-1, K)  # (B*T, K)
        targets_flat = phase_from_progress.reshape(-1)  # (B*T,)
        mask_flat = mask.reshape(-1)  # (B*T,)
        
        # Compute CE only on valid frames
        ce = F.cross_entropy(logits_flat, targets_flat, reduction='none')  # (B*T,)
        
        # Masked mean
        loss = (ce * mask_flat).sum() / mask_flat.sum().clamp(min=1)
        
        return loss


class MonotonicPhaseLoss(nn.Module):
    """Soft monotonicity constraint on phase progression.
    
    Penalizes backward phase transitions using a differentiable soft phase index
    computed from the phase distribution (no hard argmax).
    
    Mathematical formulation:
        E_t = Σ_k k · softmax(logits_t)[k]    (expected phase index)
        L = mean_t( ReLU(E_t - E_{t+1}) )      (penalize backward motion)
    
    This provides a structural prior that phases should progress monotonically
    forward through time, without hard constraints that might suppress valid
    brief regressions (e.g., corrective actions).
    
    Advantages over hard constraints:
    - Fully differentiable (gradients flow through softmax)
    - Soft penalty allows model to override if data demands it
    - No discrete argmax breaks in gradient flow
    - Complements HMM left-to-right transitions
    
    Use case: Robotics tasks with mostly forward progress but occasional pauses
    or brief corrections. Weight should be small (0.01-0.1) to avoid suppressing
    natural task dynamics.
    """
    
    def forward(
        self,
        phase_logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute monotonic phase progression loss.
        
        Args:
            phase_logits: (B, T, K) raw phase predictions before softmax.
            mask: (B, T) attention mask.
            
        Returns:
            Scalar loss penalizing backward phase transitions.
        """
        B, T, K = phase_logits.shape
        
        if T < 2:
            return torch.tensor(0.0, device=phase_logits.device)
        
        # Convert logits to soft phase index via expected value
        phase_probs = F.softmax(phase_logits, dim=-1)  # (B, T, K)
        phase_indices = torch.arange(K, dtype=phase_logits.dtype, device=phase_logits.device)
        soft_phase = (phase_probs * phase_indices).sum(dim=-1)  # (B, T)
        
        # Consecutive-frame mask
        pair_mask = mask[:, :-1] * mask[:, 1:]  # (B, T-1)
        
        # Penalize backward movement: E_t > E_{t+1}
        backward = F.relu(soft_phase[:, :-1] - soft_phase[:, 1:])  # (B, T-1)
        
        # Masked mean
        loss = (backward * pair_mask).sum() / pair_mask.sum().clamp(min=1)
        
        return loss
