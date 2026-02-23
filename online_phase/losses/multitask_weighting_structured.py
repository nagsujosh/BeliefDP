"""Multi-task loss orchestrator for the structured (Neural HMM) model.

Adds HMM-specific losses:
- L_hmm: negative log-likelihood from the forward algorithm
- L_belief_ce: cross-entropy on HMM beliefs (temporally coherent supervision)
- L_belief_ltr: left-to-right constraint on HMM belief expectations

The original sub-losses (progress, boundary, consistency) are reused as-is.
The raw-emission PhaseLoss supervises the emission head, and BeliefCELoss
supervises the HMM output. Both are needed for stable training.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from online_phase.losses.phase_losses import PhaseLoss
from online_phase.losses.progress_losses import ProgressLoss
from online_phase.losses.boundary_losses import BoundaryLoss
from online_phase.losses.consistency_losses import ConsistencyLoss
from online_phase.losses.hmm_losses import (
    HMMLoss, BeliefCELoss, BeliefLTRLoss, ProgressPhaseConsistencyLoss, MonotonicPhaseLoss
)


class MultiTaskLossStructured(nn.Module):
    """Orchestrates all sub-losses for the structured phase model.

    Loss components:
        L_total = w_emission * L_emission      (CE on raw emission logits)
                + w_belief  * L_belief_ce      (CE on HMM beliefs)
                + w_hmm     * L_hmm            (HMM NLL)
                + w_progress * L_progress      (Huber + ranking + mono)
                + w_boundary * L_boundary      (BCE + sparsity)
                + w_consistency * L_consistency (link + LTR on emissions)
                + w_belief_ltr * L_belief_ltr  (LTR on HMM beliefs)

    Recommended schedule: Start with w_hmm=0 and ramp up after ~5 epochs
    so the emission head learns basic discrimination first.

    Args:
        config: Training configuration dict.
    """

    def __init__(self, config: dict):
        super().__init__()

        num_phases = config.get("num_phases", 11)
        
        # --- Ablation flags (for clean experiments) ---
        self.use_hmm = config.get("use_hmm", True)
        self.use_prog_phase = config.get("use_prog_phase", True)
        self.use_monotonic = config.get("use_monotonic", True)
        
        # Log active components
        active_components = []
        if self.use_hmm:
            active_components.append("HMM")
        if self.use_prog_phase:
            active_components.append("ProgPhase")
        if self.use_monotonic:
            active_components.append("Monotonic")
        print(f"  Structured model active components: {', '.join(active_components) if active_components else 'Baseline only'}")

        # --- Emission supervision (same as original PhaseLoss) ---
        self.emission_loss = PhaseLoss(
            num_phases=num_phases,
            label_smoothing_max=config.get("label_smoothing_max", 0.1),
            confidence_alpha=config.get("confidence_alpha", 0.3),
            focal_gamma=config.get("focal_gamma", 0.0),
        )

        # --- HMM-specific losses ---
        self.hmm_loss = HMMLoss()
        self.belief_ce_loss = BeliefCELoss(
            confidence_alpha=config.get("confidence_alpha", 0.3),
        )
        self.belief_ltr_loss = BeliefLTRLoss()
        
        # --- Progress-phase consistency (novel contribution) ---
        self.prog_phase_loss = ProgressPhaseConsistencyLoss()
        
        # --- Monotonic phase ordering (soft structural prior) ---
        self.monotonic_loss = MonotonicPhaseLoss()

        # --- Reused sub-losses ---
        self.progress_loss = ProgressLoss(
            lambda_hub=config.get("progress_lambda_hub", 1.0),
            lambda_mono=config.get("progress_lambda_mono", 0.2),
            lambda_rank=config.get("progress_lambda_rank", 0.2),
        )

        self.boundary_loss = BoundaryLoss(
            pos_weight=config.get("boundary_pos_weight", 6.0),
            focal_gamma=config.get("focal_gamma", 0.0),
            sparsity_weight=config.get("boundary_sparsity_weight", 0.0),
        )

        self.consistency_loss = ConsistencyLoss(
            lambda_link=config.get("consistency_lambda_link", 1.0),
            lambda_ltr=config.get("consistency_lambda_ltr", 0.5),
        )

        # --- Loss weights ---
        self.w_emission = config.get("w_emission", 1.0)
        self.w_belief = config.get("w_belief", 1.0)
        self.w_hmm = config.get("w_hmm", 0.5)
        self.w_progress = config.get("w_progress", 1.0)
        self.w_boundary = config.get("w_boundary", 1.0)
        self.w_consistency = config.get("w_consistency", 0.2)
        self.w_belief_ltr = config.get("w_belief_ltr", 0.1)
        self.w_prog_phase = config.get("w_prog_phase", 0.3)  # Progress-phase consistency
        self.w_monotonic = config.get("w_monotonic", 0.05)  # Monotonic phase ordering
        
        # --- Warmup schedules ---
        self.prog_phase_warmup_epochs = config.get("prog_phase_warmup_epochs", 0)
        self._w_prog_phase_target = self.w_prog_phase

        # --- HMM weight warm-up ---
        # If enabled, w_hmm starts at 0 and ramps to config value over N epochs.
        self.hmm_warmup_epochs = config.get("hmm_warmup_epochs", 0)
        self._w_hmm_target = self.w_hmm

    def set_epoch(self, epoch: int) -> None:
        """Update epoch-dependent weights (HMM warm-up, progress-phase warm-up).

        Args:
            epoch: Current training epoch (0-indexed).
        """
        # HMM warmup
        if self.hmm_warmup_epochs > 0 and epoch < self.hmm_warmup_epochs:
            ramp = epoch / self.hmm_warmup_epochs
            self.w_hmm = self._w_hmm_target * ramp
        else:
            self.w_hmm = self._w_hmm_target
        
        # Progress-phase warmup: delayed activation
        if self.prog_phase_warmup_epochs > 0 and epoch < self.prog_phase_warmup_epochs:
            self.w_prog_phase = 0.0
        else:
            self.w_prog_phase = self._w_prog_phase_target

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute total structured multi-task loss.

        Args:
            outputs: PhaseModelStructured outputs dict with keys:
                z_logits, emission_log_probs, beliefs, log_beliefs,
                log_normalizers, progress, boundary, hidden.
            batch: Batch dict with keys: z, p, b, c, attention_mask.

        Returns:
            (total_loss, detailed_loss_components)
        """
        mask = batch["attention_mask"]          # (B, W)
        z_target = batch["z"]                   # (B, W)
        p_target = batch["p"]                   # (B, W)
        b_target = batch["b"]                   # (B, W)
        confidence = batch["c"]                 # (B, W)

        # --- Emission supervision (raw head output) ---
        l_emission = self.emission_loss(
            outputs["z_logits"], z_target, confidence, mask
        )

        # --- HMM losses ---
        if self.use_hmm:
            l_hmm = self.hmm_loss(outputs["log_normalizers"], mask)
            l_belief_ce = self.belief_ce_loss(
                outputs["log_beliefs"], z_target, confidence, mask
            )
            l_belief_ltr = self.belief_ltr_loss(outputs["beliefs"], mask)
        else:
            # Ablation: disable HMM components
            l_hmm = torch.tensor(0.0, device=mask.device)
            l_belief_ce = torch.tensor(0.0, device=mask.device)
            l_belief_ltr = torch.tensor(0.0, device=mask.device)
        
        # DEBUG: Print HMM loss magnitude (first 3 batches only)
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        if self._debug_count < 3:
            print(f"\n[DEBUG Batch {self._debug_count}] HMM Loss Components:")
            print(f"  l_emission: {l_emission.item():.4f}")
            print(f"  l_belief_ce: {l_belief_ce.item():.4f}")
            print(f"  l_hmm (raw NLL): {l_hmm.item():.4f}")
            print(f"  mean log_normalizer: {outputs['log_normalizers'][mask.bool()].mean().item():.4f}")
            print(f"  w_hmm_effective: {self.w_hmm:.6f}")
            print(f"  weighted contribution: {(self.w_hmm * l_hmm).item():.4f}")
            self._debug_count += 1

        # --- Progress & boundary (same as original) ---
        l_prog, prog_details = self.progress_loss(
            outputs["progress"], p_target, confidence, mask
        )
        l_bnd = self.boundary_loss(
            outputs["boundary"], b_target, confidence, mask
        )

        # --- Consistency (on emission logits, not beliefs) ---
        l_cons, cons_details = self.consistency_loss(
            outputs["z_logits"], outputs["boundary"], mask
        )
        
        # --- Progress-phase consistency (novel contribution) ---
        if self.use_prog_phase:
            l_prog_phase = self.prog_phase_loss(
                outputs["z_logits"], outputs["progress"], mask
            )
        else:
            # Ablation: disable progress-phase consistency
            l_prog_phase = torch.tensor(0.0, device=mask.device)
        
        # --- Monotonic phase ordering (soft structural prior) ---
        if self.use_monotonic:
            l_monotonic = self.monotonic_loss(outputs["z_logits"], mask)
        else:
            # Ablation: disable monotonic constraint
            l_monotonic = torch.tensor(0.0, device=mask.device)

        # --- Total ---
        total = (
            self.w_emission * l_emission
            + self.w_belief * l_belief_ce
            + self.w_hmm * l_hmm
            + self.w_progress * l_prog
            + self.w_boundary * l_bnd
            + self.w_consistency * l_cons
            + self.w_belief_ltr * l_belief_ltr
            + self.w_prog_phase * l_prog_phase
            + self.w_monotonic * l_monotonic
        )

        details = {
            "total": total,
            "phase": l_emission,  # Alias for logger compatibility
            "emission_ce": l_emission,
            "belief_ce": l_belief_ce,
            "hmm_nll": l_hmm,
            "belief_ltr": l_belief_ltr,
            "prog_phase": l_prog_phase,  # Progress-phase consistency
            "monotonic": l_monotonic,    # Monotonic ordering
            "progress": l_prog,
            "boundary": l_bnd,
            "consistency": l_cons,
            "prog_huber": prog_details["huber"],
            "prog_mono": prog_details["mono"],
            "prog_rank": prog_details["rank"],
            "cons_link": cons_details["link"],
            "cons_ltr": cons_details["ltr"],
            # Effective HMM weight (for logging warm-up)
            "w_hmm_eff": torch.tensor(self.w_hmm, device=total.device),
            "w_prog_phase_eff": torch.tensor(self.w_prog_phase, device=total.device),
        }

        return total, details
