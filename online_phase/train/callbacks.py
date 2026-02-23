"""Training callbacks: checkpointing, JSON logging."""
from __future__ import annotations

import time
from pathlib import Path

import torch

from online_phase.utils.io import save_json


class CheckpointManager:
    """Save and load model checkpoints.

    Saves the best model (lowest val loss) and the latest model.
    """

    def __init__(self, output_dir: str | Path, config: dict):
        self.output_dir = Path(output_dir) / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.best_val_loss = float("inf")

    def save(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
        norm_stats: dict | None = None,
    ) -> str | None:
        """Save checkpoint. Returns path if this is the new best, else None."""
        state = {
            "model_state_dict": model.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": self.config,
        }
        if norm_stats is not None:
            state["norm_stats"] = norm_stats

        # Always save latest
        latest_path = self.output_dir / "latest.pt"
        torch.save(state, latest_path)

        # Save best if improved
        saved_best = None
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.output_dir / "best.pt"
            torch.save(state, best_path)
            saved_best = str(best_path)

        return saved_best

    @staticmethod
    def load(path: str | Path, device: str = "cpu") -> dict:
        return torch.load(path, map_location=device, weights_only=False)


class MetricsLogger:
    """Log training metrics to JSON file and stdout with rich formatting."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "metrics.json"
        self.history: list[dict] = []
        self.epoch_start_time = 0.0
        self._header_printed = False

    def start_epoch(self) -> None:
        self.epoch_start_time = time.time()

    def _print_header(self) -> None:
        """Print column header for training log."""
        sep = "-" * 120
        print()
        print(sep)
        print(
            f"{'Ep':>4} | "
            f"{'Train':>9} {'Val':>9} | "
            f"{'V:Phase':>8} {'V:Prog':>8} {'V:Bnd':>8} {'V:Cons':>8} | "
            f"{'Acc':>6} {'P-MAE':>6} {'B-F1':>5} | "
            f"{'LR':>9} {'Time':>5} "
        )
        print(sep)
        self._header_printed = True

    def log_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        extra: dict | None = None,
    ) -> None:
        duration = time.time() - self.epoch_start_time
        entry = {
            "epoch": epoch,
            "duration_s": round(duration, 1),
            "train": {k: round(v, 6) for k, v in train_metrics.items()},
            "val": {k: round(v, 6) for k, v in val_metrics.items()},
        }
        if extra:
            entry["extra"] = extra

        self.history.append(entry)

        # Write full history
        save_json(self.history, self.log_path)

        # Print header on first epoch
        if not self._header_printed:
            self._print_header()

        # Extract values
        t_loss = train_metrics.get("total", 0)
        v_loss = val_metrics.get("total", 0)

        # Per-component val losses
        v_phase = val_metrics.get("phase", 0)
        v_prog = val_metrics.get("progress", 0)
        v_bnd = val_metrics.get("boundary", 0)
        v_cons = val_metrics.get("consistency", 0)
        
        # DEBUG: For structured models, also show HMM components
        v_belief_ce = val_metrics.get("belief_ce", None)
        v_hmm_nll = val_metrics.get("hmm_nll", None)

        # Val task metrics
        acc = val_metrics.get("phase_acc", 0)
        p_mae = val_metrics.get("progress_mae", 0)
        b_f1 = val_metrics.get("boundary_f1", 0)

        # Extra info
        lr = extra.get("lr", 0) if extra else 0
        is_best = bool(extra.get("saved_best")) if extra else False
        marker = " *" if is_best else ""

        # Overfit indicator
        gap = v_loss / max(t_loss, 1e-8)
        gap_warn = f" ({gap:.1f}x)" if gap > 1.8 else ""
        
        # Format HMM components if available (structured model)
        hmm_info = ""
        if v_belief_ce is not None and v_hmm_nll is not None:
            hmm_info = f" [Bel:{v_belief_ce:.2f} HMM:{v_hmm_nll:.2f}]"

        print(
            f"{epoch:4d} | "
            f"{t_loss:9.4f} {v_loss:9.4f} | "
            f"{v_phase:8.4f} {v_prog:8.4f} {v_bnd:8.4f} {v_cons:8.4f} | "
            f"{acc:6.3f} {p_mae:6.4f} {b_f1:5.3f} | "
            f"{lr:9.2e} {duration:5.1f}s{marker}{gap_warn}{hmm_info}"
        )
