"""Export trained model to TorchScript or ONNX for deployment (optional).

Usage:
    python -m online_phase.infer.export_model --ckpt path.pt --format torchscript
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from online_phase.models.phase_model_structured import PhaseModelStructured
from online_phase.train.callbacks import CheckpointManager


def export_torchscript(model: torch.nn.Module, output_path: Path, window_size: int, input_dim: int) -> None:
    model.eval()
    example_input = torch.randn(1, window_size, input_dim)
    example_mask = torch.ones(1, window_size)
    traced = torch.jit.trace(model, (example_input, example_mask))
    traced.save(str(output_path))
    print(f"TorchScript model saved to {output_path}")


def export_onnx(model: torch.nn.Module, output_path: Path, window_size: int, input_dim: int) -> None:
    model.eval()
    example_input = torch.randn(1, window_size, input_dim)
    example_mask = torch.ones(1, window_size)
    torch.onnx.export(
        model,
        (example_input, example_mask),
        str(output_path),
        input_names=["embeddings", "attention_mask"],
        output_names=["z_logits", "progress", "boundary"],
        dynamic_axes={
            "embeddings": {0: "batch"},
            "attention_mask": {0: "batch"},
        },
        opset_version=14,
    )
    print(f"ONNX model saved to {output_path}")


def main(ckpt_path: str, fmt: str = "torchscript") -> None:
    ckpt = CheckpointManager.load(ckpt_path)
    config = ckpt["config"]

    requested_model_type = config.get("model_type")
    if requested_model_type not in (None, "structured"):
        raise ValueError(
            f"Unsupported model_type='{requested_model_type}'. "
            "This repository is structured-only; use PhaseModelStructured."
        )

    model = PhaseModelStructured(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    output_dir = Path(ckpt_path).parent
    window_size = config.get("window_size", 48)
    input_dim = config.get("input_dim", 2048)

    if fmt == "torchscript":
        export_torchscript(model, output_dir / "model.pt", window_size, input_dim)
    elif fmt == "onnx":
        export_onnx(model, output_dir / "model.onnx", window_size, input_dim)
    else:
        print(f"Unknown format: {fmt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--format", type=str, default="torchscript", choices=["torchscript", "onnx"])
    args = parser.parse_args()
    main(args.ckpt, args.format)
