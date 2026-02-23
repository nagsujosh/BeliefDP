#!/usr/bin/env python3
"""Minimal end-to-end pipeline runner.

Pipeline:
1) Extract embeddings + subgoal metadata from demo.hdf5
2) Segment phases (auto-detect K from all demos, then fixed-K per demo)
3) Visualize phase boundaries
4) Train structured belief model
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
ONLINE_CONFIGS = PROJECT_ROOT / "online_phase" / "configs"


STAGE_ORDER = ["extract", "segment", "visualize", "train"]


def resolve_paths(demo: str | None, hdf5: str | None, output_dir: str | None) -> tuple[Path, Path]:
    """Resolve HDF5 and output directory from CLI arguments."""
    if demo:
        demo_root = PROJECT_ROOT / "demos" / demo
        hdf5_path = demo_root / "demo.hdf5"
        out_dir = demo_root / "output"
    else:
        hdf5_path = Path(hdf5) if hdf5 else (PROJECT_ROOT / "demo.hdf5")
        out_dir = Path(output_dir) if output_dir else (PROJECT_ROOT / "output")

    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    return hdf5_path, out_dir


def _run(cmd: list[str]) -> int:
    print(f"\n[RUN] {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def build_train_config(
    base_config_path: Path,
    out_dir: Path,
    hdf5_path: Path,
) -> Path:
    """Create a run-local training config wired to segmentation outputs."""
    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seg_json = out_dir / "phase_segmentation.json"
    subgoal_meta = out_dir / "subgoal_metadata.json"
    embeddings_dir = out_dir / "embeddings"

    cfg["segmentation_json"] = str(seg_json)
    cfg["embeddings_dir"] = str(embeddings_dir)
    cfg["hdf5_path"] = str(hdf5_path)
    cfg["label_cache_dir"] = str(out_dir / "labels_cache")
    cfg["output_dir"] = str(out_dir / "runs" / "structured")

    # Keep camera_key consistent with extraction metadata if available.
    if subgoal_meta.exists():
        with open(subgoal_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta:
            first_demo = sorted(meta.keys(), key=lambda x: int(x.split("_")[-1]))[0]
            cam = meta[first_demo].get("camera", "agentview_rgb")
            cfg["camera_keys"] = [cam]

    # Set num_phases from segmentation output when available.
    if seg_json.exists():
        with open(seg_json, "r", encoding="utf-8") as f:
            seg_data = json.load(f)
        cfg["num_phases"] = int(seg_data.get("config", {}).get("num_phases", cfg.get("num_phases", 6)))

    cfg_path = out_dir / "train_structured.generated.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path


def stage_extract(hdf5_path: Path, out_dir: Path, passthrough_args: list[str]) -> int:
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "extract_subgoals.py"),
        "--hdf5",
        str(hdf5_path),
        "--output-dir",
        str(out_dir),
        "--skip-existing",
        "--multi-view",
        "--decomp-mode",
        "ensemble",
        "--detection-source",
        "consensus_views",
        "--segmentation-source",
        "auto",
        "--fuse-mode",
        "weighted_mean",
    ] + passthrough_args
    return _run(cmd)


def stage_segment(out_dir: Path, passthrough_args: list[str]) -> int:
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "phase_segmentation.py"),
        "--metadata",
        str(out_dir / "subgoal_metadata.json"),
        "--embeddings-dir",
        str(out_dir / "embeddings"),
        "--output-dir",
        str(out_dir),
        "--scoring",
        "composite",
        "--align-mode",
        "progress",
        "--visualize",
    ] + passthrough_args
    return _run(cmd)


def stage_visualize(hdf5_path: Path, out_dir: Path, passthrough_args: list[str]) -> int:
    phase_meta = out_dir / "phase_segmentation.json"
    subgoal_meta = out_dir / "subgoal_metadata.json"
    metadata = phase_meta if phase_meta.exists() else subgoal_meta

    if not metadata.exists():
        print(f"Missing metadata for visualization: {metadata}")
        return 1

    cmd_frames = [
        sys.executable,
        str(SCRIPTS_DIR / "visualize_subgoals.py"),
        "--hdf5",
        str(hdf5_path),
        "--metadata",
        str(metadata),
        "--output-dir",
        str(out_dir / "subgoals"),
    ] + passthrough_args

    rc = _run(cmd_frames)
    if rc != 0:
        return rc

    cmd_videos = [
        sys.executable,
        str(SCRIPTS_DIR / "generate_phase_videos.py"),
        "--hdf5",
        str(hdf5_path),
        "--metadata",
        str(metadata),
        "--output-dir",
        str(out_dir / "phase_videos"),
        "--dual-view",
    ] + passthrough_args
    return _run(cmd_videos)


def stage_train(
    hdf5_path: Path,
    out_dir: Path,
    train_config: str | None,
    passthrough_args: list[str],
) -> int:
    base_cfg = Path(train_config) if train_config else (ONLINE_CONFIGS / "train_structured.yaml")
    if not base_cfg.exists():
        print(f"Training config not found: {base_cfg}")
        return 1

    generated_cfg = build_train_config(base_cfg, out_dir, hdf5_path)
    cmd = [
        sys.executable,
        "-m",
        "online_phase.train.train",
        "--config",
        str(generated_cfg),
    ] + passthrough_args
    return _run(cmd)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run BeliefDP pipeline: extract -> segment -> visualize -> train"
    )
    parser.add_argument(
        "stages",
        nargs="*",
        default=["all"],
        help="Any subset of: extract segment visualize train (or 'all')",
    )
    parser.add_argument("--demo", type=str, default=None, help="Demo under demos/<name>/")
    parser.add_argument("--hdf5", type=str, default=None, help="Path to HDF5 (used when --demo is not set)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (used when --demo is not set)")
    parser.add_argument(
        "--train-config",
        type=str,
        default=None,
        help="Optional training config template (default: online_phase/configs/train_structured.yaml)",
    )
    return parser.parse_known_args()


def main() -> int:
    args, passthrough = parse_args()

    try:
        hdf5_path, out_dir = resolve_paths(args.demo, args.hdf5, args.output_dir)
    except FileNotFoundError as e:
        print(str(e))
        return 1

    requested = args.stages or ["all"]
    if "all" in requested:
        stages = STAGE_ORDER
    else:
        unknown = [s for s in requested if s not in STAGE_ORDER]
        if unknown:
            print(f"Unknown stage(s): {', '.join(unknown)}")
            print(f"Valid stages: {', '.join(STAGE_ORDER)}")
            return 1
        stages = sorted(set(requested), key=lambda s: STAGE_ORDER.index(s))

    print(f"Pipeline stages: {' -> '.join(stages)}")
    print(f"HDF5: {hdf5_path}")
    print(f"Output: {out_dir}")

    for stage in stages:
        print(f"\n=== Stage: {stage} ===")
        if stage == "extract":
            rc = stage_extract(hdf5_path, out_dir, passthrough)
        elif stage == "segment":
            rc = stage_segment(out_dir, passthrough)
        elif stage == "visualize":
            rc = stage_visualize(hdf5_path, out_dir, passthrough)
        elif stage == "train":
            rc = stage_train(hdf5_path, out_dir, args.train_config, passthrough)
        else:
            print(f"Unhandled stage: {stage}")
            return 1

        if rc != 0:
            print(f"Stage failed: {stage} (exit={rc})")
            return rc

    print("\nPipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
