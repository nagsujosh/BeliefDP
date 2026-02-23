#!/usr/bin/env python3
"""Generate MP4 demo videos with phase number overlay for transition inspection.

Reads frames from the HDF5 file and phase boundaries from
phase_segmentation.json (or subgoal_metadata.json), assigns every frame
a phase number, draws it prominently on each frame, and writes an MP4
per demo (one video per camera, or side-by-side dual view).

Usage:
    python scripts/generate_phase_videos.py \
        --hdf5 demo.hdf5 \
        --metadata output/phase_segmentation.json \
        --output-dir output/phase_videos

    # Single demo
    python scripts/generate_phase_videos.py \
        --hdf5 demo.hdf5 \
        --metadata output/phase_segmentation.json \
        --output-dir output/phase_videos \
        --demo demo_0

    # Side-by-side dual camera view
    python scripts/generate_phase_videos.py \
        --hdf5 demo.hdf5 \
        --metadata output/phase_segmentation.json \
        --output-dir output/phase_videos \
        --dual-view
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from online_phase.utils.camera_utils import (
    detect_available_cameras,
    resolve_camera_key,
    get_short_name,
    CANONICAL_CAMERAS,
)

# Legacy camera list (for backward compatibility)
CAMERAS = ["agentview_rgb", "eye_in_hand_rgb", "wrist_rgb"]
CAMERA_SHORT = {
    "agentview_rgb": "agentview", 
    "eye_in_hand_rgb": "eye_in_hand",
    "wrist_rgb": "wrist"
}

# Phase colour palette (10 phases + background)
# Distinct colours so transitions are visually obvious
PHASE_COLORS = [
    (46, 204, 113),    # emerald green
    (52, 152, 219),    # peter-river blue
    (231, 76, 60),     # alizarin red
    (241, 196, 15),    # sunflower yellow
    (155, 89, 182),    # amethyst purple
    (230, 126, 34),    # carrot orange
    (26, 188, 156),    # turquoise
    (236, 64, 122),    # pink
    (0, 188, 212),     # cyan
    (255, 193, 7),     # amber
    (121, 85, 72),     # brown
    (96, 125, 139),    # blue grey
]


# ======================================================================= #
#  Metadata parsing (shared with visualize_subgoals.py)
# ======================================================================= #

def parse_metadata(raw: dict) -> tuple[dict, str]:
    """Return normalized per-demo dict + format name."""
    if "config" in raw:
        demo_entries = raw.get("demos", raw)
        demos = {}
        for key, val in demo_entries.items():
            if key in ("config", "phase_stats"):
                continue
            if not isinstance(val, dict) or "phase_boundaries" not in val:
                continue
            demos[key] = {
                "phase_boundaries": val["phase_boundaries"],
                "num_frames": val["num_frames"],
                "confidences": val.get("boundary_confidence"),
                "anchor_indices": val.get("anchor_indices"),
            }
        return demos, "phase"
    else:
        demos = {}
        for key, val in raw.items():
            demos[key] = {
                "phase_boundaries": val["subgoal_indices"],
                "num_frames": val["num_frames"],
                "confidences": None,
                "anchor_indices": None,
            }
        return demos, "subgoal"


def frame_to_phase(frame_idx: int, boundaries: list[int]) -> int:
    """Return the phase number for a given frame index.

    Phase 0 covers frames [0, boundaries[0]).
    Phase i covers frames [boundaries[i-1], boundaries[i]).
    Last phase covers frames [boundaries[-1], end).
    """
    for i, b in enumerate(boundaries):
        if frame_idx < b:
            return i
    return len(boundaries)


# ======================================================================= #
#  Drawing helpers
# ======================================================================= #

def draw_phase_overlay(
    frame: np.ndarray,
    phase: int,
    frame_idx: int,
    num_frames: int,
    is_boundary: bool = False,
) -> np.ndarray:
    """Draw phase number, frame counter, and colour-coded banner on a frame.

    Returns a copy with overlay drawn.
    """
    img = frame.copy()
    H, W = img.shape[:2]

    colour = PHASE_COLORS[phase % len(PHASE_COLORS)]

    # --- Top banner (semi-transparent) ---
    banner_h = 36
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (W, banner_h), colour, -1)
    alpha = 0.65
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # --- Phase text (large) ---
    phase_text = f"Phase {phase}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, phase_text, (8, 26), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # --- Frame counter (right-aligned) ---
    counter_text = f"F {frame_idx}/{num_frames - 1}"
    (tw, _), _ = cv2.getTextSize(counter_text, font, 0.45, 1)
    cv2.putText(img, counter_text, (W - tw - 8, 24), font, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)

    # --- Boundary flash: bright border on transition frames ---
    if is_boundary:
        cv2.rectangle(img, (0, 0), (W - 1, H - 1), (0, 255, 255), 3)

    # --- Small phase colour dot (bottom-left) for quick reference ---
    cv2.circle(img, (16, H - 16), 10, colour, -1)
    cv2.circle(img, (16, H - 16), 10, (255, 255, 255), 1)

    return img


# ======================================================================= #
#  Video writer
# ======================================================================= #

def write_phase_video(
    hdf5_path: Path,
    demo_name: str,
    boundaries: list[int],
    num_frames: int,
    camera: str,
    output_path: Path,
    fps: int = 20,
) -> None:
    """Write a single-camera phase-annotated video."""
    boundary_set = set(boundaries)

    with h5py.File(hdf5_path, "r") as f:
        obs = f["data"][demo_name]["obs"]
        if camera not in obs:
            tqdm.write(f"  {demo_name}: camera '{camera}' not found, skipping")
            return
        ds = obs[camera]
        T, H, W, C = ds.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

        for t in range(T):
            frame = ds[t]
            if frame.dtype != np.uint8:
                if np.issubdtype(frame.dtype, np.floating):
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)

            phase = frame_to_phase(t, boundaries)
            is_boundary = t in boundary_set
            annotated = draw_phase_overlay(frame, phase, t, num_frames, is_boundary)

            # OpenCV expects BGR
            writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        writer.release()


def write_dual_view_video(
    hdf5_path: Path,
    demo_name: str,
    boundaries: list[int],
    num_frames: int,
    output_path: Path,
    fps: int = 20,
) -> None:
    """Write a side-by-side dual-camera phase-annotated video."""
    boundary_set = set(boundaries)

    with h5py.File(hdf5_path, "r") as f:
        obs = f["data"][demo_name]["obs"]
        cams_avail = [c for c in CAMERAS if c in obs]
        if len(cams_avail) < 2:
            tqdm.write(f"  {demo_name}: need 2 cameras for dual view, only {len(cams_avail)} found")
            return

        ds0 = obs[cams_avail[0]]
        ds1 = obs[cams_avail[1]]
        T = ds0.shape[0]
        H, W = ds0.shape[1], ds0.shape[2]
        gap = 4
        out_W = W * 2 + gap

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_W, H))

        for t in range(T):
            phase = frame_to_phase(t, boundaries)
            is_boundary = t in boundary_set

            frames_pair = []
            for ds in [ds0, ds1]:
                fr = ds[t]
                if fr.dtype != np.uint8:
                    if np.issubdtype(fr.dtype, np.floating):
                        fr = (fr * 255).clip(0, 255).astype(np.uint8)
                    else:
                        fr = fr.astype(np.uint8)
                frames_pair.append(draw_phase_overlay(fr, phase, t, num_frames, is_boundary))

            canvas = np.zeros((H, out_W, 3), dtype=np.uint8)
            canvas[:, :W] = frames_pair[0]
            canvas[:, W:W + gap] = 40  # grey gap
            canvas[:, W + gap:] = frames_pair[1]

            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

        writer.release()


# ======================================================================= #
#  Main
# ======================================================================= #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate MP4 demo videos with phase number overlay.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hdf5", type=str, required=True, help="Path to demo HDF5 file")
    parser.add_argument("--metadata", type=str, required=True,
                        help="Path to phase_segmentation.json or subgoal_metadata.json")
    parser.add_argument("--output-dir", type=str, default="output/phase_videos",
                        help="Output directory for videos")
    parser.add_argument("--demo", type=str, default=None,
                        help="Generate video for a single demo only")
    parser.add_argument("--fps", type=int, default=20, help="Video frame rate")
    parser.add_argument("--dual-view", action="store_true",
                        help="Side-by-side agentview + eye_in_hand in one video")
    parser.add_argument("--camera", type=str, default=None,
                        choices=CAMERAS,
                        help="Single camera to render (default: all cameras)")
    args = parser.parse_args()

    with open(args.metadata) as f:
        raw = json.load(f)

    demos, fmt = parse_metadata(raw)
    hdf5_path = Path(args.hdf5)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    demos_to_process = sorted(demos.keys(), key=lambda x: int(x.split("_")[-1]))
    if args.demo:
        if args.demo not in demos:
            print(f"Error: {args.demo} not found in metadata")
            return 1
        demos_to_process = [args.demo]

    print(f"Generating phase videos for {len(demos_to_process)} demo(s) "
          f"[format: {fmt}, fps: {args.fps}]...")

    cameras = [args.camera] if args.camera else CAMERAS

    for demo_name in tqdm(demos_to_process, desc="Demos"):
        info = demos[demo_name]
        boundaries = info["phase_boundaries"]
        num_frames = info["num_frames"]

        if not boundaries:
            tqdm.write(f"  {demo_name}: no boundaries detected, skipping")
            continue

        if args.dual_view:
            out_path = output_dir / f"{demo_name}_dual_phase.mp4"
            write_dual_view_video(hdf5_path, demo_name, boundaries, num_frames,
                                  out_path, fps=args.fps)
            tqdm.write(f"  {demo_name}: dual-view video saved → {out_path.name}")
        else:
            for cam in cameras:
                cam_short = CAMERA_SHORT.get(cam, cam)
                out_path = output_dir / f"{demo_name}_{cam_short}_phase.mp4"
                write_phase_video(hdf5_path, demo_name, boundaries, num_frames,
                                  cam, out_path, fps=args.fps)
                tqdm.write(f"  {demo_name}: {cam_short} video saved → {out_path.name}")

    print(f"\nAll videos saved to {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
