"""Visualize detected subgoals or phase boundaries: save labeled frames and montage grids.

Supports two metadata formats:
  - subgoal_metadata.json (from extract_subgoals.py): uses "subgoal_indices"
  - phase_segmentation.json (from phase_segmentation.py): uses "phase_boundaries" + "boundary_confidence"

Usage:
    # From subgoal metadata (original)
    python scripts/visualize_subgoals.py --hdf5 demo.hdf5 --metadata output/subgoal_metadata.json --output-dir output/subgoals/

    # From phase segmentation (current approach)
    python scripts/visualize_subgoals.py --hdf5 demo.hdf5 --metadata output/phase_segmentation.json --output-dir output/subgoals/

    # Single demo
    python scripts/visualize_subgoals.py --hdf5 demo.hdf5 --metadata output/phase_segmentation.json --output-dir output/subgoals/ --demo demo_0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm


CAMERAS = ["agentview_rgb", "eye_in_hand_rgb", "wrist_rgb"]
CAMERA_SHORT = {
    "agentview_rgb": "agentview", 
    "eye_in_hand_rgb": "eye_in_hand",
    "wrist_rgb": "wrist"
}


# ======================================================================= #
#  Metadata parsing — support both formats
# ======================================================================= #

def parse_metadata(raw: dict) -> tuple[dict, str]:
    """Parse metadata file and return normalized per-demo dict + format name.

    Returns:
        (demos_dict, format_name)
        demos_dict maps demo_name -> {
            "frame_indices": list[int],
            "num_frames": int,
            "confidences": list[float] | None,   # only for phase format
            "anchor_indices": list[int] | None,   # only for phase format
        }
    """
    # Detect format: phase_segmentation.json has a top-level "config" key
    if "config" in raw:
        # Phase segmentation format — demos may be at top level or under "demos" key
        demo_entries = raw.get("demos", raw)
        demos = {}
        for key, val in demo_entries.items():
            if key in ("config", "phase_stats"):
                continue
            if not isinstance(val, dict) or "phase_boundaries" not in val:
                continue
            demos[key] = {
                "frame_indices": val["phase_boundaries"],
                "num_frames": val["num_frames"],
                "confidences": val.get("boundary_confidence"),
                "anchor_indices": val.get("anchor_indices"),
            }
        return demos, "phase"
    else:
        # Subgoal metadata format
        demos = {}
        for key, val in raw.items():
            demos[key] = {
                "frame_indices": val["subgoal_indices"],
                "num_frames": val["num_frames"],
                "confidences": None,
                "anchor_indices": None,
            }
        return demos, "subgoal"


# ======================================================================= #
#  Frame extraction
# ======================================================================= #

def extract_subgoal_frames(
    hdf5_path: str | Path,
    demo_name: str,
    frame_indices: list[int],
    camera_keys: list[str],
) -> dict[str, np.ndarray]:
    """Load only the frames at given indices from the HDF5 file.

    Returns:
        Dict mapping camera_key -> uint8 array (num_frames, H, W, 3).
    """
    frames_dict = {}
    indices = sorted(frame_indices)

    with h5py.File(hdf5_path, "r") as f:
        obs = f["data"][demo_name]["obs"]
        for cam in camera_keys:
            if cam not in obs:
                continue
            ds = obs[cam]
            frames = ds[indices]
            if frames.dtype != np.uint8:
                if np.issubdtype(frames.dtype, np.floating):
                    frames = (frames * 255).clip(0, 255).astype(np.uint8)
                else:
                    frames = frames.astype(np.uint8)
            frames_dict[cam] = frames

    return frames_dict


# ======================================================================= #
#  Labeling
# ======================================================================= #

def draw_label(image: np.ndarray, text: str, color_bg=(0, 0, 0), color_fg=(255, 255, 255)) -> np.ndarray:
    """Draw a text label on the top-left of an image (returns a copy)."""
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (0, 0), (tw + 8, th + baseline + 8), color_bg, -1)
    cv2.putText(img, text, (4, th + 4), font, scale, color_fg, thickness, cv2.LINE_AA)
    return img


def save_labeled_frames(
    frames_dict: dict[str, np.ndarray],
    demo_name: str,
    frame_indices: list[int],
    output_dir: Path,
    confidences: list[float] | None = None,
    anchor_indices: list[int] | None = None,
    fmt: str = "phase",
) -> None:
    """Save individual labeled PNG files for each boundary/subgoal and camera view."""
    output_dir.mkdir(parents=True, exist_ok=True)
    anchor_set = set(anchor_indices) if anchor_indices else set()

    for cam, frames in frames_dict.items():
        cam_short = CAMERA_SHORT.get(cam, cam)
        for i, (frame, frame_idx) in enumerate(zip(frames, frame_indices)):
            if fmt == "phase":
                label = f"Phase {i} | Frame {frame_idx}"
                if confidences and i < len(confidences):
                    label += f" | Conf {confidences[i]:.2f}"
                prefix = "phase"
            else:
                label = f"Subgoal {i} | Frame {frame_idx}"
                prefix = "subgoal"

            # Highlight anchors with a colored background
            if frame_idx in anchor_set:
                labeled = draw_label(frame, label + " [A]", color_bg=(0, 100, 0), color_fg=(255, 255, 255))
            else:
                labeled = draw_label(frame, label)

            fname = f"{prefix}_{i}_frame_{frame_idx}_{cam_short}.png"
            cv2.imwrite(str(output_dir / fname), cv2.cvtColor(labeled, cv2.COLOR_RGB2BGR))


# ======================================================================= #
#  Montage
# ======================================================================= #

def create_dual_view_montage(
    agentview_frames: np.ndarray,
    eyehand_frames: np.ndarray,
    frame_indices: list[int],
    cols: int = 5,
    confidences: list[float] | None = None,
    anchor_indices: list[int] | None = None,
    fmt: str = "phase",
) -> np.ndarray:
    """Create a montage grid with agentview + eye_in_hand side-by-side.

    Each cell shows both views horizontally concatenated with a label bar below.
    Anchor boundaries get a green label bar; interior boundaries get dark gray.

    Returns:
        RGB uint8 array of the montage image.
    """
    n = len(frame_indices)
    h, w = agentview_frames.shape[1], agentview_frames.shape[2]
    cell_w = w * 2 + 4  # two views + gap
    label_h = 28
    cell_h = h + label_h
    anchor_set = set(anchor_indices) if anchor_indices else set()

    rows = math.ceil(n / cols)
    canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for idx in range(n):
        r, c = divmod(idx, cols)
        y0 = r * cell_h
        x0 = c * cell_w

        # Paste agentview
        canvas[y0 : y0 + h, x0 : x0 + w] = agentview_frames[idx]
        # Gap
        canvas[y0 : y0 + h, x0 + w : x0 + w + 4] = 40
        # Paste eye_in_hand
        canvas[y0 : y0 + h, x0 + w + 4 : x0 + cell_w] = eyehand_frames[idx]

        # Label bar
        is_anchor = frame_indices[idx] in anchor_set
        if fmt == "phase":
            label = f"Phase {idx} | F{frame_indices[idx]}"
            if confidences and idx < len(confidences):
                label += f" | C={confidences[idx]:.2f}"
            if is_anchor:
                label += " [A]"
        else:
            label = f"Subgoal {idx} | Frame {frame_indices[idx]}"

        label_region = canvas[y0 + h : y0 + cell_h, x0 : x0 + cell_w]
        if is_anchor:
            label_region[:] = (0, 60, 0)  # dark green for anchors
        else:
            label_region[:] = 30  # dark gray
        cv2.putText(
            label_region, label, (4, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return canvas


# ======================================================================= #
#  Summary CSV
# ======================================================================= #

def generate_summary_csv(demos: dict, output_path: Path, fmt: str) -> None:
    """Write a flat CSV summarizing detections across all demos."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        if fmt == "phase":
            writer.writerow(["demo_id", "phase_index", "frame_number", "total_frames",
                             "confidence", "is_anchor"])
        else:
            writer.writerow(["demo_id", "subgoal_index", "frame_number", "total_frames",
                             "num_subgoals"])

        for demo_name, info in sorted(demos.items(), key=lambda x: int(x[0].split("_")[-1])):
            anchor_set = set(info["anchor_indices"]) if info["anchor_indices"] else set()
            for i, frame_num in enumerate(info["frame_indices"]):
                if fmt == "phase":
                    conf = info["confidences"][i] if info["confidences"] and i < len(info["confidences"]) else ""
                    is_anch = frame_num in anchor_set
                    writer.writerow([demo_name, i, frame_num, info["num_frames"],
                                     f"{conf:.4f}" if conf != "" else "", is_anch])
                else:
                    writer.writerow([demo_name, i, frame_num, info["num_frames"],
                                     len(info["frame_indices"])])


# ======================================================================= #
#  Main
# ======================================================================= #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize detected subgoals/phase boundaries with labeled frames and montages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hdf5", type=str, required=True, help="Path to demo HDF5 file")
    parser.add_argument("--metadata", type=str, required=True,
                        help="Path to subgoal_metadata.json or phase_segmentation.json")
    parser.add_argument("--output-dir", type=str, default="output/subgoals", help="Output directory for images")
    parser.add_argument("--demo", type=str, default=None, help="Visualize a single demo only")
    parser.add_argument("--montage-cols", type=int, default=5, help="Columns in montage grid")
    parser.add_argument("--no-labels", action="store_true", help="Skip text overlay on individual frames")
    args = parser.parse_args()

    with open(args.metadata) as f:
        raw = json.load(f)

    demos, fmt = parse_metadata(raw)
    fmt_label = "phase boundaries" if fmt == "phase" else "subgoals"

    hdf5_path = Path(args.hdf5)
    output_base = Path(args.output_dir)

    demos_to_process = list(demos.keys())
    if args.demo:
        if args.demo not in demos:
            print(f"Error: {args.demo} not found in metadata")
            return 1
        demos_to_process = [args.demo]

    print(f"Visualizing {fmt_label} for {len(demos_to_process)} demo(s) [format: {fmt}]...")

    for demo_name in tqdm(demos_to_process, desc="Demos"):
        info = demos[demo_name]
        frame_indices = info["frame_indices"]

        if not frame_indices:
            tqdm.write(f"  {demo_name}: no {fmt_label} detected, skipping")
            continue

        demo_dir = output_base / demo_name
        demo_dir.mkdir(parents=True, exist_ok=True)

        # Extract frames at boundary/subgoal indices from both cameras
        frames_dict = extract_subgoal_frames(hdf5_path, demo_name, frame_indices, CAMERAS)

        # Save individual labeled frames
        if not args.no_labels:
            save_labeled_frames(
                frames_dict, demo_name, frame_indices, demo_dir,
                confidences=info["confidences"],
                anchor_indices=info["anchor_indices"],
                fmt=fmt,
            )
        else:
            prefix = "phase" if fmt == "phase" else "subgoal"
            for cam, frames in frames_dict.items():
                cam_short = CAMERA_SHORT.get(cam, cam)
                for i, (frame, fidx) in enumerate(zip(frames, frame_indices)):
                    fname = f"{prefix}_{i}_frame_{fidx}_{cam_short}.png"
                    cv2.imwrite(str(demo_dir / fname), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Create montage if both views available
        if ("agentview_rgb" in frames_dict and "eye_in_hand_rgb" in frames_dict) or \
           ("agentview_rgb" in frames_dict and "wrist_rgb" in frames_dict):
            # Use eye_in_hand_rgb if available, otherwise wrist_rgb
            second_cam = "eye_in_hand_rgb" if "eye_in_hand_rgb" in frames_dict else "wrist_rgb"
            montage = create_dual_view_montage(
                frames_dict["agentview_rgb"],
                frames_dict[second_cam],
                frame_indices,
                cols=args.montage_cols,
                confidences=info["confidences"],
                anchor_indices=info["anchor_indices"],
                fmt=fmt,
            )
            montage_path = demo_dir / "montage.png"
            cv2.imwrite(str(montage_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
            tqdm.write(f"  {demo_name}: saved montage ({len(frame_indices)} {fmt_label})")
        elif frames_dict:
            cam = list(frames_dict.keys())[0]
            tqdm.write(f"  {demo_name}: only {cam} available, montage skipped")

    # Generate summary CSV
    csv_name = "phase_summary.csv" if fmt == "phase" else "subgoal_summary.csv"
    csv_path = output_base / csv_name
    generate_summary_csv(demos, csv_path, fmt)
    print(f"\nSaved summary CSV to {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
