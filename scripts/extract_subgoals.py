"""Extract visual embeddings and detect UVD subgoals from HDF5 demos.

This module keeps UVD decomposition behavior aligned with the vendored code
while adding higher-accuracy options:
1) multi-parameter decomposition ensemble
2) cross-view consensus boundaries
3) configurable embedding fusion and automatic source selection
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

# Disable wandb to avoid login prompts from UVD internals.
os.environ.setdefault("WANDB_MODE", "disabled")

# Make vendored UVD importable without a separate pip install.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_UVD_ROOT = _PROJECT_ROOT / "beliefdp" / "offline" / "uvd"
if _UVD_ROOT.exists():
    sys.path.insert(0, str(_UVD_ROOT))


def _install_allenact_shim_if_needed() -> None:
    """Install minimal allenact shim for UVD imports."""
    try:
        import allenact  # noqa: F401
        return
    except Exception:
        pass

    allenact_mod = types.ModuleType("allenact")
    utils_mod = types.ModuleType("allenact.utils")
    system_mod = types.ModuleType("allenact.utils.system")
    base_abs_mod = types.ModuleType("allenact.base_abstractions")
    dist_mod = types.ModuleType("allenact.base_abstractions.distributions")

    def get_logger():
        return logging.getLogger("uvd")

    class CategoricalDistr:
        pass

    system_mod.get_logger = get_logger
    dist_mod.CategoricalDistr = CategoricalDistr

    sys.modules["allenact"] = allenact_mod
    sys.modules["allenact.utils"] = utils_mod
    sys.modules["allenact.utils.system"] = system_mod
    sys.modules["allenact.base_abstractions"] = base_abs_mod
    sys.modules["allenact.base_abstractions.distributions"] = dist_mod

    allenact_mod.utils = utils_mod
    allenact_mod.base_abstractions = base_abs_mod
    utils_mod.system = system_mod
    base_abs_mod.distributions = dist_mod


def _install_gym_shim_if_needed() -> None:
    """Install minimal gym shim for UVD imports."""
    try:
        import gym  # noqa: F401
        return
    except Exception:
        pass

    gym_mod = types.ModuleType("gym")
    spaces_mod = types.ModuleType("gym.spaces")

    class Space:
        pass

    class Env:
        pass

    class Wrapper(Env):
        def __init__(self, env=None, *args, **kwargs):
            self.env = env
            self.action_space = getattr(env, "action_space", None)

        def step(self, action):
            if self.env is None:
                raise NotImplementedError("Shim gym.Wrapper has no wrapped env")
            return self.env.step(action)

    class Box(Space):
        def __init__(self, low=None, high=None, shape=None, dtype=None):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class Dict(Space):
        def __init__(self, spaces=None):
            self.spaces = spaces or {}

    class Discrete(Space):
        def __init__(self, n=0):
            self.n = n

    class MultiDiscrete(Space):
        def __init__(self, nvec):
            self.nvec = nvec

    def register(*args, **kwargs):
        return None

    spaces_mod.Space = Space
    spaces_mod.Box = Box
    spaces_mod.Dict = Dict
    spaces_mod.Discrete = Discrete
    spaces_mod.MultiDiscrete = MultiDiscrete

    gym_mod.Env = Env
    gym_mod.Wrapper = Wrapper
    gym_mod.register = register
    gym_mod.spaces = spaces_mod

    sys.modules["gym"] = gym_mod
    sys.modules["gym.spaces"] = spaces_mod


_install_allenact_shim_if_needed()
_install_gym_shim_if_needed()

from uvd.decomp.decomp import embedding_decomp
from uvd.models.preprocessors import get_preprocessor


CAMERA_CANDIDATES = ["agentview_rgb", "eye_in_hand_rgb", "wrist_rgb"]
PREPROCESSOR_CHOICES = ["vip", "r3m", "liv", "clip", "vc1", "dinov2"]


def get_demo_names(hdf5_path: str | Path) -> list[str]:
    """Return sorted list of demo names from the HDF5 file."""
    with h5py.File(hdf5_path, "r") as f:
        names = [k for k in f["data"].keys() if k.startswith("demo")]
    return sorted(names, key=lambda s: int(s.split("_")[-1]))


def detect_available_cameras(hdf5_path: str | Path) -> list[str]:
    """Detect camera keys from the first demo in the HDF5 file."""
    with h5py.File(hdf5_path, "r") as f:
        first_demo = sorted(f["data"].keys())[0]
        obs_keys = list(f["data"][first_demo]["obs"].keys())
    return [cam for cam in CAMERA_CANDIDATES if cam in obs_keys]


def load_demo_frames(
    hdf5_path: str | Path,
    demo_name: str,
    camera_key: str = "agentview_rgb",
) -> np.ndarray:
    """Load RGB frames for a single demo from HDF5."""
    with h5py.File(hdf5_path, "r") as f:
        frames = f["data"][demo_name]["obs"][camera_key][:]
    if frames.dtype != np.uint8:
        if np.issubdtype(frames.dtype, np.floating):
            frames = (frames * 255).clip(0, 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)
    return frames


def extract_embeddings(preprocessor, frames: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """Compute embeddings in batches."""
    chunks = []
    for i in range(0, len(frames), batch_size):
        emb = preprocessor.process(frames[i : i + batch_size], return_numpy=True)
        chunks.append(emb.astype(np.float32))
    return np.concatenate(chunks, axis=0)


def is_cuda_kernel_compat_error(exc: Exception) -> bool:
    """Detect common CUDA arch/runtime incompatibility errors."""
    msg = str(exc).lower()
    patterns = [
        "no kernel image is available for execution on the device",
        "not compatible with the current pytorch installation",
        "cuda error",
        "sm_120",
    ]
    return any(p in msg for p in patterns)


def l2_normalize_embeddings(embeddings: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-frame L2 normalization."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, eps)


def parse_int_list(raw: str) -> list[int]:
    """Parse comma-separated int list."""
    vals = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            vals.append(int(item))
    return vals


def parse_float_list(raw: str) -> list[float]:
    """Parse comma-separated float list."""
    vals = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    return vals


def motion_energy(embeddings: np.ndarray) -> float:
    """Average frame-to-frame movement in embedding space."""
    if len(embeddings) < 2:
        return 0.0
    diffs = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
    return float(np.mean(diffs))


def transition_separability_score(embeddings: np.ndarray) -> float:
    """Heuristic quality score for segmentation suitability."""
    if len(embeddings) < 8:
        return float("-inf")
    diffs = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
    if np.allclose(diffs, 0):
        return float("-inf")
    q25, q50, q75, q90 = np.percentile(diffs, [25, 50, 75, 90])
    iqr = max(q75 - q25, 1e-8)
    tail = max(q90 - q50, 0.0)
    return float(tail / iqr)


def fuse_embeddings(
    per_camera_embeddings: dict[str, np.ndarray],
    mode: str,
) -> tuple[np.ndarray | None, dict]:
    """Fuse multi-view embeddings."""
    if len(per_camera_embeddings) <= 1:
        return None, {"mode": mode, "weights": None}

    cam_list = sorted(per_camera_embeddings.keys())
    arrays = [per_camera_embeddings[c] for c in cam_list]
    min_t = min(a.shape[0] for a in arrays)
    arrays = [a[:min_t] for a in arrays]

    if mode == "concat":
        fused = np.concatenate(arrays, axis=1).astype(np.float32)
        return fused, {"mode": mode, "weights": None}

    if mode == "mean":
        fused = np.mean(np.stack(arrays, axis=0), axis=0).astype(np.float32)
        return fused, {"mode": mode, "weights": [1.0 / len(arrays)] * len(arrays)}

    if mode == "weighted_mean":
        energies = np.array([motion_energy(a) for a in arrays], dtype=np.float64)
        if np.allclose(energies.sum(), 0.0):
            weights = np.ones_like(energies) / len(energies)
        else:
            weights = energies / energies.sum()
        fused = np.zeros_like(arrays[0], dtype=np.float32)
        for w, a in zip(weights, arrays):
            fused += float(w) * a.astype(np.float32)
        return fused, {
            "mode": mode,
            "weights": {cam: float(w) for cam, w in zip(cam_list, weights)},
        }

    raise ValueError(f"Unknown fuse mode: {mode}")


def sanitize_indices(indices: list[int], length: int) -> list[int]:
    """Sort, dedupe, clamp, and ensure terminal boundary."""
    valid = sorted({int(i) for i in indices if 0 <= int(i) < length})
    if not valid or valid[-1] != length - 1:
        valid.append(length - 1)
    return valid


def detect_subgoals_single(
    embeddings: np.ndarray,
    min_interval: int = 18,
    gamma: float = 0.08,
    normalize_curve: bool = False,
    smooth_method: str = "kernel",
    window_length: int | None = None,
) -> list[int]:
    """Single UVD embedding decomposition run."""
    _, decomp_meta = embedding_decomp(
        embeddings,
        normalize_curve=normalize_curve,
        min_interval=min_interval,
        smooth_method=smooth_method,
        gamma=gamma,
        window_length=window_length,
        fill_embeddings=False,
    )
    return sanitize_indices(decomp_meta.milestone_indices, len(embeddings))


def consensus_boundaries(
    run_indices: list[list[int]],
    length: int,
    radius: int = 8,
    min_votes: int = 2,
) -> tuple[list[int], list[float], dict]:
    """Merge multiple boundary proposals by voting in a temporal radius."""
    num_runs = max(len(run_indices), 1)
    entries: list[tuple[int, int]] = []
    for run_id, idxs in enumerate(run_indices):
        for idx in idxs:
            if 0 < idx < length - 1:  # exclude implicit start/terminal
                entries.append((int(idx), run_id))

    if not entries:
        return [length - 1], [1.0], {"clusters": [], "num_runs": num_runs}

    entries.sort(key=lambda x: x[0])
    clusters: list[list[tuple[int, int]]] = [[entries[0]]]
    for item in entries[1:]:
        if item[0] - clusters[-1][-1][0] <= radius:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    kept = []
    cluster_meta = []
    for cluster in clusters:
        points = np.array([p for p, _ in cluster], dtype=np.int32)
        run_ids = {rid for _, rid in cluster}
        votes = len(run_ids)
        center = int(np.round(np.median(points)))
        spread = float(np.std(points)) if len(points) > 1 else 0.0
        if votes >= min_votes and 0 < center < length - 1:
            confidence = (votes / num_runs) * np.exp(-spread / max(radius, 1))
            kept.append((center, float(confidence), votes))
        cluster_meta.append(
            {"center": center, "votes": votes, "spread": spread, "count": int(len(points))}
        )

    # Enforce spacing by keeping strongest clusters when too close.
    kept.sort(key=lambda x: x[0])
    filtered: list[tuple[int, float, int]] = []
    for candidate in kept:
        if not filtered or candidate[0] - filtered[-1][0] >= max(1, radius // 2):
            filtered.append(candidate)
        else:
            if (candidate[1], candidate[2]) > (filtered[-1][1], filtered[-1][2]):
                filtered[-1] = candidate

    boundaries = [x[0] for x in filtered] + [length - 1]
    confidences = [round(x[1], 4) for x in filtered] + [1.0]
    return boundaries, confidences, {"clusters": cluster_meta, "num_runs": num_runs}


def detect_subgoals_ensemble(
    embeddings: np.ndarray,
    min_intervals: list[int],
    gammas: list[float],
    normalize_curve: bool,
    smooth_method: str,
    window_length: int | None,
    radius: int,
    min_votes: int,
) -> tuple[list[int], list[float], dict]:
    """Run a grid of UVD decompositions and return consensus boundaries."""
    candidates = []
    run_indices = []
    for min_interval in min_intervals:
        for gamma in gammas:
            idxs = detect_subgoals_single(
                embeddings=embeddings,
                min_interval=min_interval,
                gamma=gamma,
                normalize_curve=normalize_curve,
                smooth_method=smooth_method,
                window_length=window_length,
            )
            run_indices.append(idxs)
            candidates.append(
                {
                    "min_interval": int(min_interval),
                    "gamma": float(gamma),
                    "num_subgoals": int(len(idxs)),
                    "indices": idxs,
                }
            )

    boundaries, confidences, meta = consensus_boundaries(
        run_indices=run_indices,
        length=len(embeddings),
        radius=radius,
        min_votes=min_votes,
    )
    diagnostics = {
        "mode": "ensemble",
        "grid_size": int(len(run_indices)),
        "candidates": candidates,
        "consensus": meta,
    }
    return boundaries, confidences, diagnostics


def detect_with_mode(
    embeddings: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[int], list[float], dict]:
    """Detect subgoals with either single-run or ensemble mode."""
    if args.decomp_mode == "single":
        idxs = detect_subgoals_single(
            embeddings=embeddings,
            min_interval=args.min_interval,
            gamma=args.gamma,
            normalize_curve=args.normalize_curve,
            smooth_method=args.smooth_method,
            window_length=args.window_length,
        )
        conf = [1.0] * len(idxs)
        diag = {
            "mode": "single",
            "min_interval": args.min_interval,
            "gamma": args.gamma,
            "smooth_method": args.smooth_method,
            "normalize_curve": args.normalize_curve,
            "window_length": args.window_length,
        }
        return idxs, conf, diag

    intervals = sorted(set(parse_int_list(args.ensemble_min_intervals) + [args.min_interval]))
    gammas = sorted(set(parse_float_list(args.ensemble_gammas) + [args.gamma]))
    return detect_subgoals_ensemble(
        embeddings=embeddings,
        min_intervals=intervals,
        gammas=gammas,
        normalize_curve=args.normalize_curve,
        smooth_method=args.smooth_method,
        window_length=args.window_length,
        radius=args.consensus_radius,
        min_votes=args.consensus_min_votes,
    )


def auto_choose_segmentation_source(
    per_camera_embeddings: dict[str, np.ndarray],
    fused_embedding: np.ndarray | None,
    preferred_primary: str,
) -> str:
    """Choose embedding source that appears most separable for segmentation."""
    candidates: dict[str, np.ndarray] = {}
    if preferred_primary in per_camera_embeddings:
        candidates[preferred_primary] = per_camera_embeddings[preferred_primary]
    for cam, emb in per_camera_embeddings.items():
        candidates[cam] = emb
    if fused_embedding is not None:
        candidates["fused"] = fused_embedding

    scores = {name: transition_separability_score(emb) for name, emb in candidates.items()}
    best_name = max(scores, key=scores.get)
    return best_name


def process_demos(args: argparse.Namespace) -> dict:
    """Extract embeddings and detect subgoals for all (or selected) demos."""
    hdf5_path = Path(args.hdf5)
    output_dir = Path(args.output_dir)
    embed_dir = output_dir / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)

    available_cameras = detect_available_cameras(hdf5_path)
    primary_camera = "agentview_rgb" if "agentview_rgb" in available_cameras else available_cameras[0]

    if args.multi_view and len(available_cameras) > 1:
        camera_keys = available_cameras
        print(f"Multi-view extraction: {camera_keys}")
    else:
        camera_keys = [primary_camera]
        if args.multi_view:
            print(f"WARNING: --multi-view requested but only {available_cameras} available")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"Loading preprocessor '{args.preprocessor_name}' on {device}...")
    preprocessor = get_preprocessor(args.preprocessor_name, device=device)

    print(
        "UVD alignment: embedding_decomp("
        f"normalize_curve={args.normalize_curve}, min_interval={args.min_interval}, "
        f"smooth_method={args.smooth_method}, gamma={args.gamma})"
    )
    if args.decomp_mode == "ensemble":
        print(
            "Ensemble detection enabled:"
            f" min_intervals={args.ensemble_min_intervals},"
            f" gammas={args.ensemble_gammas},"
            f" consensus_radius={args.consensus_radius},"
            f" consensus_min_votes={args.consensus_min_votes}"
        )

    demo_names = get_demo_names(hdf5_path)
    if args.demo:
        if args.demo not in demo_names:
            print(f"Error: {args.demo} not found in {hdf5_path}")
            return {}
        demo_names = [args.demo]

    all_subgoals = {}
    for demo_name in tqdm(demo_names, desc="Demos"):
        per_camera_embeddings: dict[str, np.ndarray] = {}

        # Step 1: per-camera embeddings.
        for camera_key in camera_keys:
            cache_path = embed_dir / f"{demo_name}_{camera_key}.npy"
            emb = None
            if args.skip_existing and cache_path.exists():
                emb = np.load(cache_path)
                tqdm.write(f"  {demo_name}/{camera_key}: loaded cached {emb.shape}")

            if emb is None:
                try:
                    frames = load_demo_frames(hdf5_path, demo_name, camera_key)
                    tqdm.write(f"  {demo_name}/{camera_key}: extracting from {frames.shape[0]} frames...")
                    try:
                        emb = extract_embeddings(preprocessor, frames, args.batch_size)
                    except RuntimeError as e:
                        if device == "cuda" and is_cuda_kernel_compat_error(e):
                            tqdm.write(
                                "  CUDA kernel incompatibility detected; "
                                "switching extractor to CPU and retrying..."
                            )
                            device = "cpu"
                            preprocessor = get_preprocessor(args.preprocessor_name, device=device)
                            emb = extract_embeddings(preprocessor, frames, args.batch_size)
                        else:
                            raise
                    if args.l2_normalize_embeddings:
                        emb = l2_normalize_embeddings(emb)
                    np.save(cache_path, emb.astype(np.float32))
                    tqdm.write(f"  {demo_name}/{camera_key}: cached {cache_path.name}")
                except KeyError:
                    tqdm.write(f"  {demo_name}/{camera_key}: camera not found, skipping")
                    continue

            per_camera_embeddings[camera_key] = emb.astype(np.float32)

        if not per_camera_embeddings:
            tqdm.write(f"  {demo_name}: no camera embeddings available, skipping")
            continue

        # Step 1b: optional fused embedding.
        fused_embedding, fuse_meta = fuse_embeddings(per_camera_embeddings, args.fuse_mode)
        if fused_embedding is not None:
            fused_path = embed_dir / f"{demo_name}_fused.npy"
            if not (args.skip_existing and fused_path.exists()):
                np.save(fused_path, fused_embedding.astype(np.float32))
            else:
                fused_embedding = np.load(fused_path).astype(np.float32)
            tqdm.write(f"  {demo_name}: fused({args.fuse_mode}) -> {fused_embedding.shape}")

        # Step 1c: choose embedding source for downstream segmentation stage.
        if args.segmentation_source == "auto":
            selected_source = auto_choose_segmentation_source(
                per_camera_embeddings=per_camera_embeddings,
                fused_embedding=fused_embedding,
                preferred_primary=primary_camera,
            )
        elif args.segmentation_source == "fused":
            selected_source = "fused" if fused_embedding is not None else primary_camera
        elif args.segmentation_source == "best_view":
            # best among camera views only (exclude fused)
            scores = {
                cam: transition_separability_score(emb)
                for cam, emb in per_camera_embeddings.items()
            }
            selected_source = max(scores, key=scores.get)
        else:
            selected_source = primary_camera

        if selected_source == "fused" and fused_embedding is not None:
            segmentation_embedding = fused_embedding
        else:
            segmentation_embedding = per_camera_embeddings[selected_source]

        # Step 2: boundary detection source.
        detection_source = args.detection_source
        if detection_source == "auto":
            detection_source = "fused" if selected_source == "fused" else "primary"

        if detection_source == "consensus_views" and len(per_camera_embeddings) > 1:
            per_view = {}
            per_view_indices = []
            for cam, emb in per_camera_embeddings.items():
                idxs, conf, diag = detect_with_mode(emb, args)
                per_view[cam] = {
                    "num_subgoals": int(len(idxs)),
                    "indices": idxs,
                    "confidences": conf,
                    "diagnostics": diag,
                }
                per_view_indices.append(idxs)
            idxs, conf, consensus_diag = consensus_boundaries(
                run_indices=per_view_indices,
                length=len(segmentation_embedding),
                radius=args.view_consensus_radius,
                min_votes=args.view_consensus_min_votes,
            )
            detection_diag = {
                "mode": "consensus_views",
                "per_view": per_view,
                "consensus": consensus_diag,
            }
        else:
            if detection_source == "fused" and fused_embedding is not None:
                det_emb = fused_embedding
                det_label = "fused"
            else:
                det_emb = per_camera_embeddings.get(primary_camera, segmentation_embedding)
                det_label = primary_camera
            idxs, conf, diag = detect_with_mode(det_emb, args)
            detection_diag = {"mode": detection_source, "embedding_source": det_label, "details": diag}

        # Ensure indices are valid for selected segmentation source length.
        idxs = sanitize_indices(idxs, len(segmentation_embedding))
        if len(conf) != len(idxs):
            conf = [1.0] * len(idxs)

        all_subgoals[demo_name] = {
            "num_frames": int(len(segmentation_embedding)),
            "num_subgoals": int(len(idxs)),
            "subgoal_indices": [int(i) for i in idxs],
            "subgoal_confidence": [float(c) for c in conf],
            "camera": selected_source,
            "cameras_extracted": list(per_camera_embeddings.keys()),
            "embedding_dim": int(segmentation_embedding.shape[1]),
            "multi_view": len(per_camera_embeddings) > 1,
            "decomp_params": {
                "preprocessor_name": args.preprocessor_name,
                "mode": args.decomp_mode,
                "min_interval": args.min_interval,
                "gamma": args.gamma,
                "normalize_curve": args.normalize_curve,
                "smooth_method": args.smooth_method,
                "window_length": args.window_length,
                "detection_source": args.detection_source,
                "segmentation_source": args.segmentation_source,
                "fuse_mode": args.fuse_mode,
            },
            "fusion": fuse_meta,
            "detection_diagnostics": detection_diag,
        }

        tqdm.write(
            f"  {demo_name}: selected='{selected_source}', subgoals={len(idxs)} at {idxs}"
        )

    metadata_path = output_dir / "subgoal_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_subgoals, f, indent=2)

    print(f"\nSaved subgoal metadata: {metadata_path}")
    print(f"Embeddings cached in: {embed_dir}/")

    if all_subgoals:
        counts = [v["num_subgoals"] for v in all_subgoals.values()]
        print(f"Subgoal counts: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.1f}")
        selected = [v["camera"] for v in all_subgoals.values()]
        print(f"Segmentation sources used: {sorted(set(selected))}")

    return all_subgoals


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract embeddings and detect UVD subgoals from HDF5 demos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hdf5", type=str, required=True, help="Path to demo HDF5 file")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding extraction")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--demo", type=str, default=None, help="Process one demo only (e.g., demo_0)")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse cached embeddings if present")
    parser.add_argument("--multi-view", action="store_true", help="Extract all available camera views")

    parser.add_argument(
        "--preprocessor-name",
        type=str,
        default="r3m",
        choices=PREPROCESSOR_CHOICES,
        help="UVD visual preprocessor",
    )
    parser.add_argument(
        "--l2-normalize-embeddings",
        action="store_true",
        help="L2-normalize each frame embedding before saving",
    )

    parser.add_argument(
        "--fuse-mode",
        type=str,
        default="weighted_mean",
        choices=["concat", "mean", "weighted_mean"],
        help="How to fuse multi-view embeddings",
    )
    parser.add_argument(
        "--segmentation-source",
        type=str,
        default="auto",
        choices=["auto", "fused", "primary", "best_view"],
        help="Embedding source used by downstream segmentation",
    )
    parser.add_argument(
        "--detection-source",
        type=str,
        default="auto",
        choices=["auto", "primary", "fused", "consensus_views"],
        help="Embedding source used for subgoal boundary detection",
    )

    # UVD decomposition params (aligned with vendored defaults for 'embed').
    parser.add_argument("--min-interval", type=int, default=18, help="Min frames between subgoals")
    parser.add_argument("--gamma", type=float, default=0.08, help="Kernel smoothing gamma")
    parser.add_argument("--normalize-curve", action="store_true", help="Normalize distance curve")
    parser.add_argument(
        "--smooth-method",
        type=str,
        default="kernel",
        choices=["kernel", "savgol"],
        help="Smoothing method for embedding distance curve",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=None,
        help="Optional backward window for decomposition",
    )

    # Accuracy-focused ensemble mode.
    parser.add_argument(
        "--decomp-mode",
        type=str,
        default="ensemble",
        choices=["single", "ensemble"],
        help="Single-run UVD or parameter-ensemble consensus",
    )
    parser.add_argument(
        "--ensemble-min-intervals",
        type=str,
        default="14,18,24",
        help="Comma-separated min_interval values for ensemble",
    )
    parser.add_argument(
        "--ensemble-gammas",
        type=str,
        default="0.06,0.08,0.10",
        help="Comma-separated gamma values for ensemble",
    )
    parser.add_argument("--consensus-radius", type=int, default=8, help="Temporal radius for ensemble consensus")
    parser.add_argument("--consensus-min-votes", type=int, default=2, help="Min votes to keep a boundary")
    parser.add_argument(
        "--view-consensus-radius",
        type=int,
        default=10,
        help="Temporal radius for cross-view consensus",
    )
    parser.add_argument(
        "--view-consensus-min-votes",
        type=int,
        default=2,
        help="Min view votes for cross-view consensus boundaries",
    )

    args = parser.parse_args()
    process_demos(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
