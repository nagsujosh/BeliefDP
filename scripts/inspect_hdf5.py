"""Inspect the structure and contents of an HDF5 demonstration file.

Prints demo counts, per-demo frame counts, observation keys, shapes, and
dtypes without loading large arrays into memory.

Usage:
    python scripts/inspect_hdf5.py --file demo.hdf5
    python scripts/inspect_hdf5.py --file demo.hdf5 --demo demo_0 --verbose
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def print_tree(group: h5py.Group, indent: int = 0) -> None:
    """Recursively print the HDF5 group/dataset tree."""
    prefix = "  " * indent
    for key in sorted(group.keys()):
        item = group[key]
        if isinstance(item, h5py.Group):
            print(f"{prefix}{key}/")
            print_tree(item, indent + 1)
        else:
            print(f"{prefix}{key}  shape={item.shape}  dtype={item.dtype}")


def summarize_demos(f: h5py.File) -> None:
    """Print a summary table of all demos in the HDF5 file."""
    data = f["data"]
    demo_names = sorted(
        [k for k in data.keys() if k.startswith("demo")],
        key=lambda s: int(s.split("_")[-1]),
    )

    print(f"\nTotal demos: {len(demo_names)}")

    # Discover observation keys from first demo
    first_obs = data[demo_names[0]]["obs"]
    obs_keys = sorted(first_obs.keys())
    print(f"Observation keys: {obs_keys}")

    # Header
    header = f"{'Demo':<12} {'Timesteps':>10}"
    for k in obs_keys:
        header += f"  {k:>28}"
    if "actions" in data[demo_names[0]]:
        header += f"  {'actions':>16}"
    print(f"\n{header}")
    print("-" * len(header))

    timesteps_list = []
    for name in demo_names:
        demo = data[name]
        obs = demo["obs"]
        # Get timestep count from first obs key
        n_steps = obs[obs_keys[0]].shape[0]
        timesteps_list.append(n_steps)

        row = f"{name:<12} {n_steps:>10}"
        for k in obs_keys:
            ds = obs[k]
            row += f"  {str(ds.shape):>28}"
        if "actions" in demo:
            row += f"  {str(demo['actions'].shape):>16}"
        print(row)

    ts = np.array(timesteps_list)
    print(f"\nTimestep stats: min={ts.min()}, max={ts.max()}, "
          f"mean={ts.mean():.1f}, total={ts.sum()}")


def inspect_single_demo(f: h5py.File, demo_name: str) -> None:
    """Print detailed info for a single demo."""
    demo = f["data"][demo_name]
    print(f"\n=== {demo_name} ===")
    print_tree(demo)

    if "actions" in demo:
        actions = demo["actions"][:]
        print(f"\nActions range: min={actions.min():.4f}, max={actions.max():.4f}")
        print(f"Actions mean:  {actions.mean(axis=0)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect HDF5 demonstration file structure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--file", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--demo", type=str, default=None, help="Inspect a single demo")
    parser.add_argument("--verbose", action="store_true", help="Print full tree")
    args = parser.parse_args()

    hdf5_path = Path(args.file)
    if not hdf5_path.exists():
        print(f"Error: {hdf5_path} not found")
        return 1

    with h5py.File(hdf5_path, "r") as f:
        print(f"=== {hdf5_path.name} ===")
        print(f"Top-level keys: {list(f.keys())}")

        if args.verbose:
            print("\nFull tree:")
            print_tree(f)

        if args.demo:
            inspect_single_demo(f, args.demo)
        else:
            summarize_demos(f)

    return 0


if __name__ == "__main__":
    sys.exit(main())
