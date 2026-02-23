#!/usr/bin/env python3
"""Generic HDF5 field renaming and restructuring tool.

Flexible HDF5 transformation script that allows you to:
- Rename fields (e.g., wrist_rgb -> eye_in_hand_rgb)
- Split fields (e.g., robot0_eef_pos -> ee_states, gripper_states, joint_states)
- Copy fields as-is
- Optionally preserve depth/other companion fields

Configuration is done via preset profiles or custom command-line arguments.

Usage:
    # Use preset profile (for BeliefDP project)
    python scripts/standardize_hdf5.py --input demo.hdf5 --output demo_std.hdf5 --profile beliefdp
    
    # Custom field mappings
    python scripts/standardize_hdf5.py --input demo.hdf5 --output demo_std.hdf5 \
        --rename "wrist_rgb:eye_in_hand_rgb" \
        --split "robot0_eef_pos:ee_states,gripper_states,joint_states"
    
    # Batch processing
    python scripts/standardize_hdf5.py --input-dir demos/ --output-dir demos_std/ --profile beliefdp
    
    # Dry run to preview changes
    python scripts/standardize_hdf5.py --input demo.hdf5 --dry-run --profile beliefdp --verbose
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from tqdm import tqdm


# ============================================================================ #
#                            CONFIGURATION PROFILES                            #
# ============================================================================ #

# Preset transformation profiles for common use cases
PROFILES = {
    "beliefdp": {
        "description": "BeliefDP: wrist_rgb->eye_in_hand_rgb, split robot states",
        "renames": {
            "wrist_rgb": "eye_in_hand_rgb",
            "agentview_rgb": "agentview_rgb",
        },
        "splits": {
            "robot0_eef_pos": {
                "targets": ["ee_states", "gripper_states", "joint_states"],
                "splitter": "split_robot_state_beliefdp",
            }
        },
        "copy_depth": True,
    },
    "minimal": {
        "description": "Minimal: only rename wrist_rgb",
        "renames": {
            "wrist_rgb": "eye_in_hand_rgb",
        },
        "splits": {},
        "copy_depth": True,
    },
}


# ============================================================================ #
#                            SPLITTER FUNCTIONS                                #
# ============================================================================ #

def split_robot_state(robot_state: np.ndarray) -> dict[str, np.ndarray]:
    """Split robot state into ee_states, gripper_states, and joint_states.
    
    Args:
        robot_state: (T, D) array where D is typically 8 or more
        
    Returns:
        Dict with keys 'ee_states', 'gripper_states', 'joint_states'
        
    Logic:
        - If D == 8: assume [ee_pos(3), ee_quat(4), gripper(1)]
        - If D == 16: assume full ee_states
        - If D >= 17: assume [ee(16), gripper(1), joint(7+)]
    """
    T, D = robot_state.shape
    
    if D == 8:
        # Common case: [ee_pos(3), ee_quat(4), gripper(1)]
        ee_pos = robot_state[:, :3]
        ee_quat = robot_state[:, 3:7]
        gripper = robot_state[:, 7:8]
        
        # Construct 16-dim ee_states: [pos(3), quat(4), zeros(9)]
        ee_states = np.concatenate([
            ee_pos,
            ee_quat,
            np.zeros((T, 9), dtype=np.float32)
        ], axis=1)
        
        gripper_states = gripper
        joint_states = np.zeros((T, 7), dtype=np.float32)
        
    elif D == 16:
        # Full ee_states already
        ee_states = robot_state
        gripper_states = np.zeros((T, 1), dtype=np.float32)
        joint_states = np.zeros((T, 7), dtype=np.float32)
        
    elif D >= 17:
        # Full state: [ee(16), gripper(1), joint(7+)]
        ee_states = robot_state[:, :16]
        gripper_states = robot_state[:, 16:17]
        
        if D >= 24:
            joint_states = robot_state[:, 17:24]
        else:
            # Pad joint states if needed
            available_joints = D - 17
            joint_states = np.zeros((T, 7), dtype=np.float32)
            joint_states[:, :available_joints] = robot_state[:, 17:]
    
    else:
        # Fallback: pad everything
        ee_states = np.zeros((T, 16), dtype=np.float32)
        ee_states[:, :min(D, 16)] = robot_state[:, :min(D, 16)]
        
        gripper_states = np.zeros((T, 1), dtype=np.float32)
        joint_states = np.zeros((T, 7), dtype=np.float32)
    
    return {
        "ee_states": ee_states.astype(np.float32),
        "gripper_states": gripper_states.astype(np.float32),
        "joint_states": joint_states.astype(np.float32),
    }


def get_splitter_function(splitter_name: str):
    """Get the splitter function by name.
    
    Args:
        splitter_name: Name of the splitter function
        
    Returns:
        Splitter function
    """
    if splitter_name == "split_robot_state_beliefdp":
        return split_robot_state
    else:
        raise ValueError(f"Unknown splitter: {splitter_name}")


# ============================================================================ #
#                            TRANSFORMATION FUNCTIONS                          #
# ============================================================================ #

def standardize_demo(
    src_demo: h5py.Group,
    dst_demo: h5py.Group,
    config: dict,
    verbose: bool = False
) -> dict:
    """Transform a single demo based on configuration.
    
    Args:
        src_demo: Source demo group
        dst_demo: Destination demo group
        config: Transformation config with 'renames', 'splits', 'copy_depth'
        verbose: Print conversion details
        
    Returns:
        Dict with conversion statistics
    """
    stats = {
        "renamed": [],
        "split": [],
        "copied": [],
        "copied_depth": [],
    }
    
    src_obs = src_demo["obs"]
    dst_obs = dst_demo.create_group("obs")
    
    processed_fields = set()
    
    # -------------------------------------------------------------------------
    # 1. Apply renames
    # -------------------------------------------------------------------------
    renames = config.get("renames", {})
    for old_name, new_name in renames.items():
        if old_name in src_obs:
            data = src_obs[old_name][:]
            dst_obs.create_dataset(new_name, data=data, compression="gzip")
            stats["renamed"].append(f"{old_name} -> {new_name}")
            processed_fields.add(old_name)
            
            if verbose:
                print(f"  ✓ Renamed {old_name} -> {new_name}: {data.shape}")
            
            # Handle companion depth field if enabled
            if config.get("copy_depth", True):
                old_depth = f"{old_name}_depth"
                new_depth = f"{new_name}_depth"
                
                if old_depth in src_obs:
                    depth_data = src_obs[old_depth][:]
                    dst_obs.create_dataset(new_depth, data=depth_data, compression="gzip")
                    stats["copied_depth"].append(f"{old_depth} -> {new_depth}")
                    processed_fields.add(old_depth)
                    
                    if verbose:
                        print(f"  ✓ Copied depth {old_depth} -> {new_depth}: {depth_data.shape}")
    
    # -------------------------------------------------------------------------
    # 2. Apply splits
    # -------------------------------------------------------------------------
    splits = config.get("splits", {})
    for source_field, split_config in splits.items():
        if source_field in src_obs:
            data = src_obs[source_field][:]
            splitter_name = split_config["splitter"]
            target_fields = split_config["targets"]
            
            # Get splitter function and apply
            splitter_func = get_splitter_function(splitter_name)
            split_results = splitter_func(data)
            
            for target_field in target_fields:
                if target_field in split_results:
                    target_data = split_results[target_field]
                    dst_obs.create_dataset(target_field, data=target_data, compression="gzip")
                    stats["split"].append(f"{source_field} -> {target_field}")
                    
                    if verbose:
                        print(f"  ✓ Split {source_field} -> {target_field}: {target_data.shape}")
            
            processed_fields.add(source_field)
    
    # -------------------------------------------------------------------------
    # 3. Copy all other observation fields as-is
    # -------------------------------------------------------------------------
    for key in src_obs.keys():
        if key not in processed_fields and key not in dst_obs:
            data = src_obs[key][:]
            dst_obs.create_dataset(key, data=data, compression="gzip")
            stats["copied"].append(key)
            
            if verbose:
                print(f"  ✓ Copied {key}: {data.shape}")
    
    # -------------------------------------------------------------------------
    # 4. Copy actions
    # -------------------------------------------------------------------------
    if "actions" in src_demo:
        actions = src_demo["actions"][:]
        dst_demo.create_dataset("actions", data=actions, compression="gzip")
        
        if verbose:
            print(f"  ✓ actions: {actions.shape}")
    
    # -------------------------------------------------------------------------
    # 5. Copy attributes
    # -------------------------------------------------------------------------
    for attr_name, attr_value in src_demo.attrs.items():
        dst_demo.attrs[attr_name] = attr_value
    
    return stats


def standardize_hdf5_file(
    input_path: Path,
    output_path: Path,
    config: dict,
    verbose: bool = False,
    dry_run: bool = False,
) -> dict:
    """Transform an entire HDF5 file based on configuration.
    
    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output HDF5 file
        config: Transformation configuration
        verbose: Print detailed conversion info
        dry_run: Don't actually write output, just report what would happen
        
    Returns:
        Dict with overall statistics
    """
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing: {input_path}")
    if config.get("description"):
        print(f"Profile: {config['description']}")
    
    overall_stats = {
        "total_demos": 0,
        "renamed": set(),
        "split": set(),
        "copied": set(),
    }
    
    with h5py.File(input_path, 'r') as src_file:
        # Get all demo names
        data_group = src_file["data"]
        demo_names = sorted(
            [k for k in data_group.keys() if k.startswith("demo")],
            key=lambda s: int(s.split("_")[-1]),
        )
        overall_stats["total_demos"] = len(demo_names)
        
        if dry_run:
            print(f"\nWould process {len(demo_names)} demos")
            
            # Analyze first demo to show what would be converted
            if demo_names:
                demo_0 = data_group[demo_names[0]]
                obs_0 = demo_0["obs"]
                
                print(f"\nSource observation keys: {sorted(obs_0.keys())}")
                print(f"Source actions: {'actions' in demo_0}")
                
                print(f"\nTransformations:")
                
                # Show renames
                renames = config.get("renames", {})
                if renames:
                    print("  Renames:")
                    for old_name, new_name in renames.items():
                        if old_name in obs_0:
                            print(f"    {old_name} -> {new_name}")
                            if config.get("copy_depth", True):
                                old_depth = f"{old_name}_depth"
                                new_depth = f"{new_name}_depth"
                                if old_depth in obs_0:
                                    print(f"    {old_depth} -> {new_depth}")
                
                # Show splits
                splits = config.get("splits", {})
                if splits:
                    print("  Splits:")
                    for source_field, split_config in splits.items():
                        if source_field in obs_0:
                            targets = ", ".join(split_config["targets"])
                            print(f"    {source_field} -> {targets}")
                
                # Show what will be copied as-is
                processed = set(renames.keys()) | set(splits.keys())
                depth_fields = {f"{k}_depth" for k in renames.keys()}
                processed |= depth_fields
                
                will_copy = [k for k in obs_0.keys() if k not in processed]
                if will_copy:
                    print(f"  Copy as-is: {', '.join(sorted(will_copy))}")
            
            return overall_stats
        
        # Create output file
        with h5py.File(output_path, 'w') as dst_file:
            # Copy top-level attributes
            for attr_name, attr_value in src_file.attrs.items():
                dst_file.attrs[attr_name] = attr_value
            
            # Create data group
            dst_data = dst_file.create_group("data")
            
            # Optionally copy camera group if it exists
            if "camera" in src_file:
                src_camera = src_file["camera"]
                dst_camera = dst_file.create_group("camera")
                
                def copy_group_recursive(src_grp, dst_grp):
                    for key in src_grp.keys():
                        item = src_grp[key]
                        if isinstance(item, h5py.Group):
                            dst_subgrp = dst_grp.create_group(key)
                            copy_group_recursive(item, dst_subgrp)
                        else:
                            dst_grp.create_dataset(key, data=item[:])
                
                copy_group_recursive(src_camera, dst_camera)
            
            # Convert each demo
            print(f"\nProcessing {len(demo_names)} demos...")
            for demo_name in tqdm(demo_names, desc="Demos"):
                src_demo = data_group[demo_name]
                dst_demo = dst_data.create_group(demo_name)
                
                if verbose:
                    print(f"\n{demo_name}:")
                
                stats = standardize_demo(src_demo, dst_demo, config, verbose=verbose)
                
                # Aggregate stats
                for item in stats["renamed"]:
                    overall_stats["renamed"].add(item)
                for item in stats["split"]:
                    overall_stats["split"].add(item)
                for item in stats["copied"]:
                    overall_stats["copied"].add(item)
    
    print(f"\n✓ Transformed HDF5 saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Total demos: {overall_stats['total_demos']}")
    if overall_stats["renamed"]:
        print(f"  Renamed: {sorted(overall_stats['renamed'])}")
    if overall_stats["split"]:
        print(f"  Split: {sorted(overall_stats['split'])}")
    if overall_stats["copied"]:
        print(f"  Copied as-is: {sorted(overall_stats['copied'])}")
    
    return overall_stats


def standardize_directory(
    input_dir: Path,
    output_dir: Path,
    config: dict,
    verbose: bool = False,
    dry_run: bool = False,
) -> None:
    """Transform all HDF5 files in a directory.
    
    Args:
        input_dir: Directory containing demo.hdf5 files
        output_dir: Directory to write transformed files
        config: Transformation configuration
        verbose: Print detailed conversion info
        dry_run: Don't actually write output
    """
    # Find all demo.hdf5 files
    hdf5_files = list(input_dir.glob("*/demo.hdf5"))
    
    if not hdf5_files:
        print(f"No demo.hdf5 files found in {input_dir}")
        return
    
    print(f"Found {len(hdf5_files)} demo files")
    
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    for hdf5_file in hdf5_files:
        demo_name = hdf5_file.parent.name
        output_demo_dir = output_dir / demo_name
        
        if not dry_run:
            output_demo_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_demo_dir / "demo.hdf5"
        
        standardize_hdf5_file(hdf5_file, output_file, config, verbose=verbose, dry_run=dry_run)


# ============================================================================ #
#                                    MAIN                                      #
# ============================================================================ #

def parse_custom_mappings(rename_args, split_args):
    """Parse custom mapping arguments into config dict.
    
    Args:
        rename_args: List of "old:new" rename mappings
        split_args: List of "source:target1,target2,..." split mappings
        
    Returns:
        Config dict with renames and splits
    """
    config = {"renames": {}, "splits": {}, "copy_depth": True}
    
    if rename_args:
        for mapping in rename_args:
            old, new = mapping.split(":", 1)
            config["renames"][old.strip()] = new.strip()
    
    if split_args:
        for mapping in split_args:
            source, targets = mapping.split(":", 1)
            target_list = [t.strip() for t in targets.split(",")]
            config["splits"][source.strip()] = {
                "targets": target_list,
                "splitter": "split_robot_state_beliefdp",  # Default splitter
            }
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generic HDF5 field transformation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Input/output options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--input", type=Path, help="Input HDF5 file")
    input_group.add_argument("--input-dir", type=Path, help="Input directory with demo subdirs")
    
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output", type=Path, help="Output HDF5 file")
    output_group.add_argument("--output-dir", type=Path, help="Output directory")
    output_group.add_argument("--inplace", action="store_true", help="Overwrite input file")
    
    # Configuration options
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        help=f"Use preset profile: {', '.join(PROFILES.keys())}"
    )
    config_group.add_argument(
        "--rename",
        action="append",
        metavar="OLD:NEW",
        help="Rename field (can specify multiple times). Example: --rename wrist_rgb:eye_in_hand_rgb"
    )
    config_group.add_argument(
        "--split",
        action="append",
        metavar="SOURCE:TARGET1,TARGET2,...",
        help="Split field into multiple targets. Example: --split robot0_eef_pos:ee_states,gripper_states,joint_states"
    )
    config_group.add_argument(
        "--no-depth",
        action="store_true",
        help="Don't automatically copy companion depth fields"
    )
    
    # Options
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed conversion info")
    parser.add_argument("--dry-run", action="store_true", help="Don't write output, just show what would be done")
    parser.add_argument("--list-profiles", action="store_true", help="List available profiles and exit")
    
    args = parser.parse_args()
    
    # List profiles if requested
    if args.list_profiles:
        print("Available profiles:")
        for name, profile in PROFILES.items():
            print(f"\n  {name}:")
            print(f"    {profile['description']}")
            if profile['renames']:
                print(f"    Renames: {profile['renames']}")
            if profile['splits']:
                print(f"    Splits: {list(profile['splits'].keys())}")
        return 0
    
    # Determine configuration
    if args.profile:
        config = PROFILES[args.profile].copy()
        # Override with custom mappings if provided
        if args.rename or args.split:
            custom = parse_custom_mappings(args.rename, args.split)
            config["renames"].update(custom["renames"])
            config["splits"].update(custom["splits"])
    elif args.rename or args.split:
        config = parse_custom_mappings(args.rename, args.split)
    elif not args.list_profiles:
        print("Error: Must specify either --profile or custom --rename/--split mappings")
        return 1
    else:
        config = {}
    
    if args.no_depth:
        config["copy_depth"] = False
    
    # Validate arguments
    if not args.input and not args.input_dir and not args.list_profiles:
        print("Error: Must specify --input or --input-dir")
        return 1
    
    if args.input:
        if not args.input.exists():
            print(f"Error: Input file not found: {args.input}")
            return 1
        
        if args.inplace:
            # Create temporary file, then move
            import tempfile
            temp_file = Path(tempfile.mktemp(suffix=".hdf5"))
            standardize_hdf5_file(args.input, temp_file, config, verbose=args.verbose, dry_run=args.dry_run)
            
            if not args.dry_run:
                shutil.move(str(temp_file), str(args.input))
                print(f"\n✓ Replaced {args.input} with transformed version")
        
        elif args.output:
            standardize_hdf5_file(args.input, args.output, config, verbose=args.verbose, dry_run=args.dry_run)
        
        else:
            # Default: add _transformed suffix
            output_path = args.input.parent / f"{args.input.stem}_transformed.hdf5"
            standardize_hdf5_file(args.input, output_path, config, verbose=args.verbose, dry_run=args.dry_run)
    
    elif args.input_dir:
        if not args.input_dir.exists():
            print(f"Error: Input directory not found: {args.input_dir}")
            return 1
        
        if args.output_dir:
            standardize_directory(args.input_dir, args.output_dir, config, verbose=args.verbose, dry_run=args.dry_run)
        else:
            # Default: create output directory with _transformed suffix
            output_dir = args.input_dir.parent / f"{args.input_dir.name}_transformed"
            standardize_directory(args.input_dir, output_dir, config, verbose=args.verbose, dry_run=args.dry_run)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
