#!/usr/bin/env python3
"""Convert LeRobot parquet dataset to HDF5 format compatible with BeliefDP."""

import argparse
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import io

def convert_lerobot_to_hdf5(parquet_dir: Path, output_hdf5: Path):
    """Convert LeRobot parquet files to HDF5 format.
    
    Args:
        parquet_dir: Directory containing episode_*.parquet files
        output_hdf5: Output HDF5 file path
    """
    # Find all parquet files
    parquet_files = sorted(parquet_dir.glob("episode_*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"No episode parquet files found in {parquet_dir}")
    
    print(f"Found {len(parquet_files)} episodes")
    
    # Create HDF5 file
    with h5py.File(output_hdf5, 'w') as hf:
        hf.attrs["total"] = len(parquet_files)
        
        for i, parquet_file in enumerate(tqdm(parquet_files, desc="Converting")):
            # Read parquet file
            df = pd.read_parquet(parquet_file)
            
            demo_key = f"demo_{i}"
            demo_group = hf.create_group(f"data/{demo_key}")
            
            # Extract and convert images
            if "image" in df.columns:
                # Convert PIL images to numpy arrays
                images = []
                for img in df["image"]:
                    if isinstance(img, dict) and 'bytes' in img:
                        # Decode bytes to PIL Image
                        pil_img = Image.open(io.BytesIO(img['bytes']))
                        images.append(np.array(pil_img, dtype=np.uint8))
                    elif isinstance(img, Image.Image):
                        images.append(np.array(img, dtype=np.uint8))
                    else:
                        images.append(np.array(img, dtype=np.uint8))
                images = np.array(images, dtype=np.uint8)
                demo_group.create_dataset("obs/agentview_rgb", data=images, compression="gzip")
            
            if "wrist_image" in df.columns:
                wrist_images = []
                for img in df["wrist_image"]:
                    if isinstance(img, dict) and 'bytes' in img:
                        # Decode bytes to PIL Image
                        pil_img = Image.open(io.BytesIO(img['bytes']))
                        wrist_images.append(np.array(pil_img, dtype=np.uint8))
                    elif isinstance(img, Image.Image):
                        wrist_images.append(np.array(img, dtype=np.uint8))
                    else:
                        wrist_images.append(np.array(img, dtype=np.uint8))
                wrist_images = np.array(wrist_images, dtype=np.uint8)
                demo_group.create_dataset("obs/wrist_rgb", data=wrist_images, compression="gzip")
            
            # Extract states (convert object arrays to float arrays)
            if "state" in df.columns:
                states = np.stack([np.array(s, dtype=np.float32) for s in df["state"]])
                demo_group.create_dataset("obs/robot0_eef_pos", data=states, compression="gzip")
            
            # Extract actions (convert object arrays to float arrays)
            if "actions" in df.columns:
                actions = np.stack([np.array(a, dtype=np.float32) for a in df["actions"]])
                demo_group.create_dataset("actions", data=actions, compression="gzip")
            
            # Store episode length
            demo_group.attrs["num_samples"] = len(df)
    
    print(f"✓ Converted to {output_hdf5}")
    
    # Print summary
    with h5py.File(output_hdf5, 'r') as hf:
        print(f"\nDataset summary:")
        print(f"  Total demos: {hf.attrs['total']}")
        demo_0 = hf[f"data/demo_0"]
        print(f"  Keys in demo_0: {list(demo_0.keys())}")
        for key in demo_0.keys():
            if isinstance(demo_0[key], h5py.Group):
                print(f"    {key}/: {list(demo_0[key].keys())}")
                for subkey in demo_0[key].keys():
                    shape = demo_0[f"{key}/{subkey}"].shape
                    dtype = demo_0[f"{key}/{subkey}"].dtype
                    print(f"      {subkey}: {shape} {dtype}")
            else:
                shape = demo_0[key].shape
                dtype = demo_0[key].dtype
                print(f"    {key}: {shape} {dtype}")

def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot parquet dataset to HDF5")
    parser.add_argument("demo_name", help="Demo name in demos/ directory")
    parser.add_argument("--output", help="Output HDF5 path (default: demos/{demo_name}/demo.hdf5)")
    
    args = parser.parse_args()
    
    # Setup paths
    demo_dir = Path("demos") / args.demo_name
    parquet_dir = demo_dir / "data" / "chunk-000"
    
    if not parquet_dir.exists():
        raise ValueError(f"Parquet directory not found: {parquet_dir}")
    
    output_hdf5 = Path(args.output) if args.output else demo_dir / "demo.hdf5"
    
    # Convert
    convert_lerobot_to_hdf5(parquet_dir, output_hdf5)
    
    print(f"\n✓ Ready for pipeline:")
    print(f"  python run_pipeline.py all --demo {args.demo_name}")

if __name__ == "__main__":
    main()
