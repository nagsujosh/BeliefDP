"""Camera key utilities for handling different naming conventions.

This module provides utilities to work with both old and new camera naming:
- Old: wrist_rgb, agentview_rgb
- New: eye_in_hand_rgb, agentview_rgb

Auto-detection ensures backward compatibility while standardizing on new names.
"""
from __future__ import annotations

from typing import List, Optional
import h5py


# ============================================================================ #
#                            CAMERA NAME MAPPINGS                              #
# ============================================================================ #

# Canonical camera names (standardized format)
CANONICAL_CAMERAS = ["agentview_rgb", "eye_in_hand_rgb"]

# Legacy aliases
CAMERA_ALIASES = {
    "wrist_rgb": "eye_in_hand_rgb",
    "eye_in_hand_rgb": "eye_in_hand_rgb",
    "agentview_rgb": "agentview_rgb",
}

# Short names for display/logging
CAMERA_SHORT_NAMES = {
    "agentview_rgb": "agentview",
    "eye_in_hand_rgb": "eye_in_hand",
    "wrist_rgb": "wrist",  # legacy
}

# Depth camera names
DEPTH_SUFFIX = "_depth"


# ============================================================================ #
#                              DETECTION FUNCTIONS                             #
# ============================================================================ #

def detect_available_cameras(hdf5_file: h5py.File, demo_name: str = "demo_0") -> List[str]:
    """Detect which cameras are available in an HDF5 demo file.
    
    Args:
        hdf5_file: Open HDF5 file handle
        demo_name: Demo to check (default: first demo)
        
    Returns:
        List of canonical camera names found (e.g., ['agentview_rgb', 'eye_in_hand_rgb'])
    """
    try:
        obs = hdf5_file[f"data/{demo_name}/obs"]
    except KeyError:
        # Fallback to any demo if demo_0 doesn't exist
        data = hdf5_file["data"]
        demo_names = [k for k in data.keys() if k.startswith("demo")]
        if not demo_names:
            return []
        obs = data[demo_names[0]]["obs"]
    
    available_canonical = []
    
    # Check each possible camera name (both old and new)
    for key in obs.keys():
        # Skip depth images
        if key.endswith(DEPTH_SUFFIX):
            continue
        
        # Map to canonical name
        canonical_name = CAMERA_ALIASES.get(key, None)
        if canonical_name and canonical_name not in available_canonical:
            available_canonical.append(canonical_name)
    
    return sorted(available_canonical)


def resolve_camera_key(hdf5_file: h5py.File, canonical_name: str, demo_name: str = "demo_0") -> Optional[str]:
    """Resolve a canonical camera name to the actual key in the HDF5 file.
    
    Args:
        hdf5_file: Open HDF5 file handle
        canonical_name: Canonical camera name (e.g., 'eye_in_hand_rgb')
        demo_name: Demo to check
        
    Returns:
        Actual key in the file (e.g., 'wrist_rgb' or 'eye_in_hand_rgb'), or None if not found
    """
    try:
        obs = hdf5_file[f"data/{demo_name}/obs"]
    except KeyError:
        return None
    
    # Direct check
    if canonical_name in obs:
        return canonical_name
    
    # Check aliases
    for alias, canon in CAMERA_ALIASES.items():
        if canon == canonical_name and alias in obs:
            return alias
    
    return None


def get_camera_data(
    hdf5_group: h5py.Group,
    canonical_name: str,
    allow_legacy: bool = True,
) -> Optional[h5py.Dataset]:
    """Get camera data from an obs group, handling both old and new names.
    
    Args:
        hdf5_group: obs/ group
        canonical_name: Canonical camera name
        allow_legacy: If True, check legacy names as fallback
        
    Returns:
        Dataset or None if not found
    """
    # Try canonical name first
    if canonical_name in hdf5_group:
        return hdf5_group[canonical_name]
    
    if allow_legacy:
        # Try legacy aliases
        for alias, canon in CAMERA_ALIASES.items():
            if canon == canonical_name and alias in hdf5_group:
                return hdf5_group[alias]
    
    return None


def get_camera_depth(
    hdf5_group: h5py.Group,
    canonical_name: str,
    allow_legacy: bool = True,
) -> Optional[h5py.Dataset]:
    """Get depth data for a camera, handling both old and new names.
    
    Args:
        hdf5_group: obs/ group
        canonical_name: Canonical camera name (RGB)
        allow_legacy: If True, check legacy names as fallback
        
    Returns:
        Depth dataset or None if not found
    """
    depth_name = f"{canonical_name}{DEPTH_SUFFIX}"
    
    # Try canonical depth name first
    if depth_name in hdf5_group:
        return hdf5_group[depth_name]
    
    if allow_legacy:
        # Try legacy aliases
        for alias, canon in CAMERA_ALIASES.items():
            if canon == canonical_name:
                legacy_depth = f"{alias}{DEPTH_SUFFIX}"
                if legacy_depth in hdf5_group:
                    return hdf5_group[legacy_depth]
    
    return None


def normalize_camera_keys(camera_keys: List[str]) -> List[str]:
    """Convert a list of camera keys (possibly legacy) to canonical names.
    
    Args:
        camera_keys: List of camera names (may include legacy names)
        
    Returns:
        List of canonical camera names
    """
    canonical = []
    for key in camera_keys:
        canon = CAMERA_ALIASES.get(key, key)
        if canon not in canonical:
            canonical.append(canon)
    return canonical


def get_short_name(camera_key: str) -> str:
    """Get short display name for a camera key.
    
    Args:
        camera_key: Camera name (canonical or legacy)
        
    Returns:
        Short name for display (e.g., 'agentview', 'eye_in_hand', 'wrist')
    """
    return CAMERA_SHORT_NAMES.get(camera_key, camera_key)


# ============================================================================ #
#                           BACKWARDS COMPATIBILITY                            #
# ============================================================================ #

def ensure_backward_compatible_config(config: dict) -> dict:
    """Ensure config uses canonical camera names.
    
    Args:
        config: Training/inference config dict
        
    Returns:
        Updated config with canonical camera names
    """
    if "camera_keys" in config:
        config["camera_keys"] = normalize_camera_keys(config["camera_keys"])
    
    if "camera_key" in config:
        # Single camera key
        canonical = CAMERA_ALIASES.get(config["camera_key"], config["camera_key"])
        config["camera_key"] = canonical
    
    return config
