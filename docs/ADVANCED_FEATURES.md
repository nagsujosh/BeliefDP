# Advanced Features

## 1. Ensemble UVD Decomposition
The extractor supports a decomposition ensemble over `(min_interval, gamma)` pairs.
Each run proposes boundary indices, and a temporal voting layer produces consensus boundaries.

Benefits:
- lower sensitivity to one hyperparameter setting
- better robustness under noisy motion or brief pauses

Key arguments:
- `--decomp-mode ensemble`
- `--ensemble-min-intervals`
- `--ensemble-gammas`
- `--consensus-radius`
- `--consensus-min-votes`

## 2. Cross-View Consensus Detection
When `--multi-view` is enabled, detection can be run per camera and fused by consensus.

Key arguments:
- `--detection-source consensus_views`
- `--view-consensus-radius`
- `--view-consensus-min-votes`

## 3. Embedding Fusion Strategies
Three fusion options are available:
- `concat`: channel concat of view embeddings
- `mean`: uniform average
- `weighted_mean`: motion-energy weighted average

Key argument:
- `--fuse-mode {concat,mean,weighted_mean}`

## 4. Automatic Source Selection
The extractor can choose which embedding stream is best for segmentation using a transition separability heuristic.

Key argument:
- `--segmentation-source auto`

Alternative choices:
- `primary`
- `best_view`
- `fused`

## 5. UVD Compatibility Notes
UVD utilities import optional simulation dependencies (`allenact`, `gym`).
The extractor provides runtime import shims so offline decomposition can run without installing full simulation stacks.

## 6. Example: Full Robust Extract
```bash
python scripts/extract_subgoals.py \
  --hdf5 demos/pick_and_place_kitkat/demo.hdf5 \
  --output-dir demos/pick_and_place_kitkat/output \
  --multi-view \
  --decomp-mode ensemble \
  --detection-source consensus_views \
  --segmentation-source fused \
  --fuse-mode weighted_mean \
  --ensemble-min-intervals 14,18,24 \
  --ensemble-gammas 0.06,0.08,0.10
```
