# BeliefDP: Belief-Based Diffusion Policy at Scale

## Abstract
This repository implements a practical robotics pipeline that converts raw demonstration videos in `demo.hdf5` into globally aligned phase labels and trains an online phase-belief model. The pipeline is intentionally constrained to a single reproducible flow:

`demo.hdf5 -> embedding extraction + subgoal detection -> global fixed-K phase segmentation -> visualization -> structured belief training`

The extraction stage is UVD-aligned and enhanced with ensemble decomposition and cross-view consensus to improve boundary stability.

## 1. Problem Setting
Given multiple demonstration trajectories of the same task:
1. Detect semantically meaningful transition points.
2. Infer a global phase count shared across demonstrations.
3. Re-segment every demo onto that fixed phase count.
4. Train an online model that predicts phase belief during rollout.

## 2. Method Overview
The system is split into four executable stages.

1. `extract`: visual embedding extraction + UVD decomposition-based subgoals.
2. `segment`: global phase-count inference + fixed-K per-demo segmentation.
3. `visualize`: frame-level and video-level phase visualization.
4. `train`: structured Transformer + Neural-HMM belief model.

## 3. Repository Layout
```text
scripts/
  run_pipeline.py
  extract_subgoals.py
  phase_segmentation.py
  visualize_subgoals.py
  generate_phase_videos.py
  inspect_hdf5.py
  standardize_hdf5.py
  convert_lerobot_to_hdf5.py

online_phase/
  configs/train_structured.yaml
  train/train.py
  ...

demos/<task>/
  demo.hdf5
  output/
    embeddings/
    subgoal_metadata.json
    phase_segmentation.json
    subgoals/
    phase_videos/
    runs/structured/
```

## 4. Environment
```bash
conda env create -f environment.yml
conda activate beliefdp
pip install -e external/r3m
pip install -e .
```

## 5. Running the Pipeline
### 5.1 Full pipeline
```bash
python scripts/run_pipeline.py all --demo pick_and_place_kitkat
```

### 5.2 Stage-by-stage
```bash
python scripts/run_pipeline.py extract --demo pick_and_place_kitkat
python scripts/run_pipeline.py segment --demo pick_and_place_kitkat
python scripts/run_pipeline.py visualize --demo pick_and_place_kitkat
python scripts/run_pipeline.py train --demo pick_and_place_kitkat
```

## 6. Extraction Details (UVD-Aligned + Enhanced)
### 6.1 UVD alignment
Extraction uses vendored UVD `embedding_decomp(...)` with aligned default decomposition parameters:
- `normalize_curve=False`
- `min_interval=18`
- `smooth_method=kernel`
- `gamma=0.08`

### 6.2 Added robustness improvements
- `decomp-mode=single|ensemble`
- cross-parameter ensemble voting over `(min_interval, gamma)`
- cross-view consensus (`detection-source=consensus_views`)
- fused embedding options (`concat|mean|weighted_mean`)
- automatic embedding-source selection for downstream segmentation

### 6.3 Recommended high-accuracy extract command
```bash
python scripts/extract_subgoals.py \
  --hdf5 demos/pick_and_place_kitkat/demo.hdf5 \
  --output-dir demos/pick_and_place_kitkat/output \
  --multi-view \
  --decomp-mode ensemble \
  --detection-source consensus_views \
  --segmentation-source fused \
  --fuse-mode weighted_mean
```

This explicitly uses both camera streams when available.

## 7. Segmentation Details
`phase_segmentation.py` performs:
1. transition-signal construction from embeddings,
2. global phase-count inference from all demos,
3. fixed-K re-segmentation per demo,
4. cross-demo alignment in progress space,
5. diagnostics and plots.

Output file: `demos/<task>/output/phase_segmentation.json`

## 8. Training Details
Training entrypoint: `python -m online_phase.train.train --config <config.yaml>`

Default pipeline behavior generates demo-specific config:
- `demos/<task>/output/train_structured.generated.yaml`

The generated config auto-fills:
- segmentation path
- embeddings path
- HDF5 path
- inferred `num_phases`
- run output directory

## 9. Output Contract
### Required for segmentation
- `output/subgoal_metadata.json`
- `output/embeddings/*.npy`

### Required for training
- `output/phase_segmentation.json`
- `output/embeddings/*.npy`

## 10. Common Failure Modes
1. Missing `subgoal_metadata.json`: extraction failed or was skipped.
2. Missing `phase_segmentation.json`: segmentation failed or was skipped.
3. CUDA runtime issue: rerun extraction with `--device cpu`.
4. Missing UVD optional deps: extractor includes runtime shims for `allenact` and `gym` imports.

## 11. Reproducibility Notes
- Keep `demo.hdf5` unchanged between runs.
- Use fixed extraction parameters for comparable boundaries.
- Save and version `subgoal_metadata.json` and `phase_segmentation.json` when comparing runs.

## 12. Citation
If you use this codebase, cite your own project/paper and the underlying UVD work where appropriate.
