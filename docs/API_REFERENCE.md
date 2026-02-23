# API Reference

## 1. Pipeline Orchestrator
### `scripts/run_pipeline.py`
Runs one or more stages from:
- `extract`
- `segment`
- `visualize`
- `train`

Usage:
```bash
python scripts/run_pipeline.py all --demo <task>
python scripts/run_pipeline.py segment visualize train --demo <task>
```

Path resolution:
- input HDF5: `demos/<task>/demo.hdf5`
- output root: `demos/<task>/output/`

## 2. Extraction API
### `scripts/extract_subgoals.py`
Purpose:
- extract per-frame visual embeddings
- run UVD decomposition for subgoal boundaries
- save metadata consumed by segmentation

Core arguments:
- `--preprocessor-name {vip,r3m,liv,clip,vc1,dinov2}`
- `--multi-view`
- `--decomp-mode {single,ensemble}`
- `--detection-source {auto,primary,fused,consensus_views}`
- `--segmentation-source {auto,fused,primary,best_view}`
- `--fuse-mode {concat,mean,weighted_mean}`
- `--min-interval`, `--gamma`, `--smooth-method`, `--normalize-curve`

Example:
```bash
python scripts/extract_subgoals.py \
  --hdf5 demos/<task>/demo.hdf5 \
  --output-dir demos/<task>/output \
  --multi-view \
  --decomp-mode ensemble \
  --detection-source consensus_views \
  --segmentation-source fused \
  --fuse-mode weighted_mean
```

Outputs:
- `output/embeddings/<demo>_<camera>.npy`
- `output/embeddings/<demo>_fused.npy` (if multi-view)
- `output/subgoal_metadata.json`

## 3. Segmentation API
### `scripts/phase_segmentation.py`
Purpose:
- infer global phase count from all demos
- force each demo to fixed-K boundaries
- produce aligned phase metadata

Example:
```bash
python scripts/phase_segmentation.py \
  --metadata demos/<task>/output/subgoal_metadata.json \
  --embeddings-dir demos/<task>/output/embeddings \
  --output-dir demos/<task>/output \
  --scoring composite --align-mode progress --visualize
```

Output:
- `output/phase_segmentation.json`

## 4. Visualization API
### `scripts/visualize_subgoals.py`
Saves labeled boundary frames and montages.

### `scripts/generate_phase_videos.py`
Saves per-demo MP4 videos with phase overlays.

## 5. Training API
### `online_phase.train.train`
Runs structured online belief training.

Typical invocation (auto-generated config path):
```bash
python -m online_phase.train.train \
  --config demos/<task>/output/train_structured.generated.yaml
```

Outputs:
- `runs/structured/checkpoints/best.pt`
- `runs/structured/metrics*`
- `runs/structured/splits.json`
