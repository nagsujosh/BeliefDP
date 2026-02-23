# Configuration Guide

## 1. Design Principle
Configuration is split across three levels:
1. extraction knobs (`extract_subgoals.py`)
2. segmentation knobs (`phase_segmentation.py`)
3. model/training knobs (`online_phase/configs/train_structured.yaml`)

## 2. Extraction Configuration
### 2.1 UVD-aligned baseline
- `--min-interval=18`
- `--gamma=0.08`
- `--smooth-method=kernel`
- `--normalize-curve` off by default

### 2.2 Accuracy-focused options
- `--decomp-mode ensemble`
- `--ensemble-min-intervals 14,18,24`
- `--ensemble-gammas 0.06,0.08,0.10`
- `--consensus-radius 8`
- `--consensus-min-votes 2`
- `--detection-source consensus_views`
- `--view-consensus-radius 10`
- `--view-consensus-min-votes 2`
- `--segmentation-source fused`
- `--fuse-mode weighted_mean`

### 2.3 Camera and embedding controls
- `--multi-view`: extract all known camera streams
- `--preprocessor-name`: choose feature encoder
- `--l2-normalize-embeddings`: optional per-frame normalization

## 3. Segmentation Configuration
Main knobs:
- `--num-phases`: set fixed K manually
- omit `--num-phases`: infer K from all demos
- `--scoring composite` (recommended)
- `--align-mode progress` (recommended)
- `--min-anchor-distance`
- `--smooth-window`

## 4. Training Configuration
Base template:
- `online_phase/configs/train_structured.yaml`

Pipeline-generated run config:
- `demos/<task>/output/train_structured.generated.yaml`

Auto-filled fields:
- `segmentation_json`
- `embeddings_dir`
- `hdf5_path`
- `num_phases`
- `output_dir`

Core structured-model weights:
- `w_emission`
- `w_belief`
- `w_hmm`
- `w_progress`
- `w_boundary`

## 5. Recommended Presets
### Stable baseline
```bash
python scripts/run_pipeline.py all --demo <task>
```

### Maximum extraction robustness
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
