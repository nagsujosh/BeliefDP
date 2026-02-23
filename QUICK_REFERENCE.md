# Quick Reference

## Full Pipeline
```bash
python scripts/run_pipeline.py all --demo pick_and_place_kitkat
```

## Stages
```bash
python scripts/run_pipeline.py extract --demo pick_and_place_kitkat
python scripts/run_pipeline.py segment --demo pick_and_place_kitkat
python scripts/run_pipeline.py visualize --demo pick_and_place_kitkat
python scripts/run_pipeline.py train --demo pick_and_place_kitkat
```

## High-Accuracy Extraction (Recommended)
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

## Verify Both Cameras Are Used
```bash
python - << 'PY'
import json
m=json.load(open('demos/pick_and_place_kitkat/output/subgoal_metadata.json'))
d=next(iter(m.values()))
print('multi_view:', d['multi_view'])
print('cameras_extracted:', d['cameras_extracted'])
print('segmentation camera:', d['camera'])
print('detection mode:', d['detection_diagnostics']['mode'])
PY
```

## Key Outputs
- `demos/<task>/output/subgoal_metadata.json`
- `demos/<task>/output/phase_segmentation.json`
- `demos/<task>/output/subgoals/`
- `demos/<task>/output/phase_videos/`
- `demos/<task>/output/runs/structured/`

## Helpers
```bash
python scripts/inspect_hdf5.py --file demos/pick_and_place_kitkat/demo.hdf5
python scripts/standardize_hdf5.py --input demos/pick_and_place_kitkat/demo.hdf5 --profile beliefdp --dry-run
```
