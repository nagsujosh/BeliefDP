# Troubleshooting

## 1. Extraction failed with `No module named 'allenact'` or `gym`
Cause:
- vendored UVD imports optional simulation modules.

Status:
- extractor now installs runtime shims automatically.

Action:
```bash
python scripts/extract_subgoals.py --help
```
If help prints, import path is fixed.

## 2. Extraction failed with `No module named 'tree'`
Cause:
- missing `dm-tree` package.

Fix:
```bash
pip install dm-tree
```

## 3. Segmentation failed: missing `subgoal_metadata.json`
Cause:
- extraction did not finish successfully.

Fix:
```bash
python scripts/run_pipeline.py extract --demo <task>
ls demos/<task>/output/subgoal_metadata.json
```

## 4. Training failed: missing `phase_segmentation.json`
Cause:
- segmentation was skipped or failed.

Fix:
```bash
python scripts/run_pipeline.py segment --demo <task>
ls demos/<task>/output/phase_segmentation.json
```

## 5. CUDA initialization errors
If GPU init fails in your environment, run extraction on CPU:
```bash
python scripts/run_pipeline.py extract --demo <task> --device cpu
```

## 6. Ensure both camera views are being used
Use robust extract command:
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

Verify metadata:
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

## 7. Minimal recovery sequence after any failure
```bash
python scripts/run_pipeline.py extract --demo <task>
python scripts/run_pipeline.py segment --demo <task>
python scripts/run_pipeline.py visualize --demo <task>
python scripts/run_pipeline.py train --demo <task>
```
