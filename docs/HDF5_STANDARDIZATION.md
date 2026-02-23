# HDF5 Standardization

## Goal
Ensure incoming datasets match the expected structure before extraction.

Expected location for pipeline use:
- `demos/<task>/demo.hdf5`

## 1. Inspect Existing File
```bash
python scripts/inspect_hdf5.py --file demos/<task>/demo.hdf5
```

## 2. Convert/Standardize
```bash
python scripts/standardize_hdf5.py \
  --input demos/<task>/demo.hdf5 \
  --output demos/<task>/demo_standardized.hdf5 \
  --profile beliefdp
```

## 3. Convert LeRobot Source
```bash
python scripts/convert_lerobot_to_hdf5.py <task_name>
```

## 4. Final Placement
After conversion, place final file as:
- `demos/<task>/demo.hdf5`

Then run pipeline:
```bash
python scripts/run_pipeline.py all --demo <task>
```
