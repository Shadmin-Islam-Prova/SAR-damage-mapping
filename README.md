# SAR Damage Mapping using Coherence Change Detection

This project shows a basic workflow for mapping disaster damage with Sentinel-1 SAR coherence.

## Method
1. Load pre-event and co-event coherence images.
2. Calculate the coherence change (pre minus post).
3. Find the damage threshold automatically with the Otsu method.
4. Compare the result with a reference map and calculate accuracy (overall accuracy, precision, recall, F1, IoU).

## Requirements
pip install -r requirements.txt

## How to run
python sar_damage_mapping.py

The script runs on simulated data by default. The output is saved in the `outputs/` folder.

## Using real data
Replace `simulate_data()` with `load_real_data(pre_path, post_path, reference_path)` in the `main()` function.