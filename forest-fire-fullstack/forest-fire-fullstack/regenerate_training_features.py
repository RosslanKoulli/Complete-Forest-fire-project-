#!/usr/bin/env python3
"""
Generate X_train_processed.npy for the existing trained models.

WHY THIS EXISTS
---------------
The domain confidence indicator computes Mahalanobis distance from
new inputs to the training-data centroid. For a stable estimate in
the 11-dimensional feature space, the calculator needs hundreds of
training samples - not the 50-sample SHAP background that was
previously used.

The training pipeline doesn't currently save the full processed
training set as a numpy array, so this helper does it after the fact:
it loads the original UCI and Algerian datasets, runs them through
the saved data pipeline (which applies the same standardisation,
region encoding, and cyclic month transformation as during training),
and saves the result as X_train_processed.npy alongside the trained
models.

USAGE
-----
Run this once from the project root of the ORIGINAL training project
(not the fullstack one):

    cd /path/to/forest-fire-prediction
    python regenerate_training_features.py

The output is written to ./trained_models/X_train_processed.npy.
Restart the fullstack backend after running this and the domain
confidence calculator will pick up the new file automatically.

WHAT IF THIS SCRIPT WON'T RUN
-----------------------------
If the original training data isn't accessible (different machine,
files moved, etc), the system gracefully falls back to the SHAP
background sample for the calculator. Predictions still work; the
confidence indicator is just less well-calibrated. This script is
a quality improvement, not a critical dependency.
"""
from pathlib import Path
import sys
import numpy as np
import joblib


def main():
    # Locate the trained_models directory relative to this script
    here = Path(__file__).resolve().parent
    trained_dir = here / 'trained_models'
    if not trained_dir.exists():
        print(f'ERROR: trained_models directory not found at {trained_dir}')
        print('Run this script from the project root that contains trained_models/')
        return 1

    pipeline_path = trained_dir / 'data_pipeline.joblib'
    if not pipeline_path.exists():
        print(f'ERROR: data_pipeline.joblib not found at {pipeline_path}')
        return 1

    print(f'Loading data pipeline from {pipeline_path}')
    sys.path.insert(0, str(here))
    pipeline = joblib.load(pipeline_path)

    # Try to find the source CSVs. Typical paths in the existing project:
    candidates = [
        here / 'data' / 'forestfires.csv',                  # UCI
        here / 'data' / 'Algerian_forest_fires_dataset.csv',  # Algerian
        here / 'data' / 'Algerian_forest_fires.csv',
        here / 'data' / 'combined_fires.csv',
        here / 'data' / 'forest_fires_combined.csv',
    ]

    print('Looking for training CSVs...')
    found = [c for c in candidates if c.exists()]
    if not found:
        print('ERROR: could not find training CSVs in data/')
        print('Looked for: ' + ', '.join(c.name for c in candidates))
        print('\nIf you have a combined dataset in a different location, edit the')
        print('"candidates" list at the top of this script and re-run.')
        return 1

    print(f'Found: {[f.name for f in found]}')

    # Use the pipeline's load + preprocess flow if it exposes one
    X_train = None
    if hasattr(pipeline, 'load_combined_data'):
        try:
            X, y = pipeline.load_combined_data()
            X_train = pipeline.transform(X) if hasattr(pipeline, 'transform') else X
            print(f'Loaded via pipeline.load_combined_data: {X_train.shape}')
        except Exception as e:
            print(f'pipeline.load_combined_data failed: {e}')

    # Otherwise: load each CSV, transform, and concatenate
    if X_train is None:
        import pandas as pd
        frames = []
        for csv_path in found:
            try:
                df = pd.read_csv(csv_path)
                print(f'  {csv_path.name}: {len(df)} rows')
                frames.append(df)
            except Exception as e:
                print(f'  {csv_path.name}: could not read ({e})')

        if not frames:
            print('ERROR: no CSVs could be read')
            return 1

        df_combined = pd.concat(frames, ignore_index=True, sort=False)
        print(f'Combined: {len(df_combined)} rows')

        # Run each row through the pipeline's transform_single_input
        if hasattr(pipeline, 'transform_single_input'):
            rows = []
            for _, row in df_combined.iterrows():
                try:
                    x = pipeline.transform_single_input(row.to_dict())
                    rows.append(x.flatten() if hasattr(x, 'flatten') else x)
                except Exception:
                    continue
            if not rows:
                print('ERROR: pipeline could not transform any rows')
                return 1
            X_train = np.array(rows)
        else:
            print('ERROR: pipeline has no transform method we can call')
            return 1

    # Save
    out_path = trained_dir / 'X_train_processed.npy'
    np.save(out_path, X_train)
    print(f'\nSaved {X_train.shape} -> {out_path}')
    print('Now restart the fullstack backend; domain confidence will use this set.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
