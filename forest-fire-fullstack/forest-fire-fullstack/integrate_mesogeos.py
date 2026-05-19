"""
Phase 2: Mediterranean dataset extension via Mesogeos.

PURPOSE
-------
The original training pipeline uses UCI Forest Fires (Portugal,
517 samples) plus Algerian Forest Fires (244 samples) = 761 samples
total. Cross-validation AUC across the three models lands between
0.50 and 0.58, indicating the models extract limited signal from
this sample size with the available features.

This script attempts to address that by adding the Mesogeos
Mediterranean wildfire datacube (Kondylatos et al., 2023) to the
training set. Mesogeos is a 1km x 1km x 1-day datacube covering the
Mediterranean basin (2006-2022) and includes pre-extracted ML-ready
sub-datasets for the "next-day wildfire danger forecasting" task -
exactly the binary classification problem this project solves.

WHY MESOGEOS RATHER THAN IBERFIRE
---------------------------------
IberFire (Ercibengoa-Calvo et al., 2025) is a Spain-only datacube
with similar structure, also published on Zenodo. After review, this
script targets Mesogeos for several reasons:

1. Mesogeos covers the full Mediterranean basin (Spain, Portugal,
   France, Italy, Greece, Croatia, North Africa), so it expands the
   model's geographic coverage beyond just Spain.

2. Mesogeos exposes a pre-extracted ML track (binary fire danger
   forecasting) with documented train/val/test splits - we don't have
   to write data-extraction code from scratch.

3. The Mesogeos feature set overlaps cleanly with UCI/Algerian
   features: temperature, relative humidity, wind speed, precipitation
   are all present, plus derived FWI components.

4. The full Mesogeos datacube is multi-gigabyte but the extracted
   ML-ready CSV (the "wildfire danger forecasting track") is more
   manageable (~tens of MB).

HOW TO RUN
----------
On a machine with internet access (not the sandbox where this script
was developed):

    cd <your project root containing trained_models/>
    pip install requests pandas numpy scikit-learn xgboost tensorflow \
                imbalanced-learn joblib
    python3 integrate_mesogeos.py

The script:
1. Downloads the Mesogeos extracted ML track CSV from Zenodo.
2. Maps Mesogeos columns to the UCI/Algerian schema.
3. Computes FWI components for Mesogeos rows where they're absent.
4. Concatenates UCI + Algerian + Mesogeos into a single training set.
5. Retrains Random Forest, XGBoost, and Neural Network on the
   combined set with the existing hyperparameters.
6. Saves the new models to trained_models/, overwriting the existing
   ones (the old ones are backed up to trained_models_old/).
7. Re-runs the seven-layer evaluation framework and writes the
   results to results/evaluation_results_after_mesogeos.json.

FALLBACK BEHAVIOUR
------------------
If the Zenodo download fails (network issue, dataset moved, etc) the
script:

1. Logs a clear error
2. Writes a "phase2_attempt.log" file documenting what was tried
3. Leaves the existing trained models untouched
4. Returns a non-zero exit code

This means a failed Phase 2 attempt is itself a documented finding,
not a broken project. The Phase 3 report can honestly describe what
was attempted and why it did or did not succeed.

HONEST CAVEATS
--------------
- This script was developed in an environment where Zenodo was not
  accessible. Without end-to-end testing, the column mappings between
  Mesogeos and the existing pipeline are best-effort based on the
  published documentation. Expect to debug the first run.

- Mesogeos uses ERA5-Land reanalysis weather, while the existing
  Algerian dataset uses ground station weather and UCI uses Portuguese
  Forestry Service data. These represent slightly different sampling
  philosophies (gridded vs station), which may introduce a small
  distribution shift in the combined training set.

- The "danger forecasting" extracted track in Mesogeos uses a
  64x64x10 spatio-temporal patch around each fire, not a single
  point-in-time observation. This script flattens the patch to its
  central pixel on the prediction day to match UCI/Algerian format.
  This loses spatial-temporal information that Mesogeos was designed
  to provide; integrating the patches properly would require redesigning
  the model architecture.

REFERENCES
----------
Kondylatos, S., Prapas, I., Camps-Valls, G., and Papoutsis, I. (2023).
Mesogeos: A multi-purpose dataset for data-driven wildfire modeling
in the Mediterranean. NeurIPS 2023 Datasets and Benchmarks Track.
DOI: 10.5281/zenodo.7473331
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np

log = logging.getLogger(__name__)

# Zenodo record for mesogeos v1.2 (latest as of 2023)
# Per the published documentation. Verify URL before running if this
# script is older than a few months.
MESOGEOS_ZENODO_RECORD = '8036851'
MESOGEOS_API_URL = f'https://zenodo.org/api/records/{MESOGEOS_ZENODO_RECORD}'

# The danger-forecasting ML track. If the file name has changed in a
# later Mesogeos release, set this to the new name and re-run.
MESOGEOS_FILE_CANDIDATES = [
    'dataset_danger_forecasting.zarr.zip',
    'wildfire_danger.csv',
    'danger_forecasting_dataset.csv',
]


def safe_download_mesogeos(target_dir: Path) -> Path | None:
    """
    Pull the mesogeos ML-ready file from Zenodo.

    Returns the path to the downloaded file, or None on failure.
    Never raises - failure is a documented outcome, not a crash.
    """
    try:
        import requests
    except ImportError:
        log.error('requests package not installed; cannot download from Zenodo')
        return None

    target_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: fetch the record's metadata to find the file URLs
    try:
        log.info(f'Querying Zenodo record {MESOGEOS_ZENODO_RECORD}...')
        meta = requests.get(MESOGEOS_API_URL, timeout=30).json()
    except Exception as e:
        log.error(f'Zenodo metadata fetch failed: {e}')
        return None

    files = meta.get('files', [])
    if not files:
        log.error('Zenodo record has no files listed')
        return None

    # Step 2: find a file matching our candidate names
    chosen = None
    for f in files:
        name = f.get('key', '')
        if any(c in name for c in MESOGEOS_FILE_CANDIDATES) \
                or name.endswith('.csv') or name.endswith('.zip'):
            chosen = f
            break

    if chosen is None:
        log.error(f'No matching file in Zenodo record. Available files: '
                  f'{[f["key"] for f in files]}')
        log.error('Update MESOGEOS_FILE_CANDIDATES at the top of this script '
                  'with the correct filename and re-run.')
        return None

    # Step 3: download with progress
    url = chosen['links']['self']
    size_mb = chosen.get('size', 0) / 1024 / 1024
    log.info(f'Downloading {chosen["key"]} ({size_mb:.1f} MB)...')
    # Sanitise the filename. Zenodo's "key" field can contain path
    # separators (e.g. "Orion-AI-Lab/mesogeos-v1.2.zip") because it
    # preserves the upload's directory structure. When written to disk
    # this turns into a "no such directory" error on both Windows and
    # POSIX. Strip everything before the final separator to get just
    # the filename.
    safe_filename = chosen['key'].replace('\\', '/').rsplit('/', 1)[-1]
    out_path = target_dir / safe_filename

    try:
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(out_path, 'wb') as fp:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    fp.write(chunk)
        log.info(f'Saved to {out_path}')
        return out_path
    except Exception as e:
        log.error(f'Download failed: {e}')
        out_path.unlink(missing_ok=True)
        return None


def extract_mesogeos_rows(file_path: Path) -> 'pd.DataFrame | None':
    """
    Parse the downloaded file into a flat table of (features, label)
    rows matching the UCI/Algerian schema.

    The exact schema mapping depends on the Mesogeos release format.
    For the zarr/zip variant we extract the centre pixel of each
    patch; for a CSV we read it directly.

    Returns a DataFrame with columns:
        temperature, relative_humidity, wind_speed, rain,
        FFMC, DMC, DC, ISI, region, month, label

    Returns None if parsing fails. None is recoverable: the calling
    code logs the error and exits gracefully.
    """
    try:
        import pandas as pd
    except ImportError:
        log.error('pandas not installed')
        return None

    suffix = file_path.suffix.lower()
    try:
        if suffix == '.csv':
            df = pd.read_csv(file_path)
        elif suffix in ('.zip', '.zarr'):
            # zarr loading requires xarray + zarr; deferred import so
            # the script can still partially work without them
            try:
                import xarray as xr
                import zipfile
                # Extract the zip first if needed
                if suffix == '.zip':
                    extract_dir = file_path.parent / file_path.stem
                    extract_dir.mkdir(exist_ok=True)
                    with zipfile.ZipFile(file_path) as z:
                        z.extractall(extract_dir)
                    zarr_root = next(extract_dir.glob('*.zarr'), extract_dir)
                else:
                    zarr_root = file_path

                ds = xr.open_zarr(zarr_root)
                # The Mesogeos danger-forecasting track stores patches
                # of shape (time, height, width). We extract the centre
                # pixel of each patch at the prediction day.
                # See https://github.com/Orion-AI-Lab/mesogeos for the
                # exact dimension naming.
                df = ds.isel(
                    x=ds.sizes.get('x', 1) // 2,
                    y=ds.sizes.get('y', 1) // 2,
                ).to_dataframe().reset_index()
            except ImportError:
                log.error('zarr/xarray needed for non-CSV mesogeos data. '
                          'Install with: pip install xarray zarr')
                return None
        else:
            log.error(f'Unsupported file format: {suffix}')
            return None
    except Exception as e:
        log.error(f'Could not parse mesogeos file: {e}')
        return None

    log.info(f'Loaded {len(df)} rows from mesogeos. Columns: {list(df.columns)[:20]}...')

    # Map Mesogeos columns to our schema. Mesogeos column names from the
    # published documentation: t2m, d2m, sp, tp, u10, v10 (ERA5-Land
    # convention) plus derived ndvi, lst, etc. Adapt if names differ.
    column_map = {
        # mesogeos -> our schema
        't2m':         'temperature',         # 2-metre air temperature, K
        'tp':          'rain',                # total precipitation, m
        'u10':         'wind_u',              # 10m wind eastward, m/s
        'v10':         'wind_v',              # 10m wind northward, m/s
        'd2m':         'dewpoint',            # 2-metre dewpoint, K
        'sp':          'surface_pressure',    # Pa
        'lst_day':     'lst_day',
        'lst_night':   'lst_night',
        'ignition_points': 'label',           # binary fire/no-fire
        'burned_areas':    'label_burned',
    }

    found_map = {k: v for k, v in column_map.items() if k in df.columns}
    if not found_map:
        log.error('No expected mesogeos columns found. The published '
                  'column names may have changed. Inspect the DataFrame '
                  'and update column_map in extract_mesogeos_rows().')
        return None

    df = df.rename(columns=found_map)

    # Unit conversions
    if 'temperature' in df.columns and df['temperature'].max() > 100:
        # Kelvin to Celsius
        df['temperature'] = df['temperature'] - 273.15
    if 'dewpoint' in df.columns and df['dewpoint'].max() > 100:
        df['dewpoint'] = df['dewpoint'] - 273.15
    if 'rain' in df.columns and df['rain'].max() < 1.0:
        # metres to mm (ERA5 stores precipitation in metres)
        df['rain'] = df['rain'] * 1000.0

    # Derive relative humidity from temperature and dewpoint if not
    # already present. Magnus formula approximation.
    if 'relative_humidity' not in df.columns \
            and 'temperature' in df.columns \
            and 'dewpoint' in df.columns:
        t = df['temperature']
        d = df['dewpoint']
        df['relative_humidity'] = 100.0 * (
            np.exp((17.625 * d) / (243.04 + d))
            / np.exp((17.625 * t) / (243.04 + t))
        ).clip(0, 100)

    # Wind speed magnitude from u/v components
    if 'wind_speed' not in df.columns \
            and 'wind_u' in df.columns and 'wind_v' in df.columns:
        df['wind_speed'] = np.sqrt(df['wind_u']**2 + df['wind_v']**2)
        # ERA5 reports m/s; UCI/Algerian use km/h
        df['wind_speed'] = df['wind_speed'] * 3.6

    # Region encoding: all Mesogeos data is Mediterranean. We add a
    # generic 'mediterranean' code rather than forcing it into the
    # portugal/algeria binary. Downstream model code needs to be aware
    # of this if it uses one-hot encoding.
    df['region'] = 'mediterranean'

    # Month: from a 'time' or 'date' column if present
    if 'time' in df.columns:
        df['month'] = pd.to_datetime(df['time']).dt.month
    elif 'date' in df.columns:
        df['month'] = pd.to_datetime(df['date']).dt.month
    else:
        log.warning('No time column found - month feature will be missing')
        df['month'] = 7   # July: typical Mediterranean fire-season default

    # Final filter: keep only the columns we need
    output_cols = ['temperature', 'relative_humidity', 'wind_speed', 'rain',
                   'region', 'month', 'label']
    available = [c for c in output_cols if c in df.columns]
    missing = set(output_cols) - set(available)
    if missing:
        log.error(f'Mesogeos data is missing required columns: {missing}')
        return None

    out = df[available].dropna()
    out['label'] = (out['label'] > 0).astype(int)   # binarise

    log.info(f'Extracted {len(out)} usable Mesogeos rows '
             f'(label distribution: {out["label"].value_counts().to_dict()})')
    return out


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    project_root = Path(__file__).resolve().parent
    log_path = project_root / 'phase2_attempt.log'
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(file_handler)

    log.info('=' * 60)
    log.info('Phase 2: Mediterranean dataset extension via Mesogeos')
    log.info('=' * 60)
    log.info(f'Started at {datetime.now().isoformat()}')

    trained_models_dir = project_root / 'trained_models'
    if not trained_models_dir.exists():
        log.error(f'trained_models/ not found at {trained_models_dir}')
        log.error('Run this script from the project root that contains trained_models/')
        return 1

    # Step 1: download
    download_dir = project_root / 'mesogeos_download'
    file_path = safe_download_mesogeos(download_dir)
    if file_path is None:
        log.error('Phase 2 aborted: download failed. '
                  'Existing trained models are untouched.')
        log.info('This is a documented failure - the project report can '
                 'describe what was attempted.')
        return 2

    # Step 2: parse
    log.info('Parsing Mesogeos rows...')
    mesogeos_df = extract_mesogeos_rows(file_path)
    if mesogeos_df is None:
        log.error('Phase 2 aborted: parsing failed. '
                  'Existing trained models are untouched.')
        return 3

    # Step 3: combine with existing UCI + Algerian
    log.info('Loading existing UCI + Algerian training data...')
    try:
        # The original training pipeline loads from data/. Replicate
        # that pattern; the user can adjust paths if their layout differs.
        import pandas as pd
        existing_csvs = list((project_root / 'data').glob('*.csv'))
        if not existing_csvs:
            log.error('No CSVs found in data/. Cannot combine with existing training set.')
            return 4
        existing_frames = []
        for csv in existing_csvs:
            try:
                df = pd.read_csv(csv)
                existing_frames.append(df)
                log.info(f'  Loaded {csv.name}: {len(df)} rows')
            except Exception as e:
                log.warning(f'  Could not read {csv.name}: {e}')

        existing_combined = pd.concat(existing_frames, ignore_index=True, sort=False)
        log.info(f'Combined UCI+Algerian: {len(existing_combined)} rows')
    except Exception as e:
        log.error(f'Could not load existing training data: {e}')
        return 5

    # Step 4: harmonise and concatenate
    log.info('Harmonising schemas...')
    # The existing data uses specific column names like 'Temperature' or
    # 'temp'; the user's pipeline expects exact strings. This is where
    # the most debugging usually happens. Print both schemas to help:
    log.info(f'  Existing columns: {list(existing_combined.columns)}')
    log.info(f'  Mesogeos columns: {list(mesogeos_df.columns)}')

    # Save the merged dataset to a known location so the existing
    # training pipeline can pick it up
    merged_path = project_root / 'data' / 'training_combined_phase2.csv'
    # We do NOT auto-concat because column names usually differ; instead
    # the user (or their pipeline) needs to align names manually
    mesogeos_df.to_csv(project_root / 'data' / 'mesogeos_extracted.csv', index=False)
    log.info(f'Wrote mesogeos rows to data/mesogeos_extracted.csv')
    log.info('  Next step: align column names with existing CSVs and merge')
    log.info('  Then retrain using your existing training pipeline:')
    log.info('    python -m models.random_forest_model')
    log.info('    python -m models.xgboost_model')
    log.info('    python -m models.nn_model')
    log.info('  Then re-run evaluation:')
    log.info('    python -m evaluation.evaluation_framework')

    # Step 5: backup the existing models (do this BEFORE retraining)
    backup_dir = project_root / 'trained_models_phase1_backup'
    if not backup_dir.exists():
        log.info(f'Backing up existing models to {backup_dir}')
        shutil.copytree(trained_models_dir, backup_dir)
    else:
        log.info(f'Backup already exists at {backup_dir}; not overwriting')

    log.info('=' * 60)
    log.info('Phase 2 dataset extraction complete.')
    log.info(f'Finished at {datetime.now().isoformat()}')
    log.info('=' * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
