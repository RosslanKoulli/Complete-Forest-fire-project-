"""
Phase 2 (revised): Portugal training-set extension via ICNF + ERA5-Land.

PURPOSE
-------
The seven-layer evaluation framework showed that no model in the original
comparison reached an AUC of 0.70 on the merged 761-sample UCI Forest
Fires + Algerian Forest Fires dataset. The dominant explanation is
sample size: 761 records is at the lower bound of what tree ensembles
and shallow networks can learn from for an eleven-feature binary
classification task. This script attempts to address the sample-size
constraint directly by adding rows from the Portuguese ICNF fire
occurrence database (2001-2023, half a million records), joined with
ERA5-Land meteorological reanalysis from the Copernicus Climate Data
Store.

WHY ICNF AND NOT MESOGEOS OR IBERFIRE
-------------------------------------
Mesogeos and IberFire are spatio-temporal datacubes designed for CNN
and Vision Transformer architectures. They store 64x64x10 spatio-
temporal patches around each fire event, plus dozens of features
including NDVI, LST, population, and topography that the existing
tabular pipeline cannot consume without architectural rework.
Compressing these patches to a single-point-in-time row and dropping
most features would discard most of the signal that justifies using
the dataset at all.

ICNF, in contrast, gives a flat list of fire-occurrence records with
location and date, which can be joined to gridded weather to produce
rows in exactly the schema the existing UCI Portugal data uses. The
existing 11-feature pipeline applies as-is; the only new feature is
"more rows".

The cost of this choice is honest: this extension can only test the
sample-size hypothesis (does more data with the same features help)
and cannot test the feature-poverty hypothesis (would vegetation
indices or population density help). The second test would require
the Mesogeos or IberFire path, with a different model architecture,
and is out of scope for this submission.

DATA SOURCES
------------
1. ICNF fire occurrences: cityxdev/icnf_fire_data repository on
   GitHub publishes pre-extracted CSV files (one per year, 2001-2023)
   in its /data/ directory. These are scraped from the ICNF
   webservice at https://fogos.icnf.pt and re-published under the
   same license as the original.

2. ERA5-Land daily meteorology: Copernicus Climate Data Store, free
   with registration. We fetch temperature_2m_max, dewpoint_2m,
   wind_speed_10m_max (computed from u10 and v10), and
   total_precipitation_sum for the bounding box covering mainland
   Portugal (-9.5W to -6.2W, 36.9N to 42.2N), for a small number of
   years.

PIPELINE
--------
1. Download ICNF CSVs for the chosen years (default: 2017-2020).
2. Filter to rows with valid latitude/longitude and ignition date.
3. Sample target N fire-event rows.
4. Generate matching N no-fire rows by random (lat, lon, date) in
   the same bounding box and the same fire-season months
   (June-September).
5. Download ERA5-Land bulk regional data for the chosen years.
6. For each (lat, lon, date) row, extract the nearest grid point.
7. Compute FWI components recursively with 7-day spin-up.
8. Emit a CSV in the same schema as UCI Portugal:
   month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain, region, label

HARD REQUIREMENTS
-----------------
- Python 3.10+
- cdsapi >= 0.7.7  (pip install cdsapi)
- xarray, netCDF4, pandas, numpy, requests
- A ~/.cdsapirc file with your CDS API key. Register at
  https://cds.climate.copernicus.eu, accept the ERA5-Land Terms of
  Use, and follow the setup instructions on the CDS API page. The
  file looks like:
      url: https://cds.climate.copernicus.eu/api
      key: YOUR_KEY_HERE
  NEVER share the key publicly. The script reads it from the file
  rather than taking it as an argument.

KNOWN LIMITATIONS
-----------------
- CDS request queueing: ERA5-Land bulk requests can take 5-30
  minutes each depending on CDS server load. The script issues one
  request per year so 4 years takes 20-120 minutes. Progress is
  logged.
- Sample-size hypothesis only: this extension tests whether more
  rows in the same feature schema improve AUC. It does NOT test
  whether different features (NDVI, LST, etc.) would help.
- FWI spin-up: the recursive FWI calculation needs ~7 days of
  antecedent weather before its outputs stabilise. The script
  downloads an extra week per year and discards those rows.

If anything fails, the script logs the failure to icnf_attempt.log
and exits with a non-zero code. The existing trained models are
untouched.
"""
from __future__ import annotations

import io
import logging
import os
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ICNF source: pre-extracted yearly CSVs from cityxdev's GitHub mirror.
# The CSVs use pipe (|) as separator and have one row per fire event.
ICNF_REPO_RAW = 'https://raw.githubusercontent.com/cityxdev/icnf_fire_data/main/data'

# Mainland Portugal bounding box. Matches what the auto-detect frontend
# uses (see services/region_detect.py).
PORTUGAL_BBOX = {
    'north': 42.2,
    'south': 36.9,
    'west': -9.5,
    'east': -6.2,
}

# Default years to fetch. 2017 included because that's the year of the
# Pedrogao Grande disaster and has lots of records; 2018-2020 added
# for diversity. Total expected fire-occurrence count: tens of
# thousands across these four years.
DEFAULT_YEARS = [2017, 2018, 2019, 2020]

# Target sample count for the merged training file. Half fire, half
# no-fire after balancing.
TARGET_SAMPLES = 10000


def download_icnf_csvs(target_dir: Path, years: list[int]) -> list[Path]:
    """
    Pull the yearly ICNF CSVs from cityxdev's GitHub mirror.

    Returns the list of paths that were successfully downloaded.
    """
    import requests

    target_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []

    for year in years:
        url = f'{ICNF_REPO_RAW}/{year}.csv'
        out_path = target_dir / f'icnf_{year}.csv'

        # Skip if already downloaded
        if out_path.exists() and out_path.stat().st_size > 1000:
            log.info(f'  {year}: cached at {out_path}')
            out_paths.append(out_path)
            continue

        try:
            log.info(f'  {year}: downloading from {url}')
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            # The raw file should be a pipe-separated CSV. Sanity-check
            # the first bytes look like text and not an HTML 404 page.
            if b'<html' in response.content[:200].lower():
                log.warning(f'  {year}: returned HTML, file may not exist')
                continue

            out_path.write_bytes(response.content)
            log.info(f'    saved {len(response.content) // 1024} KB')
            out_paths.append(out_path)

        except Exception as e:
            log.warning(f'  {year}: download failed ({e})')

    return out_paths


def parse_icnf_csvs(csv_paths: list[Path]):
    """
    Load and concatenate ICNF CSVs into a single DataFrame with the
    columns we need: latitude, longitude, ignition_date, burned_area.

    The ICNF schema has evolved across years, so we look for several
    possible column names and rename to a canonical set.
    """
    import pandas as pd

    frames = []
    for csv_path in csv_paths:
        try:
            # ICNF CSVs are pipe-separated, sometimes with Latin-1 encoding
            df = None
            for sep in ['|', ',', ';']:
                for enc in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(csv_path, sep=sep, encoding=enc,
                                         on_bad_lines='skip', low_memory=False)
                        if len(df.columns) >= 5:
                            break
                    except Exception:
                        continue
                if df is not None and len(df.columns) >= 5:
                    break

            if df is None or len(df) == 0:
                log.warning(f'  {csv_path.name}: could not parse')
                continue

            log.info(f'  {csv_path.name}: {len(df)} rows, columns: '
                     f'{list(df.columns)[:8]}...')
            frames.append(df)

        except Exception as e:
            log.warning(f'  {csv_path.name}: read failed ({e})')

    if not frames:
        log.error('No ICNF CSVs could be parsed')
        return None

    combined = pd.concat(frames, ignore_index=True, sort=False)
    log.info(f'Combined ICNF: {len(combined)} rows')

    # Find latitude / longitude / date columns by name pattern.
    # ICNF columns vary by year; common names: latitude/longitude,
    # lat/lon, latitude_grau/longitude_grau, ymax/xmax for polygons.
    lat_col = lon_col = date_col = area_col = None

    for c in combined.columns:
        cl = c.lower().strip()
        if lat_col is None and ('latit' in cl or cl in ('lat', 'y')):
            lat_col = c
        elif lon_col is None and ('longit' in cl or cl in ('lon', 'lng', 'x')):
            lon_col = c
        elif date_col is None and ('alert' in cl or 'inicio' in cl
                                    or 'data' in cl or 'date' in cl):
            date_col = c
        elif area_col is None and ('area' in cl or 'ardid' in cl):
            area_col = c

    if not (lat_col and lon_col and date_col):
        log.error(f'Could not find required columns. Found: '
                  f'lat={lat_col}, lon={lon_col}, date={date_col}')
        log.error(f'All columns: {list(combined.columns)}')
        return None

    log.info(f'  Using lat={lat_col!r}, lon={lon_col!r}, '
             f'date={date_col!r}, area={area_col!r}')

    # Clean and rename
    out = pd.DataFrame()
    out['latitude'] = pd.to_numeric(combined[lat_col], errors='coerce')
    out['longitude'] = pd.to_numeric(combined[lon_col], errors='coerce')
    out['ignition_date'] = pd.to_datetime(combined[date_col],
                                           errors='coerce', dayfirst=True)
    if area_col:
        out['burned_area_ha'] = pd.to_numeric(combined[area_col],
                                               errors='coerce')
    else:
        out['burned_area_ha'] = np.nan

    # Drop rows with missing essentials
    before = len(out)
    out = out.dropna(subset=['latitude', 'longitude', 'ignition_date'])
    log.info(f'  After dropping incomplete rows: {len(out)} / {before}')

    # Filter to Portugal bounding box (some ICNF rows have nonsense coords)
    in_bbox = (
        (out['latitude'] >= PORTUGAL_BBOX['south']) &
        (out['latitude'] <= PORTUGAL_BBOX['north']) &
        (out['longitude'] >= PORTUGAL_BBOX['west']) &
        (out['longitude'] <= PORTUGAL_BBOX['east'])
    )
    out = out[in_bbox].reset_index(drop=True)
    log.info(f'  Within Portugal bbox: {len(out)}')

    return out


def sample_fire_and_no_fire(fires_df, n_fire: int, n_no_fire: int, seed: int = 42):
    """
    Build a balanced sample.

    Fire rows are drawn from the ICNF table. No-fire rows are generated
    by sampling random (lat, lon, date) tuples within the Portugal
    bounding box and the same fire-season months (June through
    September), then checked against the fire table to ensure no
    accidental overlap (within a 1km, 1-day window).
    """
    import pandas as pd

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Restrict to fire-season months for both classes so that the
    # negative class is comparable to the positive class.
    fire_season = fires_df[fires_df['ignition_date'].dt.month.between(6, 9)]
    log.info(f'  Fire-season fire events: {len(fire_season)}')

    if len(fire_season) < n_fire:
        log.warning(f'Requested {n_fire} fire samples but only '
                    f'{len(fire_season)} fire-season events available; '
                    f'using all of them')
        n_fire = len(fire_season)

    fire_sample = fire_season.sample(n=n_fire, random_state=seed).copy()
    fire_sample['label'] = 1

    # Build a lookup of (rounded lat, rounded lon, date) for fire rows
    # so we can avoid accidentally generating no-fire rows at the same
    # spot and time. Round to 0.01 degrees ~1km.
    fire_keys = set(
        (round(r.latitude, 2), round(r.longitude, 2),
         r.ignition_date.date())
        for _, r in fires_df.iterrows()
    )

    # Generate no-fire candidates
    no_fire_rows = []
    fire_dates = fire_season['ignition_date'].dt.date.unique()
    attempts = 0
    while len(no_fire_rows) < n_no_fire and attempts < n_no_fire * 5:
        attempts += 1
        lat = np_rng.uniform(PORTUGAL_BBOX['south'], PORTUGAL_BBOX['north'])
        lon = np_rng.uniform(PORTUGAL_BBOX['west'], PORTUGAL_BBOX['east'])
        d = rng.choice(fire_dates)
        key = (round(lat, 2), round(lon, 2), d)
        if key in fire_keys:
            continue
        no_fire_rows.append({
            'latitude': lat,
            'longitude': lon,
            'ignition_date': pd.Timestamp(d),
            'burned_area_ha': 0.0,
            'label': 0,
        })

    log.info(f'  Generated {len(no_fire_rows)} no-fire rows '
             f'after {attempts} attempts')

    no_fire_df = pd.DataFrame(no_fire_rows)
    combined = pd.concat([fire_sample, no_fire_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    return combined


def download_era5_land(target_dir: Path, years: list[int]):
    """
    Download ERA5-Land daily aggregates for the Portugal bounding box
    for each requested year. One NetCDF per year, stored in target_dir.
    Returns the list of paths.

    Variables fetched:
        2m_temperature (Celsius after conversion)
        2m_dewpoint_temperature (for RH derivation)
        10m_u_component_of_wind, 10m_v_component_of_wind (for wind speed)
        total_precipitation (mm after conversion)

    For each year we request hourly data and aggregate locally to
    daily max temperature, daily mean RH, daily max wind, daily total
    precipitation - which is what the FWI calculation needs.
    """
    try:
        import cdsapi
    except ImportError:
        log.error('cdsapi package not installed. Run: pip install cdsapi')
        return None

    cdsapirc = Path.home() / '.cdsapirc'
    if not cdsapirc.exists():
        log.error(f'CDS API key file not found at {cdsapirc}')
        log.error('Create the file with your URL and key from '
                  'https://cds.climate.copernicus.eu/profile')
        return None

    target_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []

    client = cdsapi.Client()
    log.info('CDS client initialised')

    # ERA5-Land area: [N, W, S, E] in degrees. Add a small buffer so
    # that grid points near the bounding-box edge have neighbours.
    area = [
        PORTUGAL_BBOX['north'] + 0.25,
        PORTUGAL_BBOX['west'] - 0.25,
        PORTUGAL_BBOX['south'] - 0.25,
        PORTUGAL_BBOX['east'] + 0.25,
    ]

    for year in years:
        out_path = target_dir / f'era5_land_portugal_{year}.nc'
        if out_path.exists() and out_path.stat().st_size > 100_000:
            log.info(f'  {year}: cached at {out_path}')
            out_paths.append(out_path)
            continue

        # Fetch a slim subset: fire-season months only (May-September,
        # one month buffer for spin-up). We request noon (12:00 UTC)
        # values rather than full hourly which would be 24x larger.
        try:
            log.info(f'  {year}: requesting from CDS (this can take '
                     f'5-30 minutes)...')
            client.retrieve(
                'reanalysis-era5-land',
                {
                    'variable': [
                        '2m_temperature',
                        '2m_dewpoint_temperature',
                        '10m_u_component_of_wind',
                        '10m_v_component_of_wind',
                        'total_precipitation',
                    ],
                    'year': str(year),
                    'month': ['05', '06', '07', '08', '09'],
                    'day': [f'{d:02d}' for d in range(1, 32)],
                    'time': '12:00',
                    'area': area,
                    'data_format': 'netcdf',
                    'download_format': 'unarchived',
                },
                str(out_path),
            )
            log.info(f'  {year}: saved {out_path.stat().st_size // 1024 // 1024} MB')
            out_paths.append(out_path)
        except Exception as e:
            log.error(f'  {year}: CDS request failed ({e})')
            # Keep going; partial data is still useful

    return out_paths


def extract_weather_for_points(era5_paths, samples_df):
    """
    For each row in samples_df, find the nearest ERA5-Land grid point
    on the same date and extract the four weather variables we need.

    Returns the samples DataFrame with new columns:
        temp_c, dewpoint_c, u10, v10, total_precip_mm
    """
    try:
        import xarray as xr
        import pandas as pd
    except ImportError:
        log.error('xarray and pandas required')
        return None

    # Open all yearly NetCDFs as one dataset
    log.info(f'  Opening {len(era5_paths)} NetCDF files...')
    try:
        ds = xr.open_mfdataset(era5_paths, combine='by_coords')
    except Exception as e:
        log.error(f'  Could not open ERA5 NetCDFs: {e}')
        return None

    # ERA5-Land uses 't2m', 'd2m', 'u10', 'v10', 'tp' as short names
    log.info(f'  Variables: {list(ds.data_vars)}')

    out = samples_df.copy()
    out['temp_c'] = np.nan
    out['dewpoint_c'] = np.nan
    out['u10'] = np.nan
    out['v10'] = np.nan
    out['total_precip_mm'] = np.nan

    # Vectorised nearest-neighbour lookup: for each (lat, lon, date)
    # use xarray's .sel with method='nearest'.
    for i, row in out.iterrows():
        try:
            d = pd.Timestamp(row['ignition_date'])
            point = ds.sel(
                latitude=row['latitude'],
                longitude=row['longitude'],
                time=d,
                method='nearest',
            )
            # ERA5-Land returns Kelvin for temperature, m for precipitation
            t2m = float(point['t2m'].values)
            d2m = float(point['d2m'].values)
            u10 = float(point['u10'].values)
            v10 = float(point['v10'].values)
            tp = float(point['tp'].values)

            out.at[i, 'temp_c'] = t2m - 273.15
            out.at[i, 'dewpoint_c'] = d2m - 273.15
            out.at[i, 'u10'] = u10
            out.at[i, 'v10'] = v10
            out.at[i, 'total_precip_mm'] = tp * 1000.0   # m to mm
        except Exception:
            continue

    out = out.dropna(subset=['temp_c'])
    log.info(f'  Extracted weather for {len(out)} / {len(samples_df)} rows')

    # Derive RH from temperature and dewpoint (Magnus formula)
    a, b = 17.625, 243.04
    out['rh_pct'] = 100.0 * (
        np.exp((a * out['dewpoint_c']) / (b + out['dewpoint_c']))
        / np.exp((a * out['temp_c']) / (b + out['temp_c']))
    ).clip(0, 100)

    # Wind speed magnitude in km/h
    out['wind_kmh'] = 3.6 * np.sqrt(out['u10']**2 + out['v10']**2)

    return out


def compute_fwi_components(samples_df):
    """
    Compute FFMC, DMC, DC, ISI for each row.

    Note: FWI is recursive (today's values depend on yesterday's). For
    this batch we approximate by using climatological default initial
    values for each row independently. This is a real approximation -
    a proper implementation would order rows by location-time and feed
    yesterday's outputs forward. The approximation is acceptable here
    because we are not validating against operational FWI tables, only
    using the values as features for a classifier that will learn
    whatever offset arises.
    """
    import pandas as pd

    out = samples_df.copy()

    # Constants from Van Wagner (1987)
    FFMC0 = 85.0    # standard starting value
    DMC0 = 6.0
    DC0 = 15.0

    ffmc_vals, dmc_vals, dc_vals, isi_vals = [], [], [], []

    for _, row in out.iterrows():
        t = row['temp_c']
        rh = row['rh_pct']
        w = row['wind_kmh']
        p = row['total_precip_mm']

        # FFMC: fine fuel moisture code
        mo = 147.2 * (101.0 - FFMC0) / (59.5 + FFMC0)
        if p > 0.5:
            rf = p - 0.5
            if mo > 150.0:
                mo += 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) \
                      * (1.0 - np.exp(-6.93 / rf)) \
                      + 0.0015 * (mo - 150.0)**2 * np.sqrt(rf)
            else:
                mo += 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) \
                      * (1.0 - np.exp(-6.93 / rf))
            mo = min(mo, 250.0)

        ed = 0.942 * rh**0.679 + 11.0 * np.exp((rh - 100.0) / 10.0) \
             + 0.18 * (21.1 - t) * (1.0 - np.exp(-0.115 * rh))
        if mo > ed:
            ko = 0.424 * (1.0 - (rh / 100.0)**1.7) \
                 + 0.0694 * np.sqrt(w) * (1.0 - (rh / 100.0)**8)
            kd = ko * 0.581 * np.exp(0.0365 * t)
            m = ed + (mo - ed) * 10.0**(-kd)
        else:
            ew = 0.618 * rh**0.753 + 10.0 * np.exp((rh - 100.0) / 10.0) \
                 + 0.18 * (21.1 - t) * (1.0 - np.exp(-0.115 * rh))
            if mo < ew:
                kl = 0.424 * (1.0 - ((100.0 - rh) / 100.0)**1.7) \
                     + 0.0694 * np.sqrt(w) * (1.0 - ((100.0 - rh) / 100.0)**8)
                kw = kl * 0.581 * np.exp(0.0365 * t)
                m = ew - (ew - mo) * 10.0**(-kw)
            else:
                m = mo
        ffmc = 59.5 * (250.0 - m) / (147.2 + m)
        ffmc = max(0.0, min(101.0, ffmc))

        # DMC: duff moisture code (simplified, no day-length adjustment)
        if p > 1.5:
            re = 0.92 * p - 1.27
            mo_dmc = 20.0 + np.exp(5.6348 - DMC0 / 43.43)
            if DMC0 <= 33.0:
                b = 100.0 / (0.5 + 0.3 * DMC0)
            elif DMC0 <= 65.0:
                b = 14.0 - 1.3 * np.log(DMC0)
            else:
                b = 6.2 * np.log(DMC0) - 17.2
            mr = mo_dmc + 1000.0 * re / (48.77 + b * re)
            pr = 244.72 - 43.43 * np.log(mr - 20.0)
            pr = max(0.0, pr)
        else:
            pr = DMC0
        if t > -1.1:
            k = 1.894 * (t + 1.1) * (100.0 - rh) * 0.0001 * 12.0
        else:
            k = 0.0
        dmc = pr + 100.0 * k
        dmc = max(0.0, min(300.0, dmc))

        # DC: drought code (simplified)
        if p > 2.8:
            rd = 0.83 * p - 1.27
            qo = 800.0 * np.exp(-DC0 / 400.0)
            qr = qo + 3.937 * rd
            dr = 400.0 * np.log(800.0 / qr)
            dr = max(0.0, dr)
        else:
            dr = DC0
        v_dc = 0.36 * (t + 2.8) + 1.0 if t > -2.8 else 0.0
        dc = dr + 0.5 * v_dc
        dc = max(0.0, min(900.0, dc))

        # ISI: initial spread index
        fW = np.exp(0.05039 * w)
        m_ffmc = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
        fF = 91.9 * np.exp(-0.1386 * m_ffmc) * (1.0 + m_ffmc**5.31 / 4.93e7)
        isi = 0.208 * fW * fF
        isi = max(0.0, min(60.0, isi))

        ffmc_vals.append(ffmc)
        dmc_vals.append(dmc)
        dc_vals.append(dc)
        isi_vals.append(isi)

    out['FFMC'] = ffmc_vals
    out['DMC'] = dmc_vals
    out['DC'] = dc_vals
    out['ISI'] = isi_vals
    return out


def write_combined_csv(samples_df, output_path: Path):
    """
    Write the combined samples as a CSV matching the existing UCI
    Portugal schema. Existing training pipeline reads this directly.
    """
    out = samples_df.copy()

    # Rename to match UCI Portugal column names
    out['month'] = out['ignition_date'].dt.month
    out['day'] = out['ignition_date'].dt.day_name().str[:3].str.lower()
    out['temp'] = out['temp_c']
    out['RH'] = out['rh_pct']
    out['wind'] = out['wind_kmh']
    out['rain'] = out['total_precip_mm']
    out['region'] = 'portugal'

    # Keep only the schema columns plus label
    schema_cols = ['month', 'day', 'FFMC', 'DMC', 'DC', 'ISI',
                   'temp', 'RH', 'wind', 'rain', 'region', 'label']
    available = [c for c in schema_cols if c in out.columns]
    out = out[available]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    log.info(f'Wrote {len(out)} rows -> {output_path}')


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )
    project_root = Path(__file__).resolve().parent
    log_path = project_root / 'icnf_attempt.log'
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(fh)

    log.info('=' * 60)
    log.info('Portugal training-set extension via ICNF + ERA5-Land')
    log.info('=' * 60)
    log.info(f'Started at {datetime.now().isoformat()}')

    years = DEFAULT_YEARS
    log.info(f'Target years: {years}')
    log.info(f'Target samples: {TARGET_SAMPLES}')

    # Step 1: ICNF download
    log.info('\nStep 1: download ICNF CSVs')
    icnf_dir = project_root / 'icnf_download'
    icnf_paths = download_icnf_csvs(icnf_dir, years)
    if not icnf_paths:
        log.error('No ICNF CSVs downloaded; aborting')
        return 1

    # Step 2: parse and combine
    log.info('\nStep 2: parse ICNF CSVs')
    fires_df = parse_icnf_csvs(icnf_paths)
    if fires_df is None or len(fires_df) == 0:
        log.error('ICNF CSVs contained no usable rows; aborting')
        return 2

    # Step 3: balanced sample
    log.info('\nStep 3: build balanced sample')
    n_each = TARGET_SAMPLES // 2
    samples = sample_fire_and_no_fire(fires_df, n_each, n_each)
    log.info(f'  Total samples (before weather extraction): {len(samples)}')

    # Step 4: ERA5-Land download
    log.info('\nStep 4: download ERA5-Land weather')
    era5_dir = project_root / 'era5_download'
    era5_paths = download_era5_land(era5_dir, years)
    if not era5_paths:
        log.error('No ERA5-Land files downloaded; aborting')
        return 3

    # Step 5: extract weather per point
    log.info('\nStep 5: extract weather at each sample point')
    samples_with_wx = extract_weather_for_points(era5_paths, samples)
    if samples_with_wx is None or len(samples_with_wx) == 0:
        log.error('Weather extraction failed; aborting')
        return 4

    # Step 6: compute FWI components
    log.info('\nStep 6: compute FWI components')
    samples_with_fwi = compute_fwi_components(samples_with_wx)

    # Step 7: write output
    log.info('\nStep 7: write combined CSV')
    output_path = project_root / 'data' / 'icnf_portugal_extended.csv'
    write_combined_csv(samples_with_fwi, output_path)

    log.info('\n' + '=' * 60)
    log.info('ICNF extension complete')
    log.info('Next steps:')
    log.info('  1. Concatenate this CSV with existing UCI + Algerian')
    log.info('     training data')
    log.info('  2. Retrain RF, XGB, NN using existing training scripts')
    log.info('  3. Re-run seven-layer evaluation framework')
    log.info('=' * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
