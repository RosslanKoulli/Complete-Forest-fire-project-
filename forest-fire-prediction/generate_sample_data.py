"""
Generate Sample Datasets
=========================
Creates synthetic datasets matching the exact column structure
and realistic distributions of:
  1. UCI Forest Fires (Montesinho, Portugal)
  2. Algerian Forest Fires (Bejaia + Sidi Bel-Abbes)

IMPORTANT: Replace these with the REAL datasets before submission!
  - UCI: https://archive.ics.uci.edu/dataset/162/forest+fires
  - Algerian: https://www.kaggle.com/datasets/nitinchoudhary012/algerian-forest-fires-dataset

The synthetic data preserves:
  - Exact column names and types
  - Realistic value ranges based on published statistics
  - Approximate class distributions
  - Correlation patterns between features
"""

import pandas as pd
import numpy as np

np.random.seed(42)


def generate_uci_forest_fires(n=517):
    """
    Generate data matching UCI Forest Fires dataset structure.
    
    Real dataset stats (Cortez & Morais, 2007):
    - 517 instances from Montesinho park, Portugal
    - ~48% have area = 0 (no significant fire)
    - Strong seasonality (Aug-Sep peak)
    - FFMC typically 80-95, DMC 0-300, DC 0-900
    """
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
              'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    
    # Month distribution weighted toward summer (fire season)
    month_weights = [0.02, 0.03, 0.08, 0.04, 0.03, 0.06,
                     0.07, 0.18, 0.16, 0.10, 0.04, 0.05]
    # Normalise to sum exactly to 1.0
    total = sum(month_weights)
    month_weights = [w / total for w in month_weights]
    # Skewed toward aug/sep matching real dataset
    
    month_col = np.random.choice(months, size=n, p=month_weights)
    day_col = np.random.choice(days, size=n)
    
    # Spatial coordinates (1-9 grid)
    X_coord = np.random.randint(1, 10, size=n)
    Y_coord = np.random.randint(1, 10, size=n)
    
    # Temperature: seasonal pattern (higher in summer)
    month_num = [months.index(m) + 1 for m in month_col]
    base_temp = 10 + 10 * np.sin(np.pi * (np.array(month_num) - 3) / 6)
    temp = base_temp + np.random.normal(0, 3, n)
    temp = np.clip(temp, 2.2, 33.3)
    
    # Relative humidity: inversely correlated with temp
    RH = 80 - 1.2 * temp + np.random.normal(0, 10, n)
    RH = np.clip(RH, 15, 100).astype(int)
    
    # Wind: 0.4 to 9.4 km/h
    wind = np.random.exponential(2.5, n) + 0.4
    wind = np.clip(wind, 0.4, 9.4).round(1)
    
    # Rain: mostly 0, occasional small amounts
    rain = np.zeros(n)
    rain_mask = np.random.random(n) < 0.08
    rain[rain_mask] = np.random.exponential(1.0, rain_mask.sum())
    rain = np.clip(rain, 0, 6.4).round(1)
    
    # FWI components: correlated with weather
    FFMC = 70 + 0.5 * temp - 0.1 * RH + np.random.normal(0, 5, n)
    FFMC = np.clip(FFMC, 18.7, 96.2).round(1)
    
    DMC = 20 + 3 * temp - 0.5 * RH + np.random.exponential(30, n)
    DMC = np.clip(DMC, 1.1, 291.3).round(1)
    
    DC = 100 + 5 * temp - 1.0 * RH + np.random.exponential(100, n)
    DC = np.clip(DC, 7.9, 860.6).round(1)
    
    ISI = 2 + 0.1 * FFMC - 0.02 * RH + np.random.exponential(2, n)
    ISI = np.clip(ISI, 0.0, 56.1).round(1)
    
    # Burned area: ~48% are 0, rest follow log-normal
    area = np.zeros(n)
    fire_mask = np.random.random(n) > 0.48
    fire_count = fire_mask.sum()
    area[fire_mask] = np.exp(np.random.normal(1.5, 1.8, fire_count))
    area = np.clip(area, 0, 1090.84).round(2)
    
    df = pd.DataFrame({
        'X': X_coord, 'Y': Y_coord,
        'month': month_col, 'day': day_col,
        'FFMC': FFMC, 'DMC': DMC, 'DC': DC, 'ISI': ISI,
        'temp': temp.round(1), 'RH': RH,
        'wind': wind, 'rain': rain,
        'area': area
    })
    
    return df


def generate_algerian_fires(n=244):
    """
    Generate data matching Algerian Forest Fires dataset structure.
    
    Real dataset:
    - 244 instances from 2 regions in Algeria, June-September 2012
    - Bejaia (122) and Sidi Bel-Abbes (122)
    - Binary classification: 'fire' / 'not fire'
    - Higher fire rate (~56%) than UCI
    """
    n_per_region = n // 2
    
    regions = []
    all_data = []
    
    for region_name in ['Bejaia', 'Sidi Bel-abbes']:
        nr = n_per_region
        
        # Day, month, year
        day = np.random.randint(1, 31, nr)
        month = np.random.choice([6, 7, 8, 9], nr, p=[0.25, 0.25, 0.30, 0.20])
        year = np.full(nr, 2012)
        
        # Temperature: 22-42°C (Algeria is hot)
        Temperature = np.random.normal(33, 4, nr)
        Temperature = np.clip(Temperature, 22, 42).astype(int)
        
        # RH: 20-90%
        RH = 70 - 0.8 * Temperature + np.random.normal(0, 8, nr)
        RH = np.clip(RH, 21, 90).astype(int)
        
        # Wind speed: 6-29 km/h
        Ws = np.random.normal(15, 4, nr)
        Ws = np.clip(Ws, 6, 29).astype(int)
        
        # Rain: mostly 0
        Rain = np.zeros(nr)
        rain_mask = np.random.random(nr) < 0.05
        Rain[rain_mask] = np.random.uniform(0.1, 1.5, rain_mask.sum())
        Rain = Rain.round(1)
        
        # FWI components
        FFMC = 70 + 0.4 * Temperature - 0.15 * RH + np.random.normal(0, 5, nr)
        FFMC = np.clip(FFMC, 28.6, 96.0).round(1)
        
        DMC = 10 + 2 * Temperature - 0.3 * RH + np.random.exponential(15, nr)
        DMC = np.clip(DMC, 1.1, 65.9).round(1)
        
        DC = 50 + 3 * Temperature - 0.5 * RH + np.random.exponential(40, nr)
        DC = np.clip(DC, 7.0, 220.4).round(1)
        
        ISI = 1 + 0.08 * FFMC + np.random.exponential(2, nr)
        ISI = np.clip(ISI, 0.0, 18.5).round(1)
        
        FWI = ISI * (1 + 0.01 * DC)
        FWI = np.clip(FWI, 0.0, 31.1).round(1)
        
        # Fire/not fire classification (~56% fire)
        fire_prob = 0.3 + 0.005 * Temperature - 0.003 * RH + 0.001 * FFMC
        fire_prob = np.clip(fire_prob, 0.1, 0.9)
        Classes = np.where(np.random.random(nr) < fire_prob, 'fire   ', 'not fire   ')
        
        region_data = pd.DataFrame({
            'day': day, 'month': month, 'year': year,
            'Temperature': Temperature, 'RH': RH, 'Ws': Ws, 'Rain': Rain,
            'FFMC': FFMC, 'DMC': DMC, 'DC': DC, 'ISI': ISI, 'FWI': FWI,
            'Classes': Classes, 'Region': region_name
        })
        all_data.append(region_data)
    
    df = pd.concat(all_data, ignore_index=True)
    return df


if __name__ == '__main__':
    print("Generating synthetic datasets...")
    
    uci = generate_uci_forest_fires()
    uci.to_csv('data/forestfires.csv', index=False)
    print(f"UCI Forest Fires: {uci.shape} → data/forestfires.csv")
    print(f"  Fire events (area > 0): {(uci['area'] > 0).sum()} "
          f"({(uci['area'] > 0).mean()*100:.1f}%)")
    
    alg = generate_algerian_fires()
    alg.to_csv('data/Algerian_forest_fires_dataset.csv', index=False)
    print(f"Algerian Fires:   {alg.shape} → data/Algerian_forest_fires_dataset.csv")
    fire_count = alg['Classes'].str.strip().str.lower().eq('fire').sum()
    print(f"  Fire events: {fire_count} ({fire_count/len(alg)*100:.1f}%)")
    
    print("\nREMEMBER: Replace with REAL datasets before submission!")
