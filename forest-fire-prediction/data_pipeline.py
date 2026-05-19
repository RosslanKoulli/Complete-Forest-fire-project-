"""
Data Pipeline Module
====================
Handles loading, cleaning, feature engineering, and preparation
of the UCI Forest Fires, Algerian Forest Fires, and ICNF Portugal
datasets for model training.

Design Decision: Datasets are combined to increase sample size and
to test cross-regional generalization. UCI (517 samples) and
Algerian (244 samples) form the original training set used for the
baseline evaluation. ICNF (Phase 2 extension, ~9851 samples for
2017-2020) is loaded when present to test whether sample size was
the binding constraint on the AUC plateau observed in the
seven-layer evaluation framework. All three datasets share the
same FWI-based feature schema, making unification straightforward.

Usage (two datasets):
    pipeline = FireDataPipeline('data/forestfires.csv',
                                 'data/Algerian_forest_fires_dataset.csv')

Usage (three datasets, Phase 2 extended):
    pipeline = FireDataPipeline('data/forestfires.csv',
                                 'data/Algerian_forest_fires_dataset.csv',
                                 'data/icnf_portugal_extended.csv')
    combined = pipeline.build_unified_dataset()
    data = pipeline.prepare_features(combined, apply_smote=True)
    # data['X_train'], data['X_test'], data['y_train'], data['y_test']
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


class FireDataPipeline:
    """
    Unified data pipeline for fire ignition prediction.
    
    Loads both datasets, maps them to a common feature schema,
    handles class imbalance via SMOTE, and scales features.
    
    Key decisions documented:
    1. UCI 'area' target converted to binary (area > 0 = fire)
    2. Month encoded cyclically with sin/cos (Dec close to Jan)
    3. SMOTE applied AFTER train/test split (prevents leakage)
    4. StandardScaler fitted on training data only
    """
    
    FEATURE_COLUMNS = [
        'temperature',
        'relative_humidity',
        'wind_speed',
        'rain',
        'FFMC',
        'DMC',
        'DC',
        'ISI',
        'region_encoded',
        'month_sin',
        'month_cos',
    ]
    
    TARGET_COLUMN = 'fire_occurred'
    
    def __init__(self, uci_path: str, algerian_path: str, icnf_path: str = None):
        """
        Parameters
        ----------
        uci_path : str
            Path to UCI Forest Fires CSV.
        algerian_path : str
            Path to Algerian Forest Fires CSV.
        icnf_path : str, optional
            Path to ICNF Portugal extended CSV produced by
            integrate_icnf.py. If None or the file is missing, ICNF
            data is skipped (backward compatible with existing
            two-dataset behaviour).
        """
        self.uci_path = uci_path
        self.algerian_path = algerian_path
        self.icnf_path = icnf_path
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def load_uci_dataset(self) -> pd.DataFrame:
        """
        Load and transform the UCI Forest Fires dataset.
        
        Converts continuous 'area' to binary classification:
        area > 0 → fire_occurred = 1 (any burning counts)
        area == 0 → fire_occurred = 0
        
        Month is cyclically encoded: sin/cos transform maps
        months onto a unit circle so December (12) is close to
        January (1), not far away as in ordinal encoding.
        """
        df = pd.read_csv(self.uci_path)
        
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
            'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
            'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        month_num = df['month'].str.lower().map(month_map)
        
        result = pd.DataFrame({
            'temperature': df['temp'],
            'relative_humidity': df['RH'],
            'wind_speed': df['wind'],
            'rain': df['rain'],
            'FFMC': df['FFMC'],
            'DMC': df['DMC'],
            'DC': df['DC'],
            'ISI': df['ISI'],
            'region_encoded': 0,  # 0 = Portugal
            'month_sin': np.sin(2 * np.pi * month_num / 12),
            'month_cos': np.cos(2 * np.pi * month_num / 12),
            'fire_occurred': (df['area'] > 0).astype(int),
            'source_dataset': 'uci'
        })
        
        return result
    
    def load_algerian_dataset(self) -> pd.DataFrame:
        """
        Load and transform the Algerian Forest Fires dataset.
        
        Two regions: Bejaia (encoded as 1) and Sidi Bel-Abbes (2).
        Target is already binary ('fire' / 'not fire').
        
        Known data quality issue: whitespace in column names and
        values. Pipeline strips these automatically.
        """
        df = pd.read_csv(self.algerian_path)
        
        # Clean whitespace issues common in this dataset
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str).str.strip()
        
        # Convert numeric columns that may be strings
        numeric_cols = ['Temperature', 'RH', 'Ws', 'Rain',
                        'FFMC', 'DMC', 'DC', 'ISI']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=[c for c in numeric_cols if c in df.columns])
        
        # Region encoding
        region_map = {'Bejaia': 1, 'Sidi Bel-abbes': 2}
        if 'Region' in df.columns:
            region_encoded = df['Region'].str.strip().map(region_map).fillna(1)
        else:
            region_encoded = pd.Series(np.ones(len(df)))
        
        # Target encoding: 'fire' = 1, 'not fire' = 0
        if 'Classes' in df.columns:
            classes_lower = df['Classes'].str.lower().str.strip()
            fire = (~classes_lower.str.contains('not')).astype(int)
        else:
            fire = pd.Series(np.zeros(len(df), dtype=int))
        
        # Month cyclical encoding
        if 'month' in df.columns:
            month_num = pd.to_numeric(df['month'], errors='coerce').fillna(7)
        else:
            month_num = pd.Series(np.full(len(df), 7))
        
        result = pd.DataFrame({
            'temperature': df['Temperature'].values,
            'relative_humidity': df['RH'].values,
            'wind_speed': df['Ws'].values,
            'rain': df['Rain'].values,
            'FFMC': df['FFMC'].values,
            'DMC': df['DMC'].values,
            'DC': df['DC'].values,
            'ISI': df['ISI'].values,
            'region_encoded': region_encoded.values,
            'month_sin': np.sin(2 * np.pi * month_num.values / 12),
            'month_cos': np.cos(2 * np.pi * month_num.values / 12),
            'fire_occurred': fire.values,
            'source_dataset': 'algerian'
        })
        
        return result
    
    def load_icnf_dataset(self) -> pd.DataFrame:
        """
        Load the ICNF Portugal extended dataset produced by
        integrate_icnf.py.
        
        This dataset is the Phase 2 extension of the training set. It
        contains thousands of additional Portuguese fire occurrences
        from the official ICNF (Instituto da Conservacao da Natureza
        e das Florestas) database for 2017-2020, joined with daily
        ERA5-Land meteorological reanalysis from the Copernicus
        Climate Data Store, with FWI components computed locally per
        Van Wagner (1987).
        
        Schema differences from UCI:
          - 'label' is already binary (0 = no fire, 1 = fire); no
            threshold step needed
          - 'month' is integer 1-12, not English text abbreviation
          - 'region' column is the string 'portugal' (same regional
            climate as the UCI Montesinho data, so encoded with the
            same region_encoded value of 0)
        
        Returns
        -------
        pd.DataFrame with the same 13 columns as load_uci_dataset and
        load_algerian_dataset, ready for concatenation in
        build_unified_dataset.
        """
        df = pd.read_csv(self.icnf_path)
        
        # ICNF month is already integer 1-12. Coerce in case of any
        # malformed rows, then default to mid-fire-season (July) if
        # parsing fails.
        month_num = pd.to_numeric(df['month'], errors='coerce').fillna(7).astype(int)
        
        result = pd.DataFrame({
            'temperature': df['temp'],
            'relative_humidity': df['RH'],
            'wind_speed': df['wind'],
            'rain': df['rain'],
            'FFMC': df['FFMC'],
            'DMC': df['DMC'],
            'DC': df['DC'],
            'ISI': df['ISI'],
            'region_encoded': 0,  # Same encoding as UCI: 0 = Portugal
            'month_sin': np.sin(2 * np.pi * month_num / 12),
            'month_cos': np.cos(2 * np.pi * month_num / 12),
            'fire_occurred': df['label'].astype(int),
            'source_dataset': 'icnf'
        })
        
        return result
    
    def build_unified_dataset(self) -> pd.DataFrame:
        """
        Combine the configured datasets into a single DataFrame.
        
        UCI and Algerian are always loaded. ICNF is also loaded and
        concatenated when an icnf_path was provided to __init__ AND
        the file exists on disk. Missing ICNF data is not an error -
        the pipeline falls back to the original two-dataset
        behaviour, so the same training script runs whether or not
        the Phase 2 extension has been generated yet.
        
        Prints summary statistics for verification, including a
        per-source breakdown so an examiner can see how many rows
        came from each dataset.
        """
        import os
        
        uci_df = self.load_uci_dataset()
        alg_df = self.load_algerian_dataset()
        
        frames = [uci_df, alg_df]
        icnf_df = None
        if self.icnf_path and os.path.exists(self.icnf_path):
            try:
                icnf_df = self.load_icnf_dataset()
                frames.append(icnf_df)
            except Exception as e:
                print(f"  Warning: ICNF load failed ({e}); continuing without it")
        
        combined = pd.concat(frames, ignore_index=True)
        
        print(f"{'='*50}")
        print(f"DATASET SUMMARY")
        print(f"{'='*50}")
        print(f"  UCI samples:      {len(uci_df)}")
        print(f"  Algerian samples: {len(alg_df)}")
        if icnf_df is not None:
            print(f"  ICNF samples:     {len(icnf_df)}")
        elif self.icnf_path:
            print(f"  ICNF samples:     (skipped; file not found at {self.icnf_path})")
        print(f"  Combined total:   {len(combined)}")
        print(f"  Fire events:      {int(combined['fire_occurred'].sum())} "
              f"({combined['fire_occurred'].mean()*100:.1f}%)")
        print(f"  No-fire events:   {int((1 - combined['fire_occurred']).sum())} "
              f"({(1 - combined['fire_occurred']).mean()*100:.1f}%)")
        print(f"{'='*50}")
        
        return combined
    
    def prepare_features(self, df: pd.DataFrame, apply_smote: bool = True,
                         test_size: float = 0.2, random_state: int = 42):
        """
        Prepare model-ready feature matrices.
        
        Steps:
        1. Extract features and target
        2. Stratified train/test split
        3. Scale features (fit on train only)
        4. Optionally apply SMOTE to training set only
        
        Parameters
        ----------
        apply_smote : bool
            True for RF and NN, False for XGBoost (uses scale_pos_weight)
        
        Returns
        -------
        dict with X_train, X_test, y_train, y_test, feature_names, etc.
        """
        X = df[self.FEATURE_COLUMNS].values
        y = df[self.TARGET_COLUMN].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y
        )
        
        # Fit scaler on train only, transform both
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.is_fitted = True
        
        if apply_smote:
            smote = SMOTE(random_state=random_state)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            print(f"  SMOTE: {len(X_train)} → {len(X_train_scaled)} training samples")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.FEATURE_COLUMNS,
            'class_distribution': {
                'train_fire': int(y_train.sum()),
                'train_no_fire': int(len(y_train) - y_train.sum()),
                'test_fire': int(y_test.sum()),
                'test_no_fire': int(len(y_test) - y_test.sum()),
            }
        }
    
    def transform_single_input(self, input_dict: dict) -> np.ndarray:
        """
        Transform a single user input for prediction.
        Used by the web app when a user submits conditions.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted yet.")
        
        features = np.array([[input_dict[col] for col in self.FEATURE_COLUMNS]])
        return self.scaler.transform(features)


if __name__ == '__main__':
    # Quick test
    pipeline = FireDataPipeline(
        'data/forestfires.csv',
        'data/Algerian_forest_fires_dataset.csv',
        'data/icnf_portugal_extended.csv'
    )
    combined = pipeline.build_unified_dataset()
    
    print("\nWith SMOTE (for RF/NN):")
    data_smote = pipeline.prepare_features(combined, apply_smote=True)
    print(f"  X_train: {data_smote['X_train'].shape}")
    print(f"  X_test:  {data_smote['X_test'].shape}")
    print(f"  Class dist: {data_smote['class_distribution']}")
    
    print("\nWithout SMOTE (for XGBoost):")
    pipeline2 = FireDataPipeline(
        'data/forestfires.csv',
        'data/Algerian_forest_fires_dataset.csv',
        'data/icnf_portugal_extended.csv'
    )
    combined2 = pipeline2.build_unified_dataset()
    data_raw = pipeline2.prepare_features(combined2, apply_smote=False)
    print(f"  X_train: {data_raw['X_train'].shape}")
    print(f"  X_test:  {data_raw['X_test'].shape}")
    print(f"  Class dist: {data_raw['class_distribution']}")
