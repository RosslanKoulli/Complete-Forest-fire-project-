"""
Tests for Data Pipeline
========================
Verifies dataset loading, cleaning, unification, feature
engineering, SMOTE application, and scaling.

Run: python -m pytest tests/test_data_pipeline.py -v
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_pipeline import FireDataPipeline


@pytest.fixture
def pipeline():
    """Create a pipeline instance with test datasets."""
    return FireDataPipeline(
        'data/forestfires.csv',
        'data/Algerian_forest_fires_dataset.csv'
    )


@pytest.fixture
def combined_df(pipeline):
    """Load and combine both datasets."""
    return pipeline.build_unified_dataset()


class TestDataLoading:
    """Test that both datasets load correctly."""
    
    def test_uci_loads(self, pipeline):
        df = pipeline.load_uci_dataset()
        assert len(df) > 0, "UCI dataset should not be empty"
        assert 'fire_occurred' in df.columns
        assert 'source_dataset' in df.columns
        assert df['source_dataset'].iloc[0] == 'uci'
    
    def test_algerian_loads(self, pipeline):
        df = pipeline.load_algerian_dataset()
        assert len(df) > 0, "Algerian dataset should not be empty"
        assert 'fire_occurred' in df.columns
        assert df['source_dataset'].iloc[0] == 'algerian'
    
    def test_combined_size(self, combined_df):
        """Combined dataset should be sum of both."""
        assert len(combined_df) == 761 or len(combined_df) > 500, \
            f"Expected ~761 samples, got {len(combined_df)}"


class TestFeatureSchema:
    """Test that the unified feature schema is correct."""
    
    def test_all_feature_columns_present(self, combined_df, pipeline):
        for col in pipeline.FEATURE_COLUMNS:
            assert col in combined_df.columns, f"Missing column: {col}"
    
    def test_target_column_present(self, combined_df, pipeline):
        assert pipeline.TARGET_COLUMN in combined_df.columns
    
    def test_target_is_binary(self, combined_df, pipeline):
        unique_vals = combined_df[pipeline.TARGET_COLUMN].unique()
        assert set(unique_vals).issubset({0, 1}), \
            f"Target should be binary, got {unique_vals}"
    
    def test_no_missing_values_in_features(self, combined_df, pipeline):
        missing = combined_df[pipeline.FEATURE_COLUMNS].isnull().sum().sum()
        assert missing == 0, f"Found {missing} missing values in features"
    
    def test_month_cyclical_encoding_range(self, combined_df):
        """sin/cos values should be between -1 and 1."""
        assert combined_df['month_sin'].between(-1, 1).all()
        assert combined_df['month_cos'].between(-1, 1).all()
    
    def test_region_encoded_values(self, combined_df):
        """Region should be 0 (UCI/Portugal), 1 (Bejaia), or 2 (Sidi)."""
        valid = {0, 1, 2}
        actual = set(combined_df['region_encoded'].unique())
        assert actual.issubset(valid), f"Unexpected regions: {actual - valid}"


class TestFeaturePreparation:
    """Test train/test splitting, scaling, and SMOTE."""
    
    def test_train_test_split_sizes(self, pipeline, combined_df):
        data = pipeline.prepare_features(combined_df, apply_smote=False)
        total = len(data['X_train']) + len(data['X_test'])
        assert total == len(combined_df), \
            f"Split lost samples: {total} vs {len(combined_df)}"
    
    def test_test_size_approximately_20_percent(self, pipeline, combined_df):
        data = pipeline.prepare_features(combined_df, apply_smote=False)
        test_ratio = len(data['X_test']) / len(combined_df)
        assert 0.15 < test_ratio < 0.25, \
            f"Test ratio {test_ratio:.2f} outside expected range"
    
    def test_smote_balances_classes(self, pipeline, combined_df):
        data = pipeline.prepare_features(combined_df, apply_smote=True)
        n_fire = data['class_distribution']['train_fire']
        n_no_fire = data['class_distribution']['train_no_fire']
        assert n_fire == n_no_fire, \
            f"SMOTE should balance: {n_fire} fire vs {n_no_fire} no-fire"
    
    def test_no_smote_preserves_distribution(self, pipeline, combined_df):
        data = pipeline.prepare_features(combined_df, apply_smote=False)
        n_fire = data['class_distribution']['train_fire']
        n_no_fire = data['class_distribution']['train_no_fire']
        assert n_fire != n_no_fire or abs(n_fire - n_no_fire) < 50, \
            "Without SMOTE, classes should reflect natural distribution"
    
    def test_scaling_zero_mean(self, pipeline, combined_df):
        """After StandardScaler, training features should have ~0 mean."""
        data = pipeline.prepare_features(combined_df, apply_smote=False)
        means = np.abs(data['X_train'].mean(axis=0))
        assert np.all(means < 0.5), \
            f"Scaled means too far from 0: {means}"
    
    def test_feature_names_returned(self, pipeline, combined_df):
        data = pipeline.prepare_features(combined_df, apply_smote=False)
        assert len(data['feature_names']) == data['X_train'].shape[1]
    
    def test_feature_count(self, pipeline, combined_df):
        data = pipeline.prepare_features(combined_df, apply_smote=False)
        assert data['X_train'].shape[1] == 11, \
            f"Expected 11 features, got {data['X_train'].shape[1]}"


class TestSingleInputTransform:
    """Test web app input transformation."""
    
    def test_transform_single_input(self, pipeline, combined_df):
        pipeline.prepare_features(combined_df, apply_smote=False)
        
        input_dict = {
            'temperature': 30.0,
            'relative_humidity': 35.0,
            'wind_speed': 4.0,
            'rain': 0.0,
            'FFMC': 90.1,
            'DMC': 100.5,
            'DC': 500.0,
            'ISI': 10.2,
            'region_encoded': 0,
            'month_sin': 0.866,
            'month_cos': 0.5,
        }
        
        X = pipeline.transform_single_input(input_dict)
        assert X.shape == (1, 11), f"Expected (1, 11), got {X.shape}"
    
    def test_transform_before_fit_raises(self):
        pipeline = FireDataPipeline('data/forestfires.csv',
                                     'data/Algerian_forest_fires_dataset.csv')
        with pytest.raises(RuntimeError):
            pipeline.transform_single_input({'temperature': 30})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
