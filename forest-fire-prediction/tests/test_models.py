"""
Tests for ML Models
====================
Tests all three prediction approaches: Random Forest, XGBoost, Neural Network.
Verifies training, prediction output format, model persistence, and metrics.

Run: python -m pytest tests/test_models.py -v
"""

import sys
import os
import numpy as np
import pytest
import joblib
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_pipeline import FireDataPipeline
from models.random_forest_model import RandomForestFireModel
from models.xgboost_model import XGBoostFireModel
from models.neural_network_model import NeuralNetworkFireModel


@pytest.fixture(scope='module')
def data_smote():
    """Prepare SMOTE-balanced data once for all tests."""
    pipeline = FireDataPipeline('../data/forestfires.csv',
                                 '../data/Algerian_forest_fires_dataset.csv',
                                 '../data/icnf_portugal_extended.csv')
    combined = pipeline.build_unified_dataset()
    return pipeline.prepare_features(combined, apply_smote=True)


@pytest.fixture(scope='module')
def data_raw():
    """Prepare non-SMOTE data for XGBoost tests."""
    pipeline = FireDataPipeline('../data/forestfires.csv',
                                 '../data/Algerian_forest_fires_dataset.csv',
                                 '../data/icnf_portugal_extended.csv')
    combined = pipeline.build_unified_dataset()
    return pipeline.prepare_features(combined, apply_smote=False)


# ================================================================
# RANDOM FOREST TESTS
# ================================================================

class TestRandomForest:
    
    def test_trains_without_error(self, data_smote):
        model = RandomForestFireModel()
        results = model.train(
            data_smote['X_train'], data_smote['y_train'],
            data_smote['X_test'], data_smote['y_test'],
            feature_names=data_smote['feature_names'],
            do_grid_search=False
        )
        assert results is not None
    
    def test_results_contain_required_keys(self, data_smote):
        model = RandomForestFireModel()
        results = model.train(
            data_smote['X_train'], data_smote['y_train'],
            data_smote['X_test'], data_smote['y_test'],
            do_grid_search=False
        )
        required = ['model_name', 'accuracy', 'auc_roc',
                     'confusion_matrix', 'classification_report',
                     'feature_importance', 'y_pred', 'y_pred_proba']
        for key in required:
            assert key in results, f"Missing key: {key}"
    
    def test_auc_above_random(self, data_smote):
        model = RandomForestFireModel()
        results = model.train(
            data_smote['X_train'], data_smote['y_train'],
            data_smote['X_test'], data_smote['y_test'],
            do_grid_search=False
        )
        # On the extended UCI + Algerian + ICNF dataset (9,598 samples)
        # Random Forest reliably reaches AUC ~0.79 on the held-out test
        # set. We assert >= 0.65 to give a safety margin for fold-to-fold
        # variation, machine differences, and library version drift,
        # while still being meaningfully above random (0.50).
        assert results['auc_roc'] >= 0.65, \
            f"AUC {results['auc_roc']:.3f} is unexpectedly low; " \
            f"check data pipeline and model hyperparameters"
    
    def test_predictions_are_binary(self, data_smote):
        model = RandomForestFireModel()
        results = model.train(
            data_smote['X_train'], data_smote['y_train'],
            data_smote['X_test'], data_smote['y_test'],
            do_grid_search=False
        )
        unique = set(np.unique(results['y_pred']))
        assert unique.issubset({0, 1})
    
    def test_probabilities_sum_to_one(self, data_smote):
        model = RandomForestFireModel()
        model.train(
            data_smote['X_train'], data_smote['y_train'],
            data_smote['X_test'], data_smote['y_test'],
            do_grid_search=False
        )
        proba = model.model.predict_proba(data_smote['X_test'])
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)
    
    def test_feature_importance_sums_to_one(self, data_smote):
        model = RandomForestFireModel()
        results = model.train(
            data_smote['X_train'], data_smote['y_train'],
            data_smote['X_test'], data_smote['y_test'],
            feature_names=data_smote['feature_names'],
            do_grid_search=False
        )
        total = sum(results['feature_importance'].values())
        assert abs(total - 1.0) < 0.01, \
            f"Feature importances sum to {total}, expected ~1.0"
    
    def test_save_and_load(self, data_smote):
        model = RandomForestFireModel()
        model.train(
            data_smote['X_train'], data_smote['y_train'],
            data_smote['X_test'], data_smote['y_test'],
            do_grid_search=False
        )
        
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            path = f.name
        
        try:
            model.save(path)
            assert os.path.exists(path)
            
            model2 = RandomForestFireModel()
            model2.load(path)
            
            pred1 = model.model.predict(data_smote['X_test'][:5])
            pred2 = model2.model.predict(data_smote['X_test'][:5])
            assert np.array_equal(pred1, pred2), "Loaded model gives different predictions"
        finally:
            os.unlink(path)
    
    def test_predict_single_sample(self, data_smote):
        model = RandomForestFireModel()
        model.train(
            data_smote['X_train'], data_smote['y_train'],
            data_smote['X_test'], data_smote['y_test'],
            do_grid_search=False
        )
        result = model.predict(data_smote['X_test'][:1])
        assert 'prediction' in result
        assert 'fire_probability' in result
        assert 0 <= result['fire_probability'] <= 1


# ================================================================
# XGBOOST TESTS
# ================================================================

class TestXGBoost:
    
    def test_trains_without_error(self, data_raw):
        model = XGBoostFireModel()
        results = model.train(
            data_raw['X_train'], data_raw['y_train'],
            data_raw['X_test'], data_raw['y_test'],
            do_grid_search=False
        )
        assert results is not None
    
    def test_results_contain_required_keys(self, data_raw):
        model = XGBoostFireModel()
        results = model.train(
            data_raw['X_train'], data_raw['y_train'],
            data_raw['X_test'], data_raw['y_test'],
            do_grid_search=False
        )
        required = ['model_name', 'accuracy', 'auc_roc',
                     'feature_importance', 'scale_pos_weight']
        for key in required:
            assert key in results, f"Missing key: {key}"
    
    def test_scale_pos_weight_calculated(self, data_raw):
        model = XGBoostFireModel()
        results = model.train(
            data_raw['X_train'], data_raw['y_train'],
            data_raw['X_test'], data_raw['y_test'],
            do_grid_search=False
        )
        assert results['scale_pos_weight'] > 0, "scale_pos_weight should be positive"
    
    def test_predictions_are_binary(self, data_raw):
        model = XGBoostFireModel()
        results = model.train(
            data_raw['X_train'], data_raw['y_train'],
            data_raw['X_test'], data_raw['y_test'],
            do_grid_search=False
        )
        unique = set(np.unique(results['y_pred']))
        assert unique.issubset({0, 1})
    
    def test_save_and_load(self, data_raw):
        model = XGBoostFireModel()
        model.train(
            data_raw['X_train'], data_raw['y_train'],
            data_raw['X_test'], data_raw['y_test'],
            do_grid_search=False
        )
        
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            path = f.name
        
        try:
            model.save(path)
            model2 = XGBoostFireModel()
            model2.load(path)
            
            pred1 = model.model.predict(data_raw['X_test'][:5])
            pred2 = model2.model.predict(data_raw['X_test'][:5])
            assert np.array_equal(pred1, pred2)
        finally:
            os.unlink(path)


# ================================================================
# NEURAL NETWORK TESTS
# ================================================================

class TestNeuralNetwork:
    
    def test_trains_without_error(self, data_smote):
        model = NeuralNetworkFireModel()
        results = model.train(
            data_smote['X_train'], data_smote['y_train'],
            data_smote['X_test'], data_smote['y_test'],
            do_grid_search=False
        )
        assert results is not None
    
    def test_results_contain_required_keys(self, data_smote):
        model = NeuralNetworkFireModel()
        results = model.train(
            data_smote['X_train'], data_smote['y_train'],
            data_smote['X_test'], data_smote['y_test'],
            do_grid_search=False
        )
        required = ['model_name', 'accuracy', 'auc_roc',
                     'training_loss_curve']
        for key in required:
            assert key in results, f"Missing key: {key}"
    
    def test_loss_curve_exists(self, data_smote):
        model = NeuralNetworkFireModel()
        results = model.train(
            data_smote['X_train'], data_smote['y_train'],
            data_smote['X_test'], data_smote['y_test'],
            do_grid_search=False
        )
        assert results['training_loss_curve'] is not None
        assert len(results['training_loss_curve']) > 1, \
            "Should have trained for at least 2 epochs"
    
    def test_loss_decreases(self, data_smote):
        """First loss should be higher than last (model learned)."""
        model = NeuralNetworkFireModel()
        results = model.train(
            data_smote['X_train'], data_smote['y_train'],
            data_smote['X_test'], data_smote['y_test'],
            do_grid_search=False
        )
        curve = results['training_loss_curve']
        assert curve[0] > curve[-1], \
            f"Loss should decrease: first={curve[0]:.4f}, last={curve[-1]:.4f}"
    
    def test_predictions_are_binary(self, data_smote):
        model = NeuralNetworkFireModel()
        results = model.train(
            data_smote['X_train'], data_smote['y_train'],
            data_smote['X_test'], data_smote['y_test'],
            do_grid_search=False
        )
        unique = set(np.unique(results['y_pred']))
        assert unique.issubset({0, 1})
    
    def test_predict_single_sample(self, data_smote):
        model = NeuralNetworkFireModel()
        model.train(
            data_smote['X_train'], data_smote['y_train'],
            data_smote['X_test'], data_smote['y_test'],
            do_grid_search=False
        )
        result = model.predict(data_smote['X_test'][:1])
        assert result['prediction'] in [0, 1]
        assert 0 <= result['fire_probability'] <= 1
        assert result['model'] == 'Neural Network'


# ================================================================
# CROSS-MODEL CONSISTENCY
# ================================================================

class TestCrossModelConsistency:
    """Verify all models produce compatible outputs."""
    
    def test_all_models_same_test_size(self, data_smote, data_raw):
        """All models evaluated on same number of test samples."""
        assert len(data_smote['X_test']) == len(data_raw['X_test'])
    
    def test_all_models_produce_probabilities(self, data_smote, data_raw):
        """Every model outputs calibrated probabilities [0, 1]."""
        models = [
            (RandomForestFireModel(), data_smote),
            (XGBoostFireModel(), data_raw),
            (NeuralNetworkFireModel(), data_smote),
        ]
        
        for model, data in models:
            model.train(
                data['X_train'], data['y_train'],
                data['X_test'], data['y_test'],
                do_grid_search=False
            )
            proba = model.model.predict_proba(data['X_test'])
            assert proba.shape[1] == 2, \
                f"{model.name}: Expected 2 probability columns"
            assert np.all(proba >= 0) and np.all(proba <= 1), \
                f"{model.name}: Probabilities outside [0, 1]"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
