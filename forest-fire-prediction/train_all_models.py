"""
Master Training Script
=======================
Runs the complete pipeline end-to-end:
1. Load + prepare data (UCI + Algerian + ICNF Phase 2 extension)
2. Generate EDA visualisations
3. Train Random Forest, XGBoost, Neural Network
4. Run 7-layer comprehensive evaluation
5. Run cross-validation + significance tests
6. Save all models, figures, and results

The data pipeline gracefully falls back to UCI + Algerian only when
icnf_portugal_extended.csv is missing from data/, so this same script
runs whether or not the Phase 2 ICNF extension has been generated.

Run: python train_all_models.py
Then: streamlit run app.py
"""

import os
import sys
import json
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from data_pipeline import FireDataPipeline
from models.random_forest_model import RandomForestFireModel
from models.xgboost_model import XGBoostFireModel
from models.neural_network_model import NeuralNetworkFireModel
from evaluation.evaluation_framework import ComprehensiveEvaluator
from eda_notebook import run_full_eda


def main():
    os.makedirs('trained_models', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("=" * 70)
    print("FOREST FIRE PREDICTION — FULL TRAINING PIPELINE")
    print("3-Model Comparison: RF vs XGBoost vs Neural Network")
    print("=" * 70)
    
    # ---- Step 1: Data ----
    print("\n[1/7] Loading and preparing data...")
    pipeline_smote = FireDataPipeline(
        'data/forestfires.csv',
        'data/Algerian_forest_fires_dataset.csv',
        'data/icnf_portugal_extended.csv'
    )
    combined = pipeline_smote.build_unified_dataset()
    data_smote = pipeline_smote.prepare_features(combined, apply_smote=True)
    joblib.dump(pipeline_smote, 'trained_models/data_pipeline.joblib')
    
    # Separate pipeline for XGBoost (no SMOTE)
    pipeline_raw = FireDataPipeline(
        'data/forestfires.csv',
        'data/Algerian_forest_fires_dataset.csv',
        'data/icnf_portugal_extended.csv'
    )
    combined_raw = pipeline_raw.build_unified_dataset()
    data_raw = pipeline_raw.prepare_features(combined_raw, apply_smote=False)
    
    # ---- Step 2: EDA ----
    print("\n[2/7] Generating EDA visualisations...")
    run_full_eda(combined, output_dir='figures')
    
    # ---- Step 3: Train Random Forest ----
    print("\n[3/7] Training Random Forest...")
    rf = RandomForestFireModel()
    rf_results = rf.train(
        data_smote['X_train'], data_smote['y_train'],
        data_smote['X_test'], data_smote['y_test'],
        feature_names=data_smote['feature_names'],
        do_grid_search=False  # Set True for final run (slower)
    )
    rf.save('trained_models/rf_fire_model.joblib')
    
    # ---- Step 4: Train XGBoost ----
    print("\n[4/7] Training XGBoost...")
    xgb = XGBoostFireModel()
    xgb_results = xgb.train(
        data_raw['X_train'], data_raw['y_train'],
        data_raw['X_test'], data_raw['y_test'],
        feature_names=data_raw['feature_names'],
        do_grid_search=False  # Set True for final run
    )
    xgb.save('trained_models/xgb_fire_model.joblib')
    
    # ---- Step 5: Train Neural Network ----
    print("\n[5/7] Training Neural Network...")
    nn = NeuralNetworkFireModel()
    nn_results = nn.train(
        data_smote['X_train'], data_smote['y_train'],
        data_smote['X_test'], data_smote['y_test'],
        feature_names=data_smote['feature_names'],
        do_grid_search=False  # Set True for final run
    )
    nn.save('trained_models/nn_fire_model.joblib')
    nn.plot_training_curve('figures/nn_loss_curve.png')
    
    # ---- Step 6: Comprehensive Evaluation ----
    print("\n[6/7] Running 7-layer evaluation...")
    evaluator = ComprehensiveEvaluator()
    evaluator.add_model('Random Forest',
                         data_smote['y_test'], rf_results['y_pred'],
                         rf_results['y_pred_proba'],
                         rf_results.get('feature_importance'))
    evaluator.add_model('XGBoost',
                         data_raw['y_test'], xgb_results['y_pred'],
                         xgb_results['y_pred_proba'],
                         xgb_results.get('feature_importance'))
    evaluator.add_model('Neural Network',
                         data_smote['y_test'], nn_results['y_pred'],
                         nn_results['y_pred_proba'], None)
    
    evaluator.run_full_evaluation('figures')
    
    # ---- Step 7: Cross-Validation + Significance ----
    print("\n[7/7] Cross-validation and significance tests...")
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.neural_network import MLPClassifier
    
    # Use raw (non-SMOTE) data for fair CV comparison
    X_full = np.vstack([data_raw['X_train'], data_raw['X_test']])
    y_full = np.concatenate([data_raw['y_train'], data_raw['y_test']])
    
    cv_models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42, eval_metric='auc', use_label_encoder=False
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(64, 32, 16), alpha=0.001,
            activation='relu', solver='adam', max_iter=500,
            early_stopping=True, random_state=42
        ),
    }
    
    cv_results = evaluator.run_cross_validation(cv_models, X_full, y_full, n_folds=5)
    significance = evaluator.test_significance(cv_results)
    
    # Save results
    cv_results.to_csv('results/cross_validation_results.csv', index=False)
    
    with open('results/significance_tests.json', 'w') as f:
        json.dump(significance, f, indent=2)
    
    # Save model results (convert numpy for JSON)
    def make_serialisable(d):
        out = {}
        for k, v in d.items():
            if hasattr(v, 'tolist'):
                out[k] = v.tolist()
            elif isinstance(v, dict):
                out[k] = make_serialisable(v)
            elif isinstance(v, np.floating):
                out[k] = float(v)
            elif isinstance(v, np.integer):
                out[k] = int(v)
            else:
                out[k] = v
        return out
    
    comparison = {
        'Random Forest': make_serialisable(rf_results),
        'XGBoost': make_serialisable(xgb_results),
        'Neural Network': make_serialisable(nn_results),
    }
    with open('results/comparison_results.json', 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    # ---- Done ----
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Models:  trained_models/*.joblib")
    print(f"  Figures: figures/*.png")
    print(f"  Results: results/*.json, results/*.csv")
    
    n_figs = len([f for f in os.listdir('figures') if f.endswith('.png')])
    print(f"\n  Total figures generated: {n_figs}")
    print(f"\n  Launch web app:  streamlit run app.py")


if __name__ == '__main__':
    main()
