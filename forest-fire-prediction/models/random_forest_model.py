"""
Random Forest Fire Ignition Classifier
=======================================
Approach 1 of 3 in the comparative study.

Ensemble of decision trees using bagging. Each tree is trained
on a random subset of features and data, then predictions are
aggregated via majority voting.

Strengths for fire prediction:
- Handles non-linear feature interactions (temp × humidity)
- Built-in feature importance for report interpretability
- Robust to outliers in environmental data
- No feature scaling required (but we scale anyway for consistency)

Hyperparameter tuning via GridSearchCV with stratified 5-fold CV.
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix
)


class RandomForestFireModel:
    
    def __init__(self):
        self.model = None
        self.best_params = None
        self.name = "Random Forest"
    
    def train(self, X_train, y_train, X_test, y_test,
              feature_names=None, do_grid_search=True):
        """
        Train with optional hyperparameter search.
        
        Grid search explores n_estimators, max_depth, min_samples_split,
        and class_weight. Scoring is AUC-ROC (threshold-independent).
        
        Returns dict with all evaluation metrics.
        """
        if do_grid_search:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'class_weight': ['balanced', 'balanced_subsample'],
            }
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42, n_jobs=-1),
                param_grid, cv=cv, scoring='roc_auc',
                n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            print(f"  Best params: {self.best_params}")
            print(f"  Best CV AUC: {grid_search.best_score_:.4f}")
        else:
            self.model = RandomForestClassifier(
                n_estimators=200, max_depth=10,
                min_samples_split=5, class_weight='balanced',
                random_state=42, n_jobs=-1
            )
            self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        feat_names = feature_names or [f'F{i}' for i in range(X_train.shape[1])]
        
        results = {
            'model_name': self.name,
            'accuracy': float((y_pred == y_test).mean()),
            'auc_roc': float(roc_auc_score(y_test, y_pred_proba)),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(
                y_test, y_pred, target_names=['No Fire', 'Fire'],
                output_dict=True
            ),
            'feature_importance': dict(zip(feat_names,
                                            self.model.feature_importances_)),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'best_params': self.best_params,
        }
        
        print(f"\n{'='*50}")
        print(f"RANDOM FOREST RESULTS")
        print(f"{'='*50}")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  AUC-ROC:   {results['auc_roc']:.4f}")
        print(classification_report(y_test, y_pred,
                                     target_names=['No Fire', 'Fire']))
        
        return results
    
    def save(self, path: str):
        joblib.dump(self.model, path)
        print(f"  Model saved: {path}")
    
    def load(self, path: str):
        self.model = joblib.load(path)
    
    def predict(self, X: np.ndarray) -> dict:
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        return {
            'prediction': int(pred),
            'fire_probability': float(proba[1]),
            'no_fire_probability': float(proba[0]),
            'model': self.name
        }


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from data_pipeline import FireDataPipeline
    
    pipeline = FireDataPipeline('../data/forestfires.csv',
                                 '../data/Algerian_forest_fires_dataset.csv')
    combined = pipeline.build_unified_dataset()
    data = pipeline.prepare_features(combined, apply_smote=True)
    
    model = RandomForestFireModel()
    results = model.train(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        feature_names=data['feature_names'],
        do_grid_search=True
    )
    model.save('trained_models/rf_fire_model.joblib')
