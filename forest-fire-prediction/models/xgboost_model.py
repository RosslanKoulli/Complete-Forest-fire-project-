"""
XGBoost Fire Ignition Classifier
=================================
Approach 2 of 3 in the comparative study.

Gradient-boosted decision trees — builds trees sequentially where
each new tree corrects errors from the previous ensemble.

Key difference from Random Forest:
- RF builds trees independently (bagging, parallel)
- XGBoost builds trees sequentially (boosting, corrective)

Uses scale_pos_weight instead of SMOTE for class imbalance,
because XGBoost was designed with this parameter and it avoids
introducing synthetic samples into a small dataset.
"""

import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix
)


class XGBoostFireModel:
    
    def __init__(self):
        self.model = None
        self.best_params = None
        self.name = "XGBoost"
    
    def train(self, X_train, y_train, X_test, y_test,
              feature_names=None, do_grid_search=True):
        """
        Train XGBoost with native class imbalance handling.
        
        scale_pos_weight = count(negative) / count(positive)
        tells XGBoost to weight fire samples more heavily.
        
        Data passed here should NOT have SMOTE applied.
        """
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_weight = n_neg / max(n_pos, 1)
        
        if do_grid_search:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
            }
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            grid_search = GridSearchCV(
                XGBClassifier(
                    scale_pos_weight=scale_weight,
                    random_state=42,
                    eval_metric='auc',
                    use_label_encoder=False,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                ),
                param_grid, cv=cv, scoring='roc_auc',
                n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            print(f"  Best params: {self.best_params}")
            print(f"  Best CV AUC: {grid_search.best_score_:.4f}")
        else:
            self.model = XGBClassifier(
                n_estimators=200, max_depth=5,
                learning_rate=0.1, subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_weight,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, eval_metric='auc',
                use_label_encoder=False,
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
            'scale_pos_weight': scale_weight,
        }
        
        print(f"\n{'='*50}")
        print(f"XGBOOST RESULTS")
        print(f"{'='*50}")
        print(f"  Accuracy:       {results['accuracy']:.4f}")
        print(f"  AUC-ROC:        {results['auc_roc']:.4f}")
        print(f"  scale_pos_wt:   {scale_weight:.3f}")
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
            'model': self.name,
        }


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from data_pipeline import FireDataPipeline
    
    pipeline = FireDataPipeline('../data/forestfires.csv',
                                 '../data/Algerian_forest_fires_dataset.csv')
    combined = pipeline.build_unified_dataset()
    # No SMOTE for XGBoost — uses scale_pos_weight instead
    data = pipeline.prepare_features(combined, apply_smote=False)
    
    model = XGBoostFireModel()
    results = model.train(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        feature_names=data['feature_names'],
        do_grid_search=False
    )
    model.save('../trained_models/xgb_fire_model.joblib')
