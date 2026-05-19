"""
Neural Network Fire Ignition Classifier
=========================================
Approach 3 of 3 in the comparative study.

Feedforward MLP using scikit-learn's MLPClassifier for API
consistency with RF and XGBoost.

Research question answered: "Does deep learning improve over
tree-based methods for fire prediction on small tabular data?"

Architecture: Input(11) → Dense(64,ReLU) → Dense(32,ReLU) → Dense(16,ReLU) → Output(1,Sigmoid)
~3,280 parameters vs ~600 training samples → 5.5:1 ratio
Early stopping + L2 regularisation (alpha) prevent overfitting.
"""

import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix
)


class NeuralNetworkFireModel:
    
    def __init__(self):
        self.model = None
        self.best_params = None
        self.name = "Neural Network"
    
    def train(self, X_train, y_train, X_test, y_test,
              feature_names=None, do_grid_search=True):
        """
        Train MLP classifier with optional hyperparameter search.
        
        Searches over architecture size, learning rate, and L2
        regularisation strength. Early stopping monitors a 15%
        internal validation split to prevent overfitting.
        """
        if do_grid_search:
            param_grid = {
                'hidden_layer_sizes': [
                    (64, 32, 16),
                    (128, 64, 32),
                    (64, 32),
                    (32, 16),
                ],
                'learning_rate_init': [0.001, 0.005, 0.01],
                'alpha': [0.0001, 0.001, 0.01],
            }
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            grid_search = GridSearchCV(
                MLPClassifier(
                    activation='relu', solver='adam',
                    batch_size=32, max_iter=500,
                    early_stopping=True, validation_fraction=0.15,
                    random_state=42,
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
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                activation='relu', solver='adam',
                alpha=0.001, batch_size=32,
                learning_rate_init=0.001, max_iter=500,
                early_stopping=True, validation_fraction=0.15,
                random_state=42,
            )
            self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        results = {
            'model_name': self.name,
            'accuracy': float((y_pred == y_test).mean()),
            'auc_roc': float(roc_auc_score(y_test, y_pred_proba)),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(
                y_test, y_pred, target_names=['No Fire', 'Fire'],
                output_dict=True
            ),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'best_params': self.best_params,
            'training_loss_curve': (self.model.loss_curve_
                                     if hasattr(self.model, 'loss_curve_') else None),
        }
        
        print(f"\n{'='*50}")
        print(f"NEURAL NETWORK RESULTS")
        print(f"{'='*50}")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  AUC-ROC:   {results['auc_roc']:.4f}")
        if self.best_params:
            print(f"  Arch:      {self.best_params.get('hidden_layer_sizes')}")
            print(f"  Alpha:     {self.best_params.get('alpha')}")
        if hasattr(self.model, 'n_iter_'):
            print(f"  Epochs:    {self.model.n_iter_}")
        print(classification_report(y_test, y_pred,
                                     target_names=['No Fire', 'Fire']))
        
        return results
    
    def plot_training_curve(self, save_path: str = '../figures/nn_loss_curve.png'):
        """
        Plot training loss curve, unique to the NN approach.
        Shows how the network learned over epochs.
        """
        if not hasattr(self.model, 'loss_curve_'):
            print("No loss curve available.")
            return
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.model.loss_curve_, color='#e74c3c', lw=2, label='Training loss')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Neural Network Training Loss Curve', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=len(self.model.loss_curve_) - 1,
                   color='gray', linestyle='--', alpha=0.5,
                   label=f'Early stop (epoch {len(self.model.loss_curve_)})')
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Loss curve saved: {save_path}")
    
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
                                 '../data/Algerian_forest_fires_dataset.csv',
                                )
    combined = pipeline.build_unified_dataset()
    data = pipeline.prepare_features(combined, apply_smote=True)
    
    model = NeuralNetworkFireModel()
    results = model.train(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        feature_names=data['feature_names'],
        do_grid_search=False  # Fast test
    )
    model.plot_training_curve()
    model.save('../trained_models/nn_fire_model.joblib')
