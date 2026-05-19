"""
Comprehensive Model Evaluation Framework
==========================================
7-layer evaluation strategy for comparing fire prediction models.

Layer 1: AUC-ROC comparison (primary metric)
Layer 2: Precision-Recall analysis (fire class focus)
Layer 3: Stratified k-fold cross-validation (stability)
Layer 4: Paired t-tests (statistical significance)
Layer 5: Calibration analysis (probability reliability)
Layer 6: Feature importance agreement (RF vs XGBoost)
Layer 7: Confusion matrix comparison (missed fires vs false alarms)

Usage:
    evaluator = ComprehensiveEvaluator()
    evaluator.add_model('Random Forest', y_test, y_pred, y_proba, importances)
    evaluator.add_model('XGBoost', y_test, y_pred, y_proba, importances)
    evaluator.add_model('Neural Network', y_test, y_pred, y_proba)
    evaluator.run_full_evaluation('figures/')
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    brier_score_loss
)
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.base import clone
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')


class ComprehensiveEvaluator:
    
    COLORS = {
        'Random Forest': '#2ecc71',
        'XGBoost': '#3498db',
        'Neural Network': '#e74c3c',
    }
    
    def __init__(self):
        self.models = {}
    
    def add_model(self, name: str, y_true: np.ndarray,
                  y_pred: np.ndarray, y_pred_proba: np.ndarray,
                  feature_importances: dict = None):
        """Register a model's test set predictions."""
        self.models[name] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_proba': y_pred_proba,
            'importances': feature_importances,
        }
    
    # ---- Layer 1: AUC-ROC ----
    def plot_roc_comparison(self, save_path: str):
        """
        ROC curves for all models on one figure.
        The curve closer to the top-left corner wins.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for name, data in self.models.items():
            fpr, tpr, _ = roc_curve(data['y_true'], data['y_proba'])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=self.COLORS.get(name, '#333'),
                    lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5,
                label='Random (AUC = 0.500)')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve Comparison', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # ---- Layer 2: Precision-Recall ----
    def plot_precision_recall(self, save_path: str):
        """
        PR curves focus on the fire class specifically.
        More informative than ROC for imbalanced data.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for name, data in self.models.items():
            prec, rec, _ = precision_recall_curve(data['y_true'], data['y_proba'])
            ap = average_precision_score(data['y_true'], data['y_proba'])
            ax.plot(rec, prec, color=self.COLORS.get(name, '#333'),
                    lw=2, label=f'{name} (AP = {ap:.3f})')
        
        first_model = list(self.models.values())[0]
        baseline = first_model['y_true'].mean()
        ax.axhline(y=baseline, color='k', linestyle='--', lw=1, alpha=0.5,
                   label=f'Baseline ({baseline:.2f})')
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve Comparison', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # ---- Layer 3: Cross-Validation ----
    def run_cross_validation(self, models_dict: dict,
                              X: np.ndarray, y: np.ndarray,
                              n_folds: int = 5) -> pd.DataFrame:
        """
        Stratified k-fold CV for all models.
        Reports mean ± std of AUC, showing performance stability.
        """
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        results = []
        
        for name, model in models_dict.items():
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                model_clone = clone(model)
                model_clone.fit(X[train_idx], y[train_idx])
                
                y_proba = model_clone.predict_proba(X[test_idx])[:, 1]
                y_pred = model_clone.predict(X[test_idx])
                
                results.append({
                    'Model': name, 'Fold': fold_idx + 1,
                    'AUC-ROC': roc_auc_score(y[test_idx], y_proba),
                    'Precision': precision_score(y[test_idx], y_pred, zero_division=0),
                    'Recall': recall_score(y[test_idx], y_pred, zero_division=0),
                    'F1': f1_score(y[test_idx], y_pred, zero_division=0),
                })
        
        df = pd.DataFrame(results)
        
        print(f"\n{'='*70}")
        print(f"STRATIFIED {n_folds}-FOLD CROSS-VALIDATION")
        print(f"{'='*70}")
        
        summary = df.groupby('Model').agg(
            AUC_mean=('AUC-ROC', 'mean'), AUC_std=('AUC-ROC', 'std'),
            F1_mean=('F1', 'mean'), F1_std=('F1', 'std'),
            Prec_mean=('Precision', 'mean'), Rec_mean=('Recall', 'mean'),
        ).round(4)
        
        for name, row in summary.iterrows():
            print(f"\n  {name}:")
            print(f"    AUC-ROC:   {row['AUC_mean']:.3f} ± {row['AUC_std']:.3f}")
            print(f"    F1:        {row['F1_mean']:.3f} ± {row['F1_std']:.3f}")
            print(f"    Precision: {row['Prec_mean']:.3f}")
            print(f"    Recall:    {row['Rec_mean']:.3f}")
        
        return df
    
    # ---- Layer 4: Statistical Significance ----
    def test_significance(self, cv_results: pd.DataFrame) -> dict:
        """
        Paired t-tests between models on CV fold scores.
        p < 0.05 means the performance difference is real.
        """
        model_names = cv_results['Model'].unique()
        results = {}
        
        print(f"\n{'='*70}")
        print(f"PAIRED T-TESTS (α = 0.05)")
        print(f"{'='*70}")
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                a, b = model_names[i], model_names[j]
                scores_a = cv_results[cv_results['Model'] == a]['AUC-ROC'].values
                scores_b = cv_results[cv_results['Model'] == b]['AUC-ROC'].values
                
                t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
                sig = p_val < 0.05
                
                key = f"{a} vs {b}"
                results[key] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_val),
                    'significant': bool(sig),
                    'mean_diff': float(scores_a.mean() - scores_b.mean()),
                }
                
                verdict = "SIGNIFICANT" if sig else "NOT significant"
                print(f"\n  {key}:")
                print(f"    Mean AUC diff: {scores_a.mean() - scores_b.mean():+.4f}")
                print(f"    p-value:       {p_val:.4f}")
                print(f"    Result:        {verdict}")
        
        return results
    
    # ---- Layer 5: Calibration ----
    def plot_calibration_curves(self, save_path: str):
        """
        Shows whether predicted probabilities match actual outcomes.
        "When model says 70%, do fires occur 70% of the time?"
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for name, data in self.models.items():
            try:
                prob_true, prob_pred = calibration_curve(
                    data['y_true'], data['y_proba'], n_bins=8, strategy='uniform'
                )
                brier = brier_score_loss(data['y_true'], data['y_proba'])
                ax.plot(prob_pred, prob_true, 's-',
                        color=self.COLORS.get(name, '#333'),
                        lw=2, label=f'{name} (Brier = {brier:.3f})')
            except Exception:
                pass
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfect calibration')
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Actual Fraction of Positives', fontsize=12)
        ax.set_title('Calibration Curves', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # ---- Layer 6: Feature Importance ----
    def plot_feature_importance(self, save_path: str):
        """
        Side-by-side feature importance for RF and XGBoost.
        If both rank the same features highly, that's strong evidence.
        """
        models_with_imp = {
            name: data['importances']
            for name, data in self.models.items()
            if data['importances'] is not None
        }
        
        if len(models_with_imp) < 1:
            print("  No models with feature importance available.")
            return
        
        n = len(models_with_imp)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
        if n == 1:
            axes = [axes]
        
        for ax, (name, imp) in zip(axes, models_with_imp.items()):
            sorted_feats = sorted(imp.items(), key=lambda x: x[1], reverse=True)
            names = [f[0] for f in sorted_feats]
            values = [f[1] for f in sorted_feats]
            
            ax.barh(range(len(names)), values,
                    color=self.COLORS.get(name, '#333'), alpha=0.8)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=10)
            ax.set_xlabel('Importance Score')
            ax.set_title(name, fontsize=13, fontweight='bold')
            ax.invert_yaxis()
        
        plt.suptitle('Feature Importance Comparison', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # ---- Layer 7: Confusion Matrices ----
    def plot_confusion_matrices(self, save_path: str):
        """
        Side-by-side confusion matrices highlighting missed fires.
        """
        n = len(self.models)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]
        
        for ax, (name, data) in zip(axes, self.models.items()):
            cm = confusion_matrix(data['y_true'], data['y_pred'])
            auc_val = roc_auc_score(data['y_true'], data['y_proba'])
            missed = cm[1, 0] if cm.shape[0] > 1 else 0
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['No Fire', 'Fire'],
                        yticklabels=['No Fire', 'Fire'])
            ax.set_title(f'{name}\nAUC: {auc_val:.3f} | Missed: {missed}',
                        fontsize=11)
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
        
        plt.suptitle('Confusion Matrix Comparison', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # ---- Summary Table ----
    def generate_summary_table(self) -> str:
        """Markdown table for the report Results chapter."""
        rows = []
        for name, data in self.models.items():
            auc_val = roc_auc_score(data['y_true'], data['y_proba'])
            report = classification_report(
                data['y_true'], data['y_pred'],
                target_names=['No Fire', 'Fire'], output_dict=True
            )
            brier = brier_score_loss(data['y_true'], data['y_proba'])
            cm = confusion_matrix(data['y_true'], data['y_pred'])
            
            rows.append({
                'Model': name,
                'AUC-ROC': f"{auc_val:.3f}",
                'Precision': f"{report['Fire']['precision']:.3f}",
                'Recall': f"{report['Fire']['recall']:.3f}",
                'F1': f"{report['Fire']['f1-score']:.3f}",
                'Brier': f"{brier:.3f}",
                'Missed Fires': int(cm[1, 0]) if cm.shape[0] > 1 else 0,
                'False Alarms': int(cm[0, 1]) if cm.shape[1] > 1 else 0,
            })
        
        df = pd.DataFrame(rows)
        table = df.to_markdown(index=False)
        print(f"\n{table}")
        return table
    
    # ---- Master Runner ----
    def run_full_evaluation(self, output_dir: str = 'figures/'):
        """Run all evaluation layers and save outputs."""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE 7-LAYER EVALUATION")
        print(f"{'='*70}")
        
        print("\n--- Layer 1: AUC-ROC ---")
        self.plot_roc_comparison(f'{output_dir}/roc_comparison.png')
        print(f"  Saved: {output_dir}/roc_comparison.png")
        
        print("\n--- Layer 2: Precision-Recall ---")
        self.plot_precision_recall(f'{output_dir}/precision_recall.png')
        print(f"  Saved: {output_dir}/precision_recall.png")
        
        print("\n--- Layer 5: Calibration ---")
        self.plot_calibration_curves(f'{output_dir}/calibration_curves.png')
        print(f"  Saved: {output_dir}/calibration_curves.png")
        
        print("\n--- Layer 6: Feature Importance ---")
        self.plot_feature_importance(f'{output_dir}/feature_importance.png')
        print(f"  Saved: {output_dir}/feature_importance.png")
        
        print("\n--- Layer 7: Confusion Matrices ---")
        self.plot_confusion_matrices(f'{output_dir}/confusion_matrices.png')
        print(f"  Saved: {output_dir}/confusion_matrices.png")
        
        print("\n--- Summary Table ---")
        self.generate_summary_table()
        
        print(f"\nAll evaluation outputs saved to {output_dir}/")


if __name__ == '__main__':
    # Quick test with dummy data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    
    evaluator = ComprehensiveEvaluator()
    evaluator.add_model('Test Model', y_true,
                         np.random.randint(0, 2, 100),
                         np.random.random(100),
                         {'feat_a': 0.3, 'feat_b': 0.5, 'feat_c': 0.2})
    evaluator.run_full_evaluation('figures/')
    print("\nEvaluation framework test passed!")
