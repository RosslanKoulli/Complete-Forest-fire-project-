# Forest Fire Ignition Prediction and Spread Modelling System

**CI601 Individual Project — University of Brighton — 2026**

A machine learning system that predicts forest fire ignition probability and simulates fire spread patterns. Compares three ML approaches (Random Forest, XGBoost, Neural Network) with a 7-layer evaluation framework.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add real datasets to data/ folder
#    - forestfires.csv (UCI Forest Fires)
#    - Algerian_forest_fires_dataset.csv

# 3. Train all models
python train_all_models.py

# 4. Launch web application
streamlit run app.py
```

## Project Structure

```
├── app.py                        # Streamlit web application
├── train_all_models.py           # Master training pipeline
├── data_pipeline.py              # Data loading, cleaning, SMOTE
├── eda_notebook.py               # Exploratory data analysis
├── models/
│   ├── random_forest_model.py    # Approach 1: Bagging ensemble
│   ├── xgboost_model.py          # Approach 2: Boosting ensemble
│   └── neural_network_model.py   # Approach 3: Neural network
├── spread_model/
│   └── cellular_automata.py      # Fire spread simulation engine
├── evaluation/
│   └── evaluation_framework.py   # 7-layer comparison framework
├── data/                         # Dataset CSVs
├── trained_models/               # Saved .joblib models
├── figures/                      # Generated visualisations
└── results/                      # Metrics, CV results, significance tests
```

## Datasets

| Dataset | Samples | Source |
|---------|---------|--------|
| UCI Forest Fires | 517 | https://archive.ics.uci.edu/dataset/162/forest+fires |
| Algerian Forest Fires | 244 | https://www.kaggle.com/datasets/nitinchoudhary012/algerian-forest-fires-dataset |

## Evaluation Framework

1. **AUC-ROC** — Primary metric (threshold-independent)
2. **Precision-Recall** — Fire class performance
3. **5-Fold Cross-Validation** — Stability (mean ± std)
4. **Paired t-tests** — Statistical significance (p < 0.05)
5. **Calibration curves** — Probability reliability
6. **Feature importance** — RF vs XGBoost agreement
7. **Confusion matrices** — Missed fires vs false alarms

## Technology Stack

Python, scikit-learn, XGBoost, Streamlit, Matplotlib, NumPy, Pandas
