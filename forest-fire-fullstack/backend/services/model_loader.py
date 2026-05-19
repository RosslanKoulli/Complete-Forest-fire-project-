"""
Model Registry: centralises model loading
=============================================

This is the bridge between the FastAPI backend and the existing
forest-fire-prediction project. It loads:

  * Three trained classifiers from trained_models/*.joblib
  * The fitted data pipeline (scaler + feature ordering)
  * Three SHAP explainers (one per model)
  * A background sample for the MLP's KernelSHAP

Everything is loaded once at app startup via FastAPI's lifespan hook
(see main.py) and stored on app.state. Endpoints reach into app.state
to grab whatever they need.

Why a class, not module-level globals
--------------------------------------
The registry is a class so that tests can construct a fresh registry
with mock models, and so that the loading logic is grouped with the
state it manages. Module-level globals would work but couple the
loading code to the API surface in a way that's harder to test.

"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np


def _locate_project_root() -> Path:
    """
    Find the original forest-fire-prediction project on disk.

    Resolution order:
      1. $FOREST_FIRE_PROJECT_ROOT environment variable, if set
      2. ../../forest-fire-prediction relative to this file
      3. ../forest-fire-prediction
      4. ./forest-fire-prediction (when backend is run from project root)

    Raises FileNotFoundError with a helpful message if nothing matches.
    """
    env = os.environ.get('FOREST_FIRE_PROJECT_ROOT')
    if env:
        p = Path(env).resolve()
        if (p / 'data_pipeline.py').exists():
            return p
        raise FileNotFoundError(
            f'FOREST_FIRE_PROJECT_ROOT={env!r} does not contain data_pipeline.py'
        )

    here = Path(__file__).resolve().parent
    candidates = [
        here.parent.parent / 'forest-fire-prediction',
        here.parent / 'forest-fire-prediction',
        here.parent.parent.parent / 'forest-fire-prediction',
        Path.cwd() / 'forest-fire-prediction',
    ]
    for c in candidates:
        if (c / 'data_pipeline.py').exists():
            return c.resolve()

    raise FileNotFoundError(
        'Could not locate forest-fire-prediction project. Set '
        'FOREST_FIRE_PROJECT_ROOT to its absolute path.\n'
        f'Tried: {[str(c) for c in candidates]}'
    )


class ModelRegistry:
    """
    Holds every trained artefact the API needs.

    Attributes (populated by load_all):
      models      - dict[str, estimator] keyed by display name
      pipeline    - fitted DataPipeline instance
      explainers  - dict[str, PredictionExplainer] (None if shap missing)
      feature_names - list[str] in training order
      project_root - Path to the original project
    """

    MODEL_FILES = {
        'Random Forest':  'rf_fire_model.joblib',
        'XGBoost':        'xgb_fire_model.joblib',
        'Neural Network': 'nn_fire_model.joblib',
    }

    def __init__(self):
        self.project_root: Optional[Path] = None
        self.models: dict = {}
        self.pipeline = None
        self.explainers: dict = {}
        self.feature_names: list = []
        self.background_sample: Optional[np.ndarray] = None
        self.domain_calc = None    # built from background_sample at startup

    def load_all(self) -> None:
        """
        Resolve the project root, register it on sys.path so we can import
        its modules, then load every artefact. Raises if anything's missing
        because a half-loaded registry is worse than a clear error.
        """
        self.project_root = _locate_project_root()
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

        self._load_models()
        self._load_pipeline()
        self._load_background()
        self._build_explainers()
        self._build_domain_calc()

    # ------------------------------------------------------------------
    # Loading steps
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        models_dir = self.project_root / 'trained_models'
        if not models_dir.exists():
            raise FileNotFoundError(
                f'{models_dir} not found. Run train_all_models.py in the '
                'main project before starting the API.'
            )

        for name, filename in self.MODEL_FILES.items():
            path = models_dir / filename
            if path.exists():
                # joblib.load returns whatever was saved -- typically our
                # project's wrapper class instances.
                self.models[name] = joblib.load(path)
            else:
                # We tolerate missing models so the API still starts in
                # partial state. Endpoints check membership and return a
                # clear error if a requested model is absent.
                print(f'[model_loader] WARN: {path} missing, skipping {name}')

    def _load_pipeline(self) -> None:
        path = self.project_root / 'trained_models' / 'data_pipeline.joblib'
        if not path.exists():
            raise FileNotFoundError(f'{path} not found')
        self.pipeline = joblib.load(path)

        # The pipeline exposes the feature names. Different versions of
        # the project have used different attribute names, so we probe.
        for attr in ('feature_names', 'features', '_feature_names'):
            v = getattr(self.pipeline, attr, None)
            if v:
                self.feature_names = list(v)
                break
        if not self.feature_names:
            # Fallback: hard-code the canonical order. This is the order
            # the Streamlit app uses, so any saved model trained from that
            # build will agree with it.
            self.feature_names = [
                'temperature', 'relative_humidity', 'wind_speed', 'rain',
                'FFMC', 'DMC', 'DC', 'ISI',
                'region_encoded', 'month_sin', 'month_cos',
            ]

    def _load_background(self) -> None:
        """
        Load the SHAP background sample if it was saved during training.
        If not present, fall back to a Gaussian approximation -- crude
        but lets the API start.
        """
        path = self.project_root / 'trained_models' / 'background_sample.npy'
        if path.exists():
            self.background_sample = np.load(path)
        else:
            print('[model_loader] No background_sample.npy; using fallback')
            n = len(self.feature_names)
            self.background_sample = np.random.RandomState(42).normal(
                size=(50, n),
            )

    def _build_explainers(self) -> None:
        """
        Construct one PredictionExplainer per model. Imports lazily because
        the shap package is heavy and we want to surface a clear error if
        it's missing rather than crashing at import time.
        """
        try:
            from model_explanations import PredictionExplainer
        except ImportError as e:
            print(f'[model_loader] SHAP module not importable: {e}')
            print('[model_loader] Explainability endpoints will be disabled.')
            return

        for name, model in self.models.items():
            try:
                self.explainers[name] = PredictionExplainer(
                    model=model,
                    model_name=name,
                    feature_names=self.feature_names,
                    background=self.background_sample if name == 'Neural Network' else None,
                )
            except Exception as e:
                # An explainer failure should not take down the entire API.
                # Predictions still work; explanations for this model
                # will return a 503 with a clear reason.
                print(f'[model_loader] Could not build explainer for {name}: {e}')

    def _build_domain_calc(self) -> None:
        """
        Build the DomainConfidenceCalculator. Prefer the full training
        feature matrix if available (X_train_processed.npy in the
        trained_models directory) because covariance estimation in 11
        dimensions needs more than 50 samples to be stable. Fall back
        to the SHAP background sample if the full matrix isn't there.

        The previous version always used the 50-sample SHAP background.
        That worked syntactically but the covariance estimate was
        unreliable, which is what made in-distribution Algerian data
        score below 20% reliability in earlier user testing.
        """
        from .domain_confidence import DomainConfidenceCalculator

        # First choice: full processed training set
        full_path = self.project_root / 'trained_models' / 'X_train_processed.npy'
        training_features = None
        source_label = ''
        if full_path.exists():
            try:
                training_features = np.load(full_path)
                source_label = f'{training_features.shape[0]} training samples (full)'
            except Exception as e:
                print(f'[model_loader] X_train_processed.npy could not be loaded: {e}')

        # Fallback: SHAP background sample
        if training_features is None:
            if self.background_sample is None:
                print('[model_loader] No training data available, domain confidence disabled')
                return
            training_features = self.background_sample
            source_label = f'{training_features.shape[0]} background samples (fallback)'

        try:
            self.domain_calc = DomainConfidenceCalculator(training_features)
            print(f'[model_loader] Domain confidence calculator built from {source_label}')
        except Exception as e:
            print(f'[model_loader] Could not build domain confidence: {e}')
            self.domain_calc = None

    # ------------------------------------------------------------------
    # Helpers used by routers
    # ------------------------------------------------------------------

    def transform_input(self, raw: dict) -> np.ndarray:
        """
        Turn a JSON-shaped input dict into the scaled numpy array that
        the models consume. Delegates to the pipeline's transform_single_input
        method, which exists in the existing project's DataPipeline class.
        """
        if self.pipeline is None:
            raise RuntimeError('Pipeline not loaded')
        if hasattr(self.pipeline, 'transform_single_input'):
            return self.pipeline.transform_single_input(raw)
        # Fallback path for older pipeline versions: assemble the row by
        # hand in the canonical feature order.
        row = np.array([[raw.get(f, 0.0) for f in self.feature_names]])
        return self.pipeline.transform(row) if hasattr(self.pipeline, 'transform') else row
