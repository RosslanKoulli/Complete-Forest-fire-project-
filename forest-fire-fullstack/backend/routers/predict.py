"""
Predict Router
================

POST /api/predict
    Run all three models on a single input. Optionally include SHAP
    explanations.

POST /api/predict/{model_name}
    Run only one named model. Useful when the UI wants to refresh a
    single card after a slider tweak without re-running everything.

The actual ML work is delegated to the registry on app.state. This
router's job is to translate between the JSON shape and the registry's
numpy/dict interface.
"""

import math
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from schemas.predict import (
    PredictionRequest,
    PredictionResponse,
    ModelPrediction,
    FeatureContribution,
)


router = APIRouter()


def get_registry(request: Request):
    """Dependency: pull the registry off app.state."""
    registry = getattr(request.app.state, 'registry', None)
    if registry is None:
        raise HTTPException(503, 'Models not loaded yet')
    return registry


def risk_band(p: float) -> str:
    """Bucket a probability into low/medium/high. Thresholds match the UI."""
    if p >= 0.7: return 'high'
    if p >= 0.4: return 'medium'
    return 'low'


def request_to_feature_dict(req: PredictionRequest) -> dict:
    """
    Transform the JSON request into the dict shape the data pipeline expects.

    The pipeline's transform_single_input takes:
        temperature, relative_humidity, wind_speed, rain,
        FFMC, DMC, DC, ISI, region_encoded, month_sin, month_cos

    Notable mappings:
      * region: string -> 0 or 1 encoding (matches training)
      * month: integer -> sin/cos pair (cyclical encoding)
    """
    region_encoded = 0 if req.region == 'portugal' else 1
    angle = 2.0 * math.pi * req.month / 12.0
    return {
        'temperature':       req.temperature,
        'relative_humidity': float(req.relative_humidity),
        'wind_speed':        req.wind_speed,
        'rain':              req.rain,
        'FFMC':              req.FFMC,
        'DMC':               req.DMC,
        'DC':                req.DC,
        'ISI':               req.ISI,
        'region_encoded':    region_encoded,
        'month_sin':         math.sin(angle),
        'month_cos':         math.cos(angle),
    }


def explain_one(registry, model_name: str, X) -> tuple[float, list[FeatureContribution], str | None]:
    """
    Run SHAP for one model. Returns (base_value, contributions, error).
    On success error is None; on failure base/contributions are None.

    Wrapped in try/except because SHAP can fail in surprising ways at
    inference time (rare but real e.g. xgboost version drift between
    training and serving). A failed explanation should not fail the
    prediction.
    """
    explainer = registry.explainers.get(model_name)
    if explainer is None:
        return None, None, 'Explainer not available for this model'

    try:
        e = explainer.explain(X)
        contribs = [
            FeatureContribution(feature=name, contribution=float(val))
            for name, val in zip(e.feature_names, e.contributions)
        ]
        return float(e.base_value), contribs, None
    except Exception as exc:
        return None, None, f'Explanation failed: {type(exc).__name__}'


# ============================================================
# Endpoints
# ============================================================

@router.post('', response_model=PredictionResponse)
async def predict_all(
    req: PredictionRequest,
    registry = Depends(get_registry),
):
    """
    Run every loaded model on the input and return their predictions.

    If explain=True, SHAP values are computed too. Tree models add ~ms;
    the MLP adds ~1-3 seconds because KernelSHAP is sampling.
    """
    if not registry.models:
        raise HTTPException(503, 'No models loaded')

    feature_dict = request_to_feature_dict(req)
    X = registry.transform_input(feature_dict)

    predictions: list[ModelPrediction] = []
    for name, model in registry.models.items():
        try:
            prob = float(model.predict_proba(X)[0, 1])
        except Exception as exc:
            raise HTTPException(500, f'Inference failed for {name}: {exc}')

        base_value = None
        contribs = None
        err = None
        if req.explain:
            base_value, contribs, err = explain_one(registry, name, X)

        predictions.append(ModelPrediction(
            model_name=name,
            probability=prob,
            risk_band=risk_band(prob),
            base_value=base_value,
            contributions=contribs,
            explanation_unavailable_reason=err,
        ))

    # Disagreement summary. max minus min, in percentage points
    probs = [p.probability for p in predictions]
    most = max(predictions, key=lambda p: p.probability)
    least = min(predictions, key=lambda p: p.probability)

    return PredictionResponse(
        predictions=predictions,
        most_alarmed_model=most.model_name,
        least_alarmed_model=least.model_name,
        disagreement_pp=round((max(probs) - min(probs)) * 100, 1),
    )


@router.post('/{model_name}', response_model=ModelPrediction)
async def predict_one(
    model_name: str,
    req: PredictionRequest,
    registry = Depends(get_registry),
):
    """
    Run a single named model. The model_name must match exactly one of
    the loaded keys ('Random Forest', 'XGBoost', 'Neural Network').
    Spaces in the URL are fine if the client URL-encodes them.
    """
    if model_name not in registry.models:
        loaded = ', '.join(registry.models.keys()) or '(none)'
        raise HTTPException(
            404,
            f'Model {model_name!r} not loaded. Available: {loaded}',
        )

    feature_dict = request_to_feature_dict(req)
    X = registry.transform_input(feature_dict)

    model = registry.models[model_name]
    prob = float(model.predict_proba(X)[0, 1])

    base_value = None
    contribs = None
    err = None
    if req.explain:
        base_value, contribs, err = explain_one(registry, model_name, X)

    return ModelPrediction(
        model_name=model_name,
        probability=prob,
        risk_band=risk_band(prob),
        base_value=base_value,
        contributions=contribs,
        explanation_unavailable_reason=err,
    )
