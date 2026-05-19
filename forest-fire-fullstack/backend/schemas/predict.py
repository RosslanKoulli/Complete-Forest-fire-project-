"""
Pydantic Schemas for Prediction Endpoints
===========================================

Pydantic enforces request/response shapes at the FastAPI boundary. Two
gains beyond hand-rolled validation:

  * The OpenAPI docs at /docs are generated from these schemas. An
    examiner browsing the docs sees exactly what fields are required,
    what types they have, and what the responses look like.
  * If a client sends garbage, FastAPI rejects it with a 422 and a
    structured error pointing at the bad field. Endpoints never see
    invalid data.

Field constraints (Field(...)) match the real-world ranges of the
features. They're not just validation -- they document the domain.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ============================================================
# Request
# ============================================================

class PredictionRequest(BaseModel):
    """The eleven environmental features the models consume."""

    temperature: float = Field(
        ..., ge=-10, le=55, description='Air temperature in degrees Celsius',
    )
    relative_humidity: float = Field(
        ..., ge=0, le=100, description='Relative humidity as a percentage',
    )
    wind_speed: float = Field(
        ..., ge=0, le=50, description='Wind speed in km/h',
    )
    rain: float = Field(
        ..., ge=0, le=20, description='Rainfall in mm',
    )

    # Fire Weather Index components
    FFMC: float = Field(..., ge=0, le=101, description='Fine Fuel Moisture Code')
    DMC: float = Field(..., ge=0, le=300, description='Duff Moisture Code')
    DC: float = Field(..., ge=0, le=900, description='Drought Code')
    ISI: float = Field(..., ge=0, le=60, description='Initial Spread Index')

    # Categorical / temporal
    month: int = Field(..., ge=1, le=12, description='Month, 1=Jan to 12=Dec')
    region: Literal['portugal', 'algeria'] = Field(
        'portugal',
        description='Region encoding -- the training data covers these two only',
    )

    # Whether to compute SHAP explanations. Tree models are fast (~ms);
    # the MLP takes a few seconds, so the client opts in explicitly.
    explain: bool = Field(
        False,
        description='If true, include per-feature SHAP contributions in the response',
    )

    class Config:
        json_schema_extra = {
            'example': {
                'temperature': 32.0,
                'relative_humidity': 35,
                'wind_speed': 12.5,
                'rain': 0.0,
                'FFMC': 88.0,
                'DMC': 120.0,
                'DC': 450.0,
                'ISI': 9.0,
                'month': 8,
                'region': 'portugal',
                'explain': True,
            }
        }


# ============================================================
# Response components
# ============================================================

class FeatureContribution(BaseModel):
    """A single feature's SHAP contribution to one prediction."""
    feature: str
    contribution: float = Field(
        ..., description='SHAP value -- positive pushes toward fire, negative away',
    )


class ModelPrediction(BaseModel):
    """One model's prediction plus optional explanation."""
    model_name: str
    probability: float = Field(..., ge=0, le=1)
    risk_band: Literal['low', 'medium', 'high']

    # Populated when the request asked for explanations
    base_value: Optional[float] = None
    contributions: Optional[list[FeatureContribution]] = None
    explanation_unavailable_reason: Optional[str] = None


class PredictionResponse(BaseModel):
    """What /api/predict returns: all three model outputs plus a summary."""
    predictions: list[ModelPrediction]
    most_alarmed_model: str
    least_alarmed_model: str
    disagreement_pp: float = Field(
        ...,
        description='Difference between max and min probability, in percentage points',
    )
