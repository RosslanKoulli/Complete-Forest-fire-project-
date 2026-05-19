"""
Hex map API endpoints.

Three operations exposed:

1. POST /api/hexmap/predict_region
   Input:  list of {h3_index, latitude, longitude}
   Output: per-hex fire probability from each ML model
   Flow:   for each hex -> fetch weather -> compute FWI -> predict -> aggregate

2. WS   /api/hexmap/simulate_stream
   Input:  list of hexes (with fire_probabilities), neighbour graph,
           ignition hex IDs, simulation config
   Output: streamed frames showing hex states over time
   Flow:   build HexFireSimulator, ignite, stream HexFrames at ~20 fps

3. GET  /api/hexmap/health
   Sanity check that the weather API and FWI calculator are reachable.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from services.fwi_calculator import spin_up_fwi
from services.hex_simulation import (
    HexCell, HexFireSimulator, HexSimulationConfig, HexState
)
from services.weather_client import (
    fetch_recent_weather, fetch_weather_for_range, classify_no_fuel,
)

log = logging.getLogger(__name__)
router = APIRouter()


def get_registry(request: Request):
    """Dependency: pull the registry off app.state. Same pattern as predict.py."""
    registry = getattr(request.app.state, 'registry', None)
    if registry is None:
        raise HTTPException(503, 'Models not loaded yet')
    return registry


# -----------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------

class HexLocation(BaseModel):
    """One hex's identity. Lat/lon is the centroid of the hex."""
    h3_index: str = Field(..., description='H3 cell index string')
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    # Per-hex region override. When the frontend auto-detects the region
    # from the hex location, it sends this value; otherwise the request-
    # level `region` field is used as a fallback for the whole batch.
    region: Optional[str] = Field(default=None,
                                  description='portugal | algeria; auto-detected from lat/lon')


class RegionPredictionRequest(BaseModel):
    """A batch prediction request for all hexes in a selected region."""
    hexes: List[HexLocation] = Field(..., min_length=1, max_length=500)
    region: str = Field(default='portugal',
                        description='Default region for any hex that did not send one')

    # Optional date window. If set, weather is fetched for the 7 days
    # ending on `start_date` (so the FWI System has a proper spin-up
    # window before the fire starts), and the predictions reflect the
    # conditions ON start_date. Without these, the existing behaviour
    # of "weather ending today" is preserved.
    start_date: Optional[str] = Field(default=None,
        description='YYYY-MM-DD: first day of the simulation window')
    duration_days: Optional[float] = Field(default=None, ge=2.0, le=7.0,
        description='Length of the simulation window in days (2 to 7)')


class HexPrediction(BaseModel):
    """Per-hex output."""
    h3_index: str
    rf_probability: float
    xgb_probability: float
    nn_probability: float
    avg_probability: float
    risk_band: str       # 'low' | 'medium' | 'high' | 'no_fuel'
    weather_source: str  # 'open-meteo' | 'cache' | 'fallback'
    inputs: dict         # the FWI + weather values used for the prediction

    # Reliability metrics. domain_confidence is "how similar are these
    # inputs to training data" (0-100). model_agreement is "how much do
    # the three models agree" (0-100). overall_reliability combines
    # both via geometric mean.
    domain_confidence: float
    model_agreement: float
    overall_reliability: float
    in_distribution: bool

    # No-fuel terrain flag. When this is set, the ML models were NOT
    # run for the hex; the probabilities are zero and risk_band is
    # 'no_fuel'. The reason ('water' or 'high_altitude') is surfaced
    # to the frontend so it can label the hex appropriately. None for
    # normal land hexes.
    no_fuel_reason: Optional[str] = None
    elevation: Optional[float] = None    # metres, surfaced for the
                                          # tooltip; helpful diagnostic


class RegionPredictionResponse(BaseModel):
    predictions: List[HexPrediction]
    total_api_calls: int
    cache_hits: int
    fallback_count: int


class SimulationStartMessage(BaseModel):
    """First WebSocket message: configures and starts the simulation."""
    hexes: List[dict]              # h3_index, latitude, longitude, fire_probability
    neighbours: Dict[str, List[str]]   # h3_index -> neighbour h3_indices
    ignition: List[str]            # hex IDs to ignite at start
    base_spread_prob: float = 0.45
    wind_speed: float = 8.0
    wind_direction: float = 45.0
    vegetation_moisture: float = 0.3
    burn_duration: int = 3
    max_steps: int = 100

    # Optional time-window mode. If duration_days is set, the
    # simulation runs for at most that many days of simulated time
    # (2 to 7). hours_per_step controls the simulation tick size; the
    # default of 12 is documented in HexSimulationConfig.
    duration_days: Optional[float] = Field(default=None, ge=2.0, le=7.0)
    hours_per_step: float = 12.0


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def _risk_band(prob: float) -> str:
    if prob >= 0.7:
        return 'high'
    if prob >= 0.4:
        return 'medium'
    return 'low'


def _build_features(fwi_state, region: str, month: int) -> dict:
    """
    Map FWI + weather + region into the eleven features the ML pipeline expects.

    Matches the dict shape consumed by ModelRegistry.transform_input. The
    pipeline handles cyclic month encoding and standardisation internally.
    """
    import math
    region_encoded = 0 if region == 'portugal' else 1
    angle = 2.0 * math.pi * month / 12.0
    return {
        'temperature':       fwi_state.temperature,
        'relative_humidity': fwi_state.humidity,
        'wind_speed':        fwi_state.wind_speed,
        'rain':              fwi_state.rain,
        'FFMC':              fwi_state.ffmc,
        'DMC':               fwi_state.dmc,
        'DC':                fwi_state.dc,
        'ISI':               fwi_state.isi,
        'region_encoded':    region_encoded,
        'month_sin':         math.sin(angle),
        'month_cos':         math.cos(angle),
    }


# -----------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------

@router.get('/health')
async def hex_health():
    """Quick sanity check."""
    return {'status': 'ok', 'service': 'hexmap'}


@router.post('/predict_region', response_model=RegionPredictionResponse)
async def predict_region(req: RegionPredictionRequest, registry = Depends(get_registry)):
    """
    Predict fire probability for every hex in a selected region.

    For each hex:
      1. Fetch the last 7 days of weather (Open-Meteo, cached)
      2. Compute FWI components via Van Wagner equations
      3. Build the 11-feature vector
      4. Run all three ML models
      5. Combine into a per-hex prediction record
    """
    api_calls = 0
    cache_hits = 0
    fallback_count = 0
    predictions: List[HexPrediction] = []

    # Resolve the date window. If start_date is set, weather is fetched
    # for the 7 days ENDING on start_date - this gives the FWI System a
    # proper spin-up window before the fire is supposed to start. The
    # last day of the spin-up IS the start of the simulation.
    spinup_end_date = None
    if req.start_date:
        try:
            from datetime import date as _date, timedelta
            spinup_end_date = _date.fromisoformat(req.start_date)
            spinup_start_date = spinup_end_date - timedelta(days=7)
        except ValueError:
            raise HTTPException(400, f'Invalid start_date {req.start_date!r}; expected YYYY-MM-DD')

        # Range check: Open-Meteo's forecast endpoint provides up to 16
        # days ahead, but quality degrades past ~7 days. We cap user
        # input at 7 days into the future. On the past side, the archive
        # supports back to 1940 but we accept whatever the user provides.
        today = _date.today()
        max_future = today + timedelta(days=7)
        if spinup_end_date > max_future:
            raise HTTPException(
                400,
                f'start_date {req.start_date} is more than 7 days in the future; '
                f'forecast quality degrades beyond that range'
            )

    for hex_loc in req.hexes:
        if spinup_end_date is not None:
            weather = fetch_weather_for_range(
                hex_loc.latitude, hex_loc.longitude,
                spinup_start_date, spinup_end_date,
            )
        else:
            weather = fetch_recent_weather(hex_loc.latitude, hex_loc.longitude)
        if weather.source == 'open-meteo' or weather.source == 'open-meteo-forecast':
            api_calls += 1
        elif weather.source == 'cache':
            cache_hits += 1
        elif weather.source == 'fallback':
            fallback_count += 1

        # No-fuel check based on elevation. If the hex is open water or
        # above the tree line, the model would extrapolate way outside
        # its training distribution AND the answer would be wrong on
        # physical grounds (no vegetation to burn). Skip the model
        # entirely and return a synthetic no-fuel hex.
        no_fuel = classify_no_fuel(weather.elevation)
        if no_fuel is not None:
            predictions.append(HexPrediction(
                h3_index=hex_loc.h3_index,
                rf_probability=0.0,
                xgb_probability=0.0,
                nn_probability=0.0,
                avg_probability=0.0,
                risk_band='no_fuel',
                weather_source=weather.source,
                inputs={
                    'elevation_m': round(weather.elevation, 0) if weather.elevation is not None else None,
                },
                # Reliability is "certain" for no-fuel hexes: we're not
                # extrapolating because we're not running the model at
                # all. The map UI will exclude these from the
                # rectangle-average reliability calculation.
                domain_confidence=100.0,
                model_agreement=100.0,
                overall_reliability=100.0,
                in_distribution=True,
                no_fuel_reason=no_fuel,
                elevation=weather.elevation,
            ))
            continue

        try:
            fwi_state = spin_up_fwi(weather.history, latitude=hex_loc.latitude)
        except ValueError as e:
            log.warning(f'FWI calc failed for {hex_loc.h3_index}: {e}')
            continue

        last_month = weather.history[-1].month
        # Use per-hex region (auto-detected by the frontend) when set,
        # else fall back to the request-level region. This lets a single
        # rectangle spanning Spain and Algeria get the correct region
        # feature for each hex without the user having to split the
        # selection.
        effective_region = hex_loc.region if hex_loc.region else req.region
        features = _build_features(fwi_state, effective_region, last_month)

        # Run the three models. Same pattern as the existing predict router:
        # transform features through the data pipeline, then call
        # predict_proba on each loaded model.
        try:
            X = registry.transform_input(features)
        except Exception as e:
            log.error(f'Feature transform failed for {hex_loc.h3_index}: {e}')
            continue

        probs = {}
        try:
            for name, model in registry.models.items():
                probs[name] = float(model.predict_proba(X)[0, 1])
        except Exception as e:
            log.error(f'Model prediction failed for {hex_loc.h3_index}: {e}')
            continue

        rf = probs.get('Random Forest', 0.0)
        xgb = probs.get('XGBoost', 0.0)
        nn = probs.get('Neural Network', 0.0)
        avg = (rf + xgb + nn) / 3.0

        # Compute reliability for this hex. Uses the same feature vector
        # that was just fed to the models, plus their three probabilities.
        from services.domain_confidence import compute_confidence
        confidence = compute_confidence(
            feature_vector=X[0],
            probabilities=[rf, xgb, nn],
            domain_calc=getattr(registry, 'domain_calc', None),
        )

        predictions.append(HexPrediction(
            h3_index=hex_loc.h3_index,
            rf_probability=rf,
            xgb_probability=xgb,
            nn_probability=nn,
            avg_probability=avg,
            risk_band=_risk_band(avg),
            weather_source=weather.source,
            inputs={
                'temperature': fwi_state.temperature,
                'relative_humidity': fwi_state.humidity,
                'wind_speed': fwi_state.wind_speed,
                'rain': fwi_state.rain,
                'FFMC': round(fwi_state.ffmc, 1),
                'DMC': round(fwi_state.dmc, 1),
                'DC': round(fwi_state.dc, 1),
                'ISI': round(fwi_state.isi, 2),
                'month': last_month,
                'elevation_m': round(weather.elevation, 0) if weather.elevation is not None else None,
            },
            domain_confidence=round(confidence.domain_confidence, 1),
            model_agreement=round(confidence.model_agreement, 1),
            overall_reliability=round(confidence.overall_reliability, 1),
            in_distribution=confidence.in_distribution,
            no_fuel_reason=None,
            elevation=weather.elevation,
        ))

    return RegionPredictionResponse(
        predictions=predictions,
        total_api_calls=api_calls,
        cache_hits=cache_hits,
        fallback_count=fallback_count,
    )


@router.websocket('/simulate_stream')
async def simulate_stream(websocket: WebSocket):
    """
    Stream a hex-based fire spread simulation frame by frame.

    The client opens the WebSocket, sends a SimulationStartMessage as JSON,
    and receives one JSON frame per simulation step until completion.

    Each frame: {step, states: {h3_index: 0|1|2|3}, burning, burnt, unburnt}
    Final message: {done: true}
    """
    await websocket.accept()
    try:
        # Receive configuration
        raw = await websocket.receive_json()
        try:
            config_msg = SimulationStartMessage(**raw)
        except Exception as e:
            await websocket.send_json({'error': f'Invalid config: {e}'})
            await websocket.close()
            return

        # Build hex graph
        cells: Dict[str, HexCell] = {}
        for h in config_msg.hexes:
            cells[h['h3_index']] = HexCell(
                h3_index=h['h3_index'],
                latitude=h['latitude'],
                longitude=h['longitude'],
                fire_probability=h.get('fire_probability', 0.5),
                state=HexState.UNBURNT,
            )

        # Filter neighbours to only include hexes that are in our set
        # (hexes near the rectangle boundary will have neighbours outside it)
        neighbours: Dict[str, List[str]] = {}
        for hid, ns in config_msg.neighbours.items():
            if hid in cells:
                neighbours[hid] = [n for n in ns if n in cells]

        sim_config = HexSimulationConfig(
            base_spread_prob=config_msg.base_spread_prob,
            wind_speed=config_msg.wind_speed,
            wind_direction=config_msg.wind_direction,
            vegetation_moisture=config_msg.vegetation_moisture,
            burn_duration=config_msg.burn_duration,
            max_steps=config_msg.max_steps,
            time_window_days=config_msg.duration_days,
            hours_per_step=config_msg.hours_per_step,
        )

        sim = HexFireSimulator(cells, neighbours, sim_config)
        sim.ignite(config_msg.ignition)

        # Stream frames at ~20 fps (50 ms per frame)
        for frame in sim.run():
            await websocket.send_json({
                'step': frame.step,
                'states': frame.states,
                'burning': frame.burning_count,
                'burnt': frame.burnt_count,
                'unburnt': frame.unburnt_count,
                'elapsed_hours': frame.elapsed_hours,
                'current_day': frame.current_day,
            })
            await asyncio.sleep(0.05)

        # Final message includes the totals so the client knows whether
        # the simulation ended due to burnout or window expiry.
        await websocket.send_json({
            'done': True,
            'final_step': sim.step,
            'final_elapsed_hours': sim.step * sim_config.hours_per_step,
            'window_capped': (
                sim_config.time_window_days is not None and
                sim.step * sim_config.hours_per_step >= sim_config.time_window_days * 24.0
            ),
        })

    except WebSocketDisconnect:
        log.info('Hex sim websocket disconnected')
    except Exception as e:
        log.exception(f'Hex sim error: {e}')
        try:
            await websocket.send_json({'error': str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
