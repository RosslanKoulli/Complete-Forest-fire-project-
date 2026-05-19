"""
Hex map API endpoints.

Three operations exposed:

1. POST /api/hexmap/predict_region
   Input:  list of {h3_index, latitude, longitude}
   Output: per-hex fire probability from each ML model
   Flow:   for each hex -> fetch weather -> compute FWI -> predict -> aggregate

2. WS   /api/hexmap/predict_region_stream
   Same logic as #1 but streams progress per hex over a WebSocket so the
   frontend can render a real progress bar and cancel mid-flight by
   closing the socket. Wire protocol:
     - Client sends a RegionPredictionRequest JSON as the first frame.
     - Server emits {type:'started', total}
     - Server emits {type:'progress', completed, total, prediction}
       for each hex (prediction may be null if that hex failed).
     - Server emits {type:'done', total_api_calls, cache_hits, fallback_count}
   If the client closes the socket mid-stream, the server stops
   processing further hexes - no wasted work, no zombie predictions.

3. WS   /api/hexmap/simulate_stream
   Input:  list of hexes (with fire_probabilities), neighbour graph,
           ignition hex IDs, simulation config
   Output: streamed frames showing hex states over time
   Flow:   build HexFireSimulator, ignite, stream HexFrames at ~20 fps

4. GET  /api/hexmap/health
   Sanity check that the weather API and FWI calculator are reachable.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketState

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

    #Date window. If set, weather is fetched for the 7 days
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

    # H3 resolution of the hexes in this simulation. Used to scale the
    # animation frame rate: bigger hexes (lower resolution) span more
    # land per cell, so each frame should linger longer to make the
    # visual spread rate feel proportional to the physical area being
    # covered. Without this, a res-5 simulation (25 km hexes) burns
    # across the map at the same wall-clock pace as a res-8 simulation
    # (1.5 km hexes), which feels uncanny. None falls back to the
    # original 20 fps for backward compatibility.
    hex_resolution: Optional[int] = Field(default=None, ge=0, le=15)


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


def _resolve_date_window(req: RegionPredictionRequest):
    """
    Validate and resolve the optional date window.

    Returns (spinup_start_date, spinup_end_date) or (None, None). Raises
    HTTPException for invalid dates; callers should catch and translate
    to the appropriate transport error (HTTP body for POST, WebSocket
    error message for the stream endpoint).
    """
    if not req.start_date:
        return None, None

    from datetime import date as _date, timedelta
    try:
        spinup_end_date = _date.fromisoformat(req.start_date)
    except ValueError:
        raise HTTPException(400, f'Invalid start_date {req.start_date!r}; expected YYYY-MM-DD')
    spinup_start_date = spinup_end_date - timedelta(days=7)

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
    return spinup_start_date, spinup_end_date


def _predict_one_hex(
    hex_loc: HexLocation,
    registry,
    req_region: str,
    spinup_start_date,
    spinup_end_date,
) -> Tuple[Optional[HexPrediction], str]:
    """
    Run the full pipeline for a single hex.

    Returns (prediction, weather_source). The weather_source is one of
    'open-meteo', 'open-meteo-forecast', 'cache', 'fallback', or '' if
    the hex failed at the FWI/transform/predict stages (in which case
    prediction is None and the caller should skip the hex).

    Extracted so both the POST and streaming endpoints share identical
    semantics - no two copies of this logic to keep in sync.
    """
    if spinup_end_date is not None:
        weather = fetch_weather_for_range(
            hex_loc.latitude, hex_loc.longitude,
            spinup_start_date, spinup_end_date,
        )
    else:
        weather = fetch_recent_weather(hex_loc.latitude, hex_loc.longitude)

    # No-fuel check based on elevation. If the hex is open water or
    # above the tree line, the model would extrapolate way outside its
    # training distribution AND the answer would be wrong on physical
    # grounds (no vegetation to burn). Skip the model entirely and
    # return a synthetic no-fuel hex.
    no_fuel = classify_no_fuel(weather.elevation)
    if no_fuel is not None:
        return HexPrediction(
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
            # all. The map UI excludes these from the rectangle-average
            # reliability calculation.
            domain_confidence=100.0,
            model_agreement=100.0,
            overall_reliability=100.0,
            in_distribution=True,
            no_fuel_reason=no_fuel,
            elevation=weather.elevation,
        ), weather.source

    try:
        fwi_state = spin_up_fwi(weather.history, latitude=hex_loc.latitude)
    except ValueError as e:
        log.warning(f'FWI calc failed for {hex_loc.h3_index}: {e}')
        return None, weather.source

    last_month = weather.history[-1].month
    # Use per-hex region (auto-detected by the frontend) when set, else
    # fall back to the request-level region. This lets a single
    # rectangle spanning Spain and Algeria get the correct region
    # feature for each hex without the user having to split the
    # selection.
    effective_region = hex_loc.region if hex_loc.region else req_region
    features = _build_features(fwi_state, effective_region, last_month)

    try:
        X = registry.transform_input(features)
    except Exception as e:
        log.error(f'Feature transform failed for {hex_loc.h3_index}: {e}')
        return None, weather.source

    probs = {}
    try:
        for name, model in registry.models.items():
            probs[name] = float(model.predict_proba(X)[0, 1])
    except Exception as e:
        log.error(f'Model prediction failed for {hex_loc.h3_index}: {e}')
        return None, weather.source

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

    return HexPrediction(
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
    ), weather.source


def _tally_source(source: str, api_calls: int, cache_hits: int, fallback_count: int):
    """Update the running tally based on a hex's weather source."""
    if source == 'open-meteo' or source == 'open-meteo-forecast':
        api_calls += 1
    elif source == 'cache':
        cache_hits += 1
    elif source == 'fallback':
        fallback_count += 1
    return api_calls, cache_hits, fallback_count


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

    This is the non-streaming variant - blocks until every hex is done
    and returns one big payload. Kept for backward compatibility and
    for non-browser API clients. The browser uses /predict_region_stream
    so it can show progress and cancel.
    """
    spinup_start_date, spinup_end_date = _resolve_date_window(req)

    api_calls = 0
    cache_hits = 0
    fallback_count = 0
    predictions: List[HexPrediction] = []

    for hex_loc in req.hexes:
        prediction, source = _predict_one_hex(
            hex_loc, registry, req.region,
            spinup_start_date, spinup_end_date,
        )
        api_calls, cache_hits, fallback_count = _tally_source(
            source, api_calls, cache_hits, fallback_count,
        )
        if prediction is not None:
            predictions.append(prediction)

    return RegionPredictionResponse(
        predictions=predictions,
        total_api_calls=api_calls,
        cache_hits=cache_hits,
        fallback_count=fallback_count,
    )


@router.websocket('/predict_region_stream')
async def predict_region_stream(websocket: WebSocket):
    """
    Streaming variant of /predict_region. Emits per-hex progress so the
    frontend can render a real progress bar, and stops processing the
    moment the client closes the socket (so cancelling a selection
    actually cancels the work, instead of letting the server burn CPU
    on hexes whose results will never be shown).

    Protocol (all messages are JSON objects with a `type` field):

        client -> server:
            One frame containing a RegionPredictionRequest payload.

        server -> client (in order):
            {type: 'started',  total: N}
            {type: 'progress', completed: i, total: N, prediction: {...}|null}
              (one per hex; null when that hex failed at any step)
            {type: 'done',     total_api_calls, cache_hits, fallback_count}
        or, on error:
            {type: 'error',    error: '...'}

    Cancellation: closing the socket between any two `progress` frames
    is the cancellation signal. The loop checks `client_state` before
    each hex and aborts cleanly when the client disconnects. There's no
    explicit cancel message because closing is unambiguous and idiomatic
    for WebSockets.
    """
    await websocket.accept()

    # Registry isn't injectable via Depends on a WebSocket endpoint the
    # same way it is for HTTP routes, so we pull it off app.state
    # directly. Same mechanism, just without the FastAPI sugar.
    registry = getattr(websocket.app.state, 'registry', None)
    if registry is None:
        await websocket.send_json({'type': 'error', 'error': 'Models not loaded yet'})
        await websocket.close()
        return

    try:
        raw = await websocket.receive_json()
        try:
            req = RegionPredictionRequest(**raw)
        except Exception as e:
            await websocket.send_json({'type': 'error', 'error': f'Invalid request: {e}'})
            await websocket.close()
            return

        # Validate the date window. HTTPException from _resolve_date_window
        # carries a friendly message; surface it as a WebSocket error
        # frame rather than letting it bubble up as a 500.
        try:
            spinup_start_date, spinup_end_date = _resolve_date_window(req)
        except HTTPException as e:
            await websocket.send_json({'type': 'error', 'error': e.detail})
            await websocket.close()
            return

        total = len(req.hexes)
        await websocket.send_json({'type': 'started', 'total': total})

        api_calls = 0
        cache_hits = 0
        fallback_count = 0

        for idx, hex_loc in enumerate(req.hexes):
            # Cancellation check. If the client closed the socket while
            # we were busy with the previous hex, stop here rather than
            # continuing to fetch weather and run models for results
            # nobody will see.
            if websocket.client_state != WebSocketState.CONNECTED:
                log.info(f'Prediction stream cancelled after {idx} / {total} hexes')
                return

            prediction, source = _predict_one_hex(
                hex_loc, registry, req.region,
                spinup_start_date, spinup_end_date,
            )
            api_calls, cache_hits, fallback_count = _tally_source(
                source, api_calls, cache_hits, fallback_count,
            )

            try:
                await websocket.send_json({
                    'type': 'progress',
                    'completed': idx + 1,
                    'total': total,
                    'prediction': prediction.model_dump() if prediction else None,
                })
            except (WebSocketDisconnect, RuntimeError) as e:
                # send_json fails when the client has closed - the same
                # cancellation case as the check above, just caught
                # from the send side.
                log.info(f'Prediction stream client gone at hex {idx + 1}: {e}')
                return

            # Yield to the event loop briefly. send_json already yields,
            # but an explicit sleep(0) makes the cancellation check
            # above more responsive and lets other clients get a turn.
            await asyncio.sleep(0)

        # All hexes done - send the summary so the client can render
        # the source breakdown and reliability badge.
        try:
            await websocket.send_json({
                'type': 'done',
                'total_api_calls': api_calls,
                'cache_hits': cache_hits,
                'fallback_count': fallback_count,
            })
        except (WebSocketDisconnect, RuntimeError):
            pass

    except WebSocketDisconnect:
        log.info('Prediction stream disconnected by client')
    except Exception as e:
        log.exception(f'Prediction stream error: {e}')
        try:
            await websocket.send_json({'type': 'error', 'error': str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


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

        # Frame pacing. We scale the inter-frame sleep by the H3
        # resolution so bigger hexes (lower res) animate slower. The
        # physical reasoning: a res-5 cell is ~25 km across, a res-8
        # cell is ~1.5 km. If we stream at the same 20 fps for both,
        # the res-5 simulation visually races across the map at an
        # implausibly fast speed for a real wildfire. Halving the
        # frame rate per resolution step keeps the apparent spread
        # rate roughly constant in physical km/hour.
        #
        # Calibration baseline: res 8 -> 0.05 s (20 fps, original).
        # Each step down doubles the per-frame sleep, capped at 0.5 s
        # so the user never waits more than half a second between
        # frames even at very low resolutions.
        if config_msg.hex_resolution is not None:
            base_sleep = 0.05 * (2 ** max(0, 8 - config_msg.hex_resolution))
            frame_sleep = min(0.5, base_sleep)
        else:
            frame_sleep = 0.05

        # Stream frames at the resolution-scaled rate
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
            await asyncio.sleep(frame_sleep)

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
