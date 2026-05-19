"""
Open-Meteo weather client.

Fetches the last 7 days of daily weather (temp, humidity, wind, rain) for a
given latitude/longitude. Used to compute FWI System components for arbitrary
map locations.

Open-Meteo (https://open-meteo.com) is free, no API key needed, and has a
generous rate limit (10,000 requests/day). For our use case where the user
might select 200 hexagons in a rectangle, we batch nearby coordinates and
cache aggressively.

Rate limiting strategy:
    - In-memory LRU cache keyed by (rounded lat, rounded lon, today's date)
    - Coordinates rounded to 0.1 degrees (~11 km) before cache lookup, since
      weather doesn't vary meaningfully at finer resolution and Open-Meteo
      itself has 9 km resolution
    - This means a 200-hexagon rectangle in a small region produces 5-20
      unique API calls, not 200

If Open-Meteo is unreachable, we fall back to climatologically reasonable
defaults for the Mediterranean basin in summer (the documented project scope).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import List, Optional, Tuple

import requests

from .fwi_calculator import DailyWeather

log = logging.getLogger(__name__)


# Open-Meteo's archive API gives historical daily values; the forecast API
# gives the same for recent days. We use forecast since "the last 7 days"
# falls inside its window.
ARCHIVE_URL = 'https://archive-api.open-meteo.com/v1/archive'
FORECAST_URL = 'https://api.open-meteo.com/v1/forecast'

# Cache TTL: weather is fetched per-day, so the cache is naturally invalidated
# by date. Within a single day, the same lat/lon + date returns identical data.
COORD_PRECISION = 1   # round to 0.1 degree for cache key
DEFAULT_TIMEOUT = 10  # seconds


def _round_coord(value: float) -> float:
    """Round to COORD_PRECISION decimal places (0.1 degree)."""
    return round(value, COORD_PRECISION)


@dataclass
class WeatherFetchResult:
    """Result of fetching recent weather. Either real data or fallback."""
    history: List[DailyWeather]
    source: str    # 'open-meteo' | 'cache' | 'fallback'
    latitude: float
    longitude: float
    elevation: Optional[float] = None    # metres above sea level from
                                          # Open-Meteo's Copernicus DEM.
                                          # None if not available (e.g.
                                          # the fallback path); negative
                                          # values indicate sea points.


# Module-level cache. Keys are (rounded_lat, rounded_lon, date_string).
# Values are (history, elevation) tuples - elevation cached too so a
# repeat request for the same nearby coordinate gets it without
# refetching. The cache is intentionally small; only a single user
# session's worth of clicks.
_cache: dict[Tuple[float, float, str], Tuple[List[DailyWeather], Optional[float]]] = {}
_CACHE_MAX = 1000


def _cache_get(lat: float, lon: float, date: str
               ) -> Optional[Tuple[List[DailyWeather], Optional[float]]]:
    return _cache.get((_round_coord(lat), _round_coord(lon), date))


def _cache_put(lat: float, lon: float, date: str,
               history: List[DailyWeather],
               elevation: Optional[float]) -> None:
    if len(_cache) >= _CACHE_MAX:
        # Drop the oldest entry (FIFO is fine for this volume)
        _cache.pop(next(iter(_cache)))
    _cache[(_round_coord(lat), _round_coord(lon), date)] = (history, elevation)


# ---------------------------------------------------------------------------
# Climatological fallback
# ---------------------------------------------------------------------------
# If Open-Meteo is unreachable, return values typical of a Mediterranean
# summer day. This is documented as a fallback only — for genuine demos we
# want real data.

def _fallback_history(latitude: float) -> List[DailyWeather]:
    """Return 7 days of typical Mediterranean summer weather."""
    today = datetime.utcnow()
    month = today.month
    return [
        DailyWeather(temperature=30.0, humidity=42.0, wind_speed=14.0,
                     rain=0.0, month=month)
        for _ in range(7)
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fetch_recent_weather(latitude: float, longitude: float,
                         days: int = 7) -> WeatherFetchResult:
    """
    Fetch the last `days` days of daily weather for the given location.

    Returns the data along with a source tag. Never raises, falls back to
    climatological values if the API is unreachable.

    Thin wrapper around fetch_weather_for_range, kept for backward
    compatibility with existing callers that don't need explicit dates.
    """
    today = datetime.utcnow().date()
    start = today - timedelta(days=days)
    return fetch_weather_for_range(latitude, longitude, start, today)


def fetch_weather_for_range(latitude: float, longitude: float,
                            start_date, end_date) -> WeatherFetchResult:
    """
    Fetch daily weather for an arbitrary date range.

    start_date, end_date: datetime.date objects. end_date inclusive.

    Used in three modes:
    - Historical fires: archive endpoint (dates more than ~5 days old)
    - Recent or near-real-time: forecast endpoint past_days parameter
    - Forward forecast: forecast endpoint forecast_days parameter
      (lets users simulate "fire starts in 3 days" against the actual
      forecast weather for that period)

    The routing happens automatically based on how the requested range
    relates to today. Same caching strategy regardless of endpoint.
    Same fallback if Open-Meteo is unreachable.
    """
    cache_key = f'{start_date.isoformat()}__{end_date.isoformat()}'

    cached = _cache_get(latitude, longitude, cache_key)
    if cached is not None:
        cached_history, cached_elev = cached
        return WeatherFetchResult(history=cached_history, source='cache',
                                  latitude=latitude, longitude=longitude,
                                  elevation=cached_elev)

    today = datetime.utcnow().date()
    # Archive endpoint lag is typically 5 days. If end_date is within
    # 5 days of today (or in the future), the archive will be missing
    # the most recent values - the forecast endpoint covers both
    # past_days (recent measurements/reanalysis) and forecast_days
    # (predictions). Threshold of 5 days is conservative.
    ARCHIVE_LAG_DAYS = 5
    use_forecast_endpoint = end_date >= (today - timedelta(days=ARCHIVE_LAG_DAYS))

    try:
        if use_forecast_endpoint:
            # Compute how many days of past data and how many days into
            # the future we need. Open-Meteo's forecast endpoint accepts
            # past_days (0-92) and forecast_days (1-16); we cover any
            # date range by setting both.
            past_days = max(0, (today - start_date).days)
            forecast_days = max(1, (end_date - today).days + 1)
            # Sanity-clamp to the endpoint's documented limits
            past_days = min(past_days, 92)
            forecast_days = min(forecast_days, 16)

            url = FORECAST_URL
            params = {
                'latitude': latitude,
                'longitude': longitude,
                'past_days': past_days,
                'forecast_days': forecast_days,
                'daily': ','.join([
                    'temperature_2m_max',
                    'temperature_2m_mean',
                    'relative_humidity_2m_mean',
                    'wind_speed_10m_max',
                    'precipitation_sum',
                ]),
                'timezone': 'auto',
                'wind_speed_unit': 'kmh',
            }
        else:
            # Pure historical range, archive endpoint is the right one
            url = ARCHIVE_URL
            params = {
                'latitude': latitude,
                'longitude': longitude,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'daily': ','.join([
                    'temperature_2m_max',
                    'temperature_2m_mean',
                    'relative_humidity_2m_mean',
                    'wind_speed_10m_max',
                    'precipitation_sum',
                ]),
                'timezone': 'auto',
                'wind_speed_unit': 'kmh',
            }

        response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        # Elevation comes as a top-level scalar in the response. Open-Meteo
        # returns it from the Copernicus DEM (90m resolution). It can be
        # negative (sea points) or large positive (high terrain).
        elevation = data.get('elevation')
        if elevation is not None:
            try:
                elevation = float(elevation)
            except (TypeError, ValueError):
                elevation = None

        history = _parse_open_meteo_response(data)
        if not history:
            log.warning(f'Open-Meteo returned empty data for ({latitude}, {longitude})')
            return WeatherFetchResult(
                history=_fallback_history(latitude), source='fallback',
                latitude=latitude, longitude=longitude, elevation=elevation,
            )

        _cache_put(latitude, longitude, cache_key, history, elevation)
        source_label = 'open-meteo-forecast' if use_forecast_endpoint else 'open-meteo'
        return WeatherFetchResult(history=history, source=source_label,
                                  latitude=latitude, longitude=longitude,
                                  elevation=elevation)

    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning(f'Open-Meteo fetch failed for ({latitude}, {longitude}): {e}')
        return WeatherFetchResult(
            history=_fallback_history(latitude), source='fallback',
            latitude=latitude, longitude=longitude, elevation=None,
        )


def _parse_open_meteo_response(data: dict) -> List[DailyWeather]:
    """
    Open-Meteo returns parallel arrays under 'daily'. Convert to our
    DailyWeather dataclass.
    """
    daily = data.get('daily', {})
    times = daily.get('time', [])
    temps_max = daily.get('temperature_2m_max', [])
    temps_mean = daily.get('temperature_2m_mean', [])
    humidities = daily.get('relative_humidity_2m_mean', [])
    winds = daily.get('wind_speed_10m_max', [])
    rains = daily.get('precipitation_sum', [])

    history: List[DailyWeather] = []
    for i, date_str in enumerate(times):
        # Use mean temp where available (closer to noon Van Wagner convention)
        # and max temp as fallback.
        temp = (temps_mean[i] if i < len(temps_mean) and temps_mean[i] is not None
                else temps_max[i] if i < len(temps_max) and temps_max[i] is not None
                else 25.0)
        rh = humidities[i] if i < len(humidities) and humidities[i] is not None else 50.0
        wind = winds[i] if i < len(winds) and winds[i] is not None else 10.0
        rain = rains[i] if i < len(rains) and rains[i] is not None else 0.0

        try:
            month = datetime.fromisoformat(date_str).month
        except ValueError:
            month = datetime.utcnow().month

        history.append(DailyWeather(
            temperature=float(temp),
            humidity=float(rh),
            wind_speed=float(wind),
            rain=float(rain),
            month=month,
        ))

    return history


def fetch_recent_weather_batch(coords: List[Tuple[float, float]]
                               ) -> List[WeatherFetchResult]:
    """
    Fetch weather for multiple locations.

    Coordinates are rounded to 0.1 degrees for cache lookup, so requests for
    nearby points share the same API call. Order of results matches order of
    inputs.
    """
    results: List[WeatherFetchResult] = []
    for lat, lon in coords:
        results.append(fetch_recent_weather(lat, lon))
    return results


# ---------------------------------------------------------------------------
# No-fuel classification
# ---------------------------------------------------------------------------
# Some terrain on Earth cannot sustain a forest fire because there is
# no vegetation. The two we can detect reliably from elevation alone
# are:
#
#   * Open ocean and large inland seas: elevation is at or below sea level
#     in the Copernicus DEM. (Land just inland from the coast still has
#     positive elevation, so the false-positive rate is low.)
#
#   * Above the tree line in Mediterranean / temperate latitudes. The
#     tree line is not a single number - it varies by mountain range:
#
#       Pyrenees (Spain/France):   ~2200-2400m
#       Sierra Nevada (Andalusia): ~2200-2400m
#       Apennines (Italy):         ~1900-2200m
#       Atlas Mountains (Algeria): ~2500-2700m (Atlas cedar reaches
#                                  the highest of any Mediterranean
#                                  conifer)
#
#     Above the tree line you still get alpine grass and scrub which
#     CAN burn, but it's sparse, low-fuel, and well outside the ML
#     model's training distribution (UCI Portugal + Algerian datasets
#     are drawn from areas mostly below 1500m).
#
#     We use 3000m as a documented threshold. This is above the active
#     forest tree-line in every Mediterranean range, while still
#     catching the highest snow/rock zones (Mulhacen 3479m in Sierra
#     Nevada, Pico Aneto 3404m in the Pyrenees, Toubkal 4167m in the
#     High Atlas). A threshold of 3500m or higher would be above
#     almost every Iberian peak and effectively disable the check.
#
# What we CANNOT reliably detect from elevation alone:
#
#   * Small inland lakes (the DEM's land-sea mask doesn't catch them).
#   * Rivers (sub-grid-cell features).
#   * Glaciers below 3000m (rare in our scope but possible).
#   * Bare rock outcrops below the tree line.
#   * Recently burned areas with no current fuel load.
#
# For these we'd need a separate land-cover dataset. Not in scope.


# The high-altitude threshold above which we flag a hex as no-fuel.
# Set to 3000m: above the forest tree line in every Mediterranean
# range, below all but the very highest summits. See the comment
# block above for the reasoning.
ELEVATION_TREE_LINE_M = 3000.0
ELEVATION_SEA_LEVEL_M = 0.0


def classify_no_fuel(elevation: Optional[float]) -> Optional[str]:
    """
    Decide if a hex's terrain cannot sustain a forest fire.

    Returns a string reason ('water' or 'high_altitude') if the hex is
    no-fuel, or None if it's normal land that can burn.

    None elevation (e.g. from the fallback path when Open-Meteo is
    unreachable) returns None - we err on the side of letting the
    model run rather than blocking it. The user can interpret the
    weather_source field in the response to see if the call was a
    real lookup.
    """
    if elevation is None:
        return None
    if elevation <= ELEVATION_SEA_LEVEL_M:
        return 'water'
    if elevation >= ELEVATION_TREE_LINE_M:
        return 'high_altitude'
    return None
