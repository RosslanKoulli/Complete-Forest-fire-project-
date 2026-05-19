"""
Van Wagner FWI System calculator.

Implements the recursive equations of the Canadian Forest Fire Weather Index
System to compute FFMC, DMC, DC, ISI from a sequence of daily weather
observations.

The system is RECURSIVE: each day's index value depends on the previous day's
value plus today's weather. There is no formula that computes FFMC from a single
day's weather. To get a meaningful value, we feed in several days of recent
weather starting from a documented "spin-up" initial state.

Standard practice is to start from "fire season start" values (FFMC=85, DMC=6,
DC=15) and walk forward day by day. For our use case (predict on demand for an
arbitrary location), we fetch the last 7 days of weather from Open-Meteo and
spin up from those. Seven days is enough for FFMC and DMC to converge to their
true values; DC takes longer but is dominated by recent rainfall in any case.

"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence


# ---------------------------------------------------------------------------
# Day-length factors
# ---------------------------------------------------------------------------
# DMC's drying rate depends on day length, which depends on month and latitude
# zone. Van Wagner provides three sets of factors: northern (lat > 30), tropical
# (-30 < lat < 30), and southern (lat < -30). For the Mediterranean basin this
# project covers, the northern set is correct.

DAY_LENGTH_NORTHERN = [
    6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0,
]   # months 1..12 (January index 0)

DAY_LENGTH_TROPICAL = [9.0] * 12  # constant near equator

DAY_LENGTH_SOUTHERN = [
    11.5, 10.5, 9.2, 7.9, 6.8, 6.2, 6.5, 7.4, 8.7, 10.0, 11.2, 11.8,
]

# DC's drying rate depends on month (different factors than DMC).
DC_LF_NORTHERN = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
DC_LF_TROPICAL = [1.4] * 12
DC_LF_SOUTHERN = [6.4, 5.0, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8]


def _day_length(month: int, lat: float) -> float:
    """Return day-length factor for DMC drying rate."""
    table = (DAY_LENGTH_NORTHERN if lat > 30
             else DAY_LENGTH_SOUTHERN if lat < -30
             else DAY_LENGTH_TROPICAL)
    return table[month - 1]


def _dc_day_length(month: int, lat: float) -> float:
    """Return day-length factor for DC drying rate."""
    table = (DC_LF_NORTHERN if lat > 30
             else DC_LF_SOUTHERN if lat < -30
             else DC_LF_TROPICAL)
    return table[month - 1]


# ---------------------------------------------------------------------------
# FFMC equation (Van Wagner 1987, eq. 2-7)
# ---------------------------------------------------------------------------
# FFMC tracks moisture in surface litter. Two-phase model: rain wetting,
# then drying or wetting toward equilibrium with the atmosphere.

def update_ffmc(prev_ffmc: float, temp: float, rh: float,
                wind: float, rain: float) -> float:
    """
    Compute today's FFMC given yesterday's FFMC and today's weather.

    temp:   noon temperature in degrees Celsius
    rh:     noon relative humidity in percent (0-100)
    wind:   noon wind speed in km/h at 10m height
    rain:   24-hour rainfall in mm (since previous noon)

    Returns FFMC on the 0-101 scale.
    """
    rh = min(100.0, max(0.0, rh))   # clamp to valid range

    # Convert previous FFMC to moisture content (Van Wagner eq. 1).
    # The 147.2 constant is a Fine Fuel Moisture scaling factor.
    mo = 147.2 * (101.0 - prev_ffmc) / (59.5 + prev_ffmc)

    # Rain wetting: only if rainfall exceeds 0.5 mm (the canopy interception
    # threshold). Otherwise rain is intercepted by canopy and doesn't reach
    # the litter layer.
    if rain > 0.5:
        rf = rain - 0.5
        # Van Wagner eq. 3a: standard rain effect
        if mo <= 150.0:
            mo = (mo
                  + 42.5 * rf * math.exp(-100.0 / (251.0 - mo))
                  * (1.0 - math.exp(-6.93 / rf)))
        else:
            # Eq. 3b: wet fuels gain less moisture (saturation effect)
            mo = (mo
                  + 42.5 * rf * math.exp(-100.0 / (251.0 - mo))
                  * (1.0 - math.exp(-6.93 / rf))
                  + 0.0015 * (mo - 150.0) ** 2 * math.sqrt(rf))
        mo = min(mo, 250.0)   # absolute upper bound

    # Equilibrium moisture content for drying (Ed) and wetting (Ew).
    # Van Wagner eq. 4 and 5.
    ed = (0.942 * rh ** 0.679
          + 11.0 * math.exp((rh - 100.0) / 10.0)
          + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh)))

    ew = (0.618 * rh ** 0.753
          + 10.0 * math.exp((rh - 100.0) / 10.0)
          + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh)))

    # Drying or wetting toward equilibrium.
    if mo > ed:
        # Drying: move from mo toward ed
        ko = (0.424 * (1.0 - (rh / 100.0) ** 1.7)
              + 0.0694 * math.sqrt(wind) * (1.0 - (rh / 100.0) ** 8))
        kd = ko * 0.581 * math.exp(0.0365 * temp)
        m = ed + (mo - ed) * 10.0 ** (-kd)
    elif mo < ew:
        # Wetting: move from mo toward ew
        kl = (0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7)
              + 0.0694 * math.sqrt(wind) * (1.0 - ((100.0 - rh) / 100.0) ** 8))
        kw = kl * 0.581 * math.exp(0.0365 * temp)
        m = ew - (ew - mo) * 10.0 ** (-kw)
    else:
        # Already at equilibrium
        m = mo

    # Convert moisture content back to FFMC scale (eq. 6 inverted).
    new_ffmc = 59.5 * (250.0 - m) / (147.2 + m)
    return min(101.0, max(0.0, new_ffmc))


# ---------------------------------------------------------------------------
# DMC equation (Van Wagner 1987, eq. 11-17)
# ---------------------------------------------------------------------------
# DMC tracks moisture in the loose organic layer below surface litter.
# Slower-responding than FFMC. Day-length affects drying rate.

def update_dmc(prev_dmc: float, temp: float, rh: float, rain: float,
               month: int, latitude: float) -> float:
    """
    Compute today's DMC given yesterday's DMC and today's weather.

    Same units as update_ffmc, plus month (1-12) and latitude (degrees).
    """
    rh = min(100.0, max(0.0, rh))
    if temp < -1.1:
        temp = -1.1   # Van Wagner's lower bound on temperature for DMC drying

    # Drying rate K depends on temperature, humidity, and day length.
    le = _day_length(month, latitude)
    k = 1.894 * (temp + 1.1) * (100.0 - rh) * le * 1e-6

    # Rain effect (only if rain > 1.5 mm, the canopy threshold for DMC).
    if rain > 1.5:
        re = 0.92 * rain - 1.27   # effective rainfall reaching the duff
        # Wetting depends on current DMC (drier fuels absorb more).
        if prev_dmc <= 33.0:
            b = 100.0 / (0.5 + 0.3 * prev_dmc)
        elif prev_dmc <= 65.0:
            b = 14.0 - 1.3 * math.log(prev_dmc)
        else:
            b = 6.2 * math.log(prev_dmc) - 17.2
        # Convert DMC to moisture, add rain, convert back.
        mo = 20.0 + math.exp(5.6348 - prev_dmc / 43.43)
        mr = mo + 1000.0 * re / (48.77 + b * re)
        wet_dmc = 244.72 - 43.43 * math.log(mr - 20.0)
        wet_dmc = max(0.0, wet_dmc)
    else:
        wet_dmc = prev_dmc

    return max(0.0, wet_dmc + 100.0 * k)


# ---------------------------------------------------------------------------
# DC equation (Van Wagner 1987, eq. 18-22)
# ---------------------------------------------------------------------------
# DC tracks moisture in the deep compact organic layer. Slowest-responding
# index. Tracks long-term drought.

def update_dc(prev_dc: float, temp: float, rain: float,
              month: int, latitude: float) -> float:
    """
    Compute today's DC given yesterday's DC and today's weather.

    Note that DC does not depend on humidity directly — it represents
    such a deep layer that humidity at the surface is irrelevant.
    """
    if temp < -2.8:
        temp = -2.8

    # Potential evapotranspiration: drying rate depends on temperature and
    # day length factor for the month.
    lf = _dc_day_length(month, latitude)
    pe = max(0.0, 0.36 * (temp + 2.8) + lf) / 2.0

    # Rain effect (only if rain > 2.8 mm).
    if rain > 2.8:
        rd = 0.83 * rain - 1.27
        # Convert DC to moisture equivalent, add rain, convert back.
        qo = 800.0 * math.exp(-prev_dc / 400.0)
        qr = qo + 3.937 * rd
        wet_dc = 400.0 * math.log(800.0 / qr)
        wet_dc = max(0.0, wet_dc)
    else:
        wet_dc = prev_dc

    return wet_dc + pe


# ---------------------------------------------------------------------------
# ISI (eq. 24, 25): Initial Spread Index = FFMC + wind
# ---------------------------------------------------------------------------

def calc_isi(ffmc: float, wind: float) -> float:
    """ISI is non-recursive, depends only on today's FFMC and wind."""
    fw = math.exp(0.05039 * wind)
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    ff = 91.9 * math.exp(-0.1386 * m) * (1.0 + m ** 5.31 / 4.93e7)
    return 0.208 * fw * ff


# ---------------------------------------------------------------------------
# Spin-up: compute current FWI components from a sequence of daily weather
# ---------------------------------------------------------------------------

@dataclass
class DailyWeather:
    """One day of weather. All fields required for spin-up."""
    temperature: float       # noon temp in C
    humidity: float          # noon RH in %
    wind_speed: float        # 10m wind in km/h
    rain: float              # 24h precipitation in mm
    month: int               # 1-12


@dataclass
class FWIState:
    """Computed FWI components plus the inputs that drove them."""
    ffmc: float
    dmc: float
    dc: float
    isi: float
    temperature: float
    humidity: float
    wind_speed: float
    rain: float


def spin_up_fwi(history: Sequence[DailyWeather], latitude: float,
                initial_ffmc: float = 85.0,
                initial_dmc: float = 6.0,
                initial_dc: float = 15.0) -> FWIState:
    """
    Walk through a sequence of daily weather, returning the final-day FWI state.

    history: oldest day first, most recent day last. At least one day required;
             7+ days recommended for FFMC and DMC to converge.

    initial_*: Van Wagner's standard "fire season start" values. With 7+ days
               of spin-up, the influence of these initial values is small for
               FFMC and DMC; DC is more persistent but also dominated by any
               recent rainfall.
    """
    if not history:
        raise ValueError('Need at least one day of weather to compute FWI')

    ffmc = initial_ffmc
    dmc = initial_dmc
    dc = initial_dc

    for day in history:
        ffmc = update_ffmc(ffmc, day.temperature, day.humidity,
                           day.wind_speed, day.rain)
        dmc = update_dmc(dmc, day.temperature, day.humidity, day.rain,
                         day.month, latitude)
        dc = update_dc(dc, day.temperature, day.rain, day.month, latitude)

    last = history[-1]
    isi = calc_isi(ffmc, last.wind_speed)

    return FWIState(
        ffmc=ffmc, dmc=dmc, dc=dc, isi=isi,
        temperature=last.temperature, humidity=last.humidity,
        wind_speed=last.wind_speed, rain=last.rain,
    )
