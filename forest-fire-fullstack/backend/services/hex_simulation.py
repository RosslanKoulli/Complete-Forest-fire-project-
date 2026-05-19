"""
Hex-based fire spread simulator.

Different from the grid-based cellular_automata.py module: this operates on a
graph of H3 hexagonal cells, where each hex has up to 6 neighbours rather than
the 8 of a square Moore neighbourhood. This matches the geographic reality of
the user's selection: when they draw a rectangle on the map and we tessellate
it with H3 hexagons, fire should spread from neighbour to neighbour through
those actual hexagons, not on a synthetic grid laid over them.

State machine (same conceptual states as the grid version):
    UNBURNT  -> can be ignited by burning neighbours
    BURNING  -> spreading fire; transitions to BURNT after burn_duration steps
    BURNT    -> spent; cannot reignite
    FIREBREAK -> blocks transmission; not used in current map UI but supported

Spread probability per neighbour per step:
    p_spread = base_rate * fire_modifier * wind_modifier * moisture_modifier

where
    base_rate = the global base spread probability (user-configurable)
    fire_modifier = function of the source hex's fire probability (high-risk
                    hexes propagate more aggressively)
    wind_modifier = directional bias from wind (1.0 in line with wind, 0.5
                    against, calibrated like Trunfio (2004) but on hex topology)
    moisture_modifier = 1 - vegetation_moisture (drier = spreads more)

This formulation is documented as a simplification of the Trunfio (2004) CA.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple


class HexState(IntEnum):
    UNBURNT = 0
    BURNING = 1
    BURNT = 2
    FIREBREAK = 3


@dataclass
class HexCell:
    """One hex in the simulation."""
    h3_index: str
    latitude: float
    longitude: float
    state: HexState = HexState.UNBURNT
    fire_probability: float = 0.0    # per-hex probability from the ML models (0-1)
    burn_steps_remaining: int = 0


@dataclass
class HexFrame:
    """One time step of the simulation."""
    step: int
    states: Dict[str, int]    # h3_index -> state code
    burning_count: int
    burnt_count: int
    unburnt_count: int

    # Real-time fields (populated when running in time-window mode;
    # zeroed otherwise so consumers can ignore them safely).
    elapsed_hours: float = 0.0
    current_day: int = 0      # 1-indexed; day 1 is the first day of the window


@dataclass
class HexSimulationConfig:
    """User-configurable parameters."""
    base_spread_prob: float = 0.45
    wind_speed: float = 8.0           # km/h
    wind_direction: float = 45.0      # degrees, 0=N, 90=E
    vegetation_moisture: float = 0.3  # 0-1
    burn_duration: int = 3            # steps a hex stays BURNING
    max_steps: int = 100
    seed: Optional[int] = None        # for reproducibility

    # ----- Time-window mode -----
    # If `time_window_days` is set, the simulation runs over a fixed
    # real-world duration rather than continuing until burnout. Each
    # simulation step represents `hours_per_step` real hours; when the
    # cumulative simulated time reaches the window length, the
    # simulation terminates whether the fire is out or not.
    #
    # The default (12 hours per step) is a calibration choice: a
    # typical Mediterranean wildfire passes through a hex of a few
    # square km in roughly half a day under moderate spread conditions.
    # Without ground-truth spread-rate data we cannot do better than
    # this default; the value is exposed so callers can override it.
    time_window_days: Optional[float] = None
    hours_per_step: float = 12.0


class HexFireSimulator:
    """
    Run a fire spread simulation on a graph of H3 hexagons.

    Usage:
        sim = HexFireSimulator(hexes, neighbours, config)
        sim.ignite(['hex_id_1'])
        for frame in sim.run():
            yield frame
    """

    def __init__(self,
                 cells: Dict[str, HexCell],
                 neighbours: Dict[str, List[str]],
                 config: HexSimulationConfig):
        """
        cells: dict mapping h3 index to HexCell. Initial state is whatever's
               in each cell (typically all UNBURNT before user ignites).
        neighbours: dict mapping h3 index to list of neighbour h3 indices.
                    Only includes neighbours that are also in `cells` — hexes
                    on the rectangle boundary will have fewer than 6 neighbours.
        config: simulation parameters.
        """
        self.cells = cells
        self.neighbours = neighbours
        self.config = config
        self.step = 0

        # Use a dedicated RNG so we don't perturb global state.
        self.rng = random.Random(config.seed)

    # ----------------------- Public API -----------------------

    def ignite(self, hex_ids: List[str]) -> None:
        """Set the given hexes to BURNING. Caller chooses ignition points."""
        for hid in hex_ids:
            if hid in self.cells:
                self.cells[hid].state = HexState.BURNING
                self.cells[hid].burn_steps_remaining = self.config.burn_duration

    def run(self):
        """
        Generator yielding one HexFrame per simulation step.

        Stops when no hex is BURNING (fire is out), max_steps reached, OR
        (in time-window mode) when the cumulative elapsed time exceeds the
        configured window length. Returning early on window expiry is the
        whole point of time-window mode: the user wants to know "what
        burned in the first N days", not "how long until everything burns".
        """
        # First frame: the initial state after ignition
        yield self._snapshot()

        # Compute the time-window cap once. None means "no cap".
        max_hours = (self.config.time_window_days * 24.0
                     if self.config.time_window_days is not None
                     else None)

        for _ in range(self.config.max_steps):
            burning = [c for c in self.cells.values()
                       if c.state == HexState.BURNING]
            if not burning:
                return   # fire is out

            # Time-window check: stop if we've exceeded the window.
            # We check BEFORE stepping so the user gets exactly N days of
            # spread and no more.
            if max_hours is not None:
                elapsed = (self.step + 1) * self.config.hours_per_step
                if elapsed > max_hours:
                    return

            self._step()
            self.step += 1
            yield self._snapshot()

    # ----------------------- Internals -----------------------

    def _step(self) -> None:
        """
        Advance the simulation by one step.

        Two phases per step:
        1. Burning hexes try to ignite each unburnt neighbour
        2. Burning hexes age toward BURNT

        Phases are separated so that hexes ignited THIS step can't propagate
        until next step, which is the standard CA invariant.
        """
        new_ignitions: Set[str] = set()

        for hid, cell in self.cells.items():
            if cell.state != HexState.BURNING:
                continue
            for nid in self.neighbours.get(hid, []):
                if nid in new_ignitions:
                    continue
                neighbour = self.cells.get(nid)
                if neighbour is None or neighbour.state != HexState.UNBURNT:
                    continue
                if self._attempt_ignition(cell, neighbour):
                    new_ignitions.add(nid)

        # Apply ignitions
        for nid in new_ignitions:
            self.cells[nid].state = HexState.BURNING
            self.cells[nid].burn_steps_remaining = self.config.burn_duration

        # Age burning hexes
        for cell in self.cells.values():
            if cell.state == HexState.BURNING:
                cell.burn_steps_remaining -= 1
                if cell.burn_steps_remaining <= 0:
                    cell.state = HexState.BURNT

    def _attempt_ignition(self, source: HexCell, target: HexCell) -> bool:
        """
        Roll the dice on whether `source` ignites `target` this step.

        Probability formula:
            p = base * fire_factor * wind_factor * moisture_factor

        Each factor is in [0, 2] roughly, so p stays in a sensible range.
        """
        base = self.config.base_spread_prob

        # Fire-probability factor: hexes in high-risk areas propagate more.
        # Clamp to [0.5, 1.5] so even low-probability hexes can spread some.
        fire_factor = 0.5 + target.fire_probability

        # Wind factor: cosine of angle between wind direction and the
        # geographic bearing from source to target. With wind = 1.5 in line,
        # 0.5 against, 1.0 perpendicular.
        wind_factor = self._wind_factor(source, target)

        # Moisture factor: drier vegetation propagates more.
        moisture_factor = max(0.05, 1.0 - self.config.vegetation_moisture)

        p = base * fire_factor * wind_factor * moisture_factor
        p = min(0.95, max(0.0, p))   # clamp

        return self.rng.random() < p

    def _wind_factor(self, source: HexCell, target: HexCell) -> float:
        """
        Compute the directional wind multiplier for spread from source to target.

        Returns 1.5 if spread direction matches wind direction exactly,
        0.5 if opposite, 1.0 if perpendicular.
        """
        # Bearing from source to target in degrees (0 = north, 90 = east).
        dlon = math.radians(target.longitude - source.longitude)
        lat1 = math.radians(source.latitude)
        lat2 = math.radians(target.latitude)
        x = math.sin(dlon) * math.cos(lat2)
        y = (math.cos(lat1) * math.sin(lat2)
             - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
        bearing = math.degrees(math.atan2(x, y))
        # Normalise to [0, 360)
        bearing = (bearing + 360.0) % 360.0

        # Angular difference between bearing and wind direction.
        # Wind direction convention: "wind from N at 0 degrees" means wind
        # blowing from north toward south, so fire is pushed south. We want
        # spread south to be enhanced, so we compare bearing to (wind_dir + 180).
        push_direction = (self.config.wind_direction + 180.0) % 360.0
        angle = abs(bearing - push_direction)
        if angle > 180.0:
            angle = 360.0 - angle

        # Map angle [0, 180] to factor [1.5, 0.5] linearly.
        factor = 1.5 - (angle / 180.0)
        return factor

    def _snapshot(self) -> HexFrame:
        """Build the current HexFrame for streaming."""
        states = {hid: int(c.state) for hid, c in self.cells.items()}
        burning = sum(1 for s in states.values() if s == HexState.BURNING)
        burnt = sum(1 for s in states.values() if s == HexState.BURNT)
        unburnt = sum(1 for s in states.values() if s == HexState.UNBURNT)

        # Real-time tagging. In step-mode (time_window_days=None) these
        # stay at 0 and clients can ignore them.
        elapsed = self.step * self.config.hours_per_step
        # 1-indexed day-of-window. Step 0 -> day 1, then day rolls over
        # every 24 hours of simulated time.
        day_idx = int(elapsed // 24) + 1 if self.config.time_window_days else 0

        return HexFrame(
            step=self.step,
            states=states,
            burning_count=burning,
            burnt_count=burnt,
            unburnt_count=unburnt,
            elapsed_hours=elapsed,
            current_day=day_idx,
        )
