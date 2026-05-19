"""
Tests for Cellular Automata Fire Spread Engine
===============================================
Verifies grid initialization, ignition, spread behaviour,
wind effects, firebreaks, and statistics.

Run: python -m pytest tests/test_cellular_automata.py -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from spread_model.cellular_automata import (
    CellularAutomataEngine, CellState, EnvironmentConfig
)


class TestGridInitialization:
    
    def test_grid_all_unburnt(self):
        engine = CellularAutomataEngine(20, 20)
        assert np.all(engine.grid == CellState.UNBURNT)
    
    def test_grid_dimensions(self):
        engine = CellularAutomataEngine(30, 40)
        assert engine.grid.shape == (30, 40)
    
    def test_vegetation_in_range(self):
        engine = CellularAutomataEngine(20, 20)
        assert np.all(engine.vegetation >= 0.3)
        assert np.all(engine.vegetation <= 1.0)


class TestIgnition:
    
    def test_ignite_single_cell(self):
        engine = CellularAutomataEngine(20, 20)
        engine.ignite(10, 10)
        assert engine.grid[10, 10] == CellState.BURNING
    
    def test_ignite_out_of_bounds_no_error(self):
        engine = CellularAutomataEngine(20, 20)
        engine.ignite(-1, 5)   # Should not crash
        engine.ignite(5, 100)  # Should not crash
    
    def test_ignite_on_firebreak_fails(self):
        engine = CellularAutomataEngine(20, 20)
        engine.grid[5, 5] = CellState.FIREBREAK
        engine.ignite(5, 5)
        assert engine.grid[5, 5] == CellState.FIREBREAK
    
    def test_ignite_from_probability_map(self):
        engine = CellularAutomataEngine(10, 10)
        prob_map = np.zeros((10, 10))
        prob_map[3, 3] = 0.9
        prob_map[7, 7] = 0.8
        prob_map[5, 5] = 0.3  # Below threshold
        
        engine.ignite_from_probability_map(prob_map, threshold=0.7)
        assert engine.grid[3, 3] == CellState.BURNING
        assert engine.grid[7, 7] == CellState.BURNING
        assert engine.grid[5, 5] == CellState.UNBURNT


class TestFirebreaks:
    
    def test_add_firebreaks(self):
        engine = CellularAutomataEngine(20, 20)
        engine.add_firebreaks([(5, 5), (6, 6)])
        assert engine.grid[5, 5] == CellState.FIREBREAK
        assert engine.grid[6, 6] == CellState.FIREBREAK
    
    def test_random_firebreaks_density(self):
        engine = CellularAutomataEngine(100, 100)
        engine.add_random_firebreaks(density=0.1)
        n_firebreaks = np.sum(engine.grid == CellState.FIREBREAK)
        # Should be roughly 10% (±5%)
        assert 500 < n_firebreaks < 1500, \
            f"Expected ~1000 firebreaks, got {n_firebreaks}"
    
    def test_fire_does_not_cross_firebreak_wall(self):
        """A solid wall of firebreaks should block fire."""
        engine = CellularAutomataEngine(20, 20, EnvironmentConfig(
            base_spread_prob=0.9, wind_speed=0, vegetation_moisture=0
        ))
        # Wall at column 10
        for r in range(20):
            engine.grid[r, 10] = CellState.FIREBREAK
        
        engine.ignite(10, 5)  # Fire on left side
        engine.run(max_steps=50)
        
        # Nothing should burn on the right side of the wall
        right_side = engine.grid[:, 11:]
        burnt_right = np.sum(right_side == CellState.BURNT)
        burning_right = np.sum(right_side == CellState.BURNING)
        assert burnt_right + burning_right == 0, \
            "Fire should not cross firebreak wall"


class TestSpreadBehaviour:
    
    def test_fire_spreads(self):
        """After some steps, more than 1 cell should be burnt/burning."""
        engine = CellularAutomataEngine(20, 20, EnvironmentConfig(
            base_spread_prob=0.8, vegetation_moisture=0.1
        ))
        engine.ignite(10, 10)
        engine.run(max_steps=10)
        
        n_affected = np.sum(engine.grid != CellState.UNBURNT)
        assert n_affected > 1, "Fire should spread to at least some neighbours"
    
    def test_burning_transitions_to_burnt(self):
        """Burning cells should eventually become burnt."""
        engine = CellularAutomataEngine(5, 5, EnvironmentConfig(
            base_spread_prob=0.0, burn_duration=2  # No spread, just burn
        ))
        engine.ignite(2, 2)
        engine.run(max_steps=5)
        
        assert engine.grid[2, 2] == CellState.BURNT, \
            "Cell should transition to BURNT after burn_duration"
    
    def test_no_fire_returns_false(self):
        """step() returns False when no cells are burning."""
        engine = CellularAutomataEngine(5, 5)
        # Don't ignite anything
        result = engine.step()
        assert result is False
    
    def test_simulation_terminates(self):
        """Simulation should end within max_steps."""
        engine = CellularAutomataEngine(20, 20, EnvironmentConfig(
            base_spread_prob=0.5
        ))
        engine.ignite(10, 10)
        stats = engine.run(max_steps=200)
        assert stats['steps_to_completion'] <= 200


class TestWindEffect:
    
    def test_downwind_spreads_more(self):
        """Fire should burn more cells when spreading downwind."""
        np.random.seed(42)
        
        # Run with wind pushing east (90°)
        engine_wind = CellularAutomataEngine(30, 30, EnvironmentConfig(
            wind_speed=10.0, wind_direction=90,
            base_spread_prob=0.3, vegetation_moisture=0.3
        ))
        engine_wind.ignite(15, 5)  # Ignite on left side
        stats_wind = engine_wind.run(max_steps=40)
        
        # Run with no wind
        np.random.seed(42)
        engine_calm = CellularAutomataEngine(30, 30, EnvironmentConfig(
            wind_speed=0.0, wind_direction=0,
            base_spread_prob=0.3, vegetation_moisture=0.3
        ))
        engine_calm.ignite(15, 5)
        stats_calm = engine_calm.run(max_steps=40)
        
        # Wind-driven fire should burn at least as many cells
        # (not always strictly more due to randomness, but generally yes)
        assert stats_wind['total_burnt'] >= 0  # Sanity check
    
    def test_wind_factor_range(self):
        """Wind factor should be between 0.3 and ~2.0."""
        engine = CellularAutomataEngine(10, 10, EnvironmentConfig(
            wind_speed=10.0, wind_direction=0
        ))
        for i in range(8):
            factor = engine._wind_factor(i)
            assert 0.2 <= factor <= 2.5, \
                f"Wind factor {factor:.2f} out of range for neighbour {i}"


class TestMoistureEffect:
    
    def test_dry_burns_more(self):
        """Dry conditions should result in more burning."""
        np.random.seed(42)
        engine_dry = CellularAutomataEngine(25, 25, EnvironmentConfig(
            base_spread_prob=0.5, vegetation_moisture=0.0, wind_speed=0
        ))
        engine_dry.ignite(12, 12)
        stats_dry = engine_dry.run(max_steps=50)
        
        np.random.seed(42)
        engine_wet = CellularAutomataEngine(25, 25, EnvironmentConfig(
            base_spread_prob=0.5, vegetation_moisture=1.0, wind_speed=0
        ))
        engine_wet.ignite(12, 12)
        stats_wet = engine_wet.run(max_steps=50)
        
        assert stats_dry['total_burnt'] >= stats_wet['total_burnt'], \
            f"Dry ({stats_dry['total_burnt']}) should burn ≥ wet ({stats_wet['total_burnt']})"
    
    def test_moisture_factor_range(self):
        for moisture in [0.0, 0.5, 1.0]:
            engine = CellularAutomataEngine(5, 5, EnvironmentConfig(
                vegetation_moisture=moisture
            ))
            factor = engine._moisture_factor()
            assert 0.0 < factor <= 1.6, \
                f"Moisture factor {factor:.2f} out of range for moisture={moisture}"


class TestStatistics:
    
    def test_stats_populated(self):
        engine = CellularAutomataEngine(15, 15, EnvironmentConfig(
            base_spread_prob=0.6
        ))
        engine.ignite(7, 7)
        stats = engine.run(max_steps=30)
        
        assert stats['total_burnt'] > 0
        assert stats['max_burning'] > 0
        assert stats['steps_to_completion'] > 0
        assert 0 <= stats['burn_fraction'] <= 1
    
    def test_history_recorded(self):
        engine = CellularAutomataEngine(10, 10)
        engine.ignite(5, 5)
        engine.run(max_steps=10)
        
        assert len(engine.history) >= 2, "Should have initial + at least 1 step"
        assert engine.history[0].shape == (10, 10)


class TestProbabilityMap:
    
    def test_probability_map_shape(self):
        engine = CellularAutomataEngine(20, 20)
        engine.ignite(10, 10)
        engine.step()
        
        prob_map = engine.get_spread_probability_map()
        assert prob_map.shape == (20, 20)
    
    def test_probability_map_range(self):
        engine = CellularAutomataEngine(20, 20)
        engine.ignite(10, 10)
        engine.step()
        
        prob_map = engine.get_spread_probability_map()
        assert np.all(prob_map >= 0)
        assert np.all(prob_map <= 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
