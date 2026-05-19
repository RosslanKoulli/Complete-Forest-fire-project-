"""
Cellular Automata Fire Spread Engine
======================================
Simulates fire propagation across a grid landscape using
probabilistic state transitions.

Cell States: UNBURNT(0), BURNING(1), BURNT(2), FIREBREAK(3)

Spread probability:
  P = P_base × F_vegetation × F_wind × F_moisture

Wind factor uses cosine of angle between wind direction and
spread direction — fire spreads faster downwind, slower upwind.

References:
  Trunfio (2004) - Hexagonal CA for wildfire prediction
  Li (2023) - Parallel CA implementation (GitHub)
"""

import numpy as np
from enum import IntEnum
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


class CellState(IntEnum):
    UNBURNT = 0
    BURNING = 1
    BURNT = 2
    FIREBREAK = 3


@dataclass
class EnvironmentConfig:
    """Environmental parameters controlling spread behaviour."""
    wind_speed: float = 5.0          # km/h
    wind_direction: float = 0.0     # degrees, 0=North, clockwise
    base_spread_prob: float = 0.4   # baseline probability
    vegetation_moisture: float = 0.5 # 0=bone dry, 1=saturated
    temperature: float = 25.0       # °C (affects drying)
    burn_duration: int = 3          # steps before BURNING→BURNT


class CellularAutomataEngine:
    """
    Fire spread simulation on a square grid.
    
    Usage:
        config = EnvironmentConfig(wind_speed=8, wind_direction=90)
        engine = CellularAutomataEngine(50, 50, config)
        engine.add_random_firebreaks(0.05)
        engine.ignite(25, 25)
        stats = engine.run(max_steps=200)
        engine.generate_animation('fire_spread.gif')
    """
    
    # Moore neighbourhood: 8 surrounding cells
    NEIGHBOURS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    
    # Angle from centre to each neighbour (degrees from North)
    NEIGHBOUR_ANGLES = [315, 0, 45, 270, 90, 225, 180, 135]
    
    COLORS = ['#228B22', '#FF4500', '#2F2F2F', '#4169E1']
    
    def __init__(self, rows: int = 50, cols: int = 50,
                 config: Optional[EnvironmentConfig] = None):
        self.rows = rows
        self.cols = cols
        self.config = config or EnvironmentConfig()
        
        self.grid = np.full((rows, cols), CellState.UNBURNT, dtype=int)
        self.burn_timer = np.zeros((rows, cols), dtype=int)
        
        # Random vegetation density (affects spread)
        self.vegetation = np.random.uniform(0.3, 1.0, (rows, cols))
        
        self.history: List[np.ndarray] = []
        self.step_count = 0
        self.stats = {'total_burnt': 0, 'max_burning': 0,
                      'steps_to_completion': 0, 'burn_fraction': 0.0}
    
    def add_firebreaks(self, positions: List[Tuple[int, int]]):
        """Mark specific cells as firebreaks (water, roads)."""
        for r, c in positions:
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.grid[r, c] = CellState.FIREBREAK
    
    def add_random_firebreaks(self, density: float = 0.05):
        """Scatter random firebreaks across the grid."""
        mask = np.random.random((self.rows, self.cols)) < density
        self.grid[mask] = CellState.FIREBREAK
    
    def ignite(self, row: int, col: int):
        """Start fire at a specific cell."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            if self.grid[row, col] == CellState.UNBURNT:
                self.grid[row, col] = CellState.BURNING
                self.burn_timer[row, col] = 1
    
    def ignite_from_probability_map(self, prob_map: np.ndarray,
                                     threshold: float = 0.7):
        """
        Ignite cells based on ML-predicted probability map.
        Integration point between ML models and CA simulation.
        """
        mask = (prob_map >= threshold) & (self.grid == CellState.UNBURNT)
        self.grid[mask] = CellState.BURNING
        self.burn_timer[mask] = 1
    
    def _wind_factor(self, neighbour_idx: int) -> float:
        """
        How wind affects spread to a specific neighbour.
        
        Uses cosine of angle difference between wind and spread direction.
        Downwind: factor up to 2.0 (fire spreads fast)
        Upwind: factor down to 0.3 (fire struggles)
        No wind: factor = 1.0 (uniform spread)
        """
        speed_norm = min(self.config.wind_speed / 10.0, 1.0)
        spread_angle = self.NEIGHBOUR_ANGLES[neighbour_idx]
        angle_diff = np.radians(self.config.wind_direction - spread_angle)
        alignment = np.cos(angle_diff)
        return max(1.0 + speed_norm * alignment, 0.3)
    
    def _moisture_factor(self) -> float:
        """
        Moisture reduces spread.
        Dry (0.0) → 1.5, Normal (0.5) → 0.9, Wet (1.0) → 0.3
        """
        return max(1.5 - self.config.vegetation_moisture * 1.2, 0.1)
    
    def step(self) -> bool:
        """
        Advance simulation by one timestep.
        Returns False when no cells are burning (simulation done).
        """
        new_grid = self.grid.copy()
        new_timers = self.burn_timer.copy()
        
        burning = np.argwhere(self.grid == CellState.BURNING)
        if len(burning) == 0:
            return False
        
        moisture_f = self._moisture_factor()
        
        for r, c in burning:
            for idx, (dr, dc) in enumerate(self.NEIGHBOURS):
                nr, nc = r + dr, c + dc
                
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if self.grid[nr, nc] != CellState.UNBURNT:
                    continue
                
                wind_f = self._wind_factor(idx)
                veg_f = self.vegetation[nr, nc]
                
                p_spread = min(
                    self.config.base_spread_prob * veg_f * wind_f * moisture_f,
                    1.0
                )
                
                if np.random.random() < p_spread:
                    new_grid[nr, nc] = CellState.BURNING
                    new_timers[nr, nc] = 1
            
            new_timers[r, c] += 1
            if new_timers[r, c] >= self.config.burn_duration:
                new_grid[r, c] = CellState.BURNT
        
        self.grid = new_grid
        self.burn_timer = new_timers
        self.step_count += 1
        
        n_burning = int(np.sum(self.grid == CellState.BURNING))
        n_burnt = int(np.sum(self.grid == CellState.BURNT))
        self.stats['max_burning'] = max(self.stats['max_burning'], n_burning)
        self.stats['total_burnt'] = n_burnt
        self.stats['steps_to_completion'] = self.step_count
        
        self.history.append(self.grid.copy())
        return True
    
    def run(self, max_steps: int = 200) -> dict:
        """Run until fire dies or max steps reached."""
        self.history = [self.grid.copy()]
        
        for _ in range(max_steps):
            if not self.step():
                break
        
        total_burnable = np.sum(self.grid != CellState.FIREBREAK)
        self.stats['burn_fraction'] = (
            self.stats['total_burnt'] / max(total_burnable, 1)
        )
        return self.stats
    
    def get_spread_probability_map(self) -> np.ndarray:
        """
        Current probability map for each unburnt cell.
        Used for web app colour-coded risk visualisation.
        """
        prob_map = np.zeros((self.rows, self.cols))
        burning = np.argwhere(self.grid == CellState.BURNING)
        moisture_f = self._moisture_factor()
        
        for r, c in burning:
            for idx, (dr, dc) in enumerate(self.NEIGHBOURS):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if self.grid[nr, nc] != CellState.UNBURNT:
                    continue
                
                wind_f = self._wind_factor(idx)
                veg_f = self.vegetation[nr, nc]
                p = min(self.config.base_spread_prob * veg_f * wind_f * moisture_f, 1.0)
                prob_map[nr, nc] = max(prob_map[nr, nc], p)
        
        return prob_map
    
    def plot_final_state(self, save_path: str = 'figures/fire_spread_final.png'):
        """Save the final grid state as an image."""
        cmap = ListedColormap(self.COLORS)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.grid, cmap=cmap, vmin=0, vmax=3)
        
        labels = ['Unburnt', 'Burning', 'Burnt', 'Firebreak']
        patches = [mpatches.Patch(color=c, label=l)
                   for c, l in zip(self.COLORS, labels)]
        ax.legend(handles=patches, loc='upper right', fontsize=9)
        
        ax.set_title(
            f'Fire Spread — Step {self.step_count} | '
            f'Burnt: {self.stats["total_burnt"]} | '
            f'{self.stats["burn_fraction"]*100:.1f}%',
            fontsize=13
        )
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Final state saved: {save_path}")
    
    def generate_animation(self, save_path: str = 'figures/fire_spread.gif',
                           fps: int = 5):
        """Create animated GIF for report/exhibition."""
        if len(self.history) < 2:
            print("  Not enough history for animation.")
            return
        
        cmap = ListedColormap(self.COLORS)
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(self.history[0], cmap=cmap, vmin=0, vmax=3)
        
        labels = ['Unburnt', 'Burning', 'Burnt', 'Firebreak']
        patches = [mpatches.Patch(color=c, label=l)
                   for c, l in zip(self.COLORS, labels)]
        ax.legend(handles=patches, loc='upper right', fontsize=9)
        title = ax.set_title('Step 0')
        ax.axis('off')
        
        def update(frame):
            im.set_data(self.history[frame])
            n_burn = int(np.sum(self.history[frame] == CellState.BURNING))
            n_burnt = int(np.sum(self.history[frame] == CellState.BURNT))
            title.set_text(f'Step {frame} | Burning: {n_burn} | Burnt: {n_burnt}')
            return [im, title]
        
        anim = animation.FuncAnimation(
            fig, update, frames=len(self.history),
            interval=1000 // fps, blit=True
        )
        anim.save(save_path, writer='pillow', fps=fps)
        plt.close()
        print(f"  Animation saved: {save_path} ({len(self.history)} frames)")


def run_wind_comparison(grid_size: int = 40, save_path: str = 'figures/wind_comparison.png'):
    """
    Compare fire spread under 8 wind directions.
    Great figure for the report showing model sensitivity.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    cmap = ListedColormap(CellularAutomataEngine.COLORS)
    
    for i, ax in enumerate(axes.flat):
        config = EnvironmentConfig(
            wind_speed=8.0,
            wind_direction=i * 45,
            base_spread_prob=0.45,
            vegetation_moisture=0.3,
        )
        engine = CellularAutomataEngine(grid_size, grid_size, config)
        engine.add_random_firebreaks(0.03)
        engine.ignite(grid_size // 2, grid_size // 2)
        stats = engine.run(max_steps=80)
        
        ax.imshow(engine.grid, cmap=cmap, vmin=0, vmax=3)
        ax.set_title(f'Wind: {directions[i]} ({i*45}°)\n'
                     f'Burnt: {stats["total_burnt"]}',
                     fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Fire Spread Under Different Wind Directions', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Wind comparison saved: {save_path}")


if __name__ == '__main__':
    print("Testing Cellular Automata Engine...")
    
    # Basic test
    config = EnvironmentConfig(
        wind_speed=6.0, wind_direction=90,  # East wind
        base_spread_prob=0.45, vegetation_moisture=0.3,
    )
    engine = CellularAutomataEngine(50, 50, config)
    engine.add_random_firebreaks(0.05)
    engine.ignite(25, 25)
    
    stats = engine.run(max_steps=100)
    
    print(f"\n  Grid:         {engine.rows}×{engine.cols}")
    print(f"  Steps:        {stats['steps_to_completion']}")
    print(f"  Total burnt:  {stats['total_burnt']}")
    print(f"  Max burning:  {stats['max_burning']}")
    print(f"  Burn fraction:{stats['burn_fraction']*100:.1f}%")
    
    engine.plot_final_state()
    engine.generate_animation(fps=8)
    
    # Wind comparison
    print("\nGenerating wind comparison...")
    run_wind_comparison()
    
    print("\nCA Engine test complete!")
