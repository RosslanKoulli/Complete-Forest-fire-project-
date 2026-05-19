"""
Map Integration Module
=======================
Generates hexagonal grids on real geographic coordinates using
Uber's H3 library, runs ML fire predictions per hexagon, and
visualises spread probabilities on interactive Folium maps.

Three operational modes:
  1. User-defined weather (sliders) + simulated vegetation
  2. Live weather from Open-Meteo API + simulated vegetation
  3. Full geospatial (weather + vegetation + terrain) — future work

Dependencies:
  pip install h3 folium streamlit-folium

Usage:
  from map_integration import HexFireMap
  
  hex_map = HexFireMap(ml_models, data_pipeline)
  hex_map.generate_grid(lat, lng, radius_km=5, resolution=8)
  hex_map.predict_ignition(weather_params)
  folium_map = hex_map.build_map()
"""

import numpy as np
import h3
import folium
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class HexCell:
    """
    A single hexagonal cell on the map with environmental data
    and fire prediction results.
    """
    hex_id: str                    # H3 hex index
    lat: float                     # Centre latitude
    lng: float                     # Centre longitude
    boundary: List[Tuple[float, float]]  # Vertex coordinates
    
    # Environmental features (populated during prediction)
    temperature: float = 25.0
    humidity: float = 50.0
    wind_speed: float = 5.0
    rain: float = 0.0
    ffmc: float = 80.0
    dmc: float = 50.0
    dc: float = 200.0
    isi: float = 5.0
    vegetation_density: float = 0.5   # 0-1, randomised per cell
    elevation: float = 500.0          # metres, randomised per cell
    
    # Prediction results
    ignition_probability: float = 0.0
    spread_probability: float = 0.0
    is_ignition_point: bool = False
    state: str = 'unburnt'  # unburnt, burning, burnt


class HexFireMap:
    """
    Manages a hexagonal grid overlay on a geographic area,
    runs fire predictions per hexagon, and builds interactive maps.
    
    The hex grid uses Uber's H3 system:
      Resolution 7 = ~1.22 km edge length (coarse, fast)
      Resolution 8 = ~0.46 km edge length (medium, recommended)
      Resolution 9 = ~0.17 km edge length (fine, slow for large areas)
    
    For a 10km × 10km area at resolution 8, you get ~200-400 hexagons.
    Each hexagon is roughly 0.7 km² — similar to a small forest block.
    """
    
    # Colour scales for map visualisation
    IGNITION_COLORS = {
        'very_low': '#27ae60',    # Green: <20%
        'low': '#2ecc71',         # Light green: 20-35%
        'moderate': '#f1c40f',    # Yellow: 35-50%
        'high': '#e67e22',        # Orange: 50-70%
        'very_high': '#e74c3c',   # Red: 70-85%
        'extreme': '#8b0000',     # Dark red: >85%
    }
    
    SPREAD_COLORS = {
        'none': '#228B22',        # Forest green: 0%
        'low': '#90EE90',         # Light green: <20%
        'moderate': '#FFD700',    # Gold: 20-50%
        'high': '#FF6347',        # Tomato: 50-80%
        'extreme': '#8B0000',     # Dark red: >80%
    }
    
    def __init__(self, ml_models: dict = None, data_pipeline=None):
        """
        Parameters
        ----------
        ml_models : dict
            {'Random Forest': sklearn_model, 'XGBoost': model, ...}
        data_pipeline : FireDataPipeline
            Fitted pipeline with scaler for feature transformation.
        """
        self.ml_models = ml_models or {}
        self.pipeline = data_pipeline
        self.hexagons: Dict[str, HexCell] = {}
        self.center_lat = 0.0
        self.center_lng = 0.0
        self.selected_model = 'Random Forest'
    
    def generate_grid(self, center_lat: float, center_lng: float,
                      radius_km: float = 5.0, resolution: int = 8):
        """
        Generate a hexagonal grid covering a circular area.
        
        Parameters
        ----------
        center_lat, center_lng : float
            Centre point of the area to cover.
        radius_km : float
            Approximate radius in kilometres.
        resolution : int
            H3 resolution (7=coarse, 8=medium, 9=fine).
            Resolution 8 gives ~460m hex edges — good balance of
            detail and performance.
        
        How it works:
        1. Convert radius to approximate lat/lng delta
        2. Sample points across the bounding box
        3. Convert each point to its H3 hex index
        4. Use set() to deduplicate (many points map to same hex)
        5. Create HexCell objects with randomised vegetation/elevation
        """
        self.center_lat = center_lat
        self.center_lng = center_lng
        self.hexagons = {}
        
        # Approximate degrees per km (varies with latitude)
        lat_delta = radius_km / 111.0
        lng_delta = radius_km / (111.0 * np.cos(np.radians(center_lat)))
        
        # Sample points to find all hexagons in the area
        step = 0.001 if resolution >= 9 else 0.002
        hex_ids = set()
        
        for lat in np.arange(center_lat - lat_delta,
                             center_lat + lat_delta, step):
            for lng in np.arange(center_lng - lng_delta,
                                 center_lng + lng_delta, step):
                # Only include hexagons within the circular radius
                dist_km = np.sqrt(
                    ((lat - center_lat) * 111) ** 2 +
                    ((lng - center_lng) * 111 * np.cos(np.radians(center_lat))) ** 2
                )
                if dist_km <= radius_km:
                    hex_ids.add(h3.latlng_to_cell(lat, lng, resolution))
        
        # Create HexCell objects with randomised local conditions
        np.random.seed(42)
        for hex_id in hex_ids:
            lat, lng = h3.cell_to_latlng(hex_id)
            boundary = h3.cell_to_boundary(hex_id)
            
            self.hexagons[hex_id] = HexCell(
                hex_id=hex_id,
                lat=lat,
                lng=lng,
                boundary=list(boundary),
                vegetation_density=np.random.uniform(0.2, 1.0),
                elevation=np.random.uniform(200, 1200),
            )
        
        return len(self.hexagons)
    
    def set_weather_conditions(self, temperature: float, humidity: float,
                                wind_speed: float, rain: float,
                                ffmc: float, dmc: float,
                                dc: float, isi: float):
        """
        Set uniform weather conditions across all hexagons.
        
        In Version 1 (user-defined), all hexagons get the same
        weather but have different vegetation density and elevation,
        which causes spatial variation in ignition probability.
        
        In a real system, you'd fetch per-location weather from
        an API like Open-Meteo and set different values per hex.
        """
        for cell in self.hexagons.values():
            cell.temperature = temperature
            cell.humidity = humidity
            cell.wind_speed = wind_speed
            cell.rain = rain
            cell.ffmc = ffmc
            cell.dmc = dmc
            cell.dc = dc
            cell.isi = isi
    
    def add_spatial_variation(self):
        """
        Add realistic spatial variation to environmental features.
        
        Even with uniform weather, local conditions vary:
        - Higher elevation = cooler temperature, higher wind
        - Dense vegetation = slightly higher humidity (transpiration)
        - South-facing slopes = warmer (more sun exposure)
        
        This creates meaningful variation in the ignition probability
        map even when base weather is uniform.
        """
        for cell in self.hexagons.values():
            # Elevation effect: -0.6°C per 100m rise (lapse rate)
            elev_diff = (cell.elevation - 500) / 100
            cell.temperature = cell.temperature - 0.6 * elev_diff
            
            # Higher elevation = windier
            cell.wind_speed = cell.wind_speed * (1 + 0.05 * elev_diff)
            
            # Vegetation moisture effect
            cell.humidity = cell.humidity + 5 * cell.vegetation_density
            
            # Adjust FFMC based on local conditions
            cell.ffmc = cell.ffmc - 2 * cell.vegetation_density + 0.5 * elev_diff
            cell.ffmc = np.clip(cell.ffmc, 18, 96)
    
    def predict_ignition(self, model_name: str = 'Random Forest'):
        """
        Run fire ignition prediction for every hexagon using
        the specified ML model.
        
        Each hexagon's 11 features are transformed through the
        same pipeline used during training (StandardScaler), then
        fed to the model's predict_proba() method.
        
        The spatial variation in vegetation and elevation means
        each hexagon gets a slightly different probability, creating
        the heat-map effect on the map.
        """
        self.selected_model = model_name
        
        if model_name not in self.ml_models or self.pipeline is None:
            # No model loaded — use simulated probabilities
            self._simulate_ignition_probabilities()
            return
        
        model = self.ml_models[model_name]
        
        for cell in self.hexagons.values():
            month_sin = np.sin(2 * np.pi * 8 / 12)  # August default
            month_cos = np.cos(2 * np.pi * 8 / 12)
            
            input_dict = {
                'temperature': cell.temperature,
                'relative_humidity': cell.humidity,
                'wind_speed': cell.wind_speed,
                'rain': cell.rain,
                'FFMC': cell.ffmc,
                'DMC': cell.dmc,
                'DC': cell.dc,
                'ISI': cell.isi,
                'region_encoded': 0,
                'month_sin': month_sin,
                'month_cos': month_cos,
            }
            
            try:
                X = self.pipeline.transform_single_input(input_dict)
                proba = model.predict_proba(X)[0][1]
                cell.ignition_probability = float(proba)
            except Exception:
                cell.ignition_probability = 0.5
    
    def _simulate_ignition_probabilities(self):
        """
        Fallback when no ML model is loaded.
        Generates plausible-looking spatial probabilities based on
        vegetation density and temperature.
        """
        for cell in self.hexagons.values():
            # Simple heuristic: hot + dry + dense vegetation = high risk
            temp_factor = np.clip((cell.temperature - 15) / 25, 0, 1)
            humid_factor = np.clip(1 - cell.humidity / 100, 0, 1)
            veg_factor = cell.vegetation_density
            
            prob = 0.5 * temp_factor + 0.3 * humid_factor + 0.2 * veg_factor
            prob += np.random.normal(0, 0.08)
            cell.ignition_probability = float(np.clip(prob, 0.02, 0.98))
    
    def simulate_spread(self, ignition_hex_id: str,
                         wind_direction: float = 90.0,
                         wind_speed: float = 5.0,
                         steps: int = 10):
        """
        Run fire spread simulation from a selected ignition hexagon.
        
        Uses H3's k_ring to find neighbouring hexagons, then
        calculates spread probability based on wind direction,
        vegetation density, and distance from the fire front.
        
        Parameters
        ----------
        ignition_hex_id : str
            H3 index of the hexagon where fire starts.
        wind_direction : float
            Degrees from north (0=N, 90=E, 180=S, 270=W).
        wind_speed : float
            Wind speed in km/h.
        steps : int
            Number of spread iterations to simulate.
        """
        # Reset spread state
        for cell in self.hexagons.values():
            cell.spread_probability = 0.0
            cell.state = 'unburnt'
            cell.is_ignition_point = False
        
        if ignition_hex_id not in self.hexagons:
            return
        
        # Set ignition point
        self.hexagons[ignition_hex_id].is_ignition_point = True
        self.hexagons[ignition_hex_id].state = 'burning'
        self.hexagons[ignition_hex_id].spread_probability = 1.0
        
        burning = {ignition_hex_id}
        burnt = set()
        
        wind_rad = np.radians(wind_direction)
        
        for step in range(steps):
            new_burning = set()
            
            for hex_id in burning:
                cell = self.hexagons[hex_id]
                
                # Get H3 neighbours (ring of 1 = immediate neighbours)
                neighbours = h3.grid_ring(hex_id, 1)
                
                for neighbour_id in neighbours:
                    if neighbour_id not in self.hexagons:
                        continue
                    if neighbour_id in burning or neighbour_id in burnt:
                        continue
                    
                    neighbour = self.hexagons[neighbour_id]
                    
                    # Calculate spread probability
                    # 1. Wind factor: fire spreads faster downwind
                    dlat = neighbour.lat - cell.lat
                    dlng = neighbour.lng - cell.lng
                    spread_angle = np.arctan2(dlng, dlat)
                    angle_diff = spread_angle - wind_rad
                    wind_factor = 1.0 + (wind_speed / 10.0) * np.cos(angle_diff)
                    wind_factor = max(wind_factor, 0.3)
                    
                    # 2. Vegetation factor
                    veg_factor = neighbour.vegetation_density
                    
                    # 3. Moisture factor (higher humidity = slower spread)
                    moisture_factor = max(1.5 - neighbour.humidity / 80, 0.2)
                    
                    # 4. Base probability decays with distance from origin
                    base_prob = 0.4 * (1 - step * 0.05)
                    
                    prob = min(base_prob * wind_factor * veg_factor * moisture_factor, 0.95)
                    
                    # Stochastic spread decision
                    if np.random.random() < prob:
                        new_burning.add(neighbour_id)
                        neighbour.state = 'burning'
                    
                    # Update spread probability (max of all attempts)
                    neighbour.spread_probability = max(
                        neighbour.spread_probability, prob
                    )
            
            # Transition burning → burnt
            for hex_id in burning:
                self.hexagons[hex_id].state = 'burnt'
                burnt.add(hex_id)
            
            burning = new_burning
            
            if not burning:
                break
    
    def _prob_to_ignition_color(self, prob: float) -> str:
        """Map ignition probability to a hex colour string."""
        if prob < 0.20: return self.IGNITION_COLORS['very_low']
        elif prob < 0.35: return self.IGNITION_COLORS['low']
        elif prob < 0.50: return self.IGNITION_COLORS['moderate']
        elif prob < 0.70: return self.IGNITION_COLORS['high']
        elif prob < 0.85: return self.IGNITION_COLORS['very_high']
        else: return self.IGNITION_COLORS['extreme']
    
    def _prob_to_spread_color(self, prob: float, state: str) -> str:
        """Map spread probability/state to a colour."""
        if state == 'burnt': return '#2F2F2F'
        if state == 'burning': return '#FF4500'
        if prob < 0.01: return self.SPREAD_COLORS['none']
        elif prob < 0.20: return self.SPREAD_COLORS['low']
        elif prob < 0.50: return self.SPREAD_COLORS['moderate']
        elif prob < 0.80: return self.SPREAD_COLORS['high']
        else: return self.SPREAD_COLORS['extreme']
    
    def build_ignition_map(self) -> folium.Map:
        """
        Build a Folium map showing ignition probability per hexagon.
        
        Each hexagon is coloured by its ML-predicted fire ignition
        probability. Clicking a hexagon shows its probability and
        feature values in a popup.
        """
        m = folium.Map(
            location=[self.center_lat, self.center_lng],
            zoom_start=13, tiles='OpenStreetMap'
        )
        
        for hex_id, cell in self.hexagons.items():
            boundary = cell.boundary
            coords = [[lng, lat] for lat, lng in boundary]
            coords.append(coords[0])
            
            color = self._prob_to_ignition_color(cell.ignition_probability)
            
            geojson = {
                'type': 'Feature',
                'geometry': {'type': 'Polygon', 'coordinates': [coords]},
                'properties': {
                    'hex_id': hex_id,
                    'probability': cell.ignition_probability,
                }
            }
            
            popup_html = (
                f"<b>Fire ignition probability: {cell.ignition_probability*100:.1f}%</b><br>"
                f"Model: {self.selected_model}<br>"
                f"<hr>"
                f"Temp: {cell.temperature:.1f}°C<br>"
                f"Humidity: {cell.humidity:.0f}%<br>"
                f"Wind: {cell.wind_speed:.1f} km/h<br>"
                f"FFMC: {cell.ffmc:.1f}<br>"
                f"Vegetation: {cell.vegetation_density:.0%}<br>"
                f"Elevation: {cell.elevation:.0f}m"
            )
            
            folium.GeoJson(
                geojson,
                style_function=lambda x, c=color: {
                    'fillColor': c, 'color': '#333',
                    'weight': 0.5, 'fillOpacity': 0.6,
                },
                tooltip=f"Ignition: {cell.ignition_probability*100:.1f}%",
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(m)
        
        # Add legend
        self._add_legend(m, 'ignition')
        
        return m
    
    def build_spread_map(self) -> folium.Map:
        """
        Build a Folium map showing fire spread probabilities.
        
        Hexagons are coloured by spread probability / burn state:
        - Dark grey: burnt
        - Orange-red: currently burning
        - Red to green gradient: spread probability for unburnt cells
        """
        m = folium.Map(
            location=[self.center_lat, self.center_lng],
            zoom_start=13, tiles='OpenStreetMap'
        )
        
        for hex_id, cell in self.hexagons.items():
            boundary = cell.boundary
            coords = [[lng, lat] for lat, lng in boundary]
            coords.append(coords[0])
            
            color = self._prob_to_spread_color(
                cell.spread_probability, cell.state
            )
            opacity = 0.7 if cell.state != 'unburnt' or cell.spread_probability > 0.01 else 0.3
            
            geojson = {
                'type': 'Feature',
                'geometry': {'type': 'Polygon', 'coordinates': [coords]},
                'properties': {'hex_id': hex_id}
            }
            
            if cell.is_ignition_point:
                label = "IGNITION POINT"
            elif cell.state == 'burnt':
                label = f"Burnt (was {cell.spread_probability*100:.0f}% likely)"
            elif cell.state == 'burning':
                label = "Currently burning"
            else:
                label = f"Spread risk: {cell.spread_probability*100:.1f}%"
            
            folium.GeoJson(
                geojson,
                style_function=lambda x, c=color, o=opacity: {
                    'fillColor': c, 'color': '#333',
                    'weight': 0.5, 'fillOpacity': o,
                },
                tooltip=label,
            ).add_to(m)
        
        self._add_legend(m, 'spread')
        
        return m
    
    def _add_legend(self, m: folium.Map, mode: str):
        """Add a colour legend to the map."""
        if mode == 'ignition':
            colors = self.IGNITION_COLORS
            labels = ['<20% Very low', '20-35% Low', '35-50% Moderate',
                      '50-70% High', '70-85% Very high', '>85% Extreme']
        else:
            colors = self.SPREAD_COLORS
            labels = ['No risk', '<20% Low', '20-50% Moderate',
                      '50-80% High', '>80% Extreme']
        
        legend_html = '<div style="position:fixed;bottom:30px;left:30px;z-index:1000;'
        legend_html += 'background:white;padding:10px;border-radius:5px;border:1px solid #ccc;'
        legend_html += 'font-size:12px;">'
        legend_html += f'<b>{"Ignition" if mode == "ignition" else "Spread"} probability</b><br>'
        
        for color, label in zip(colors.values(), labels):
            legend_html += (
                f'<span style="background:{color};width:14px;height:14px;'
                f'display:inline-block;margin-right:4px;border:1px solid #999;">'
                f'</span>{label}<br>'
            )
        legend_html += '</div>'
        
        m.get_root().html.add_child(folium.Element(legend_html))
    
    def get_hex_at_coords(self, lat: float, lng: float) -> Optional[str]:
        """Find which hexagon contains the given coordinates."""
        resolution = 8  # Must match the resolution used in generate_grid
        if self.hexagons:
            # Infer resolution from existing hexagons
            sample_hex = list(self.hexagons.keys())[0]
            resolution = h3.get_resolution(sample_hex)
        
        hex_id = h3.latlng_to_cell(lat, lng, resolution)
        return hex_id if hex_id in self.hexagons else None
    
    def get_stats(self) -> dict:
        """Return summary statistics for the current grid."""
        probs = [c.ignition_probability for c in self.hexagons.values()]
        spread_probs = [c.spread_probability for c in self.hexagons.values()]
        
        return {
            'total_hexagons': len(self.hexagons),
            'avg_ignition_prob': float(np.mean(probs)) if probs else 0,
            'max_ignition_prob': float(np.max(probs)) if probs else 0,
            'high_risk_hexagons': sum(1 for p in probs if p > 0.7),
            'burnt_hexagons': sum(1 for c in self.hexagons.values() if c.state == 'burnt'),
            'burning_hexagons': sum(1 for c in self.hexagons.values() if c.state == 'burning'),
            'max_spread_prob': float(np.max(spread_probs)) if spread_probs else 0,
        }


if __name__ == '__main__':
    # Test the module without ML models
    print("Testing HexFireMap...")
    
    hfm = HexFireMap()
    
    # Generate grid over Montesinho Natural Park, Portugal
    n_hex = hfm.generate_grid(41.86, -6.73, radius_km=3, resolution=8)
    print(f"Generated {n_hex} hexagons")
    
    # Set weather and predict
    hfm.set_weather_conditions(
        temperature=35, humidity=25, wind_speed=8,
        rain=0, ffmc=92, dmc=150, dc=600, isi=12
    )
    hfm.add_spatial_variation()
    hfm.predict_ignition()
    
    stats = hfm.get_stats()
    print(f"Stats: {stats}")
    
    # Build ignition map
    m1 = hfm.build_ignition_map()
    m1.save('figures/hex_ignition_map.html')
    print("Ignition map saved: figures/hex_ignition_map.html")
    
    # Simulate spread from highest-probability hex
    best_hex = max(hfm.hexagons.items(),
                   key=lambda x: x[1].ignition_probability)[0]
    print(f"Ignition point: {best_hex} "
          f"({hfm.hexagons[best_hex].ignition_probability*100:.1f}%)")
    
    hfm.simulate_spread(best_hex, wind_direction=90, wind_speed=8, steps=8)
    
    stats2 = hfm.get_stats()
    print(f"After spread: {stats2}")
    
    # Build spread map
    m2 = hfm.build_spread_map()
    m2.save('figures/hex_spread_map.html')
    print("Spread map saved: figures/hex_spread_map.html")
    
    print("\nHexFireMap test complete!")
