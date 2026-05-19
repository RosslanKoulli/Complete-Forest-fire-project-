# Forest Fire Prediction - Full-stack version

A FastAPI backend with a hand-written vanilla JavaScript frontend, layered on
top of the existing forest-fire-prediction project. Compares Random Forest,
XGBoost, and an MLP on fire ignition prediction; runs hex-based cellular
automata fire spread simulations. The user interface is a Leaflet map: draw
a rectangle, get hex-by-hex predictions, click to ignite and simulate spread.

This replaces the earlier slider-based prototype. The slider page and the
grid-based simulator are no longer in the codebase. Their URLs (`/predict`
and `/simulate`) now redirect to the map page so old bookmarks still work.

## What is in this directory

```
forest-fire-fullstack/
+- backend/                    FastAPI app
|  +- main.py                  app entry, mounts API + static frontend
|  +- requirements.txt
|  +- routers/                 one file per endpoint group
|  |  +- health.py
|  |  +- predict.py            old single-prediction endpoint (kept)
|  |  +- compare.py
|  |  +- hexmap.py             new: per-hex prediction + WS spread sim
|  +- schemas/                 Pydantic request/response models
|  +- services/
|     +- model_loader.py       loads trained models + SHAP explainers
|     +- fwi_calculator.py     Van Wagner FFMC/DMC/DC/ISI equations
|     +- weather_client.py     Open-Meteo client with caching
|     +- hex_simulation.py     hex-based cellular automaton
+- frontend/                   hand-written HTML/CSS/JS, no build step
|  +- index.html               landing page
|  +- map.html                 NEW: Leaflet map + draw + simulate
|  +- compare.html             metric cards, ROC, t-tests
|  +- about.html               methodology and scoping
|  +- css/styles.css           single CSS file
|  +- js/
|     +- api.js                fetch wrapper for the older endpoints
|     +- compare.js
|     +- map.js                NEW: map page logic
|     +- utils.js              DOM helpers and formatters
+- fix_imports.py              utility for fixing the data_pipeline import error
+- README.md
```

## Quick start

```
cd backend
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export FOREST_FIRE_PROJECT_ROOT=/path/to/forest-fire-prediction
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000 in a browser. The Map link in the nav goes to
the new interactive page.

## Architecture

```
   Browser (Leaflet + h3-js + vanilla JS)
   |
   | HTTP + WebSocket (same origin)
   v
   FastAPI (port 8000)
   |   +- /api/hexmap/predict_region    POST batch prediction
   |   +- /api/hexmap/simulate_stream   WS frame-by-frame spread
   |   +- /api/predict, /api/compare    older endpoints (kept for compatibility)
   |   +- /static/*                     serves the frontend files
   |
   v
   Trained models + Van Wagner FWI calculator
   |
   v
   Open-Meteo API (free, no key, archive endpoint)
```

One process, one URL, one terminal. No Node.js, no npm, no build step.

## How the map page works

1. User draws a rectangle on the Leaflet map using the rectangle tool.
2. JavaScript tessellates the rectangle with H3 hexagons at the chosen
   resolution (5-8). The cap is 500 hexes per request.
3. The frontend POSTs the hexes to `/api/hexmap/predict_region`.
4. For each hex the backend fetches the last 7 days of weather from
   Open-Meteo (cached at 0.1 degree resolution to share calls between
   nearby hexes), walks them through the Van Wagner equations to compute
   FFMC, DMC, DC, ISI, then runs all three trained models.
5. Hexes are coloured low/medium/high based on the average probability.
6. User clicks any hex. Frontend opens a WebSocket and sends the hex
   graph plus simulation parameters. Backend runs the hex-based cellular
   automaton, streaming one frame per step at ~20 fps. Frontend recolours
   hexes as they ignite and burn out.

## Required external services

- **Open-Meteo** (api.open-meteo.com, archive-api.open-meteo.com).
  Free, no API key, generous rate limits. The deployment server needs
  outbound HTTPS to these domains. If unreachable, the backend falls
  back to climatologically reasonable Mediterranean summer values and
  marks the prediction's `weather_source` as `fallback` so the user can
  see what happened.

## Modifying the frontend

No build step. Edit any HTML, CSS, or JS file and refresh the browser.
The map page lives in `frontend/map.html` (markup, controls, legend) and
`frontend/js/map.js` (Leaflet setup, draw handler, prediction request,
WebSocket simulation loop). External libraries (Leaflet, leaflet-draw,
h3-js) are loaded from unpkg's CDN as global scripts; our own code uses
ES modules.

## Endpoints

| Method | Path                              | Purpose                                  |
|--------|-----------------------------------|------------------------------------------|
| GET    | `/api/health`                     | Loaded models, pipeline status           |
| POST   | `/api/predict`                    | Single-input prediction (legacy)         |
| POST   | `/api/predict/{model_name}`       | Single model only                        |
| POST   | `/api/hexmap/predict_region`      | Batch per-hex prediction                 |
| WS     | `/api/hexmap/simulate_stream`     | Hex-based fire spread streaming          |
| GET    | `/api/hexmap/health`              | Hex map service health                   |
| GET    | `/api/compare/metrics`            | AUC, t-tests, feature importance         |
| GET    | `/api/compare/figure/{name}`      | Pre-rendered evaluation figure           |

Auto-generated interactive docs at `/docs`.

## Common errors and fixes

**ModuleNotFoundError: No module named 'data_pipeline'** when running a
file in `models/` directly. This is the import-path issue. Run
`fix_imports.py` from the project root, then run model files with `-m`:
`python3 -m models.xgboost_model`. See the script's docstring for the
explanation.

**Open-Meteo 403 Forbidden** in the logs. The deployment server cannot
reach Open-Meteo. Check the firewall and ensure outbound HTTPS to
api.open-meteo.com is allowed. The app will keep working with fallback
values until this is fixed.

**No hexes appear after drawing a rectangle.** Check the browser console
for errors. The most common cause is the H3 resolution being too high
for the rectangle size (over 500 hexes triggers the cap). Lower the
resolution slider or draw a smaller rectangle.
