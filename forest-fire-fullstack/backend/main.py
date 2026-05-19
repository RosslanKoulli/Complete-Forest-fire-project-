"""
FastAPI Entry Point
====================

This is the top-level application module. Its job is to:

  1. Configure the FastAPI app with metadata that surfaces in /docs
  2. Set up CORS so the Next.js frontend (running on a different port in
     development) can call the API without browser errors
  3. Load the trained ML models exactly once at startup, not on every
     request -- joblib.load is slow, and we want the cost amortised
  4. Mount the routers that handle the actual endpoints

Run in development:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000 For Server
    python -m uvicorn main:app --reload --port  8000 For Windows local device
    uvicorn main:app --reload --port 8000 For Linux local device

The auto-generated interactive docs live at http://localhost:8000/docs.
That documentation: hand it to an examiner and they can call the API themselves with no other tooling.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import os
import secrets
import base64

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

from routers import predict, hexmap, health
from services.model_loader import ModelRegistry


# ---------------------------------------------------------------
# Application state
# ---------------------------------------------------------------
#
# We keep the loaded models on app.state rather than importing them as
# module-level globals. Two reasons:
#   - Test isolation: we can override app.state.models in a test fixture
#     to swap in mock models without monkeypatching the import.
#   - Restart safety: if a model file is corrupted on startup, we surface
#     the error immediately rather than silently importing None.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan hook - runs once on startup, once on shutdown.

    The startup branch loads the three trained models, the data pipeline,
    and the SHAP explainers. Loading is single-threaded and blocking, but
    it happens exactly once before the first request lands.
    """
    print('[startup] Loading models and pipeline...')
    registry = ModelRegistry()
    registry.load_all()
    app.state.registry = registry
    print(f'[startup] Loaded models: {list(registry.models.keys())}')
    yield
    # Nothing to clean up on shutdown - joblib-loaded models are
    # garbage-collected when the registry goes out of scope.


# ---------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------

app = FastAPI(
    title='Forest Fire Prediction API',
    description=(
        'REST API for the CI601 forest fire prediction project. Serves '
        'three machine learning classifiers (Random Forest, XGBoost, MLP) '
        'and a cellular automata fire spread simulator. See /docs for '
        'interactive API documentation.'
    ),
    version='1.0.0',
    lifespan=lifespan,
)


# ---------------------------------------------------------------
# CORS - allow the Next.js dev server and the production deployment
# ---------------------------------------------------------------
#
# In development, Next.js runs on http://localhost:3000 and the API runs
# on http://localhost:8000. Browsers block cross-origin requests unless
# the API explicitly opts in. The list below covers local dev and the
# typical production patterns; expand it for your actual deployment.

ALLOWED_ORIGINS = [
    'http://localhost:3000',          # dev server
    'http://127.0.0.1:3000',
    'http://localhost:3001',          # Alternate dev port
    # Add your production frontend URL here, e.g.:
    # 'https://your-app.vercel.app',
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=['GET', 'POST'],    # We only use these two
    allow_headers=['*'],
)


# ---------------------------------------------------------------
# Optional HTTP Basic auth for production deployments
# ---------------------------------------------------------------
#
# When APP_PASSWORD is set in the environment, every request to the app
# must include the matching Basic auth credentials. When the variable is
# unset (the default for local development), no auth is required and
# the middleware passes through.
#
# This is a single shared password, not a user account system. The
# username can be anything (the middleware ignores it). Only the
# password is checked. The browser remembers the credentials for the
# current session, so users authenticate once per browser tab/window.
#
# Username/password comparison uses secrets.compare_digest to avoid
# timing-attack leakage of the correct password.

# Password lookup order:
# 1. APP_PASSWORD environment variable (useful for one-off testing or CI)
# 2. A file named .env in the backend directory containing APP_PASSWORD=...
# 3. Empty -> auth disabled
#
# The .env approach is the friendlier option for self-hosted deployments
# because the friend running the server can edit the file with a text
# editor without needing to remember the syntax for setting environment
# variables permanently. The file is read once at startup; restart the
# server to apply a password change.

def _load_app_password() -> str:
    # First check the environment variable (highest priority)
    env_value = os.environ.get('APP_PASSWORD', '').strip()
    if env_value:
        return env_value

    # Fall back to .env file next to this script
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        try:
            for line in env_file.read_text().splitlines():
                line = line.strip()
                # Skip comments and blank lines
                if not line or line.startswith('#'):
                    continue
                if line.startswith('APP_PASSWORD='):
                    value = line[len('APP_PASSWORD='):].strip()
                    # Strip surrounding quotes if user wrote them
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                        value = value[1:-1]
                    return value
        except OSError as e:
            print(f'[auth] could not read .env: {e}')

    return ''


APP_PASSWORD = _load_app_password()
if APP_PASSWORD:
    print(f'[auth] HTTP Basic auth enabled (password set, length {len(APP_PASSWORD)})')
else:
    print('[auth] HTTP Basic auth DISABLED (no APP_PASSWORD set in env or .env)')


@app.middleware('http')
async def basic_auth_middleware(request: Request, call_next):
    # No password configured -> auth disabled, pass through.
    if not APP_PASSWORD:
        return await call_next(request)

    auth_header = request.headers.get('authorization', '')
    if not auth_header.startswith('Basic '):
        return Response(
            status_code=401,
            content='Authentication required',
            headers={'WWW-Authenticate': 'Basic realm="Forest Fire App"'},
        )

    try:
        encoded = auth_header.split(' ', 1)[1]
        decoded = base64.b64decode(encoded).decode('utf-8')
        # Format is "username:password". We ignore the username and
        # only check the password against APP_PASSWORD.
        _, submitted_password = decoded.split(':', 1)
    except (ValueError, UnicodeDecodeError):
        return Response(
            status_code=401,
            content='Invalid authentication header',
            headers={'WWW-Authenticate': 'Basic realm="Forest Fire App"'},
        )

    # secrets.compare_digest is constant-time: it doesn't leak how many
    # characters of the password matched via timing differences.
    if not secrets.compare_digest(submitted_password, APP_PASSWORD):
        return Response(
            status_code=401,
            content='Invalid credentials',
            headers={'WWW-Authenticate': 'Basic realm="Forest Fire App"'},
        )

    return await call_next(request)


# ---------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------
#
# Each router is a self-contained module that owns a related cluster of
# endpoints. The prefix gets prepended to every route inside the router.
# Tags group routes in the OpenAPI docs.

app.include_router(health.router,   prefix='/api',          tags=['health'])
app.include_router(predict.router,  prefix='/api/predict',  tags=['predict'])
app.include_router(hexmap.router,   prefix='/api/hexmap',   tags=['hexmap'])


# ---------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------
#
# The vanilla JS frontend lives in ../frontend/. We mount it at /static
# for CSS and JS, and serve specific HTML files for each route. This
# means the whole app is one process at one URL, you visit
# http://localhost:8000 and see the app, with the API hanging off /api/*.
#
# The order matters: API routes are registered first so they take
# precedence. The static file handler only catches requests that
# weren't already handled by an API route.

FRONTEND_DIR = Path(__file__).resolve().parent.parent / 'frontend'

if FRONTEND_DIR.exists():
    # Mount /static so /static/css/styles.css resolves to
    # frontend/css/styles.css. Same for /static/js/*.
    app.mount('/static', StaticFiles(directory=FRONTEND_DIR), name='static')

    # Page routes: each returns the corresponding HTML file. We don't
    # use FastAPI's templating because the frontend is fully static; the
    # browser does all the rendering via vanilla JS.
    @app.get('/', include_in_schema=False)
    async def home():
        return FileResponse(FRONTEND_DIR / 'index.html')

    # The map page replaces the old slider-based predict and the old
    # grid-based simulate. Both URLs route to it for backward compatibility
    # with bookmarks; the page itself does ignition-and-spread.
    @app.get('/map', include_in_schema=False)
    async def map_page():
        return FileResponse(FRONTEND_DIR / 'map.html')

    @app.get('/predict', include_in_schema=False)
    async def predict_page():
        return FileResponse(FRONTEND_DIR / 'map.html')

    @app.get('/simulate', include_in_schema=False)
    async def simulate_page():
        return FileResponse(FRONTEND_DIR / 'map.html')

    @app.get('/about', include_in_schema=False)
    async def about_page():
        return FileResponse(FRONTEND_DIR / 'about.html')
else:
    # Frontend folder missing. API still works, just no UI
    @app.get('/', include_in_schema=False)
    async def root():
        return {
            'name': 'Forest Fire Prediction API',
            'docs': '/docs',
            'health': '/api/health',
            'note': 'Frontend folder not found at ../frontend/',
        }
