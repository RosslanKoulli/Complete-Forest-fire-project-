# Forest Fire Prediction

A forest-fire ignition prediction and spread simulation system for the Mediterranean climate region, built as a CI601 final-year individual project at the University of Brighton.

This README covers running the project on your own machine for development (Windows or macOS or Linux) and deploying it to a Linux server so other people can reach it from a URL. If you only want to read about *what* the project does and *how it works internally*, that's covered on the `/about` page once the app is running.

---

## What you're looking at

- **Backend:** FastAPI + uvicorn. Loads three trained ML models at startup (Random Forest, XGBoost, MLP), fetches live weather from Open-Meteo per hex, computes the FWI fire-weather indices, returns per-hex ignition probabilities, and streams a cellular-automaton fire spread over WebSocket.
- **Frontend:** Vanilla JavaScript, no build step. Leaflet for the map, leaflet-draw for rectangle selection, H3-js for the hexagonal grid. Loads in any modern browser straight from the FastAPI static-file mount.
- **ML pipeline:** Lives in a separate `forest-fire-prediction/` directory next to this one. Training is done there; this app only consumes the saved `.pkl` files.

Repo layout (relevant bits):

```
forest-fire-fullstack/
├── backend/
│   ├── main.py                  # FastAPI app, lifespan, routes
│   ├── requirements.txt
│   ├── routers/                 # predict, hexmap, health
│   ├── services/                # model_loader, weather_client,
│   │                            # fwi_calculator, hex_simulation,
│   │                            # domain_confidence
│   └── schemas/
├── frontend/
│   ├── index.html               # landing page
│   ├── map.html                 # main map view
│   ├── about.html               # project background
│   ├── js/                      # map.js (the big one), common.js
│   ├── css/styles.css
│   └── data/
│       └── mediterranean_climate.geojson   # Köppen overlay
└── scripts/
    └── build_koppen_geojson.py  # one-off raster→geojson converter
```

The ML training project itself (datasets, training scripts, saved models) sits in a sibling `forest-fire-prediction/` directory. The backend imports the `.pkl` files from there at startup. If you're running on a server, make sure that directory is present at the path the model loader expects (check `services/model_loader.py` for the exact relative path).

---

## Quick start: Windows

These steps assume Python 3.10-3.12 already installed and on PATH. Tested with 3.12.

### 1. Open PowerShell in the project folder

```powershell
cd "C:\path\to\Forest_Fire_version_1.0\forest-fire-fullstack"
```

### 2. Create a virtual environment

**This is not optional.** If you skip it and `pip install` globally, you will end up with NumPy 2.x in one site-packages folder and NumPy 1.x dependencies in another, and the backend will fail to start with a cryptic *"A module that was compiled using NumPy 1.x cannot be run in NumPy 2.4.6"* error. See the Troubleshooting section if you've already done this; otherwise just use the venv from the start.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell refuses with an *execution policy* error, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

You should now see `(.venv)` at the start of your prompt.

### 3. Install dependencies

```powershell
pip install -r backend/requirements.txt
```

This pulls FastAPI, uvicorn, scikit-learn, xgboost, numpy, pandas, scipy, and a handful of supporting libraries. Takes a couple of minutes on a fresh install.

### 4. Run the backend

```powershell
cd backend
python -m uvicorn main:app --reload --port 8000
```

You should see:

```
[startup] Loading models and pipeline...
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 5. Open it

Browse to `http://127.0.0.1:8000`. The landing page loads, click through to the map, draw a rectangle anywhere in the Mediterranean.

---

## Quick start: Linux

This is also what you'd run on the production server. Same workflow as Windows, with one or two distro-specific quirks.

### 1. Make sure you have the basics

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git
```

(Use `dnf` on Fedora/RHEL, `pacman` on Arch; adjust to taste.)

### 2. Get the code on the box

If you have it on GitHub:

```bash
cd ~
git clone <your-repo-url> forest-fire
cd forest-fire/forest-fire-fullstack
```

If you're copying from Windows over SSH, run this **on the Windows machine** from the directory containing the project folder:

```powershell
scp -r forest-fire-fullstack <user>@<server-ip>:~/forest-fire/
```

### 3. Virtual environment + install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 4. Run it

```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Note: `--host 0.0.0.0` (not `127.0.0.1`) so the server accepts connections from outside the box itself.

If your server has a firewall (most VPS images do), open the port:

```bash
sudo ufw allow 8000/tcp
```

Now from any machine: `http://<server-ip>:8000` should load the app.

---

## Making it run forever (Linux production)

Holding an SSH session open with uvicorn running isn't how you actually deploy this. Use systemd to run it as a managed service that auto-restarts on crash and starts on boot.

### 1. Create a service file

```bash
sudo nano /etc/systemd/system/forest-fire.service
```

Paste this, adjusting `User`, `WorkingDirectory`, and `ExecStart` to match your setup:

```ini
[Unit]
Description=Forest Fire Prediction API
After=network.target

[Service]
Type=simple
User=rosslan
WorkingDirectory=/home/rosslan/forest-fire/forest-fire-fullstack/backend
ExecStart=/home/rosslan/forest-fire/forest-fire-fullstack/.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 2. Enable and start it

```bash
sudo systemctl daemon-reload
sudo systemctl enable forest-fire
sudo systemctl start forest-fire
sudo systemctl status forest-fire
```

Logs go to the journal:

```bash
sudo journalctl -u forest-fire -f       # live tail
sudo journalctl -u forest-fire -n 100   # last 100 lines
```

### 3. (Optional) Nginx reverse proxy for a clean URL

If you want `http://your-server-ip/` instead of `http://your-server-ip:8000/`:

```bash
sudo apt install -y nginx
sudo nano /etc/nginx/sites-available/forest-fire
```

Paste:

```nginx
server {
    listen 80;
    server_name _;   # or your domain

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;

        proxy_read_timeout 600s;   # WebSocket sims can run long
    }
}
```

Enable it:

```bash
sudo ln -s /etc/nginx/sites-available/forest-fire /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t                  # test config
sudo systemctl reload nginx
sudo ufw allow 80/tcp
```

Now `http://your-server-ip/` works on port 80, and you can close port 8000 to the outside if you want:

```bash
sudo ufw deny 8000/tcp
```

The `Upgrade`/`Connection` headers and `proxy_read_timeout 600s` are essential. Without them, the WebSocket fire-spread simulation and the prediction-progress stream will drop connection mid-flight.

### 4. (Optional) HTTPS

If you have a domain pointing at the server, free TLS with Let's Encrypt:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

Certbot rewrites the nginx config in place and sets up auto-renewal.

---

## The Köppen-Geiger climate overlay

The `frontend/data/mediterranean_climate.geojson` file that ships with the project is a smoothed approximation of the Csa/Csb/Csc zones. For an accurate 1:1 overlay matching the published Köppen-Geiger maps, regenerate it from a real climate raster.

### 1. Download one of the standard rasters

- **Beck et al. (2018):** 1 km resolution, present-day, recommended.
  https://www.gloh2o.org/koppen/ → `Beck_KG_V1_present_0p0083.tif`

- **Kottek et al. (2006):** 0.5° (~50 km), matches the canonical paper.
  http://koeppen-geiger.vu-wien.ac.at/present.htm → `KG_World_v1.1.txt`

### 2. Run the converter

```bash
pip install rasterio shapely
python scripts/build_koppen_geojson.py <raster-file> \
    --output frontend/data/mediterranean_climate.geojson
```

The Beck dataset uses class codes 8/9/10 for Csa/Csb/Csc (script defaults). The Kottek dataset numbering varies, so pass `--csa-code N --csb-code N --csc-code N` to override.

Restart the backend (or hard-refresh the browser if running with `--reload`) and the new overlay appears with the correct boundaries.

---

## Troubleshooting

### `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.4.6`

You're not in a virtual environment, and there's a NumPy 2 somewhere in your user-level site-packages that's shadowing the NumPy 1 the rest of the stack expects.

Cleanest fix: create the venv (instructions above), do everything inside it. If you absolutely can't, force-remove all the NumPy copies and install a 1.x one:

```powershell
pip uninstall -y numpy
pip uninstall -y numpy
pip uninstall -y numpy
pip install "numpy<2"
```

Run `pip uninstall numpy` repeatedly until pip says "not installed". Python can install the same package in multiple locations and each `uninstall` only finds one copy.

### `ssh: Permission denied`

The username, password, or the auth method is wrong. Linux usernames are case-sensitive and almost always lowercase. If your friend handed you a six-word hyphenated phrase like `morbidly-walk-impure-disrupt-gore-last`, double-check it's an SSH password rather than a Tailscale auth key (Tailscale uses that exact format).

After three failed attempts the server will close the connection. Wait a minute before retrying, since `fail2ban` may also rate-limit your IP for several minutes.

### Backend starts but the map page returns 404

The static-file mount is missing the frontend directory. Check that `backend/main.py` finds `frontend/` at `FRONTEND_DIR = Path(__file__).resolve().parent.parent / 'frontend'`. If you've copied only `backend/` to the server, copy `frontend/` too.

### `WebSocket connection failed` in the browser console

If you're behind an nginx reverse proxy, you forgot the `Upgrade`/`Connection` headers and the `proxy_read_timeout`. See the nginx config block above.

If you're not behind a proxy, the most likely cause is a firewall blocking port 8000 (or whichever port you ran uvicorn on). `sudo ufw allow 8000/tcp` opens it.

### Predictions hang forever / Open-Meteo timing out

The free Open-Meteo API has a soft rate limit. A few hundred hexes in one prediction is fine; thousands will cause some calls to fail. The backend retries with a fallback, but if the network is genuinely unreachable from the server, predictions will hang. Test with:

```bash
curl 'https://api.open-meteo.com/v1/forecast?latitude=40&longitude=-8&current=temperature_2m'
```

from inside an SSH session on the server. If that doesn't return JSON, the server can't reach Open-Meteo at all.

### `Permission denied` writing to `frontend/data/mediterranean_climate.geojson`

When running the converter script, you're writing as a different user than the one that owns the file. On the server: `sudo chown -R $USER:$USER frontend/data/` once.

### Hex labels overlap when zoomed out

Fixed in the latest map.js. Labels hide entirely below zoom level 8 and are font-capped at 16px so they can't outgrow their hex. If you still see overlap, you're on an older `map.js`; replace it.

---

## Context

- CI601 Individual Project, University of Brighton, 2025-2026
- v1.0: full-stack iteration (FastAPI + vanilla JS), replacing an earlier Streamlit prototype
- Scope: Mediterranean climate region (Köppen Csa, Csb, Csc)
- Not an operational forecasting service. See the `/about` page for the full scope statement
