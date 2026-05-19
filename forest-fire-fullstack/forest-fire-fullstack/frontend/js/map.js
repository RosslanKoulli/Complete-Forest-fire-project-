/*
 * Map page logic
 * ================
 *
 * The full interactive flow:
 *
 *   1. User opens the page; default Leaflet map centred on the Mediterranean.
 *   2. User draws a rectangle with the rectangle tool (top-left of the map).
 *   3. We tessellate the rectangle with H3 hexagons at the configured resolution.
 *   4. Each hex's centroid is sent to /api/hexmap/predict_region.
 *   5. The response is per-hex fire probabilities; we colour each hex
 *      polygon by its risk band (low/medium/high).
 *   6. User clicks any coloured hex to ignite a fire there.
 *   7. We open a WebSocket to /api/hexmap/simulate_stream and feed it the
 *      hexes, neighbours, ignition point, and spread parameters.
 *   8. As frames stream back, we recolour each hex by its current state
 *      (unburnt / burning / burnt).
 *
 * Three external libraries are loaded as globals via the HTML page:
 *   - L (Leaflet): the mapping library
 *   - L.Draw: the drawing tool extension
 *   - h3 (h3-js): the hexagon tessellation library
 *
 * Our own code uses ES module imports for utils and api, but we treat
 * Leaflet and h3-js as globals because that is how their CDN builds work.
 */

import {
    $, $$, setActiveNavLink, updateStatusDot,
} from '/static/js/utils.js';

// ============================================================
// State
// ============================================================
//
// One central state object. Subviews read from this and write back.
// We do not use any reactive framework, so all state changes go through
// helper functions that update both state and DOM.

// Default start date for the time-window picker: 5 days ago. Open-Meteo's
// archive endpoint has a 5-day lag, and 5 days is close enough to "now"
// to be useful while staying inside what the archive reliably returns.
// The user can pick anywhere from 1940 (archive coverage) up to 7 days
// in the future (forecast quality limit).
function defaultStartDate() {
    const d = new Date();
    d.setDate(d.getDate() - 5);
    return d.toISOString().slice(0, 10);   // YYYY-MM-DD
}


const state = {
    // Configuration (driven by sliders)
    region: 'portugal',
    hexResolution: 7,
    windSpeed: 8.0,
    windDirection: 45,
    vegetationMoisture: 0.3,
    baseSpreadProb: 0.45,

    // Time-window state. The UI used to have a checkbox to make this
    // optional, but the checkbox was removed in favour of always sending
    // a date and duration - so this is always true. Kept as a state
    // field so the rest of the code that conditionally checks it
    // continues to work without invasive refactoring.
    timeWindowEnabled: true,

    // Active model for displayed probabilities. The backend returns all
    // three model probabilities on every prediction; this picks which
    // one drives the hex colours, labels, and simulation. Switching
    // does NOT re-call the backend - we already have all three.
    activeModel: 'average',     // 'average' | 'rf' | 'xgb' | 'nn'
    windowStartDate: defaultStartDate(),
    durationDays: 5,

    // Currently selected rectangle bounds (Leaflet LatLngBounds), or null
    selectedBounds: null,

    // Currently rendered hexes: Map of h3_index -> {polygon, prediction}
    // polygon is the Leaflet polygon object; prediction is the per-hex
    // result from /api/hexmap/predict_region.
    hexes: new Map(),

    // Neighbour graph for the current hex set: h3_index -> [neighbour h3 indices]
    neighbours: new Map(),

    // WebSocket connection during a simulation, or null
    simulationSocket: null,

    // Captured frames from the most recent simulation. Each entry is
    // {step, states, elapsed_hours, current_day, burning, burnt}.
    // Used by the day-scrubber to let the user drag through history.
    simulationFrames: [],

    // Whether a prediction request is currently in flight (used to
    // disable the rectangle tool to prevent overlapping requests)
    predictionInFlight: false,
};


// ============================================================
// Map setup
// ============================================================
//
// Default view: Mediterranean basin so the user sees Portugal and Algeria
// at once. They can pan and zoom freely.

const DEFAULT_CENTRE = [38.5, 0.0];   // Mediterranean basin
const DEFAULT_ZOOM = 5;

const map = L.map('leaflet-map', {
    center: DEFAULT_CENTRE,
    zoom: DEFAULT_ZOOM,
    zoomControl: true,
    minZoom: 3,
    maxZoom: 12,
});

// Tile layer. We use OpenStreetMap's standard tiles; they are free, do
// not require an API key, and have global coverage. The dark filter is
// applied via CSS to fit our colour scheme.
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors',
    className: 'map-tile-dark',
}).addTo(map);

// Update hex percentage labels when the user zooms. At low zoom levels
// the labels are too cluttered to be useful; we add them only when
// the zoom is high enough that each hex covers enough pixels.
map.on('zoomend', () => {
    // updateHexLabelVisibility is defined later in this file; using
    // function declaration so it's hoisted and available here.
    updateHexLabelVisibility();
});


// ============================================================
// Drawing tool (rectangle selection)
// ============================================================
//
// Leaflet.Draw provides the rectangle drawing tool. We configure it
// to allow exactly one rectangle at a time; drawing a new one clears
// the previous selection.

const drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);

const drawControl = new L.Control.Draw({
    draw: {
        // Only the rectangle tool is enabled. Disable the others so
        // the toolbar is clean.
        polygon: false,
        polyline: false,
        circle: false,
        marker: false,
        circlemarker: false,
        rectangle: {
            shapeOptions: {
                color: '#f97316',
                weight: 2,
                fillOpacity: 0.05,
            },
        },
    },
    edit: {
        featureGroup: drawnItems,
        edit: false,    // disable edit and remove buttons; we provide our own
        remove: false,
    },
});
map.addControl(drawControl);

map.on(L.Draw.Event.CREATED, async (event) => {
    // User finished drawing a rectangle. Clear any previous selection,
    // then run the prediction flow on the new one.
    drawnItems.clearLayers();
    drawnItems.addLayer(event.layer);
    state.selectedBounds = event.layer.getBounds();
    await runPredictionFlow();
});


// ============================================================
// Hex tessellation
// ============================================================
//
// Given a Leaflet LatLngBounds, generate the set of H3 cells that cover
// the rectangle at the configured resolution. h3-js provides
// polygonToCells which takes a polygon and returns the H3 indices.
//
// We cap at 500 hexes (matching the backend's max_length=500) to prevent
// runaway requests when users draw very large rectangles at high resolution.

const HEX_LIMIT = 500;

function generateHexagons(bounds, resolution) {
    // Convert Leaflet bounds to a polygon in [lat, lng] format
    // (h3-js v4 expects lat-lng pairs).
    const sw = bounds.getSouthWest();
    const ne = bounds.getNorthEast();
    const polygon = [
        [sw.lat, sw.lng],
        [sw.lat, ne.lng],
        [ne.lat, ne.lng],
        [ne.lat, sw.lng],
        [sw.lat, sw.lng],
    ];

    // h3.polygonToCells returns an array of H3 indices covering the polygon
    const cells = h3.polygonToCells(polygon, resolution);

    if (cells.length > HEX_LIMIT) {
        // Too many hexes; tell the user to pick a smaller area or lower
        // resolution. We do not silently truncate because that could give
        // misleading results.
        return { ok: false, count: cells.length };
    }

    return { ok: true, cells };
}


// ============================================================
// Prediction flow
// ============================================================
//
// Called when the user finishes drawing a rectangle. Generates hexes,
// posts them to the API, then renders the polygons on the map.

async function runPredictionFlow() {
    if (state.predictionInFlight) {
        return;   // ignore overlapping requests
    }

    // Stop any in-progress simulation; the new selection invalidates it.
    closeSimulation();

    // Clear previously rendered hexes
    clearHexes();

    if (!state.selectedBounds) return;

    const { ok, cells, count } = generateHexagons(
        state.selectedBounds, state.hexResolution
    );

    if (!ok) {
        showStatus(
            `That selection contains ${count} hexagons, which is over the limit of ${HEX_LIMIT}. Pick a smaller area or lower the H3 resolution slider.`,
            'warn'
        );
        return;
    }

    if (cells.length === 0) {
        showStatus('No hexagons found in that selection.', 'warn');
        return;
    }

    showStatus(`Computing predictions for ${cells.length} hexagons...`, 'info');
    state.predictionInFlight = true;

    // Build the request payload. Each hex needs h3_index plus its
    // centroid (lat, lon) for the weather lookup. Region is detected
    // per-hex from the location rather than from a single user-picked
    // value: a rectangle that spans Spain and Algeria gets the right
    // region for each side automatically.
    const hexPayload = cells.map(idx => {
        const [lat, lng] = h3.cellToLatLng(idx);
        return {
            h3_index: idx,
            latitude: lat,
            longitude: lng,
            region: detectRegionForLocation(lat, lng),
        };
    });

    let response;
    try {
        response = await fetch('/api/hexmap/predict_region', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                hexes: hexPayload,
                region: state.region,
                // Time-window fields: only included when the user has
                // enabled the time window. Sending nulls keeps the
                // backend's existing behaviour (weather ending today).
                start_date: state.timeWindowEnabled ? state.windowStartDate : null,
                duration_days: state.timeWindowEnabled ? state.durationDays : null,
            }),
        });
    } catch (err) {
        showStatus(`Network error contacting the API: ${err.message}`, 'error');
        state.predictionInFlight = false;
        return;
    }

    if (!response.ok) {
        const detail = await response.text();
        showStatus(`Prediction failed (HTTP ${response.status}): ${detail}`, 'error');
        state.predictionInFlight = false;
        return;
    }

    const data = await response.json();
    state.predictionInFlight = false;

    // Render each prediction as a coloured hex polygon.
    for (const pred of data.predictions) {
        renderHexPolygon(pred);
    }

    // Build the neighbour graph for the simulation step. We compute it
    // here, after we have all the predicted hexes, so that the simulation
    // doesn't have to re-derive it.
    buildNeighbourGraph();

    // Update the stats strip
    showStats({
        hexCount: data.predictions.length,
        step: 0,
        burning: 0,
        burnt: 0,
        burntPct: 0,
    });

    // Compute reliability summary across the selection and surface it
    // as a badge over the rectangle. Per-hex reliability still appears
    // in tooltips; this is the at-a-glance view.
    //
    // No-fuel hexes are excluded from this average. They score 100%
    // domain confidence (the model didn't run, so there's no
    // extrapolation), but including that in the average would inflate
    // it misleadingly - the user wants to know "how reliable are the
    // ACTUAL fire predictions in my selection", not "how confident am
    // I that water doesn't burn".
    const predictedHexes = data.predictions.filter(p => p.risk_band !== 'no_fuel');
    const reliabilities = predictedHexes.map(p => p.overall_reliability);
    const avgReliability = reliabilities.length
        ? reliabilities.reduce((a, b) => a + b, 0) / reliabilities.length
        : 0;
    const lowCount = reliabilities.filter(r => r < 40).length;
    showConfidenceBadge(avgReliability, predictedHexes.length, lowCount);

    // If the user has the basemap-dim toggle enabled, redraw the mask
    // around the new selection.
    updateDimMask();

    const sourceSummary = describeSources(data);
    showStatus(
        `Predicted ${data.predictions.length} hexagons. ${sourceSummary} Click any hexagon to start a fire there.`,
        'success'
    );
}


function describeSources(data) {
    const parts = [];
    if (data.total_api_calls > 0) parts.push(`${data.total_api_calls} live weather lookups`);
    if (data.cache_hits > 0) parts.push(`${data.cache_hits} cache hits`);
    if (data.fallback_count > 0) parts.push(`${data.fallback_count} fallback values`);

    // Same filtering as the badge: no-fuel hexes don't represent
    // model predictions, so they shouldn't count toward the average
    // reliability or the out-of-distribution flag count.
    if (data.predictions && data.predictions.length > 0) {
        const predicted = data.predictions.filter(p => p.risk_band !== 'no_fuel');
        const noFuelCount = data.predictions.length - predicted.length;
        if (noFuelCount > 0) {
            parts.push(`${noFuelCount} no-fuel hexes`);
        }
        if (predicted.length > 0) {
            const reliabilities = predicted.map(p => p.overall_reliability);
            const avgReliability = reliabilities.reduce((a, b) => a + b, 0) / reliabilities.length;
            const lowCount = reliabilities.filter(r => r < 40).length;
            parts.push(`average reliability ${avgReliability.toFixed(0)}%`);
            if (lowCount > 0) {
                parts.push(`${lowCount} hexes flagged as out-of-distribution`);
            }
        }
    }

    return parts.length ? '(' + parts.join(', ') + ').' : '';
}


// ============================================================
// Rendering hex polygons
// ============================================================

function renderHexPolygon(prediction) {
    const boundary = h3.cellToBoundary(prediction.h3_index);
    // h3-js returns boundary as [[lat, lng], ...]; Leaflet wants the same.
    //
    // Hex opacity is constant across all hexes. We use 0.85 (slightly
    // less than full opaque) so the underlying basemap is faintly
    // visible through the colour. Reliability is shown separately via
    // the badge over the selection rectangle, not via per-hex fading.
    //
    // Class names use hyphens for CSS, but the risk_band field from
    // the backend uses underscores ('no_fuel'). Normalise here so
    // both ends remain idiomatic in their own languages.
    const cssClass = `hex-${prediction.risk_band.replace(/_/g, '-')}`;

    // Tag the prediction with its Köppen zone (or null if outside any
    // Mediterranean polygon). The tooltip uses this to flag hexes
    // outside the training climate zone. Computed once at render time
    // and cached on the prediction object - no recomputation needed
    // when the model selector changes.
    const [centroidLat, centroidLng] = h3.cellToLatLng(prediction.h3_index);
    prediction.koppen_zone = pointInKoppenZone(centroidLat, centroidLng);

    const polygon = L.polygon(boundary, {
        className: cssClass,
        weight: 1,
        fillOpacity: HEX_FILL_OPACITY,
        opacity: HEX_STROKE_OPACITY,
    });

    // Tooltip on hover: shows the prediction details.
    polygon.bindTooltip(buildTooltipHtml(prediction), {
        sticky: true,
        direction: 'top',
        opacity: 0.95,
    });

    // Click handler. No-fuel hexes (water, high altitude) cannot be
    // ignited - the simulation refuses because there is no fuel to
    // burn. Other hexes proceed to the normal ignition flow.
    polygon.on('click', () => {
        if (prediction.risk_band === 'no_fuel') {
            const reason = prediction.no_fuel_reason === 'water'
                ? 'open water'
                : 'above the tree line';
            showStatus(`This hex is ${reason} - no fuel to ignite. Try a hex over vegetation.`, 'warn');
            return;
        }
        igniteHex(prediction.h3_index);
    });

    polygon.addTo(map);

    // Per-hex percentage label. Only added for hexes that ran the
    // ML model; no_fuel hexes don't get a label because the number
    // would just be "0%" everywhere and clutter the view.
    let labelMarker = null;
    if (prediction.risk_band !== 'no_fuel') {
        labelMarker = createHexLabel(prediction);
    }

    state.hexes.set(prediction.h3_index, {
        polygon,
        labelMarker,
        prediction,
        currentState: 0,   // 0=unburnt, 1=burning, 2=burnt, 3=firebreak
    });
}


// Constants for hex appearance. Both fill and stroke at 0.85 so the
// basemap shows through faintly, giving a sense of where on the world
// the prediction is, but the fire-risk colour is the dominant signal.
const HEX_FILL_OPACITY = 0.85;
const HEX_STROKE_OPACITY = 0.85;

// Minimum map zoom at which per-hex percentage labels become visible.
// At lower zooms the hexes are too small for text and the labels
// would overlap into noise. At resolution 7 (~5 km hexes) zoom 8+
// gives enough pixel area; at resolution 5 (~25 km hexes) zoom 6+
// works. The thresholds below are a compromise that works reasonably
// at the default resolution.
const LABEL_MIN_ZOOM = 8;


function createHexLabel(prediction) {
    // The hex centroid is where the label sits. Leaflet's L.marker
    // with a divIcon gives us full HTML/CSS control; the .hex-label
    // CSS class handles styling, including the text shadow that
    // keeps the number legible on any of the four fill colours.
    const [lat, lng] = h3.cellToLatLng(prediction.h3_index);
    const pct = Math.round(prediction.avg_probability * 100);

    const marker = L.marker([lat, lng], {
        icon: L.divIcon({
            html: `<div class="hex-label">${pct}%</div>`,
            className: '',         // suppress Leaflet's default styling
            iconSize: null,
            iconAnchor: [14, 6],   // approximate centring
        }),
        interactive: false,         // never block clicks on the hex below
        keyboard: false,
    });

    // Only add to the map if we're zoomed in enough. updateHexLabelVisibility
    // toggles this for all hexes on zoom changes.
    if (map.getZoom() >= LABEL_MIN_ZOOM) {
        marker.addTo(map);
    }
    return marker;
}


function updateHexLabelVisibility() {
    // Called on zoom changes. Adds labels to the map when zoomed in,
    // removes them when zoomed out. We don't destroy the markers
    // (creating them is expensive) - just toggle their map membership.
    const shouldShow = map.getZoom() >= LABEL_MIN_ZOOM;
    for (const hex of state.hexes.values()) {
        if (!hex.labelMarker) continue;
        const onMap = map.hasLayer(hex.labelMarker);
        if (shouldShow && !onMap) {
            hex.labelMarker.addTo(map);
        } else if (!shouldShow && onMap) {
            map.removeLayer(hex.labelMarker);
        }
    }
}


function buildTooltipHtml(p) {
    // No-fuel hexes get a different tooltip - showing probabilities
    // would be misleading (they're synthetic zeroes, not model output).
    // We surface the reason and the elevation instead.
    if (p.risk_band === 'no_fuel') {
        const reasonLabel = p.no_fuel_reason === 'water'
            ? 'Water / sea level'
            : p.no_fuel_reason === 'high_altitude'
                ? 'Above tree line'
                : 'No fuel';
        const elev = p.elevation !== null && p.elevation !== undefined
            ? `${Math.round(p.elevation)}m`
            : 'unknown';
        return `
            <div style="font-family: Inter, sans-serif; font-size: 12px; line-height: 1.5;">
                <div style="font-weight: 500; margin-bottom: 4px;">
                    ${p.h3_index.slice(-6)} - no fuel
                </div>
                <div style="font-size: 11px; color: #cbd5e1;">
                    ${reasonLabel}<br>
                    Elevation: ${elev}<br>
                    The ML models were not run for this hex because
                    no forest fuel is present.
                </div>
            </div>
        `;
    }

    // Compact tooltip showing the three model probabilities, the average,
    // the inputs, and the reliability breakdown.
    const i = p.inputs || {};

    // Colour the reliability values: green if high, amber if marginal, red if low.
    const reliabilityColour = (v) =>
        v >= 70 ? '#34d399' : v >= 40 ? '#fbbf24' : '#f87171';

    // Köppen zone string for display: the actual zone name if inside
    // one of our Csa/Csb/Csc polygons, or a warning if outside the
    // training climate region. p.koppen_zone is set by renderHexPolygon.
    let koppenLine = '';
    if (p.koppen_zone) {
        koppenLine = `<div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid #444; font-size: 11px; color: #a1a1aa;">
            Climate: <span style="color: #34d399;">${p.koppen_zone}</span> Mediterranean
        </div>`;
    } else {
        koppenLine = `<div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid #444; font-size: 11px; color: #fbbf24;">
            ⚠ Outside Köppen Csa/Csb/Csc - the model was trained on Mediterranean climates only
        </div>`;
    }

    return `
        <div style="font-family: Inter, sans-serif; font-size: 12px; line-height: 1.5;">
            <div style="font-weight: 500; margin-bottom: 4px;">
                ${p.h3_index.slice(-6)} - ${p.risk_band} risk
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums;">
                RF:&nbsp;&nbsp;${(p.rf_probability * 100).toFixed(0)}%<br>
                XGB:&nbsp;${(p.xgb_probability * 100).toFixed(0)}%<br>
                NN:&nbsp;&nbsp;${(p.nn_probability * 100).toFixed(0)}%<br>
                <span style="color: #f97316;">Avg: ${(p.avg_probability * 100).toFixed(0)}%</span>
            </div>
            <div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid #444;">
                <div style="font-size: 11px; margin-bottom: 2px; color: #a1a1aa;">Reliability</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; font-size: 11px;">
                    Domain conf: <span style="color: ${reliabilityColour(p.domain_confidence)};">${p.domain_confidence.toFixed(0)}%</span><br>
                    Model agree: <span style="color: ${reliabilityColour(p.model_agreement)};">${p.model_agreement.toFixed(0)}%</span><br>
                    <strong>Overall:&nbsp;&nbsp;<span style="color: ${reliabilityColour(p.overall_reliability)};">${p.overall_reliability.toFixed(0)}%</span></strong>
                </div>
            </div>
            ${koppenLine}
            <div style="margin-top: 6px; font-size: 11px; color: #71717a;">
                T:${i.temperature?.toFixed(0)}C
                RH:${i.relative_humidity?.toFixed(0)}%
                W:${i.wind_speed?.toFixed(0)}km/h<br>
                FFMC:${i.FFMC} DMC:${i.DMC} DC:${i.DC} ISI:${i.ISI}
            </div>
        </div>
    `;
}


// ============================================================
// Basemap dimming overlay
// ============================================================
//
// Optional visual aid: darken the world outside the user's current
// selection, focusing attention on the predicted region. Implemented
// as a single Leaflet polygon covering the whole world with a hole
// punched out for the selection rectangle.
//
// Off by default because it can feel heavy-handed; users toggle it
// on if they want the focus effect.

function clearHexes() {
    for (const hex of state.hexes.values()) {
        map.removeLayer(hex.polygon);
        // Labels: removeLayer is safe whether or not the label is on
        // the map (it's a no-op when not added). Saves us tracking
        // visibility state separately for cleanup.
        if (hex.labelMarker) {
            map.removeLayer(hex.labelMarker);
        }
    }
    state.hexes.clear();
    state.neighbours.clear();
    state.simulationFrames = [];
    hideConfidenceBadge();
    hideStats();
    hideDayScrubber();
}

// ============================================================
// Confidence badge over the selected rectangle
// ============================================================
//
// A small floating label anchored to the top edge of the user's
// rectangle, showing the average reliability across all hexes in the
// selection. Coloured green / amber / red so the user can see at a
// glance whether the predictions in the visible area are trustworthy.
//
// Implemented as a Leaflet Marker with a divIcon, so we get full
// HTML/CSS control and Leaflet handles re-positioning on pan/zoom.

let confidenceBadge = null;   // Leaflet marker, or null

// Basemap dim mask state (declared here alongside the badge state)
let dimMask = null;           // Leaflet polygon, or null
let dimEnabled = false;

function showConfidenceBadge(avgReliability, hexCount, lowCount) {
    // Remove any previous badge before drawing the new one
    hideConfidenceBadge();

    if (!state.selectedBounds) return;

    // Pick a colour by reliability. Same thresholds the tooltip uses:
    // >=70 green, 40-69 amber, <40 red.
    let bg, fg, label;
    if (avgReliability >= 70) {
        bg = '#10b981';   // emerald
        fg = '#022c22';
        label = 'high reliability';
    } else if (avgReliability >= 40) {
        bg = '#f59e0b';   // amber
        fg = '#451a03';
        label = 'moderate reliability';
    } else {
        bg = '#ef4444';   // red
        fg = '#450a0a';
        label = 'low reliability - out of distribution';
    }

    // Out-of-distribution hex count, only shown when relevant
    const oodSuffix = lowCount > 0
        ? `<div style="font-size: 10px; opacity: 0.85; margin-top: 2px;">${lowCount} of ${hexCount} hexes flagged</div>`
        : '';

    const html = `
        <div style="
            background: ${bg};
            color: ${fg};
            font-family: Inter, sans-serif;
            font-size: 12px;
            font-weight: 600;
            padding: 6px 10px;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            white-space: nowrap;
            border: 1px solid rgba(0, 0, 0, 0.3);
            text-align: center;
        ">
            <div>Avg reliability: ${avgReliability.toFixed(0)}%</div>
            <div style="font-size: 10px; font-weight: 500; opacity: 0.9;">${label}</div>
            ${oodSuffix}
        </div>
    `;

    // Anchor at the top-centre of the rectangle, slightly above so the
    // badge sits in the strip above the selection rather than over it.
    const ne = state.selectedBounds.getNorthEast();
    const sw = state.selectedBounds.getSouthWest();
    const anchorLat = ne.lat;
    const anchorLng = (ne.lng + sw.lng) / 2;

    confidenceBadge = L.marker([anchorLat, anchorLng], {
        icon: L.divIcon({
            html,
            className: 'confidence-badge-wrapper',
            iconSize: null,        // let CSS size it naturally
            iconAnchor: [70, 50],  // approximate centring; not pixel-perfect
                                    // but close enough for the visual
        }),
        interactive: false,        // don't intercept clicks meant for hexes
        keyboard: false,
    });
    confidenceBadge.addTo(map);
}

function hideConfidenceBadge() {
    if (confidenceBadge) {
        map.removeLayer(confidenceBadge);
        confidenceBadge = null;
    }
}

function updateDimMask() {
    // Remove existing mask
    if (dimMask) {
        map.removeLayer(dimMask);
        dimMask = null;
    }

    if (!dimEnabled || !state.selectedBounds) return;

    // World bounds (a generous outer rectangle)
    const worldRing = [
        [85, -180],
        [85, 180],
        [-85, 180],
        [-85, -180],
        [85, -180],
    ];

    // Inner hole = the user's selection. Order matters for the hole
    // to be cut out: outer ring clockwise, holes counter-clockwise (or
    // vice versa). Leaflet handles this automatically when you pass
    // multiple rings.
    const sw = state.selectedBounds.getSouthWest();
    const ne = state.selectedBounds.getNorthEast();
    const selectionRing = [
        [sw.lat, sw.lng],
        [sw.lat, ne.lng],
        [ne.lat, ne.lng],
        [ne.lat, sw.lng],
        [sw.lat, sw.lng],
    ];

    dimMask = L.polygon([worldRing, selectionRing], {
        color: '#000000',
        weight: 0,
        fillColor: '#000000',
        fillOpacity: 0.55,
        interactive: false,    // user can still click hexes through it
    });
    dimMask.addTo(map);

    // Make sure the hexes are drawn on top of the mask. Leaflet's
    // default z-order respects insertion order, so we re-add each
    // hex's polygon to bring it to the front.
    for (const hex of state.hexes.values()) {
        hex.polygon.bringToFront();
    }
}


// ============================================================
// Neighbour graph for the simulation
// ============================================================
//
// The hex simulation needs to know which hexes are adjacent. h3-js gives
// us gridDisk(idx, 1) which returns the hex itself plus its 6 neighbours.
// We filter to just the neighbours that are also in our prediction set.

function buildNeighbourGraph() {
    state.neighbours.clear();
    for (const idx of state.hexes.keys()) {
        // gridDisk(idx, 1) includes idx itself plus all neighbours within 1 step
        const ring = h3.gridDisk(idx, 1);
        const inSet = ring.filter(h => h !== idx && state.hexes.has(h));
        state.neighbours.set(idx, inSet);
    }
}


// ============================================================
// Ignition and simulation streaming
// ============================================================

function igniteHex(h3Index) {
    if (state.simulationSocket) {
        // Already running; user must reset first
        showStatus('A simulation is already running. Click "Stop simulation" first.', 'warn');
        return;
    }

    // Reset all hex states to unburnt before igniting (in case there was
    // a previous simulation showing residual burnt hexes)
    for (const hex of state.hexes.values()) {
        hex.currentState = 0;
        applyHexClass(hex);
    }

    // Build the simulation start payload
    const hexes = [];
    for (const [idx, h] of state.hexes) {
        hexes.push({
            h3_index: idx,
            latitude: h3.cellToLatLng(idx)[0],
            longitude: h3.cellToLatLng(idx)[1],
            fire_probability: h.prediction.avg_probability,
        });
    }

    const neighboursObj = {};
    for (const [idx, ns] of state.neighbours) {
        neighboursObj[idx] = ns;
    }

    const startPayload = {
        hexes,
        neighbours: neighboursObj,
        ignition: [h3Index],
        base_spread_prob: state.baseSpreadProb,
        wind_speed: state.windSpeed,
        wind_direction: state.windDirection,
        vegetation_moisture: state.vegetationMoisture,
        burn_duration: 3,
        max_steps: 100,
        // Time-window mode: only set when the user has enabled it.
        // If null/undefined, the simulation runs to natural burnout
        // exactly like before.
        duration_days: state.timeWindowEnabled ? state.durationDays : null,
        hours_per_step: 12.0,
    };

    // Open the WebSocket. We use the same host as the page, automatically
    // upgrading to wss:// when the page is loaded over https.
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${proto}//${window.location.host}/api/hexmap/simulate_stream`;

    const ws = new WebSocket(wsUrl);
    state.simulationSocket = ws;
    $('#reset-sim-btn').disabled = false;
    $('#clear-burnt-btn').disabled = false;

    ws.onopen = () => {
        // Clear any previous frame buffer when starting a new simulation
        state.simulationFrames = [];
        hideDayScrubber();
        ws.send(JSON.stringify(startPayload));
        showStatus(`Fire ignited at ${h3Index.slice(-6)}. Watching the spread...`, 'info');
    };

    ws.onmessage = (event) => {
        let data;
        try {
            data = JSON.parse(event.data);
        } catch (e) {
            return;
        }

        if (data.error) {
            showStatus(`Simulation error: ${data.error}`, 'error');
            closeSimulation();
            return;
        }

        if (data.done) {
            showStatus('Simulation finished. Drag the scrubber below to replay.', 'success');
            closeSimulation();
            // Reveal the scrubber so the user can replay frames
            showDayScrubber();
            return;
        }

        // Apply state changes per frame
        if (data.states) {
            for (const [idx, stateCode] of Object.entries(data.states)) {
                const hex = state.hexes.get(idx);
                if (!hex) continue;
                if (hex.currentState !== stateCode) {
                    hex.currentState = stateCode;
                    applyHexClass(hex);
                }
            }
        }

        // Buffer the full frame for the scrubber. We capture a shallow
        // copy of the per-hex state map so future state changes don't
        // mutate the captured frame.
        state.simulationFrames.push({
            step: data.step,
            states: { ...(data.states || {}) },
            burning: data.burning,
            burnt: data.burnt,
            elapsed_hours: data.elapsed_hours || 0,
            current_day: data.current_day || 0,
        });

        showStats({
            hexCount: state.hexes.size,
            step: data.step,
            burning: data.burning,
            burnt: data.burnt,
            burntPct: state.hexes.size
                ? Math.round((data.burnt / state.hexes.size) * 100)
                : 0,
            // Time-window mode fields. These are 0 when the simulation
            // is running in step mode; the showStats function decides
            // whether to display them based on whether the time window
            // is currently active.
            elapsedHours: data.elapsed_hours || 0,
            currentDay: data.current_day || 0,
        });
    };

    ws.onerror = () => {
        showStatus('WebSocket error during simulation.', 'error');
        closeSimulation();
    };

    ws.onclose = () => {
        if (state.simulationSocket === ws) {
            state.simulationSocket = null;
            $('#reset-sim-btn').disabled = true;
        }
    };
}


function applyHexClass(hex) {
    // Map the hex's current sim state to a CSS class on the polygon path.
    // The risk_band may use underscores (no_fuel) so we normalise to
    // hyphens for the CSS class name, matching the convention used at
    // hex creation time.
    const baseClass = `hex-${hex.prediction.risk_band.replace(/_/g, '-')}`;
    const stateMap = {
        0: baseClass,            // unburnt - back to risk colour
        1: 'hex-burning',
        2: 'hex-burnt',
        3: 'hex-firebreak',
    };
    const cls = stateMap[hex.currentState] || baseClass;
    hex.polygon.setStyle({ className: cls });

    // setStyle does not always re-apply the className on existing paths in
    // all Leaflet versions, so we also poke the underlying SVG path element.
    const path = hex.polygon._path;
    if (path) {
        path.setAttribute('class', `leaflet-interactive ${cls}`);
    }

    // The percentage label is only meaningful for unburnt hexes. While
    // a hex is burning or burnt, the original probability is irrelevant
    // - hide the label. We don't destroy the marker, just remove it
    // from the map, so it can come back if the user resets.
    if (hex.labelMarker) {
        if (hex.currentState === 0 && map.getZoom() >= LABEL_MIN_ZOOM) {
            if (!map.hasLayer(hex.labelMarker)) hex.labelMarker.addTo(map);
        } else {
            if (map.hasLayer(hex.labelMarker)) map.removeLayer(hex.labelMarker);
        }
    }
}


// Map the active model selection to the per-prediction field that
// holds its probability. The backend always returns rf/xgb/nn plus
// avg as four separate fields, so model-switching is a pure render
// operation.
function probabilityForActiveModel(prediction) {
    switch (state.activeModel) {
        case 'rf':  return prediction.rf_probability;
        case 'xgb': return prediction.xgb_probability;
        case 'nn':  return prediction.nn_probability;
        case 'average':
        default:
            return prediction.avg_probability;
    }
}


// Mirror of the backend's _risk_band function. We need it client-side
// because switching the model changes the active probability, which
// can shift the hex across the low/medium/high bands.
function riskBandForProb(p) {
    if (p >= 0.7) return 'high';
    if (p >= 0.4) return 'medium';
    return 'low';
}


// Re-render every hex's colour and label using the active model's
// probability. Triggered when the user switches model selector. Skips
// no-fuel hexes (those don't have model output) and skips hexes that
// are currently burning/burnt (the simulation owns their colour).
function recolourHexesForActiveModel() {
    for (const hex of state.hexes.values()) {
        const p = hex.prediction;
        if (p.risk_band === 'no_fuel') continue;

        // Update the prediction object in place so future operations
        // (tooltip rebuilds, ignition) see the new band.
        const newProb = probabilityForActiveModel(p);
        const newBand = riskBandForProb(newProb);
        p.avg_probability = newProb;
        p.risk_band = newBand;

        // If this hex is currently in an unburnt state, refresh its
        // colour and label. If it's burning or burnt, leave it alone -
        // the simulation will reset it back to the new colour on
        // clear-burnt.
        if (hex.currentState === 0) {
            applyHexClass(hex);
        }

        // Update the label text. createHexLabel encodes the % in the
        // divIcon HTML, so the cleanest update is to rebuild the icon.
        if (hex.labelMarker) {
            const pct = Math.round(newProb * 100);
            hex.labelMarker.setIcon(L.divIcon({
                html: `<div class="hex-label">${pct}%</div>`,
                className: '',
                iconSize: null,
                iconAnchor: [14, 6],
            }));
        }

        // Refresh the tooltip so the highlighted Avg matches.
        hex.polygon.unbindTooltip();
        hex.polygon.bindTooltip(buildTooltipHtml(p), {
            sticky: true,
            direction: 'top',
            opacity: 0.95,
        });
    }
}


function closeSimulation() {
    if (state.simulationSocket) {
        try {
            state.simulationSocket.close();
        } catch (e) {}
        state.simulationSocket = null;
    }
    $('#reset-sim-btn').disabled = true;
}


// ============================================================
// Status messages and stats
// ============================================================

function showStatus(message, kind = 'info') {
    const card = $('#status-message');
    const body = $('#status-message-body');
    const colourMap = {
        info:    'var(--text-secondary)',
        success: 'var(--risk-low)',
        warn:    'var(--risk-med)',
        error:   'var(--risk-high)',
    };
    body.style.color = colourMap[kind] || 'var(--text-secondary)';
    body.textContent = message;
    card.style.display = 'block';
}


// ============================================================
// Day scrubber (replay through captured simulation frames)
// ============================================================
//
// Once a simulation completes (or is stopped), the user can drag a
// slider to replay any frame. The slider's max is the number of
// captured frames; the value picks which frame's per-hex state map
// to apply.

function showDayScrubber() {
    const total = state.simulationFrames.length;
    if (total === 0) return;

    const wrap = $('#day-scrubber');
    const input = $('#day-scrubber-input');
    const totalLabel = $('#scrubber-total');

    wrap.style.display = 'block';
    input.max = String(total - 1);
    input.value = String(total - 1);   // start at the final frame
    totalLabel.textContent = String(total - 1);
    applyScrubberFrame(total - 1);
}

function hideDayScrubber() {
    const wrap = $('#day-scrubber');
    if (wrap) wrap.style.display = 'none';
}

function applyScrubberFrame(index) {
    const frame = state.simulationFrames[index];
    if (!frame) return;

    // Apply this frame's hex states (no animation, just snap).
    for (const hex of state.hexes.values()) {
        const newState = frame.states[hex.prediction.h3_index];
        if (newState !== undefined && newState !== hex.currentState) {
            hex.currentState = newState;
            applyHexClass(hex);
        }
    }

    // Update the labels on the scrubber itself
    $('#scrubber-frame').textContent = String(index);
    $('#scrubber-hours').textContent = String(Math.round(frame.elapsed_hours));

    // Update the stats strip to match the scrubbed frame
    showStats({
        hexCount: state.hexes.size,
        step: frame.step,
        burning: frame.burning,
        burnt: frame.burnt,
        burntPct: state.hexes.size
            ? Math.round((frame.burnt / state.hexes.size) * 100)
            : 0,
        elapsedHours: frame.elapsed_hours,
        currentDay: frame.current_day,
    });
}


function showStats(s) {
    $('#stats-strip').style.display = 'grid';
    $('#stat-hex-count').textContent = s.hexCount;
    $('#stat-burning').textContent = s.burning;
    $('#stat-burnt').textContent = s.burnt;
    $('#stat-burnt-pct').textContent = `${s.burntPct}%`;

    // The "Step" cell repurposes itself in time-window mode to show
    // "Day X" instead of the abstract step counter. The label below
    // the number toggles to match. We look up the cell's label
    // sibling rather than using a separate ID, so the markup stays
    // simple.
    const stepNumber = $('#stat-step');
    const stepLabel = stepNumber.parentElement.querySelector('.stat-strip-label');
    if (state.timeWindowEnabled && s.currentDay > 0) {
        stepNumber.textContent = `${s.currentDay} / ${state.durationDays}`;
        stepLabel.textContent = 'Day';
    } else {
        stepNumber.textContent = s.step;
        stepLabel.textContent = 'Step';
    }
}


function hideStats() {
    $('#stats-strip').style.display = 'none';
}


// ============================================================
// Control wiring
// ============================================================

function wireControls() {
    // Model picker. Switching the model re-colours the hexes using the
    // probability for that model (no re-prediction needed - the backend
    // returns all three on every call). When no model is explicitly
    // selected this stays at 'average' which uses avg_probability.
    $$('#model-picker .segmented-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            $$('#model-picker .segmented-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.activeModel = btn.dataset.model;
            // Re-render hex colours with the new model's probabilities.
            // No backend call needed since we already have all three.
            recolourHexesForActiveModel();
        });
    });

    // Sliders. Same convention as the older predict page: data-input
    // names a state field, data-display names the value display element.
    $$('input[type="range"][data-input]').forEach(input => {
        const field = input.dataset.input;
        const display = $(`[data-display="${field}"]`);
        const step = parseFloat(input.step);

        input.addEventListener('input', () => {
            const v = parseFloat(input.value);

            // Map slider field names to state field names.
            const stateField = {
                wind_speed:           'windSpeed',
                wind_direction:       'windDirection',
                vegetation_moisture:  'vegetationMoisture',
                base_spread_prob:     'baseSpreadProb',
                hex_resolution:       'hexResolution',
                duration_days:        'durationDays',
            }[field];

            if (stateField) {
                state[stateField] = v;
            }

            if (display) {
                if (field === 'vegetation_moisture') {
                    display.textContent = Math.round(v * 100);
                } else if (step >= 1) {
                    display.textContent = v.toFixed(0);
                } else if (field === 'base_spread_prob') {
                    display.textContent = v.toFixed(2);
                } else {
                    display.textContent = v.toFixed(1);
                }
            }
        });

        // If the resolution slider changes, regenerate hexes for the
        // current selection.
        if (field === 'hex_resolution') {
            input.addEventListener('change', () => {
                if (state.selectedBounds) {
                    runPredictionFlow();
                }
            });
        }
    });

    // Reset selection
    $('#reset-btn').addEventListener('click', () => {
        drawnItems.clearLayers();
        state.selectedBounds = null;
        clearHexes();
        closeSimulation();
        $('#status-message').style.display = 'none';
        // Remove any active dim mask too
        updateDimMask();
    });

    // Stop simulation
    $('#reset-sim-btn').addEventListener('click', () => {
        closeSimulation();
        // Reset all hexes to their unburnt-coloured state
        for (const hex of state.hexes.values()) {
            hex.currentState = 0;
            applyHexClass(hex);
        }
        showStatus('Simulation stopped. Click another hex to ignite again.', 'info');
        $('#clear-burnt-btn').disabled = true;
    });

    // Clear burnt areas. Resets the simulation state on every hex back
    // to the original prediction colour, without clearing the rectangle
    // or re-running prediction. Useful when the user wants to look at
    // the predicted heatmap again after running a simulation.
    $('#clear-burnt-btn').addEventListener('click', () => {
        for (const hex of state.hexes.values()) {
            hex.currentState = 0;
            applyHexClass(hex);
        }
        // Drop the captured frames - they belong to the cleared sim
        state.simulationFrames = [];
        hideDayScrubber();
        $('#clear-burnt-btn').disabled = true;
        showStatus('Burnt areas cleared. Click any hex to ignite again.', 'info');
    });

    // Basemap dimming toggle. When checked, darken the world outside
    // the current selection.
    const dimToggle = $('#dim-basemap-toggle');
    if (dimToggle) {
        dimToggle.addEventListener('change', () => {
            dimEnabled = dimToggle.checked;
            updateDimMask();
        });
    }

    // Day scrubber. Drag-to-replay through captured simulation frames.
    // 'input' event fires on every drag step for live feedback;
    // applyScrubberFrame is light enough to run at 60fps.
    const scrubberInput = $('#day-scrubber-input');
    if (scrubberInput) {
        scrubberInput.addEventListener('input', () => {
            const idx = parseInt(scrubberInput.value, 10);
            if (!isNaN(idx)) applyScrubberFrame(idx);
        });
    }

    // Time-window date picker. The toggle checkbox was removed - the
    // time window is always-on now. Changing the date re-runs the
    // prediction so the colours reflect the new period's weather.
    const windowDateInput = $('#window-start-date');
    if (windowDateInput) {
        windowDateInput.value = state.windowStartDate;
        windowDateInput.addEventListener('change', () => {
            state.windowStartDate = windowDateInput.value;
            if (state.selectedBounds) {
                runPredictionFlow();
            }
        });
    }
}


// ============================================================
// Init
// ============================================================
// Köppen-Geiger Mediterranean climate zones
// ============================================================
//
// We overlay the Csa/Csb/Csc zones on the map and use them to flag
// hexes that fall outside the model's training climate region. This
// is the project's defensible scope boundary: the training data is
// drawn from Mediterranean climates, and predictions outside that
// climate are extrapolating.
//
// The GeoJSON is a simplified hand-curated approximation of the
// boundaries published by Beck et al. (2018). It is NOT a precise
// scientific raster - the polygons are smoothed to ~0.5 degree
// resolution for visual clarity and fast point-in-polygon checks.

let koppenLayer = null;          // Leaflet GeoJSON layer (or null)
let koppenFeatures = null;       // raw feature array for point-in-polygon
let koppenVisible = true;        // toggle state

function pointInPolygonRing(lat, lng, ring) {
    // Ray casting algorithm: count horizontal crossings.
    // Ring is array of [lng, lat] pairs in GeoJSON convention.
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
        const xi = ring[i][0], yi = ring[i][1];
        const xj = ring[j][0], yj = ring[j][1];
        const intersect = ((yi > lat) !== (yj > lat))
            && (lng < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

function pointInKoppenZone(lat, lng) {
    // Returns the Köppen zone string ('Csa', 'Csb', 'Csc') if the
    // point is inside any Mediterranean polygon, or null otherwise.
    if (!koppenFeatures) return null;
    for (const feature of koppenFeatures) {
        const geom = feature.geometry;
        let rings = [];
        if (geom.type === 'Polygon') {
            rings = [geom.coordinates[0]];   // outer ring only
        } else if (geom.type === 'MultiPolygon') {
            rings = geom.coordinates.map(poly => poly[0]);
        }
        for (const ring of rings) {
            if (pointInPolygonRing(lat, lng, ring)) {
                return feature.properties.zone || 'Cs';
            }
        }
    }
    return null;
}

async function loadKoppenZones() {
    try {
        const response = await fetch('/static/data/mediterranean_climate.geojson');
        if (!response.ok) {
            console.warn(`Köppen overlay: HTTP ${response.status}, skipping`);
            return;
        }
        const geojson = await response.json();
        koppenFeatures = geojson.features;

        // Render as a translucent layer. Different fill per zone so the
        // user can distinguish hot-summer (Csa) from warm-summer (Csb)
        // Mediterranean climates.
        const zoneStyle = {
            Csa: { color: '#dc2626', fillColor: '#dc2626', fillOpacity: 0.10, weight: 1.5, dashArray: '4 3' },
            Csb: { color: '#d97706', fillColor: '#d97706', fillOpacity: 0.10, weight: 1.5, dashArray: '4 3' },
            Csc: { color: '#0891b2', fillColor: '#0891b2', fillOpacity: 0.10, weight: 1.5, dashArray: '4 3' },
        };

        koppenLayer = L.geoJSON(geojson, {
            style: (feature) => zoneStyle[feature.properties.zone] || zoneStyle.Csa,
            interactive: false,   // don't intercept clicks meant for hexes
        });
        if (koppenVisible) koppenLayer.addTo(map);
    } catch (e) {
        console.warn('Köppen overlay failed to load:', e);
    }
}

function setKoppenVisible(visible) {
    koppenVisible = visible;
    if (!koppenLayer) return;
    if (visible && !map.hasLayer(koppenLayer)) {
        koppenLayer.addTo(map);
    } else if (!visible && map.hasLayer(koppenLayer)) {
        map.removeLayer(koppenLayer);
    }
}


// ============================================================
// Auto-detect region from hex location
// ============================================================
//
// The trained model uses a "region" categorical feature with values
// for Portugal and Algeria (the two countries in the training set).
// At prediction time we have to provide a value. Auto-detection
// removes this from the user's mental model: clicks in northern Iberia
// get region=portugal, clicks in North Africa get region=algeria,
// everywhere else falls back to the closer of the two (which is
// usually portugal for European hexes, algeria for African ones).

function detectRegionForLocation(lat, lng) {
    // Coarse country-shape filters. Not pixel-accurate but good enough
    // for setting a categorical feature value that the model uses as
    // one input among eleven.
    //
    // Portugal: roughly 37-42N, -9.5 to -6.2W
    if (lat >= 36.9 && lat <= 42.2 && lng >= -9.6 && lng <= -6.0) {
        return 'portugal';
    }
    // Algeria: roughly 19-37N, -8.7 to 12.0E
    if (lat >= 19.0 && lat <= 37.0 && lng >= -8.7 && lng <= 12.0) {
        return 'algeria';
    }
    // Mediterranean Europe (Iberia, southern France, Italy, Greece,
    // Balkans): default to portugal because climate is similar
    if (lat >= 36.0 && lat <= 46.0 && lng >= -10.0 && lng <= 30.0) {
        return 'portugal';
    }
    // North Africa: default to algeria
    if (lat >= 15.0 && lat <= 38.0 && lng >= -18.0 && lng <= 36.0) {
        return 'algeria';
    }
    // Everything else: nearest by latitude band
    return lat >= 30 ? 'portugal' : 'algeria';
}


// ============================================================
// Init
// ============================================================

setActiveNavLink();
updateStatusDot();
wireControls();

// Load the Köppen overlay asynchronously - if it fails the rest of
// the app still works, we just don't show the climate-zone outline
loadKoppenZones();


// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    closeSimulation();
});
