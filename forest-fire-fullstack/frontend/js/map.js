/*
 * Map page logic
 * ================
 *
 * The full interactive flow:
 *
 *   1. User opens the page; default Leaflet map centred on the Mediterranean.
 *   2. User draws a rectangle with the rectangle tool (top-left of the map).
 *   3. We tessellate the rectangle with H3 hexagons at the configured resolution.
 *   4. Each hex's centroid is sent to /api/hexmap/predict_region_stream over a
 *      WebSocket so we can render a real progress bar and cancel mid-flight.
 *   5. As predictions stream back, each hex polygon is added to the map
 *      immediately - the user sees the rectangle fill in.
 *   6. User clicks any coloured hex to ignite a fire there.
 *   7. We open a WebSocket to /api/hexmap/simulate_stream and feed it the
 *      hexes, neighbours, ignition point, and spread parameters.
 *   8. As frames stream back, we recolour each hex by its current state
 *      (unburnt / burning / burnt).
 *
 * Cancellation: drawing a new rectangle or clicking Clear Selection while a
 * prediction is in progress closes the WebSocket. The backend stops on the
 * next hex boundary - no more wasted weather lookups, no late-arriving
 * hexes appearing in a cleared selection.
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
    // result from /api/hexmap/predict_region_stream.
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

    // Active prediction WebSocket. Held so Clear Selection / new
    // rectangle can close it and stop the backend mid-flight rather
    // than letting the prediction continue silently in the background.
    predictionSocket: null,

    // Set true when the user explicitly cancels (Clear Selection,
    // Cancel button, new rectangle). The message handler checks this
    // before applying late-arriving frames - belt and braces, since
    // we also close the socket which stops the backend.
    predictionAborted: false,
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
// the zoom is high enough that each hex covers enough pixels. We also
// rescale the label font on every zoom so labels stay proportional to
// the hex size on screen (otherwise they look comically large when
// zoomed out and stay tiny when zoomed in).
map.on('zoomend', () => {
    updateHexLabelVisibility();
    updateHexLabelScale();
});

// Belt-and-braces fix for the tooltip-stuck-on-held-click glitch.
// Even with sticky:false on the tooltip binding, very fast mouse
// gestures (press + drag + release) can occasionally leave a tooltip
// open after the cursor has moved off the hex. We dismiss any open
// tooltips on map drag start and on any mousedown outside a hex so
// the user never has to fight to clear one.
map.on('mousedown dragstart', () => {
    dismissAllTooltips();
});

function dismissAllTooltips() {
    for (const hex of state.hexes.values()) {
        if (hex.polygon && hex.polygon.isTooltipOpen && hex.polygon.isTooltipOpen()) {
            hex.polygon.closeTooltip();
        }
    }
}


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
    // User finished drawing a rectangle. runPredictionFlow handles
    // cancelling any in-flight prediction first, so we don't need to
    // do that here - just kick off the new flow.
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
// Prediction flow (streaming WebSocket)
// ============================================================
//
// Called when the user finishes drawing a rectangle. Generates hexes
// client-side, opens a WebSocket to the streaming endpoint, and renders
// each hex as the backend finishes its prediction.
//
// The flow returns a Promise that resolves when the stream completes
// OR is cancelled, so the L.Draw.Event.CREATED `await` resolves cleanly
// in both cases.

async function runPredictionFlow() {
    // Cancel any prior in-flight prediction. This closes the active
    // socket, hides the progress UI, and sets predictionAborted so any
    // late-arriving messages get ignored.
    closePrediction();

    // Dismiss any leftover warning overlay from a prior failed attempt.
    hideMapWarning();

    // Stop any in-progress simulation; the new selection invalidates it.
    closeSimulation();

    // Clear previously rendered hexes
    clearHexes();

    if (!state.selectedBounds) return;

    const { ok, cells, count } = generateHexagons(
        state.selectedBounds, state.hexResolution
    );

    if (!ok) {
        // The selection at this resolution exceeds the hex cap. We show
        // a prominent across-the-map warning rather than just a status
        // line below the map - the previous wording was easy to miss
        // and people would just sit waiting for predictions that were
        // never going to arrive.
        showMapWarning(
            'Selection too large',
            `That rectangle contains ${count.toLocaleString()} hexagons at H3 resolution ${state.hexResolution}. ` +
            `The limit is ${HEX_LIMIT}. Either draw a smaller rectangle or lower the resolution slider.`
        );
        return;
    }

    if (cells.length === 0) {
        showStatus('No hexagons found in that selection.', 'warn');
        return;
    }

    // A selection inside the limit but still substantial: warn the user
    // in the loading bar so they know to be patient. Bigger selections
    // mean more Open-Meteo calls (one per uncached hex), which can take
    // tens of seconds for a few hundred hexes.
    const sizeHint = cells.length > 200
        ? `Large selection (${cells.length} hexagons) - this may take a while.`
        : cells.length > 50
            ? `${cells.length} hexagons - moderate wait expected.`
            : `${cells.length} hexagons - should be quick.`;

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

    // The accumulator collects predictions as they stream in so that
    // the post-completion summary (reliability badge, source breakdown)
    // can be computed once at the end. The hex polygons are already
    // rendered incrementally onto the map; this is just for the
    // aggregate stats.
    const collected = [];
    const sourceStats = { total_api_calls: 0, cache_hits: 0, fallback_count: 0 };

    // Open the streaming WebSocket. Same host as the page, upgraded to
    // wss when the page is served over HTTPS.
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${proto}//${window.location.host}/api/hexmap/predict_region_stream`;

    return new Promise((resolve) => {
        let ws;
        try {
            ws = new WebSocket(wsUrl);
        } catch (err) {
            showStatus(`Could not open prediction stream: ${err.message}`, 'error');
            resolve();
            return;
        }

        state.predictionSocket = ws;
        state.predictionInFlight = true;
        state.predictionAborted = false;

        // Show the progress UI immediately with the size-aware hint -
        // the user gets feedback even before the first hex completes
        // (during connection + request validation on the backend), and
        // the message tells them to expect a longer wait if they drew
        // a large rectangle.
        showPredictionProgress(0, cells.length, sizeHint);

        // Hide the regular status message; we'll show it again on done.
        $('#status-message').style.display = 'none';

        ws.onopen = () => {
            if (state.predictionAborted) {
                try { ws.close(); } catch {}
                return;
            }
            ws.send(JSON.stringify({
                hexes: hexPayload,
                region: state.region,
                // Time-window fields: only included when the user has
                // enabled the time window. Sending nulls keeps the
                // backend's default behaviour (weather ending today).
                start_date: state.timeWindowEnabled ? state.windowStartDate : null,
                duration_days: state.timeWindowEnabled ? state.durationDays : null,
            }));
            showPredictionProgress(0, cells.length, sizeHint);
        };

        ws.onmessage = (event) => {
            // Belt-and-braces: if the user cancelled while a message
            // was in transit, drop it on the floor. The socket close
            // is the primary mechanism; this guards against the brief
            // window between cancel and close completing.
            if (state.predictionAborted) return;

            let data;
            try {
                data = JSON.parse(event.data);
            } catch (e) {
                return;
            }

            if (data.type === 'started') {
                showPredictionProgress(0, data.total, 'Computing predictions...');
                return;
            }

            if (data.type === 'progress') {
                showPredictionProgress(data.completed, data.total, 'Computing predictions...');
                // Render this hex's polygon immediately so the user
                // sees the rectangle fill in as the stream progresses.
                // null prediction = the hex failed at FWI/transform/
                // predict; skip silently (same behaviour as the POST
                // endpoint which drops failed hexes).
                if (data.prediction) {
                    renderHexPolygon(data.prediction);
                    collected.push(data.prediction);
                }
                return;
            }

            if (data.type === 'done') {
                sourceStats.total_api_calls = data.total_api_calls || 0;
                sourceStats.cache_hits = data.cache_hits || 0;
                sourceStats.fallback_count = data.fallback_count || 0;
                finalizePredictionFlow(collected, sourceStats);
                hidePredictionProgress();
                state.predictionInFlight = false;
                state.predictionSocket = null;
                try { ws.close(); } catch {}
                resolve();
                return;
            }

            if (data.type === 'error') {
                showStatus(`Prediction failed: ${data.error}`, 'error');
                hidePredictionProgress();
                state.predictionInFlight = false;
                state.predictionSocket = null;
                try { ws.close(); } catch {}
                resolve();
                return;
            }
        };

        ws.onerror = () => {
            // Only treat as an error if we weren't expecting the
            // socket to close (i.e. the user didn't cancel). Otherwise
            // a user-initiated close shows up here as an error event
            // followed by a close event, which would surface a spurious
            // error message.
            if (!state.predictionAborted) {
                showStatus('Network error during prediction.', 'error');
            }
            hidePredictionProgress();
            state.predictionInFlight = false;
            if (state.predictionSocket === ws) {
                state.predictionSocket = null;
            }
            resolve();
        };

        ws.onclose = () => {
            // onclose always fires - either after 'done', after an error,
            // or after the user cancelled. The first two paths already
            // resolved; this branch handles the cancel case (and any
            // unexpected disconnect).
            if (state.predictionSocket === ws) {
                state.predictionSocket = null;
            }
            if (state.predictionInFlight) {
                state.predictionInFlight = false;
                hidePredictionProgress();
            }
            resolve();
        };
    });
}


// Cancel any in-flight prediction. Safe to call when there's nothing in
// flight - it's a no-op in that case. Used by Clear Selection, the
// Cancel button in the progress UI, and beforeunload.
function closePrediction() {
    if (state.predictionSocket) {
        state.predictionAborted = true;
        try {
            state.predictionSocket.close();
        } catch (e) {
            // Closing an already-closing socket throws on some browsers;
            // the underlying connection is dead either way.
        }
        state.predictionSocket = null;
    }
    state.predictionInFlight = false;
    hidePredictionProgress();
}


// After all hex predictions have streamed in, compute the aggregate
// post-processing: neighbour graph, stats strip, reliability badge,
// dim mask, and the final status line summarising the run.
function finalizePredictionFlow(predictions, sourceStats) {
    // Build the neighbour graph now that all hexes are in state.hexes
    buildNeighbourGraph();

    showStats({
        hexCount: predictions.length,
        step: 0,
        burning: 0,
        burnt: 0,
        burntPct: 0,
    });

    // No-fuel hexes are excluded from the reliability average. They
    // score 100% domain confidence (the model didn't run, so there's
    // no extrapolation), but including that in the average would
    // inflate it misleadingly - the user wants to know "how reliable
    // are the actual fire predictions in my selection", not "how
    // confident am I that water doesn't burn".
    const predicted = predictions.filter(p => p.risk_band !== 'no_fuel');
    const reliabilities = predicted.map(p => p.overall_reliability);
    const avgReliability = reliabilities.length
        ? reliabilities.reduce((a, b) => a + b, 0) / reliabilities.length
        : 0;
    const lowCount = reliabilities.filter(r => r < 40).length;
    showConfidenceBadge(avgReliability, predicted.length, lowCount);

    updateDimMask();

    // Status line with the same source-summary format as before
    const parts = [];
    if (sourceStats.total_api_calls > 0) parts.push(`${sourceStats.total_api_calls} live weather lookups`);
    if (sourceStats.cache_hits > 0) parts.push(`${sourceStats.cache_hits} cache hits`);
    if (sourceStats.fallback_count > 0) parts.push(`${sourceStats.fallback_count} fallback values`);

    const noFuelCount = predictions.length - predicted.length;
    if (noFuelCount > 0) {
        parts.push(`${noFuelCount} no-fuel hexes`);
    }
    if (predicted.length > 0) {
        parts.push(`average reliability ${avgReliability.toFixed(0)}%`);
        if (lowCount > 0) {
            parts.push(`${lowCount} hexes flagged as out-of-distribution`);
        }
    }
    const sourceSummary = parts.length ? '(' + parts.join(', ') + ').' : '';

    showStatus(
        `Predicted ${predictions.length} hexagons. ${sourceSummary} Click any hexagon to start a fire there.`,
        'success'
    );
}


// ============================================================
// Map warning overlay (full-width over the map)
// ============================================================
//
// Used for blocking conditions that the small status card under the
// map is too discreet for - chiefly when the user's selection contains
// too many hexagons to predict, or when they tried to bump the H3
// resolution up too high for the current selection. The overlay sits
// on top of the map with high z-index so it's impossible to miss; it
// stays until dismissed.

function showMapWarning(title, body) {
    const wrap = $('#map-warning');
    if (!wrap) return;
    const titleEl = $('#map-warning-title');
    const bodyEl  = $('#map-warning-body');
    if (titleEl) titleEl.textContent = title;
    if (bodyEl)  bodyEl.textContent  = body;
    wrap.style.display = 'flex';
}

function hideMapWarning() {
    const wrap = $('#map-warning');
    if (wrap) wrap.style.display = 'none';
}


// ============================================================
// Progress UI helpers
// ============================================================

function showPredictionProgress(completed, total, label) {
    const wrap = $('#prediction-progress');
    if (!wrap) return;
    wrap.style.display = 'block';

    const fill = $('#prediction-progress-fill');
    const text = $('#prediction-progress-text');
    const labelEl = $('#prediction-progress-label');

    const pct = total > 0 ? (completed / total) * 100 : 0;
    if (fill) fill.style.width = `${pct}%`;
    if (text) text.textContent = `${completed} / ${total} (${Math.round(pct)}%)`;
    if (labelEl && label) labelEl.textContent = label;
}

function hidePredictionProgress() {
    const wrap = $('#prediction-progress');
    if (wrap) wrap.style.display = 'none';
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

    // Tooltip on hover: shows the prediction details. We pass the hex
    // wrapper as well as the prediction so the tooltip can use the
    // current displayProb (which reflects the active model selection)
    // for the "Avg" line, while still showing the unmodified rf/xgb/nn
    // numbers from the backend underneath.
    //
    // Note: no `sticky: true` here. Sticky tooltips follow the cursor
    // and, on some browsers, get into a wedged state when the left
    // mouse button is held down then released over a different hex -
    // the tooltip ends up anchored to the click point and won't close
    // until the user clicks elsewhere. Plain hover tooltips anchor to
    // the polygon and close cleanly on mouseleave.
    polygon.bindTooltip(buildTooltipHtml(prediction), {
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

    // Initial display values reflect the active model selection. For
    // the default 'average' model these match the backend's avg_probability,
    // but we recompute fresh from rf/xgb/nn so the value is correct
    // regardless of what was last in avg_probability. These live on the
    // wrapper (never on the prediction) so model switches are pure
    // render operations.
    const initialProb = probabilityForActiveModel(prediction);
    const initialBand = prediction.risk_band === 'no_fuel'
        ? 'no_fuel'
        : riskBandForProb(initialProb);

    const hexWrapper = {
        polygon,
        labelMarker,
        prediction,
        displayProb: initialProb,
        displayBand: initialBand,
        currentState: 0,   // 0=unburnt, 1=burning, 2=burnt, 3=firebreak
    };

    // Re-bind the tooltip now that we have the wrapper - this lets
    // buildTooltipHtml read displayProb/displayBand off the wrapper
    // so the "active model" line is correct from the very first hover.
    polygon.unbindTooltip();
    polygon.bindTooltip(buildTooltipHtml(prediction, hexWrapper), {
        direction: 'top',
        opacity: 0.95,
    });

    state.hexes.set(prediction.h3_index, hexWrapper);
}


// Constants for hex appearance. Both fill and stroke at 0.85 so the
// basemap shows through faintly, giving a sense of where on the world
// the prediction is, but the fire-risk colour is the dominant signal.
const HEX_FILL_OPACITY = 0.85;
const HEX_STROKE_OPACITY = 0.85;

// Minimum map zoom at which per-hex percentage labels become visible.
// Below this zoom, labels are hidden entirely - the alternative (tiny
// text squashed inside tiny hexes) renders as overlapping noise that
// hurts more than it helps. The colour gradient still conveys risk at
// any zoom; numbers are an enhancement only at usable zoom levels.
//
// Tuned to zoom 8 because at zoom 7 a default Leaflet viewport shows
// a country-sized area, and even resolution-7 hexes (~5 km) appear at
// only ~25px on screen - too small to fit a "65%" label without
// overlapping its neighbours. Zoom 8 doubles that and gives labels
// room to breathe.
const LABEL_MIN_ZOOM = 8;

// Font-size scaling. Labels grow slowly with zoom so they stay readable
// as the user zooms in (without an upper cap, however, labels at low
// zoom expand past the hex size and overlap each other - that's the
// "ugly when zoomed out" bug).
//
// The cap matters more than the growth rate. We pin the font between
// LABEL_MIN_PX and LABEL_MAX_PX no matter what zoom you're at:
//
//   - Below LABEL_MIN_ZOOM: labels are hidden entirely (updateHexLabelVisibility)
//   - At LABEL_MIN_ZOOM exactly: font is LABEL_MIN_PX (10px)
//   - For each zoom step above: font grows by LABEL_GROWTH_PX_PER_ZOOM
//   - Above the cap zoom: font stays at LABEL_MAX_PX (16px)
//
// Linear growth rather than the original multiplicative one keeps the
// numbers feeling consistent. 1.5 pixels per zoom is enough that the
// user feels the labels respond to zoom, but small enough that they
// never explode in sisze.
const LABEL_MIN_PX = 12;
const LABEL_MAX_PX = 16;
const LABEL_GROWTH_PX_PER_ZOOM = 1.5;


function updateHexLabelScale() {
    // Set a CSS variable on the map container that drives the font-size
    // of every .hex-label inside it. Updating one variable is dramatically
    // cheaper than rebuilding every label's divIcon HTML on each zoomend.
    //
    // Clamp between LABEL_MIN_PX and LABEL_MAX_PX so labels never
    // shrink to illegible at very low zoom (we hide them entirely
    // below LABEL_MIN_ZOOM via updateHexLabelVisibility anyway) nor
    // grow past the hex outline at very high zoom.
    const zoom = map.getZoom();
    const zoomDelta = Math.max(0, zoom - LABEL_MIN_ZOOM);
    const rawSize = LABEL_MIN_PX + zoomDelta * LABEL_GROWTH_PX_PER_ZOOM;
    const fontSize = Math.min(LABEL_MAX_PX, Math.max(LABEL_MIN_PX, rawSize));
    map.getContainer().style.setProperty('--hex-label-size', `${fontSize.toFixed(1)}px`);
}


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


function buildTooltipHtml(p, hex) {
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

    // Compact tooltip showing the three model probabilities, the
    // currently-selected model's value (highlighted), the inputs,
    // and the reliability breakdown.
    const i = p.inputs || {};

    // The highlighted "active model" line uses the hex wrapper's
    // displayProb / displayBand when present. These reflect the active
    // model selection without mutating the underlying prediction
    // object - so switching between RF / XGB / NN / Average always
    // shows the right value and switches are fully reversible. When
    // no hex is passed (e.g. very first render before recolouring),
    // we fall back to recomputing the average fresh from rf/xgb/nn.
    const displayProb = (hex && hex.displayProb !== undefined)
        ? hex.displayProb
        : (p.rf_probability + p.xgb_probability + p.nn_probability) / 3.0;
    const displayBand = (hex && hex.displayBand)
        ? hex.displayBand
        : p.risk_band;

    // The label next to the highlighted probability reflects whichever
    // model the user has currently selected. Reading from state keeps
    // this in sync without having to thread the model name through.
    const activeLabel = (
        state.activeModel === 'rf'  ? 'RF'  :
        state.activeModel === 'xgb' ? 'XGB' :
        state.activeModel === 'nn'  ? 'NN'  :
        'Avg'
    );

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
                ${p.h3_index.slice(-6)} - ${displayBand} risk
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums;">
                RF:&nbsp;&nbsp;${(p.rf_probability * 100).toFixed(0)}%<br>
                XGB:&nbsp;${(p.xgb_probability * 100).toFixed(0)}%<br>
                NN:&nbsp;&nbsp;${(p.nn_probability * 100).toFixed(0)}%<br>
                <span style="color: #f97316;">${activeLabel}: ${(displayProb * 100).toFixed(0)}%</span>
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
    if (state.predictionInFlight) {
        showStatus('Predictions are still streaming in. Wait for them to finish before igniting.', 'warn');
        return;
    }

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
        // H3 resolution: the backend uses this to scale the per-frame
        // sleep so bigger hexes (lower resolution) animate slower,
        // giving the visual spread rate a physically-grounded feel
        // rather than racing across the map at the same wall-clock
        // speed regardless of how much area each cell covers.
        hex_resolution: state.hexResolution,
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
    // For the unburnt state we use displayBand (set by recolourHexesForActiveModel)
    // when present, falling back to the backend's original risk_band on the
    // prediction object. This keeps the colour driven by whatever model the
    // user has selected without mutating the prediction itself.
    const baseBand = hex.displayBand || hex.prediction.risk_band;
    const baseClass = `hex-${baseBand.replace(/_/g, '-')}`;
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


// Map the active model selection to the per-prediction probability.
// The backend always returns rf/xgb/nn plus avg as four separate fields
// on each prediction, so model-switching is a pure render operation.
//
// IMPORTANT: when activeModel is 'average', we recompute the mean from
// the three raw model probabilities rather than reading prediction.avg_probability.
// Earlier versions of this code mutated avg_probability in place when
// the user switched models, which meant switching to RF and back to
// Average would leave avg_probability holding the RF value instead of
// the actual three-model mean. Computing fresh from rf/xgb/nn each
// time means switches are fully reversible and the "Average" view
// always reflects the true ensemble average.
function probabilityForActiveModel(prediction) {
    switch (state.activeModel) {
        case 'rf':  return prediction.rf_probability;
        case 'xgb': return prediction.xgb_probability;
        case 'nn':  return prediction.nn_probability;
        case 'average':
        default:
            return (prediction.rf_probability
                  + prediction.xgb_probability
                  + prediction.nn_probability) / 3.0;
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
//
// IMPORTANT: this function does NOT mutate prediction.avg_probability
// or prediction.risk_band. The original backend values are preserved
// so the Average view always recomputes correctly. Instead, the
// hex wrapper stores a `displayProb` and `displayBand` used purely
// for rendering.
function recolourHexesForActiveModel() {
    for (const hex of state.hexes.values()) {
        const p = hex.prediction;
        if (p.risk_band === 'no_fuel') continue;

        // Compute the display values for this model selection. These
        // live on the hex wrapper, never on the prediction object, so
        // the underlying backend data stays pristine.
        const newProb = probabilityForActiveModel(p);
        const newBand = riskBandForProb(newProb);
        hex.displayProb = newProb;
        hex.displayBand = newBand;

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

        // Refresh the tooltip so the highlighted "Avg" matches the
        // display value, but the underlying rf/xgb/nn breakdown still
        // shows the true backend numbers.
        hex.polygon.unbindTooltip();
        hex.polygon.bindTooltip(buildTooltipHtml(p, hex), {
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
        // current selection. Before we kick off the prediction we
        // check whether the new resolution would push the hex count
        // over the limit; if so, we revert the slider to its prior
        // value and show the big warning overlay rather than silently
        // doing nothing. This is point (9): "if the resolution
        // increase is too big for the square that was made, give the
        // user a notification".
        if (field === 'hex_resolution') {
            // Track the value before each change so we can revert.
            let priorValue = parseInt(input.value, 10);
            input.addEventListener('change', () => {
                if (!state.selectedBounds) {
                    priorValue = parseInt(input.value, 10);
                    return;
                }
                const newRes = parseInt(input.value, 10);
                const probe = generateHexagons(state.selectedBounds, newRes);
                if (!probe.ok) {
                    // Revert visually and in state - this resolution
                    // doesn't fit the current selection.
                    input.value = String(priorValue);
                    state.hexResolution = priorValue;
                    if (display) display.textContent = String(priorValue);
                    showMapWarning(
                        'Resolution too high for this selection',
                        `Increasing the H3 resolution to ${newRes} would create ${probe.count.toLocaleString()} hexagons, ` +
                        `which is over the ${HEX_LIMIT} limit. Either keep the current resolution or draw a smaller rectangle first.`
                    );
                    return;
                }
                priorValue = newRes;
                runPredictionFlow();
            });
        }
    });

    // Reset selection. This is the "Clear Selection" button. Critical
    // behaviour here: it must cancel any in-flight prediction so that
    // late-arriving hexes don't repopulate the cleared map. closePrediction
    // closes the WebSocket, which makes the backend stop processing
    // further hexes too.
    $('#reset-btn').addEventListener('click', () => {
        closePrediction();
        drawnItems.clearLayers();
        state.selectedBounds = null;
        clearHexes();
        closeSimulation();
        hideMapWarning();
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

        // Reset the stats strip too. Without this, Burnt Percentage
        // would keep displaying the final value from the cleared
        // simulation even though the map no longer shows any burnt
        // hexes - confusing for the user.
        showStats({
            hexCount: state.hexes.size,
            step: 0,
            burning: 0,
            burnt: 0,
            burntPct: 0,
        });
        showStatus('Burnt areas cleared. Click any hex to ignite again.', 'info');
    });

    // Cancel button inside the progress card. Closes the prediction
    // socket, which stops the backend on the next hex boundary, and
    // clears whatever's been rendered so far - same end state as
    // pressing Clear Selection mid-flight.
    const cancelBtn = $('#cancel-prediction-btn');
    if (cancelBtn) {
        cancelBtn.addEventListener('click', () => {
            closePrediction();
            drawnItems.clearLayers();
            state.selectedBounds = null;
            clearHexes();
            updateDimMask();
            showStatus('Prediction cancelled.', 'info');
        });
    }

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

    // Dismiss button on the map warning overlay (points 8 and 9).
    // The overlay is modal-feeling but not actually modal - the user
    // can still interact with the rest of the page; the button just
    // hides it once they've read the message.
    const warningDismiss = $('#map-warning-dismiss');
    if (warningDismiss) {
        warningDismiss.addEventListener('click', () => hideMapWarning());
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
// The GeoJSON at /static/data/mediterranean_climate.geojson is
// expected to be a direct conversion of a real Köppen-Geiger raster
// (Beck et al. 2018 at 1 km, or Kottek et al. 2006 at 0.5 deg) into
// polygons - see scripts/build_koppen_geojson.py for the converter.
// Each feature has a `properties.zone` of "Csa", "Csb", or "Csc",
// which drives both the fill colour and the in-distribution check.
//
// The colour palette below matches the canonical Kottek 2006 scheme
// used on the Wikipedia Köppen-Geiger world map: yellow / olive /
// darker olive, in zone order.

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

        // Render as a translucent layer matching the canonical
        // Köppen-Geiger colour scheme from Kottek et al. (2006), the
        // same palette used on the standard Wikipedia world map and
        // every Köppen reference figure:
        //
        //   Csa - hot-summer Mediterranean   - pure yellow
        //   Csb - warm-summer Mediterranean  - olive
        //   Csc - cool-summer Mediterranean  - darker olive
        //
        // No dashArray (the older render dashed the borders, which
        // looked tentative - solid borders read "these are the actual
        // boundaries" which matches the 1:1 fidelity of the real
        // Köppen raster the GeoJSON is built from).
        //
        // Fill opacity is bumped slightly higher than the previous
        // value of 0.10. Olive and dark-olive on a dark basemap need
        // more saturation to be legible, but going much higher
        // overwhelms the hex predictions on top.
        const zoneStyle = {
            Csa: { color: '#ffff00', fillColor: '#ffff00', fillOpacity: 0.22, weight: 1.2, opacity: 0.95 },
            Csb: { color: '#c6c600', fillColor: '#c6c600', fillOpacity: 0.22, weight: 1.2, opacity: 0.95 },
            Csc: { color: '#969600', fillColor: '#969600', fillOpacity: 0.22, weight: 1.2, opacity: 0.95 },
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

// Set the initial hex-label font scale based on the default zoom.
// Subsequent zoom events update it via the zoomend handler.
updateHexLabelScale();

// Load the Köppen overlay asynchronously - if it fails the rest of
// the app still works, we just don't show the climate-zone outline
loadKoppenZones();


// Cleanup on page unload. Close both sockets so the backend isn't
// left processing hexes for a page that's been navigated away from.
window.addEventListener('beforeunload', () => {
    closePrediction();
    closeSimulation();
});
