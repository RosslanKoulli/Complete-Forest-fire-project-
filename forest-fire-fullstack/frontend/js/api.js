/*
 * API client
 * ============
 *
 * One function per backend endpoint. All UI code goes through these
 * rather than calling fetch directly, so:
 *   - Errors get handled the same way everywhere
 *   - If we ever change the URL scheme or add auth headers, we only
 *     change it here
 *
 * Everything is async. Callers use await or .then(). No build step,
 * no bundler, just JavaScript that works directly in the browser.
 *
 * Usage:
 *   import { predict, simulate } from '/static/js/api.js';
 *   const result = await predict({ temperature: 32, ... });
 */

/**
 * Custom error class so callers can distinguish API errors from network
 * errors. instanceof ApiError works in JS just like in TypeScript.
 */
export class ApiError extends Error {
    constructor(status, message, details) {
        super(message);
        this.name = 'ApiError';
        this.status = status;
        this.details = details;
    }
}

/**
 * Internal helper: all requests go through this. It centralises
 * JSON serialisation, error handling, and content-type headers.
 */
async function request(path, options = {}) {
    const res = await fetch(path, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...(options.headers || {}),
        },
    });

    if (!res.ok) {
        let details;
        try { details = await res.json(); } catch {}
        throw new ApiError(res.status, `${res.status} ${res.statusText}`, details);
    }

    return res.json();
}

// ============ Health ============

/** Returns { status, models_loaded, ... }. Used by the status dot. */
export const checkHealth = () => request('/api/health');

// ============ Predict ============

/**
 * Run all three models on one input. The request shape is documented
 * by the backend's Pydantic schema, see backend/schemas/predict.py.
 *
 * @param {object} req - PredictionRequest
 * @returns {Promise<object>} PredictionResponse
 */
export const predict = (req) =>
    request('/api/predict', { method: 'POST', body: JSON.stringify(req) });

/** Run one named model. */
export const predictOne = (modelName, req) =>
    request(`/api/predict/${encodeURIComponent(modelName)}`, {
        method: 'POST', body: JSON.stringify(req),
    });

// ============ Simulate ============

/**
 * Full simulation: blocks until done, returns all frames.
 * Use streamSimulation for live frame-by-frame rendering.
 */
export const simulate = (cfg) =>
    request('/api/simulate', { method: 'POST', body: JSON.stringify(cfg) });

/**
 * Stream a simulation over WebSocket. Returns a function that you call
 * to close the connection (e.g. on Pause or page leave).
 *
 * @param {object} cfg          SimulationConfig
 * @param {function} onFrame    Called for each step's frame
 * @param {function} [onDone]   Called when simulation ends
 * @param {function} [onError]  Called on connection or server error
 * @returns {function}          Call to close the WebSocket
 */
export function streamSimulation(cfg, onFrame, onDone, onError) {
    // Build the WebSocket URL from the current page location. This works
    // whether we're on http://localhost or https://my-deployment.app
    // because we just swap the protocol.
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${proto}//${window.location.host}/api/simulate/stream`;

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => ws.send(JSON.stringify(cfg));

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.error) { onError && onError(data.error); return; }
            if (data.done)  { onDone && onDone(); return; }
            onFrame(data);
        } catch (e) {
            onError && onError(`Could not parse frame: ${e}`);
        }
    };

    ws.onerror = () => onError && onError('WebSocket connection failed');
    ws.onclose = () => onDone && onDone();

    return () => {
        if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
            ws.close();
        }
    };
}

// ============ Compare ============

export const getCompareMetrics = () => request('/api/compare/metrics');
export const getFiguresList = () => request('/api/compare/figures');
export const getFigureUrl = (name) => `/api/compare/figure/${name}`;
