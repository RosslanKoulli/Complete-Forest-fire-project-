/*
 * Utilities
 * ===========
 *
 * Small helpers used across pages. Two categories:
 *
 *   1. Formatters -- turn numbers into display strings
 *   2. DOM helpers -- shorter alternatives to verbose DOM API
 *
 * The DOM helpers are the closest thing this codebase has to a
 * framework. The pattern is "create an element with classes, attrs,
 * and children in one expression" rather than the standard
 * createElement/setAttribute/appendChild dance, which is needlessly
 * verbose for the volume of DOM construction we need.
 */

// ============ Formatters ============

/** 0.733 -> "73%"  (decimals=0)  or "73.3%" (decimals=1) */
export function fmtPct(p, decimals = 0) {
    return `${(p * 100).toFixed(decimals)}%`;
}

/** Force a leading sign. 0.082 -> "+0.082", -0.214 -> "-0.214" */
export function fmtSigned(n, decimals = 3) {
    const v = n.toFixed(decimals);
    return n >= 0 ? `+${v}` : v;
}

/** Bucket a probability into low/medium/high. Thresholds match the UI. */
export function riskBand(p) {
    if (p >= 0.7) return 'high';
    if (p >= 0.4) return 'medium';
    return 'low';
}

// ============ DOM helpers ============

/**
 * Shorthand for document.querySelector / querySelectorAll.
 * Usage: $('#my-id'), $('.my-class')
 */
export const $ = (selector, parent = document) =>
    parent.querySelector(selector);

export const $$ = (selector, parent = document) =>
    Array.from(parent.querySelectorAll(selector));

/**
 * Create a DOM element with attributes and children in one go.
 *
 * Examples:
 *   el('div', { className: 'card' }, 'Hello')
 *   el('button', { onclick: () => alert('hi') }, 'Click me')
 *   el('div', {}, [
 *     el('span', {}, 'nested'),
 *     el('span', {}, 'children'),
 *   ])
 *
 * Special attribute names:
 *   - className: sets class
 *   - dataset: object whose keys become data-* attributes
 *   - on{event}: attaches an event listener (e.g. onclick, onchange)
 *   - style: object of CSS properties
 *   - any other: setAttribute
 */
export function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);

    for (const [key, value] of Object.entries(attrs)) {
        if (value === null || value === undefined || value === false) continue;

        if (key === 'className') {
            node.className = value;
        } else if (key === 'dataset') {
            for (const [dk, dv] of Object.entries(value)) {
                node.dataset[dk] = dv;
            }
        } else if (key === 'style' && typeof value === 'object') {
            Object.assign(node.style, value);
        } else if (key.startsWith('on') && typeof value === 'function') {
            // onclick, onchange, oninput, etc.
            node.addEventListener(key.slice(2).toLowerCase(), value);
        } else if (key === 'innerHTML') {
            node.innerHTML = value;
        } else {
            node.setAttribute(key, value);
        }
    }

    appendChildren(node, children);
    return node;
}

function appendChildren(parent, children) {
    if (!children) return;
    if (!Array.isArray(children)) children = [children];

    for (const child of children) {
        if (child === null || child === undefined || child === false) continue;
        if (Array.isArray(child)) {
            appendChildren(parent, child);
        } else if (child instanceof Node) {
            parent.appendChild(child);
        } else {
            parent.appendChild(document.createTextNode(String(child)));
        }
    }
}

/**
 * Replace all children of an element. Cleaner than `innerHTML = ''`
 * because it avoids re-parsing HTML and preserves event listeners
 * on the parent.
 */
export function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
}

/** Replace all children with the given new children. */
export function setChildren(node, children) {
    clear(node);
    appendChildren(node, children);
}

// ============ Misc ============

/**
 * Debounce a function. Returns a wrapped version that only fires after
 * `delay` ms have passed since the last call.
 *
 * Used on the prediction form: the slider fires onChange continuously
 * while dragged. We only want to call /api/predict after the user
 * stops moving the slider. Without debouncing, we'd hammer the backend
 * with one request per slider tick.
 */
export function debounce(fn, delay) {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => fn(...args), delay);
    };
}

/**
 * Set the active state of nav links based on current pathname.
 * Called from each page so the right nav link is highlighted.
 */
export function setActiveNavLink() {
    const path = window.location.pathname;
    $$('.nav-link').forEach((link) => {
        const href = link.getAttribute('href');
        if (href === path || (href !== '/' && path.startsWith(href))) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
}

/**
 * Status-dot updater. Called from every page on load to ping /api/health
 * and reflect the result in the nav status indicator.
 */
export async function updateStatusDot() {
    const dot = $('#status-dot');
    const text = $('#status-text');
    if (!dot) return;

    try {
        const res = await fetch('/api/health');
        if (res.ok) {
            dot.className = 'status-dot ok';
            if (text) text.textContent = 'API connected';
        } else {
            dot.className = 'status-dot down';
            if (text) text.textContent = 'API offline';
        }
    } catch {
        dot.className = 'status-dot down';
        if (text) text.textContent = 'API offline';
    }
}

/** Model accent colour table -- keeps colours consistent across pages. */
export const MODEL_COLORS = {
    'Random Forest':  '#10b981',
    'XGBoost':        '#3b82f6',
    'Neural Network': '#f97316',
};

/** Map a full model name to its CSS stripe class. */
export const MODEL_STRIPES = {
    'Random Forest':  'rf',
    'XGBoost':        'xgb',
    'Neural Network': 'nn',
};
