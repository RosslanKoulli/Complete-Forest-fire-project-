/*
 * Compare page logic
 * ====================
 *
 * Fetches /api/compare/metrics on load, renders:
 *   - 3 metric cards (one per model, with AUC and stddev)
 *   - ROC curve chart drawn as raw SVG
 *   - Pairwise significance test results
 *
 * The ROC chart is hand-rolled SVG. ~80 lines of JS gets us a chart
 * that matches the design system exactly. Recharts (in the React
 * version) was easier to write but pulled in ~100KB of library code
 * for one chart on one page. Hand-rolled is more code but ships nothing.
 */

import { getCompareMetrics } from '/static/js/api.js';
import {
    $, $$, el, setChildren,
    setActiveNavLink, updateStatusDot,
    MODEL_COLORS, MODEL_STRIPES,
} from '/static/js/utils.js';

// ============ Init ============

setActiveNavLink();
updateStatusDot();
loadMetrics();

async function loadMetrics() {
    try {
        const metrics = await getCompareMetrics();
        $('#loading-msg').style.display = 'none';
        $('#main-content').style.display = 'block';

        if (!metrics.available) {
            $('#compare-subtitle').textContent +=
                ' (Showing placeholder values -- run train_all_models.py to populate.)';
        }

        const comparison = metrics.sections.comparison || {};
        const significance = metrics.sections.significance || {};

        renderMetricCards(comparison);

        // Build the AUC table for the ROC chart
        const aucs = {};
        for (const [model, m] of Object.entries(comparison)) {
            if (m.auc !== undefined) aucs[model] = m.auc;
        }

        if (Object.keys(aucs).length > 0) {
            $('#roc-card').style.display = 'block';
            renderROCChart(aucs);
            renderROCLegend(aucs);
        }

        if (Object.keys(significance).length > 0) {
            $('#sig-card').style.display = 'block';
            renderSignificance(significance);
        }
    } catch (err) {
        $('#loading-msg').style.display = 'none';
        $('#error-state').style.display = 'block';
        $('#error-text').textContent = err.message || 'Unknown error';
    }
}

// ============ Metric cards ============

function renderMetricCards(comparison) {
    const container = $('#metric-cards');
    setChildren(container, Object.entries(comparison).map(([model, m]) => {
        const stripe = MODEL_STRIPES[model] || '';
        return el('div', { className: 'card' }, [
            el('div', { className: 'metric-card-inner' }, [
                el('div', {
                    className: 'metric-card-stripe',
                    style: { backgroundColor: MODEL_COLORS[model] || '#71717a' },
                }),
                el('div', { className: 'metric-card-content' }, [
                    el('div', { className: 'metric-card-name' }, model),
                    el('div', { className: 'metric-card-auc' }, [
                        el('span', { className: 'metric-card-auc-value' },
                          (m.auc !== undefined ? m.auc.toFixed(3) : '--')),
                        el('span', { className: 'metric-card-auc-label' }, 'AUC'),
                    ]),
                    el('div', { className: 'metric-card-detail' }, [
                        m.auc_std !== undefined && el('div', { className: 'metric-card-detail-row' }, [
                            el('span', {}, 'Stddev (5-fold)'),
                            el('span', { className: 'mono' }, `±${m.auc_std.toFixed(3)}`),
                        ]),
                        m.brier !== undefined && el('div', { className: 'metric-card-detail-row' }, [
                            el('span', {}, 'Brier score'),
                            el('span', { className: 'mono' }, m.brier.toFixed(3)),
                        ]),
                    ]),
                ]),
            ]),
        ]);
    }));
}

// ============ ROC chart ============

/**
 * Synthesise a plausible ROC curve from a target AUC.
 *
 * The curve follows the family tpr = 1 - (1-fpr)^k, which is concave
 * and passes through (0,0) and (1,1). Larger k produces higher AUC.
 * We binary-search k to match the target AUC numerically.
 *
 * This is an approximation -- the real fold-mean ROC curve would
 * require per-fold FPR/TPR arrays that the training script doesn't
 * currently export. Documented limitation.
 */
function curveForAUC(targetAUC) {
    let lo = 1, hi = 50;
    for (let i = 0; i < 30; i++) {
        const k = (lo + hi) / 2;
        // Numerically integrate to estimate AUC at this k
        let area = 0;
        const N = 100;
        for (let j = 0; j < N; j++) {
            const fpr1 = j / N, fpr2 = (j + 1) / N;
            const tpr1 = 1 - Math.pow(1 - fpr1, k);
            const tpr2 = 1 - Math.pow(1 - fpr2, k);
            area += (tpr1 + tpr2) / 2 / N;
        }
        if (area < targetAUC) lo = k;
        else hi = k;
    }
    const k = (lo + hi) / 2;

    const points = [];
    for (let i = 0; i <= 50; i++) {
        const fpr = i / 50;
        points.push([fpr, 1 - Math.pow(1 - fpr, k)]);
    }
    return points;
}

/**
 * Render the ROC chart into the SVG element. Coordinate transforms:
 *   - data x: fpr in [0, 1]    -> svg x: [margin.l, W - margin.r]
 *   - data y: tpr in [0, 1]    -> svg y: [H - margin.b, margin.t]   (flipped)
 *
 * SVG y-axis grows downward; we flip so higher tpr = higher on screen.
 */
function renderROCChart(aucs) {
    const W = 600, H = 320;
    const m = { l: 50, r: 16, t: 16, b: 40 };

    const px = (x) => m.l + x * (W - m.l - m.r);
    const py = (y) => H - m.b - y * (H - m.t - m.b);

    const svg = $('#roc-svg');
    setChildren(svg, []);    // clear

    // Helper: create SVG element. createElement doesn't work for SVG;
    // we need createElementNS with the SVG namespace.
    const ns = 'http://www.w3.org/2000/svg';
    const svgEl = (tag, attrs = {}, children = []) => {
        const node = document.createElementNS(ns, tag);
        for (const [k, v] of Object.entries(attrs)) {
            if (v !== null && v !== undefined) node.setAttribute(k, v);
        }
        if (!Array.isArray(children)) children = [children];
        for (const child of children) {
            if (child instanceof Node) node.appendChild(child);
            else if (child !== null && child !== undefined) {
                node.appendChild(document.createTextNode(String(child)));
            }
        }
        return node;
    };

    // Grid lines + tick labels
    for (let i = 0; i <= 5; i++) {
        const t = i / 5;
        // Horizontal grid line
        svg.appendChild(svgEl('line', {
            x1: px(0), y1: py(t), x2: px(1), y2: py(t),
            stroke: '#27272a', 'stroke-width': 0.5,
        }));
        // Vertical grid line
        svg.appendChild(svgEl('line', {
            x1: px(t), y1: py(0), x2: px(t), y2: py(1),
            stroke: '#27272a', 'stroke-width': 0.5,
        }));
        // X-axis tick label
        svg.appendChild(svgEl('text', {
            x: px(t), y: H - 16,
            'font-size': 11,
            'text-anchor': 'middle',
            fill: '#71717a',
        }, t.toFixed(1)));
        // Y-axis tick label
        svg.appendChild(svgEl('text', {
            x: m.l - 8, y: py(t) + 4,
            'font-size': 11,
            'text-anchor': 'end',
            fill: '#71717a',
        }, t.toFixed(1)));
    }

    // Diagonal random-classifier line
    svg.appendChild(svgEl('line', {
        x1: px(0), y1: py(0), x2: px(1), y2: py(1),
        stroke: '#52525b', 'stroke-dasharray': '4 4', 'stroke-width': 1,
    }));

    // Each model's curve as a polyline
    for (const [model, auc] of Object.entries(aucs)) {
        const points = curveForAUC(auc)
            .map(([x, y]) => `${px(x)},${py(y)}`)
            .join(' ');
        svg.appendChild(svgEl('polyline', {
            points,
            fill: 'none',
            stroke: MODEL_COLORS[model] || '#71717a',
            'stroke-width': 2,
            'stroke-linecap': 'round',
            'stroke-linejoin': 'round',
        }));
    }

    // Axis labels
    svg.appendChild(svgEl('text', {
        x: W / 2, y: H - 4,
        'font-size': 12,
        'text-anchor': 'middle',
        fill: '#a1a1aa',
    }, 'False positive rate'));

    svg.appendChild(svgEl('text', {
        x: 14, y: H / 2,
        'font-size': 12,
        'text-anchor': 'middle',
        fill: '#a1a1aa',
        transform: `rotate(-90, 14, ${H / 2})`,
    }, 'True positive rate'));
}

function renderROCLegend(aucs) {
    const legend = $('#roc-legend');
    const items = Object.keys(aucs).map((model) =>
        el('span', { className: 'roc-legend-item' }, [
            el('span', {
                className: 'roc-legend-line',
                style: { backgroundColor: MODEL_COLORS[model] || '#71717a' },
            }),
            model,
        ])
    );
    items.push(el('span', { className: 'roc-legend-item' }, [
        el('span', { className: 'roc-legend-line dashed' }),
        'Random',
    ]));
    setChildren(legend, items);
}

// ============ Significance tests ============

function renderSignificance(significance) {
    const body = $('#sig-body');
    setChildren(body, Object.entries(significance).map(([pair, result]) =>
        el('div', { className: 'sig-row' }, [
            el('span', { className: 'sig-label' }, pair),
            el('div', { className: 'sig-result' }, [
                el('span', { className: 'sig-pvalue' }, `p = ${result.p_value.toFixed(3)}`),
                el('span', {
                    className: `sig-tag ${result.significant ? 'significant' : 'not-significant'}`,
                }, result.significant ? 'significant' : 'n.s.'),
            ]),
        ])
    ));
}
