"""
Compare Router
================

GET /api/compare/metrics
    Returns the cross-validation results, AUC scores, t-test outcomes,
    and feature importance rankings produced by the existing project's
    train_all_models.py script.

GET /api/compare/figure/{name}
    Streams one of the pre-rendered evaluation figures (PNG) from the
    project's figures/ directory. Lets the frontend embed them without
    knowing the file paths.

The metrics live in trained_models/ as JSON files written by the
training script. We just read and reformat them here -- no statistical
work happens at request time.
"""

import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse


router = APIRouter()


def _project_root(request: Request) -> Path:
    """Pull the project root off the registry that main.py loaded."""
    registry = getattr(request.app.state, 'registry', None)
    if registry is None or registry.project_root is None:
        raise HTTPException(503, 'Project root not resolved yet')
    return registry.project_root


@router.get('/metrics')
async def get_metrics(request: Request):
    """
    Read every JSON metrics file in trained_models/ and return them as
    one combined object. The file names are conventional:
        comparison_results.json   -- per-model AUC, PR, etc.
        cv_results.json           -- cross-validation fold scores
        significance_tests.json   -- pairwise t-test outcomes
        feature_importance.json   -- importance rankings

    Missing files are tolerated -- the response just lacks that section.
    """
    root = _project_root(request)
    results_dir = root / 'results'

    out = {'available': True, 'sections': {}}
    files = {
        'comparison':       'comparison_results.json',
        'cross_validation': 'cv_results.json',
        'significance':     'significance_tests.json',
        'feature_importance': 'feature_importance.json',
    }

    for key, fname in files.items():
        path = results_dir / fname
        if path.exists():
            try:
                out['sections'][key] = json.loads(path.read_text())
            except json.JSONDecodeError as e:
                out['sections'][key] = {'error': f'Could not parse: {e}'}

    if not out['sections']:
        # Fallback so the frontend still has something useful to show.
        # These numbers are placeholders; replace with real values from
        # your training run.
        out['available'] = False
        out['sections'] = {
            'comparison': {
                'Random Forest':  {'auc': 0.831, 'auc_std': 0.034, 'brier': 0.18},
                'XGBoost':        {'auc': 0.847, 'auc_std': 0.021, 'brier': 0.16},
                'Neural Network': {'auc': 0.812, 'auc_std': 0.048, 'brier': 0.14},
            },
        }
    return out


@router.get('/figure/{name}')
async def get_figure(name: str, request: Request):
    """
    Serve a PNG figure from the project's figures/ directory.

    Whitelist-only: only files matching a known set of names are served.
    Without this, a malicious client could read arbitrary files via
    path traversal (../../etc/passwd).
    """
    ALLOWED = {
        'roc_comparison', 'pr_curves', 'calibration_curves',
        'feature_importance', 'confusion_matrices', 'cv_distributions',
        'class_distribution', 'correlation_heatmap', 'feature_distributions',
    }
    if name not in ALLOWED:
        raise HTTPException(404, f'Unknown figure: {name}')

    root = _project_root(request)
    path = root / 'figures' / f'{name}.png'
    if not path.exists():
        raise HTTPException(404, f'Figure file not found: {path.name}')

    return FileResponse(path, media_type='image/png')


@router.get('/figures')
async def list_figures(request: Request):
    """List the figures that actually exist on disk, with display titles."""
    root = _project_root(request)
    figures_dir = root / 'figures'
    if not figures_dir.exists():
        return {'figures': []}

    titles = {
        'roc_comparison':       'ROC curve comparison',
        'pr_curves':            'Precision-recall curves',
        'calibration_curves':   'Probability calibration',
        'feature_importance':   'Feature importance',
        'confusion_matrices':   'Confusion matrices',
        'cv_distributions':     'Cross-validation AUC distribution',
        'class_distribution':   'Class distribution',
        'correlation_heatmap':  'Feature correlations',
        'feature_distributions': 'Feature distributions by class',
    }

    out = []
    for name, title in titles.items():
        if (figures_dir / f'{name}.png').exists():
            out.append({'name': name, 'title': title})
    return {'figures': out}
