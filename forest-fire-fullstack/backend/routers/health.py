"""
Health Check Router
=====================

GET /api/health
    Returns the API status and what models are loaded. Useful for
    container health probes, deployment monitoring, and the frontend's
    initial connection check on page load.
"""

from fastapi import APIRouter, Request


router = APIRouter()


@router.get('/health')
async def health(request: Request):
    """Return basic status info about the running API."""
    registry = getattr(request.app.state, 'registry', None)
    if registry is None:
        return {'status': 'starting', 'models_loaded': []}

    return {
        'status': 'ok',
        'models_loaded': list(registry.models.keys()),
        'explainers_loaded': list(registry.explainers.keys()),
        'pipeline_loaded': registry.pipeline is not None,
        'project_root': str(registry.project_root) if registry.project_root else None,
    }
