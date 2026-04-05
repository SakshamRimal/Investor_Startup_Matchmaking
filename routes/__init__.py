from routes.entities import router as entities_router
from routes.predictions import router as predictions_router
from routes.dashboard import router as dashboard_router
from routes.frontend import router as frontend_router


__all__ = [
    "entities_router",
    "predictions_router",
    "dashboard_router",
    "frontend_router",
]
