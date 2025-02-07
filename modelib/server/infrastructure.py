import fastapi

from modelib.core import schemas

router = fastapi.APIRouter(prefix="", tags=["Infrastructure"])


@router.get("/healthz", response_model=schemas.HealthCheckStausSchema)
async def healthz():
    """Health check endpoint. Expecting an empty response with status code 200 when the service is in health state. The /healthz endpoint is deprecated. (since Kubernetes v1.16)"""
    return schemas.HealthCheckStausSchema(status="ok")


@router.get("/readyz", response_model=schemas.HealthCheckStausSchema)
async def readyz():
    """A 200 OK status from /readyz endpoint indicated the service is ready to accept traffic. From that point and onward, Kubernetes will use /livez endpoint to perform periodic health checks."""
    return schemas.HealthCheckStausSchema(status="ok")


@router.get("/livez", response_model=schemas.HealthCheckStausSchema)
async def livez():
    """Health check endpoint for Kubernetes. Healthy endpoint responses with a 200 OK status."""
    return schemas.HealthCheckStausSchema(status="ok")


def init_app(app: fastapi.FastAPI):
    app.include_router(router)
