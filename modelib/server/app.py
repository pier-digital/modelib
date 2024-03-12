import fastapi
import typing

from . import infrastructure
from modelib.core import exceptions

from modelib.runners.base import BaseRunner
from modelib.core import endpoint_factory


def init_app(
    *,
    runners: typing.List[BaseRunner],
    app: fastapi.FastAPI = fastapi.FastAPI(),
    include_infrastructure: bool = True,
    **runners_router_kwargs,
) -> fastapi.FastAPI:
    """Initialize FastAPI application with Modelib runners.

    Args:
        runners: List of runners to be included in the application.
        app: FastAPI application to be initialized. If not provided, a new application will be created.
        include_infrastructure: Whether to include infrastructure endpoints.
        **runners_router_kwargs: Additional keyword arguments to be passed to the runners router.

    Returns:
        FastAPI application with Modelib runners.
    """
    exceptions.init_app(app)

    app.include_router(
        endpoint_factory.create_runners_router(runners, **runners_router_kwargs)
    )

    if include_infrastructure:
        infrastructure.init_app(app)

    return app
