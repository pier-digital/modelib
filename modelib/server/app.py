import fastapi
import typing

from . import infrastructure
from modelib.core import exceptions

from modelib.runners.base import BaseRunner
from modelib.core import endpoint_factory


def init_app(
    app: fastapi.FastAPI,
    runners: typing.List[BaseRunner],
    include_infrastructure: bool = True,
) -> fastapi.FastAPI:
    exceptions.init_app(app)

    app.include_router(endpoint_factory.create_runners_router(runners))

    if include_infrastructure:
        infrastructure.init_app(app)

    return app
