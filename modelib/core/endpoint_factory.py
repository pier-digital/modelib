import typing

import fastapi
from slugify import slugify

from modelib.core import schemas
from modelib.runners.base import BaseRunner


def create_runner_endpoint(
    app: fastapi.FastAPI,
    runner: BaseRunner,
    **kwargs,
) -> fastapi.FastAPI:
    path = f"/{slugify(runner.name)}"

    route_kwargs = {
        "name": runner.name,
        "methods": ["POST"],
        "response_model": runner.response_model,
    }
    route_kwargs.update(kwargs)

    app.add_api_route(
        path,
        runner.get_runner_func(),
        **route_kwargs,
    )

    return app


def create_runners_router(runners: typing.List[BaseRunner]) -> fastapi.APIRouter:
    router = fastapi.APIRouter(
        tags=["runners"],
        responses={
            500: {
                "model": schemas.JsonApiErrorModel,
                "description": "Inference Internal Server Error",
            }
        },
    )

    for runner in runners:
        router = create_runner_endpoint(router, runner)

    return router
