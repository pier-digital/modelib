import typing

import fastapi
from slugify import slugify

from modelib.core import schemas
from modelib.runners.base import PayloadManager, BaseRunner


def create_runner_endpoint(
    app: fastapi.FastAPI,
    runner_func: typing.Callable,
    payload_manager: PayloadManager,
    **kwargs,
) -> fastapi.FastAPI:
    path = f"/{slugify(payload_manager.name)}"

    route_kwargs = {
        "name": payload_manager.name,
        "methods": ["POST"],
        "response_model": payload_manager.response_model,
    }
    route_kwargs.update(kwargs)

    app.add_api_route(
        path,
        runner_func,
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
        router = create_runner_endpoint(
            router,
            runner_func=runner.get_runner_func(),
            payload_manager=runner.payload_manager,
        )

    return router
