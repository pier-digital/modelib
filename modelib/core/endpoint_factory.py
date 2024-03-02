import typing

import fastapi


from modelib.core import schemas
from modelib.runners.base import EndpointMetadataManager, BaseRunner


def create_runner_endpoint(
    app: fastapi.FastAPI,
    runner_func: typing.Callable,
    endpoint_metadata_manager: EndpointMetadataManager,
    **kwargs,
) -> fastapi.FastAPI:
    path = f"/{endpoint_metadata_manager.slug}"

    route_kwargs = {
        "name": endpoint_metadata_manager.name,
        "methods": ["POST"],
        "response_model": endpoint_metadata_manager.response_model,
    }
    route_kwargs.update(kwargs)

    app.add_api_route(
        path,
        runner_func,
        **route_kwargs,
    )

    return app


def create_runners_router(
    runners: typing.List[BaseRunner], **runners_router_kwargs
) -> fastapi.APIRouter:
    responses = runners_router_kwargs.pop("responses", {})
    if 500 not in responses:
        responses[500] = {
            "model": schemas.JsonApiErrorModel,
            "description": "Inference Internal Server Error",
        }

    runners_router_kwargs["responses"] = responses

    runners_router_kwargs["tags"] = runners_router_kwargs.get("tags", ["runners"])

    router = fastapi.APIRouter(**runners_router_kwargs)

    for runner in runners:
        router = create_runner_endpoint(
            router,
            runner_func=runner.get_runner_func(),
            endpoint_metadata_manager=runner.endpoint_metadata_manager,
        )

    return router
