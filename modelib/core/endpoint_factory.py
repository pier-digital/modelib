import typing

import fastapi


from modelib.runners.base import BaseRunner


def create_runner_endpoint(
    app: fastapi.FastAPI,
    runner: BaseRunner,
    **kwargs,
) -> fastapi.FastAPI:
    path = f"/{runner.slug}"

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


def create_runners_router(
    runners: typing.List[BaseRunner], **runners_router_kwargs
) -> fastapi.APIRouter:
    runners_router_kwargs["tags"] = runners_router_kwargs.get("tags", ["runners"])

    router = fastapi.APIRouter(**runners_router_kwargs)

    for runner in runners:
        router = create_runner_endpoint(
            router,
            runner=runner,
        )

    return router
