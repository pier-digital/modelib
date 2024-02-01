import traceback

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class JsonAPIException(Exception):
    def __init__(
        self,
        status_code,
        title,
        detail=None,
        headers=None,
        type_="JsonAPIException",
        kind="json-api-exception",
    ):
        self.status_code = status_code
        self.title = title
        self.detail = detail
        self.headers = headers or {}
        if not isinstance(detail, dict):
            self.detail = {"message": detail}

        self.detail["type"] = type_
        self.detail["kind"] = kind

        super().__init__(self.detail or self.title)

    def to_dict(self):
        return {
            "status": self.status_code,
            "title": self.title,
            "detail": self.detail,
        }

    def content(self):
        return {"errors": [self.to_dict()]}

    def to_json_response(self):
        return JSONResponse(
            status_code=self.status_code,
            content=self.content(),
            headers=self.headers,
        )


async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    response = JSONResponse(
        status_code=exc.status_code,
        content={
            "errors": [
                {
                    "status": exc.status_code,
                    "title": "HTTP Exception",
                    "detail": {
                        "message": exc.detail,
                        "type": "HTTPException",
                        "kind": "http-exception",
                    },
                }
            ]
        },
    )

    return response


async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    response = JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"errors": exc.errors()},
    )

    return response


async def json_api_exception_handler(request: Request, exc: JsonAPIException):
    response = exc.to_json_response()

    return response


async def generic_exception_handler(request: Request, exc: Exception):
    try:
        traceback_info = traceback.format_exception(exc)
    except Exception:
        traceback_info = "Unknown"

    response = JSONResponse(
        status_code=500,
        content={
            "errors": [
                {
                    "status": 500,
                    "title": "Internal Server Error",
                    "detail": {
                        "message": str(exc),
                        "type": type(exc).__name__,
                        "kind": "unknown-generic-exception",
                        "traceback": traceback_info,
                    },
                }
            ]
        },
    )

    return response


def init_app(app: FastAPI):
    app.exception_handler(StarletteHTTPException)(custom_http_exception_handler)
    app.exception_handler(RequestValidationError)(request_validation_exception_handler)
    app.exception_handler(JsonAPIException)(json_api_exception_handler)
    app.exception_handler(Exception)(generic_exception_handler)
