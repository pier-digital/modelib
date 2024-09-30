from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


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


async def json_api_exception_handler(request: Request, exc: JsonAPIException):
    response = exc.to_json_response()

    return response


def init_app(app: FastAPI):
    app.exception_handler(JsonAPIException)(json_api_exception_handler)
