import fastapi
from fastapi.responses import JSONResponse
import traceback


def parse_exception(exception: Exception) -> dict:
    data = {
        "message": str(exception),
        "type": type(exception).__name__,
        "traceback": None,
    }

    try:
        data["traceback"] = "".join(
            traceback.format_exception(None, exception, exception.__traceback__)
        ).strip()

    except Exception:
        pass

    return data


async def internal_exception_handler(request: fastapi.Request, exc: Exception):
    if isinstance(exc, fastapi.HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail,
        )

    return JSONResponse(
        status_code=500,
        content={
            "title": "Internal server error",
            "error": parse_exception(exc),
        },
    )


def init_app(app: fastapi.FastAPI):
    app.exception_handler(500)(internal_exception_handler)
