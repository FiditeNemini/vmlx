"""JSON-safe request validation errors, including non-standard float input."""

import math

from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse


async def request_validation_error_response(request, exc):
    # Python's JSON decoder accepts NaN/Infinity. An invalid request can thus
    # carry non-finite floats in Pydantic's echoed `input`. Preserve ordinary
    # FastAPI error details while preventing its JSON response from raising 500.
    details = jsonable_encoder(
        exc.errors(),
        custom_encoder={float: lambda value: value if math.isfinite(value) else repr(value)},
    )
    return JSONResponse(status_code=422, content={"detail": details})
