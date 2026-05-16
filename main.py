"""If-Vest API entry point.

Minimal FastAPI app. Only /health is implemented.
See docs/api.yaml for the full contract.
"""
from typing import Optional

from dotenv import load_dotenv  # Load .env BEFORE any module that reads env vars
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from routers.calculate import router as calculate_router
from routers.optimal_timing import router as optimal_timing_router
from routers.parse_decision import router as parse_decision_router
from routers.reflect import router as reflect_router
from routers.tickers import router as tickers_router


APP_VERSION = "0.2.0"

app = FastAPI(
    title="If-Vest API",
    version=APP_VERSION,
    description="Backend API for If-Vest. See docs/api.yaml for the spec.",
)

# CORS — see docs/conventions.md §11
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://if-vest.vercel.app",
    ],
    # Vercel preview deploys: https://if-vest-git-<branch>-<scope>.vercel.app
    # The scope segment is verified once we have a real preview URL; update if wrong.
    allow_origin_regex=r"https://if-vest-.+-eunjeonghur\.vercel\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Admin-Token"],
)

# --- Error envelope normalization ---
# Per docs/conventions.md §5, all error responses must have shape:
#   {"error": {"code": "...", "message": "...", "details": [...]?}}
# FastAPI's defaults wrap errors as {"detail": ...}, so we override.

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Unwrap detail when callers already passed `{"error": {...}}`. Otherwise
    wrap a string detail into the envelope.
    """
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "HTTP_ERROR",
                "message": str(exc.detail) if exc.detail else "Request failed.",
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Pydantic / Query validation failures -> envelope with per-field details."""
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed.",
                "details": [
                    {
                        "field": ".".join(str(p) for p in err.get("loc", [])),
                        "message": err.get("msg", "Invalid value."),
                    }
                    for err in exc.errors()
                ],
            }
        },
    )


app.include_router(calculate_router)
app.include_router(optimal_timing_router)
app.include_router(parse_decision_router)
app.include_router(reflect_router)
app.include_router(tickers_router)

class HealthResponse(BaseModel):
    status: str
    version: str
    db: Optional[str] = None
    yfinance: Optional[str] = None


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness probe. Always returns 200 unless the process can't respond.
    Subsystem fields (db, yfinance) report 'ok' / 'degraded' once those are
    wired in; for now they're set to 'ok' as placeholders.
    """
    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        db="ok",
        yfinance="ok",
    )
