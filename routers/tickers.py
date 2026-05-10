"""Routes for /tickers/validate and /tickers/search.

See docs/api.yaml for the contract and docs/conventions.md §8 for the
behavior rules these endpoints implement.
"""
from fastapi import APIRouter, HTTPException, Query

from schemas.tickers import (
    TickerSearchResponse,
    TickerValidationResponse,
)
from services.ticker_service import (
    is_valid_ticker_format,
    normalize_ticker,
    search_tickers,
    validate_ticker,
)

router = APIRouter(prefix="/tickers", tags=["tickers"])


@router.get("/validate", response_model=TickerValidationResponse)
def validate_endpoint(
    ticker: str = Query(..., min_length=1, max_length=15, description="Ticker symbol; case- and separator-insensitive."),
) -> TickerValidationResponse:
    """Validate a ticker. Invalid format -> 422. Valid format but unknown
    ticker -> 200 with `valid: false` (per spec — keeps client debouncing simple).
    """
    normalized = normalize_ticker(ticker)
    if not is_valid_ticker_format(normalized):
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "code": "INVALID_TICKER_FORMAT",
                    "message": (
                        "Ticker must be 1-10 characters, start with a letter, "
                        "and contain only letters, digits, '.' or '-'."
                    ),
                }
            },
        )
    return validate_ticker(normalized)


@router.get("/search", response_model=TickerSearchResponse)
def search_endpoint(
    q: str = Query("", max_length=50, description="Query string; matched against ticker prefix, then name."),
    limit: int = Query(10, ge=1, le=25),
) -> TickerSearchResponse:
    """Autocomplete tickers. Empty `q` returns empty `items`."""
    return search_tickers(q, limit)
