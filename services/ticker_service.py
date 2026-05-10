"""Ticker validation and search.

Source-of-truth lookup order (per docs/conventions.md §8):
  1. Static index loaded from data/tickers_index.json (NASDAQ + NYSE listings)
  2. yfinance fallback for tickers not in the static index (handles new IPOs)

Search additionally augments results with yfinance.Search when the static
index returns fewer than 3 results. yfinance failures degrade silently —
search must never 502 (it's autocomplete UX).

In-memory TTL caches:
  - validate: 24h per ticker
  - search:    1h per (q, limit) pair
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import yfinance as yf
from cachetools import TTLCache

from schemas.tickers import (
    TickerSearchItem,
    TickerSearchResponse,
    TickerValidationResponse,
)

logger = logging.getLogger(__name__)

# Pattern from docs/api.yaml `Ticker` schema (post-normalization).
TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")

INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "tickers_index.json"

# In-memory caches. NOTE: not thread-safe; fine for single-worker uvicorn.
_validate_cache: TTLCache = TTLCache(maxsize=10_000, ttl=86_400)  # 24h
_search_cache: TTLCache = TTLCache(maxsize=1_000, ttl=3_600)  # 1h

# Loaded once at import time.
_INDEX: list[dict] = []
_INDEX_BY_TICKER: dict[str, dict] = {}


def _load_index() -> None:
    global _INDEX, _INDEX_BY_TICKER
    if not INDEX_PATH.exists():
        raise RuntimeError(
            f"Ticker index missing at {INDEX_PATH}. "
            "Run `python scripts/build_tickers_index.py` to generate it."
        )
    _INDEX = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    _INDEX_BY_TICKER = {entry["ticker"]: entry for entry in _INDEX}
    logger.info("Loaded %d tickers from static index", len(_INDEX))


_load_index()


def normalize_ticker(raw: str) -> str:
    """Match scripts/build_tickers_index.py: whitespace/slash/dot -> dash."""
    t = raw.strip().upper()
    t = re.sub(r"[\s/.]+", "-", t)
    return t.strip("-")


def is_valid_ticker_format(ticker: str) -> bool:
    return bool(TICKER_PATTERN.fullmatch(ticker))


# ============================================================================
# /tickers/validate
# ============================================================================
def validate_ticker(raw: str) -> TickerValidationResponse:
    """Validate a ticker. Always returns a 200-shaped result; the router layer
    decides on 422 for malformed input (caller's responsibility to call
    `is_valid_ticker_format` after `normalize_ticker`).
    """
    ticker = normalize_ticker(raw)

    cached = _validate_cache.get(ticker)
    if cached is not None:
        return cached

    # 1. Static index (fast)
    entry = _INDEX_BY_TICKER.get(ticker)
    if entry is not None:
        result = TickerValidationResponse(
            ticker=ticker,
            valid=True,
            name=entry["name"],
            exchange=entry.get("exchange"),
        )
        _validate_cache[ticker] = result
        return result

    # 2. yfinance fallback (slower; covers new IPOs not yet in static index)
    result = _yfinance_validate(ticker)
    _validate_cache[ticker] = result
    return result


def _yfinance_validate(ticker: str) -> TickerValidationResponse:
    """Try yfinance. Returns valid=False on any failure or empty result."""
    try:
        # history() is faster than .info; if data comes back, the ticker exists.
        hist = yf.Ticker(ticker).history(period="5d")
        if hist.empty:
            return TickerValidationResponse(ticker=ticker, valid=False)

        # Try to enrich with metadata. Both calls can fail; both are optional.
        name: Optional[str] = None
        exchange: Optional[str] = None
        try:
            fast = yf.Ticker(ticker).fast_info
            exchange = getattr(fast, "exchange", None)
        except Exception:
            pass
        try:
            info = yf.Ticker(ticker).info or {}
            name = info.get("longName") or info.get("shortName")
            if exchange is None:
                exchange = info.get("exchange") or info.get("fullExchangeName")
        except Exception:
            pass

        return TickerValidationResponse(
            ticker=ticker, valid=True, name=name, exchange=exchange
        )
    except Exception as exc:
        logger.warning("yfinance validate failed for %s: %s", ticker, exc)
        return TickerValidationResponse(ticker=ticker, valid=False)


# ============================================================================
# /tickers/search
# ============================================================================
def search_tickers(q: str, limit: int) -> TickerSearchResponse:
    """Autocomplete tickers by symbol/name. Empty `q` -> empty list (200).
    EQUITY + ETF only. yfinance failures degrade silently; never 502.
    """
    q_clean = q.strip()
    if not q_clean:
        return TickerSearchResponse(items=[])

    q_upper = q_clean.upper()
    cache_key = (q_upper, limit)
    cached = _search_cache.get(cache_key)
    if cached is not None:
        return cached

    # 1. Static index search with 3-tier ranking.
    ticker_prefix: list[dict] = []
    name_prefix: list[dict] = []
    name_substring: list[dict] = []

    for entry in _INDEX:
        ticker = entry["ticker"]
        name_upper = entry["name"].upper()
        if ticker.startswith(q_upper):
            ticker_prefix.append(entry)
        elif name_upper.startswith(q_upper):
            name_prefix.append(entry)
        elif q_upper in name_upper:
            name_substring.append(entry)

    for tier in (ticker_prefix, name_prefix, name_substring):
        tier.sort(key=lambda e: e["ticker"])

    results: list[dict] = ticker_prefix + name_prefix + name_substring

    # 2. Augment from yfinance only if static is sparse.
    if len(results) < 3:
        try:
            yf_results = _yfinance_search(q_clean, limit)
        except Exception as exc:
            logger.warning("yfinance search failed for %r: %s", q_clean, exc)
            yf_results = []

        seen = {e["ticker"] for e in results}
        for entry in yf_results:
            if entry["ticker"] not in seen:
                results.append(entry)
                seen.add(entry["ticker"])

    items = [
        TickerSearchItem(
            ticker=e["ticker"],
            name=e["name"],
            exchange=e.get("exchange"),
        )
        for e in results[:limit]
    ]
    response = TickerSearchResponse(items=items)
    _search_cache[cache_key] = response
    return response


def _yfinance_search(q: str, limit: int) -> list[dict]:
    """Try yfinance.Search (available in yfinance >= 0.2.20). Return [] on any
    failure so the caller falls back to static-only results.
    """
    try:
        from yfinance import Search
    except ImportError:
        return []

    search = Search(q, max_results=limit * 2)  # over-fetch to allow filtering
    quotes = getattr(search, "quotes", None) or []

    out: list[dict] = []
    for quote in quotes:
        quote_type = (quote.get("quoteType") or "").upper()
        if quote_type not in ("EQUITY", "ETF"):
            continue
        symbol = (quote.get("symbol") or "").upper()
        if not symbol or not is_valid_ticker_format(symbol):
            continue
        name = quote.get("shortname") or quote.get("longname") or ""
        if not name:
            continue
        out.append(
            {
                "ticker": symbol,
                "name": name,
                "exchange": quote.get("exchange"),
                "type": quote_type,
            }
        )
    return out[:limit]
