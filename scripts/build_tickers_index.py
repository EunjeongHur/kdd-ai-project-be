"""Build the static ticker index used by /tickers/validate and /tickers/search.

Pulls NASDAQ Trader's official symbol directories (free, public), filters out
test issues, warrants, units, etc., and writes data/tickers_index.json.

Run this once and commit the JSON. Refresh manually every month or so.

Usage:
    .\\.venv\\Scripts\\python.exe scripts/build_tickers_index.py
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "tickers_index.json"

# otherlisted.txt 'Exchange' code -> human-readable
EXCHANGE_MAP = {
    "A": "NYSE American",
    "N": "NYSE",
    "P": "NYSE Arca",
    "Z": "BATS",
    "V": "IEX",
}

# Security Name keywords that mean "not a normal stock/ETF"
EXCLUDE_KEYWORDS = [
    "warrant",
    "warrants",
    "right",
    "rights",
    " unit",
    " units",
    "preferred",
    "convertible",
    "depositary",
    "subordinated",
    "notes",
    "bonds",
    "bond ",
    "trust unit",
    "limited partnership",
    "when issued",
    "when-issued",
]

# Suffixes commonly tacked onto Security Name; trim for display
NAME_SUFFIXES = [
    " - Common Stock",
    " Common Stock",
    " - American Depositary Shares",
    " - Class A Common Stock",
    " - Class B Common Stock",
    " - Class C Common Stock",
    " - Ordinary Shares",
    " Ordinary Shares",
]


def fetch_text(url: str) -> str:
    print(f"Downloading {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "if-vest-build/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")


def parse_pipe_table(text: str) -> list[dict[str, str]]:
    """Parse NASDAQ Trader's pipe-separated format.
    First line = header. Last line = 'File Creation Time:' marker (skip it).
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    header = [h.strip() for h in lines[0].split("|")]
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        if line.startswith("File Creation Time"):
            break
        cells = line.split("|")
        if len(cells) != len(header):
            continue
        rows.append({h: c.strip() for h, c in zip(header, cells)})
    return rows


def clean_name(name: str) -> str:
    """Trim noisy suffixes from Security Name for nicer display."""
    cleaned = name.strip()
    # Repeatedly strip suffixes (some entries have multiple)
    for _ in range(3):
        for suffix in NAME_SUFFIXES:
            if cleaned.lower().endswith(suffix.lower()):
                cleaned = cleaned[: -len(suffix)].strip()
                break
        else:
            break
    return cleaned


def is_keepable(name: str, ticker: str) -> bool:
    """Skip warrants, rights, units, preferreds, etc."""
    if not ticker or not name:
        return False
    # Tickers with $ usually denote preferred / warrant / right share classes
    if "$" in ticker:
        return False
    name_lower = name.lower()
    return not any(kw in name_lower for kw in EXCLUDE_KEYWORDS)


def normalize_ticker(t: str) -> str:
    """Backend uses dash for share classes (BRK-A, BRK-B). NASDAQ files use
    space (`BRK A`), slash (`BRK/A`), or dot (`BRK.A`) inconsistently across
    columns. Collapse any of those separators to a single dash so we end up
    with the Yahoo-Finance-style symbol users actually type — and that yfinance
    accepts as input on lookup.
    """
    t = t.strip().upper()
    # Whitespace, slash, dot -> dash. Yahoo / yfinance expect dash for share classes.
    t = re.sub(r"[\s/.]+", "-", t)
    # Strip leading/trailing dashes if any survived
    return t.strip("-")


def parse_nasdaq(text: str) -> list[dict[str, str]]:
    rows = parse_pipe_table(text)
    out: list[dict[str, str]] = []
    for row in rows:
        if row.get("Test Issue", "Y") == "Y":
            continue
        ticker = normalize_ticker(row.get("Symbol", ""))
        name = clean_name(row.get("Security Name", ""))
        if not is_keepable(name, ticker):
            continue
        # Validate against the API's Ticker pattern
        if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", ticker):
            continue
        out.append({
            "ticker": ticker,
            "name": name,
            "exchange": "NASDAQ",
            "type": "ETF" if row.get("ETF", "N") == "Y" else "EQUITY",
        })
    return out


def parse_other(text: str) -> list[dict[str, str]]:
    rows = parse_pipe_table(text)
    out: list[dict[str, str]] = []
    for row in rows:
        if row.get("Test Issue", "Y") == "Y":
            continue
        ticker = normalize_ticker(row.get("ACT Symbol", ""))
        name = clean_name(row.get("Security Name", ""))
        if not is_keepable(name, ticker):
            continue
        if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", ticker):
            continue
        exchange = EXCHANGE_MAP.get(row.get("Exchange", ""), row.get("Exchange", "OTHER"))
        out.append({
            "ticker": ticker,
            "name": name,
            "exchange": exchange,
            "type": "ETF" if row.get("ETF", "N") == "Y" else "EQUITY",
        })
    return out


def dedupe_prefer_nasdaq(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: dict[str, dict[str, str]] = {}
    for e in entries:
        existing = seen.get(e["ticker"])
        if existing is None:
            seen[e["ticker"]] = e
        else:
            # NASDAQ entry takes priority over others
            if e["exchange"] == "NASDAQ" and existing["exchange"] != "NASDAQ":
                seen[e["ticker"]] = e
    return list(seen.values())


def main() -> None:
    nasdaq_text = fetch_text(NASDAQ_URL)
    other_text = fetch_text(OTHER_URL)

    nasdaq_entries = parse_nasdaq(nasdaq_text)
    other_entries = parse_other(other_text)
    print(f"  NASDAQ:    {len(nasdaq_entries)} entries (post-filter)")
    print(f"  NYSE/etc.: {len(other_entries)} entries (post-filter)")

    combined = dedupe_prefer_nasdaq(nasdaq_entries + other_entries)
    combined.sort(key=lambda e: e["ticker"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    equity_count = sum(1 for e in combined if e["type"] == "EQUITY")
    etf_count = sum(1 for e in combined if e["type"] == "ETF")

    print()
    print(f"Wrote {len(combined)} tickers -> {OUTPUT_PATH.relative_to(Path.cwd())}")
    print(f"  EQUITY: {equity_count}")
    print(f"  ETF:    {etf_count}")
    print()

    # Sanity check on common tickers
    by_ticker = {e["ticker"]: e for e in combined}
    for t in ["AAPL", "NVDA", "TSLA", "MSFT", "SPY", "QQQ", "BRK-A", "BRK-B"]:
        e = by_ticker.get(t)
        if e:
            print(f"  ✓ {t}: {e['name']} ({e['exchange']}, {e['type']})")
        else:
            print(f"  ✗ {t}: MISSING")


if __name__ == "__main__":
    main()
