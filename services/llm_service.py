"""LLM-backed extraction service (F-05).

Uses Claude Haiku 4.5 with tool_use for structured output. The system prompt
and tool schema are static and tagged with cache_control so Anthropic's prompt
cache can reuse them across requests (cache hits land at ~10% of base cost).

NOTE: Haiku 4.5 requires ~4096-token cacheable prefix to actually trigger
caching. Our current prompt + tool schema is shorter, so caching is a no-op
right now — kept for forward compatibility as the prompt grows.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import date
from typing import Any, Optional

import anthropic
from fastapi import HTTPException

from schemas.parse_decision import (
    ConfidenceScores,
    ExtractedFields,
    ParseDecisionResponse,
)

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1024

# Per docs/api.yaml Ticker pattern (post-normalization).
_TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


# --------------------------------------------------------------------------
# Static, cacheable system prompt
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a financial decision extractor for If-Vest, a personal-investor reflection tool.

Your job: read a user's free-text description of a trading decision (or non-decision) and call the `extract_decision` tool with structured fields. Always call the tool — never respond in plain text.

## Fields

### ticker
- US stock ticker symbol (UPPERCASE).
- Map company names from your knowledge:
  - "Nvidia" / "엔비디아" → NVDA
  - "Tesla" / "테슬라" → TSLA
  - "Apple" / "애플" → AAPL
  - "Microsoft" / "마이크로소프트" → MSFT
  - "Google" / "Alphabet" / "구글" → GOOGL
  - "Meta" / "메타" / "Facebook" → META
  - "Amazon" / "아마존" → AMZN
  - "Netflix" / "넷플릭스" → NFLX
  - "Berkshire Hathaway" / "버크셔" → BRK-B (use BRK-A only if class A explicit)
- Format: 1-10 chars, starts with letter, may include "." or "-" (e.g., BRK-A, BF-B).
- For non-US tickers (e.g., Samsung, TSMC), prefer the US ADR (TSM for TSMC). If no US listing, set ticker to null and flag in reasoning.

### scenario_type
Pick exactly one:
- `no_buy` — User considered buying but didn't. Phrases: "thought about buying", "almost bought", "should have bought", "사려다", "안 샀어".
- `no_sell` — User held a position; considered selling but didn't. Phrases: "should have sold", "kept holding", "didn't take profits", "안 팔았어", "버텼어".
- `sold_too_early` — User sold and price kept rising. Phrases: "sold too early", "took profits and missed", "팔았는데 더 올랐어".

### decision_date
ISO 8601 (YYYY-MM-DD). Resolve relative phrases against the "Today's date" line that begins every user message.
- "yesterday" / "어제" → today − 1 day
- "last week" / "지난주" → today − 7 days
- "X days ago" / "X일 전" → today − X days
- "last month" / "지난달" → today − 30 days
- "early/mid/late <Month>" / "초/중순/말" → 5th / 15th / 25th of that month
- "<Month>" alone → 15th of that month
- "Q1 2025" / "2025년 1분기" → 2025-02-15
- Specific dates → exact YYYY-MM-DD

If a relative phrase resolves to a future date, choose the most recent matching past date (e.g., "March" in May 2025 → 2025-03-15; "March" in February 2025 → 2024-03-15).

### quantity
- Number of shares mentioned ("10 shares", "10주"). Fractional allowed.
- Null if the user only mentioned dollar amount or no size at all.

### amount
- Total USD amount mentioned. Strip "$" and commas. "$10K" → 10000. "$2.5M" → 2500000.
- If user gives BOTH share count AND per-share price (e.g., "10 shares at $250"), set quantity=10 AND amount = 10 * 250 = 2500. Both fields populated represents the actual transaction value.
- Null if the user only mentioned share count or no size at all.

### confidence
Per-field self-assessment in [0, 1]:
- 1.0 — explicit and unambiguous in input
- 0.7-0.9 — strong inference (e.g., "Nvidia" → NVDA, "late March" → March 25)
- 0.4-0.6 — weak inference (ambiguous date, unclear scenario)
- 0.0 — field not mentioned (value is null)

### reasoning
1-2 sentences explaining how each non-null field was derived. Mention any ambiguity.

## Rules

- Always call `extract_decision`. No plain-text response.
- `decision_date` must be ≤ today.
- Set fields to null when not in the input — do NOT guess.
- If you cannot extract ticker (e.g., user mentioned only a sector), set ticker=null and confidence.ticker=0.

## Examples

Input: "Late March, I was thinking about buying 10 shares of Nvidia but didn't pull the trigger."
Today: 2026-05-10
extract_decision:
  ticker: "NVDA"
  scenario_type: "no_buy"
  decision_date: "2026-03-25"
  quantity: 10
  amount: null
  confidence: { ticker: 1.0, scenario_type: 1.0, decision_date: 0.8, quantity: 1.0, amount: 0.0 }
  reasoning: "User considered buying Nvidia (NVDA) but did not — no_buy. 'Late March' resolves to March 25, current year (most recent past March)."

Input: "지난주 테슬라 100주 $250에 팔았는데, 지금 보니 너무 일찍 팔았네."
Today: 2026-05-10
extract_decision:
  ticker: "TSLA"
  scenario_type: "sold_too_early"
  decision_date: "2026-05-03"
  quantity: 100
  amount: 25000
  confidence: { ticker: 1.0, scenario_type: 1.0, decision_date: 0.8, quantity: 1.0, amount: 0.95 }
  reasoning: "User sold Tesla (TSLA) and price rose afterward — sold_too_early. 'Last week' = today - 7. $250 is per-share, multiplied by 100 shares = $25000 total amount."

Input: "Held onto Apple through the September dip — should've sold at $220."
Today: 2026-05-10
extract_decision:
  ticker: "AAPL"
  scenario_type: "no_sell"
  decision_date: "2025-09-15"
  quantity: null
  amount: null
  confidence: { ticker: 1.0, scenario_type: 1.0, decision_date: 0.7, quantity: 0.0, amount: 0.0 }
  reasoning: "User held Apple (AAPL) and considered selling — no_sell. 'September' alone resolves to mid-month, last year (most recent past September)."

Input: "어제 마이크로소프트에 5천 달러 정도 넣을까 했어."
Today: 2026-05-10
extract_decision:
  ticker: "MSFT"
  scenario_type: "no_buy"
  decision_date: "2026-05-09"
  quantity: null
  amount: 5000
  confidence: { ticker: 1.0, scenario_type: 1.0, decision_date: 1.0, quantity: 0.0, amount: 0.95 }
  reasoning: "User considered buying Microsoft (MSFT) — no_buy. '어제' = today - 1 day. '5천 달러' = $5000."

Input: "tech stocks 좀 살까 했는데 안 샀어."
Today: 2026-05-10
extract_decision:
  ticker: null
  scenario_type: "no_buy"
  decision_date: null
  quantity: null
  amount: null
  confidence: { ticker: 0.0, scenario_type: 0.9, decision_date: 0.0, quantity: 0.0, amount: 0.0 }
  reasoning: "User considered buying — no_buy. 'tech stocks' is a sector, not a specific ticker; cannot extract. No date or size mentioned."
"""


# --------------------------------------------------------------------------
# Tool schema (Anthropic JSON Schema)
# --------------------------------------------------------------------------
EXTRACT_TOOL: dict[str, Any] = {
    "name": "extract_decision",
    "description": (
        "Extract structured trading-decision data from natural-language input. "
        "Always call this tool. Use null for fields not present in the input."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": ["string", "null"],
                "description": "Uppercase US ticker (e.g., NVDA, AAPL, BRK-A). Null if not extractable.",
            },
            "scenario_type": {
                "type": ["string", "null"],
                "enum": ["no_buy", "no_sell", "sold_too_early", None],
                "description": "Type of decision regret.",
            },
            "decision_date": {
                "type": ["string", "null"],
                "description": "ISO 8601 date (YYYY-MM-DD) the user was making the decision. Null if not extractable.",
            },
            "quantity": {
                "type": ["number", "null"],
                "description": "Shares (fractional allowed). Null if not mentioned.",
            },
            "amount": {
                "type": ["number", "null"],
                "description": "Total USD amount. Null if not mentioned. Set both this and quantity for actual transactions.",
            },
            "confidence": {
                "type": "object",
                "description": "Per-field self-rated confidence in [0, 1]. Use 0 for null fields.",
                "properties": {
                    "ticker": {"type": "number", "minimum": 0, "maximum": 1},
                    "scenario_type": {"type": "number", "minimum": 0, "maximum": 1},
                    "decision_date": {"type": "number", "minimum": 0, "maximum": 1},
                    "quantity": {"type": "number", "minimum": 0, "maximum": 1},
                    "amount": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "ticker",
                    "scenario_type",
                    "decision_date",
                    "quantity",
                    "amount",
                ],
            },
            "reasoning": {
                "type": "string",
                "description": "1-2 sentences explaining derivation of non-null fields.",
            },
        },
        "required": [
            "ticker",
            "scenario_type",
            "decision_date",
            "quantity",
            "amount",
            "confidence",
            "reasoning",
        ],
    },
}


def _get_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "SERVER_MISCONFIGURED",
                    "message": "ANTHROPIC_API_KEY is not set on the server.",
                }
            },
        )
    return anthropic.Anthropic(api_key=api_key)


def _validate_ticker_format(ticker: Optional[str]) -> bool:
    if ticker is None:
        return False
    return bool(_TICKER_PATTERN.fullmatch(ticker))


def parse_decision_text(text: str) -> ParseDecisionResponse:
    """Send `text` to Claude Haiku 4.5 and return structured extraction.

    Always returns a 200-shaped response. Raises HTTPException for upstream
    failures (auth, rate limit, unparsable model output).
    """
    client = _get_client()
    today_iso = date.today().isoformat()

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[EXTRACT_TOOL],
            tool_choice={"type": "tool", "name": "extract_decision"},
            messages=[
                {
                    "role": "user",
                    "content": f"Today's date: {today_iso}\n\nDecision text: {text}",
                }
            ],
        )
    except anthropic.RateLimitError:
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "code": "RATE_LIMITED",
                    "message": "LLM provider rate limit hit. Try again in a moment.",
                }
            },
        )
    except anthropic.AuthenticationError:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "INVALID_API_KEY",
                    "message": "Server LLM authentication failed.",
                }
            },
        )
    except anthropic.APIError as exc:
        logger.warning("Anthropic API error: %s", exc)
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "code": "LLM_PROVIDER_ERROR",
                    "message": str(exc),
                }
            },
        )

    # Extract the tool_use block
    tool_use_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None,
    )
    if tool_use_block is None:
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "code": "EXTRACTION_FAILED",
                    "message": "Model did not call the extraction tool.",
                }
            },
        )

    raw: dict[str, Any] = tool_use_block.input

    try:
        extracted = ExtractedFields(
            ticker=raw.get("ticker"),
            scenario_type=raw.get("scenario_type"),
            decision_date=raw.get("decision_date"),
            quantity=raw.get("quantity"),
            amount=raw.get("amount"),
        )
        confidence = ConfidenceScores(**(raw.get("confidence") or {}))
    except Exception as exc:
        logger.warning("Tool output failed schema validation: %s; raw=%r", exc, raw)
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "code": "EXTRACTION_FAILED",
                    "message": "Model output did not match expected schema.",
                }
            },
        )

    return ParseDecisionResponse(
        extracted=extracted,
        confidence=confidence,
        # TODO: replace with services.ticker_service.validate_ticker once that PR merges.
        ticker_validated=_validate_ticker_format(extracted.ticker),
        reasoning=raw.get("reasoning") or "",
    )
