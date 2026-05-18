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
from schemas.reflect import ReflectRequest, ReflectResponse
from services import ticker_service

logger = logging.getLogger(__name__)

# Different models per task. Extraction is bounded structured output that
# Haiku does very well; reflection is creative writing where Sonnet's
# instruction following + tone calibration is meaningfully better.
EXTRACT_MODEL = "claude-haiku-4-5"
REFLECT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024

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

### title
- Short event-style label (3-7 words, Title Case) used as the reflection's headline in journal lists.
- Should evoke the SITUATION or EVENT around the decision, not restate enums.
  - GOOD: "NVDA Earnings Dip Skipped", "TSLA Profit-Take Regret", "AAPL September Hold"
  - BAD:  "No Buy of NVDA", "User did not sell" (just restating scenario_type)
- Pull concrete context the user mentioned: catalyst (earnings, dip, rally), timing (March, last week), or outcome framing (skipped, held, sold early).
- No emoji, no em-dash, no AI buzzwords (no "AI", "smart", "intelligent").
- Set to null when the input is too sparse for a meaningful headline (e.g., "bought NVDA"). Don't pad with generic words.

### confidence
Per-field self-assessment in [0, 1]:
- 1.0 — explicit and unambiguous in input
- 0.7-0.9 — strong inference (e.g., "Nvidia" → NVDA, "late March" → March 25)
- 0.4-0.6 — weak inference (ambiguous date, unclear scenario)
- 0.0 — field not mentioned (value is null)
- For `title`: 0.9+ when input has clear catalyst/timing/outcome; 0.7-0.8 when somewhat thin; 0.0 when null.

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
  title: "NVDA March Entry Skipped"
  confidence: { ticker: 1.0, scenario_type: 1.0, decision_date: 0.8, quantity: 1.0, amount: 0.0, title: 0.9 }
  reasoning: "User considered buying Nvidia (NVDA) but did not — no_buy. 'Late March' resolves to March 25, current year (most recent past March). Title captures ticker + timing + skipped entry."

Input: "지난주 테슬라 100주 $250에 팔았는데, 지금 보니 너무 일찍 팔았네."
Today: 2026-05-10
extract_decision:
  ticker: "TSLA"
  scenario_type: "sold_too_early"
  decision_date: "2026-05-03"
  quantity: 100
  amount: 25000
  title: "TSLA Early Exit Regret"
  confidence: { ticker: 1.0, scenario_type: 1.0, decision_date: 0.8, quantity: 1.0, amount: 0.95, title: 0.9 }
  reasoning: "User sold Tesla (TSLA) and price rose afterward — sold_too_early. 'Last week' = today - 7. $250 is per-share, multiplied by 100 shares = $25000 total amount. Title reflects early-exit framing."

Input: "Held onto Apple through the September dip — should've sold at $220."
Today: 2026-05-10
extract_decision:
  ticker: "AAPL"
  scenario_type: "no_sell"
  decision_date: "2025-09-15"
  quantity: null
  amount: null
  title: "AAPL September Dip Hold"
  confidence: { ticker: 1.0, scenario_type: 1.0, decision_date: 0.7, quantity: 0.0, amount: 0.0, title: 0.9 }
  reasoning: "User held Apple (AAPL) and considered selling — no_sell. 'September' alone resolves to mid-month, last year (most recent past September). Title captures the dip catalyst plus hold action."

Input: "어제 마이크로소프트에 5천 달러 정도 넣을까 했어."
Today: 2026-05-10
extract_decision:
  ticker: "MSFT"
  scenario_type: "no_buy"
  decision_date: "2026-05-09"
  quantity: null
  amount: 5000
  title: "MSFT $5K Entry Hesitation"
  confidence: { ticker: 1.0, scenario_type: 1.0, decision_date: 1.0, quantity: 0.0, amount: 0.95, title: 0.85 }
  reasoning: "User considered buying Microsoft (MSFT) — no_buy. '어제' = today - 1 day. '5천 달러' = $5000. Title captures hesitation around a specific dollar amount."

Input: "tech stocks 좀 살까 했는데 안 샀어."
Today: 2026-05-10
extract_decision:
  ticker: null
  scenario_type: "no_buy"
  decision_date: null
  quantity: null
  amount: null
  title: null
  confidence: { ticker: 0.0, scenario_type: 0.9, decision_date: 0.0, quantity: 0.0, amount: 0.0, title: 0.0 }
  reasoning: "User considered buying — no_buy. 'tech stocks' is a sector, not a specific ticker; cannot extract. No date or size mentioned. Title null because no concrete catalyst/ticker to anchor the headline."
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
            "title": {
                "type": ["string", "null"],
                "description": (
                    "Short event-style label (3-7 words, Title Case) used as the "
                    "reflection's headline. Pull catalyst/timing/outcome framing "
                    "from the input. Null if input is too sparse for a meaningful title."
                ),
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
                    "title": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "ticker",
                    "scenario_type",
                    "decision_date",
                    "quantity",
                    "amount",
                    "title",
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
            "title",
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


def _ticker_exists(ticker: Optional[str]) -> bool:
    """Real existence check via ticker_service: format -> static index -> yfinance."""
    if ticker is None:
        return False
    normalized = ticker_service.normalize_ticker(ticker)
    if not ticker_service.is_valid_ticker_format(normalized):
        return False
    try:
        return ticker_service.validate_ticker(normalized).valid
    except Exception as exc:
        logger.warning("Ticker validation failed for %r: %s", ticker, exc)
        return False


def parse_decision_text(text: str) -> ParseDecisionResponse:
    """Send `text` to Claude Haiku 4.5 and return structured extraction.

    Always returns a 200-shaped response. Raises HTTPException for upstream
    failures (auth, rate limit, unparsable model output).
    """
    client = _get_client()
    today_iso = date.today().isoformat()

    try:
        response = client.messages.create(
            model=EXTRACT_MODEL,
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
            title=raw.get("title"),
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
        ticker_validated=_ticker_exists(extracted.ticker),
        reasoning=raw.get("reasoning") or "",
    )


# ==========================================================================
# Per-decision reflection (single-decision narrative)
# ==========================================================================

REFLECT_MAX_ATTEMPTS = 3
REFLECT_MAX_TOKENS = 200

# Phrases that violate PRD AI guardrails (5.2). Targeted at *recommendations*
# and *predictions* — factual language like "you opted not to buy" is allowed,
# since the no_buy scenario is meaningless without saying "buy".
_FORBIDDEN_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # Prescriptive language (telling user what to do)
        r"\b(?:should|shouldn'?t|must|need\s+to|have\s+to|ought\s+to)\b",
        # Explicit recommendations
        r"\brecommend",
        r"\b(?:I'?d|I\s+would)\s+(?:suggest|recommend|advise)\b",
        r"\bmy\s+(?:advice|recommendation)\b",
        # Imperative buy/sell phrasing (vs factual past-tense)
        r"\b(?:time\s+to\s+(?:buy|sell)|opportunity\s+to\s+(?:buy|sell)|consider\s+(?:buying|selling))\b",
        # Future price predictions
        r"\bwill\s+(?:rise|fall|drop|jump|reach|hit|increase|decrease|gain|lose|continue)\b",
        r"\b(?:likely|expected|going)\s+to\s+(?:rise|fall|drop|jump|reach|hit|increase|decrease|gain|lose|move|continue)\b",
        # Prescriptive future references
        r"\bnext\s+time\b",
        r"\bgoing\s+forward\b",
        r"\bin\s+the\s+future\b",
        # Strategy suggestions
        r"\bdiversify\b",
        r"\bhold\s+longer\b",
        r"\b(?:rebalance|stop[- ]loss|take\s+profits)\b",
        # Internal enum / code-style values that should never appear in user copy.
        # We only ban the underscore_form — plain English words like
        # "outcome", "direction" must be allowed (they appear naturally).
        r"\b(?:no_buy|no_sell|sold_too_early)\b",
        r"\b(?:missed_gain|avoided_loss|kept_gain|endured_loss|cut_short_gain|well_timed_exit)\b",
        r"\b(?:scenario_type|diff_percent|diff_amount|decision_price)\b",
    ]
]


REFLECT_SYSTEM_PROMPT = """You are a decision reflection assistant for If-Vest, a personal-investor reflection tool. Given the result of a user's decision and (when available) their decision history, write a brief, neutral 1-2 sentence reflection that helps the user notice their own pattern — without telling them what to do.

## Input format

The user message contains:
1. A "Current decision" block (always present) with ticker, action, price change, and result described in plain language.
2. A "Previous decisions" list — either empty (first reflection) or up to 10 entries, most-recent-first, in the same block format.

## Output language — IMPORTANT

The input describes decisions in plain language (e.g. "considered buying but did not", "missed a potential gain"). **Your output must use the same plain language style.** Never use internal code-like terms in your reflection:

NEVER WRITE:
- "no_buy", "no_sell", "sold_too_early"
- "missed_gain", "avoided_loss", "cut_short_gain", "well_timed_exit", "endured_loss", "kept_gain"
- "scenario_type", "direction", "outcome"

INSTEAD WRITE (natural English):
- "you considered buying but didn't"
- "you missed a substantial gain"
- "you exited before a further rise"
- "you held through a decline"

## Behavior depends on history

- **No previous decisions**: Reflect on this single decision. Don't claim to see "patterns" or "recurring" anything from one data point — that's dishonest. Use phrasing like "a useful first data point", "as you log more decisions, patterns will emerge", or just describe what happened without speculating.

- **1-2 previous decisions**: Still too few for pattern claims. You may note "this is your second/third logged decision" or compare to one specific prior decision, but don't generalize.

- **3+ previous decisions**: Now you can find genuine patterns. Count specific recurrences ("your fourth missed_gain in a row"), notice shifts ("your first favorable outcome after three unfavorable"), or identify themes (sectors, scenario types). Anchor every claim in the actual data shown — never invent.

## What you write
- Sentence 1: A factual observation about THIS decision (what happened, scale).
- Sentence 2 (optional): A gentle, non-prescriptive observation that invites self-reflection — e.g. "worth noting if X is recurring for you."

## Tone
- Neutral and matter-of-fact. Like a mirror, not a coach.
- Same weight for good outcomes and bad outcomes — don't dramatize losses or downplay wins.
- Plain language. No jargon. No emojis. No exclamation marks.
- One short paragraph. 1-2 sentences. ≤ 60 words.

## STRICT prohibitions (NEVER do these)
- DON'T recommend buying or selling. No "should buy", "time to sell", "consider buying", "opportunity to sell".
- DON'T predict future prices. No "will rise", "will fall", "likely to drop", "going to recover".
- DON'T give prescriptive advice. No "should", "shouldn't", "must", "need to", "ought to", "next time", "going forward".
- DON'T suggest strategies. No "diversify", "hold longer", "rebalance", "take profits", "stop-loss".
- DON'T use "I recommend", "my advice", "I'd suggest".

## What IS allowed (and necessary)
- Factual past-tense description of what they did: "you opted not to buy", "you sold AMZN", "you held AAPL through" — these are required to describe the scenario at all.
- Restating the magnitude in plain language: "a 193% move", "modest", "substantial".
- Mentioning the scenario: "you considered buying", "you held", "you exited".
- Inviting reflection: "worth noting", "worth observing", "this kind of pattern shows up".
- Tracking framing: "tracking how often X coincides with Y", "useful data point alongside other decisions".

The line is: describe the past and invite reflection. Never tell them what to do or predict what comes next.

## Examples — no history (first reflection)

Input:
Current decision:
  - Ticker: NVDA
  - User action: considered buying but did not on 2024-03-15
  - Price change since: +193.32%
  - Result: missed a potential gain (unfavorable outcome)
Previous decisions: (none — this is the user's first reflection)

Output:
"You considered NVDA back then; the stock has nearly tripled since. As you log more decisions, this becomes the first data point in your own decision record."

Input:
Current decision:
  - Ticker: META
  - User action: considered buying but did not on 2025-12-01
  - Price change since: -5.20%
  - Result: avoided a potential loss (favorable outcome)
Previous decisions: (none — this is the user's first reflection)

Output:
"You opted not to buy META; the stock has since declined 5.2%. A useful first data point — the value of this record grows as you add more decisions."

## Examples — with history (genuine pattern observation)

Input:
Current decision:
  - Ticker: NVDA
  - User action: considered buying but did not on 2025-03-15
  - Price change since: +193.32%
  - Result: missed a potential gain (unfavorable outcome)

Previous decisions (3, most recent first):

Previous decision #1:
  - Ticker: TSLA
  - User action: considered buying but did not on 2024-12-10
  - Price change since: +45.00%
  - Result: missed a potential gain (unfavorable outcome)

Previous decision #2:
  - Ticker: AMD
  - User action: considered buying but did not on 2024-10-01
  - Price change since: +28.00%
  - Result: missed a potential gain (unfavorable outcome)

Previous decision #3:
  - Ticker: MSFT
  - User action: sold the position on 2024-08-15
  - Price change since: +12.00%
  - Result: sold before further upside (unfavorable outcome)

Output:
"This is your fourth unfavorable outcome in a row, and the third time you stepped back from a high-conviction name and watched it run. NVDA's 193% move stands out as the largest by a wide margin — hesitation around momentum names is consistent across your record so far."

Input:
Current decision:
  - Ticker: META
  - User action: considered buying but did not on 2025-12-01
  - Price change since: -5.20%
  - Result: avoided a potential loss (favorable outcome)

Previous decisions (5, most recent first):

Previous decision #1:
  - Ticker: NVDA
  - User action: considered buying but did not on 2025-10-01
  - Price change since: +15.00%
  - Result: missed a potential gain (unfavorable outcome)

Previous decision #2:
  - Ticker: AMD
  - User action: considered buying but did not on 2025-08-15
  - Price change since: +22.00%
  - Result: missed a potential gain (unfavorable outcome)

Previous decision #3:
  - Ticker: TSLA
  - User action: sold the position on 2025-06-10
  - Price change since: +8.00%
  - Result: sold before further upside (unfavorable outcome)

Previous decision #4:
  - Ticker: AAPL
  - User action: held a position and considered selling but did not on 2025-04-01
  - Price change since: -12.00%
  - Result: absorbed a decline by holding (unfavorable outcome)

Previous decision #5:
  - Ticker: GOOGL
  - User action: considered buying but did not on 2025-02-15
  - Price change since: +18.00%
  - Result: missed a potential gain (unfavorable outcome)

Output:
"Your first favorable outcome in your last six decisions — opting out of META aligned with a 5% pullback. A sharp contrast with your prior pattern of stepping back from names that subsequently rose."

Input:
Current decision:
  - Ticker: AAPL
  - User action: held a position and considered selling but did not on 2026-04-01
  - Price change since: -8.50%
  - Result: absorbed a decline by holding (unfavorable outcome)

Previous decisions (2, most recent first):

Previous decision #1:
  - Ticker: TSLA
  - User action: considered buying but did not on 2026-02-10
  - Price change since: +12.00%
  - Result: missed a potential gain (unfavorable outcome)

Previous decision #2:
  - Ticker: NVDA
  - User action: considered buying but did not on 2025-12-15
  - Price change since: +8.00%
  - Result: missed a potential gain (unfavorable outcome)

Output:
"You held AAPL through an 8.5% drawdown — your third logged decision, and the first time you've reflected on a hold-through-decline. Too few entries to call any pattern, but the record now spans different decision types."

## Output rules

- Respond with ONLY the reflection text. No preamble, no quotes, no markdown.
- Use the same natural language found in your examples above.
- Never use code-style enum values (no_buy, missed_gain, sold_too_early, etc.) — translate to plain English."""


def _has_forbidden(text: str) -> Optional[str]:
    """Return the first forbidden phrase found, or None if clean."""
    for pattern in _FORBIDDEN_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


# Natural-language translations of internal enum values. We feed these to the
# model instead of raw enum strings so it never echoes "no_buy" or
# "missed_gain" verbatim in the output.
from schemas.calculate import Direction as _Direction
from schemas.calculate import Outcome as _Outcome
from schemas.calculate import ScenarioType as _ScenarioType

_SCENARIO_DESCRIPTIONS = {
    _ScenarioType.NO_BUY: "considered buying but did not",
    _ScenarioType.NO_SELL: "held a position and considered selling but did not",
    _ScenarioType.SOLD_TOO_EARLY: "sold the position",
}

_DIRECTION_DESCRIPTIONS = {
    _Direction.MISSED_GAIN: "missed a potential gain",
    _Direction.AVOIDED_LOSS: "avoided a potential loss",
    _Direction.KEPT_GAIN: "captured a gain by holding",
    _Direction.ENDURED_LOSS: "absorbed a decline by holding",
    _Direction.CUT_SHORT_GAIN: "sold before further upside",
    _Direction.WELL_TIMED_EXIT: "sold before a decline",
    _Direction.NEUTRAL: "saw minimal price change",
}

_OUTCOME_LABELS = {
    _Outcome.FAVORABLE: "favorable",
    _Outcome.UNFAVORABLE: "unfavorable",
    _Outcome.NEUTRAL: "neutral",
}


def _format_decision_block(d, label: str) -> str:
    """Render one decision as natural-language prose. Used for both the
    current decision and each history entry.
    """
    return (
        f"{label}:\n"
        f"  - Ticker: {d.ticker}\n"
        f"  - User action: {_SCENARIO_DESCRIPTIONS[d.scenario_type]} on {d.decision_date.isoformat()}\n"
        f"  - Price change since: {d.diff_percent:+.2f}%\n"
        f"  - Result: {_DIRECTION_DESCRIPTIONS[d.direction]} ({_OUTCOME_LABELS[d.outcome]} outcome)"
    )


def _format_reflect_input(req: ReflectRequest) -> str:
    """Compose the user message. With history, the model anchors observations
    in the user's actual record; without it, the reflection describes only
    this decision.
    """
    current = _format_decision_block(req, "Current decision")
    if not req.previous_decisions:
        return f"{current}\n\nPrevious decisions: (none — this is the user's first reflection)"

    history = "\n\n".join(
        _format_decision_block(d, f"Previous decision #{idx + 1}")
        for idx, d in enumerate(req.previous_decisions)
    )
    return (
        f"{current}\n\n"
        f"Previous decisions ({len(req.previous_decisions)}, most recent first):\n\n"
        f"{history}"
    )


def generate_reflection(req: ReflectRequest) -> ReflectResponse:
    """Generate a single-decision reflection. Retries up to 3 times if the
    response trips a guardrail; returns degraded=True if all 3 fail.
    """
    client = _get_client()
    user_message = _format_reflect_input(req)

    for attempt in range(1, REFLECT_MAX_ATTEMPTS + 1):
        try:
            response = client.messages.create(
                model=REFLECT_MODEL,
                max_tokens=REFLECT_MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": REFLECT_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_message}],
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
            logger.warning("Anthropic API error during reflect: %s", exc)
            raise HTTPException(
                status_code=502,
                detail={
                    "error": {
                        "code": "LLM_PROVIDER_ERROR",
                        "message": str(exc),
                    }
                },
            )

        # Pull the text out
        text_block = next(
            (block for block in response.content if block.type == "text"),
            None,
        )
        if text_block is None:
            logger.warning("Reflect attempt %d returned no text block", attempt)
            continue
        candidate = text_block.text.strip()
        if not candidate:
            logger.warning("Reflect attempt %d returned empty text", attempt)
            continue

        forbidden = _has_forbidden(candidate)
        if forbidden is None:
            return ReflectResponse(
                reflection=candidate,
                degraded=False,
                attempts=attempt,
            )

        logger.info(
            "Reflect attempt %d hit guardrail (%r): %s",
            attempt,
            forbidden,
            candidate,
        )

    # All attempts violated guardrails — degrade gracefully
    return ReflectResponse(
        reflection="",
        degraded=True,
        attempts=REFLECT_MAX_ATTEMPTS,
    )


# ==========================================================================
# Pattern-level insights (F-06) — multi-decision behavioral observations
# ==========================================================================

from dataclasses import dataclass

INSIGHTS_MODEL = REFLECT_MODEL  # Sonnet 4.6 — same tone calibration as /reflect
INSIGHTS_MAX_TOKENS = 800
INSIGHTS_MAX_ATTEMPTS = 3
INSIGHTS_MIN_COUNT = 3
INSIGHTS_MAX_COUNT = 5


@dataclass
class InsightsResult:
    """Internal return type for generate_insights — the pattern service wraps
    this with cache + timestamp metadata before returning to the API layer."""
    insights: list[str]
    degraded: bool


INSIGHTS_SYSTEM_PROMPT = """You are analyzing a personal investor's full decision history for If-Vest to surface observational patterns.

Your job: read the aggregated stats in the user message and call the `report_insights` tool with 3-5 one-sentence behavioral observations.

## Rules
- Observe, never advise. No "you should..." sentences, no imperatives.
- Every insight must reference SPECIFIC numbers from the stats. Vague claims like "you're improving" are not insights.
- No predictions about future market moves or individual stock prices.
- No stock recommendations or strategy suggestions.
- Neutral, factual tone. A quiet analyst's voice, not a coach's.
- If a stat category has under 3 entries, don't generalize from it.

## Good insights (write like these)
- "Of your 12 logged decisions, 7 were no_buy; 5 of those 7 turned unfavorable — hesitation around eventually-rising names is your dominant pattern so far."
- "Decisions logged while anxious have a 25% favorable rate (1 of 4), versus 80% (4 of 5) when confident."
- "NVDA appears in 5 of your 12 decisions — heavier single-ticker concentration than any other name in your record."
- "Your last 4 outcomes are unfavorable, the longest such streak in your logged history."

## Bad insights (NEVER write these)
- "You should diversify more." (advice)
- "NVDA is likely to keep rising." (prediction)
- "Your decision-making has improved." (vague, no number)
- "Consider sitting out of volatile stocks." (advice)

## Output language
The input uses internal terms like `no_buy`, `no_sell`, `sold_too_early`, `missed_gain`, `favorable`, etc. Your OUTPUT may use the scenario types (`no_buy`, `no_sell`, `sold_too_early`) verbatim since users see those labels on their decisions. Translate the rest to plain English (e.g., "favorable outcome" stays as "favorable", "missed_gain" becomes "missing a gain").

## STRICT prohibitions
- No "should", "shouldn't", "must", "need to", "ought to", "next time", "going forward"
- No "recommend", "suggest", "advise"
- No "will rise/fall/drop/jump", "likely to", "expected to", "going to"
- No "diversify", "rebalance", "take profits", "stop-loss", "hold longer"
- No emojis, no exclamation marks
- No code-leaked enums beyond the scenario_type names (no `missed_gain`, `avoided_loss`, `direction`, `diff_percent` in your output)

## Tone
- Plain observational. 1 sentence per insight. Compact. Each insight stands alone.
- Order from most striking to supporting.

## Output rules
- Always call `report_insights`. Never respond in plain text.
- Return 3-5 insights. If you can only find 3 honest, data-anchored observations, return 3 — never pad.
"""


INSIGHTS_TOOL: dict[str, Any] = {
    "name": "report_insights",
    "description": (
        "Return 3-5 one-sentence behavioral observations from the user's decision history. "
        "Each must reference specific numbers from the input stats."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "insights": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": INSIGHTS_MIN_COUNT,
                "maxItems": INSIGHTS_MAX_COUNT,
                "description": "Each item is one self-contained observation, 1 sentence.",
            }
        },
        "required": ["insights"],
    },
}


def _count_dict(items, key_fn) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        k = key_fn(it)
        if k is None:
            continue
        out[k] = out.get(k, 0) + 1
    return out


def format_insights_context(items) -> str:
    """Format decision history as natural-language stats for the LLM.

    `items` is a list of DecisionWithCurrent (most-recent-first by created_at,
    per get_user_decisions' default sort). Imported lazily to avoid circular
    imports — pattern_service imports llm_service, not the other way around.
    """
    total = len(items)
    lines = [f"Total logged decisions: {total}"]

    scenario_counts = _count_dict(items, lambda it: it.scenario_type.value)
    outcome_counts = _count_dict(items, lambda it: it.outcome.value if it.outcome else None)
    emotion_counts = _count_dict(items, lambda it: it.emotion.value if it.emotion else None)
    direction_counts = _count_dict(items, lambda it: it.direction.value if it.direction else None)

    lines.append("\nScenario distribution:")
    for s, c in sorted(scenario_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  - {s}: {c}")

    lines.append("\nOutcome distribution:")
    for o, c in sorted(outcome_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  - {o}: {c}")

    lines.append("\nDirection distribution:")
    for d, c in sorted(direction_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  - {d}: {c}")

    if emotion_counts:
        lines.append("\nEmotion at decision time:")
        for e, c in sorted(emotion_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  - {e}: {c}")
    else:
        lines.append("\nEmotion at decision time: not logged for any decision yet.")

    # Scenario × outcome
    scenario_outcome: dict[str, dict[str, int]] = {}
    for it in items:
        if not it.outcome:
            continue
        s_key = it.scenario_type.value
        o_key = it.outcome.value
        scenario_outcome.setdefault(s_key, {})[o_key] = (
            scenario_outcome.get(s_key, {}).get(o_key, 0) + 1
        )
    if scenario_outcome:
        lines.append("\nScenario x outcome breakdown:")
        for s, counts in scenario_outcome.items():
            total_s = sum(counts.values())
            fav = counts.get("favorable", 0)
            unfav = counts.get("unfavorable", 0)
            neutral = counts.get("neutral", 0)
            lines.append(
                f"  - {s} (n={total_s}): {fav} favorable, {unfav} unfavorable, {neutral} neutral"
            )

    # Emotion × outcome
    if emotion_counts:
        emotion_outcome: dict[str, dict[str, int]] = {}
        for it in items:
            if not it.emotion or not it.outcome:
                continue
            e_key = it.emotion.value
            o_key = it.outcome.value
            emotion_outcome.setdefault(e_key, {})[o_key] = (
                emotion_outcome.get(e_key, {}).get(o_key, 0) + 1
            )
        if emotion_outcome:
            lines.append("\nEmotion x outcome breakdown (only emotions logged):")
            for e, counts in emotion_outcome.items():
                total_e = sum(counts.values())
                fav = counts.get("favorable", 0)
                unfav = counts.get("unfavorable", 0)
                lines.append(f"  - {e} (n={total_e}): {fav} favorable, {unfav} unfavorable")

    # Top tickers (top 5 by frequency)
    ticker_counts = _count_dict(items, lambda it: it.ticker)
    top_tickers = sorted(ticker_counts.items(), key=lambda x: -x[1])[:5]
    if top_tickers and top_tickers[0][1] >= 2:
        lines.append("\nMost frequent tickers (top 5 by count):")
        for t, c in top_tickers:
            lines.append(f"  - {t}: {c}")

    # Recent streak of outcomes (most-recent-first)
    recent = [it.outcome.value for it in items[:5] if it.outcome]
    if recent:
        lines.append(f"\nMost recent outcomes (newest first, up to 5): {', '.join(recent)}")

    # Average diff_percent — give LLM a single-number magnitude sense
    diffs = [it.current.diff_percent for it in items if it.current]
    if diffs:
        avg_diff = sum(diffs) / len(diffs)
        lines.append(f"\nAverage price change since decision across all: {avg_diff:+.2f}%")

    return "\n".join(lines)


def generate_insights(items) -> InsightsResult:
    """Generate 3-5 behavioral insights via LLM. Retries on guardrail violations
    or under-count responses. Returns degraded=True with empty insights when
    all attempts fail.

    Caller is responsible for verifying len(items) >= 10 before invoking.
    """
    client = _get_client()
    user_message = format_insights_context(items)

    for attempt in range(1, INSIGHTS_MAX_ATTEMPTS + 1):
        try:
            response = client.messages.create(
                model=INSIGHTS_MODEL,
                max_tokens=INSIGHTS_MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": INSIGHTS_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=[INSIGHTS_TOOL],
                tool_choice={"type": "tool", "name": "report_insights"},
                messages=[{"role": "user", "content": user_message}],
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
            logger.warning("Anthropic API error during insights: %s", exc)
            raise HTTPException(
                status_code=502,
                detail={
                    "error": {
                        "code": "LLM_PROVIDER_ERROR",
                        "message": str(exc),
                    }
                },
            )

        tool_use_block = next(
            (b for b in response.content if b.type == "tool_use"),
            None,
        )
        if tool_use_block is None:
            logger.warning("Insights attempt %d returned no tool_use block", attempt)
            continue

        raw_insights = tool_use_block.input.get("insights") or []
        if not isinstance(raw_insights, list) or len(raw_insights) < INSIGHTS_MIN_COUNT:
            logger.warning(
                "Insights attempt %d returned %d items (need >= %d)",
                attempt,
                len(raw_insights) if isinstance(raw_insights, list) else 0,
                INSIGHTS_MIN_COUNT,
            )
            continue

        # Drop empty strings, then guardrail-check the survivors.
        cleaned = [str(s).strip() for s in raw_insights if str(s).strip()]
        if len(cleaned) < INSIGHTS_MIN_COUNT:
            logger.warning("Insights attempt %d had only %d non-empty entries", attempt, len(cleaned))
            continue

        forbidden_hit: Optional[str] = None
        for ins in cleaned:
            hit = _has_forbidden(ins)
            if hit:
                forbidden_hit = hit
                logger.info("Insight tripped guardrail (%r): %s", hit, ins)
                break

        if forbidden_hit is None:
            return InsightsResult(insights=cleaned[:INSIGHTS_MAX_COUNT], degraded=False)

    return InsightsResult(insights=[], degraded=True)
