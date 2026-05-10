# API Conventions

Global rules every endpoint must follow. The OpenAPI spec at `api.yaml` is the source of truth for shapes; this doc explains the why and the rules that aren't easy to encode in OpenAPI.

## 1. Naming

- **JSON keys: `snake_case`** (e.g. `decision_date`, `current_price`).
- **URL paths: `kebab-case`** (e.g. `/optimal-timing`, `/parse-decision`).
- **Query params: `snake_case`** (e.g. `?start_date=...`).
- **Tickers: always uppercase in both request and response.** Backend normalizes (`nvda` → `NVDA`).

## 2. Dates and times

- **Date-only fields** (e.g. `decision_date`): `YYYY-MM-DD` (ISO 8601 date).
- **Datetime fields** (e.g. `created_at`): full ISO 8601 with timezone (`2026-05-05T14:32:00Z`).
- **Market data is ET-based.** When a user supplies a date and the US market was closed (weekend/holiday), the backend silently uses the **most recent prior trading day** and includes `actual_date_used` in the response so the UI can disclose it.
- **`current_date` is always the most recent close.** Even when the market is open intraday, we use yesterday's (or today's if after close) adjusted close. This guarantees the same decision returns the same result whenever the user reloads.
- **Reject future dates** with 422.

## 3. Money

- **Currency: USD only for MVP.** No conversion, no `currency` field in MVP responses.
- **Prices: number with up to 4 decimal places** (e.g. `118.5024`).
- **Amounts: number with 2 decimal places** (e.g. `1185.00`).
- **Use Yahoo Finance Adjusted Close** for all calculations (handles splits/dividends). UI shows "split-adjusted" disclaimer where relevant.

## 4. Authentication

- **Bearer**: Supabase JWT in `Authorization: Bearer <token>`. Verify with HS256, audience `authenticated`. The `sub` claim is the Supabase `auth.users.id` (uuid). Use as `user_id`.
- **Admin token**: `X-Admin-Token` header. Used only by `/admin/*` endpoints. Value comes from `ADMIN_TOKEN` env var (backend-only secret, not in client).
- **Public endpoints** (no auth): `/tickers/validate`, `/tickers/search`, `/calculate`, `/optimal-timing`, `/price-history`, `/health`. These are pure compute or upstream proxy.
- **Authed endpoints**: anything that touches `decisions`, `patterns`, or LLM (F-05/F-06).

## 5. Errors

Single shape across all endpoints:

```json
{
  "error": {
    "code": "TICKER_NOT_FOUND",
    "message": "Ticker 'ABCXYZ' could not be found.",
    "details": null
  }
}
```

- `code`: SCREAMING_SNAKE_CASE, stable, programmatic.
- `message`: human-readable, English (target audience), no stack trace.
- `details`: optional. For 422 validation errors, an array of `{ field, message }`. Null otherwise.

### Status codes

| Status | When |
|---|---|
| `200` | Success with body |
| `201` | Resource created (POST /decisions, POST /admin/seed-demo) |
| `204` | Success no body (DELETE /decisions/:id) |
| `400` | Malformed request (bad JSON, etc.) |
| `401` | Missing or invalid JWT / admin token |
| `403` | Authenticated but not authorized |
| `404` | Resource not found |
| `409` | Business-rule conflict (e.g. 50-decision cap reached) |
| `422` | Validation failure (bad date, missing field, range too large) |
| `429` | Rate limit hit |
| `500` | Unhandled server error |
| `502` / `503` | Yahoo Finance / LLM provider down |

### Error code catalog (extend as needed)

- `TICKER_NOT_FOUND` — 422
- `TICKER_DELISTED` — 422 (still computes, but flagged in response)
- `DATE_IN_FUTURE` — 422
- `DATE_BEFORE_LISTING` — 422
- `RANGE_TOO_LARGE` — 422 (>5y on /price-history or /optimal-timing)
- `QUANTITY_AND_AMOUNT_NEITHER` — 422 (must provide at least one)
- `END_DATE_BEFORE_START` — 422
- `DECISION_LIMIT_REACHED` — 409 (50-cap)
- `RATE_LIMITED` — 429
- `LLM_PROVIDER_ERROR` — 502
- `YAHOO_FINANCE_ERROR` — 502
- `INVALID_TOKEN` — 401
- `INVALID_ADMIN_TOKEN` — 401
- `UNAUTHORIZED` — 403

## 6. Response shape

- **No envelope.** Endpoints return the resource directly.
- **Lists return `{ items, total }`.** No cursor for MVP — the user cap is 50, so always cheap to send all rows.
- **Always return what was used.** If backend silently coerced (e.g. weekend → prior trading day), include the actual value used (`actual_date_used`).

## 7. Filters and sorting (`/decisions` GET)

- `?ticker=NVDA`
- `?scenario_type=no_buy`
- `?from=2024-01-01&to=2024-12-31` (decision_date range)
- `?sort=decision_date` or `?sort=-decision_date` (desc with `-` prefix). Default: `-created_at`.

No `limit` or `cursor` — we always return everything that matches.

## 8. Ticker validation flow

### Source of truth

`/tickers/validate` is the **only** authoritative source for "is this a real ticker". All other endpoints that take a ticker should call the validation logic internally and return `TICKER_NOT_FOUND` (422) on miss.

### Static index + yfinance hybrid

The backend ships with `data/tickers_index.json` — a pre-built list of all NASDAQ + NYSE listings (~5000 entries: `{ticker, name, exchange}`). Generated once from NASDAQ Trader's public symbol directory, committed to the repo, refreshed manually when needed.

**Lookup order** for both `/tickers/validate` and `/tickers/search`:

1. **Static index first** (in-memory, sub-millisecond).
2. **yfinance fallback** only when the static index doesn't cover the case:
   - `validate`: ticker not in static index → check yfinance (handles new IPOs, symbols added since last refresh).
   - `search`: static index returns < 3 results → augment with yfinance Search.

### Filtering

`/tickers/search` returns only `EQUITY` and `ETF` quote types. No mutual funds, currencies, futures, or crypto. The static index is pre-filtered to these types; yfinance results are filtered post-fetch.

### Empty input

`/tickers/search?q=` (empty) returns `200 { items: [] }`. NOT 422. The frontend often clears its input field; the backend handles that case so the client doesn't have to guard every call site.

### Ranking

Within `/tickers/search`:

1. Ticker prefix match (highest)
2. Name prefix match
3. Name substring match (lowest)

Within a tier: alphabetical by ticker. yfinance results, when augmenting, append after static results unless they offer a higher-tier match.

### Upstream failures

`/tickers/validate` can return 502 on yfinance failure (the caller needs to know the lookup didn't complete).

`/tickers/search` must **never** return 502 — degrading to "fewer results from static index" or "empty list" is acceptable. Search is non-critical UX; a hard error blocks the input.

### Caching

In-memory TTL cache (no DB needed):

| Endpoint | Key | TTL | Reason |
|---|---|---|---|
| `/tickers/validate` | normalized ticker | 24 hours | Existence rarely changes |
| `/tickers/search` | `(q, limit)` | 1 hour | Names/exchanges rarely change |

Use `cachetools.TTLCache` or equivalent. Reset on backend restart is fine.

### Normalization

Always uppercase + strip whitespace before pattern check or lookup. `?ticker= aapl ` → treated as `AAPL`. Spec's `Ticker` regex (`^[A-Z][A-Z0-9.-]{0,9}$`) is checked **after** normalization.

## 9. Caching

- `GET /price-history` and `/calculate` may hit Yahoo Finance. Backend caches in `price_cache` table by `(ticker, date)`. Cache fresh for closed market days; refresh same-day for current trading day.
- `GET /tickers/validate` and `/tickers/search` use in-memory TTL caches (see §8).
- LLM responses (F-06 patterns/insights) cached per-user in an `insights_cache` table. Invalidated lazily: on `GET /patterns/insights`, compare cache `generated_at` vs the user's `decisions.updated_at` MAX. If decisions are newer, regenerate.
- Public endpoints set `Cache-Control: public, max-age=300` for non-current dates; `no-store` for current trading day.

## 10. Rate limiting

Per-user (when authed) or per-IP (when public):

| Endpoint | Limit |
|---|---|
| `POST /parse-decision` | 30 / day / user |
| `GET /patterns/insights` | 5 / day / user (and only on decisions change due to lazy regen) |
| `POST /admin/seed-demo` | 10 / day / admin token |
| All others | 600 / hour / IP (default) |

Returns `429` with `Retry-After` header.

## 11. CORS

Allow only:

- `http://localhost:3000` (dev)
- `https://if-vest.vercel.app` (production frontend)
- `https://*-eunjeonghur.vercel.app` (Vercel preview deploys for this scope)
- Custom domain (TBD when we move off `*.vercel.app`)

Methods: `GET, POST, DELETE, OPTIONS`. Headers: `Authorization, Content-Type, X-Admin-Token`. No wildcard origins.

## 12. AI guardrails (applies to F-05, F-06)

- **Output post-processing**: Reject any response containing forbidden phrases (`buy`, `sell`, `recommend`, `will rise`, `will fall`, `should`). Regenerate up to 3 times. After 3 failures, return only the F-04 numeric stats with `degraded: true`.
- **Structured output**: Both endpoints use JSON Schema mode. No free-form text in `extracted` or `insights` fields beyond what the schema allows.
- **Token logging**: Every LLM call logs `{ user_id, endpoint, model, input_tokens, output_tokens, duration_ms }` for cost monitoring.
- **Provider-agnostic**: Backend abstracts behind an `LLMProvider` interface. F-05 uses cheap/fast model (Claude Haiku / GPT-4o-mini); F-06 uses stronger model (Claude Sonnet / GPT-4o).

## 13. Computed metrics

These are not just "data we store" but formulas the backend executes. Spelled out here so the implementation matches the spec.

### Direction / Outcome

```
direction = f(scenario_type, sign(diff_percent))

| scenario_type   | up               | down             |
|-----------------|------------------|------------------|
| no_buy          | missed_gain      | avoided_loss     |
| no_sell         | kept_gain        | endured_loss     |
| sold_too_early  | cut_short_gain   | well_timed_exit  |

When |diff_percent| < 0.5 → direction = neutral.

outcome:
  favorable    = direction in { avoided_loss, kept_gain, well_timed_exit }
  unfavorable  = direction in { missed_gain, endured_loss, cut_short_gain }
  neutral      = direction == neutral

was_decision_correct:
  true   when outcome == favorable
  false  when outcome == unfavorable
  null   when outcome == neutral
```

### Peak (used by F-04)

For each saved decision:
- `peak_in_window` = max(adjusted_close) over (`actual_date_used`, `current_date`]
- `decision_distance_from_peak_percent` = ((decision_price - peak_in_window) / peak_in_window) * 100
  - Always ≤ 0 (peak is by definition ≥ decision_price within the window).

### avg_distance_from_peak_percent

Mean of `decision_distance_from_peak_percent` across all of the user's decisions.

### consistency_score

```
distances = [decision_distance_from_peak_percent for d in user.decisions]
sd = stddev(distances)
consistency_score = 1 - min(sd / 30, 1)
```

- Range: [0, 1]. Higher = more consistent.
- Threshold `30` is the value of stddev considered "fully inconsistent" (score = 0). Anything more variable than 30% spread also caps at 0.
- **This threshold is preliminary.** Will be retuned with simulated/seeded data before week 4 development. Do not let it bake into client logic.

### regret_score (used to rank `most_regretted_top3`)

```
regret_score = abs(current.diff_percent)
```

Sort descending, take top 3. Ties broken by `created_at` descending (more recent first).

## 14. Request validation rules (server-side, regardless of client)

- At least one of `quantity` / `amount` must be present; both may be supplied.
  When both are present, the effective `decision_price` is computed as
  `amount / quantity` (treated as the user's actual transaction price) and
  the response field `decision_price_source` is `user`. When only one is
  supplied, the yfinance adjusted close on `actual_date_used` is used and
  `decision_price_source` is `yfinance`.
- `quantity > 0`, `amount > 0`.
- `decision_date <= today`.
- `end_date >= decision_date` (when provided).
- `ticker` matches `^[A-Z][A-Z0-9.-]{0,9}$` (1-10 chars, starts with letter).
- `notes` ≤ 500 chars.
- `text` (parse-decision) ≤ 1000 chars.
- For `/optimal-timing` and `/price-history`: `end_date - start_date <= 5 years` else `RANGE_TOO_LARGE`.

## 15. Versioning

- Path-based versioning if and when needed: `/v2/calculate`. MVP is unversioned (root `/calculate`).
- Breaking changes require a v2 path; non-breaking additions go on the existing path.

## 16. OpenAPI as source of truth

- `docs/api.yaml` is canonical. PRs that change endpoint shape must update it first.
- We use **OpenAPI 3.0.3** (not 3.1) for max tooling compatibility.
- Frontend generates types: `npx openapi-typescript ../kdd-ai-project-be/docs/api.yaml -o lib/api/types.ts`.
- Backend either implements by hand against the spec, or uses `datamodel-code-generator` to scaffold Pydantic models.
- CI should diff the spec against the running server's `/openapi.json` and fail if they drift (TBD post-MVP).

## 17. Demo seeding

`/admin/seed-demo` exists specifically for presentation day. Before a demo:

1. Create a real Supabase user (`demo@if-vest.com`) via signup.
2. Get the user's id from Supabase Dashboard → Auth → Users.
3. `POST /admin/seed-demo` with that user_id + `count: 50` and `clear_existing: true`.
4. Log in as that user during the demo. F-04/F-06 are unlocked; insights show realistic patterns.

The seed should pull from a fixed scenario library committed in the backend repo (e.g. `app/data/seed_scenarios.json`) so demos are reproducible.

## 18. Open questions (resolve before week 4)

- [ ] **`consistency_score` threshold (30)**: retune with seeded data.
- [ ] **Peak window definition**: confirmed as (`actual_date_used`, `current_date`]. If a decision has `end_date`, do we cap the window at end_date? (Currently no; document if changed.)
- [ ] **Beta user recruitment channel.**
- [ ] **Domain decision** for production CORS + custom email-from address.
- [ ] **Email confirmation toggle** in Supabase.
- [ ] **Vercel preview CORS glob**: `*-eunjeonghur.vercel.app` is the assumed pattern; verify with the actual scope in vercel deploy URLs.
