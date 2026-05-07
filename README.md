# If-Vest Backend

FastAPI service for If-Vest. The contract is defined in [`docs/api.yaml`](docs/api.yaml) (OpenAPI 3.0.3); read [`docs/conventions.md`](docs/conventions.md) for the global rules.

Right now this is a minimal scaffold: only `GET /health` is wired up, just enough to deploy.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # fill in real values
uvicorn main:app --reload --port 8000
```

Smoke test:

```bash
curl http://localhost:8000/health
# -> {"status":"ok","version":"0.2.0","db":"ok","yfinance":"ok"}
```

FastAPI auto-generates interactive docs at:

- Swagger UI: http://localhost:8000/docs
- ReDoc:      http://localhost:8000/redoc
- Raw spec:   http://localhost:8000/openapi.json

## Deploy to Render

This repo includes [`render.yaml`](render.yaml). Two ways:

**Blueprint (recommended)**

1. Render dashboard → New → Blueprint → connect this repo.
2. Render reads `render.yaml`, creates the Web Service.
3. After it boots, set env vars in the service's **Environment** tab:
   - `SUPABASE_URL`
   - `SUPABASE_JWT_SECRET`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - `ANTHROPIC_API_KEY`
   - `ADMIN_TOKEN`
4. The service URL appears in the dashboard. Update `docs/api.yaml`'s `servers:` block with it and add it to the frontend's `NEXT_PUBLIC_API_URL`.

**Manual**

If you prefer to set it up by hand:

- Type: **Web Service** (NOT Static Site).
- Runtime: Python 3.
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health Check Path: `/health`
- Add env vars (same list as above).

## Free-tier warning

Render free Web Services sleep after 15 min idle. The next request takes 30+ seconds (cold start). For demos, hit `/health` from an external pinger (e.g. cron-job.org) every 5-10 min, or upgrade to Starter ($7/mo) the week of the demo.

## What's next

The current `main.py` is one file with a single endpoint. As more endpoints come in, restructure to:

```
app/
  main.py              # FastAPI app, middleware, router includes
  config.py            # Pydantic settings (env vars)
  routers/
    health.py
    tickers.py
    calculate.py
    decisions.py
    patterns.py
    parse_decision.py
    admin.py
  services/
    yfinance.py        # market data + caching
    supabase.py        # auth + DB client wrappers
    llm/
      provider.py      # provider-agnostic interface
      anthropic.py
      openai.py
    guardrails.py      # AI output post-processing
  schemas/             # Pydantic models (or generate from docs/api.yaml)
  db/
    client.py
    migrations/        # if not using Supabase migrations directly
```

Don't preemptively create empty files. Add modules as endpoints get implemented.
