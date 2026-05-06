# If-Vest API docs

Two files:

- **`api.yaml`** — OpenAPI 3.1 spec. Source of truth for endpoint shapes.
- **`conventions.md`** — Global rules (errors, dates, money, auth, CORS, rate limits, AI guardrails). Read this first.

## How each side uses these

### Backend (FastAPI, Python)

Two options:

1. **Generate Pydantic models from the spec:**
   ```bash
   pip install datamodel-code-generator
   datamodel-codegen \
     --input docs/api.yaml \
     --output app/schemas/generated.py \
     --output-model-type pydantic_v2.BaseModel
   ```
   Then write route handlers that accept/return those models.

2. **Hand-write Pydantic models** following the spec. Faster for small projects, easier to customize. FastAPI auto-generates an `/openapi.json` from your code; diff it against `docs/api.yaml` periodically to catch drift.

Either way, **the spec drives the contract**, not the code. Update the spec first, then the code.

### Frontend (Next.js, TypeScript)

Generate types:

```bash
npm i -D openapi-typescript
npx openapi-typescript ../kdd-ai-project-be/docs/api.yaml -o lib/api/types.ts
```

Then import:

```ts
import type { paths } from "@/lib/api/types";

type CalculateRequest =
  paths["/calculate"]["post"]["requestBody"]["content"]["application/json"];
type CalculateResponse =
  paths["/calculate"]["post"]["responses"][200]["content"]["application/json"];
```

A typed fetch client (`openapi-fetch` or hand-rolled) reads from these.

## Workflow

1. Spec change goes to `docs/api.yaml` first, in a PR.
2. Both sides review.
3. Once merged, frontend regenerates types, backend updates route handlers.
4. CI (TBD) compares the running server's `/openapi.json` to `docs/api.yaml` and fails on drift.

## Open questions (track here, resolve before week 4)

- [ ] **`consistency_score` threshold (30)**: formula is locked (`1 - min(sd/30, 1)`); the 30 is preliminary. Retune with seeded data.
- [ ] **Beta user recruitment channel.** PRD targets US-stock-investing 30s Korean. Where do we get them? Affects whether persona stays as-is or shifts.
- [ ] **Domain decision.** Need a real domain for production CORS + email-from address (currently using Supabase default SMTP).
- [ ] **Email confirmation toggle.** Decide whether to require email verification at signup. Affects `/signup` flow on the frontend.
- [ ] **Vercel preview CORS glob.** Verify the `*-eunjeonghur.vercel.app` pattern matches Vercel's actual preview URL format for this project.
- [ ] **`peak` window cap with `end_date`.** Currently the peak window for F-04 is (`actual_date_used`, `current_date`]. If a decision has `end_date` set, should we cap at `end_date` instead? Right now we don't.
