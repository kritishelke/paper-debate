# PaperDebate

PaperDebate is a local multi-agent scientific paper debate system. It runs three heterogeneous agents across ingestion, claim-graph debate, sycophancy-triggered prompt optimization, and final team-answer scoring.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL and start a debate from an abstract. The backend has deterministic offline fallbacks, so the app and tests work without API keys. Set these for real providers:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
```

Optional environment variables:

```bash
GROBID_URL=http://localhost:8070
WANDB_PROJECT=paper-debate
RATINGS_DB_PATH=training/ratings.db
PO_MODEL_ID=...
```

## API

- `POST /debate` with `{ "abstract": "...", "focus": "..." }` or `{ "pdf_bytes": "base64..." }`
- `GET /debate/{id}/stream` for SSE events
- `GET /debate/{id}/graph`
- `GET /debate/{id}/metrics`
- `POST /debate/{id}/rating`
- `GET /training/status`
- `POST /training/trigger`

## Tests

```bash
pytest
```

The tests cover t0/t1/t2 trigger logic, team scoring, prompt optimizer fallback validation, training score weighting, agent JSON retry handling, and one mocked full debate loop.
