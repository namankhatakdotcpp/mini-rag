# Mini RAG

A production‑minded Retrieval‑Augmented Generation (RAG) system built with FastAPI, PostgreSQL + pgvector, and OpenAI.

## Backend

### Run locally

```bash
cd backend
./venv/bin/uvicorn app.main:app --reload
```

### Environment

Create `backend/.env` with:

```
OPENAI_API_KEY=your_key_here
SUPABASE_DB_URL=postgresql://user:password@host:5432/postgres
```

Note: If your DB password contains special characters like `@`, URL‑encode them (e.g. `@` → `%40`).

### Testing

Run the full unit test suite:

```bash
cd backend
./venv/bin/pytest -q
```

Default CI‑safe run (includes mocked e2e coverage):

```bash
cd backend
./venv/bin/pytest -q -m e2e_mocked
```

Live OpenAI e2e (optional, requires real credentials and network access):

```bash
cd backend
./venv/bin/pytest -q -m e2e
```

## Frontend

The frontend is a static HTML console that calls the backend directly.

Open `frontend/index.html` in a browser and set the API base to your running backend (default `http://localhost:8000`).

If you serve the frontend from a different origin, ensure CORS is enabled on the backend.

