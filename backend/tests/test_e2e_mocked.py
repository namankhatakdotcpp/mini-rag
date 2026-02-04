import hashlib
import re
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.db.client import SessionLocal
from app.main import app
from app.models.schemas import QueryResponse

TOKEN_PATTERN = re.compile(r"e2e-[0-9a-f\-]{36}")
VECTOR_DIMS = 1536


def _vector_from_key(key: str) -> list[float]:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    rng = seed
    vector = []
    for _ in range(VECTOR_DIMS):
        rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
        vector.append((rng % 1000) / 1000.0)
    return vector


def _extract_token(text: str) -> str:
    match = TOKEN_PATTERN.search(text)
    return match.group(0) if match else text


class MockEmbedder:
    def embed(self, texts):
        vectors = []
        for item in texts:
            token = _extract_token(item)
            vectors.append(_vector_from_key(token))
        return vectors


class MockLLMClient:
    def generate_answer(self, question, context_chunks):
        if not context_chunks:
            return "I don't know."
        return "Mocked answer based on [S1]."


@pytest.mark.e2e_mocked
def test_ingest_and_query_e2e_mocked(monkeypatch):
    monkeypatch.setattr("app.api.ingest.Embedder", MockEmbedder)
    monkeypatch.setattr("app.api.query.Embedder", MockEmbedder)
    monkeypatch.setattr("app.api.query.LLMClient", MockLLMClient)

    client = TestClient(app)

    unique_token = f"e2e-{uuid.uuid4()}"
    content = (
        f"This is a mocked e2e test document. Unique token: {unique_token}. "
        "It exists only to validate ingestion and retrieval."
    )

    ingest_response = client.post(
        "/api/ingest",
        data={"text": content},
    )

    assert ingest_response.status_code == 200

    db = SessionLocal()
    try:
        row = db.execute(
            text(
                """
                SELECT content, source, chunk_index, vector_dims(embedding) AS dims
                FROM documents
                WHERE content LIKE :pattern
                ORDER BY created_at DESC
                LIMIT 1
                """
            ),
            {"pattern": f"%{unique_token}%"},
        ).fetchone()
    finally:
        db.close()

    assert row is not None
    assert row.dims == VECTOR_DIMS

    query_response = client.post(
        "/api/query",
        json={"question": f"What is the unique token {unique_token}?"},
    )

    assert query_response.status_code == 200
    body = query_response.json()

    QueryResponse.model_validate(body)
    assert any(unique_token in src["content"] for src in body["sources"])
