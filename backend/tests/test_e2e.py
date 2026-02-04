import os
import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.e2e
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or not os.getenv("SUPABASE_DB_URL"),
    reason="Missing OPENAI_API_KEY or SUPABASE_DB_URL",
)
def test_ingest_and_query_e2e():
    client = TestClient(app)

    unique_token = f"e2e-{uuid.uuid4()}"
    content = (
        f"This is an end-to-end test document. Unique token: {unique_token}. "
        "It exists only to validate ingestion and retrieval."
    )

    ingest_response = client.post(
        "/api/ingest",
        data={"text": content},
    )

    assert ingest_response.status_code == 200

    query_response = client.post(
        "/api/query",
        json={"question": f"What is the unique token?"},
    )

    assert query_response.status_code == 200
    body = query_response.json()

    assert "answer" in body
    assert "sources" in body
    assert any(unique_token in src["content"] for src in body["sources"])
