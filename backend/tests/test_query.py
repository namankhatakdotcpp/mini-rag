from fastapi.testclient import TestClient

from app.main import app


class DummyEmbedder:
    def embed(self, texts):
        return [[0.1, 0.2, 0.3]]


class DummyRetriever:
    def __init__(self, db):
        self.db = db

    def search(self, query_embedding, top_k):
        return [
            {
                "id": "00000000-0000-0000-0000-000000000000",
                "content": "Alpha chunk",
                "source": "unit-test",
                "created_at": None,
                "distance": 0.01,
            }
        ]


class DummyLLM:
    def generate_answer(self, question, context_chunks):
        return "Answer from [S1]"


def test_query_endpoint(monkeypatch):
    monkeypatch.setattr("app.api.query.Embedder", DummyEmbedder)
    monkeypatch.setattr("app.api.query.Retriever", DummyRetriever)
    monkeypatch.setattr("app.api.query.LLMClient", DummyLLM)

    client = TestClient(app)

    response = client.post("/api/query", json={"question": "What is alpha?"})

    assert response.status_code == 200
    body = response.json()

    assert body["answer"] == "Answer from [S1]"
    assert len(body["sources"]) == 1
    assert body["sources"][0]["content"] == "Alpha chunk"
