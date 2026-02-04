from typing import List, Dict, Any
from sqlalchemy import text


class Retriever:
    """
    Performs similarity search against pgvector using raw SQL.
    """

    def __init__(self, db):
        self.db = db

    @staticmethod
    def _format_embedding(embedding: List[float]) -> str:
        """
        pgvector accepts embeddings as a vector literal string.
        We format to a stable, compact representation for SQL binding.
        """
        values = ",".join(f"{x:.8f}" for x in embedding)
        return f"[{values}]"

    def search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """
        Return top_k most similar chunks using L2 distance (<->).
        """
        embedding_literal = self._format_embedding(query_embedding)

        stmt = text(
            """
            SELECT
                id,
                content,
                source,
                section,
                chunk_index,
                created_at,
                embedding <-> :embedding::vector AS distance
            FROM documents
            ORDER BY embedding <-> :embedding::vector
            LIMIT :top_k
            """
        )

        result = self.db.execute(
            stmt,
            {
                "embedding": embedding_literal,
                "top_k": top_k,
            },
        )

        rows = result.fetchall()
        chunks: List[Dict[str, Any]] = []
        for row in rows:
            chunks.append(
                {
                    "id": str(row.id),
                    "content": row.content,
                    "source": row.source,
                    "section": row.section,
                    "chunk_index": row.chunk_index,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "distance": float(row.distance) if row.distance is not None else None,
                }
            )

        return chunks
