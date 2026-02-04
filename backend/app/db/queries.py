from sqlalchemy import text

def insert_chunks(db, chunks, embeddings, source):
    """
    Inserts chunk + embedding pairs into the database.
    """
    for idx, (content, embedding) in enumerate(zip(chunks, embeddings)):
        db.execute(
            text("""
                INSERT INTO documents (content, embedding, source, chunk_index)
                VALUES (:content, :embedding, :source, :chunk_index)
            """),
            {
                "content": content,
                "embedding": embedding,
                "source": source,
                "chunk_index": idx
            }
        )
    db.commit()
