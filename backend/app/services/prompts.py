from typing import List, Dict


def build_rag_messages(
    question: str,
    context_chunks: List[Dict],
    system_prompt: str,
) -> List[Dict[str, str]]:
    """
    Build chat messages for RAG answering.
    Returns a list of {role, content} dicts.
    """
    context_lines = []
    for idx, chunk in enumerate(context_chunks, start=1):
        tag = f"[S{idx}]"
        context_lines.append(f"{tag} {chunk['content']}")

    context_block = "\n\n".join(context_lines)

    user_message = (
        "Context:\n"
        f"{context_block}\n\n"
        f"Question: {question}\n"
        "Answer with citations to the sources."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
