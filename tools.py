from langchain_core.tools import tool

from data_loader import embed_texts
from vector_db import QdrantStorage


@tool
def retrieve_documents(query: str, top_k: int = 5) -> str:
    """Search indexed PDF chunks for information relevant to the query."""
    query_vec = embed_texts([query])[0]
    found = QdrantStorage().search(query_vec, top_k)
    if not found["chunks"]:
        return "No relevant documents found."

    blocks = [
        f"[Source: {c['source']}]\n{c['text']}" for c in found["chunks"]
    ]
    return "\n\n---\n\n".join(blocks)


@tool
def list_indexed_documents() -> str:
    """List all PDF documents that have been indexed in the knowledge base."""
    sources = QdrantStorage().list_sources()
    if not sources:
        return "No documents have been indexed yet."
    return "Indexed documents:\n" + "\n".join(f"- {s}" for s in sources)


