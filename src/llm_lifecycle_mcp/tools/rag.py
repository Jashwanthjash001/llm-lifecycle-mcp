"""RAG tools — talk to the local-document-assistant FastAPI app over HTTP."""
from __future__ import annotations

from ..clients.http import rag_post


def register(mcp) -> None:
    @mcp.tool()
    async def search_documents(query: str, top_k: int = 5) -> dict:
        """Semantic search over the indexed documents (no LLM). Returns the top-k
        passages with page numbers and cosine distances."""
        try:
            return await rag_post("/search", {"query": query, "top_k": top_k})
        except Exception as e:  # noqa: BLE001
            return {"error": f"RAG search unavailable: {e}"}

    @mcp.tool()
    async def ask_documents(query: str) -> dict:
        """Ask a grounded question over the indexed documents. Returns an answer with
        page citations, or abstains when the documents don't support one."""
        try:
            return await rag_post("/chat", {"query": query})
        except Exception as e:  # noqa: BLE001
            return {"error": f"RAG chat unavailable: {e}"}
