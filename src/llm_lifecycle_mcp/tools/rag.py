"""RAG tools — talk to the local-document-assistant FastAPI app over HTTP, and
read its measured eval results."""
from __future__ import annotations

import json

from ..clients.http import rag_post
from ..config import config


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

    @mcp.tool()
    def rag_eval() -> dict:
        """The RAG app's measured evals (read-only): retrieval-mode comparison
        (dense vs BM25 vs hybrid) and quantization vs RAG answer quality."""
        results = config.rag_app_root / "evals" / "results"
        out: dict = {}
        rc = results / "retrieval_compare.json"
        qr = results / "quant_rag.json"
        if rc.exists():
            out["retrieval_modes"] = json.loads(rc.read_text())
        if qr.exists():
            out["quantization_vs_rag"] = json.loads(qr.read_text())
        if not out:
            return {"error": f"No eval results under {results}. Run evals/run_compare.py "
                             "and evals/run_rag_quality.py in the RAG app first."}
        return out
