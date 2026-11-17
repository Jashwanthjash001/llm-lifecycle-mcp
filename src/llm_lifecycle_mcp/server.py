"""llm-lifecycle-mcp — an MCP server that lets an agent operate the local-LLM
lifecycle: quantize a model, measure its quality, serve it, and build RAG on it.

Transport is stdio (for Claude Desktop / Claude Code). Everything is wired through
config (env vars), so it points at your own llama.cpp, RAG, and vLLM setups — see
config.example.env.
"""
from __future__ import annotations

import sys

import httpx
from mcp.server.fastmcp import FastMCP

from .config import config
from .jobs import jobs
from .tools import benchmarks, quantization, rag, serving

mcp = FastMCP("llm-lifecycle")


@mcp.tool()
async def server_status() -> dict:
    """What this server is wired to, and whether each backend is reachable."""
    status = {
        "edge_ai_root": config.edge_ai_root or "(unset)",
        "edge_ai_ok": bool(config.edge_ai_root) and config.models_dir.exists(),
        "rag_api_url": config.rag_api_url or "(unset)",
        "vllm_url": config.vllm_url or "(unset)",
        "serving_lab_root": config.serving_lab_root or "(unset)",
        "running_job": jobs.current,
    }
    if config.rag_api_url:
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                status["rag_reachable"] = (await c.get(f"{config.rag_api_url}/health")).status_code == 200
        except Exception:  # noqa: BLE001
            status["rag_reachable"] = False
    if config.vllm_url:
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                status["vllm_reachable"] = (await c.get(f"{config.vllm_url}/v1/models")).status_code == 200
        except Exception:  # noqa: BLE001
            status["vllm_reachable"] = False
    return status


rag.register(mcp)
benchmarks.register(mcp)
quantization.register(mcp)
serving.register(mcp)


def main() -> None:
    for problem in config.startup_problems():
        print(f"llm-lifecycle-mcp: warning — {problem}", file=sys.stderr)
    mcp.run()


if __name__ == "__main__":
    main()
