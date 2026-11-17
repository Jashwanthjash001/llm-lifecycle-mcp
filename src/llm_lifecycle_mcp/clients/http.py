"""Thin HTTP wrappers for the RAG app and the vLLM server."""
from __future__ import annotations

import httpx

from ..config import config


async def rag_post(path: str, payload: dict, timeout: float = 60.0) -> dict:
    if not config.rag_api_url:
        raise RuntimeError("RAG_API_URL is not set.")
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{config.rag_api_url}{path}", json=payload)
        r.raise_for_status()
        return r.json()


async def vllm_get_text(path: str, timeout: float = 10.0) -> str:
    if not config.vllm_url:
        raise RuntimeError("VLLM_URL is not set.")
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(f"{config.vllm_url}{path}")
        r.raise_for_status()
        return r.text
