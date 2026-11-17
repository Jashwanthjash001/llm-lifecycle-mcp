"""Configuration from the environment.

An MCP client passes these via its "env" block; for local runs, export them or
drop them in a `config.env` next to the server (or point `LLM_LIFECYCLE_ENV` at a file).
Only EDGE_AI_ROOT is required — RAG, vLLM, and serving-lab tools disable themselves
cleanly when their variable is unset.
"""
from __future__ import annotations

import os
from pathlib import Path


def _load_env_file() -> None:
    """Minimal KEY=VALUE loader so local runs don't need python-dotenv."""
    path = Path(os.getenv("LLM_LIFECYCLE_ENV") or "config.env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


_load_env_file()


class Config:
    def __init__(self) -> None:
        self.edge_ai_root = os.getenv("EDGE_AI_ROOT", "").rstrip("/")
        self.rag_api_url = os.getenv("RAG_API_URL", "").rstrip("/")
        self.vllm_url = os.getenv("VLLM_URL", "").rstrip("/")
        self.serving_lab_root = os.getenv("SERVING_LAB_ROOT", "").rstrip("/")

    # edge-ai (llama.cpp) layout
    @property
    def bin_dir(self) -> Path:
        return Path(self.edge_ai_root) / "bin"

    @property
    def models_dir(self) -> Path:
        return Path(self.edge_ai_root) / "models"

    @property
    def results_dir(self) -> Path:
        return Path(self.edge_ai_root) / "results"

    @property
    def bench_sh(self) -> Path:
        return Path(self.edge_ai_root) / "bench.sh"

    @property
    def rag_app_root(self) -> Path:
        # EDGE_AI_ROOT is <local-document-assistant>/serving; the app and its
        # evals live one directory up.
        return Path(self.edge_ai_root).parent if self.edge_ai_root else Path()

    def startup_problems(self) -> list[str]:
        problems: list[str] = []
        if not self.edge_ai_root:
            problems.append("EDGE_AI_ROOT is not set — quantization/benchmark tools need it.")
        elif not Path(self.edge_ai_root).exists():
            problems.append(f"EDGE_AI_ROOT path does not exist: {self.edge_ai_root}")
        return problems


config = Config()
