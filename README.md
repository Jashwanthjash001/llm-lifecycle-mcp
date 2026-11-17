# llm-lifecycle-mcp

One MCP server that lets an AI agent **operate a local-LLM stack end to end** — quantize a model,
measure its quality, benchmark it, read a live serving endpoint, and ask questions of a RAG app —
without you touching the terminal.

It ties together three standalone projects through config (paths + URLs), copying none of their code:

```
                         ┌─────────────────────────────┐
   Claude (agent) ──MCP──│      llm-lifecycle-mcp       │
     stdio               │  server_status, compare,    │
                         │  eval, benchmark, quantize, │
                         │  rag, vllm_metrics, recommend│
                         └───┬──────────┬───────────┬───┘
            subprocess (bin/)│   HTTP   │   HTTP    │
                             ▼          ▼           ▼
                   ┌──────────────┐ ┌─────────┐ ┌──────────────┐
                   │ llama.cpp    │ │ RAG app │ │ vLLM /metrics│
                   │ quantize /   │ │ /search │ │ (read-only)  │
                   │ perplexity / │ │ /chat   │ │              │
                   │ bench        │ │         │ │              │
                   └──────────────┘ └─────────┘ └──────────────┘
          EDGE_AI_ROOT            RAG_API_URL      VLLM_URL
     (local-document-assistant/   (FastAPI RAG)  (serving-lab's
          serving)                                vLLM server)
```

## Tools

| Tool | What it does | Backend |
| ---- | ------------ | ------- |
| `server_status` | What's wired up and reachable | — |
| `compare_results` | Compare Q4 vs Q8 speed (llama-bench) + quality (perplexity) | `EDGE_AI_ROOT/results` |
| `search_documents` | Semantic search over indexed docs | RAG `/search` |
| `ask_documents` | Grounded answer with citations | RAG `/chat` |
| `rag_eval` | Measured RAG evals: retrieval modes + quant × RAG quality (read-only) | RAG app `evals/results` |
| `eval_perplexity` | Measure a GGUF's perplexity on a text file *(job)* | `bin/llama-perplexity` |
| `quantize_model` | Quantize a GGUF to a lower precision *(job)* | `bin/llama-quantize` |
| `run_benchmark` | llama-bench throughput on a local GGUF *(job)* | `bench.sh` |
| `job_status` | Poll a running/finished job | — |
| `vllm_metrics` | Live running/waiting requests + KV-cache use (read-only) | vLLM `/metrics` |
| `recommend_config` | Operating concurrency from the serving-lab sweep | `SERVING_LAB_ROOT/results` |

Long operations (`eval_perplexity`, `quantize_model`, `run_benchmark`) return a `job_id` — poll it
with `job_status`.

## Setup

```bash
cp config.example.env config.env   # fill in your paths/URLs
```

Run it from a client over stdio. Claude Desktop / Claude Code config:

```json
{
  "mcpServers": {
    "llm-lifecycle": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/Jashwanthjash001/llm-lifecycle-mcp", "llm-lifecycle-mcp"],
      "env": {
        "EDGE_AI_ROOT": "/path/to/local-document-assistant/serving",
        "RAG_API_URL": "http://localhost:8000",
        "VLLM_URL": "",
        "SERVING_LAB_ROOT": "/path/to/llm-serving-optimization-lab"
      }
    }
  }
}
```

Only `EDGE_AI_ROOT` is required — leave a URL empty and its tools disable themselves cleanly.

Local dev:

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
LLM_LIFECYCLE_ENV=config.env .venv/bin/python -m llm_lifecycle_mcp.server   # runs on stdio
```

## Guardrails

- **One job at a time**, with a cooldown between runs — no stacking GPU-heavy work.
- **Allow-lists, not arbitrary commands:** models must already exist in the models dir, quant types
  are a fixed set, and paths are stripped of traversal. No shell string is ever built from free text.
- **The serving endpoint is read-only.** `vllm_metrics` only reads `/metrics`; nothing here sends load
  to the vLLM server (it may be shared / in production).

## Limitations

- llama.cpp tools are macOS/Apple-Silicon by default (the bundled build); point `EDGE_AI_ROOT` at a
  Linux build to run elsewhere.
- Jobs are in-memory — they don't survive a server restart.
- `vllm_metrics` surfaces a few key series, not the full Prometheus set.
- No auth layer; it's meant to run locally over stdio, not exposed on a network.
