"""Serving tools: read the vLLM server's live metrics (read-only) and recommend an
operating concurrency from the measured sweep. These never send load to the server."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import config
from ..clients.http import vllm_get_text

# A few Prometheus series from vLLM's /metrics worth surfacing.
# (The KV-cache series was renamed across vLLM versions — accept either name.)
_WANTED = {
    "vllm:num_requests_running": "requests_running",
    "vllm:num_requests_waiting": "requests_waiting",
    "vllm:kv_cache_usage_perc": "kv_cache_usage",
    "vllm:gpu_cache_usage_perc": "kv_cache_usage",
}


def _parse_prom(text: str) -> dict:
    out: dict = {}
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        key, _, val = line.partition(" ")
        base = key.split("{", 1)[0]
        if base in _WANTED:
            try:
                out[_WANTED[base]] = float(val)
            except ValueError:
                pass
    return out


def _find_knee(rows: list[dict]) -> int | None:
    """Smallest concurrency reaching 95% of peak aggregate throughput."""
    rows = [r for r in rows if r.get("aggregate_tps")]
    if not rows:
        return None
    peak = max(r["aggregate_tps"] for r in rows)
    hit = [int(r["concurrency"]) for r in rows if r["aggregate_tps"] >= 0.95 * peak]
    return min(hit) if hit else None


def register(mcp) -> None:
    @mcp.tool()
    async def vllm_metrics() -> dict:
        """Live, read-only snapshot of the vLLM server: running/waiting requests and
        KV-cache usage (from /metrics). This only reads — it never sends load."""
        try:
            text = await vllm_get_text("/metrics")
        except Exception as e:  # noqa: BLE001
            return {"error": f"vLLM metrics unavailable: {e}"}
        return _parse_prom(text)

    @mcp.tool()
    def recommend_config() -> dict:
        """Recommend an operating concurrency from the serving-lab sweep — the point of
        max throughput before latency spikes (the saturation knee)."""
        if not config.serving_lab_root:
            return {"error": "SERVING_LAB_ROOT not set (needs the serving-lab results)."}
        sweep = None
        for p in sorted(Path(config.serving_lab_root, "results").glob("sweep_*.json")):
            sweep = json.loads(p.read_text())
            break
        if not sweep:
            return {"error": "no sweep_*.json found in the serving-lab results."}
        knee = _find_knee(sweep)
        peak = max((r.get("aggregate_tps") or 0) for r in sweep)
        return {
            "recommended_concurrency": knee,
            "peak_throughput_tps": round(peak, 1),
            "rationale": (
                f"Throughput plateaus around concurrency {knee}; beyond it latency climbs "
                "for ~no extra throughput. Add a replica rather than push one engine harder."
            ),
        }
