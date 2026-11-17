"""Benchmark tools: compare existing results, run a new llama-bench, check jobs."""
from __future__ import annotations

import csv
from pathlib import Path

from ..config import config
from ..jobs import jobs


def _read_llama_bench_csv(path: Path) -> dict:
    """Pull prefill (pp) and decode (tg) throughput out of a llama-bench CSV."""
    rows = list(csv.DictReader(path.read_text().splitlines()))
    out: dict = {"file": path.name}
    for r in rows:
        tps = r.get("avg_ts") or r.get("t/s")
        out.setdefault("model", r.get("model_type") or r.get("model_filename"))
        if r.get("n_gen") not in (None, "", "0"):
            out["decode_tps"] = float(tps) if tps else None
        elif r.get("n_prompt") not in (None, "", "0"):
            out["prefill_tps"] = float(tps) if tps else None
    return out


def register(mcp) -> None:
    @mcp.tool()
    def compare_results() -> dict:
        """Compare the quantization results on disk — speed from the llama-bench CSVs,
        quality from perplexity.csv. Answers 'compare my Q4 and Q8 results'."""
        rdir = config.results_dir
        if not rdir.exists():
            return {"error": f"results dir not found: {rdir}"}
        speed = {}
        for csvf in sorted(rdir.glob("*.csv")):
            if csvf.name == "perplexity.csv":
                continue
            speed[csvf.stem] = _read_llama_bench_csv(csvf)
        quality = {}
        ppl = rdir / "perplexity.csv"
        if ppl.exists():
            for r in csv.DictReader(ppl.read_text().splitlines()):
                quality[r["quant"]] = {
                    "perplexity": float(r["perplexity"]),
                    "stderr": float(r["stderr"]),
                }
        return {"speed": speed, "quality": quality}

    @mcp.tool()
    async def run_benchmark(model_filename: str, label: str = "") -> dict:
        """Run a llama-bench throughput benchmark on a LOCAL GGUF model (submits a job;
        poll job_status). model_filename must be a .gguf already in the models dir.
        This never targets the production vLLM server — it only benchmarks local llama.cpp."""
        name = Path(model_filename).name  # strip any path — no traversal
        model = config.models_dir / name
        if not name.endswith(".gguf") or not model.exists():
            avail = [p.name for p in config.models_dir.glob("*.gguf")]
            return {"error": f"unknown model '{model_filename}'. Available: {avail}"}
        label = (label or model.stem).replace("/", "_")
        cmd = ["bash", str(config.bench_sh), str(model), label]
        try:
            job_id = await jobs.submit("benchmark", cmd, cwd=config.edge_ai_root)
        except RuntimeError as e:
            return {"error": str(e)}
        return {"job_id": job_id, "note": "poll job_status(job_id)"}

    @mcp.tool()
    def job_status(job_id: str) -> dict:
        """Status + result of a submitted job (benchmark / perplexity / quantize)."""
        job = jobs.get(job_id)
        if not job:
            return {"error": f"no job {job_id}"}
        return {
            "id": job.id,
            "kind": job.kind,
            "status": job.status,
            "returncode": job.returncode,
            "result": job.result,
            "output_tail": job.output_tail[-1200:],
        }
