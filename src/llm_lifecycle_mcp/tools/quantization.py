"""Quantization tools: measure quality (perplexity) and create quantized models."""
from __future__ import annotations

import re
from pathlib import Path

from ..config import config
from ..jobs import jobs

ALLOWED_QUANTS = {"Q4_0", "Q4_K_M", "Q5_0", "Q5_K_M", "Q6_K", "Q8_0"}
_PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)")


def _parse_ppl(text: str) -> dict:
    m = _PPL_RE.search(text)
    if not m:
        return {"perplexity": None, "note": "could not parse PPL from output"}
    return {"perplexity": float(m.group(1)), "stderr": float(m.group(2))}


def _known_models() -> list[str]:
    return [p.name for p in config.models_dir.glob("*.gguf")]


def register(mcp) -> None:
    @mcp.tool()
    async def eval_perplexity(model_filename: str, dataset_path: str) -> dict:
        """Measure a GGUF model's perplexity (quality) on a text file via llama-perplexity.
        Submits a job; poll job_status for the number. Lower is better."""
        name = Path(model_filename).name
        model = config.models_dir / name
        if not name.endswith(".gguf") or not model.exists():
            return {"error": f"unknown model '{model_filename}'. Available: {_known_models()}"}
        data = Path(dataset_path).expanduser()
        if not data.exists():
            return {"error": f"dataset not found: {dataset_path}"}
        cmd = [
            str(config.bin_dir / "llama-perplexity"),
            "-m", str(model), "-f", str(data), "-ngl", "99", "-c", "512",
        ]
        try:
            job_id = await jobs.submit("perplexity", cmd, cwd=config.edge_ai_root, parse=_parse_ppl)
        except RuntimeError as e:
            return {"error": str(e)}
        return {"job_id": job_id, "note": "poll job_status(job_id) for the perplexity number"}

    @mcp.tool()
    async def quantize_model(source_filename: str, quant_type: str) -> dict:
        """Quantize a source GGUF to a lower precision (e.g. Q4_K_M) via llama-quantize.
        Submits a job; poll job_status. Output lands in the models dir."""
        qt = quant_type.upper()
        if qt not in ALLOWED_QUANTS:
            return {"error": f"quant_type must be one of {sorted(ALLOWED_QUANTS)}"}
        name = Path(source_filename).name
        src = config.models_dir / name
        if not name.endswith(".gguf") or not src.exists():
            return {"error": f"unknown source '{source_filename}'. Available: {_known_models()}"}
        out = config.models_dir / f"{src.stem}-{qt}.gguf"
        cmd = [str(config.bin_dir / "llama-quantize"), str(src), str(out), qt]
        try:
            job_id = await jobs.submit("quantize", cmd, cwd=config.edge_ai_root)
        except RuntimeError as e:
            return {"error": str(e)}
        return {"job_id": job_id, "output": out.name, "note": "poll job_status(job_id)"}
