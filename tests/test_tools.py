"""Unit tests — pure parsers + the job manager. No real GPU, HTTP, or models."""
import asyncio

import pytest

from llm_lifecycle_mcp.jobs import JobManager
from llm_lifecycle_mcp.tools.benchmarks import _read_llama_bench_csv
from llm_lifecycle_mcp.tools.quantization import _parse_ppl
from llm_lifecycle_mcp.tools.serving import _find_knee, _parse_prom


def test_parse_ppl():
    out = _parse_ppl("... Final estimate: PPL = 8.0496 +/- 0.29324\n")
    assert out["perplexity"] == 8.0496
    assert out["stderr"] == 0.29324


def test_parse_ppl_missing():
    assert _parse_ppl("no number here")["perplexity"] is None


def test_parse_prom():
    text = (
        "# HELP vllm:num_requests_running\n"
        "vllm:num_requests_running{model=\"q\"} 3.0\n"
        "vllm:num_requests_waiting{model=\"q\"} 1.0\n"
        "vllm:gpu_cache_usage_perc{model=\"q\"} 0.42\n"
        "some_other_metric 99\n"
    )
    out = _parse_prom(text)
    assert out == {"requests_running": 3.0, "requests_waiting": 1.0, "kv_cache_usage": 0.42}


def test_find_knee():
    sweep = [
        {"concurrency": 1, "aggregate_tps": 30},
        {"concurrency": 4, "aggregate_tps": 120},
        {"concurrency": 8, "aggregate_tps": 238},
        {"concurrency": 16, "aggregate_tps": 238},
    ]
    assert _find_knee(sweep) == 8


def test_read_llama_bench_csv(tmp_path):
    csv = tmp_path / "q4.csv"
    csv.write_text(
        "model_type,n_prompt,n_gen,avg_ts\n"
        "llama 8B Q4_K - Medium,512,0,204.19\n"
        "llama 8B Q4_K - Medium,0,128,24.43\n"
    )
    out = _read_llama_bench_csv(csv)
    assert out["prefill_tps"] == 204.19
    assert out["decode_tps"] == 24.43


@pytest.mark.asyncio
async def test_job_runs_and_parses():
    jm = JobManager(cooldown_s=0)
    jid = await jm.submit("test", ["echo", "Final estimate: PPL = 7.95 +/- 0.10"], parse=_parse_ppl)
    for _ in range(100):
        if jm.get(jid).status != "running":
            break
        await asyncio.sleep(0.05)
    job = jm.get(jid)
    assert job.status == "done"
    assert job.result["perplexity"] == 7.95
    assert jm.current is None  # frees the slot when done


@pytest.mark.asyncio
async def test_one_job_at_a_time():
    jm = JobManager(cooldown_s=0)
    await jm.submit("slow", ["sleep", "0.4"])
    with pytest.raises(RuntimeError, match="already running"):
        await jm.submit("second", ["echo", "hi"])


@pytest.mark.asyncio
async def test_cooldown_blocks_back_to_back():
    jm = JobManager(cooldown_s=60)
    jid = await jm.submit("one", ["echo", "done"])
    for _ in range(100):
        if jm.get(jid).status != "running":
            break
        await asyncio.sleep(0.05)
    with pytest.raises(RuntimeError, match="Cooldown"):
        await jm.submit("two", ["echo", "hi"])
