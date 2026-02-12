"""A/B benchmark: eager vs torch.compile (cold start vs warm start).

Runs three passes on the same workload and writes a markdown results table.
Usage: python benchmark.py
"""

import atexit
import gc
import os
import shutil
import time
from random import randint, seed

import torch
from nanovllm import LLM, SamplingParams


MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-14B-Base/")
CACHE_ROOT = os.path.join(os.path.expanduser("~"), ".cache", "nanovllm", "torch_compile")
NUM_SEQS = 256
MAX_INPUT_LEN = 1024
MAX_OUTPUT_LEN = 1024
MAX_MODEL_LEN = 4096
TEMPERATURE = 0.6


def make_workload(num_seqs, max_input_len, max_output_len):
    """Generate deterministic random prompts and sampling params."""
    seed(0)
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=TEMPERATURE, ignore_eos=True,
                       max_tokens=randint(100, max_output_len))
        for _ in range(num_seqs)
    ]
    total_output_tokens = sum(sp.max_tokens for sp in sampling_params)
    return prompt_token_ids, sampling_params, total_output_tokens


def run_benchmark(enforce_eager: bool, label: str) -> dict:
    """Run one full benchmark pass. Returns a dict of metrics."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    prompt_token_ids, sampling_params, total_output_tokens = make_workload(
        NUM_SEQS, MAX_INPUT_LEN, MAX_OUTPUT_LEN
    )

    # --- Startup (model load + optional compilation) ---
    t_startup = time.perf_counter()
    llm = LLM(MODEL_PATH, enforce_eager=enforce_eager, max_model_len=MAX_MODEL_LEN)
    startup_time = time.perf_counter() - t_startup
    print(f"  Startup: {startup_time:.1f}s")

    # --- Warmup (first batch, may trigger remaining JIT work) ---
    llm.generate(["warmup"], SamplingParams(), use_tqdm=False)

    # --- Timed generation ---
    t_gen = time.perf_counter()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    gen_time = time.perf_counter() - t_gen
    throughput = total_output_tokens / gen_time

    print(f"  Generation: {gen_time:.2f}s, {throughput:.0f} tok/s")

    # --- Cleanup: destroy process group so next run can re-init ---
    # atexit handler would only fire at process exit, too late for us.
    llm.exit()
    atexit.unregister(llm.exit)
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "label": label,
        "startup_s": startup_time,
        "gen_time_s": gen_time,
        "throughput": throughput,
        "total_tokens": total_output_tokens,
    }


def write_results(results: list[dict], path: str):
    """Write benchmark results as a markdown table."""
    eager_throughput = results[0]["throughput"]

    lines = [
        "# Benchmark Results",
        "",
        f"Model: Qwen3-14B-Base | GPU: {torch.cuda.get_device_name(0)} | "
        f"Sequences: {NUM_SEQS} | Input: 100-{MAX_INPUT_LEN} tok | "
        f"Output: 100-{MAX_OUTPUT_LEN} tok",
        "",
        "| Mode | Startup (s) | Gen Time (s) | Throughput (tok/s) | vs Eager |",
        "|------|-------------|--------------|--------------------|----------|",
    ]
    for r in results:
        speedup = r["throughput"] / eager_throughput
        lines.append(
            f"| {r['label']} | {r['startup_s']:.1f} | "
            f"{r['gen_time_s']:.1f} | {r['throughput']:.0f} | "
            f"{speedup:.2f}x |"
        )

    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nResults saved to {path}")


def main():
    assert os.path.isdir(MODEL_PATH), (
        f"Model not found at {MODEL_PATH}. Download with:\n"
        f"  huggingface-cli download Qwen/Qwen3-14B-Base "
        f"--local-dir {MODEL_PATH} --local-dir-use-symlinks False"
    )

    results = []

    # Run 1: Eager baseline (no torch.compile, no CUDA graphs)
    results.append(run_benchmark(enforce_eager=True, label="Eager"))

    # Run 2: Compiled, cold start (clear Inductor cache first)
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    results.append(run_benchmark(enforce_eager=False, label="Compiled (cold)"))

    # Run 3: Compiled, warm start (cache populated from Run 2)
    results.append(run_benchmark(enforce_eager=False, label="Compiled (warm)"))

    write_results(results, "benchmark_results.md")


if __name__ == "__main__":
    main()
