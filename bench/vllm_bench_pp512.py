"""
vLLM pp512 benchmark for nsys profiling.
Must be run with 'if __name__ == "__main__"' guard for vLLM's multiprocessing.
"""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    import time
    import statistics
    from vllm import LLM, SamplingParams

    MODEL = "/models/Qwen3-Coder-30B-A3B-Instruct-FP4"
    QUANT = "modelopt_fp4"
    MAX_MODEL_LEN = 2048
    WARMUP_REPS = 2
    MEASURED_REPS = 3

    # Build a ~512 token prompt using simple repeated tokens
    PROMPT = "x " * 512

    print(f"Loading model {MODEL} with quantization={QUANT}", flush=True)
    llm = LLM(
        model=MODEL,
        quantization=QUANT,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1,
        gpu_memory_utilization=0.90,
    )

    sampling = SamplingParams(temperature=0.0, max_tokens=4)

    print(f"Warming up ({WARMUP_REPS} reps)...", flush=True)
    for i in range(WARMUP_REPS):
        outputs = llm.generate([PROMPT], sampling)

    print(f"Measuring ({MEASURED_REPS} reps)...", flush=True)
    times = []
    for i in range(MEASURED_REPS):
        t0 = time.perf_counter()
        outputs = llm.generate([PROMPT], sampling)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        n_prompt = len(outputs[0].prompt_token_ids or [])
        pp_toks = n_prompt / (elapsed_ms / 1000.0)
        times.append(elapsed_ms)
        print(f"  rep {i}: {elapsed_ms:.1f} ms, {n_prompt} prompt tokens, {pp_toks:.0f} pp tok/s", flush=True)

    mean_ms = statistics.mean(times)
    n_prompt = len(outputs[0].prompt_token_ids or [])
    mean_pp = n_prompt / (mean_ms / 1000.0)
    print(f"Summary: mean={mean_ms:.1f} ms, mean pp_tok/s={mean_pp:.0f}", flush=True)
