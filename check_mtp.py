"""Validate Qwen3.5 MTP speculative decoding.

Greedy-equivalent sampling (temperature ~ 0): speculative decoding with rejection
sampling is output-exact w.r.t. the target model, so spec on/off should produce
(nearly) identical token sequences — small fp differences between the 1-token
decode path and the multi-token verify path can occasionally flip near-tied tokens.
"""
import gc
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from nanovllm import LLM, SamplingParams


PATH = "/workspace/models/Qwen3.5-0.8B"
PROMPTS = [
    "介绍一下你自己",
    "1 + 1 = ",
    "The capital of France is",
]


def run(spec_decode: bool, num_draft: int = 1, max_tokens: int = 64, temperature: float = 1e-6):
    llm = LLM(PATH, spec_decode=spec_decode, spec_num_draft_tokens=num_draft,
              max_model_len=2048, max_num_seqs=4)
    sp = [SamplingParams(temperature=temperature, max_tokens=max_tokens)] * len(PROMPTS)
    llm.generate(PROMPTS[:1], sp[:1], use_tqdm=False)  # warmup
    if spec_decode:
        llm.model_runner.reset_spec_stats()  # keep warmup steps out of the stats
    t0 = time.time()
    outs = llm.generate(PROMPTS, sp, use_tqdm=False)
    dt = time.time() - t0
    num_tokens = sum(len(o["token_ids"]) for o in outs)
    stats = llm.model_runner.format_spec_stats() if spec_decode else ""
    del llm
    gc.collect()
    return [o["token_ids"] for o in outs], num_tokens / dt, stats


def compare(plain, spec):
    total = matched = 0
    for a, b in zip(plain, spec):
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        matched += n
        total += max(len(a), len(b))
    return matched, total


def main():
    plain, t_plain, _ = run(spec_decode=False)
    print(f"plain: {t_plain:.1f} tok/s")
    for k in (3, 8):
        spec, t_spec, stats = run(spec_decode=True, num_draft=k)
        matched, total = compare(plain, spec)
        print(f"\nk={k}: sequence match: {matched}/{total} tokens identical | "
              f"spec: {t_spec:.1f} tok/s | speedup: {t_spec / t_plain:.2f}x")
        print(stats)
        for a, b in zip(plain, spec):
            print("plain:", repr(a[:16]))
            print("spec :", repr(b[:16]))


if __name__ == "__main__":
    main()
