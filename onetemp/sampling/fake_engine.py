#!/usr/bin/env python3
"""A deterministic stand-in for vLLM's LLM and a tokenizer, for the smoke test and dry runs (no GPU, no model).

`FakeLLM.generate` has the shape of vLLM's: it takes prompts ({"prompt_token_ids": [...]}) and one SamplingParams
or a list of them, and returns one result per prompt with `.outputs`, each output having `.index`, `.text`,
`.token_ids`, `.finish_reason` and `.stop_reason`. The texts are synthetic answers in the shape of each benchmark;
a sample is "right" with a probability that peaks at a temperature fixed per prompt, so a screen over a grid has
something to find. The token ids are the UTF-8 bytes of the text, and `FakeTokenizer.batch_decode` inverts that.
"""
import hashlib
import math
import random


class FakeSamplingParams:
    def __init__(self, n=1, temperature=1.0, max_tokens=16, **kw):
        self.n, self.temperature, self.max_tokens = int(n), float(temperature), int(max_tokens)
        self.min_p = float(kw.pop("min_p", 0.0))
        self.top_p, self.top_k, self.seed = 1.0, 0, None
        self.extra = kw

    def __repr__(self):
        return (f"FakeSamplingParams(n={self.n}, temperature={self.temperature}, min_p={self.min_p}, "
                f"max_tokens={self.max_tokens}, extra={self.extra})")


class _Out:
    def __init__(self, index, text, finish_reason, stop_reason, detokenize=True):
        self.index = index
        self.token_ids = list(text.encode("utf-8"))
        self.text = text if detokenize else ""
        self.finish_reason = finish_reason
        self.stop_reason = stop_reason


class _Result:
    def __init__(self, prompt_token_ids, outputs):
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs


class FakeTokenizer:
    def batch_decode(self, ids_list, skip_special_tokens=False):
        return [bytes(ids).decode("utf-8") for ids in ids_list]

    def decode(self, ids, skip_special_tokens=False):
        return bytes(ids).decode("utf-8")


def _answer(benchmark, right, rng):
    if benchmark == "math500":
        return "We compute step by step. \\boxed{" + ("7" if right else str(rng.randint(8, 99))) + "}"
    if benchmark == "hitom":
        return ("A. red_box" if right else "B. green_basket") + ". Because the agent exited first."
    return "n = int(input())\nprint(" + ("n" if right else "n + 1") + ")\n```"


class FakeLLM:
    def __init__(self, benchmark, seed=0, **engine_kwargs):
        self.benchmark = benchmark
        self.rng = random.Random(0 if seed is None else seed)
        self.engine_kwargs = engine_kwargs
        self.requests = 0

    def get_tokenizer(self):
        return FakeTokenizer()

    def generate(self, prompts, sampling_params, use_tqdm=False):
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        assert len(sampling_params) == len(prompts)
        results = []
        for p, sp in zip(prompts, sampling_params):
            ids = p["prompt_token_ids"]
            h = int(hashlib.sha256(str(ids).encode()).hexdigest()[:8], 16)
            peak_t = 0.2 + (h % 18) / 10.0                  # each prompt has its own best temperature
            peak_p = 0.05 + (h % 7) / 10.0
            x = sp.temperature if sp.min_p == 0 else sp.temperature * (1 - sp.min_p)
            prob = peak_p * math.exp(-((x - peak_t) ** 2) / 0.5)
            outs = []
            for i in range(sp.n):
                right = self.rng.random() < prob
                text = _answer(self.benchmark, right, self.rng)
                trunc = self.rng.random() < 0.02
                outs.append(_Out(i, text, "length" if trunc else "stop", None,
                                 detokenize=sp.extra.get("detokenize", True)))
            self.requests += 1
            results.append(_Result(ids, outs))
        return results
