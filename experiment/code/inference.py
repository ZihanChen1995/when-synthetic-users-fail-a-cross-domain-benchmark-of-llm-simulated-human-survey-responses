"""Inference clients + response parsing.

Backends:
  api    -> any hosted chat-completion API; implement APIClient.generate()
  vllm   -> local OpenAI-compatible server

Parsing turns raw model text into either a single answer code (style A) or a
probability distribution over answer codes (style C), plus a validity flag so we
can report invalid/refusal rates as a first-class stability metric.

Running inference is optional: every table and figure in the paper is computed
from the cached outputs under datasets/<DS>/build/llm_runs/.
"""
import json
import os
import re

import config as C


# ----------------------------------------------------------------- parsing
def parse_style_A(text: str, mapping: dict):
    """Extract a single answer key -> answer code. Returns (code|None, valid).

    Keys are either letters (A, B, ...) or scale numbers (1..10). For numeric keys
    we match the longest key first so "10" is not read as "1".
    """
    if not text:
        return None, False
    keys = list(mapping.keys())
    numeric = all(str(k).isdigit() for k in keys)
    if numeric:
        # find the first standalone integer that is a valid key (longest keys first)
        for tok in re.findall(r"\d+", text):
            if tok in mapping:
                return mapping[tok], True
        return None, False
    m = re.search(r"[A-Z]", text.strip().upper())
    if not m or m.group(0) not in mapping:
        return None, False
    return mapping[m.group(0)], True


def parse_style_C(text: str, mapping: dict):
    """Extract a JSON prob dist -> {answer_code: prob} normalized. Returns (dist|None, valid)."""
    if not text:
        return None, False
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None, False
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, False
    dist = {}
    for letter, code in mapping.items():
        try:
            dist[code] = float(raw.get(letter, 0.0))
        except (TypeError, ValueError):
            return None, False
    s = sum(dist.values())
    if s <= 0:
        return None, False
    return {k: v / s for k, v in dist.items()}, True


# ----------------------------------------------------------------- hosted API
class APIClient:
    """Client for a hosted chat-completion API. Implement `generate()` for the
    API you use; the rest of the pipeline only depends on this contract:

        generate(prompt: str, seed: int | None) -> str   # raw completion text
    """

    def __init__(self, model_id, max_tokens=100, temperature=1.0):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt: str, seed: int | None = None) -> str:
        raise NotImplementedError(
            "Implement APIClient.generate() for your API, or use --backend vllm. "
            "Not needed to reproduce the paper: all model outputs are cached."
        )


# ----------------------------------------------------------------- local vllm (OpenAI-compatible)
class VLLMClient:
    def __init__(self, model_id, base_url="http://localhost:8000/v1",
                 max_tokens=64, temperature=1.0):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get("LLM_API_KEY", "EMPTY"))
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt: str, seed: int | None = None) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=seed,
        )
        return resp.choices[0].message.content


PARSERS = {"A": parse_style_A, "C": parse_style_C}
