"""Optional OpenAI chat backend for teacher rollouts and zero-shot parses.

Used by ``generate_rollouts.py --backend openai`` and ``parse_split.py
--backend openai`` so the local Qwen pipeline and an API teacher (e.g. GPT-4o)
can be compared on the same prompts and the same downstream filtering.

Reproducibility notes, recorded in each output's ``.meta.json``:

* pin a dated snapshot (``gpt-4o-2024-08-06``), not an alias;
* ``seed`` makes sampling best-effort deterministic only; the per-response
  ``system_fingerprint`` is stored so backend changes are visible;
* the frozen rollout file, with its sha256, is the reproducible artifact.

The API key is read from ``OPENAI_API_KEY`` by the client.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

__all__ = ["DEFAULT_OPENAI_MODEL", "ChatResult", "OpenAIChat"]

DEFAULT_OPENAI_MODEL = "gpt-4o-2024-08-06"


@dataclass
class ChatResult:
    texts: list[str]
    system_fingerprint: str | None
    finish_reasons: list[str | None]
    usage: dict


class OpenAIChat:
    """Thin wrapper: one request returns ``n`` samples for one prompt."""

    def __init__(self, model: str = DEFAULT_OPENAI_MODEL, *, max_retries: int = 8,
                 timeout: float = 120.0, workers: int = 4):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise SystemExit("the openai package is not installed: "
                             "pip install -e '.[training]'") from e
        self.client = OpenAI(max_retries=max_retries, timeout=timeout)
        self.model = model
        self.workers = max(1, workers)

    def sample(self, system: str, user: str, *, n: int = 1, temperature: float = 0.0,
               max_tokens: int = 320, seed: int | None = None) -> ChatResult:
        kwargs = dict(model=self.model,
                      messages=[{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                      n=n, temperature=temperature, max_tokens=max_tokens)
        if seed is not None:
            kwargs["seed"] = seed
        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except Exception as e:  # client already retries 429/5xx; this is the last net
                if attempt == 4:
                    raise
                wait = 10 * (attempt + 1)
                print(f"  openai error ({type(e).__name__}: {e}); retry in {wait}s",
                      flush=True)
                time.sleep(wait)
        choices = sorted(resp.choices, key=lambda c: c.index)
        usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
        return ChatResult(texts=[c.message.content or "" for c in choices],
                          system_fingerprint=getattr(resp, "system_fingerprint", None),
                          finish_reasons=[c.finish_reason for c in choices],
                          usage=usage)

    def sample_many(self, system: str, users: list[str], seeds=None,
                    **kwargs) -> list[ChatResult]:
        """``sample`` for several prompts concurrently, results in input order."""
        seeds = seeds or [None] * len(users)
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(
                lambda us: self.sample(system, us[0], seed=us[1], **kwargs),
                zip(users, seeds)))
