"""Client for a local llama.cpp ``llama-server`` (GGUF models).

Lets a GGUF build of a teacher (Qwen3-30B-A3B or Qwen3-Coder-30B-A3B-Instruct
at Q4_K_M) run on a GPU too small to hold it: llama-server keeps the MoE
expert weights of some layers in system RAM (``--n-cpu-moe``) and the rest on
the GPU. This is the PC counterpart of how the original MLX run fitted on an
Apple Silicon laptop.

The prompt is rendered here, with the served model's own Hugging Face chat
template (thinking off where the model has a thinking mode), exactly as the
PyTorch and former MLX paths rendered it, and sent as raw text to
``/completion``. The server's own chat template is not used, so all three
backends condition the model on the same string.

Sampling is pure temperature sampling, as in the other backends: top-k, top-p,
min-p and every penalty llama.cpp enables by default are switched off per
request. Every sample is its own request with its own seed. Only the Python
standard library is needed.

Start the server first, e.g.::

    llama-server -m models/Qwen3-30B-A3B-Q4_K_M.gguf -ngl 99 --n-cpu-moe 20 \\
        -c 16384 -np 8 --port 8080
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["DEFAULT_SERVER_URL", "LlamaCppServer", "Completion", "add_llamacpp_args",
           "resolve_template_tokenizer"]

DEFAULT_SERVER_URL = "http://127.0.0.1:8080"

#: Per-request sampler settings that make llama.cpp sample from temperature only
NEUTRAL_SAMPLING = {
    "top_k": 0, "top_p": 1.0, "min_p": 0.0, "typical_p": 1.0,
    "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
    "xtc_probability": 0.0, "dry_multiplier": 0.0, "mirostat": 0,
}

STOP = ["<|im_end|>", "<|endoftext|>"]


@dataclass
class Completion:
    text: str
    tokens: int = 0
    stop_type: str | None = None
    extra: dict = field(default_factory=dict)


class LlamaCppServer:
    """Thin client for ``llama-server``'s native ``/completion`` endpoint."""

    def __init__(self, url: str = DEFAULT_SERVER_URL, *, workers: int = 8,
                 timeout: float = 600.0, retries: int = 3):
        self.url = url.rstrip("/")
        self.workers = max(1, workers)
        self.timeout = timeout
        self.retries = retries

    # -- plumbing ---------------------------------------------------------
    def _request(self, path: str, body: dict | None = None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            self.url + path, data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if body is None else "POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def wait_ready(self, wait_s: float = 600.0) -> dict:
        """Block until the model is loaded; return ``/props`` (model path etc.)."""
        start = time.time()
        last = None
        while time.time() - start < wait_s:
            try:
                self._request("/health")
                return self._request("/props")
            except urllib.error.HTTPError as e:   # 503 while the model loads
                last = e
                print("  llama-server is loading the model ...", flush=True)
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                last = e
                # Nothing listening: give a just-started server a minute, no more
                if time.time() - start > 60:
                    break
            time.sleep(5)
        raise SystemExit(
            f"llama-server is not reachable at {self.url} ({last}).\n"
            "Start it first, e.g.\n  llama-server -m models/Qwen3-30B-A3B-Q4_K_M.gguf "
            "-ngl 99 --n-cpu-moe 20 -c 16384 -np 8 --port 8080\n"
            "and pass --server-url if it listens elsewhere.")

    @staticmethod
    def model_name(props: dict) -> str:
        path = props.get("model_path") or props.get("model_alias") or "unknown.gguf"
        return Path(str(path)).name

    # -- generation -------------------------------------------------------
    def complete(self, prompt: str, *, max_tokens: int, temperature: float,
                 seed: int | None) -> Completion:
        body = {"prompt": prompt, "n_predict": max_tokens, "temperature": temperature,
                "stop": STOP, "cache_prompt": True, "stream": False,
                **NEUTRAL_SAMPLING}
        if seed is not None:
            body["seed"] = int(seed)
        for attempt in range(self.retries + 1):
            try:
                out = self._request("/completion", body)
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                if attempt == self.retries:
                    raise
                print(f"  llama-server error ({e}); retrying", flush=True)
                time.sleep(5 * (attempt + 1))
        return Completion(text=out.get("content", ""),
                          tokens=int(out.get("tokens_predicted", 0) or 0),
                          stop_type=out.get("stop_type"))

    def complete_many(self, jobs: list[tuple[str, int | None]], *, max_tokens: int,
                      temperature: float) -> list[Completion]:
        """``(prompt, seed)`` jobs, run concurrently; results in input order.

        Concurrency should match the server's ``-np`` slots: more requests
        than slots just queue on the server.
        """
        with ThreadPoolExecutor(self.workers) as pool:
            futures = [pool.submit(self.complete, p, max_tokens=max_tokens,
                                   temperature=temperature, seed=s) for p, s in jobs]
            return [f.result() for f in futures]


def add_llamacpp_args(ap) -> None:
    ap.add_argument("--server-url", default=DEFAULT_SERVER_URL,
                    help="llamacpp: llama-server address (default: %(default)s)")
    ap.add_argument("--tokenizer", default=None,
                    help="llamacpp: Hugging Face tokenizer whose chat template "
                         "renders the prompt (only tokenizer files are "
                         "downloaded). Default: the --teacher preset's, else the "
                         "one matching the served GGUF (Qwen3-30B-A3B or "
                         "Qwen3-Coder-30B-A3B-Instruct); a mismatch is refused")


def resolve_template_tokenizer(args, gguf: str | None, teacher: str | None = None) -> str:
    """Fill ``args.tokenizer`` for the served GGUF; exit on a template mismatch."""
    from grounding.parsing.models import template_tokenizer

    try:
        args.tokenizer = template_tokenizer(args.tokenizer, teacher=teacher, gguf=gguf)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    return args.tokenizer


def load_template_tokenizer(name: str):
    """The tokenizer used only to render chat prompts as text."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)
