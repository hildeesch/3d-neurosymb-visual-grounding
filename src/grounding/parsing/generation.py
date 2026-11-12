"""Chat prompts, model loading and batched generation for the local parsers.

PyTorch / Hugging Face ``transformers`` backend. It replaces the earlier mlx-lm
backend and keeps its call shape -- ``load_model`` returns ``(model, tokenizer,
sampler)`` and ``batch_generate(...).texts`` holds one completion per prompt --
so the experiment scripts changed only in their imports.

Decoding matches what the mlx-lm sampler did: ``temperature == 0`` is greedy,
otherwise pure temperature sampling with top-p, top-k and min-p disabled. The
model's own ``generation_config`` (Qwen3 ships top_p=0.95, top_k=20, T=0.6) is
overridden on purpose, so that ``--temp 0.7`` means the same thing it meant
under MLX. Outputs are still not token-identical to MLX runs: kernels,
quantisation and random streams differ.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence

__all__ = [
    "encode_chat",
    "template_prompt",
    "load_model",
    "batch_generate",
    "BatchResponse",
    "Sampler",
    "add_model_args",
    "batch_seed",
    "free_memory",
    "pick_device",
]


def encode_chat(tokenizer, system: str, user: str) -> list[int]:
    """The Qwen chat layout, encoded directly rather than through a template."""
    return tokenizer.encode(f"<|im_start|>system\n{system}<|im_end|>\n"
                            f"<|im_start|>user\n{user}<|im_end|>\n"
                            f"<|im_start|>assistant\n",
                            add_special_tokens=False)


def template_prompt(tokenizer, system: str, user: str, *, tokenize=True,
                    enable_thinking=False, fallback_suffix=" /no_think"):
    """The tokenizer's own chat template, with thinking off by default.

    Templates that predate ``enable_thinking`` reject it; they get
    ``fallback_suffix`` appended to the user turn instead, which is the same
    switch for Qwen3.
    """
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    try:
        out = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                            tokenize=tokenize,
                                            enable_thinking=enable_thinking)
    except TypeError:
        messages[-1]["content"] += fallback_suffix
        out = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                            tokenize=tokenize)
    return _as_ids(out) if tokenize else out


def _as_ids(out) -> list[int]:
    """apply_chat_template returns a list or a BatchEncoding depending on version."""
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


# ---------------------------------------------------------------------------
# Device, precision and loading

def pick_device(device: str | None = None) -> str:
    """``cuda`` when available, else ``mps`` (Apple), else ``cpu``."""
    import torch

    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_for(device: str, dtype: str | None):
    import torch

    if dtype and dtype != "auto":
        return {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float16": torch.float16, "fp16": torch.float16,
                "float32": torch.float32, "fp32": torch.float32}[dtype]
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


@dataclass(frozen=True)
class Sampler:
    """Decoding settings; the torch counterpart of mlx-lm's ``make_sampler``."""
    temperature: float = 0.0
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0

    def describe(self) -> dict:
        return {"sampler": "greedy" if self.greedy else "temperature",
                "temp": self.temperature, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
                "seed": self.seed}


def load_model(model: str, adapter: str | None = None, temperature: float = 0.0,
               *, seed: int | None = None, quantize: str = "none",
               device: str | None = None, dtype: str | None = None):
    """``(model, tokenizer, sampler)`` for a Hugging Face causal LM.

    ``adapter`` is a PEFT LoRA directory (as written by train_lora.py, or by
    scripts/prepare/convert_mlx_adapter.py from an old MLX adapter).
    ``quantize`` is ``none``, ``8bit`` or ``4bit`` (bitsandbytes, CUDA only).
    With CUDA, ``device_map="auto"`` lets a model too large for the GPU spill
    to CPU memory instead of failing -- slow, but it runs.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = pick_device(device)
    torch_dtype = _dtype_for(dev, dtype)
    kwargs: dict = {"dtype": torch_dtype, "low_cpu_mem_usage": True}

    if quantize not in ("none", None):
        if dev != "cuda":
            raise SystemExit(f"--quantize {quantize} needs a CUDA GPU "
                             "(bitsandbytes); use --quantize none on CPU")
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch_dtype,
                               bnb_4bit_use_double_quant=True)
            if quantize == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
    if dev == "cuda":
        kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(model)
    # Left padding: every prompt in a batch then ends where generation starts
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        loaded = AutoModelForCausalLM.from_pretrained(model, **kwargs)
    except TypeError:  # transformers < 4.56 calls it torch_dtype
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        loaded = AutoModelForCausalLM.from_pretrained(model, **kwargs)
    if "device_map" not in kwargs:
        loaded.to(dev)

    if adapter:
        from peft import PeftModel
        loaded = PeftModel.from_pretrained(loaded, adapter)
    loaded.eval()
    return loaded, tokenizer, Sampler(temperature=temperature, seed=seed)


def add_model_args(ap, *, quantize_default: str = "none") -> None:
    """The --device / --dtype / --quantize flags shared by the generators."""
    ap.add_argument("--device", default="auto",
                    help="auto | cuda | cpu | mps (default: auto)")
    ap.add_argument("--dtype", default="auto",
                    help="auto | bfloat16 | float16 | float32 (default: bf16 on "
                         "CUDA when supported, fp32 on CPU)")
    ap.add_argument("--quantize", default=quantize_default,
                    choices=["none", "8bit", "4bit"],
                    help="bitsandbytes weight quantisation, CUDA only "
                         f"(default: {quantize_default})")


def batch_seed(base: int | None, key) -> int | None:
    """A per-batch seed that depends only on the run seed and the batch key.

    Seeding per batch (keyed by e.g. the first row_idx) keeps a resumed run on
    the same random stream as an uninterrupted one, given the same batch size.
    """
    if base is None:
        return None
    digest = hashlib.sha256(f"{base}:{key}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def free_memory() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Generation

@dataclass
class BatchResponse:
    texts: list[str] = field(default_factory=list)


def _input_device(model):
    """Where input ids must go; with device_map="auto" that is the embedding's device."""
    try:
        return model.get_input_embeddings().weight.device
    except AttributeError:  # pragma: no cover - defensive
        return next(model.parameters()).device


def _stop_ids(model, tokenizer) -> set[int]:
    ids: set[int] = set()
    cfg = getattr(model, "generation_config", None)
    eos = getattr(cfg, "eos_token_id", None) if cfg is not None else None
    for value in (eos, tokenizer.eos_token_id):
        if value is None:
            continue
        ids.update(value if isinstance(value, (list, tuple)) else [value])
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end != tokenizer.unk_token_id:
        ids.add(im_end)
    return ids


def batch_generate(model, tokenizer, prompts: Sequence, max_tokens: int,
                   sampler: Sampler | None = None, verbose: bool = False,
                   seed: int | None = None) -> BatchResponse:
    """Generate one completion per prompt.

    ``prompts`` are token-id lists (as ``encode_chat``/``template_prompt``
    return) or strings, which are tokenized without adding special tokens.
    Completions stop at the first end-of-turn token and exclude it, as the
    mlx-lm texts did. ``seed`` overrides ``sampler.seed`` for this call.
    """
    import torch

    sampler = sampler or Sampler()
    ids = [tokenizer.encode(p, add_special_tokens=False) if isinstance(p, str)
           else list(p) for p in prompts]
    if not ids:
        return BatchResponse([])

    pad = tokenizer.pad_token_id
    width = max(len(x) for x in ids)
    input_ids = torch.full((len(ids), width), pad, dtype=torch.long)
    attention = torch.zeros((len(ids), width), dtype=torch.long)
    for i, x in enumerate(ids):
        input_ids[i, width - len(x):] = torch.tensor(x, dtype=torch.long)
        attention[i, width - len(x):] = 1
    device = _input_device(model)
    input_ids, attention = input_ids.to(device), attention.to(device)

    stop = _stop_ids(model, tokenizer)
    gen_kwargs = dict(max_new_tokens=max_tokens, pad_token_id=pad,
                      eos_token_id=sorted(stop) or None)
    if sampler.greedy:
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None,
                          top_k=None, min_p=None)
    else:
        # Pure temperature sampling, as mlx-lm's make_sampler(temp=T) did
        gen_kwargs.update(do_sample=True, temperature=sampler.temperature,
                          top_p=1.0, top_k=0, min_p=None)
        s = seed if seed is not None else sampler.seed
        if s is not None:
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)

    with torch.inference_mode():
        out = model.generate(input_ids=input_ids, attention_mask=attention,
                             **gen_kwargs)

    texts = []
    for row in out[:, width:].tolist():
        cut = len(row)
        for j, tok in enumerate(row):
            if tok in stop:
                cut = j
                break
        texts.append(tokenizer.decode(row[:cut], skip_special_tokens=False))
    if verbose:
        print(f"generated {len(texts)} completions")
    return BatchResponse(texts)
