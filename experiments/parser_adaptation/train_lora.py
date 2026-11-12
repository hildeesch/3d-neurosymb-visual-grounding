#!/usr/bin/env python3
"""LoRA supervised fine-tuning of the parser student with PyTorch + PEFT.

Replaces ``python -m mlx_lm lora`` and accepts the same flags and YAML config
(``lora_config_sft_v4_ext.yaml``), so train_sft_arms.py only swapped the
executable. The data directory holds ``train.jsonl`` and ``valid.jsonl`` with
one ``{"messages": [...]}`` chat per line, as build_uniform_sets.py writes.

What is kept from mlx-lm 0.31.3, so arms stay comparable with each other:

* tokenisation: ``tokenizer.apply_chat_template(messages)``, truncated to
  ``max_seq_length``; loss on every token (``mask_prompt: false``);
* batching: rows stably sorted by key count, cut into consecutive batches
  (remainder dropped), batch order a ``numpy`` permutation per epoch, and each
  evaluation draws one permutation of the validation batches and keeps the
  first ``--val-batches``. ``numpy`` is seeded with ``--seed`` and draws in the
  same order as mlx-lm, so eval_parser_assistant_loss.py can replay them;
* LoRA on every linear layer of the selected blocks, rank and ``scale`` from
  the YAML (PEFT ``lora_alpha = scale * rank``), constant learning rate, Adam;
* an evaluation at iteration 1, every ``--steps-per-eval`` and at the end,
  computed *before* that iteration's update; checkpoints every
  ``--save-every`` as ``<adapter-path>/NNNNNNN_adapters/``; the final adapter
  in ``<adapter-path>`` followed by the log line ``Saved final weights to``.

Known differences, so numbers are not identical to the MLX runs: PyTorch Adam
applies bias correction (mlx's Adam defaults to none); the logged loss covers
real target tokens only (mlx also scored one padding target per row); LoRA A
uses PEFT's Kaiming init; kernels and random streams differ.

Run settings are written to ``<adapter-path>/training_config.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

DEFAULTS = {
    "fine_tune_type": "lora",
    "num_layers": 16,
    "lora_parameters": {"rank": 8, "scale": 20.0, "dropout": 0.0},
    "learning_rate": 1e-5,
    "max_seq_length": 2048,
    "optimizer": "adam",
    "batch_size": 4,
    "iters": 1000,
    "seed": 0,
    "steps_per_report": 10,
    "steps_per_eval": 200,
    "save_every": 100,
    "val_batches": 25,
    "grad_checkpoint": False,
    "mask_prompt": False,
}

#: Sequence positions per lm_head chunk; bounds the float32 logits held at once
LOSS_CHUNK = 256


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", action="store_true",
                    help="accepted for mlx_lm compatibility; training always runs")
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--adapter-path", required=True, type=Path)
    ap.add_argument("-c", "--config", type=Path, default=None)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--iters", type=int)
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--max-seq-length", type=int)
    ap.add_argument("--num-layers", type=int)
    ap.add_argument("--save-every", type=int)
    ap.add_argument("--steps-per-eval", type=int)
    ap.add_argument("--steps-per-report", type=int)
    ap.add_argument("--val-batches", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--grad-checkpoint", action="store_true", default=None)
    ap.add_argument("--mask-prompt", action="store_true", default=None)
    ap.add_argument("--report-to", default=None, help="wandb, or omit")
    ap.add_argument("--project-name", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    return ap.parse_args(argv)


def resolve_config(args) -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))
    if args.config:
        import yaml
        loaded = yaml.safe_load(Path(args.config).read_text()) or {}
        lora = loaded.pop("lora_parameters", {}) or {}
        cfg.update(loaded)
        cfg["lora_parameters"].update(lora)
    for key in ("batch_size", "iters", "learning_rate", "max_seq_length", "num_layers",
                "save_every", "steps_per_eval", "steps_per_report", "val_batches",
                "seed", "grad_checkpoint", "mask_prompt"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if cfg["fine_tune_type"] != "lora":
        sys.exit(f"fine_tune_type {cfg['fine_tune_type']!r} is not supported; use lora")
    if str(cfg["optimizer"]).lower() != "adam":
        sys.exit(f"optimizer {cfg['optimizer']!r} is not supported; use adam")
    cfg["model"] = args.model
    cfg["data"] = (str(args.data.relative_to(REPO))
                   if args.data.resolve().is_relative_to(REPO) else str(args.data))
    return cfg


# ---------------------------------------------------------------------------
# Data, in mlx-lm's order

def load_split(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def chat_ids(tokenizer, messages: list[dict]) -> list[int]:
    out = tokenizer.apply_chat_template(messages, tokenize=True)
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
    return list(out)


def chat_ids_prompt(tokenizer, messages: list[dict]) -> list[int]:
    out = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                        tokenize=True)
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
    return list(out)


def encode_rows(rows: list[dict], tokenizer, max_seq_length: int,
                mask_prompt: bool) -> list[dict]:
    out, truncated = [], 0
    for row in rows:
        messages = row["messages"]
        ids = chat_ids(tokenizer, messages)
        # mask_prompt trains on the completion only, as mlx-lm's --mask-prompt
        offset = (len(chat_ids_prompt(tokenizer, messages[:-1]))
                  if mask_prompt else 0)
        if len(ids) > max_seq_length:
            truncated += 1
            ids = ids[:max_seq_length]
        out.append({"ids": ids, "offset": offset})
    if truncated:
        print(f"[WARN] {truncated} rows longer than max_seq_length "
              f"{max_seq_length} were truncated", flush=True)
    return out


def batch_groups(rows: list[dict], batch_size: int) -> list[list[int]]:
    """mlx-lm's grouping: stable sort by key count, consecutive full batches."""
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
    if len(order) < batch_size:
        sys.exit(f"dataset has {len(order)} rows, fewer than batch size {batch_size}")
    return [order[i:i + batch_size]
            for i in range(0, len(order) - batch_size + 1, batch_size)]


def train_batches(groups: list[list[int]]):
    """Endless iterator; draws one numpy permutation per epoch, lazily."""
    while True:
        for j in np.random.permutation(len(groups)):
            yield groups[j]


def collate(encoded: list[dict], group: list[int], pad_id: int, device):
    import torch

    rows = [encoded[i] for i in group]
    width = max(len(r["ids"]) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attn = torch.zeros((len(rows), width), dtype=torch.long)
    # Target mask over positions 1..len-1 (after an optional prompt offset)
    tmask = torch.zeros((len(rows), width), dtype=torch.float32)
    for j, r in enumerate(rows):
        n = len(r["ids"])
        ids[j, :n] = torch.tensor(r["ids"])
        attn[j, :n] = 1
        tmask[j, max(1, r["offset"]):n] = 1.0
    return ids.to(device), attn.to(device), tmask.to(device)


# ---------------------------------------------------------------------------
# Model and loss

def linear_leaf_names(model) -> list[str]:
    import torch.nn as nn

    names = set()
    for name, module in model.named_modules():
        if ".layers." in f".{name}." and isinstance(module, nn.Linear):
            names.add(name.rsplit(".", 1)[-1])
    return sorted(names)


def build_model(cfg: dict, device: str, dtype: str):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from grounding.parsing.generation import _dtype_for

    torch_dtype = _dtype_for(device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch_dtype)
    except TypeError:  # transformers < 4.56
        model = AutoModelForCausalLM.from_pretrained(cfg["model"],
                                                     torch_dtype=torch_dtype)
    model.to(device)
    model.config.use_cache = False

    n_layers = model.config.num_hidden_layers
    k = cfg["num_layers"]
    layers = list(range(n_layers)) if k in (-1, None) or k >= n_layers \
        else list(range(n_layers - k, n_layers))
    lora = cfg["lora_parameters"]
    rank = int(lora["rank"])
    targets = lora.get("keys") or linear_leaf_names(model)
    if lora.get("keys"):
        targets = sorted({t.rsplit(".", 1)[-1] for t in targets})
    peft_cfg = LoraConfig(r=rank, lora_alpha=float(lora["scale"]) * rank,
                          lora_dropout=float(lora.get("dropout", 0.0)),
                          target_modules=targets, layers_to_transform=layers,
                          bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_cfg)
    if cfg["grad_checkpoint"]:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model, tokenizer, targets, layers


def token_ce_sum(model, ids, attn, tmask):
    """Summed cross-entropy over masked targets, lm_head applied in chunks.

    Avoids materialising (batch, seq, vocab) float32 logits; each chunk is
    checkpointed, so its logits are recomputed in the backward pass.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    base = model.get_base_model()
    hidden = base.model(input_ids=ids, attention_mask=attn).last_hidden_state
    head = base.get_output_embeddings()
    h, targets, mask = hidden[:, :-1], ids[:, 1:], tmask[:, 1:]

    def chunk_loss(hc, tc, mc):
        logits = head(hc).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tc.reshape(-1),
                             reduction="none")
        return (ce * mc.reshape(-1)).sum()

    total = torch.zeros((), device=ids.device, dtype=torch.float32)
    for s in range(0, h.shape[1], LOSS_CHUNK):
        args = (h[:, s:s + LOSS_CHUNK], targets[:, s:s + LOSS_CHUNK],
                mask[:, s:s + LOSS_CHUNK])
        if torch.is_grad_enabled():
            total = total + checkpoint(chunk_loss, *args, use_reentrant=False)
        else:
            total = total + chunk_loss(*args)
    return total, mask.sum()


def evaluate(model, encoded, groups, val_batches, pad_id, device) -> float:
    import torch

    drawn = np.random.permutation(len(groups))[:val_batches]
    model.eval()
    total, count = 0.0, 0.0
    with torch.no_grad():
        for j in drawn:
            ids, attn, tmask = collate(encoded, groups[j], pad_id, device)
            loss, n = token_ce_sum(model, ids, attn, tmask)
            total += loss.item()
            count += n.item()
    model.train()
    return total / max(count, 1.0)


def save_adapter(model, path: Path, cfg: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    (path / "training_config.json").write_text(json.dumps(cfg, indent=2) + "\n")


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    import torch

    from grounding.parsing.generation import pick_device

    args = parse_args(argv)
    cfg = resolve_config(args)
    device = pick_device(args.device)

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    print(f"Loading {cfg['model']} on {device} ...", flush=True)
    model, tokenizer, targets, layers = build_model(cfg, device, args.dtype)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA r={cfg['lora_parameters']['rank']} scale="
          f"{cfg['lora_parameters']['scale']} on {targets} in {len(layers)} layers; "
          f"trainable {trainable / 1e6:.3f}M / {total / 1e6:.1f}M", flush=True)

    train_rows = load_split(args.data / "train.jsonl")
    valid_path = args.data / "valid.jsonl"
    valid_rows = load_split(valid_path) if valid_path.exists() else []
    enc_train = encode_rows(train_rows, tokenizer, cfg["max_seq_length"], cfg["mask_prompt"])
    enc_valid = encode_rows(valid_rows, tokenizer, cfg["max_seq_length"], cfg["mask_prompt"])
    train_groups = batch_groups(train_rows, cfg["batch_size"])
    valid_groups = batch_groups(valid_rows, cfg["batch_size"]) if valid_rows else []
    print(f"train {len(train_rows)} rows / {len(train_groups)} batches, "
          f"valid {len(valid_rows)} rows", flush=True)

    cfg.update({"trainer": "experiments/parser_adaptation/train_lora.py (torch+peft)",
                "lora_targets": targets, "lora_layers": layers,
                "adapter_path": str(args.adapter_path), "device": device})
    args.adapter_path.mkdir(parents=True, exist_ok=True)
    (args.adapter_path / "training_config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    run = None
    if args.report_to == "wandb":
        import wandb
        run = wandb.init(project=args.project_name, name=args.adapter_path.name,
                         config=cfg)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg["learning_rate"])
    pad_id = tokenizer.pad_token_id
    batches = train_batches(train_groups)
    iters = cfg["iters"]

    model.train()
    losses, ntoks, trained_tokens = 0.0, 0.0, 0
    t_report = time.perf_counter()
    for it in range(1, iters + 1):
        # The batch is fetched before evaluating, as mlx-lm's zip() did
        group = next(batches)
        if valid_groups and (it == 1 or it % cfg["steps_per_eval"] == 0 or it == iters):
            t0 = time.perf_counter()
            val = evaluate(model, enc_valid, valid_groups, cfg["val_batches"], pad_id, device)
            print(f"Iter {it}: Val loss {val:.3f}, Val took "
                  f"{time.perf_counter() - t0:.3f}s", flush=True)
            if run:
                run.log({"val_loss": val}, step=it)
            t_report = time.perf_counter()

        ids, attn, tmask = collate(enc_train, group, pad_id, device)
        loss_sum, n = token_ce_sum(model, ids, attn, tmask)
        loss = loss_sum / n.clamp(min=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        losses += loss_sum.item()
        ntoks += n.item()
        trained_tokens += int(attn.sum().item())

        if it % cfg["steps_per_report"] == 0 or it == iters:
            elapsed = time.perf_counter() - t_report
            train_loss = losses / max(ntoks, 1.0)
            peak = (torch.cuda.max_memory_allocated() / 1e9
                    if device == "cuda" else float("nan"))
            print(f"Iter {it}: Train loss {train_loss:.3f}, Learning Rate "
                  f"{cfg['learning_rate']:.3e}, It/sec "
                  f"{cfg['steps_per_report'] / max(elapsed, 1e-9):.3f}, Tokens/sec "
                  f"{ntoks / max(elapsed, 1e-9):.3f}, Trained Tokens {trained_tokens}, "
                  f"Peak mem {peak:.3f} GB", flush=True)
            if run:
                run.log({"train_loss": train_loss}, step=it)
            losses, ntoks = 0.0, 0.0
            t_report = time.perf_counter()

        if it % cfg["save_every"] == 0:
            ckpt = args.adapter_path / f"{it:07d}_adapters"
            save_adapter(model, ckpt, {**cfg, "iteration": it})
            save_adapter(model, args.adapter_path, {**cfg, "iteration": it})
            print(f"Iter {it}: Saved adapter weights to {args.adapter_path} and {ckpt}.",
                  flush=True)

    save_adapter(model, args.adapter_path, {**cfg, "iteration": iters})
    print(f"Saved final weights to {args.adapter_path}.", flush=True)
    if run:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
