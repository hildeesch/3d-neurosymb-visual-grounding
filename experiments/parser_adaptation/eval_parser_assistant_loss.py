#!/usr/bin/env python3
"""Recompute parser validation loss over chosen token spans of saved adapters.

Forward passes only (PyTorch + PEFT): no optimizer, no weight update, nothing
written except the optional --out file, which must not already exist. Token ids
and batching follow train_lora.py (which mirrors mlx_lm 0.31.3's
`ChatDataset.process` and `iterate_batches`). Three spans are reported, each a
token-weighted mean cross-entropy over the tokens it contains:

  full       every real token after the first; what train_lora.py logs as
             "Val loss" (mlx_lm's logged loss also scored one padding target
             per row, so old MLX logs sit slightly off this number)
  assistant  from the end of the `add_generation_prompt=True` prefix to the end
             (what `--mask-prompt` would train on: includes the empty think
             block the Qwen3 template writes into the final assistant turn)
  program    the JSON program and its closing <|im_end|>; the prefix is rendered
             with `enable_thinking=False`, as parse_split.py prompts at inference

Span offsets are lengths of the templated prefix and are checked to be a token
prefix of the templated conversation; a row where they are not stops the run.

--rows all scores every validation row. --rows logged scores the 25 batches
train_lora.py drew for the "Val loss" line at --logged-iter, rebuilt from the
run's seed. That line is computed before that iteration's update, so a saved
NNNNNNN_adapters/ checkpoint is one update later than the logged value.
Omitting --adapter scores the base model, which at iteration 1 is exactly
the model that produced the first logged value (LoRA B starts at zero).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from grounding.resolution.artifacts import sha256

REPO = Path(__file__).resolve().parents[2]
SPANS = ("full", "assistant", "program")
#: Sequence positions per lm_head chunk when scoring
LOSS_CHUNK = 256


def load_run_config(args: argparse.Namespace) -> dict:
    """Take run settings from the adapter's training_config.json.

    train_lora.py writes it; scripts/prepare/convert_mlx_adapter.py writes one
    for converted MLX adapters from their old adapter_config.json.
    """
    if args.adapter is None:
        if args.data is None:
            sys.exit("--data is required when no --adapter is given")
        return {"model": args.model, "data": str(args.data), "seed": args.seed,
                "batch_size": 4, "max_seq_length": 2048, "val_batches": 25,
                "steps_per_eval": 100, "iters": 1000, "mask_prompt": False}
    path = args.adapter / "training_config.json"
    if not path.exists():
        sys.exit(f"{path} is missing; was {args.adapter} written by train_lora.py "
                 "or convert_mlx_adapter.py?")
    config = json.loads(path.read_text())
    if args.data is not None:
        config["data"] = str(args.data)
    if args.seed is not None:
        config["seed"] = args.seed
    return config


def logged_iterations(steps_per_eval: int, iters: int) -> list[int]:
    """Iterations at which train_lora.py (like mlx_lm) runs evaluate()."""
    return [it for it in range(1, iters + 1)
            if it == 1 or it % steps_per_eval == 0 or it == iters]


def logged_batches(config: dict, n_valid: int, iteration: int) -> list[int]:
    """Replay the global numpy draws of train_lora.py up to that evaluation.

    numpy is seeded with the run seed; the training iterator draws one
    permutation of its batches whenever an epoch starts (fetched before that
    iteration's evaluation), and every evaluation draws one permutation of the
    validation batches and keeps the first val_batches.
    """
    batch_size = config["batch_size"]
    train_path = REPO / config["data"] / "train.jsonl"
    with train_path.open() as f:
        n_train = sum(1 for _ in f)
    n_train_batches = n_train // batch_size
    iterations = logged_iterations(config["steps_per_eval"], config["iters"])
    if iteration not in iterations:
        sys.exit(f"--logged-iter {iteration} is not an evaluated iteration: "
                 f"{iterations}")
    np.random.seed(config["seed"])
    evaluated = set(iterations)
    drawn = None
    for it in range(1, iteration + 1):
        if (it - 1) % n_train_batches == 0:
            np.random.permutation(n_train_batches)
        if it in evaluated:
            drawn = np.random.permutation(n_valid // batch_size)
    return drawn[:config["val_batches"]].tolist()


def _ids(out) -> list[int]:
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
    return list(out)


def encode(rows: list[dict], tokenizer) -> list[dict]:
    encoded = []
    for index, row in enumerate(rows):
        messages = row["messages"]
        if messages[-1]["role"] != "assistant":
            sys.exit(f"valid row {index} does not end in an assistant turn")
        tokens = _ids(tokenizer.apply_chat_template(messages, tokenize=True))
        assistant = _ids(tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=True, tokenize=True))
        program = _ids(tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=True, enable_thinking=False,
            tokenize=True))
        for name, prefix in (("assistant", assistant), ("program", program)):
            if tokens[:len(prefix)] != prefix:
                sys.exit(f"valid row {index}: the {name} prefix is not a token "
                         "prefix of the templated conversation")
        end = len(tokens) - tokens[::-1].index(tokenizer.eos_token_id)
        if end <= len(program):
            sys.exit(f"valid row {index}: no {tokenizer.eos_token!r} after the program")
        encoded.append({"tokens": tokens, "spans": {
            "full": (1, len(tokens)),
            "assistant": (len(assistant), len(tokens)),
            "program": (len(program), end),
        }})
    return encoded


def batches(n_rows: int, batch_size: int, mode: str, drawn: list[int] | None):
    # Batches group rows by raw JSON key count, not token count; equal counts keep file order
    order = list(range(n_rows))
    groups = [order[i:i + batch_size]
              for i in range(0, n_rows - batch_size + 1, batch_size)]
    if mode == "logged":
        return [groups[i] for i in drawn]
    covered = batch_size * len(groups)
    return groups + ([order[covered:]] if covered < n_rows else [])


def score(model, encoded: list[dict], groups, max_seq_length: int,
          pad_id: int) -> dict:
    import torch
    import torch.nn.functional as F

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    head = base.get_output_embeddings()
    device = head.weight.device
    totals = {span: 0.0 for span in SPANS}
    counts = {span: 0 for span in SPANS}
    for group in groups:
        rows = [encoded[i] for i in group]
        longest = max(len(row["tokens"]) for row in rows)
        if longest > max_seq_length:
            sys.exit(f"a row has {longest} tokens > max_seq_length "
                     f"{max_seq_length}; truncation is not replayed")
        batch = torch.full((len(rows), longest), pad_id, dtype=torch.long)
        attn = torch.zeros((len(rows), longest), dtype=torch.long)
        for j, row in enumerate(rows):
            batch[j, :len(row["tokens"])] = torch.tensor(row["tokens"])
            attn[j, :len(row["tokens"])] = 1
        batch, attn = batch.to(device), attn.to(device)
        with torch.no_grad():
            hidden = base.model(input_ids=batch, attention_mask=attn).last_hidden_state
            # Position t predicts token t + 1; ce[:, t] scores token t + 1, in float32
            parts = []
            for s0 in range(0, longest - 1, LOSS_CHUNK):
                logits = head(hidden[:, s0:min(s0 + LOSS_CHUNK, longest - 1)]).float()
                tgt = batch[:, s0 + 1:s0 + 1 + logits.shape[1]]
                parts.append(F.cross_entropy(logits.transpose(1, 2), tgt,
                                             reduction="none"))
            ce = torch.cat(parts, dim=1)
        positions = torch.arange(ce.shape[1], device=device)
        for span in SPANS:
            # Tokens [start, end) are scored at positions [start - 1, end - 1)
            starts = torch.tensor([row["spans"][span][0] - 1 for row in rows],
                                  device=device)[:, None]
            ends = torch.tensor([row["spans"][span][1] - 1 for row in rows],
                                device=device)[:, None]
            mask = (positions >= starts) & (positions < ends)
            totals[span] += (ce * mask).sum().item()
            counts[span] += sum(row["spans"][span][1] - row["spans"][span][0]
                                for row in rows)
    return {span: {"loss": totals[span] / counts[span],
                   "perplexity": math.exp(totals[span] / counts[span]),
                   "tokens": counts[span]} for span in SPANS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path,
                    help="adapter directory (reads its adapter_config.json); "
                         "omit to score the base model")
    ap.add_argument("--weights", type=Path,
                    help="checkpoint directory to score with --adapter's config, "
                         "e.g. <adapter>/0000400_adapters (default: the final "
                         "adapter in --adapter)")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B",
                    help="base model when no --adapter is given")
    ap.add_argument("--data", type=Path,
                    help="training-set directory holding valid.jsonl "
                         "(default: the adapter's recorded data path)")
    ap.add_argument("--seed", type=int,
                    help="run seed for --rows logged (default: the adapter's)")
    ap.add_argument("--rows", choices=("all", "logged"), default="all")
    ap.add_argument("--logged-iter", type=int, default=1000)
    ap.add_argument("--out", type=Path, help="write the report here; must not exist")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()

    if args.out is not None and args.out.exists():
        sys.exit(f"{args.out} exists; not overwriting")
    config = load_run_config(args)
    if config.get("mask_prompt"):
        print("note: this run trained with mask_prompt=true", file=sys.stderr)
    if args.rows == "logged" and config["seed"] is None:
        sys.exit("--rows logged needs the run seed (--seed)")

    valid_path = REPO / config["data"] / "valid.jsonl"
    rows = [json.loads(line) for line in valid_path.open() if line.strip()]
    if len({len(row) for row in rows}) != 1:
        sys.exit("valid rows have differing key counts; the trainer's batch "
                 "order is not file order and is not replayed here")

    from grounding.parsing.generation import load_model

    weights = None
    if args.adapter is not None:
        weights = args.weights or args.adapter
        if not (weights / "adapter_config.json").exists():
            sys.exit(f"{weights} is not a PEFT adapter directory")
    model, tokenizer, _ = load_model(config["model"],
                                     None if weights is None else str(weights),
                                     device=args.device, dtype=args.dtype)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = encode(rows, tokenizer)
    drawn = (logged_batches(config, len(rows), args.logged_iter)
             if args.rows == "logged" else None)
    groups = batches(len(rows), config["batch_size"], args.rows, drawn)
    metrics = score(model, encoded, groups, config["max_seq_length"],
                    tokenizer.pad_token_id)

    report = {
        "model": config["model"],
        "adapter": None if args.adapter is None else str(args.adapter),
        "weights": None if weights is None else str(weights),
        "weights_sha256": None if weights is None else sha256(
            next(p for p in (weights / "adapter_model.safetensors",
                             weights / "adapter_model.bin") if p.exists())),
        "valid": str(valid_path.relative_to(REPO)
                     if valid_path.is_relative_to(REPO) else valid_path),
        "rows": args.rows,
        "logged_iter": args.logged_iter if args.rows == "logged" else None,
        "examples": sum(len(group) for group in groups),
        "metrics": metrics,
    }
    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
