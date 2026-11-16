#!/usr/bin/env python3
"""Generate parse rollouts for Nr3D training utterances.

Three teacher backends write the same row format, so filter_rollouts.py and
the training-set builders consume any of them unchanged:

  hf        a local Hugging Face model through PyTorch (default: Qwen3-30B-A3B)
  llamacpp  a GGUF model served by a running llama.cpp ``llama-server``; fits
            Qwen3-30B-A3B on a 16 GB GPU by keeping some experts in system RAM
  openai    an OpenAI chat model (default: gpt-4o-2024-08-06), n samples per call

Every output gets ``<output>.meta.json`` recording backend, model, prompt hash,
decoding and seed. Resuming into a file written under different settings is
refused, so two teachers never end up in one pool.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from grounding.parsing import PROMPTS, extract_json_objects
from grounding.parsing.generation import (add_model_args, batch_generate,
                                          batch_seed, load_model,
                                          template_prompt)
from grounding.parsing.llamacpp_backend import add_llamacpp_args
from grounding.parsing.openai_backend import DEFAULT_OPENAI_MODEL

DEFAULT_HF_MODEL = "Qwen/Qwen3-30B-A3B"


def load_rows(path: Path, limit: int) -> list[dict]:
    """Load and deterministically shuffle Nr3D rows."""
    rows = []

    with path.open(newline="") as f:
        for row_idx, row in enumerate(csv.DictReader(f)):
            utterance = (row.get("utterance") or "").strip()
            if not utterance:
                continue

            rows.append({
                "row_idx": row_idx,
                "stimulus_id": row.get("stimulus_id", ""),
                "scan_id": row.get("scan_id", ""),
                "target_id": row.get("target_id", ""),
                "instance_type": row.get("instance_type", ""),
                "utterance": utterance,
            })

    random.Random(42).shuffle(rows)

    return rows[:limit] if limit > 0 else rows


def completed_rows(path: Path) -> set[int]:
    """Return row indices already present in a rollout file."""
    if not path.exists():
        return set()

    done = set()

    with path.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)["row_idx"])
            except (json.JSONDecodeError, KeyError):
                continue

    return done


def user_message(utterance: str) -> str:
    return f"Sentence: {utterance}\n\nParsed JSON:"


def invocation(args, system: str) -> dict:
    """Settings that decide what a rollout looks like; compared on resume."""
    meta = {
        "producer": "experiments/parser_adaptation/generate_rollouts.py",
        "command": " ".join(sys.argv),
        "input": str(args.input),
        "backend": args.backend,
        "model": args.model,
        "prompt": args.prompt,
        "prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "decoding": {"n_samples": args.n_samples, "temp": args.temp,
                     "top_p": 1.0, "top_k": 0, "max_tokens": args.max_tokens,
                     "seed": args.seed, "thinking": False},
    }
    if args.backend == "hf":
        meta["decoding"].update(batch_size=args.batch_size, quantize=args.quantize,
                                dtype=args.dtype)
    if args.backend == "llamacpp":
        meta["decoding"].update(template_tokenizer=args.tokenizer,
                                min_p=0.0, penalties="off")
        # Recorded, not compared on resume: where and how the GGUF was served
        meta["server"] = getattr(args, "server_info", {})
    return meta


def check_resume(meta_path: Path, meta: dict, have_rows: bool) -> None:
    keys = ("backend", "model", "prompt", "prompt_sha256", "decoding")
    if not have_rows:
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        return
    if not meta_path.exists():
        raise SystemExit(
            f"{meta_path.with_suffix('').name} has rows but no {meta_path.name}; "
            "it may come from another teacher. Pass a different --output.")
    recorded = json.loads(meta_path.read_text())
    drift = {k: (recorded.get(k), meta[k]) for k in keys if recorded.get(k) != meta[k]}
    if drift:
        raise SystemExit("output was written under different settings; refusing "
                         f"to mix rollouts: {json.dumps(drift, indent=1)}")


def row_record(row: dict, raw_samples: list[str], extra: dict | None = None) -> dict:
    samples = [{"raw": raw, "parsed": extract_json_objects(raw)} for raw in raw_samples]
    rec = {k: row[k] for k in ("row_idx", "stimulus_id", "scan_id", "target_id",
                               "instance_type", "utterance")}
    rec["samples"] = samples
    if extra:
        rec.update(extra)
    return rec


def load_hf(args):
    print(f"Loading {args.model} ...", flush=True)
    return load_model(args.model, temperature=args.temp, seed=args.seed,
                      quantize=args.quantize, device=args.device, dtype=args.dtype)


def run_hf(args, rows, system, fout, stats, loaded) -> None:
    model, tokenizer, sampler = loaded

    prompts = [template_prompt(tokenizer, system, user_message(row["utterance"]))
               for row in rows]

    # Each row contributes n_samples sequences to a generation batch
    rows_per_batch = max(1, args.batch_size // args.n_samples)

    for start in range(0, len(rows), rows_per_batch):
        group = rows[start:start + rows_per_batch]

        batch_prompts = []
        for i in range(len(group)):
            batch_prompts.extend([prompts[start + i]] * args.n_samples)

        response = batch_generate(
            model, tokenizer, prompts=batch_prompts, max_tokens=args.max_tokens,
            sampler=sampler, seed=batch_seed(args.seed, group[0]["row_idx"]))

        for i, row in enumerate(group):
            raw_samples = response.texts[i * args.n_samples:(i + 1) * args.n_samples]
            rec = row_record(row, raw_samples)
            stats.add(rec, sum(len(tokenizer.encode(r)) for r in raw_samples))
            fout.write(json.dumps(rec) + "\n")

        fout.flush()
        stats.report(min(start + len(group), len(rows)), len(rows))


def connect_llamacpp(args):
    from grounding.parsing.llamacpp_backend import LlamaCppServer

    server = LlamaCppServer(args.server_url, workers=args.workers)
    print(f"Connecting to llama-server at {args.server_url} ...", flush=True)
    props = server.wait_ready()
    gguf = server.model_name(props)
    args.model = args.model or gguf
    args.server_info = {"url": args.server_url, "gguf": gguf,
                        "total_slots": props.get("total_slots"),
                        "build_info": props.get("build_info")}
    slots = props.get("total_slots")
    if slots and args.workers > slots:
        print(f"  note: {args.workers} workers but the server has {slots} slots; "
              "extra requests queue", flush=True)
    print(f"  serving {gguf} ({slots} slots)", flush=True)
    return server


def run_llamacpp(args, rows, system, fout, stats, server) -> None:
    from grounding.parsing.llamacpp_backend import load_template_tokenizer

    tokenizer = load_template_tokenizer(args.tokenizer)
    # Same rendered prompt as the hf (and former MLX) path: template, thinking off
    prompts = [template_prompt(tokenizer, system, user_message(row["utterance"]),
                               tokenize=False) for row in rows]

    # Enough rows per chunk to keep every worker (server slot) busy
    rows_per_chunk = max(1, -(-args.workers // args.n_samples))

    for start in range(0, len(rows), rows_per_chunk):
        group = rows[start:start + rows_per_chunk]
        jobs = [(prompts[start + i], batch_seed(args.seed, f"{row['row_idx']}:{k}"))
                for i, row in enumerate(group) for k in range(args.n_samples)]
        done = server.complete_many(jobs, max_tokens=args.max_tokens,
                                    temperature=args.temp)

        for i, row in enumerate(group):
            outs = done[i * args.n_samples:(i + 1) * args.n_samples]
            rec = row_record(row, [o.text for o in outs],
                             {"stop_types": [o.stop_type for o in outs]})
            stats.add(rec, sum(o.tokens for o in outs))
            fout.write(json.dumps(rec) + "\n")

        fout.flush()
        stats.report(min(start + len(group), len(rows)), len(rows))


def run_openai(args, rows, system, fout, stats, loaded=None) -> None:
    from grounding.parsing.openai_backend import OpenAIChat

    chat = OpenAIChat(args.model, workers=args.workers)
    print(f"Sampling {args.model} via the OpenAI API, {args.workers} workers")
    step = max(1, args.workers)

    for start in range(0, len(rows), step):
        group = rows[start:start + step]
        with ThreadPoolExecutor(len(group)) as pool:
            futures = [pool.submit(chat.sample, system, user_message(r["utterance"]),
                                   n=args.n_samples, temperature=args.temp,
                                   max_tokens=args.max_tokens,
                                   seed=batch_seed(args.seed, r["row_idx"]))
                       for r in group]
            # Results are written in row order, so a resumed file stays a prefix
            results = [f.result() for f in futures]

        for row, res in zip(group, results):
            rec = row_record(row, res.texts, {
                "system_fingerprint": res.system_fingerprint,
                "finish_reasons": res.finish_reasons})
            stats.add(rec, res.usage.get("completion_tokens", 0))
            fout.write(json.dumps(rec) + "\n")

        fout.flush()
        stats.report(min(start + len(group), len(rows)), len(rows))


class Stats:
    def __init__(self):
        self.samples = self.valid = self.tokens = 0
        self.t0 = time.time()

    def add(self, rec: dict, tokens: int) -> None:
        self.samples += len(rec["samples"])
        self.valid += sum(s["parsed"] is not None for s in rec["samples"])
        self.tokens += tokens

    def rate(self) -> float:
        elapsed = time.time() - self.t0
        return self.tokens / elapsed if elapsed else 0.0

    def report(self, done: int, total: int) -> None:
        pct = 100 * self.valid / self.samples if self.samples else 0
        print(f"[{done}/{total}] {self.rate():.1f} tok/s | valid JSON {pct:.0f}%",
              file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--input", type=Path,
                        default=Path("data/corpora/nr3d_train.csv"))
    parser.add_argument("--output", type=Path, default=None,
                        help="default: data/rollouts/nr3d_rollouts_<prompt>"
                             "[_<openai model>].jsonl")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="v3")
    parser.add_argument("--backend", choices=["hf", "llamacpp", "openai"],
                        default="hf",
                        help="hf: local PyTorch model; llamacpp: running "
                             "llama-server (GGUF); openai: OpenAI chat API")
    parser.add_argument("--model", default=None,
                        help=f"default: {DEFAULT_HF_MODEL} (hf), the served GGUF "
                             f"file name (llamacpp) or {DEFAULT_OPENAI_MODEL} (openai)")
    parser.add_argument("--limit", type=int, default=2800,
                        help="Number of utterances to process; 0 uses all rows.")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--seed", type=int, default=0,
                        help="base seed; each batch (hf) or row (openai) is "
                             "seeded from it, so resumed runs stay on one stream")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="hf: maximum number of generated sequences per batch.")
    parser.add_argument("--workers", type=int, default=None,
                        help="concurrent requests: openai (default 4) or "
                             "llamacpp (default 8; match llama-server -np)")
    add_model_args(parser)
    add_llamacpp_args(parser)

    args = parser.parse_args()
    args.workers = args.workers or (8 if args.backend == "llamacpp" else 4)

    # Connect before anything is written, so a missing server leaves no file
    server = connect_llamacpp(args) if args.backend == "llamacpp" else None
    args.model = args.model or (DEFAULT_OPENAI_MODEL if args.backend == "openai"
                                else DEFAULT_HF_MODEL)

    if args.output is None:
        tag = ("" if args.backend == "hf"
               else "_" + Path(args.model.replace("/", "_")).stem)
        args.output = Path(f"data/rollouts/nr3d_rollouts_{args.prompt}{tag}.jsonl")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = Path(str(args.output) + ".meta.json")

    system = PROMPTS[args.prompt]
    rows = load_rows(args.input, args.limit)

    done = completed_rows(args.output)
    check_resume(meta_path, invocation(args, system), bool(done))
    rows = [row for row in rows if row["row_idx"] not in done]

    if done:
        print(f"Resuming: {len(done)} rows already complete, {len(rows)} remaining.")

    if not rows:
        print("Nothing to do.")
        return 0

    # Load before opening the output, so a failed load leaves no empty file
    loaded = load_hf(args) if args.backend == "hf" else server
    runner = {"hf": run_hf, "llamacpp": run_llamacpp, "openai": run_openai}[args.backend]
    stats = Stats()
    with args.output.open("a") as fout:
        runner(args, rows, system, fout, stats, loaded)

    print(f"Done. {len(rows)} prompts x {args.n_samples} = {stats.samples} samples "
          f"({stats.valid} parsed). {stats.rate():.1f} tok/s. Wrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
