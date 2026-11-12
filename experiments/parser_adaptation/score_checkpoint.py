"""Evaluate parser checkpoints through the selected representation (default: extended).

``--rescore`` consumes stored generations without loading a language model.
The resolver reads FROZEN_RESOLVER and records its actual settings. Selection
credits targets tied at the top; headline evaluate.py/compare_baseline.py use
the declared tie-break policy. These are different evaluation protocols.
``--expected-top1`` also records tie-aware credit: 1/k for a k-way top tie."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path


from grounding.parsing import PROMPTS
from grounding.parsing.generation import (add_model_args, batch_generate,
                                          encode_chat, free_memory, load_model)
from grounding.parsing.training_set import user_turn
from grounding.resolution.config import resolver_metadata
from grounding.paths import CORPORA, REPO_ROOT, RESULTS_FINAL, RESULTS_STAGE3, ROLLOUTS
from grounding.paths import SPLITS_PATH as SPLITS
from grounding.resolution import engine
from grounding.resolution.engine import (Scenes, rel_num, score,
                                         target_rank)
from grounding.resolution.ranking import expected_rank

def _default_out(split: str, tag: str, expected_top1: bool = False) -> str:
    """Choose the result directory for the checkpoint-selection protocol."""
    directory = RESULTS_FINAL if split == "test" else RESULTS_STAGE3
    from grounding.resolution.artifacts import checkpoint_result_name
    if split == "dev" and expected_top1:
        directory = directory.parent / "expected_top1"
    if split != "dev":
        directory = directory / "checkpoint_scoring" / split
    return str((directory / checkpoint_result_name(tag)).relative_to(REPO_ROOT))

CSV = {
    "train": CORPORA / "nr3d_train.csv",
    "test": CORPORA / "nr3d_test.csv",
}
#: The teacher pool --split train subtracts; a stale name scores a model on its own rows
POOL = ROLLOUTS / "nr3d_rollouts_fft_v4.jsonl"
#: Frozen train-derived dev; --n takes a prefix for smoke runs, --n 0 uses all rows
DEV = CORPORA / "nr3d_dev.jsonl"
RULE_PARSES = CORPORA / "nr3d.jsonl"


def pick_rows(split: str, n: int, seed: int, exclude_pool: bool) -> list[dict]:
    if split == "dev":
        if not DEV.exists():
            sys.exit(f"{DEV} does not exist. Build it once with\n"
                     "  python scripts/prepare/build_dev_set.py")
        rows = [json.loads(l) for l in open(DEV) if l.strip()]
        sidecar = Path(str(DEV) + ".stats.json")
        sha = (json.loads(sidecar.read_text())["sha256"][:16] + "..."
               if sidecar.exists() else "no sidecar")
        print(f"  {len(rows)} frozen dev rows, sha256 {sha}")
        # A PREFIX, never a sample: build_dev_set's draw already has the prefix
        # property, so `--n k` is the same k rows for every arm and is safe for
        # a smoke run. Comparing two arms at different k is still wrong, and
        # select_adapter.py refuses it, not this note
        if n and n < len(rows):
            rows = rows[:n]
            print(f"  truncated to the first {n} (smoke run, not a comparison)")
        return [{
            "dataset_idx": r["row_idx"],
            "scan_id": r["scan_id"],
            "target_id": r["target_id"],
            "target_label": r["instance_type"],
            "utterance": r["utterance"].strip(),
            "is_hard": r["is_hard"],
        } for r in rows]

    rows = list(csv.DictReader(open(CSV[split], newline="")))
    idxs = list(range(len(rows)))

    if split == "train" and exclude_pool:
        if not POOL.exists():
            sys.exit(f"{POOL} does not exist, so the pool cannot be excluded "
                     "and this run would score the model on its own training "
                     "utterances. Use --split dev, or pass --include-pool if "
                     "you really mean to.")
        seen = {json.loads(l)["row_idx"] for l in open(POOL) if l.strip()}
        idxs = [i for i in idxs if i not in seen]
        print(f"  {len(seen)} pool utterances excluded; {len(idxs)} unseen remain")

    if n and n < len(idxs):
        idxs = random.Random(seed).sample(sorted(idxs), n)

    idxs.sort()
    return [{
        "dataset_idx": i,
        "scan_id": rows[i]["scan_id"],
        "target_id": rows[i]["target_id"],
        "target_label": rows[i]["instance_type"],
        "utterance": rows[i]["utterance"].strip(),
    } for i in idxs]


def load_raw(path: Path) -> dict[int, str]:
    """dataset_idx -> raw generation, from a result JSON or a parses jsonl."""
    if path.suffix == ".json":
        d = json.loads(path.read_text())
        return {int(r["dataset_idx"]): r["raw"] for r in d["rows"]}
    out = {}
    for line in path.open():
        if line.strip():
            r = json.loads(line)
            out[int(r.get("dataset_idx", r["row_idx"]))] = r["raw"]
    return out


def expected_credit(scores, obj_ids: list, target_id) -> tuple[float, int | None]:
    """(top-1 credit, top-tie size) under expected_rank; (0.0, None) if absent."""
    ids = [int(i) for i in obj_ids]
    if int(target_id) not in ids:
        return 0.0, None
    ranked = expected_rank([float(v) for v in scores], ids.index(int(target_id)))
    return ranked["top1"], ranked["tied"]


def resolver_config() -> dict:
    return resolver_metadata(representation=engine.bound())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # dev is the split to choose on; test is for the number reported once nothing is left to choose
    ap.add_argument("--split", choices=["dev", "train", "test"], default="dev")
    ap.add_argument("--n", type=int, default=500,
                    help="0 = every row in the split. For --split dev this is "
                         "a deterministic PREFIX of the frozen draw, for smoke "
                         "runs; leave it at 0 for a real comparison")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default=None,
                    help="PEFT LoRA adapter directory (train_lora.py output, or "
                         "an MLX adapter converted with "
                         "scripts/prepare/convert_mlx_adapter.py)")
    ap.add_argument("--representation", dest="representation",
                    default="extended",
                    help="which arm's resolver scores the programs: baseline | "
                         "extended. The final system is extended")
    ap.add_argument("--rescore", type=Path, default=None,
                    help="re-score existing generations instead of generating: "
                         "a result JSON from this command or a parse_split.py "
                         "parses jsonl covering the split. The model is not loaded")
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--no-think", action="store_true", help="Append ' /no_think' to prompt")
    ap.add_argument("--include-pool", action="store_true", help="Do not exclude rollout pool utterances")
    # A checkpoint must be scored under the prompt it was trained under. The
    # sets built by build_uniform_sets.py are v4, whose parsing rules differ from
    # v2's (v2 still declares the `viewpoint` field v4 removed), so a v4 adapter
    # scored under v2 measures out-of-distribution behaviour
    ap.add_argument("--prompt", default="v4", choices=sorted(PROMPTS),
                    help="system prompt version the checkpoint was trained "
                         "under. v4 is the current generation; pass v2 to "
                         "reproduce results from before that switch")
    ap.add_argument("--expected-top1", action="store_true",
                    help="also record per-row tie-aware credit (model_expected, "
                         "model_tied) and its totals; model stays the "
                         "every-top-tie selection flag")
    ap.add_argument("--out", type=Path, default=None)
    add_model_args(ap)
    args = ap.parse_args()

    engine.bind(args.representation)
    picks = pick_rows(args.split, args.n, args.seed, not args.include_pool)
    generated_by = None
    if args.rescore:
        src = args.rescore if args.rescore.is_absolute() else REPO_ROOT / args.rescore
        if src.suffix == ".json":
            prev = json.loads(src.read_text())
            if prev.get("split") != args.split:
                sys.exit(f"{src.name} is split {prev.get('split')!r}, not {args.split!r}")
            args.model, args.adapter = prev["model"], prev.get("adapter")
        generated_by = str(src.relative_to(REPO_ROOT)) if src.is_relative_to(REPO_ROOT) else str(src)
    tag = Path(args.adapter or args.model).name
    out_path = REPO_ROOT / (args.out or _default_out(args.split, tag, args.expected_top1))
    print(f"{len(picks)} utterances from {args.split}, scored through the "
          f"{engine.bound()} arm")

    from grounding.parsing import clean_program, extract_json_objects, lower_program

    rn = rel_num()
    t0 = time.time()
    if args.rescore:
        raw = load_raw(src)
        missing = [p["dataset_idx"] for p in picks if p["dataset_idx"] not in raw]
        if missing:
            sys.exit(f"{len(missing)} of {len(picks)} {args.split} rows have no "
                     f"generation in {src.name}; a rescoring must cover the split")
        for p in picks:
            p["raw"] = raw[p["dataset_idx"]]
        print(f"  {len(picks)} generations read from {generated_by}; model not loaded")
    else:
        model, tok, sampler = load_model(args.model, args.adapter,
                                         quantize=args.quantize,
                                         device=args.device, dtype=args.dtype)
        suffix = " /no_think" if args.no_think else ""

        system = PROMPTS[args.prompt]
        print(f"  scoring under prompt {args.prompt} "
              f"({len(system)} chars), shared v4 SFT user turn")

        def prompt(sentence: str):
            # user_turn rendered the targets, so the scored prompt matches training
            return encode_chat(tok, system, user_turn(sentence) + suffix)

        for i in range(0, len(picks), args.batch_size):
            chunk = picks[i:i + args.batch_size]
            resp = batch_generate(model, tok, prompts=[prompt(p["utterance"]) for p in chunk],
                                  max_tokens=args.max_tokens, sampler=sampler, verbose=False)
            for p, text in zip(chunk, resp.texts):
                p["raw"] = text.split("<|im_end|>")[0]
            if (i // args.batch_size) % 10 == 0:
                print(f"  generated {min(i + args.batch_size, len(picks))}/{len(picks)}  ({time.time() - t0:.0f}s)", flush=True)

        del model
        free_memory()

    rule = {int(json.loads(l)["dataset_idx"]): json.loads(l).get("json_obj")
            for l in open(RULE_PARSES)} if args.split == "test" and RULE_PARSES.exists() else {}
    splits = json.loads(SPLITS.read_text()) if args.split == "test" else {}
    # Dev rows use build_dev_set.py's is_hard rule, so hard/easy needs no second definition
    dev_meta = ({str(p["dataset_idx"]): {"is_hard": p["is_hard"]} for p in picks}
                if args.split == "dev" else {})

    scenes = Scenes()
    rows, n_json = [], 0
    
    for p in sorted(picks, key=lambda q: q["scan_id"]):
        parsed = extract_json_objects(p["raw"])
        n_json += parsed is not None
        
        rec = {k: p[k] for k in ("dataset_idx", "scan_id", "target_label", "utterance")}
        rec.update({"json_ok": parsed is not None, "raw": p["raw"], "program": parsed})

        for name, obj in (("model", parsed), ("rule", rule.get(p["dataset_idx"]))):
            hit, credit, tied = False, 0.0, None
            if name == "model" and obj is not None:
                try:
                    obj = lower_program(clean_program(obj, p["utterance"], recover=False), rn)
                except Exception:
                    obj = None
            if obj:
                try:
                    ids, sc = score(scenes, p["scan_id"], obj)
                    hit = target_rank(sc, ids, p["target_id"]) == 1
                    credit, tied = expected_credit(sc, ids, p["target_id"])
                except (KeyError, FileNotFoundError) as e:
                    raise SystemExit(
                        f"Scene data missing for {p['scan_id']}: {e!r}\n"
                        f"Scans load through the baseline tree, which needs the "
                        f"referit3d point clouds under data/raw/referit3d/.") from e
                except Exception:
                    hit, credit, tied = False, 0.0, None
            rec[name] = hit
            if args.expected_top1 and name == "model":
                rec.update(model_expected=credit, model_tied=tied)

        if meta := splits.get(str(p["dataset_idx"])) or dev_meta.get(str(p["dataset_idx"])):
            rec["is_hard"] = meta["is_hard"]
            if "is_view_dependent" in meta:
                rec["is_view_dependent"] = meta["is_view_dependent"]
            
        rows.append(rec)
        scenes.drop(p["scan_id"])

    n, acc = len(rows), sum(r["model"] for r in rows)
    print(f"\n=== {args.split}: {tag} through the {engine.bound()} arm ({n} utterances) ===")
    print(f"  valid JSON      {n_json}/{n}")
    print(f"  top-1 (model)   {acc}/{n} ({acc / n:.1%})")
    
    summary = {"n": n, "json_ok": n_json, "model_top1": acc}
    if args.expected_top1:
        expected = sum(r["model_expected"] for r in rows)
        summary["model_expected_top1"] = expected
        print(f"  expected top-1  {expected:.2f}/{n} ({expected / n:.1%})")
    if rule:
        rb = sum(r["rule"] for r in rows)
        both = sum(r["model"] and r["rule"] for r in rows)
        summary["rule_top1"] = rb
        print(f"  top-1 (rule)    {rb}/{n} ({rb / n:.1%})   <- paired baseline")
        print(f"  model wins {acc - both}, rule wins {rb - both}, both {both}")

    if splits or dev_meta:
        print(f"\n  {'split':<18}{'model':>14}{'rule':>14}")
        for name, key, want in (("easy", "is_hard", False), ("hard", "is_hard", True),
                                ("view_dependent", "is_view_dependent", True),
                                ("view_independent", "is_view_dependent", False)):
            sub = [r for r in rows if r.get(key) is want]
            if not sub: continue
            m = sum(r["model"] for r in sub)
            cells = f"{m:>6}/{len(sub)} {m / len(sub):>5.1%}"
            if rule:
                q = sum(r["rule"] for r in sub)
                cells += f"{q:>7}/{len(sub)} {q / len(sub):>5.1%}"
            summary[name] = {"n": len(sub), "model": m}
            if args.expected_top1:
                summary[name]["model_expected"] = sum(r["model_expected"] for r in sub)
            print(f"  {name:<18}{cells}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "split": args.split, "model": args.model, "adapter": args.adapter,
        "prompt": args.prompt,
        # The arm and resolver the programs ran under; selections do not carry across them
        "resolver": resolver_config(),
        "generations_from": generated_by,
        "summary": summary, "rows": rows}, indent=2))
    print(f"\nWrote {out_path}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
