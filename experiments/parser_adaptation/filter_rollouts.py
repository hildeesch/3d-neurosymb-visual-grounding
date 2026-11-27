"""Score rollout samples through the bound engine and keep the ranking ones.

The output is a *keeper pool*, not the final distillation set: every program that
ranks the target is written with its confidence margin. Downstream training-set
builders apply their own threshold and per-prompt selection policy. Thresholding
here would mean re-executing the whole pool to move it.

Each distinct tree goes through ``to_executable`` -- the grammar, checked after
the fact, which rejects and never repairs -- and the checked program is scored.
The stored ``program`` is the model's tree as emitted: the program that earned
the rank is the SFT target. Trees outside the grammar are counted by
InvalidTree reason, per distinct program (``invalid_by_reason``) and per sample
(``sample_outcomes``, ``sample_invalid_rate``), and the depth of the valid and
kept trees is reported. The rollouts' ``.meta.json`` says whether the pool was
decoded under the grammar.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


from grounding.parsing.tree import (InvalidTree, depth_of, is_tree_prompt,
                                    to_executable)
from grounding.paths import REPO_ROOT
from grounding.resolution.config import DEFAULT_RESOLVER
from grounding.resolution import engine
from grounding.resolution.engine import Scenes, score


def canonical(program: dict) -> str:
    """Key identifying a program up to key order, for deduplication."""
    return json.dumps(program, sort_keys=True, ensure_ascii=False)


def rank_and_margin(scores, obj_ids: list, target_id) -> tuple[int | None, float]:
    """Strictly-greater target rank and relative margin against the best distractor.

    Top ties receive rank 1. Margin is (target - best_other) / target when the
    target score is positive, otherwise zero. A sole candidate gets margin 1;
    an absent target returns (None, 0.0).
    """
    target_id = int(target_id)
    ids = [int(i) for i in obj_ids]
    if target_id not in ids:
        return None, 0.0
    i = ids.index(target_id)
    vals = [float(v) for v in scores]
    t = vals[i]
    rank = sum(1 for v in vals if v > t) + 1
    others = vals[:i] + vals[i + 1:]
    if not others:
        return rank, 1.0
    best_other = max(others)
    # A non-positive target score carries no confidence to measure
    margin = (t - best_other) / t if t > 0 else 0.0
    return rank, margin


#: The Qwen3-30B-A3B teacher's pool (teacher_student_grid.rollouts_path("qwen"))
MAIN_POOL = "data/rollouts/nr3d_rollouts_qwen_tree_v1.jsonl"


def default_keepers(rollouts: str, representation: str) -> str:
    """``.../nr3d_rollouts_<x>.jsonl`` -> ``data/distilled/nr3d_keepers_<x>[_ext].jsonl``."""
    stem = Path(rollouts).name.removesuffix(".jsonl")
    if not stem.startswith("nr3d_rollouts_"):
        raise SystemExit(f"cannot name keepers for {rollouts!r}; pass --output")
    arm = "_ext" if representation == "extended" else ""
    return f"data/distilled/nr3d_keepers_{stem.removeprefix('nr3d_rollouts_')}{arm}.jsonl"


def rollouts_meta(path: Path) -> dict:
    """The rollouts' ``.meta.json`` written by generate_rollouts.py."""
    meta = Path(str(path) + ".meta.json")
    if not meta.exists():
        raise SystemExit(f"{path} has no {meta.name}; it was not written by this "
                         "branch's generate_rollouts.py")
    recorded = json.loads(meta.read_text())
    if not is_tree_prompt(recorded.get("prompt")):
        raise SystemExit(f"{path} was generated under prompt "
                         f"{recorded.get('prompt')!r}, not a tree prompt")
    return recorded


def sample_outcome(sample: dict) -> str:
    """Grammar outcome of one sample: ok, no_json or the InvalidTree reason."""
    parsed = sample.get("parsed")
    if parsed is None:
        return "no_json"
    try:
        to_executable(parsed)
    except InvalidTree as e:
        return e.reason
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts", default=MAIN_POOL)
    ap.add_argument("--output", default=None,
                    help="default: data/distilled/nr3d_keepers_<rollouts "
                         "name>[_ext].jsonl")
    ap.add_argument("--keep-rank", type=int, default=1,
                    help="keep programs whose target rank is <= this")
    ap.add_argument("--limit", type=int, default=0, help="0 = every prompt")
    # Acceptance is evidence about one arm, so this must match the arm the student is evaluated in
    ap.add_argument("--representation", dest="representation",
                    default="extended",
                    help="which arm executes the candidates: extended (the "
                         "tree's vocabulary) | baseline")
    args = ap.parse_args()

    meta = rollouts_meta(REPO_ROOT / args.rollouts)
    engine.bind(args.representation)
    args.output = args.output or default_keepers(args.rollouts, engine.bound())

    rows = [json.loads(l) for l in open(REPO_ROOT / args.rollouts) if l.strip()]
    if not rows:
        raise SystemExit(
            f"{args.rollouts} has no rollout rows. generate_rollouts.py most "
            "likely stopped before its first batch (check its output for a "
            "model-loading or CUDA error) or is still running.")
    if args.limit:
        rows = rows[: args.limit]
    # Grouped by scan so each scene is built once and dropped once
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_scan[r["scan_id"]].append(r)

    # Dedupe before the scene loop: cost is then per candidate, not per sample
    total_samples = sum(len(r["samples"]) for r in rows)
    distinct = sum(
        len({canonical(s["parsed"]) for s in r["samples"] if s.get("parsed") is not None})
        for r in rows
    )
    print(f"{len(rows)} prompts over {len(by_scan)} scans")
    if distinct:
        print(f"{total_samples} samples -> {distinct} distinct programs "
              f"({total_samples / distinct:.1f}x fewer executions)")

    scenes = Scenes()
    outcomes: Counter = Counter()
    # Grammar outcome per distinct program and per sample, and the depth of
    # the trees that passed the check / were kept
    invalid: Counter = Counter()
    sample_outcomes: Counter = Counter()
    depth_valid: Counter = Counter()
    depth_kept: Counter = Counter()
    kept_prompts = 0
    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as fout:
        for n, (scan_id, prompts) in enumerate(sorted(by_scan.items()), 1):
            for row in prompts:
                target_id = str(row["target_id"])
                utterance = row["utterance"]
                kept_this = False
                # First sample index per distinct program, for traceability
                first_seen: dict[str, int] = {}
                for si, sample in enumerate(row["samples"]):
                    sample_outcomes[sample_outcome(sample)] += 1
                    parsed = sample.get("parsed")
                    if parsed is None:
                        outcomes["parse_null"] += 1
                        continue
                    key = canonical(parsed)
                    if key in first_seen:
                        outcomes["duplicate"] += 1
                        continue
                    first_seen[key] = si

                    # The grammar, checked after the fact: reject, never repair
                    try:
                        program = to_executable(parsed)
                    except InvalidTree as e:
                        outcomes["invalid_tree"] += 1
                        invalid[e.reason] += 1
                        continue
                    depth_valid[depth_of(program)] += 1
                    try:
                        ids, scores = score(scenes, scan_id, program)
                    except Exception:   # noqa: BLE001 - a bad parse is data
                        outcomes["exec_error"] += 1
                        continue

                    rank, margin = rank_and_margin(scores, ids, target_id)
                    if rank is None:
                        outcomes["no_target"] += 1
                        continue
                    if rank > args.keep_rank:
                        outcomes["wrong_rank"] += 1
                        continue

                    outcomes["kept"] += 1
                    depth_kept[depth_of(program)] += 1
                    kept_this = True
                    fout.write(json.dumps({
                        "row_idx": row["row_idx"], "scan_id": scan_id,
                        "target_id": target_id, "utterance": utterance,
                        "program": parsed, "sample_index": si,
                        "target_rank": rank, "margin": margin,
                    }, ensure_ascii=False) + "\n")
                kept_prompts += kept_this
            scenes.drop(scan_id)
            if n % 25 == 0 or n == len(by_scan):
                print(f"  {n}/{len(by_scan)} scans, kept {outcomes['kept']}", flush=True)

    executed = outcomes["kept"] + outcomes["wrong_rank"] + outcomes["no_target"]
    n_samples = sum(sample_outcomes.values())
    stats = {
        # The pool a keeper set came from is part of its identity
        "rollouts": args.rollouts,
        "prompt": meta["prompt"],
        # How the pool was decoded (generate_rollouts.py); None = unconstrained
        "grammar": meta.get("grammar"),
        "engine": f"the {engine.bound()} arm (load_one_scene, label_type=gt)",
        "representation": engine.bound(),
        "resolver": {"norm_mode": DEFAULT_RESOLVER.norm_mode,
                     "color_mode": DEFAULT_RESOLVER.color_mode, "combine": DEFAULT_RESOLVER.combine,
                     "ordinal_tau": DEFAULT_RESOLVER.ordinal_tau},
        "prompts": len(rows), "samples": total_samples,
        "distinct_programs": distinct,
        "outcomes": dict(outcomes), "executed_programs": executed,
        "kept": outcomes["kept"],
        "keep_rate_of_executed": outcomes["kept"] / executed if executed else 0.0,
        "prompts_with_at_least_one_keep": kept_prompts,
        "prompt_coverage": kept_prompts / len(rows) if rows else 0.0,
        "keep_rank": args.keep_rank,
        "invalid_by_reason": dict(invalid.most_common()),
        "sample_outcomes": dict(sample_outcomes.most_common()),
        "sample_invalid_rate": (1 - sample_outcomes["ok"] / n_samples
                                if n_samples else 0.0),
        "depth_of_valid_distinct": dict(sorted(depth_valid.items())),
        "depth_of_kept": dict(sorted(depth_kept.items())),
    }
    Path(str(out_path) + ".stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\n--- execution-feedback filter (the {engine.bound()} arm, "
          f"{DEFAULT_RESOLVER.norm_mode}) ---")
    for k, v in outcomes.most_common():
        print(f"  {k:14s} {v}")
    print(f"\nKept {outcomes['kept']} programs across {kept_prompts}/{len(rows)} "
          f"prompts ({stats['prompt_coverage']:.1%} prompt coverage).")
    if executed:
        print(f"Keep rate among executed programs: {outcomes['kept']}/{executed} "
              f"= {stats['keep_rate_of_executed']:.1%}")
    mode = "grammar" if meta.get("grammar") else "no grammar"
    print(f"\n{meta['prompt']}, decoded with {mode}: "
          f"{stats['sample_invalid_rate']:.1%} of samples outside the grammar")
    for k, v in invalid.most_common():
        print(f"  invalid {k:16s} {v}")
    print(f"  depth of valid distinct trees {stats['depth_of_valid_distinct']}")
    print(f"  depth of kept trees           {stats['depth_of_kept']}")
    print(f"Wrote {out_path} (+ .stats.json)")
    print("Build training-set variants with build_winner_pool.py and build_uniform_sets.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
