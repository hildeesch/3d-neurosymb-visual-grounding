"""Score rollout samples through the bound engine and keep the ranking ones.

The output is a *keeper pool*, not the final distillation set: every program that
ranks the target is written with its confidence margin. Downstream training-set
builders apply their own threshold and per-prompt selection policy. Thresholding
here would mean re-executing the whole pool to move it.

The stored program is the *cleaned* one -- the program that actually earned the
rank -- so training targets and admission decisions refer to the same object.
Lowering is used only for scoring; a CleanedProgram is still flat.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


from grounding.paths import REPO_ROOT
from grounding.resolution.config import DEFAULT_RESOLVER
from grounding.resolution import engine
from grounding.resolution.engine import Scenes, rel_num, score


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


#: The R=8 teacher pool. Any other --rollouts must name its own --output
MAIN_POOL = "data/rollouts/nr3d_rollouts_fft_v4.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts", default=MAIN_POOL)
    ap.add_argument("--output", default=None,
                    help="default: nr3d_keepers_fft_v4<arm>.jsonl under "
                         "data/distilled/, where <arm> is '_ext' for the "
                         "extended tree and empty for baseline")
    ap.add_argument("--keep-rank", type=int, default=1,
                    help="keep programs whose target rank is <= this")
    # Recovery would teach inferred constraints the model did not emit
    ap.add_argument("--recover", action="store_true", default=False,
                    help="let clean_program infer constraints the rollout omitted")
    ap.add_argument("--emit-raw", action="store_true", default=False,
                    help="store the model's program instead of the cleaned one")
    ap.add_argument("--limit", type=int, default=0, help="0 = every prompt")
    # Acceptance is evidence about one arm, so this must match the arm the student is evaluated in
    ap.add_argument("--representation", dest="representation",
                    default="baseline",
                    help="which arm executes the candidates: baseline | extended")
    args = ap.parse_args()

    engine.bind(args.representation)

    # The arm goes in the name, not the sidecar alone, so an alias resolves to its canonical arm
    if args.output is None:
        # The default name encodes the arm, not the pool, so other pools must name their output
        if args.rollouts != MAIN_POOL:
            raise SystemExit(
                f"--rollouts is {args.rollouts!r}, not the main pool "
                f"{MAIN_POOL!r}, and the default --output would overwrite the "
                "main keeper pool. Pass --output explicitly, e.g. "
                "data/distilled/nr3d_keepers_fft_v4_recovery_ext.jsonl")
        arm = "_ext" if engine.bound() == "extended" else ""
        args.output = f"data/distilled/nr3d_keepers_fft_v4{arm}.jsonl"

    from grounding.parsing import clean_program, lower_program
    rn = rel_num()

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
    cleaning: Counter = Counter()
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
                    parsed = sample.get("parsed")
                    if parsed is None:
                        outcomes["parse_null"] += 1
                        continue
                    key = canonical(parsed)
                    if key in first_seen:
                        outcomes["duplicate"] += 1
                        continue
                    first_seen[key] = si

                    try:
                        cleaned = clean_program(json.loads(key), utterance, recover=args.recover)
                    except Exception:   # noqa: BLE001 - a bad parse is data
                        outcomes["clean_error"] += 1
                        continue
                    was_cleaned = canonical(cleaned) != key
                    try:
                        ids, scores = score(scenes, scan_id, lower_program(cleaned, rn))
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
                    cleaning["cleaned" if was_cleaned else "untouched"] += 1
                    kept_this = True
                    record = {
                        "row_idx": row["row_idx"], "scan_id": scan_id,
                        "target_id": target_id, "utterance": utterance,
                        "program": parsed if args.emit_raw else cleaned,
                        "cleaned": was_cleaned, "sample_index": si,
                        "target_rank": rank, "margin": margin,
                    }
                    # The preimage is worth keeping only when it differs from the training target
                    if was_cleaned and not args.emit_raw:
                        record["raw_program"] = parsed
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept_prompts += kept_this
            scenes.drop(scan_id)
            if n % 25 == 0 or n == len(by_scan):
                print(f"  {n}/{len(by_scan)} scans, kept {outcomes['kept']}", flush=True)

    executed = outcomes["kept"] + outcomes["wrong_rank"] + outcomes["no_target"]
    stats = {
        # The pool a keeper set came from is part of its identity
        "rollouts": args.rollouts,
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
        "kept_by_cleaning": dict(cleaning),
        "keep_rank": args.keep_rank, "recover": args.recover,
        "stored_program": "raw" if args.emit_raw else "cleaned",
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
    if outcomes["kept"]:
        touched = cleaning["cleaned"]
        print(f"Of the keepers, {touched} ({touched / outcomes['kept']:.1%}) "
              f"were repaired by clean_program.")
    print(f"Wrote {out_path} (+ .stats.json)")
    print("Build training-set variants with build_winner_pool.py and build_uniform_sets.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
