#!/usr/bin/env python3
"""Aggregate multi-seed Stage III dev results by overall, hard, and easy rows.

Seed variation measures training noise on a fixed dataset. This reports each
run and per-arm mean plus sample standard deviation; it never selects a winner.
``--metric expected`` reads the tie-aware credit ``score_checkpoint.py
--expected-top1`` records instead of the every-top-tie flag.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from grounding.parsing.tree import DEFAULT_PROMPT
from grounding.paths import RESULTS_STAGE3
from grounding.resolution.config import FROZEN_RESOLVER, shared_resolver


RUN_RE = re.compile(r"^(?P<arm>.+?)_seed_(?P<seed>\d+)\.json$")
METRICS = ("overall", "hard", "easy")
#: Per-row field each --metric sums
ROW_FIELDS = {"top1": "model", "expected": "model_expected"}
REFERENCE = ("qwen3_1p7b_zeroshot",)
DOSES = (
    ("sft_tree_ext_prompt_uniform", "1.0"),
    ("sft_tree_ext_prompt_uniform_hard150", "1.5"),
    ("sft_tree_ext_prompt_uniform_hard200", "2.0"),
)


def strata(rows: list[dict], field: str = "model") -> dict[str, tuple[float, int]]:
    """Return correct/total counts for overall, hard, and easy rows."""
    counts = {"overall": (sum(row[field] for row in rows), len(rows))}
    for name, is_hard in (("hard", True), ("easy", False)):
        subset = [row for row in rows if row.get("is_hard") is is_hard]
        counts[name] = (sum(row[field] for row in subset), len(subset))
    return counts


def load_run(path: Path, field: str = "model") -> dict:
    data = json.loads(path.read_text())
    rows = data.get("rows")
    if not rows:
        sys.exit(f"{path} carries no per-row results; re-run score_checkpoint.py")
    missing = sum("is_hard" not in row for row in rows)
    if missing:
        sys.exit(
            f"{path}: {missing}/{len(rows)} rows carry no is_hard label, so the "
            "hard/easy break-down would be computed on a silently smaller "
            "denominator. Re-score against data/corpora/nr3d_dev.jsonl."
        )
    if any(field not in row for row in rows):
        sys.exit(f"{path} has no {field!r} per row; rescore it with "
                 "score_checkpoint.py --expected-top1")
    return {"resolver": data.get("resolver") or {}, "strata": strata(rows, field)}


def collect_runs(results_dir: Path, field: str = "model") -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = defaultdict(dict)
    for path in sorted(results_dir.glob("*_seed_*.json")):
        if match := RUN_RE.match(path.name):
            runs["sft_tree_ext_" + match["arm"]][int(match["seed"])] = load_run(path, field)
    if not runs:
        sys.exit(f"no <arm>_seed_<k>.json under {results_dir}")
    return runs


def pct(correct: float, total: int) -> float:
    return 100.0 * correct / total if total else 0.0


def rates(run: dict) -> dict[str, float]:
    return {metric: pct(*run["strata"][metric]) for metric in METRICS}


def print_runs(runs: dict[str, dict[int, dict]]) -> None:
    print("PER RUN")
    print(f"  {'arm':<40}{'seed':>5}{'overall':>10}{'hard':>10}{'easy':>10}")
    for arm in sorted(runs):
        for seed, run in sorted(runs[arm].items()):
            values = rates(run)
            print(f"  {arm:<40}{seed:>5}"
                  f"{values['overall']:>9.2f}%{values['hard']:>9.2f}%"
                  f"{values['easy']:>9.2f}%")


def aggregate_arm(seed_runs: dict[int, dict], seeds: list[int]) -> dict:
    values = {metric: [rates(seed_runs[seed])[metric] for seed in seeds]
              for metric in METRICS}
    entry = {"seeds": seeds}
    for metric, sample in values.items():
        entry[metric] = {
            "mean": statistics.fmean(sample),
            # Undefined on one seed, not 0.0: printing 0.00 would read as stable, not as unmeasured
            "std": statistics.stdev(sample) if len(sample) > 1 else None,
            "values": sample,
        }
    return entry


def print_aggregate(runs: dict[str, dict[int, dict]], expected_seeds: int) -> dict:
    print("\nAGGREGATE (mean +- sample std over seeds)")
    print(f"  {'arm':<40}{'overall':>16}{'hard':>16}{'easy':>16}{'seeds':>7}")
    report = {"runs": {}, "aggregate": {}}
    for arm in sorted(runs):
        seed_runs = runs[arm]
        seeds = sorted(seed_runs)
        values = {seed: rates(run) for seed, run in seed_runs.items()}
        report["runs"][arm] = {str(seed): values[seed] for seed in seeds}
        if arm in REFERENCE:
            reference = values[seeds[0]]
            print(f"  {arm:<40}" + "".join(
                f"{reference[metric]:>15.2f}%" for metric in METRICS) + f"{'ref':>7}")
            report["aggregate"][arm] = {"reference_line": True, **reference}
        elif len(seeds) != expected_seeds:
            print(f"  {arm:<40}  INCOMPLETE: {len(seeds)}/{expected_seeds} "
                  f"seeds present ({seeds}) -- not aggregated")
        else:
            entry = aggregate_arm(seed_runs, seeds)
            cells = [
                f"{entry[metric]['mean']:>9.2f}+-{entry[metric]['std']:<5.2f}"
                if entry[metric]["std"] is not None
                else f"{entry[metric]['mean']:>9.2f}{'  n/a':<7}"
                for metric in METRICS]
            print(f"  {arm:<40}" + "".join(cells) + f"{len(seeds):>7}")
            report["aggregate"][arm] = entry
    return report


def add_dose_ordering(report: dict) -> None:
    available = [(arm, dose) for arm, dose in DOSES
                 if arm in report["aggregate"] and "hard" in report["aggregate"][arm]]
    if len(available) != len(DOSES):
        return

    means = [report["aggregate"][arm]["hard"]["mean"] for arm, _ in available]
    ordered = means[0] < means[1] < means[2]
    spread = max(means) - min(means)
    # None on a single seed: there is no spread to compare the doses against
    stds = [report["aggregate"][arm]["hard"]["std"] for arm, _ in available]
    largest_std = max(stds) if all(sd is not None for sd in stds) else None
    print("\nhard-subset means by dose: "
          + "  ".join(f"{dose}={mean:.2f}%" for (_, dose), mean in zip(available, means)))
    print(f"ordering uniform < 1.5 < 2.0 holds on the means: {ordered}")
    if largest_std is None:
        print(f"  spread across doses {spread:.2f} pp; per-arm seed std NOT "
              "MEASURED (one seed per arm) -- the ordering is a single draw, "
              "not a result")
    else:
        print(f"  spread across doses {spread:.2f} pp vs largest per-arm "
              f"seed std {largest_std:.2f} pp")
    report["hard_dose_ordering"] = {
        "means": dict(zip([dose for _, dose in available], means)),
        "ordered_uniform_lt_150_lt_200": ordered,
        "spread_pp": spread,
        "largest_seed_std_pp": largest_std,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expect-seeds", type=int, default=3)
    parser.add_argument("--metric", choices=sorted(ROW_FIELDS), default="top1",
                        help="top1: every top tie counts (selection protocol). "
                             "expected: 1/k for a k-way top tie")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.results is None:
        args.results = (RESULTS_STAGE3.parent / "expected_top1"
                        if args.metric == "expected" else RESULTS_STAGE3) / DEFAULT_PROMPT
    runs = collect_runs(args.results, ROW_FIELDS[args.metric])
    resolver = shared_resolver(
        {f"{arm}__seed{seed}": run["resolver"]
         for arm, arm_runs in runs.items() for seed, run in arm_runs.items()},
        hint="Rescore the runs under one resolver configuration.")
    drift = {key: resolver.get(key) for key, value in FROZEN_RESOLVER.items()
             if resolver.get(key) != value}
    if drift:
        print(f"  WARNING: resolver differs from the frozen configuration: {drift}")

    print(f"resolver: {json.dumps(resolver)}\nmetric: {args.metric}\n")
    print_runs(runs)
    report = {"resolver": resolver, "metric": args.metric,
              **print_aggregate(runs, args.expect_seeds)}
    add_dose_ordering(report)

    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
