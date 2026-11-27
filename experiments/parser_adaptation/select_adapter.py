#!/usr/bin/env python3
"""Select a parser adapter from paired, held-out dev-set results only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from grounding.parsing.tree import DEFAULT_PROMPT
from grounding.paths import REPO_ROOT, RESULTS_STAGE3
from grounding.resolution.config import FROZEN_RESOLVER, shared_resolver
from grounding.resolution.statistics import mcnemar


def load_arm(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data.get("split") != "dev":
        sys.exit(
            f"{path.name} is split {data.get('split')!r}, not 'dev'.\n"
            "  Adapters are chosen on dev. Scoring a choice on test makes the "
            "reported number the number that made the choice."
        )

    rows = {int(row["dataset_idx"]): bool(row["model"]) for row in data["rows"]}
    hard = {
        int(row["dataset_idx"]): bool(row["is_hard"])
        for row in data["rows"]
        if "is_hard" in row
    }
    return {
        "name": Path(data.get("adapter") or data["model"]).name,
        "path": path,
        "rows": rows,
        "hard": hard,
        "json_ok": data["summary"]["json_ok"],
        "resolver": data.get("resolver"),
    }


def shared_keys(arms: list[dict]) -> list[int]:
    """Require every arm to have scored the same rows."""
    keysets = {frozenset(arm["rows"]) for arm in arms}
    if len(keysets) != 1:
        sizes = {arm["name"]: len(arm["rows"]) for arm in arms}
        common = set.intersection(*(set(arm["rows"]) for arm in arms))
        sys.exit(
            f"the arms were scored on different rows: {sizes}\n"
            f"  ({len(common)} rows in common). Re-run the short ones at --n 0 "
            "so every arm sees the whole frozen dev set."
        )
    return sorted(keysets.pop())


def resolver_status(resolver: dict) -> tuple[bool, str]:
    missing = sorted(FROZEN_RESOLVER.keys() - resolver.keys())
    drift = {
        key: (resolver.get(key), expected)
        for key, expected in FROZEN_RESOLVER.items()
        if key in resolver and resolver[key] != expected
    }
    if drift:
        detail = ", ".join(
            f"{key}={actual!r} but the frozen configuration is {expected!r}"
            for key, (actual, expected) in sorted(drift.items())
        )
        return False, (
            f"PROVISIONAL: scored under {detail} -- rescore the arms and "
            "re-select before citing this"
        )

    if missing:
        return False, (
            f"INCOMPLETE METADATA: {', '.join(missing)} not recorded; the recorded "
            "settings match the target. Verify the run's configuration before citing it."
        )

    frozen = ", ".join(f"{key}={value}" for key, value in sorted(FROZEN_RESOLVER.items()))
    return True, (
        f"scored under the frozen resolver configuration {frozen}. "
        "Selection applies to the supplied dev runs only."
    )


def accuracy(arm: dict, keys: list[int]) -> float:
    return sum(arm["rows"][key] for key in keys) / len(keys) if keys else 0.0


def arm_report(arm: dict, leader: dict, keys: list[int], hard_keys: list[int],
               easy_keys: list[int]) -> dict:
    comparison = None
    if arm is not leader:
        wins = sum(arm["rows"][key] and not leader["rows"][key] for key in keys)
        losses = sum(leader["rows"][key] and not arm["rows"][key] for key in keys)
        comparison = {
            "wins": wins,
            "losses": losses,
            "mcnemar_p": round(mcnemar(wins, losses), 4),
        }
    return {
        "name": arm["name"],
        "result_file": str(arm["path"].relative_to(REPO_ROOT)),
        "correct": arm["correct"],
        "accuracy": arm["accuracy"],
        "hard_accuracy": accuracy(arm, hard_keys),
        "easy_accuracy": accuracy(arm, easy_keys),
        "json_ok": arm["json_ok"],
        "vs_leader": comparison,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        nargs="*",
        help="dev result JSON files (default: seeded adapter runs only; zero-shot controls require explicit --results)",
    )
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="significance for separation from the leader")
    parser.add_argument("--out", default=str(RESULTS_STAGE3 / DEFAULT_PROMPT / "adapter_selection_dev.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = ([REPO_ROOT / path for path in args.results] if args.results
             else sorted((RESULTS_STAGE3 / DEFAULT_PROMPT).glob("*_seed_*.json")))
    if len(paths) < 2:
        sys.exit(
            f"found {len(paths)} dev result files; need at least two to choose between. "
            "Run score_checkpoint.py --split dev first."
        )

    arms = [load_arm(path) for path in paths]
    keys = shared_keys(arms)
    resolver = shared_resolver(
        {str(arm["path"]): arm["resolver"] for arm in arms},
        hint="Rescore every arm with score_checkpoint.py --rescore under one "
             "arm and configuration.")
    for arm in arms:
        arm["correct"] = sum(arm["rows"][key] for key in keys)
        arm["accuracy"] = arm["correct"] / len(keys)
    arms.sort(key=lambda arm: -arm["correct"])
    leader = arms[0]

    hard_keys = [key for key in keys if leader["hard"].get(key)]
    easy_keys = [key for key in keys if key in leader["hard"] and not leader["hard"][key]]
    matches_frozen_resolver, note = resolver_status(resolver)
    reports = [arm_report(arm, leader, keys, hard_keys, easy_keys) for arm in arms]

    print(f"\n=== adapter selection on {len(keys)} held-out Nr3D-train dev rows ===")
    print(f"  {'arm':<38}{'top-1':>14}{'hard':>9}{'easy':>9}{'vs leader':>22}")
    for report in reports:
        comparison = report["vs_leader"]
        cell = "--" if comparison is None else (
            f"+{comparison['wins']}/-{comparison['losses']} "
            f"p={mcnemar(comparison['wins'], comparison['losses']):.3f}"
        )
        score = f"{report['correct']}/{len(keys)} {report['accuracy']:.3f}"
        print(
            f"  {report['name']:<38}{score:>14}"
            f"{report['hard_accuracy']:>9.3f}{report['easy_accuracy']:>9.3f}{cell:>22}"
        )

    tied = [
        report for report in reports
        if report["vs_leader"] is None or report["vs_leader"]["mcnemar_p"] >= args.alpha
    ]
    print(f"\n  leader:  {leader['name']}  ({leader['accuracy']:.3f})")
    print(f"  resolver: {note}")
    if len(tied) > 1:
        names = ", ".join(report["name"] for report in tied)
        print(f"  {len(tied)} arms are not separated from it at alpha={args.alpha}: {names}")
        print("  -> the dev set does not distinguish them; the leader stands, and the margin is not a result.")

    report = {
        "n": len(keys),
        "alpha": args.alpha,
        "rows": "dev",
        "resolver": resolver,
        "frozen_resolver": FROZEN_RESOLVER,
        "matches_frozen_resolver": matches_frozen_resolver,
        "note": note,
        "arms": reports,
        "selected": leader["name"],
        "separated_from_leader": [
            arm["name"] for arm in reports
            if arm["vs_leader"] and arm["vs_leader"]["mcnemar_p"] < args.alpha
        ],
        "tied_with_leader": [arm["name"] for arm in tied],
    }
    out = Path(args.out)
    out = out if out.is_absolute() else REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
