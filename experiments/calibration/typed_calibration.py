#!/usr/bin/env python3
"""Typed calibration: per-condition mapping by semantic type, product fixed.

Each top-level condition is put in a group (exclusive / relative / absolute,
see grounding.resolution.condition_types) and each group gets its own mapping.
Colour can stay raw or be standardised among the category's candidates. The
arms recombine the traced raw scores, like `score_aggregation.py norm`; the
`shipped` arm rebuilds the engine's own prediction and the run exits non-zero
if any row disagrees.

Only top-level conditions are re-mapped. Anchors are still resolved with the
engine's calibration, as in `score_aggregation.py final`.

  python -m experiments.calibration.typed_calibration
  python -m experiments.calibration.typed_calibration \\
      --arm "my_arm=exclusive:zscore,absolute:raw,colour:category"
  python -m experiments.calibration.typed_calibration --arms shipped,typed \\
      --parses data/parses/test/<file>.jsonl --confirm-on-test

Default arms (every unnamed group/colour keeps the shipped mapping):
  shipped           everything zscore_sigmoid, raw colour (control)
  all_zscore        everything z-score + softmax (Table VII's z-score row)
  excl_softmax      exclusive -> z-score + softmax
  abs_raw           absolute  -> raw [0,1] score, anchor-mass normalised
  colour_category   colour    -> z-score + sigmoid among category members
  typed             excl_softmax + abs_raw
  typed_colour      typed + colour_category
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

from grounding.resolution import engine
from grounding.resolution.condition_types import (
    COLOUR_MAPPINGS, GROUPS, MAPPINGS, appearance_typed, calibrate,
    condition_group, match_relations, negate, traced_with_masses)
from grounding.resolution.config import DEFAULT_RESOLVER
from grounding.resolution.stage2 import (_EXEC_ERRORS, CGE2_PARSES, DIAGNOSTICS_OUT,
                                         STAGE2_OUT, accuracy_table, guard_split,
                                         load_rows, load_tree, predicted_id, row_key,
                                         scene_batches, write_report)

SHIPPED = DEFAULT_RESOLVER.norm_mode


def arm_spec(**changes):
    spec = {group: SHIPPED for group in GROUPS}
    spec["colour"] = "raw"
    spec.update(changes)
    return spec


DEFAULT_ARMS = {
    "shipped": arm_spec(),
    "all_zscore": arm_spec(exclusive="zscore", relative="zscore", absolute="zscore"),
    "excl_softmax": arm_spec(exclusive="zscore"),
    "abs_raw": arm_spec(absolute="raw"),
    "colour_category": arm_spec(colour="category"),
    "typed": arm_spec(exclusive="zscore", absolute="raw"),
    "typed_colour": arm_spec(exclusive="zscore", absolute="raw", colour="category"),
}


def parse_arm(text):
    """'name=group:mapping,...' -> (name, spec)."""
    name, _, body = text.partition("=")
    if not name or not body:
        sys.exit(f"--arm {text!r}: expected name=group:mapping,...")
    changes = {}
    for item in body.split(","):
        group, _, mapping = item.strip().partition(":")
        allowed = COLOUR_MAPPINGS if group == "colour" else MAPPINGS
        if group not in (*GROUPS, "colour") or mapping not in allowed:
            sys.exit(f"--arm {text!r}: bad item {item!r}; groups {GROUPS + ('colour',)}, "
                     f"mappings {MAPPINGS}, colour {COLOUR_MAPPINGS}")
        changes[group] = mapping
    return name.strip(), arm_spec(**changes)


def factor_affinity(factor, relation, mass, mapping, floor):
    vec = calibrate(mapping, factor["raw"], mass if mapping == "raw" else None,
                    floor=floor)
    if relation.get("negative") is True:
        vec = negate(vec, mapping, floor=floor)
    return vec


def row_strata(groups, has_colour):
    out = [f"has {g}" for g in GROUPS if g in groups]
    if has_colour:
        out.append("has colour")
    if set(groups) == {"relative"} and not has_colour:
        out.append("relative only")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parses", default=CGE2_PARSES)
    parser.add_argument("--tree", default="extended")
    parser.add_argument("--out", default=STAGE2_OUT)
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                        help="default arms to run; shipped is always run first")
    parser.add_argument("--arm", action="append", default=[],
                        help="extra arm: name=group:mapping,... (repeatable)")
    parser.add_argument("--raw-floor", type=float, default=1e-3,
                        help="lowest affinity a raw (absolute) condition can give")
    parser.add_argument("--confirm-on-test", action="store_true",
                        help="allow test parses, only to confirm a dev choice")
    args = parser.parse_args()
    guard_split(args.parses, args.confirm_on_test)
    if args.confirm_on_test and Path(args.out) == Path(STAGE2_OUT):
        args.out = DIAGNOSTICS_OUT

    names = [n.strip() for n in args.arms.split(",") if n.strip()]
    if unknown := set(names) - set(DEFAULT_ARMS):
        sys.exit(f"unknown --arms {sorted(unknown)}; define them with --arm")
    arms = {"shipped": DEFAULT_ARMS["shipped"]}
    arms.update({n: DEFAULT_ARMS[n] for n in names})
    for text in args.arm:
        name, spec = parse_arm(text)
        if name in arms:
            sys.exit(f"--arm {name!r} is already defined")
        arms[name] = spec

    tree = load_tree(args.tree)
    stats = {"skipped": 0, "unmatched": 0}
    results, strata = {}, defaultdict(list)
    coverage, mismatched = defaultdict(Counter), 0
    batches = scene_batches(tree, engine.Scenes(), load_rows(args.parses), stats)
    for group_rows, concepts, valid, obj_ids in batches:
        for row in group_rows:
            program = row["json_obj"]
            score, top, masses = traced_with_masses(tree, program, concepts, valid,
                                                    _EXEC_ERRORS)
            factors = [f for f in top if f["kind"] == "rel"]
            if score is None or not any(f["kind"] == "cat" for f in top):
                stats["skipped"] += 1
                continue
            try:
                relations = match_relations(factors, program)
            except ValueError:
                stats["unmatched"] += 1
                continue
            groups = [condition_group(f["name"], r) for f, r in zip(factors, relations)]
            for f, g in zip(factors, groups):
                coverage[g][f["name"]] += 1
            has_colour = any(f["kind"] == "color" for f in top)

            key, target = row_key(row), int(row["target_id"])
            results[key] = {}
            for arm, spec in arms.items():
                with torch.no_grad():
                    s = appearance_typed(top, DEFAULT_RESOLVER.color_mode, spec["colour"])
                    for f, r, m, g in zip(factors, relations, masses, groups):
                        s = s * factor_affinity(f, r, m, spec[g], args.raw_floor)
                pred = predicted_id(obj_ids, s)
                results[key][arm] = pred == target
                if arm == "shipped":
                    mismatched += pred != predicted_id(obj_ids, score)
            for stratum in row_strata(groups, has_colour):
                strata[stratum].append(key)

    arm_names = list(arms)
    print(f"\ncontrol: shipped rebuilt from the trace matches the engine on "
          f"{len(results) - mismatched}/{len(results)} rows")
    if mismatched:
        print(f"FAILED: {mismatched} rows differ, so the tables measure the "
              "capture, not the arms", file=sys.stderr)
    print("\ncondition groups (top-level factors):")
    for g in GROUPS:
        top_names = ", ".join(f"{n} {c}" for n, c in coverage[g].most_common(12))
        print(f"  {g:<10}{sum(coverage[g].values()):>6}  {top_names}")

    report = {"parses": args.parses, "tree": args.tree, "arms": arms,
              "raw_floor": args.raw_floor, "n_rows": len(results), **stats,
              "control": {"arm": "shipped", "rebuilt": len(results),
                          "agree": len(results) - mismatched},
              "coverage": {g: dict(coverage[g]) for g in GROUPS},
              "overall": {}, "strata": {}}
    accuracy_table("OVERALL (paired against shipped)", [("all rows", list(results))],
                   report["overall"], results=results, arms=arm_names, base="shipped")
    order = [f"has {g}" for g in GROUPS] + ["has colour", "relative only"]
    accuracy_table("BY CONDITION GROUP PRESENT",
                   [(s, strata[s]) for s in order if strata[s]],
                   report["strata"], results=results, arms=arm_names, base="shipped")
    write_report(args, "typed_calibration", report)
    return 1 if mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
