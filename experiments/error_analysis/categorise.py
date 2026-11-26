#!/usr/bin/env python3
"""Categorise the rows the full system still grounds wrongly.

Runs the shipped resolver on one parse file, and gives every failure one
primary category in pipeline order (parser -> category -> conditions ->
combination); see grounding.resolution.failure_analysis for the definitions.
Writes a JSON report and a CSV with one line per failure for manual reading.

  python -m experiments.error_analysis.categorise \\
      --parses data/parses/dev/nr3d_dev_student_mlxadam_seed2.jsonl

Use dev parses to decide what to fix. Test parses need --confirm-on-test and
are only for describing the final system's errors.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

from grounding.data.splits import VIEW_DEPENDENT_TOKENS
from grounding.paths import REPO_ROOT
from grounding.resolution import engine, tracing
from grounding.resolution.failure_analysis import (
    CATEGORIES, EVIDENCE_CATEGORIES, classify, colour_words, dropped_cues)
from grounding.resolution.stage2 import (_EXEC_ERRORS, bind_multiplicity,
                                         guard_split, load_tree, row_key)
from grounding.representation.features import get_all_colors

OUT = "artifacts/results/error_analysis"


class _PositionedBinds(list):
    """Each bind also records how many factors were traced before it."""

    def append(self, item):
        item["at"] = len(tracing._TRACE)
        super().append(item)


def traced(tree, program, concepts, valid):
    """(score or None, depth-0 factors, per depth-0 rel factor anchor multiplicity)."""
    binds = _PositionedBinds()
    trace = tracing.start_trace(binds=binds)
    try:
        score = tree.parse("", program, concepts, valid)
    except _EXEC_ERRORS:
        score = None
    finally:
        tracing.stop_trace()
    top_idx = [i for i, f in enumerate(trace) if f["depth"] == 0]
    rel_idx = [i for i in top_idx if trace[i]["kind"] == "rel"]
    mult = {}
    for b in binds:
        if b["depth"] != 0:
            continue
        # A relation's anchors are resolved, bound, then its factor is traced
        owner = next((i for i in rel_idx if i >= b["at"]), None)
        if owner is not None and owner not in mult:
            mult[owner] = bind_multiplicity(b)
    return score, [trace[i] for i in top_idx], [mult.get(i) for i in rel_idx]


def top_relations(program):
    """Top-level relations in program order, for the negation flag."""
    return list(program.get("relations") or [])


def rel_meta_for(factors, program, multiplicities):
    """Negation and anchor multiplicity per depth-0 rel factor."""
    relations, i, out = top_relations(program), 0, []
    for f, mult in zip(factors, multiplicities):
        name = f["name"].split("|")[0]
        while i < len(relations) and relations[i].get("relation_name") != name:
            i += 1
        rel = relations[i] if i < len(relations) else {}
        i += 1
        out.append({"negated": rel.get("negative") is True, "multiplicity": mult})
    return out


def view_dependent(sentence):
    tokens = (sentence or "").lower().replace(",", " ").replace(".", " ").split()
    return any(t in VIEW_DEPENDENT_TOKENS for t in tokens)


def load_all_rows(parses):
    path = REPO_ROOT / parses
    if not path.exists():
        sys.exit(f"no parses at {parses}")
    return [json.loads(line) for line in open(path) if line.strip()]


def pct(n, d):
    return round(100 * n / d, 2) if d else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parses", required=True)
    parser.add_argument("--tree", default="extended")
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--top", type=int, default=15,
                        help="entries per frequency list in the report")
    parser.add_argument("--confirm-on-test", action="store_true",
                        help="allow test parses, only to describe the final system")
    args = parser.parse_args()
    guard_split(args.parses, args.confirm_on_test)

    tree = load_tree(args.tree)
    rows = load_all_rows(args.parses)
    scenes = engine.Scenes()
    by_scan = defaultdict(list)
    for row in rows:
        by_scan[row["scan_id"]].append(row)

    records, stats = [], Counter()
    for n, (scan_id, group) in enumerate(sorted(by_scan.items()), 1):
        try:
            scene = scenes.load(scan_id)
        except (FileNotFoundError, KeyError):
            stats["scene_missing"] += len(group)
            continue
        labels = list(scene["inst_labels"])
        obj_ids = [int(i) for i in scene["obj_ids"]]
        programs = [r["json_obj"] for r in group if r.get("json_obj")]
        wanted = set()
        for program in programs:
            wanted |= set(tree.get_all_relations(program))
            wanted |= set(tree.get_all_categories(program))
            wanted |= {f"color:{c}" for c in get_all_colors(program)}
        concepts = (tree.get_all_features(scan_id, sorted(wanted), scenes, "gt")
                    if programs else {})
        valid = torch.arange(len(scene["pred_locs"])).to(tree.device)

        for row in group:
            target_id, sentence = int(row["target_id"]), row.get("sentence") or row.get("utterance", "")
            if target_id not in obj_ids:
                stats["target_not_candidate"] += 1
                continue
            target = obj_ids.index(target_id)
            n_same = labels.count(labels[target])
            rec = {"key": row_key(row), "scan_id": scan_id, "target_id": target_id,
                   "sentence": sentence, "target_label": labels[target],
                   "n_same_class": n_same, "hard": n_same > 2,
                   "view_dependent": view_dependent(sentence),
                   "colour_words": colour_words(sentence)}
            program = row.get("json_obj")
            if not program:
                rec.update(category="no_program", correct=False, factors=[], against=[],
                           dropped_cues=[])
                records.append(rec)
                continue
            rec["program"] = program
            rec["dropped_cues"] = dropped_cues(sentence, program)
            score, top, mults = traced(tree, program, concepts, valid)
            if score is None:
                rec.update(category="execution_error", correct=False, factors=[], against=[])
                records.append(rec)
                continue
            pred = int(torch.argmax(score))
            rec["pred_id"], rec["pred_label"] = obj_ids[pred], labels[pred]
            rec["rel_names"] = [f["name"] if f["kind"] == "rel" else "colour"
                                for f in top if f["kind"] in ("rel", "color")]
            rec["correct"] = pred == target
            if rec["correct"]:
                records.append(rec)
                continue
            rels = [f for f in top if f["kind"] == "rel"]
            rec.update(classify(target=target, pred=pred, labels=labels, score=score,
                                top=top, rel_meta=rel_meta_for(rels, program, mults)))
            records.append(rec)
        scenes.drop(scan_id)
        if n % 50 == 0:
            print(f"  [{n}/{len(by_scan)}] scans", flush=True)

    report = summarise(records, stats, args)
    out_dir = REPO_ROOT / args.out / Path(args.parses).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_failures(out_dir / "failures.csv", records)
    print_summary(report)
    print(f"\nwrote {out_dir / 'report.json'} and {out_dir / 'failures.csv'}")
    return 0


def summarise(records, stats, args):
    n = len(records)
    fails = [r for r in records if not r["correct"]]
    report = {"parses": args.parses, "tree": args.tree, "n_rows": n,
              "n_correct": n - len(fails), "accuracy": pct(n - len(fails), n),
              "n_failures": len(fails), "excluded": dict(stats), "categories": {}}

    for cat in CATEGORIES:
        sub = [r for r in fails if r["category"] == cat]
        report["categories"][cat] = {
            "n": len(sub), "pct_of_failures": pct(len(sub), len(fails)),
            "pct_of_rows": pct(len(sub), n),
            "hard": sum(r["hard"] for r in sub),
            "view_dependent": sum(r["view_dependent"] for r in sub),
        }

    def top(counter):
        return [[k, v] for k, v in counter.most_common(args.top)]

    # Class errors: which head/label pairs
    report["wrong_head_pairs"] = top(Counter(
        f"{r['head']} -> target {r['target_label']} (won: {r['pred_label']})"
        for r in fails if r["category"] == "wrong_head"))
    report["other_class_pairs"] = top(Counter(
        f"{r['target_label']} lost to {r['pred_label']}"
        for r in fails if r["category"] == "other_class"))

    # Underspecified: did the sentence name something the program lacks?
    under = [r for r in fails if r["category"] == "underspecified"]
    report["underspecified"] = {
        "n": len(under),
        "sentence_names_unencoded_cue": sum(bool(r["dropped_cues"]) for r in under),
        "sentence_names_colour": sum(bool(r["colour_words"]) for r in under),
        "top_unencoded_cues": top(Counter(c for r in under for c in r["dropped_cues"])),
    }

    # Evidence failures: which conditions point at the wrong object
    ev = [r for r in fails if r["category"] in EVIDENCE_CATEGORIES]
    tag_counts = Counter(t for r in ev for f in r["factors"]
                         if f["prefers"] == "winner" for t in f["tags"])
    report["evidence"] = {
        "n": len(ev),
        "against_factor_tags": dict(tag_counts),
        "rows_with_decisive_factor": sum(any(f.get("decisive") for f in r["factors"])
                                         for r in ev),
    }

    # Per predicate: how often it occurs, and how often it sinks the target
    occ, occ_fail = Counter(), Counter()
    against, decisive = Counter(), Counter()
    for r in records:
        for name in r.get("rel_names", []):
            occ[name] += 1
            occ_fail[name] += not r["correct"]
    for r in ev:
        for f in r["factors"]:
            key = f["name"] if f["kind"] == "rel" else "colour"
            if f["prefers"] == "winner":
                against[key] += 1
                decisive[key] += bool(f.get("decisive"))
    names = sorted(set(occ) | set(against), key=lambda k: (-against[k], -occ[k], k))
    report["predicates"] = {
        name: {"rows": occ[name], "failed_rows": occ_fail[name],
               "against_target": against[name], "decisive": decisive[name],
               "decisive_per_100_rows": pct(decisive[name], occ[name])}
        for name in names}

    # Parser omissions: cues named in the sentence but not in the program
    succ = [r for r in records if r["correct"]]
    drop_f = Counter(c for r in fails for c in r.get("dropped_cues", []))
    drop_s = Counter(c for r in succ for c in r.get("dropped_cues", []))
    report["unencoded_cues"] = {
        "failures_with_any": sum(bool(r.get("dropped_cues")) for r in fails),
        "successes_with_any": sum(bool(r.get("dropped_cues")) for r in succ),
        "by_cue": {c: {"in_failures": drop_f[c], "in_successes": drop_s[c]}
                   for c, _ in (drop_f + drop_s).most_common(args.top)},
    }

    # Easy/hard and view-dependent accuracy, for context
    def acc(sub):
        return {"n": len(sub), "accuracy": pct(sum(r["correct"] for r in sub), len(sub))}
    report["splits"] = {
        "easy": acc([r for r in records if not r["hard"]]),
        "hard": acc([r for r in records if r["hard"]]),
        "view_independent": acc([r for r in records if not r["view_dependent"]]),
        "view_dependent": acc([r for r in records if r["view_dependent"]]),
    }
    return report


def write_failures(path, records):
    cols = ["key", "scan_id", "target_id", "pred_id", "category", "target_label",
            "pred_label", "head", "n_same_class", "hard", "view_dependent", "sentence",
            "against", "decisive", "against_tags", "factors", "dropped_cues",
            "colour_words", "program"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in records:
            if r["correct"]:
                continue
            facs = r.get("factors", [])
            w.writerow({
                **{c: r.get(c, "") for c in cols},
                "against": "; ".join(r.get("against", [])),
                "decisive": "; ".join(f["name"] for f in facs if f.get("decisive")),
                "against_tags": "; ".join(sorted({t for f in facs if f["prefers"] == "winner"
                                                   for t in f["tags"]})),
                "factors": "; ".join(f"{f['name']} t={f['target']:.3g} w={f['winner']:.3g}"
                                     for f in facs),
                "dropped_cues": "; ".join(r.get("dropped_cues", [])),
                "colour_words": "; ".join(r.get("colour_words", [])),
                "program": json.dumps(r.get("program"), ensure_ascii=False),
            })


def print_summary(report):
    print(f"\n{report['n_rows']} rows, accuracy {report['accuracy']}%, "
          f"{report['n_failures']} failures  (excluded: {report['excluded']})")
    print(f"\n  {'category':<20}{'n':>6}{'% fail':>9}{'% rows':>9}{'hard':>7}{'view-dep':>10}")
    for cat, v in report["categories"].items():
        print(f"  {cat:<20}{v['n']:>6}{v['pct_of_failures']:>9}{v['pct_of_rows']:>9}"
              f"{v['hard']:>7}{v['view_dependent']:>10}")
    ev = report["evidence"]
    print(f"\n  evidence failures: {ev['n']}, with a decisive factor: "
          f"{ev['rows_with_decisive_factor']}, tags on wrong factors: {ev['against_factor_tags']}")
    print("\n  predicates (rows / against target / decisive / decisive per 100 rows):")
    for name, v in list(report["predicates"].items())[:12]:
        print(f"    {name:<20}{v['rows']:>6}{v['against_target']:>6}{v['decisive']:>6}"
              f"{v['decisive_per_100_rows']:>8}")


if __name__ == "__main__":
    raise SystemExit(main())
