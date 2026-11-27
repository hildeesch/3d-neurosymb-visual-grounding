#!/usr/bin/env python3
"""Component ablations and factor interactions on one evaluation split.

    plan     list the cells, their tree commands, and any teacher parse
             still to generate (parse_split.py runs in the training env)
    run      build features per parser arm, score every cell, then analyse
    analyse  ablation table and factorial effects from the scored cells

Factors (full system level first):

    parser          student | student_coder | student_from_qwen_coder |
                    student_coder_from_qwen_coder | teacher_qwen |
                    teacher_qwen_coder | teacher_gpt4o (opt-in)
    representation  extended | lasp
    executor        thesis | lasp

The student arms are the teacher x student distillation grid (Qwen3-1.7B and
Qwen2.5-Coder-1.5B, each trained on Qwen3-30B-A3B and on Qwen3-Coder-30B-A3B
rollouts); see experiments/parser_adaptation/teacher_student_grid.py. The
analysis runs three families of factorial designs, each crossed with the
representation and executor factors where both levels were scored:

    grid          teacher[qwen|qwen_coder] x student[qwen3|coder] over the four
                  trained students: does the code teacher help, does the code
                  student help, and does either depend on the other
    distillation  parser[student arm | its own teacher, zero-shot], per pair
    teachers      parser[teacher_qwen | other teacher], zero-shot

Interaction p-values are Holm-adjusted within each family.

Every parser arm needs features built from ITS parses file, and the feature
cache holds one tensor per tree and split, so ``run`` scores one parser arm
completely before building the next. A parses file must cover the same rows
as the others: cells are compared expression by expression.

    python -m experiments.ablation plan
    python -m experiments.ablation run        # the whole local grid
    python -m experiments.ablation run --parsers student teacher_qwen
    python -m experiments.ablation analyse

Every parser arm is the tree parser (grammar-constrained). ``--no-grammar``
scores every arm's unconstrained parses instead -- the grammar ablation -- and
GPT-4o is always unconstrained. Results go to
``artifacts/results/ablation/<split>_tree_v1[_nogrammar]``, beside the v4
parser's results, which are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from experiments.ablation import design as D
from experiments.ablation import stats as S
from grounding.parsing.tree import DEFAULT_PROMPT
from grounding.paths import REPO_ROOT, RESULTS, resolve_under_root
from grounding.resolution.artifacts import (git_commit, parse_scores,
                                            relative_path as _rel, sha256)
from grounding.resolution.evaluate import TOTALS_MODES, row_key
from grounding.resolution.harness import (DATASET_SPLITS, SPLIT_NAMES,
                                          dataset_split, feature_relpath,
                                          run_in_tree, tree_dir)

MANIFEST = "cells.json"


# --- inputs --------------------------------------------------------------

def _constrained(args) -> bool:
    return not getattr(args, "no_grammar", False)


def parses_paths(args) -> dict[str, Path]:
    paths = {name: D.parses_for(name, _constrained(args)) for name in args.parsers}
    for item in args.parses or []:
        name, _, path = item.partition("=")
        if name not in D.PARSERS or not path:
            sys.exit(f"--parses takes NAME=PATH with NAME in {sorted(D.PARSERS)}")
        paths[name] = path
    return {name: resolve_under_root(paths[name]).resolve() for name in args.parsers}


def population(parses: Path) -> list[int]:
    keys = []
    with parses.open() as source:
        for line in source:
            if line.strip():
                keys.append(int(row_key(json.loads(line))))
    if len(set(keys)) != len(keys):
        sys.exit(f"{_rel(parses)} repeats row keys; one row per expression")
    return sorted(keys)


def missing_teacher_commands(paths: dict[str, Path],
                             constrained: bool = True) -> list[str]:
    lines = []
    for name, path in paths.items():
        if not path.exists() and name in D.TEACHER_PARSERS:
            lines.append(shlex.join(D.teacher_parse_command(
                name, out=_rel(path), constrained=constrained)))
    return lines


def missing_student_hints(paths: dict[str, Path]) -> list[str]:
    return [f"{name}: {D.student_parse_hint(name)}" for name, path in paths.items()
            if not path.exists() and name in D.STUDENT_PARSERS]


def check_inputs(paths: dict[str, Path], requested_split: str,
                 constrained: bool = True) -> tuple[str, list[int]]:
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        msg = [f"missing parses for: {', '.join(missing)}"]
        teacher = missing_teacher_commands(paths, constrained)
        if teacher:
            msg += ["generate the teacher arms in the training environment:",
                    *[f"  {line}" for line in teacher]]
        student = missing_student_hints(paths)
        if student:
            msg += ["train and parse the student arms (training environment):",
                    *[f"  {line}" for line in student]]
        sys.exit("\n".join(msg))
    splits = {name: dataset_split(path, requested_split) for name, path in paths.items()}
    if len(set(splits.values())) != 1:
        sys.exit(f"parser arms are on different splits: {splits}")
    keys = {name: population(path) for name, path in paths.items()}
    reference = next(iter(keys))
    for name, k in keys.items():
        if k != keys[reference]:
            only_a = len(set(keys[reference]) - set(k))
            only_b = len(set(k) - set(keys[reference]))
            sys.exit(f"{name} and {reference} cover different rows "
                     f"({only_b} only in {name}, {only_a} only in {reference}); "
                     "an interaction needs every cell on the same expressions")
    return next(iter(splits.values())), keys[reference]


# --- features ------------------------------------------------------------

def feature_tensor(tree: str, split: str) -> Path:
    return (tree_dir(tree) / feature_relpath(tree, split)).resolve()


def features_built_from(tree: str, split: str, parses: Path) -> bool:
    tensor = feature_tensor(tree, split)
    meta = Path(str(tensor) + ".meta.json")
    if not tensor.exists() or not meta.exists():
        return False
    recorded = json.loads(meta.read_text())
    return (recorded.get("parses", {}).get("sha256") == sha256(parses)
            and recorded.get("tensor", {}).get("sha256") == sha256(tensor))


def ensure_features(trees: set[str], split: str, parses: Path, rebuild: bool) -> None:
    stale = sorted(t for t in trees
                   if rebuild or not features_built_from(t, split, parses))
    if not stale:
        print(f"  features: reusing {', '.join(sorted(trees))} tensors built "
              f"from {parses.name}")
        return
    cmd = [sys.executable, str(REPO_ROOT / "scripts/evaluation/build_features.py"),
           "--parses", str(parses), "--split", split]
    for tree in stale:
        cmd += ["--representation", tree]
    print(f"  features: building {', '.join(stale)} from {parses.name}", flush=True)
    if subprocess.run(cmd, cwd=REPO_ROOT).returncode != 0:
        sys.exit(f"feature build failed: {shlex.join(cmd)}")
    for tree in stale:
        if not features_built_from(tree, split, parses):
            sys.exit(f"{tree} features do not record {parses.name} after the build")


# --- run -----------------------------------------------------------------

def load_manifest(out_dir: Path) -> dict:
    path = out_dir / MANIFEST
    return json.loads(path.read_text()) if path.exists() else {"cells": {}}


def write_manifest(out_dir: Path, manifest: dict) -> None:
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")


def score_cell(cell: D.Cell, parses: Path, split: str, totals: str,
               out_dir: Path) -> dict:
    dump = out_dir / "cells" / f"{cell.name}.jsonl"
    dump.parent.mkdir(parents=True, exist_ok=True)
    tree, args = D.cell_command(cell, parses, split, totals, dump)
    print(f"  {cell.name:<40} ({tree} tree)", flush=True)
    output = run_in_tree(tree, "src.eval.eval_nr3d", args)
    scores = parse_scores(output)
    if not scores or not dump.exists():
        sys.exit(f"cell {cell.name} produced no scores; rerun its command by hand:\n"
                 f"  (cd {tree_dir(tree)} && python -m src.eval.eval_nr3d "
                 f"{shlex.join(args)})")
    features = feature_tensor(tree, split)
    return {
        "cell": dict(zip(D.FACTORS, cell.levels)),
        "tree": tree,
        "args": args,
        "parses": {"path": _rel(parses), "sha256": sha256(parses)},
        "features": {"path": _rel(features), "sha256": sha256(features)},
        "dump": _rel(dump),
        "scores": {name: {"correct": c, "total": n} for name, (c, n) in scores.items()},
        "scored": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def cmd_run(args) -> int:
    paths = parses_paths(args)
    split, keys = check_inputs(paths, args.split, _constrained(args))
    totals = args.totals or ("official" if split == "test" else "parses")
    out_dir = out_directory(args, split)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(out_dir)
    if manifest.get("population") not in (None, keys):
        sys.exit(f"{_rel(out_dir / MANIFEST)} was scored on other rows; "
                 "pass a different --out-dir")
    if manifest.get("totals") not in (None, totals):
        sys.exit(f"{_rel(out_dir / MANIFEST)} used --totals {manifest['totals']}")
    manifest.update(split=split, totals=totals, population=keys)

    trees = {D.REPRESENTATION_LEVELS[r] for r in args.representations}
    for parser in args.parsers:
        parses = paths[parser]
        print(f"\n=== parser: {parser} ({_rel(parses)}) ===")
        ensure_features(trees, split, parses, args.rebuild_features)
        for cell in D.cells([parser], args.representations, args.executors):
            manifest["cells"][cell.name] = score_cell(cell, parses, split,
                                                      totals, out_dir)
            manifest.update(git_commit=git_commit(), command=" ".join(sys.argv))
            write_manifest(out_dir, manifest)

    print(f"\nwrote {_rel(out_dir / MANIFEST)}")
    return analyse(out_dir, args)


# --- analyse -------------------------------------------------------------

def scored_cells(manifest: dict) -> list[D.Cell]:
    return [D.Cell(*(entry["cell"][f] for f in D.FACTORS))
            for entry in manifest["cells"].values()]


#: Design families, in report order; Holm runs within each
FAMILIES = ("grid", "distillation", "teachers")


def designs(present: list[D.Cell]) -> list[tuple[str, str, dict, list[D.Cell]]]:
    """(family, name, design, cells) of every complete 2^k design in ``present``.

    Representation and executor join every design in which both their levels
    were scored. A design needs at least two factors and all 2^k cells.

    * grid: teacher x student over the four trained students (the full
      system's teacher and student are the +1 levels);
    * distillation: each student arm against its own teacher, zero-shot;
    * teachers: the thesis teacher against each other teacher, zero-shot.
    """
    have = {f: {c.level(f) for c in present} for f in D.FACTORS}
    full = dict(zip(D.FACTORS, D.FULL))
    crossed = {f: (full[f], "lasp") for f in ("representation", "executor")
               if {full[f], "lasp"} <= have[f]}

    candidates = []
    arms = D.grid_arms()
    if len(D.GRID_TEACHERS) == 2 and len(D.GRID_STUDENTS) == 2 \
            and set(arms.values()) <= have["parser"]:
        candidates.append(("grid", {"teacher": D.GRID_TEACHERS,
                                    "student": D.GRID_STUDENTS},
                           set(arms.values())))
    for name, (_, teacher) in D.STUDENT_PARSERS.items():
        zero_shot = D.teacher_arm(teacher)
        if name in have["parser"] and zero_shot in have["parser"]:
            candidates.append(("distillation", {"parser": (name, zero_shot)},
                               {name, zero_shot}))
    reference = D.teacher_arm(D.GRID_TEACHERS[0])
    for name in D.TEACHER_PARSERS:
        if name != reference and {name, reference} <= have["parser"]:
            candidates.append(("teachers", {"parser": (reference, name)},
                               {reference, name}))

    out = []
    for family, parser_design, parser_arms in candidates:
        design = {**parser_design, **crossed}
        if len(design) < 2:
            continue
        needed = [c for c in present
                  if c.parser in parser_arms
                  and all(c.level(f) in lv for f, lv in crossed.items())
                  and all(c.level(f) == full[f]
                          for f in ("representation", "executor") if f not in crossed)]
        if len(needed) == 2 ** len(design):
            name = " x ".join(f"{f}[{a}|{b}]" for f, (a, b) in design.items())
            out.append((family, name, design, needed))
    return out


def grid_tables(table: list[dict]) -> list[dict]:
    """Accuracy per (student model, teacher model), one table per rep/executor."""
    out = []
    for rep_ex in sorted({(r["representation"], r["executor"]) for r in table},
                         key=lambda re: (re != D.FULL[1:], re)):
        rows = {r["parser"]: r["accuracy_pp"] for r in table
                if (r["representation"], r["executor"]) == rep_ex}
        teachers = [t for t in (*D.GRID_TEACHERS, "gpt4o")
                    if any(D.PARSER_ARMS[p].teacher == t for p in rows)]
        students = [D.ZERO_SHOT, *D.GRID_STUDENTS]
        matrix = {}
        for student in students:
            for teacher in teachers:
                arm = next((n for n, a in D.PARSER_ARMS.items()
                            if a.teacher == teacher
                            and (a.student or D.ZERO_SHOT) == student), None)
                if arm in rows:
                    matrix.setdefault(student, {})[teacher] = rows[arm]
        if any(s != D.ZERO_SHOT for s in matrix):
            out.append({"representation": rep_ex[0], "executor": rep_ex[1],
                        "teachers": teachers, "accuracy_pp": matrix})
    return out


def analyse(out_dir: Path, args) -> int:
    manifest = load_manifest(out_dir)
    if not manifest["cells"]:
        sys.exit(f"no scored cells in {_rel(out_dir)}; run first")
    keys = manifest["population"]
    present = scored_cells(manifest)
    y = {c: S.load_correct(REPO_ROOT / manifest["cells"][c.name]["dump"], keys)
         for c in present}
    kw = {"n_boot": args.n_boot, "seed": args.seed}

    table, warnings = [], []
    for cell in present:
        entry = manifest["cells"][cell.name]
        overall = entry["scores"].get("overall", {})
        correct = int(y[cell].sum())
        if manifest["totals"] == "official" and overall.get("correct") != correct:
            warnings.append(f"{cell.name}: tree reports {overall.get('correct')} "
                            f"correct, dump has {correct}")
        table.append({"cell": cell.name, **entry["cell"], "correct": correct,
                      "n": len(keys), "accuracy_pp": 100 * correct / len(keys),
                      "strata": entry["scores"]})

    full = D.Cell(*D.FULL)
    ablations = []
    if full in y:
        for cell in present:
            if cell == full:
                continue
            swapped = [f for f in D.FACTORS if cell.level(f) != full.level(f)]
            ablations.append({"cell": cell.name, "swapped": swapped,
                              "label": " + ".join(f"{f}={cell.level(f)}" for f in swapped),
                              **S.simple_contrast(y[full], y[cell], **kw)})
        ablations.sort(key=lambda a: (len(a["swapped"]), a["cell"]))
        for a in ablations:
            # full - ablated: positive means the full system's level is better
            a["delta_full_minus_ablated_pp"] = a.pop("estimate_pp")

    factorial = []
    for family, name, design, needed in designs(present):
        effects = S.factorial_effects(design, {c: y[c] for c in needed}, **kw)
        factorial.append({"family": family, "design": name,
                          "levels": {f: {"full": a, "ablated": b}
                                     for f, (a, b) in design.items()},
                          "effects": effects})
    for family in FAMILIES:
        interactions = [e for d in factorial if d["family"] == family
                        for e in d["effects"] if e["order"] >= 2]
        for e, p in zip(interactions, S.holm([e["p_exact"] for e in interactions])):
            e["p_holm"] = p

    payload = {
        "producer": "experiments/ablation/cli.py",
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "split": manifest["split"], "totals": manifest["totals"],
        "n_expressions": len(keys),
        "conventions": {
            "scale": "accuracy, percentage points",
            "sign": "+1 = the first level of each design factor (the full "
                    "system's; in a distillation design, the student); ablation "
                    "deltas are full - ablated",
            "missing_rows": "rows absent from a dump (null parse, executor error) "
                            "count as misses, as under the official denominator",
            "test": S.__doc__.strip().splitlines()[0],
            "p_exact": "two-sided exact sign-flip test (= exact McNemar for two cells)",
            "ci95_pp": f"percentile bootstrap over expressions, {args.n_boot} "
                       f"resamples, seed {args.seed}",
            "p_holm": "Holm-adjusted over the interactions (order >= 2) of each "
                      "design family (grid, distillation, teachers)",
            "grid": "teacher and student are the model behind a parser arm; "
                    "+1 = the thesis teacher (qwen) and student (qwen3)",
        },
        "parsers": {c.parser: {"label": D.PARSER_ARMS[c.parser].label,
                               "teacher": c.level("teacher"),
                               "student": c.level("student")}
                    for c in present},
        "cells": table,
        "grid": grid_tables(table),
        "ablations": ablations,
        "factorial": factorial,
        "warnings": warnings,
    }
    (out_dir / "ablation.json").write_text(json.dumps(payload, indent=2) + "\n")
    report = markdown(payload)
    (out_dir / "ablation.md").write_text(report)
    print(report)
    for w in warnings:
        print(f"WARNING {w}")
    print(f"wrote {_rel(out_dir / 'ablation.json')} and ablation.md")
    return 0


def _p(p: float) -> str:
    return f"{p:.2g}" if p >= 1e-4 else f"{p:.1e}"


def markdown(payload: dict) -> str:
    lines = [f"# Ablation and interaction tests ({payload['split']}, "
             f"n={payload['n_expressions']}, totals={payload['totals']})", "",
             "## Parsers", "", "| parser | model |", "|---|---|"]
    for name, info in payload.get("parsers", {}).items():
        lines.append(f"| {name} | {info['label']} |")
    for g in payload.get("grid", []):
        lines += ["", f"## Teacher x student ({g['representation']} representation, "
                  f"{g['executor']} executor), accuracy %", "",
                  "| student \\ teacher | " + " | ".join(
                      D.model_label(t) for t in g["teachers"]) + " |",
                  "|---|" + "---:|" * len(g["teachers"])]
        for student, row in g["accuracy_pp"].items():
            label = ("none (zero-shot teacher)" if student == D.ZERO_SHOT
                     else D.model_label(student))
            lines.append(f"| {label} | " + " | ".join(
                f"{row[t]:.2f}" if t in row else "-" for t in g["teachers"]) + " |")
    lines += ["", "## Cells", "",
             "| parser | representation | executor | acc % | easy | hard | VD | VI |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in payload["cells"]:
        strata = []
        for name in ("easy", "hard", "view_dependent", "view_independent"):
            s = row["strata"].get(name)
            strata.append(f"{100 * s['correct'] / s['total']:.2f}"
                          if s and s["total"] else "-")
        lines.append(f"| {row['parser']} | {row['representation']} | {row['executor']} "
                     f"| {row['accuracy_pp']:.2f} | " + " | ".join(strata) + " |")
    if payload["ablations"]:
        lines += ["", "## Ablations (full system minus ablated cell)", "",
                  "| ablated | Δ pp | 95% CI | wins | losses | p |",
                  "|---|---:|---:|---:|---:|---:|"]
        for a in payload["ablations"]:
            lo, hi = a["ci95_pp"]
            lines.append(f"| {a['label']} | {a['delta_full_minus_ablated_pp']:+.2f} "
                         f"| [{lo:+.2f}, {hi:+.2f}] | {a['wins']} | {a['losses']} "
                         f"| {_p(a['p_exact'])} |")
        lines += ["", "wins: expressions only the full system gets right; "
                  "losses: only the ablated cell."]
    for d in payload["factorial"]:
        lines += ["", f"## Factorial ({d['family']}): {d['design']}", "",
                  "| effect | estimate pp | 95% CI | p | p (Holm) |",
                  "|---|---:|---:|---:|---:|"]
        for e in d["effects"]:
            lo, hi = e["ci95_pp"]
            holm = _p(e["p_holm"]) if "p_holm" in e else "-"
            lines.append(f"| {e['effect']} | {e['estimate_pp']:+.2f} "
                         f"| [{lo:+.2f}, {hi:+.2f}] | {_p(e['p_exact'])} | {holm} |")
    lines += ["", "In each design name the first level of a factor is +1 (the "
              "full system's level; in a distillation design, the student). "
              "Main effects average the +1 level's gain over the other "
              "factors' levels. A positive interaction A x B means A gains more "
              "when B is at its +1 level. In the grid, +1 is the thesis "
              "teacher (qwen) and student (qwen3), so a negative teacher or "
              "student effect means the code model is better. Holm adjusts "
              "within each family.", ""]
    return "\n".join(lines)


def cmd_analyse(args) -> int:
    split = args.split if args.split != "auto" else "test"
    return analyse(out_directory(args, split), args)


# --- plan ----------------------------------------------------------------

def cmd_plan(args) -> int:
    paths = parses_paths(args)
    split = args.split if args.split != "auto" else "test"
    totals = args.totals or ("official" if split == "test" else "parses")
    out_dir = out_directory(args, split)
    print(f"split {split}, totals {totals}, results -> {_rel(out_dir)}\n")
    for name, path in paths.items():
        state = "ok" if path.exists() else "MISSING"
        print(f"  {name:<30} {D.PARSER_ARMS[name].label:<55} {_rel(path)}  [{state}]")
    teacher = missing_teacher_commands(paths, _constrained(args))
    if teacher:
        print("\nteacher parses to generate (training environment):")
        for line in teacher:
            print(f"  {line}")
    student = missing_student_hints(paths)
    if student:
        print("\nstudent arms to train and parse (training environment):")
        for line in student:
            print(f"  {line}")
    print("\ncells (run scores them; parser arms need features rebuilt from "
          "their own parses, and run leaves the cache on the last arm's):")
    for cell in D.cells(args.parsers, args.representations, args.executors):
        tree, cell_args = D.cell_command(cell, paths[cell.parser], split, totals,
                                         out_dir / "cells" / f"{cell.name}.jsonl")
        module = ("grounding.resolution.cli" if tree == "extended"
                  else "src.eval.eval_nr3d")
        print(f"  {cell.name}\n    (cd {_rel(tree_dir(tree))} && python -m {module} "
              f"{shlex.join(cell_args)})")
    return 0


# --- arguments -----------------------------------------------------------

def out_directory(args, split: str) -> Path:
    name = f"{split}_{DEFAULT_PROMPT}" + ("" if _constrained(args) else "_nogrammar")
    out = args.out_dir or (RESULTS / "ablation" / name)
    return resolve_under_root(out).resolve()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--parsers", nargs="+", default=list(D.DEFAULT_PARSERS),
                       choices=list(D.PARSERS),
                       help="default: the local teacher x student grid "
                            "(every arm but teacher_gpt4o)")
        p.add_argument("--representations", nargs="+",
                       default=list(D.REPRESENTATION_LEVELS),
                       choices=list(D.REPRESENTATION_LEVELS))
        p.add_argument("--executors", nargs="+", default=list(D.EXECUTOR_LEVELS),
                       choices=list(D.EXECUTOR_LEVELS))
        p.add_argument("--parses", action="append", metavar="NAME=PATH",
                       help="override a parser arm's parses file; repeatable")
        p.add_argument("--split", choices=["auto", *DATASET_SPLITS], default="auto")
        p.add_argument("--totals", choices=list(TOTALS_MODES), default=None,
                       help="default: official on test, parses otherwise")
        p.add_argument("--out-dir", type=Path, default=None,
                       help="default: artifacts/results/ablation/"
                            "<split>_tree_v1[_nogrammar]")
        p.add_argument("--n-boot", type=int, default=10_000)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--no-grammar", action="store_true",
                       help="score every arm's parses decoded without the "
                            "grammar (the grammar ablation)")

    p = sub.add_parser("plan", help="list cells and missing inputs")
    common(p)
    p.set_defaults(run=cmd_plan)
    p = sub.add_parser("run", help="build features and score every cell")
    common(p)
    p.add_argument("--rebuild-features", action="store_true",
                   help="rebuild even when the cached tensor records this parses file")
    p.set_defaults(run=cmd_run)
    p = sub.add_parser("analyse", help="tables and tests from scored cells")
    common(p)
    p.set_defaults(run=cmd_analyse)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
