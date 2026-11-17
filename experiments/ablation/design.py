"""Factor levels of the ablation design and the tree command each cell runs.

The executor ("the resolver") is NOT identical to LaSP's ``parse``: the thesis
resolver standardises every relation score and squashes it with a sigmoid
(``zscore_sigmoid``) where LaSP applies a softmax. Run with LaSP's arithmetic
(``softmax``, sum binding, no rank, no colour) on LaSP's features it reproduces
LaSP exactly -- tests/test_executor_equivalence.py checks this -- so the
executor factor below isolates that one change.

What each level holds fixed:

* parser: the parses file. Everything else is identical, including lowering
  against the extended vocabulary (the student's own lowering), so the parser
  factor changes the model only.
* representation: which encoders build the feature tensor and which predicate
  table the resolver accepts. ``lasp`` is upstream LaSP's table and encoders
  (the baseline tree): no colour, isolation, thesis wall encoder,
  smallest/lowest, or ordinal rank. Programs are not re-lowered, so a
  predicate the LaSP table lacks is dropped at execution, as in LaSP.
* executor: per-relation normalisation, thesis vs LaSP. Colour composition
  (product) and rank stay with the representation they belong to.

The (lasp, lasp) cell is upstream LaSP scoring, run by the baseline tree's own
``parse``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

from grounding.resolution.config import BASELINE_RESOLVER, FROZEN_RESOLVER
from grounding.resolution.harness import feature_relpath

__all__ = ["PARSERS", "TEACHER_PARSERS", "REPRESENTATION_LEVELS",
           "EXECUTOR_LEVELS", "FULL", "FACTORS", "Cell", "cells",
           "cell_command", "default_parses", "teacher_parse_command"]


#: parser level -> default parses file (test split). The student is the
#: maintained run; the teachers are what ``parse_split.py --teacher`` writes.
PARSERS = {
    "student": "data/parses/test/nr3d_test_sft_v4_ext_random_size_matched__seed2.jsonl",
    "teacher_qwen": "data/parses/test/nr3d_test_teacher_qwen_v4_ext.jsonl",
    "teacher_gpt4o": "data/parses/test/nr3d_test_teacher_gpt4o_v4_ext.jsonl",
}
TEACHER_PARSERS = {"teacher_qwen": "qwen", "teacher_gpt4o": "gpt4o"}

#: representation level -> the execution tree whose encoders build features.
REPRESENTATION_LEVELS = {"extended": "extended", "lasp": "baseline"}

#: executor level -> per-relation normalisation.
EXECUTOR_LEVELS = {"thesis": FROZEN_RESOLVER["norm_mode"],
                   "lasp": BASELINE_RESOLVER["norm_mode"]}

#: The full system's level of every factor, in factor order.
FULL = ("student", "extended", "thesis")
FACTORS = ("parser", "representation", "executor")


@dataclass(frozen=True)
class Cell:
    parser: str
    representation: str
    executor: str

    @property
    def levels(self) -> tuple[str, str, str]:
        return (self.parser, self.representation, self.executor)

    @property
    def name(self) -> str:
        return "__".join(self.levels)

    @property
    def tree(self) -> str:
        return REPRESENTATION_LEVELS[self.representation]

    def level(self, factor: str) -> str:
        return self.levels[FACTORS.index(factor)]


def cells(parsers, representations, executors) -> list[Cell]:
    for name, allowed in (("parser", PARSERS), ("representation", REPRESENTATION_LEVELS),
                          ("executor", EXECUTOR_LEVELS)):
        levels = {"parser": parsers, "representation": representations,
                  "executor": executors}[name]
        unknown = set(levels) - set(allowed)
        if unknown:
            raise ValueError(f"unknown {name} level(s): {sorted(unknown)}")
    return [Cell(*levels) for levels in product(parsers, representations, executors)]


def cell_command(cell: Cell, parses: Path, split: str, totals: str,
                 dump: Path) -> tuple[str, list[str]]:
    """(tree, evaluator args) for one cell; run with harness.run_in_tree."""
    args = [
        "--features_path", feature_relpath(cell.tree, split),
        "--top_k", "5",
        "--threshold", "0.9",
        "--label_type", "gt",
        "--split", split,
        "--totals", totals,
        "--parses", str(Path(parses).resolve()),
        "--dump", str(Path(dump).resolve()),
    ]
    norm_mode = EXECUTOR_LEVELS[cell.executor]
    if cell.tree == "extended":
        # Passed, not inherited, so the record says what ran. Colour and rank
        # are representation features and stay on at both executor levels.
        args += ["--norm_mode", norm_mode,
                 "--color_mode", FROZEN_RESOLVER["color_mode"], "--use_rank"]
    else:
        # The aligned top-k tie-break the extended tree always uses.
        args += ["--tie_break", "legacy"]
        if cell.executor == "thesis":
            args += ["--executor", "thesis", "--norm_mode", norm_mode]
        else:
            args += ["--executor", "lasp"]
    return cell.tree, args


def default_parses(parser: str) -> str:
    return PARSERS[parser]


def teacher_parse_command(parser: str, rows: str = "data/corpora/nr3d_test.csv",
                          out: str | None = None) -> list[str]:
    """The parse_split invocation (training environment) for a teacher arm."""
    cmd = ["python", "-m", "experiments.parser_adaptation.parse_split",
           "--representation", "extended", "--teacher", TEACHER_PARSERS[parser],
           "--rows", rows]
    if out:
        cmd += ["--out", out]
    return cmd
