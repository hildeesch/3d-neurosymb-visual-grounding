"""Factor levels of the ablation design and the tree command each cell runs.

The executor ("the resolver") is NOT identical to LaSP's ``parse``: the thesis
resolver standardises every relation score and squashes it with a sigmoid
(``zscore_sigmoid``) where LaSP applies a softmax. Run with LaSP's arithmetic
(``softmax``, sum binding, no rank, no colour) on LaSP's features it reproduces
LaSP exactly -- tests/test_executor_equivalence.py checks this -- so the
executor factor below isolates that one change.

Parser arms are the distillation grid of the tree parser (experiments/
parser_adaptation/teacher_student_grid.py): every student model trained on
every teacher's winners, plus every teacher parsing zero-shot, all under the
tree prompt and grammar. ``parses_for(arm, constrained=False)`` is the same arm
decoded without the grammar (the grammar ablation); GPT-4o cannot decode under
a grammar, so its arm is always the unconstrained parse.

    student                        Qwen3-1.7B         <- Qwen3-30B-A3B (full system)
    student_coder                  Qwen2.5-Coder-1.5B <- Qwen3-30B-A3B
    student_from_qwen_coder        Qwen3-1.7B         <- Qwen3-Coder-30B-A3B
    student_coder_from_qwen_coder  Qwen2.5-Coder-1.5B <- Qwen3-Coder-30B-A3B
    teacher_qwen / teacher_qwen_coder / teacher_gpt4o   zero-shot teachers

Besides its own level, a cell answers ``level("teacher")`` (the model whose
rollouts trained the student, or that parsed zero-shot) and
``level("student")`` (the student model, ``none`` for a zero-shot teacher), so
the teacher x student grid is a factorial design of its own.

What each level holds fixed:

* parser: the parses file. Everything else is identical -- every arm's
  ``json_obj`` is ``to_executable`` of its tree -- so the parser factor
  changes the model only.
* representation: which encoders build the feature tensor and which predicate
  table the resolver accepts. ``lasp`` is upstream LaSP's table and encoders
  (the baseline tree): no colour, isolation, thesis wall encoder,
  smallest/lowest, or ordinal rank. Programs are not rewritten, so a
  relation the LaSP table lacks is dropped at execution, as in LaSP.
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

from experiments.parser_adaptation import teacher_student_grid as grid
from grounding.parsing.models import STUDENTS, TEACHERS as TEACHER_MODELS

__all__ = ["ParserArm", "PARSER_ARMS", "PARSERS", "TEACHER_PARSERS",
           "STUDENT_PARSERS", "DEFAULT_PARSERS", "GRID_TEACHERS", "GRID_STUDENTS",
           "ZERO_SHOT", "REPRESENTATION_LEVELS", "EXECUTOR_LEVELS", "FULL",
           "FACTORS", "PARSER_ATTRIBUTES", "Cell", "cells", "cell_command",
           "default_parses", "parses_for", "teacher_parse_command",
           "student_parse_hint",
           "grid_arms", "teacher_arm", "model_label"]


@dataclass(frozen=True)
class ParserArm:
    name: str
    teacher: str             # teacher model key: qwen | qwen_coder | gpt4o
    student: str | None      # student model key; None = the teacher, zero-shot
    label: str
    parses: str              # default parses file (test split)


#: ``level("student")`` of a zero-shot teacher arm
ZERO_SHOT = "none"

GRID_TEACHERS = tuple(TEACHER_MODELS)     # (qwen, qwen_coder): full level first
GRID_STUDENTS = tuple(STUDENTS)           # (qwen3, coder): full level first


def _arms() -> dict[str, ParserArm]:
    arms = {}
    for pair in grid.pairs():
        arms[pair.name] = ParserArm(pair.name, pair.teacher, pair.student,
                                    pair.label, pair.test_parses)
    for key, model in TEACHER_MODELS.items():
        name = grid.teacher_parser_name(key)
        arms[name] = ParserArm(name, key, None, f"{model.label} (zero-shot)",
                               grid.teacher_parses_path(key))
    arms["teacher_gpt4o"] = ParserArm("teacher_gpt4o", "gpt4o", None,
                                      "GPT-4o (zero-shot, no grammar)",
                                      grid.teacher_parses_path("gpt4o", False))
    return arms


#: parser level -> what it is. Students are the grid's test parses (the full
#: system's is the maintained run); teachers are what ``parse_split.py
#: --teacher`` writes.
PARSER_ARMS: dict[str, ParserArm] = _arms()
PARSERS = {name: arm.parses for name, arm in PARSER_ARMS.items()}
TEACHER_PARSERS = {name: arm.teacher for name, arm in PARSER_ARMS.items()
                   if arm.student is None}
STUDENT_PARSERS = {name: (arm.student, arm.teacher)
                   for name, arm in PARSER_ARMS.items() if arm.student is not None}
#: What ``run`` scores without --parsers: the whole local grid. GPT-4o is opt-in.
DEFAULT_PARSERS = [name for name in PARSERS if name != "teacher_gpt4o"]

#: Attributes of the parser level that designs can treat as factors
PARSER_ATTRIBUTES = ("teacher", "student")


def model_label(key: str) -> str:
    """Display name of a teacher or student model key."""
    if key in TEACHER_MODELS:
        return TEACHER_MODELS[key].label
    if key in STUDENTS:
        return STUDENTS[key].label
    return {"gpt4o": "GPT-4o"}.get(key, key)


def grid_arms() -> dict[tuple[str, str], str]:
    """(student, teacher) -> parser arm, for every trained pair."""
    return {(s, t): name for name, (s, t) in STUDENT_PARSERS.items()}


def teacher_arm(teacher: str) -> str | None:
    """The zero-shot parser arm of a teacher model."""
    return next((n for n, t in TEACHER_PARSERS.items() if t == teacher), None)

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
        if factor == "teacher":
            return PARSER_ARMS[self.parser].teacher
        if factor == "student":
            return PARSER_ARMS[self.parser].student or ZERO_SHOT
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


def parses_for(parser: str, constrained: bool = True) -> str:
    """Default test parses of an arm, with or without the grammar."""
    if parser in STUDENT_PARSERS:
        student, teacher = STUDENT_PARSERS[parser]
        return grid.Pair(student, teacher).parses(constrained)
    teacher = TEACHER_PARSERS[parser]
    # The OpenAI API has no grammar: GPT-4o's arm is always unconstrained
    return grid.teacher_parses_path(teacher, constrained and teacher != "gpt4o")


def student_parse_hint(parser: str) -> str:
    """Where a missing student arm's parses come from (training environment)."""
    student, teacher = STUDENT_PARSERS[parser]
    return ("python -m experiments.parser_adaptation.teacher_student_grid plan "
            f"--students {student} --teachers {teacher}   # then `run` the steps it lists")


def teacher_parse_command(parser: str, rows: str = "data/corpora/nr3d_test.csv",
                          out: str | None = None, constrained: bool = True) -> list[str]:
    """The parse_split invocation (training environment) for a teacher arm.

    Grammar-constrained unless ``constrained`` is False (``--no-grammar``);
    GPT-4o is always unconstrained.
    """
    teacher = TEACHER_PARSERS[parser]
    cmd = ["python", "-m", "experiments.parser_adaptation.parse_split",
           "--representation", "extended", "--teacher", teacher, "--rows", rows]
    if not constrained or teacher == "gpt4o":
        cmd.append("--no-grammar")
    if out:
        cmd += ["--out", out]
    return cmd
