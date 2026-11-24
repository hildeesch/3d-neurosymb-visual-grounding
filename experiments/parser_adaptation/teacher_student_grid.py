#!/usr/bin/env python3
"""Every (student, teacher) pair of the distillation grid: paths, status, commands.

Two teachers x two students (grounding.parsing.models):

    teachers  qwen        Qwen3-30B-A3B                  (thesis teacher)
              qwen_coder  Qwen3-Coder-30B-A3B-Instruct
    students  qwen3       Qwen3-1.7B                     (thesis student)
              coder       Qwen2.5-Coder-1.5B-Instruct

Each student is trained on each teacher's winners with the thesis recipe and
nothing else changed: the maintained arm (``sft_v4_ext_random_size_matched``,
3908 prompts x 2 rows), the shared LoRA YAML, 1000 updates, seeds 0-2, the seed
chosen on dev by select_adapter.py, then a greedy test parse. Together with the
two zero-shot teachers that gives the six parser arms of the ablation
(experiments/ablation): ``student`` (qwen3 <- qwen, the full system),
``student_coder``, ``student_from_qwen_coder``, ``student_coder_from_qwen_coder``,
``teacher_qwen`` and ``teacher_qwen_coder``.

The reference pair (qwen3 <- qwen) is the maintained student: its parses file
is used as it is and no step is re-run for it.

Steps, in order (``plan`` prints them with their status; ``run`` executes the
ones whose outputs are missing, in this environment, stopping at the first
failure):

    teacher  rollouts  R=8 rollouts on the frozen 8750 train prompts (llama-server
                       must serve that teacher's GGUF; training env)
             filter    filter_rollouts.py through the extended arm
             winners   build_winner_pool.py
             sets      build_uniform_sets.py --only random_size_matched
             zeroshot  zero-shot test parse (llama-server; training env)
    student  train     train_sft_arms.py, three seeds
             dev       score_checkpoint.py --split dev --n 0, per seed
             select    select_adapter.py over the three seeds
             parse     parse_split.py --style sft with the selected adapter

    python -m experiments.parser_adaptation.teacher_student_grid plan
    python -m experiments.parser_adaptation.teacher_student_grid run \\
        --teachers qwen_coder --steps filter winners sets
    python -m experiments.parser_adaptation.teacher_student_grid run \\
        --students coder --steps train dev select parse
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from grounding.parsing.models import (DEFAULT_STUDENT, DEFAULT_TEACHER,
                                      STUDENTS, TEACHERS)
from grounding.paths import REPO_ROOT

#: The maintained training arm: every pair is trained on it
ARM = "sft_v4_ext_random_size_matched"
SEEDS = (0, 1, 2)
#: generate_rollouts.load_rows prefix the frozen R=8 pool was drawn from
ROLLOUT_PROMPTS = 8750
TEST_ROWS = "data/corpora/nr3d_test.csv"

TEACHER_STEPS = ("rollouts", "filter", "winners", "sets", "zeroshot")
STUDENT_STEPS = ("train", "dev", "select", "parse")
STEPS = TEACHER_STEPS + STUDENT_STEPS

#: The maintained student's test parses (seed 2, selected on dev)
REFERENCE_PARSES = "data/parses/test/nr3d_test_sft_v4_ext_random_size_matched__seed2.jsonl"


def is_reference(student: str, teacher: str) -> bool:
    return student == DEFAULT_STUDENT and teacher == DEFAULT_TEACHER


def parser_name(student: str, teacher: str) -> str:
    """Ablation parser-arm name: ``student[_<s>][_from_<t>]``, thesis parts omitted."""
    name = "student"
    if student != DEFAULT_STUDENT:
        name += f"_{student}"
    if teacher != DEFAULT_TEACHER:
        name += f"_from_{teacher}"
    return name


def teacher_parser_name(teacher: str) -> str:
    return f"teacher_{teacher}"


# --- paths (repo-relative strings) ----------------------------------------

def rollouts_path(teacher: str) -> str:
    if teacher == DEFAULT_TEACHER:
        return "data/rollouts/nr3d_rollouts_fft_v4.jsonl"
    return f"data/rollouts/nr3d_rollouts_{teacher}_v4.jsonl"


def keepers_path(teacher: str) -> str:
    tag = "fft" if teacher == DEFAULT_TEACHER else teacher
    return f"data/distilled/nr3d_keepers_{tag}_v4_ext.jsonl"


def winners_path(teacher: str) -> str:
    tag = "fft" if teacher == DEFAULT_TEACHER else teacher
    return f"data/distilled/nr3d_winners_{tag}_v4_ext.jsonl"


def sets_path(teacher: str) -> str:
    if teacher == DEFAULT_TEACHER:
        return "data/training_sets_zscore_sigmoid"
    return f"data/training_sets_{teacher}"


def teacher_parses_path(teacher: str) -> str:
    """What ``parse_split --teacher <t>`` writes by default for the test rows."""
    return f"data/parses/test/nr3d_test_teacher_{teacher}_v4_ext.jsonl"


@dataclass(frozen=True)
class Pair:
    student: str
    teacher: str

    @property
    def tag(self) -> str:
        return f"{self.student}_from_{self.teacher}"

    @property
    def name(self) -> str:
        return parser_name(self.student, self.teacher)

    @property
    def reference(self) -> bool:
        return is_reference(self.student, self.teacher)

    @property
    def label(self) -> str:
        return f"{STUDENTS[self.student].label} <- {TEACHERS[self.teacher].label}"

    @property
    def model(self) -> str:
        return STUDENTS[self.student].hf

    @property
    def adapters(self) -> str:
        return "artifacts/adapters" if self.reference else f"artifacts/adapters_{self.tag}"

    @property
    def logs(self) -> str:
        return "runs/logs" if self.reference else f"runs/logs_{self.tag}"

    @property
    def dev_dir(self) -> str:
        return f"artifacts/results/parser_adaptation/dev/grid/{self.tag}"

    def run(self, seed: int) -> str:
        return f"{ARM}__seed{seed}"

    def dev_result(self, seed: int) -> str:
        # score_checkpoint's own naming (checkpoint_result_name)
        return f"{self.dev_dir}/{ARM.removeprefix('sft_v4_ext_')}_seed_{seed}.json"

    @property
    def selection(self) -> str:
        return f"{self.dev_dir}/selection.json"

    @property
    def test_parses(self) -> str:
        if self.reference:
            return REFERENCE_PARSES
        return f"data/parses/test/nr3d_test_{self.name}_v4_ext.jsonl"


def pairs(students=None, teachers=None) -> list[Pair]:
    return [Pair(s, t) for t in (teachers or TEACHERS) for s in (students or STUDENTS)]


# --- status ----------------------------------------------------------------

def _p(rel: str) -> Path:
    return REPO_ROOT / rel


def _has_rows(rel: str) -> bool:
    path = _p(rel)
    return path.exists() and path.stat().st_size > 0


def train_done(pair: Pair, seed: int) -> bool:
    log = _p(pair.logs) / f"{pair.run(seed)}.log"
    cfg = _p(pair.adapters) / pair.run(seed) / "training_config.json"
    return (log.exists() and "Saved final weights to" in log.read_text(errors="ignore")
            and cfg.exists())


def parses_complete(rel: str) -> bool:
    """A parse_split output whose sidecar records a finished run."""
    meta = Path(str(_p(rel)) + ".meta.json")
    if not _has_rows(rel) or not meta.exists():
        return False
    return "written" in json.loads(meta.read_text())


def selected_run(pair: Pair) -> str | None:
    path = _p(pair.selection)
    return json.loads(path.read_text()).get("selected") if path.exists() else None


# --- commands --------------------------------------------------------------

@dataclass
class Step:
    step: str
    who: str                      # teacher key or pair tag
    argv: list[str] | None        # None: nothing to run (blocked or manual)
    done: bool
    note: str = ""

    def line(self) -> str:
        return shlex.join(self.argv) if self.argv else ""


def _py(module: str, *args) -> list[str]:
    return ["python", "-m", f"experiments.parser_adaptation.{module}", *map(str, args)]


def llama_server_hint(teacher: str) -> str:
    t = TEACHERS[teacher]
    return (f"llama-server must serve {t.gguf_file} (hf download {t.gguf_repo} "
            f"--include '*Q4_K_M*' --local-dir models/; llama-server -m "
            f"models/{t.gguf_file} -ngl 99 --n-cpu-moe 16 -c 16384 -np 8 --port 8080)")


def teacher_steps(teacher: str) -> list[Step]:
    r, k, w, s = (rollouts_path(teacher), keepers_path(teacher),
                  winners_path(teacher), sets_path(teacher))
    train_set = f"{s}/{ARM}/train.jsonl"
    foreign = [] if teacher == DEFAULT_TEACHER else ["--foreign-pool"]
    # rollouts -> keepers -> winners -> training set: a step is done once its
    # output or any later one exists (the thesis teacher's training set may be
    # present without the intermediate pools)
    chain = [_has_rows(x) for x in (r, k, w, train_set)]
    done = [any(chain[i:]) for i in range(len(chain))]
    hint = llama_server_hint(teacher)
    if teacher == DEFAULT_TEACHER:
        hint = ("the frozen thesis pool; regenerating it replaces the thesis "
                "teacher's rollouts. " + hint)
    return [
        Step("rollouts", teacher,
             _py("generate_rollouts", "--teacher", teacher, "--backend", "llamacpp",
                 "--prompt", "v4", "--limit", ROLLOUT_PROMPTS, "--n-samples", 8,
                 "--temp", 0.7, "--max-tokens", 320, "--seed", 0, "--output", r),
             done[0], hint),
        Step("filter", teacher,
             _py("filter_rollouts", "--representation", "extended",
                 "--rollouts", r, "--output", k), done[1]),
        Step("winners", teacher,
             _py("build_winner_pool", "--keepers", k, "--output", w), done[2]),
        Step("sets", teacher,
             _py("build_uniform_sets", "--pool", w, "--out", s, "--only", ARM,
                 *foreign), done[3]),
        Step("zeroshot", teacher,
             _py("parse_split", "--representation", "extended", "--teacher", teacher,
                 "--rows", TEST_ROWS, "--out", teacher_parses_path(teacher)),
             parses_complete(teacher_parses_path(teacher)),
             llama_server_hint(teacher)),
    ]


def student_steps(pair: Pair, wandb: bool = False) -> list[Step]:
    if pair.reference:
        return [Step(step, pair.tag, None, True,
                     f"reference pair: the maintained student ({REFERENCE_PARSES})")
                for step in STUDENT_STEPS]
    # Relative paths: every command runs from the repository root
    train = _py("train_sft_arms", "--student", pair.student,
                "--sets", sets_path(pair.teacher),
                "--adapters", pair.adapters, "--logs", pair.logs,
                "--only", ARM, "--skip-done")
    if not wandb:
        train.append("--no-wandb")
    steps = [Step("train", pair.tag, train, all(train_done(pair, k) for k in SEEDS),
                  f"needs {sets_path(pair.teacher)}/{ARM}")]
    for seed in SEEDS:
        steps.append(Step(
            "dev", pair.tag,
            _py("score_checkpoint", "--split", "dev", "--n", 0, "--model", pair.model,
                "--adapter", f"{pair.adapters}/{pair.run(seed)}",
                "--out", pair.dev_result(seed)),
            _p(pair.dev_result(seed)).exists(), f"seed {seed}"))
    steps.append(Step(
        "select", pair.tag,
        _py("select_adapter", "--results", *[pair.dev_result(k) for k in SEEDS],
            "--out", pair.selection),
        _p(pair.selection).exists()))
    chosen = selected_run(pair)
    parse = (_py("parse_split", "--representation", "extended", "--style", "sft",
                 "--student", pair.student, "--adapter", f"{pair.adapters}/{chosen}",
                 "--rows", TEST_ROWS, "--out", pair.test_parses)
             if chosen else None)
    steps.append(Step("parse", pair.tag, parse, parses_complete(pair.test_parses),
                      "" if chosen else f"after select (reads {pair.selection})"))
    return steps


def all_steps(teachers, students, wandb=False, wanted=STEPS) -> list[Step]:
    out = []
    for t in teachers:
        out += [s for s in teacher_steps(t) if s.step in wanted]
    for pair in pairs(students, teachers):
        out += [s for s in student_steps(pair, wandb) if s.step in wanted]
    return out


# --- CLI -------------------------------------------------------------------

def cmd_plan(args) -> int:
    print("distillation grid (student <- teacher -> ablation parser arm, test parses):")
    for pair in pairs(args.students, args.teachers):
        state = "ok" if _has_rows(pair.test_parses) else "missing"
        print(f"  {pair.label:<58} {pair.name:<30} {pair.test_parses} [{state}]")
    for t in args.teachers:
        path = teacher_parses_path(t)
        state = "ok" if _has_rows(path) else "missing"
        print(f"  {TEACHERS[t].label + ' (zero-shot)':<58} "
              f"{teacher_parser_name(t):<30} {path} [{state}]")
    print()
    for s in all_steps(args.teachers, args.students, args.wandb, args.steps):
        if args.todo and s.done:
            continue
        mark = "done" if s.done else ("blocked" if s.argv is None else "todo")
        print(f"[{mark:>7}] {s.step:<8} {s.who:<24} {s.note}")
        if s.argv and not s.done:
            print(f"          {s.line()}")
    print("\nthen, in the evaluation environment:\n"
          "  python -m experiments.ablation plan\n  python -m experiments.ablation run")
    return 0


def cmd_run(args) -> int:
    # Re-planned after every command: later steps read earlier outputs
    for step_name in [s for s in STEPS if s in args.steps]:
        while True:
            todo = [s for s in all_steps(args.teachers, args.students, args.wandb,
                                         (step_name,)) if not s.done and s.argv]
            if not todo:
                break
            step = todo[0]
            argv = [sys.executable, *step.argv[1:]]
            print(f"\n=== {step.step} {step.who}: {shlex.join(step.argv)}", flush=True)
            if subprocess.run(argv, cwd=REPO_ROOT).returncode != 0:
                sys.exit(f"{step.step} failed for {step.who}; fix it and re-run "
                         "(finished outputs are skipped)")
            if not [s for s in all_steps(args.teachers, args.students, args.wandb,
                                         (step_name,))
                    if s.who == step.who and s.line() == step.line() and s.done]:
                sys.exit(f"{step.step} for {step.who} exited 0 but its output is "
                         "not there; see the command's output")
        blocked = [s for s in all_steps(args.teachers, args.students, args.wandb,
                                        (step_name,)) if not s.done and not s.argv]
        for s in blocked:
            print(f"  {s.step} {s.who}: not runnable yet ({s.note})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name, fn, help_ in (("plan", cmd_plan, "list pairs, steps, status and commands"),
                            ("run", cmd_run, "run the missing steps here")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--teachers", nargs="+", choices=list(TEACHERS),
                       default=list(TEACHERS))
        p.add_argument("--students", nargs="+", choices=list(STUDENTS),
                       default=list(STUDENTS))
        p.add_argument("--steps", nargs="+", choices=list(STEPS), default=list(STEPS))
        p.add_argument("--wandb", action="store_true",
                       help="log training to wandb (default: off)")
        if name == "plan":
            p.add_argument("--todo", action="store_true", help="hide finished steps")
        p.set_defaults(run=fn)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
