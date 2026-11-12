#!/usr/bin/env python3
"""Run five parser-training arms at seeds 0, 1 and 2.

Defaults are 1000 updates, batch size 4 and the shared LoRA YAML. The margin and
size-matched random controls compare selection; prompt-uniform, hard150 and
hard200 compare sampling weights. Completion requires the final-save log marker,
not merely a checkpoint file. Training runs train_lora.py (PyTorch + PEFT),
which takes the mlx_lm lora flags. See README.md for inputs and scoring."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: Updates every arm receives. Not scaled by dataset size -- see the docstring
ITERS = 1000

# Fixed seed replicates for each retained arm
TRAIN_SEEDS = (0, 1, 2)

# Margin filtering uses the random size-matched control; the uniform arms vary weight only
MARGIN_FILTERED = "sft_v4_ext_margin_filtered"
RANDOM_SIZE_MATCHED = "sft_v4_ext_random_size_matched"

ARMS = [
    MARGIN_FILTERED,
    RANDOM_SIZE_MATCHED,
    "sft_v4_ext_prompt_uniform",
    "sft_v4_ext_prompt_uniform_hard150",
    "sft_v4_ext_prompt_uniform_hard200",
]

MODEL = "Qwen/Qwen3-1.7B"
CONFIG = "lora_config_sft_v4_ext.yaml"
BATCH_SIZE = 4
SAVE_EVERY = 200
STEPS_PER_EVAL = 100
STEPS_PER_REPORT = 10
VAL_BATCHES = 25
PROJECT = "nr3d-sft-v4-ext"


def train_rows(name: str, sets_dir: Path) -> int:
    path = sets_dir / name / "train.jsonl"
    if not path.exists():
        sys.exit(f"missing {path}")
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def schedule(sets_dir: Path, iters: int = ITERS) -> list[dict]:
    """One entry per (arm, seed). Every run gets the same update budget.

    `epochs` is still reported, because it is no longer held constant and so
    becomes a fact about each arm worth reading: at a fixed budget a larger set
    is covered less. It is a diagnostic here, not a control.
    """
    out = []
    for a in ARMS:
        rows = train_rows(a, sets_dir)
        for seed in TRAIN_SEEDS:
            out.append({"name": a, "seed": seed, "rows": rows, "iters": iters,
                        "run": f"{a}__seed{seed}",
                        "rows_consumed": iters * BATCH_SIZE,
                        "epochs": iters * BATCH_SIZE / rows})
    return out


# Periodic checkpoints can survive an interrupted run; require the final-save log
DONE_MARKER = "Saved final weights to"


def finished(name: str, logs: Path) -> bool:
    log = logs / f"{name}.log"
    return log.exists() and DONE_MARKER in log.read_text(errors="ignore")


def command(arm: dict, sets_dir: Path, adapters: Path, wandb: bool) -> list[str]:
    cmd = [
        sys.executable, "-m", "experiments.parser_adaptation.train_lora",
        "--model", MODEL, "--train",
        "--data", str(sets_dir / arm["name"]),
        # Per-(arm, seed) adapter dir so three runs of one arm do not overwrite each other
        "--adapter-path", str(adapters / arm["run"]),
        "-c", CONFIG,
        "--batch-size", str(BATCH_SIZE),
        "--iters", str(arm["iters"]),
        "--save-every", str(SAVE_EVERY),
        "--steps-per-eval", str(STEPS_PER_EVAL),
        "--steps-per-report", str(STEPS_PER_REPORT),
        "--val-batches", str(VAL_BATCHES),
        "--seed", str(arm["seed"]),
        # Activation checkpointing keeps the retained runs within GPU memory
        "--grad-checkpoint",
    ]
    if wandb:
        cmd += ["--report-to", "wandb", "--project-name", PROJECT]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=ITERS,
                    help="updates every run receives. Fixed across arms: the "
                         "optimization budget is the controlled quantity")
    ap.add_argument("--sets", type=Path,
                    default=REPO / "data/training_sets_zscore_sigmoid")
    ap.add_argument("--adapters", type=Path, default=REPO / "artifacts/adapters")
    ap.add_argument("--logs", type=Path, default=REPO / "runs/logs")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the schedule and the commands, run nothing")
    # Adapter checkpoints omit optimizer state, so resume only between completed runs
    ap.add_argument("--skip-done", action="store_true",
                    help="skip arms that already FINISHED; how an interrupted "
                         "sequence is resumed without redoing completed arms")
    ap.add_argument("--only", action="append", default=None,
                    help="run just this arm; repeatable")
    ap.add_argument("--seeds", type=int, action="append", default=None,
                    help="run just this seed; repeatable (default: all of "
                         f"{TRAIN_SEEDS})")
    args = ap.parse_args()

    plan = schedule(args.sets, args.iters)
    if args.only:
        unknown = set(args.only) - {a["name"] for a in plan}
        if unknown:
            sys.exit(f"unknown arm(s): {sorted(unknown)}")
        plan = [a for a in plan if a["name"] in set(args.only)]
    if args.seeds:
        plan = [a for a in plan if a["seed"] in set(args.seeds)]

    w = max(len(a["run"]) for a in plan)
    print(f"{'run':<{w}}  {'train rows':>10}  {'iters':>7}  {'rows used':>10}  "
          f"{'epochs':>8}")
    print("-" * (w + 42))
    for a in plan:
        print(f"{a['run']:<{w}}  {a['rows']:>10}  {a['iters']:>7}  "
              f"{a['rows_consumed']:>10}  {a['epochs']:>8.4f}")
    # Epochs vary by arm: the consequence of a fixed update budget, reported not smoothed
    print(f"\nruns: {len(plan)}   total iterations: {sum(a['iters'] for a in plan)}")
    print(f"epochs range {min(a['epochs'] for a in plan):.4f} - "
          f"{max(a['epochs'] for a in plan):.4f}  (not held constant: every run "
          f"gets {plan[0]['iters']} updates)")

    if args.dry_run:
        print("\ncommands:")
        for a in plan:
            print("  " + " ".join(command(a, args.sets, args.adapters, not args.no_wandb)))
        return 0

    args.adapters.mkdir(parents=True, exist_ok=True)
    args.logs.mkdir(parents=True, exist_ok=True)
    failed, skipped = [], []
    for a in plan:
        if args.skip_done and finished(a["run"], args.logs):
            skipped.append(a["run"])
            print(f"\n=== {a['run']}  already trained, skipping", flush=True)
            continue
        log = args.logs / f"{a['run']}.log"
        print(f"\n=== {a['run']}  {a['iters']} iters -> {log}", flush=True)
        with log.open("w") as f:
            p = subprocess.run(command(a, args.sets, args.adapters,
                                       not args.no_wandb),
                               cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
        # One arm's failure must not take the other three down with it
        if p.returncode != 0:
            failed.append(a["run"])
            print(f"  FAILED (exit {p.returncode}); see {log}", flush=True)
    if skipped:
        print(f"\nskipped (already trained): {', '.join(skipped)}")
    print("\n" + ("all arms finished" if not failed
                  else f"FAILED: {', '.join(failed)}"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
