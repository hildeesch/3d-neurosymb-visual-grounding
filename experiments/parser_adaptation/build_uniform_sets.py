"""Build fixed-budget LoRA training sets from verified winners.

Weighting duplicates prompts because the trainer (train_lora.py, which mirrors
mlx_lm.lora) has no per-example weight.
Each arm uses the same per-prompt budget and a fixed seed; the validation split
holds out complete prompts. The margin and random controls are size-matched.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
from functools import partial
from collections import Counter, defaultdict
from pathlib import Path

from grounding.parsing import PROMPTS, cues
from grounding.parsing.training_set import user_turn
from grounding.paths import REPO_ROOT

SEED = 0
#: Rows per covered training prompt. The budget every weighting arm shares
BASE_ROWS_PER_PROMPT = 2

#: The frozen validation split, pinned by prompt id; it controls training eligibility only
VALID_PROMPTS_FILE = REPO_ROOT / "experiments/parser_adaptation/validation_split.json"

#: Seed for the random_size_matched draw, and nothing else
CONSTRUCTION_SEED = 1234


def load_valid_prompts(covered: list[int]) -> set[int]:
    """Load frozen validation IDs; fail if any are absent from the current pool."""
    if not VALID_PROMPTS_FILE.exists():
        raise SystemExit(
            f"missing {VALID_PROMPTS_FILE}. The validation split is frozen by "
            "id; it is not redrawn. Restore the artifact from git.")
    art = json.loads(VALID_PROMPTS_FILE.read_text())
    ids = set(art["valid_prompt_ids"])
    missing = ids - set(covered)
    if missing:
        raise SystemExit(
            f"{len(missing)} frozen validation prompts are absent from the "
            f"winner pool (e.g. {sorted(missing)[:5]}). The pool was rebuilt "
            "under a configuration the frozen split does not describe -- "
            "re-freeze deliberately or rebuild the pool, do not redraw here.")
    if art["covered_prompts"] != len(covered):
        print(f"  NOTE: pool covers {len(covered)} prompts, the frozen split "
              f"was taken over {art['covered_prompts']}; every frozen id is "
              "still present, so the partition holds.")
    return ids


def quantile(sorted_values: list[float], q: float) -> float:
    """Select round(q * (n - 1)), clamped to the array; Python rounds ties to even."""
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1,
                max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[index]


def original_prompt_split(pool: dict[int, list[dict]]) -> tuple[list[int], set[int], list[int]]:
    """Return the original prompts and their frozen train/validation split."""
    covered = sorted(prompt for prompt, winners in pool.items()
                     if any(winner["source"] == "original" for winner in winners))
    valid = load_valid_prompts(covered)
    return covered, valid, [prompt for prompt in covered if prompt not in valid]

# Hard:easy per-prompt sampling weight; all other construction settings are shared
VARIANTS: dict[str, float] = {
    "sft_v4_ext_prompt_uniform":         1.0,
    "sft_v4_ext_prompt_uniform_hard150": 1.5,
    "sft_v4_ext_prompt_uniform_hard200": 2.0,
}

def load_pool(path: Path) -> dict[int, list[dict]]:
    by_prompt: dict[int, list[dict]] = defaultdict(list)
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            by_prompt[r["row_idx"]].append(r)
    for rows in by_prompt.values():
        rows.sort(key=lambda c: c["winner_rank"])
    return by_prompt


def base_pair(winners: list[dict]) -> list[dict]:
    """The two programs a prompt draws from, whatever its pool size."""
    best = winners[:BASE_ROWS_PER_PROMPT]
    return best if len(best) == BASE_ROWS_PER_PROMPT else [best[0]] * BASE_ROWS_PER_PROMPT


def split_budget(n_easy: int, n_hard: int, budget: int, weight: float) -> int:
    """Rows to the easy stratum so hard-per-prompt / easy-per-prompt == weight.

    The exact solution is real; both integers either side of it are tried and
    the one whose realised ratio is closest to the request wins, so the rounding
    is decided by the quantity being controlled, not by a convention.
    """
    exact = n_easy * budget / (n_easy + weight * n_hard)

    def realised(easy_rows: int) -> float:
        return ((budget - easy_rows) / n_hard) / (easy_rows / n_easy)

    # Each stratum has to keep at least one row per prompt
    lo = max(n_easy, int(exact))
    hi = min(budget - n_hard, lo + 1)
    return min((lo, hi), key=lambda x: abs(realised(x) - weight))


def spread(prompts: list[int], total: int, seed: int) -> dict[int, int]:
    """`total` rows over `prompts`, as evenly as the integers allow.

    Which prompts carry the extra row is a seeded draw over the sorted stratum
    and nothing else. Tying it to margin, winner count or utterance length would
    reintroduce exactly the confidence-shaped selection this experiment tests.
    """
    n = len(prompts)
    base, extra = divmod(total, n)
    if base < 1:
        raise SystemExit(f"budget {total} cannot give {n} prompts one row each")
    chosen = set(random.Random(seed).sample(sorted(prompts), extra))
    return {p: base + (1 if p in chosen else 0) for p in prompts}


def example(row: dict, system: str, repeat: int, source: str) -> dict:
    """One chat row for train_lora.py, in the control set's exact shape."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_turn(row["utterance"])},
            {"role": "assistant",
             "content": json.dumps(row["program"], ensure_ascii=False)},
        ],
        "row_idx": row["row_idx"], "scan_id": row["scan_id"],
        "margin": row["margin"],
        # Provenance the control rows did not need: which winner, which pool, which copy
        "winner_rank": row["winner_rank"], "source": source,
        "repeat": repeat, "is_hard": row["is_hard"],
    }


def refuse_recovery(name: str, train: list[dict], valid: list[dict]) -> None:
    """Reject recovery rows even when --pool supplies a compatible external file."""
    for split, rows in (("train", train), ("valid", valid)):
        bad = [r["row_idx"] for r in rows if r.get("source") == "recovery"]
        if bad:
            raise SystemExit(
                f"{name}: {len(bad)} recovery rows reached {split}.jsonl "
                f"({bad[:5]}). Recovery is an R=16 diagnostic over prompts the "
                "R=8 teacher missed; training on it confounds coverage with "
                "sampling budget. The pool passed in is not R=8 only -- see "
                "build_winner_pool.py and the archived recovery diagnostic.")


def retention(rows: list[dict], pool: dict[int, list[dict]],
              prompts: set[int]) -> dict:
    """How many prompts still carry two distinct trajectories, and of how many.

    The ceiling is the prompts whose pool holds two or more verified winners;
    everything below that ceiling is the fixed budget spending a row elsewhere,
    which is a secondary characteristic of the weighting, not a defect.
    """
    used: dict[int, set[int]] = defaultdict(set)
    for r in rows:
        used[r["row_idx"]].add(r["winner_rank"])
    two = sum(1 for p in prompts if len(used[p]) >= 2)
    eligible = sum(1 for p in prompts if len(pool[p]) >= 2)
    return {
        "prompts_with_two_distinct_programs": two,
        "fraction_prompts_two_distinct": two / len(prompts) if prompts else 0.0,
        "prompts_eligible_for_two": eligible,
        "retention_of_available_pairs": two / eligible if eligible else 0.0,
    }


def describe(name: str, rows: list[dict], pool: dict[int, list[dict]],
             n_valid: int) -> dict:
    prompts = {r["row_idx"] for r in rows}
    hard = {p for p in prompts if pool[p][0]["is_hard"]}
    easy = prompts - hard
    hard_rows = [r for r in rows if r["is_hard"]]
    easy_rows = [r for r in rows if not r["is_hard"]]
    per_prompt = Counter(r["row_idx"] for r in rows)
    distinct = {p: len(pool[p]) for p in prompts}
    row_progs = [pool[r["row_idx"]][r["winner_rank"]] for r in rows]

    def mean(vals):
        return statistics.fmean(vals) if vals else 0.0

    return {
        "name": name,
        "physical_training_rows": len(rows),
        "valid_rows": n_valid,
        "unique_source_prompts": len(prompts),
        "easy_prompts": len(easy), "hard_prompts": len(hard),
        "easy_training_rows": len(easy_rows),
        "hard_training_rows": len(hard_rows),
        "rows_per_easy_prompt": (len(easy_rows) / len(easy)) if easy else 0.0,
        "rows_per_hard_prompt": (len(hard_rows) / len(hard)) if hard else 0.0,
        # Per-prompt weighting differs from the raw row ratio below
        "realized_hard_easy_weight": (
            (len(hard_rows) / len(hard)) / (len(easy_rows) / len(easy))
            if easy and hard else 0.0),

        "raw_hard_easy_row_ratio": (
            len(hard_rows) / len(easy_rows) if easy_rows else 0.0),
        "equal_mass_per_prompt": len(set(per_prompt.values())) == 1,
        "rows_per_prompt_histogram": dict(sorted(Counter(per_prompt.values()).items())),
        "rows_per_prompt_easy": dict(sorted(Counter(
            per_prompt[p] for p in easy).items())),
        "rows_per_prompt_hard": dict(sorted(Counter(
            per_prompt[p] for p in hard).items())),
        "distinct_programs_used": len({(r["row_idx"], r["winner_rank"])
                                       for r in rows}),
        # Report alternative trajectories lost to fixed-budget weighting
        **retention(rows, pool, prompts),
        "mean_distinct_winners_per_prompt": mean(list(distinct.values())),
        "prompts_with_1_winner": sum(1 for v in distinct.values() if v == 1),
        "prompts_with_2_winners": sum(1 for v in distinct.values() if v == 2),
        "prompts_with_3plus_winners": sum(1 for v in distinct.values() if v >= 3),
        "mean_utterance_words_rows": mean([p["word_count"] for p in row_progs]),
        "mean_executable_predicates_per_row": mean(
            [cues.predicate_count(p["program"]) for p in row_progs]),
        "multi_constraint_rate_rows": mean(
            [float(cues.multi_constraint(p["program"])) for p in row_progs]),
        "mean_cue_coverage_per_row": mean([p["coverage"] for p in row_progs]),
        "mean_margin_per_row": mean([p["margin"] for p in row_progs]),
        "recovery_training_rows": sum(1 for r in rows if r["source"] == "recovery"),
    }


def write_set(out_dir: Path, train: list[dict], valid: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # File order is shuffled so a scan-ordered pool does not become a curriculum
    random.Random(SEED).shuffle(train)
    for split, rows in (("train", train), ("valid", valid)):
        with open(out_dir / f"{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def render_rows(prompts, winners, system, counts=None):
    """Render prompts in their supplied order, cycling the selected winners."""
    return [example(winners[p][i % len(winners[p])], system, i,
                    winners[p][i % len(winners[p])]["source"])
            for p in prompts for i in range(1 if counts is None else counts[p])]


def write_variant(name, pool, train, valid, out_root, config, **details):
    """Validate, shuffle, serialize and describe every training arm identically."""
    refuse_recovery(name, train, valid)
    write_set(out_root / name, train, valid)
    stats = describe(name, train, pool, len(valid))
    stats["config"] = config
    stats.update(details)
    (out_root / name / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def build_weighted(name: str, weight: float, pool: dict[int, list[dict]],
                   system: str, out_root: Path) -> dict:
    """Allocate a fixed row budget using the requested hard/easy prompt weight."""
    _, valid_prompts, train_prompts = original_prompt_split(pool)

    budget = BASE_ROWS_PER_PROMPT * len(train_prompts)
    easy = [p for p in train_prompts if not pool[p][0]["is_hard"]]
    hard = [p for p in train_prompts if pool[p][0]["is_hard"]]
    easy_rows = split_budget(len(easy), len(hard), budget, weight)
    counts = {**spread(easy, easy_rows, SEED),
              **spread(hard, budget - easy_rows, SEED + 1)}

    train = render_rows(train_prompts, {p: base_pair(pool[p]) for p in train_prompts},
                        system, counts)
    valid = render_rows(sorted(valid_prompts), {p: [pool[p][0]] for p in valid_prompts},
                        system)

    assert len(train) == budget, (len(train), budget)
    config = {"hard_easy_weight": weight, "row_budget": budget,
                       "rows_per_prompt_base": BASE_ROWS_PER_PROMPT,
                       "includes_recovery": False,
                       "seed": SEED,
                       "valid_split": "frozen (experiments/parser_adaptation/validation_split.json)"}
    return write_variant(name, pool, train, valid, out_root, config)


#: The confidence cut the original control was selected under. Read off the eligible pool
MARGIN_PERCENTILE = 25.0

#: Name of the margin-filtered control
MARGIN_FILTERED = "sft_v4_ext_margin_filtered"

#: Name of the size-matched random control
RANDOM_SIZE_MATCHED = "sft_v4_ext_random_size_matched"

#: What the p25 cut yields on the frozen train pool
EXPECTED_MARGIN_TRAIN_PROMPTS = 3908


def build_margin_filtered(name: str, pool: dict[int, list[dict]], system: str,
                          out_root: Path) -> dict:
    """Keep the highest-ranked original winner above the positive-margin p25 cut."""
    eligible = {p: [c for c in v if c["source"] == "original"] for p, v in pool.items()}
    eligible = {p: v for p, v in eligible.items() if v}

    # Exclude target/distractor ties when calculating the confidence threshold
    positive = sorted(c["margin"] for v in eligible.values()
                      for c in v if c["margin"] > 0)
    tau = quantile(positive, MARGIN_PERCENTILE / 100.0)

    # Split before filtering so validation remains a subset of the frozen IDs
    covered, valid_prompts, _ = original_prompt_split(eligible)

    # Preserve cue-coverage ranking among winners that clear the margin cut
    kept: dict[int, dict] = {}
    for p, v in eligible.items():
        survivors = [c for c in v if c["margin"] >= tau]
        if survivors:
            kept[p] = min(survivors, key=lambda c: c["winner_rank"])

    train_prompts = sorted(p for p in kept if p not in valid_prompts)
    kept_valid = sorted(p for p in kept if p in valid_prompts)

    # Refuse a changed pool before producing size-matched comparisons
    if len(train_prompts) != EXPECTED_MARGIN_TRAIN_PROMPTS:
        raise SystemExit(
            f"margin cut yields {len(train_prompts)} train prompts, expected "
            f"{EXPECTED_MARGIN_TRAIN_PROMPTS} (tau={tau:.6f}). The winner pool "
            "or the margin policy has moved since the split was frozen. "
            "Re-verify the Stage III counts before rebuilding; do not relax "
            "this number to match.")

    winners = {p: [winner] for p, winner in kept.items()}
    train = render_rows(train_prompts, winners, system,
                        dict.fromkeys(train_prompts, BASE_ROWS_PER_PROMPT))
    valid = render_rows(kept_valid, winners, system)

    assert len(train) == BASE_ROWS_PER_PROMPT * len(train_prompts)
    assert not (set(train_prompts) & valid_prompts)
    removed_train = sorted(set(covered) - valid_prompts - set(train_prompts))
    config = {
        "margin_percentile": MARGIN_PERCENTILE,
        "margin_threshold": tau,
        "rows_per_prompt_base": BASE_ROWS_PER_PROMPT,
        "includes_recovery": False,
        "fixed_row_budget": False,
        "seed": SEED,
        "valid_split": "frozen (experiments/parser_adaptation/validation_split.json)",
        "split_procedure": "same frozen prompt split as build_weighted",
    }
    selection = {
        "eligible_prompts": len(covered),
        "eligible_original_winners": sum(len(v) for v in eligible.values()),
        "positive_margin_winners": len(positive),
        "zero_margin_winners": sum(len(v) for v in eligible.values()) - len(positive),
        "prompts_surviving_threshold": len(kept),
        "prompts_removed_by_margin": len(covered) - len(kept),
        "train_prompts": len(train_prompts),
        "valid_prompts": len(kept_valid),
        "valid_prompts_removed_by_margin": len(valid_prompts) - len(kept_valid),
        "train_prompts_removed_by_margin": len(removed_train),
        # Reported beside the pool's own rate so the margin cut's effect on difficulty is readable
        "hard_prompts": sum(1 for p in train_prompts if kept[p]["is_hard"]),
        "hard_prompt_fraction": (
            sum(1 for p in train_prompts if kept[p]["is_hard"])
            / len(train_prompts)),
        "pool_hard_prompt_fraction": (
            sum(1 for p in covered
                if p not in valid_prompts and pool[p][0]["is_hard"])
            / len([p for p in covered if p not in valid_prompts])),
    }
    return write_variant(name, pool, train, valid, out_root, config, selection=selection)


def build_random_size_matched(name: str, pool: dict[int, list[dict]],
                              system: str, out_root: Path) -> dict:
    """Draw the margin control's prompt count uniformly using CONSTRUCTION_SEED."""
    _, valid_prompts, train_prompts = original_prompt_split(pool)

    n = EXPECTED_MARGIN_TRAIN_PROMPTS
    if n > len(train_prompts):
        raise SystemExit(f"cannot draw {n} prompts from {len(train_prompts)}")
    # The ONE thing CONSTRUCTION_SEED governs
    chosen = sorted(random.Random(CONSTRUCTION_SEED).sample(train_prompts, n))

    # Highest-ranked winner per prompt, minus the margin test, so the arms differ in membership only
    kept = {p: pool[p][0] for p in chosen}
    train = render_rows(chosen, {p: [kept[p]] for p in chosen}, system,
                        dict.fromkeys(chosen, BASE_ROWS_PER_PROMPT))
    valid = render_rows(sorted(valid_prompts), {p: [pool[p][0]] for p in valid_prompts},
                        system)

    assert len(train) == BASE_ROWS_PER_PROMPT * n, (len(train), n)
    assert not (set(chosen) & valid_prompts)
    hard_p = sum(1 for p in chosen if kept[p]["is_hard"])
    config = {
        "selection": "uniform random, no margin criterion",
        "construction_seed": CONSTRUCTION_SEED,
        "size_matched_to": MARGIN_FILTERED,
        "rows_per_prompt_base": BASE_ROWS_PER_PROMPT,
        "includes_recovery": False,
        "fixed_row_budget": False,
        "valid_split": "frozen (experiments/parser_adaptation/validation_split.json)",
    }
    selection_detail = {
        "eligible_train_prompts": len(train_prompts),
        "drawn_prompts": n,
        "hard_prompts": hard_p,
        "easy_prompts": n - hard_p,
        "hard_prompt_fraction": hard_p / n,
        "pool_hard_prompt_fraction": (
            sum(1 for p in train_prompts if pool[p][0]["is_hard"])
            / len(train_prompts)),
    }
    return write_variant(name, pool, train, valid, out_root, config,
                         selection_detail=selection_detail)


BUILDERS = {
    MARGIN_FILTERED: build_margin_filtered,
    RANDOM_SIZE_MATCHED: build_random_size_matched,
    **{name: partial(build_weighted, weight=weight) for name, weight in VARIANTS.items()},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool",
                    default="data/distilled/nr3d_winners_fft_v4_ext.jsonl")
    ap.add_argument("--out", default="data/training_sets_zscore_sigmoid")
    ap.add_argument("--prompt", default="v4", choices=sorted(PROMPTS))
    ap.add_argument("--only", choices=list(BUILDERS),
                    help="build a single variant")
    ap.add_argument("--force", action="store_true",
                    help="replace a variant directory this script wrote before")
    args = ap.parse_args()

    system = PROMPTS[args.prompt]
    pool = load_pool(REPO_ROOT / args.pool)

    # A current pool is R=8 only; a prompt holding both sources means the pools overlapped
    sources = {c["source"] for v in pool.values() for c in v}
    if sources - {"original"}:
        print(f"  pool carries non-R=8 sources {sorted(sources)}; "
              "every arm drops them and refuse_recovery re-checks the output")
    mixed = [p for p, v in pool.items()
             if {c["source"] for c in v} == {"original", "recovery"}]
    if mixed:
        raise SystemExit(f"{len(mixed)} prompts appear in both pools: {mixed[:5]}")

    out_root = REPO_ROOT / args.out
    # Two experiments: difficulty weighting and margin versus random selection
    names = ([args.only] if args.only
             else list(BUILDERS))
    all_stats = []
    for name in names:
        target = out_root / name
        if target.exists() and any(target.iterdir()):
            if not args.force:
                raise SystemExit(f"{target} already has contents; pass --force "
                                 f"to replace it")
            shutil.rmtree(target)
        s = BUILDERS[name](name=name, pool=pool, system=system, out_root=out_root)
        all_stats.append(s)
        print(f"\n=== {name} ===")
        for k, v in s.items():
            if k in ("name", "config"):
                continue
            print(f"  {k:44s} {v}")
    (out_root / "variants.stats.json").write_text(json.dumps(all_stats, indent=2))
    print(f"\nWrote {len(all_stats)} training sets under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
