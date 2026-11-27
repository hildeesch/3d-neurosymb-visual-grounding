"""Input fingerprints and score-table parsing for evaluation commands."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

from grounding.paths import REPO_ROOT, RESULTS_FINAL, RESULTS_STAGE1
from grounding.resolution.harness import SPLIT_NAMES, tree_dir


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def relative_path(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def provenance(rep: str, parses: Path | None, features: Path,
               tie_break: str | None, totals: str) -> dict:
    """Record the input rows, feature tensor and evaluation conventions."""
    if parses is None:
        parses = (tree_dir(rep) / "data/symbolic_exp/nr3d.jsonl").resolve()
    with parses.open() as source:
        n_rows = sum(bool(line.strip()) for line in source)
    return {
        "representation": rep,
        "parses": {"path": relative_path(parses), "sha256": sha256(parses), "n": n_rows},
        "features": {"path": relative_path(features.resolve()), "sha256": sha256(features)},
        "tie_break": tie_break,
        "totals": totals,
    }


def parse_scores(output: str) -> dict[str, tuple[int, int]]:
    scores = {}
    for name in SPLIT_NAMES:
        match = re.search(rf"^{name}\s*:?\s+(\d+)\s*/?\s*(\d+)", output, re.M)
        if match:
            scores[name] = (int(match.group(1)), int(match.group(2)))
    return scores


def comparison_path(out_dir: Path, parses: Path, representation: str | None = None) -> Path:
    """Use concise names only for the maintained run in its final directory."""
    if (out_dir.resolve() == RESULTS_FINAL.resolve()
            and parses.stem == "nr3d_test_sft_v4_ext_random_size_matched__seed2"):
        name = (f"test_predictions_{representation}.jsonl" if representation
                else "test_comparison.json")
    elif (out_dir.resolve() == (RESULTS_STAGE1 / "dev_comparison").resolve()
          and parses.stem == "nr3d_dev_30b_v4"):
        name = f"predictions_{representation}.jsonl" if representation else "comparison.json"
    else:
        name = (f"compare_{parses.stem}_{representation}.jsonl" if representation
                else f"compare_{parses.stem}.json")
    return out_dir / name


def checkpoint_result_name(tag: str) -> str:
    """Readable checkpoint name; run identity remains in the result metadata."""
    # v4 and tree runs share the readable form; they live in different directories
    return (tag.removeprefix("sft_v4_ext_").removeprefix("sft_tree_ext_")
            .replace("__seed", "_seed_")
            .replace("zeroshot", "zero_shot") + ".json")


def calibration_result_name(name: str, parses: Path) -> str:
    experiment = {"final_aggregation_calibration": "aggregation",
                  "normalization_mechanism": "normalization"}.get(name, name)
    populations = {"nr3d_dev_30b_v4": "all_programs",
                   "nr3d_dev_30b_v4_cge2": "multi_constraint",
                   "nr3d_dev_30b_v4_anchor_c1": "single_anchor"}
    population = populations.get(parses.stem, parses.stem.removeprefix("nr3d_test_sft_v4_ext_"))
    if population.endswith("_cge2"):
        return str(Path(population.removesuffix("_cge2")) / f"{experiment}_multi_constraint.json")
    return f"{experiment}_{population}.json"
