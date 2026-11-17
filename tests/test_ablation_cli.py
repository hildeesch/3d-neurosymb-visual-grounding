"""Orchestration of `python -m experiments.ablation run/analyse`, trees mocked."""

import json
import random

import pytest

from experiments.ablation import cli


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    keys = list(range(200))
    paths = {}
    for parser in ("student", "teacher_qwen"):
        path = tmp_path / f"{parser}.jsonl"
        path.write_text("".join(json.dumps({"dataset_idx": k, "scan_id": "s",
                                            "target_id": 1, "json_obj": {}}) + "\n"
                                for k in keys))
        paths[parser] = path

    # Accuracy rises with each full-system level; the student x representation
    # interaction is planted on rows 150-169.
    def correct(cell_name, k):
        parser, rep, ex = cell_name.split("__")
        level = (parser == "student") + (rep == "extended") + (ex == "thesis")
        hit = random.Random(f"{k}").random() < 0.3 + 0.1 * level
        if 150 <= k < 170:
            hit = parser == "student" and rep == "extended"
        return hit

    calls = []

    def fake_run_in_tree(tree, module, args):
        calls.append((tree, args))
        dump = args[args.index("--dump") + 1]
        name = dump.rsplit("/", 1)[1].removesuffix(".jsonl")
        hits = [k for k in keys if correct(name, k)]
        with open(dump, "w") as out:
            for k in keys[:-5]:          # 5 rows unscored -> misses
                out.write(json.dumps({"dataset_idx": k, "correct": k in hits}) + "\n")
        n = sum(1 for k in hits if k < keys[-5])
        return f"overall: {n} / {len(keys)} (x)\neasy: 1 / 2 (x)\n"

    monkeypatch.setattr(cli, "run_in_tree", fake_run_in_tree)
    monkeypatch.setattr(cli, "dataset_split", lambda path, split: "test")
    monkeypatch.setattr(cli, "ensure_features", lambda *a, **k: None)
    monkeypatch.setattr(cli, "feature_tensor", lambda tree, split: paths["student"])
    return tmp_path, paths, calls


def test_run_scores_every_cell_and_analyses(workspace):
    tmp_path, paths, calls = workspace
    out = tmp_path / "out"
    rc = cli.main(["run", "--parsers", "student", "teacher_qwen",
                   "--parses", f"student={paths['student']}",
                   "--parses", f"teacher_qwen={paths['teacher_qwen']}",
                   "--out-dir", str(out), "--n-boot", "300"])
    assert rc == 0
    assert len(calls) == 8
    assert {t for t, _ in calls} == {"extended", "baseline"}

    report = json.loads((out / "ablation.json").read_text())
    assert report["warnings"] == []
    assert len(report["cells"]) == 8
    assert {a["label"] for a in report["ablations"]} >= {
        "parser=teacher_qwen", "representation=lasp", "executor=lasp"}
    (factorial,) = report["factorial"]
    effects = {e["effect"]: e for e in factorial["effects"]}
    assert set(effects) == {"parser", "representation", "executor",
                            "parser x representation", "parser x executor",
                            "representation x executor",
                            "parser x representation x executor"}
    assert effects["parser x representation"]["estimate_pp"] > 0
    assert all("p_holm" in e for e in effects.values() if e["order"] >= 2)
    assert (out / "ablation.md").read_text().startswith("# Ablation")


def test_run_refuses_mismatched_rows(workspace):
    tmp_path, paths, _ = workspace
    lines = paths["teacher_qwen"].read_text().splitlines()[:-1]
    paths["teacher_qwen"].write_text("\n".join(lines) + "\n")
    with pytest.raises(SystemExit, match="different rows"):
        cli.main(["run", "--parsers", "student", "teacher_qwen",
                  "--parses", f"student={paths['student']}",
                  "--parses", f"teacher_qwen={paths['teacher_qwen']}",
                  "--out-dir", str(tmp_path / "o")])


def test_missing_teacher_prints_parse_command(workspace, tmp_path):
    _, paths, _ = workspace
    with pytest.raises(SystemExit, match="--teacher gpt4o"):
        cli.main(["run", "--parsers", "student", "teacher_gpt4o",
                  "--parses", f"student={paths['student']}",
                  "--parses", f"teacher_gpt4o={tmp_path / 'nope.jsonl'}",
                  "--out-dir", str(tmp_path / "o")])
