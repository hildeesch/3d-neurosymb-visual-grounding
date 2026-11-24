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


@pytest.fixture
def grid_workspace(tmp_path, monkeypatch):
    """Every default parser arm: four students of the grid and two teachers."""
    from experiments.ablation import design as D

    keys = list(range(240))
    paths = {}
    for parser in D.DEFAULT_PARSERS:
        path = tmp_path / f"{parser}.jsonl"
        path.write_text("".join(json.dumps({"dataset_idx": k}) + "\n" for k in keys))
        paths[parser] = path

    # Rows 100-139: the code student only works with the code teacher and the
    # Qwen3 student only with the Qwen3 teacher -> a teacher x student
    # interaction, positive under the +1 = (qwen, qwen3) convention.
    def correct(cell_name, k):
        cell = D.Cell(*cell_name.split("__"))
        if 100 <= k < 140 and cell.level("student") != D.ZERO_SHOT:
            return (cell.level("teacher") == "qwen_coder") == \
                (cell.level("student") == "coder")
        level = (cell.representation == "extended") + (cell.executor == "thesis")
        return random.Random(f"{k}").random() < 0.35 + 0.1 * level

    calls = []

    def fake_run_in_tree(tree, module, args):
        calls.append((tree, args))
        dump = args[args.index("--dump") + 1]
        name = dump.rsplit("/", 1)[1].removesuffix(".jsonl")
        n = 0
        with open(dump, "w") as out:
            for k in keys:
                hit = correct(name, k)
                n += hit
                out.write(json.dumps({"dataset_idx": k, "correct": hit}) + "\n")
        return f"overall: {n} / {len(keys)} (x)\n"

    monkeypatch.setattr(cli, "run_in_tree", fake_run_in_tree)
    monkeypatch.setattr(cli, "dataset_split", lambda path, split: "test")
    monkeypatch.setattr(cli, "ensure_features", lambda *a, **k: None)
    monkeypatch.setattr(cli, "feature_tensor", lambda tree, split: paths["student"])
    return tmp_path, paths, calls


def test_default_run_scores_and_analyses_the_whole_grid(grid_workspace):
    from experiments.ablation import design as D

    tmp_path, paths, calls = grid_workspace
    out = tmp_path / "out"
    argv = ["run", "--out-dir", str(out), "--n-boot", "200"]
    for name, path in paths.items():
        argv += ["--parses", f"{name}={path}"]
    assert cli.main(argv) == 0
    assert len(calls) == 6 * 2 * 2

    report = json.loads((out / "ablation.json").read_text())
    families = [d["family"] for d in report["factorial"]]
    assert families.count("grid") == 1
    assert families.count("distillation") == 4      # every pair vs its teacher
    assert families.count("teachers") == 1          # qwen vs qwen_coder zero-shot

    (grid,) = [d for d in report["factorial"] if d["family"] == "grid"]
    assert grid["levels"]["teacher"] == {"full": "qwen", "ablated": "qwen_coder"}
    assert grid["levels"]["student"] == {"full": "qwen3", "ablated": "coder"}
    effects = {e["effect"]: e for e in grid["effects"]}
    assert len(effects) == 15                       # 2^4 - 1
    ts = effects["teacher x student"]
    assert ts["estimate_pp"] > 0 and ts["p_exact"] < 1e-3
    assert all("p_holm" in e for d in report["factorial"]
               for e in d["effects"] if e["order"] >= 2)

    distilled = {d["levels"]["parser"]["full"]: d["levels"]["parser"]["ablated"]
                 for d in report["factorial"] if d["family"] == "distillation"}
    assert distilled == {"student": "teacher_qwen", "student_coder": "teacher_qwen",
                         "student_from_qwen_coder": "teacher_qwen_coder",
                         "student_coder_from_qwen_coder": "teacher_qwen_coder"}

    # one teacher x student accuracy table per representation/executor cell,
    # full system's first
    assert len(report["grid"]) == 4
    first = report["grid"][0]
    assert (first["representation"], first["executor"]) == ("extended", "thesis")
    assert set(first["accuracy_pp"]) == {D.ZERO_SHOT, "qwen3", "coder"}
    assert set(first["accuracy_pp"]["coder"]) == {"qwen", "qwen_coder"}
    md = (out / "ablation.md").read_text()
    assert "## Teacher x student (extended representation, thesis executor)" in md
    assert "## Factorial (grid): teacher[qwen|qwen_coder] x student[qwen3|coder]" in md


def test_missing_student_arm_points_to_the_grid(workspace, tmp_path):
    _, paths, _ = workspace
    with pytest.raises(SystemExit, match="teacher_student_grid plan --students coder"):
        cli.main(["run", "--parsers", "student", "student_coder",
                  "--parses", f"student={paths['student']}",
                  "--parses", f"student_coder={tmp_path / 'nope.jsonl'}",
                  "--out-dir", str(tmp_path / "o")])
