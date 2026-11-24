"""The (student, teacher) grid: names, paths, step status and the training guard."""

import json
import sys

import pytest

from experiments.parser_adaptation import teacher_student_grid as G

tsa = pytest.importorskip("experiments.parser_adaptation.train_sft_arms")


def test_every_pair_has_its_own_paths():
    pairs = G.pairs()
    assert {(p.student, p.teacher) for p in pairs} == {
        ("qwen3", "qwen"), ("coder", "qwen"),
        ("qwen3", "qwen_coder"), ("coder", "qwen_coder")}
    assert [p.name for p in pairs] == ["student", "student_coder",
                                       "student_from_qwen_coder",
                                       "student_coder_from_qwen_coder"]
    for attr in ("adapters", "logs", "dev_dir", "test_parses"):
        assert len({getattr(p, attr) for p in pairs}) == 4, attr
    ref = G.Pair("qwen3", "qwen")
    assert ref.reference and ref.test_parses == G.REFERENCE_PARSES
    assert ref.adapters == "artifacts/adapters"
    # dev results use score_checkpoint's naming
    from grounding.resolution.artifacts import checkpoint_result_name
    p = G.Pair("coder", "qwen_coder")
    assert p.dev_result(1).endswith(checkpoint_result_name(p.run(1)))


def test_teacher_paths_keep_the_thesis_pool():
    assert G.rollouts_path("qwen") == "data/rollouts/nr3d_rollouts_fft_v4.jsonl"
    assert G.winners_path("qwen") == "data/distilled/nr3d_winners_fft_v4_ext.jsonl"
    assert G.sets_path("qwen") == "data/training_sets_zscore_sigmoid"
    assert G.rollouts_path("qwen_coder") == "data/rollouts/nr3d_rollouts_qwen_coder_v4.jsonl"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "REPO_ROOT", tmp_path)
    return tmp_path


def _touch(root, rel, text="x\n"):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_teacher_steps_and_foreign_pool_flag(repo):
    steps = {s.step: s for s in G.teacher_steps("qwen_coder")}
    assert not any(s.done for s in steps.values())
    roll = steps["rollouts"].argv
    assert roll[roll.index("--teacher") + 1] == "qwen_coder"
    assert roll[roll.index("--limit") + 1] == "8750"
    assert "--foreign-pool" in steps["sets"].argv
    assert "--foreign-pool" not in {s.step: s for s in G.teacher_steps("qwen")}["sets"].argv
    assert steps["zeroshot"].argv[steps["zeroshot"].argv.index("--teacher") + 1] == "qwen_coder"
    # a later output completes the earlier steps (thesis sets without pools)
    _touch(repo, f"{G.sets_path('qwen')}/{G.ARM}/train.jsonl")
    qwen = {s.step: s.done for s in G.teacher_steps("qwen")}
    assert qwen == {"rollouts": True, "filter": True, "winners": True,
                    "sets": True, "zeroshot": False}


def test_student_steps_follow_the_pipeline(repo):
    pair = G.Pair("coder", "qwen")
    steps = G.student_steps(pair)
    assert [s.step for s in steps] == ["train", "dev", "dev", "dev", "select", "parse"]
    train = steps[0].argv
    assert train[train.index("--student") + 1] == "coder"
    assert "--no-wandb" in train and "--skip-done" in train
    dev = steps[1].argv
    assert dev[dev.index("--model") + 1] == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert dev[dev.index("--n") + 1] == "0"
    assert steps[-1].argv is None                     # waits for the selection

    for seed in G.SEEDS:
        run = pair.run(seed)
        _touch(repo, f"{pair.logs}/{run}.log", "... Saved final weights to x\n")
        _touch(repo, f"{pair.adapters}/{run}/training_config.json", "{}")
        _touch(repo, pair.dev_result(seed), "{}")
    _touch(repo, pair.selection, json.dumps({"selected": pair.run(2)}))
    steps = G.student_steps(pair)
    assert all(s.done for s in steps[:-1])
    parse = steps[-1].argv
    assert parse[parse.index("--adapter") + 1] == f"{pair.adapters}/{pair.run(2)}"
    assert parse[parse.index("--student") + 1] == "coder"
    assert parse[parse.index("--out") + 1] == pair.test_parses
    assert not steps[-1].done
    _touch(repo, pair.test_parses)
    _touch(repo, pair.test_parses + ".meta.json", json.dumps({"written": {}}))
    assert G.student_steps(pair)[-1].done


def test_reference_pair_is_never_rerun(repo):
    steps = G.student_steps(G.Pair("qwen3", "qwen"))
    assert all(s.done and s.argv is None for s in steps)


def test_plan_lists_the_grid(repo, capsys):
    assert G.main(["plan", "--todo"]) == 0
    out = capsys.readouterr().out
    for name in ("student_coder_from_qwen_coder", "teacher_qwen_coder",
                 "Qwen2.5-Coder-1.5B-Instruct <- Qwen3-Coder-30B-A3B-Instruct"):
        assert name in out


def test_run_executes_missing_steps_in_order(repo, monkeypatch):
    ran = []

    def fake_run(argv, cwd):
        ran.append(argv)
        module = argv[2].rsplit(".", 1)[1]
        out = argv[argv.index("--output") + 1] if "--output" in argv else None
        if module == "build_uniform_sets":
            out = f"{argv[argv.index('--out') + 1]}/{G.ARM}/train.jsonl"
        _touch(repo, out)
        return type("P", (), {"returncode": 0})()

    monkeypatch.setattr(G.subprocess, "run", fake_run)
    _touch(repo, G.rollouts_path("qwen_coder"))
    assert G.main(["run", "--teachers", "qwen_coder", "--steps",
                   "filter", "winners", "sets"]) == 0
    assert [a[2].rsplit(".", 1)[1] for a in ran] == [
        "filter_rollouts", "build_winner_pool", "build_uniform_sets"]
    assert ran[0][0] == sys.executable


def test_train_sft_arms_uses_the_student_model(tmp_path):
    arm = {"name": G.ARM, "run": f"{G.ARM}__seed0", "iters": 10, "seed": 0}
    cmd = tsa.command(arm, tmp_path, tmp_path / "ad", False,
                      "Qwen/Qwen2.5-Coder-1.5B-Instruct")
    assert cmd[cmd.index("--model") + 1] == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert tsa.command(arm, tmp_path, tmp_path / "ad", False)[
        cmd.index("--model") + 1] == "Qwen/Qwen3-1.7B"


def test_train_sft_arms_refuses_another_pairs_adapter(tmp_path):
    arm = {"name": G.ARM, "run": f"{G.ARM}__seed0"}
    sets, adapters = tmp_path / "sets", tmp_path / "ad"
    assert tsa.adapter_conflict(arm, sets, adapters, "Qwen/Qwen3-1.7B") is None
    cfg = adapters / arm["run"] / "training_config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"model": "Qwen/Qwen3-1.7B",
                               "data": tsa._data_key(sets / G.ARM)}))
    assert tsa.adapter_conflict(arm, sets, adapters, "Qwen/Qwen3-1.7B") is None
    msg = tsa.adapter_conflict(arm, sets, adapters, "Qwen/Qwen2.5-Coder-1.5B-Instruct")
    assert msg and "Qwen2.5-Coder" in msg
    assert tsa.adapter_conflict(arm, tmp_path / "other_sets", adapters,
                                "Qwen/Qwen3-1.7B")
