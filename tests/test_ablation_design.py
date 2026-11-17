import pytest

from experiments.ablation import design as D


def _args(cell):
    tree, args = D.cell_command(cell, "p.jsonl", "test", "official", "d.jsonl")
    return tree, args, dict(zip(args[::1], args[1:] + [None]))


def test_full_system_cell_runs_frozen_resolver_in_extended_tree():
    tree, args, kv = _args(D.Cell(*D.FULL))
    assert tree == "extended"
    assert kv["--norm_mode"] == "zscore_sigmoid"
    assert kv["--color_mode"] == "product" and "--use_rank" in args
    assert "extended" in kv["--features_path"]


def test_executor_ablation_keeps_representation_features():
    tree, args, kv = _args(D.Cell("student", "extended", "lasp"))
    assert tree == "extended" and kv["--norm_mode"] == "softmax"
    assert "--use_rank" in args and kv["--color_mode"] == "product"


def test_representation_ablation_runs_lasp_features_under_thesis_executor():
    tree, args, kv = _args(D.Cell("student", "lasp", "thesis"))
    assert tree == "baseline"
    assert kv["--executor"] == "thesis" and kv["--norm_mode"] == "zscore_sigmoid"
    assert kv["--tie_break"] == "legacy"
    assert "extended" not in kv["--features_path"]


def test_lasp_corner_is_upstream_parse():
    tree, args, kv = _args(D.Cell("teacher_gpt4o", "lasp", "lasp"))
    assert tree == "baseline" and kv["--executor"] == "lasp"
    assert "--norm_mode" not in args


def test_cells_reject_unknown_levels():
    with pytest.raises(ValueError):
        D.cells(["student"], ["extended"], ["optuna"])
    assert len(D.cells(list(D.PARSERS), ["extended", "lasp"], ["thesis", "lasp"])) == 12


def test_teacher_parse_command():
    cmd = D.teacher_parse_command("teacher_qwen")
    assert cmd[cmd.index("--teacher") + 1] == "qwen"
    assert cmd[cmd.index("--representation") + 1] == "extended"
