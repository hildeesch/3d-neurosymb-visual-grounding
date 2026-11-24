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
    assert len(D.cells(list(D.PARSERS), ["extended", "lasp"], ["thesis", "lasp"])) == 28


def test_teacher_parse_command():
    cmd = D.teacher_parse_command("teacher_qwen")
    assert cmd[cmd.index("--teacher") + 1] == "qwen"
    assert cmd[cmd.index("--representation") + 1] == "extended"


def test_parser_arms_cover_every_teacher_student_pair():
    assert D.FULL[0] == "student"
    assert set(D.STUDENT_PARSERS.values()) == {
        (s, t) for s in ("qwen3", "coder") for t in ("qwen", "qwen_coder")}
    assert D.STUDENT_PARSERS["student"] == ("qwen3", "qwen")
    assert D.STUDENT_PARSERS["student_coder_from_qwen_coder"] == ("coder", "qwen_coder")
    assert set(D.TEACHER_PARSERS) == {"teacher_qwen", "teacher_qwen_coder",
                                      "teacher_gpt4o"}
    # the default run is the whole local grid: 4 students + 2 local teachers
    assert set(D.DEFAULT_PARSERS) == set(D.PARSERS) - {"teacher_gpt4o"}
    # the maintained student keeps its parses file
    assert D.PARSERS["student"].endswith("random_size_matched__seed2.jsonl")
    assert len(set(D.PARSERS.values())) == len(D.PARSERS)


def test_cells_expose_teacher_and_student():
    cell = D.Cell("student_coder_from_qwen_coder", "lasp", "thesis")
    assert (cell.level("teacher"), cell.level("student")) == ("qwen_coder", "coder")
    zero_shot = D.Cell("teacher_qwen_coder", "extended", "thesis")
    assert zero_shot.level("student") == D.ZERO_SHOT
    assert zero_shot.level("teacher") == "qwen_coder"
    assert D.teacher_arm("qwen_coder") == "teacher_qwen_coder"


def test_coder_teacher_parse_command_and_student_hint():
    cmd = D.teacher_parse_command("teacher_qwen_coder")
    assert cmd[cmd.index("--teacher") + 1] == "qwen_coder"
    hint = D.student_parse_hint("student_coder")
    assert "--students coder --teachers qwen" in hint
