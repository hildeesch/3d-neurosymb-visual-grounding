import argparse
from pathlib import Path

import pytest

ps = pytest.importorskip("experiments.parser_adaptation.parse_split")


class _Ap:
    def error(self, msg):
        raise SystemExit(msg)


def ns(**kw):
    base = dict(teacher=None, style=None, backend=None, model=None,
                max_tokens=None, adapter=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_no_teacher_keeps_student_defaults():
    args = ns()
    ps.apply_teacher(args, _Ap())
    assert (args.style, args.backend, args.max_tokens) == ("sft", "hf", 256)


def test_qwen_teacher_defaults_to_llamacpp_rollout_style():
    args = ns(teacher="qwen")
    ps.apply_teacher(args, _Ap())
    assert args.style == "rollout" and args.backend == "llamacpp"
    assert args.model is None and args.max_tokens == 320


def test_qwen_teacher_on_transformers():
    args = ns(teacher="qwen", backend="hf")
    ps.apply_teacher(args, _Ap())
    assert args.model == "Qwen/Qwen3-30B-A3B"


def test_gpt4o_teacher():
    args = ns(teacher="gpt4o")
    ps.apply_teacher(args, _Ap())
    assert args.backend == "openai" and args.model.startswith("gpt-4o")


@pytest.mark.parametrize("bad", [dict(teacher="gpt4o", backend="hf"),
                                 dict(teacher="qwen", adapter="a"),
                                 dict(teacher="qwen", style="teacher")])
def test_teacher_rejects_confounds(bad):
    with pytest.raises(SystemExit):
        ps.apply_teacher(ns(**bad), _Ap())


def test_rollout_style_matches_rollout_prompt():
    from experiments.parser_adaptation.generate_rollouts import user_message
    from grounding.parsing import user_turn
    # the teacher sees the chat template, the student plain ChatML; both get
    # the same system prompt and user turn, and the same to_executable step
    assert ps.STYLES == {"sft": False, "rollout": True}
    assert user_turn("the red chair") == user_message("the red chair")
    assert user_turn("the red chair") == "Sentence: the red chair\n\nParsed JSON:"


def test_teacher_out():
    assert ps.teacher_out("gpt4o", Path("data/corpora/nr3d_test.csv"), constrained=False) == \
        "data/parses/test/nr3d_test_teacher_gpt4o_tree_v1_nogrammar_ext.jsonl"


def test_qwen_coder_teacher_defaults_to_llamacpp():
    args = ns(teacher="qwen_coder")
    ps.apply_teacher(args, _Ap())
    assert args.style == "rollout" and args.backend == "llamacpp"
    assert args.model is None and args.max_tokens == 320


def test_qwen_coder_teacher_on_transformers():
    args = ns(teacher="qwen_coder", backend="hf")
    ps.apply_teacher(args, _Ap())
    assert args.model == "Qwen/Qwen3-Coder-30B-A3B-Instruct"


def test_coder_student_preset():
    args = ns(student="coder")
    ps.apply_teacher(args, _Ap())
    assert args.model == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert (args.style, args.backend) == ("sft", "hf")
    # an explicit --model still wins
    args = ns(student="coder", model="local/dir")
    ps.apply_teacher(args, _Ap())
    assert args.model == "local/dir"


@pytest.mark.parametrize("bad", [dict(student="coder", teacher="qwen"),
                                 dict(student="coder", backend="llamacpp")])
def test_student_rejects_confounds(bad):
    with pytest.raises(SystemExit):
        ps.apply_teacher(ns(**bad), _Ap())


def test_coder_teacher_out():
    assert ps.teacher_out("qwen_coder", Path("data/corpora/nr3d_test.csv")) == \
        "data/parses/test/nr3d_test_teacher_qwen_coder_tree_v1_ext.jsonl"
