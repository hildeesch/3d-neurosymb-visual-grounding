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
    from grounding.parsing import sft_user_turn
    version, chat, user_msg, recover = ps.STYLES["rollout"]
    assert (version, chat, user_msg, recover) == ("v4", True, None, False)
    assert sft_user_turn("the red chair") == user_message("the red chair")
    # same post-processing as the student
    assert recover == ps.STYLES["sft"][3]


def test_teacher_out():
    assert ps.teacher_out("gpt4o", Path("data/corpora/nr3d_test.csv")) == \
        "data/parses/test/nr3d_test_teacher_gpt4o_v4_ext.jsonl"
