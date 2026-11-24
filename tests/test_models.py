"""Teacher/student registry and the llama.cpp chat-template guard."""

import argparse

import pytest

from grounding.parsing.llamacpp_backend import resolve_template_tokenizer
from grounding.parsing.models import (STUDENTS, TEACHERS, teacher_for_gguf,
                                      template_tokenizer)

QWEN, CODER = "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-Coder-30B-A3B-Instruct"


def test_registry():
    assert TEACHERS["qwen"].hf == QWEN
    assert TEACHERS["qwen_coder"].hf == CODER
    assert TEACHERS["qwen_coder"].gguf_file == "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
    assert STUDENTS["qwen3"].hf == "Qwen/Qwen3-1.7B"
    assert STUDENTS["coder"].hf == "Qwen/Qwen2.5-Coder-1.5B-Instruct"


@pytest.mark.parametrize("gguf, teacher", [
    ("Qwen3-30B-A3B-Q4_K_M.gguf", "qwen"),
    ("qwen3-coder-30b-a3b-instruct-q4_k_m.gguf", "qwen_coder"),
    ("Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf", "qwen_coder"),
    ("Llama-3-8B.gguf", None),
    (None, None),
])
def test_teacher_for_gguf(gguf, teacher):
    assert teacher_for_gguf(gguf) == teacher


def test_template_follows_served_gguf():
    # The historical default when nothing is known
    assert template_tokenizer(None) == QWEN
    assert template_tokenizer(None, gguf="Qwen3-30B-A3B-Q4_K_M.gguf") == QWEN
    assert template_tokenizer(None, gguf="Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf") == CODER
    assert template_tokenizer(None, teacher="qwen_coder") == CODER
    # an unknown GGUF keeps an explicit choice
    assert template_tokenizer("my/tok", gguf="other.gguf") == "my/tok"


@pytest.mark.parametrize("kw", [
    dict(explicit=QWEN, gguf="Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"),
    dict(explicit=CODER, gguf="Qwen3-30B-A3B-Q4_K_M.gguf"),
    dict(explicit=None, teacher="qwen_coder", gguf="Qwen3-30B-A3B-Q4_K_M.gguf"),
    dict(explicit=None, teacher="qwen", gguf="Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"),
])
def test_template_mismatch_is_refused(kw):
    explicit = kw.pop("explicit")
    with pytest.raises(ValueError):
        template_tokenizer(explicit, **kw)


def test_resolve_template_tokenizer_fills_args_and_exits_on_mismatch():
    args = argparse.Namespace(tokenizer=None)
    assert resolve_template_tokenizer(
        args, "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf") == CODER
    assert args.tokenizer == CODER
    with pytest.raises(SystemExit, match="Restart llama-server"):
        resolve_template_tokenizer(argparse.Namespace(tokenizer=None),
                                   "Qwen3-30B-A3B-Q4_K_M.gguf", teacher="qwen_coder")


def test_generate_rollouts_default_output():
    gr = pytest.importorskip("experiments.parser_adaptation.generate_rollouts")
    from pathlib import Path
    assert gr.default_output("qwen_coder", "llamacpp", "x.gguf", "v4") == \
        Path("data/rollouts/nr3d_rollouts_qwen_coder_v4.jsonl")
    # the thesis teacher keeps the historical names
    assert gr.default_output("qwen", "hf", QWEN, "v4") == \
        Path("data/rollouts/nr3d_rollouts_v4.jsonl")
    assert gr.default_output(None, "llamacpp", "Qwen3-30B-A3B-Q4_K_M.gguf", "v4") == \
        Path("data/rollouts/nr3d_rollouts_v4_Qwen3-30B-A3B-Q4_K_M.jsonl")
