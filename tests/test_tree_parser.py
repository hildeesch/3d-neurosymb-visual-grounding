"""The tree parser through the pipeline: constrained generation, rollouts ->
filter -> winners -> SFT rows, parse_split / score_checkpoint, and the ablation.

The grammar itself is tested in test_tree_grammar.py; here the question is
whether every stage uses it (and to_executable, never clean/lower), and
records it."""

from __future__ import annotations

import argparse
import json
import sys

import pytest

from grounding.parsing import tree as T
from grounding.parsing import cues


def _node(cat="chair", color=None, *rels):
    return {"category": cat, "color": color, "relations": list(rels)}


def _rel(name, *objs, rank=None, negative=False):
    return {"relation_name": name, "negative": negative, "rank": rank, "objects": list(objs)}


GOOD = _node("chair", None, _rel("near", _node("desk", None, _rel("below", _node("monitor")))))
FAR = _node("chair", None, _rel("far", _node("desk")))
SYNONYM = _node("chair", None, _rel("next to", _node("desk")))
REORDERED = {"color": None, "category": "chair",
             "relations": [_rel("near", _node("table"))]}


# ------------------------------------------------------------------ helpers

def test_grammar_for_and_record():
    assert T.grammar_for("v4") is None
    assert T.grammar_for("tree_v1", enabled=False) is None
    g = T.grammar_for("tree_v1")
    assert g == T.gbnf()
    rec = T.grammar_record(g)
    assert rec["format"] == "gbnf" and len(rec["sha256"]) == 64
    assert T.grammar_record(None) is None
    assert T.grammar_record(T.gbnf(2)) != rec


def test_in_layout_size_and_names():
    assert T.in_layout(GOOD) and not T.in_layout(REORDERED)
    T.to_executable(REORDERED)                 # valid, only the order differs
    assert T.size(GOOD) == 4                   # near+desk, below+monitor
    assert T.relation_names(GOOD) == ["near", "below"]
    assert T.depth_of(GOOD) == 3


def test_tree_cue_names_are_tree_relations():
    from grounding.parsing.cues import _CONCEPTS
    assert set(cues.TREE_NAME_OF_CUE) == set(_CONCEPTS)
    assert set(cues.TREE_NAME_OF_CUE.values()) <= set(T.RELATIONS)


def test_tree_cue_names_share_the_cue_encoder():
    pytest.importorskip("torch")
    from grounding.parsing.lowering import normalise_predicate
    from grounding.representation.pipeline import CONCEPTS, rel_num
    for cue, name in cues.TREE_NAME_OF_CUE.items():
        lowered = normalise_predicate(cue, rel_num)
        assert lowered is not None, cue
        assert CONCEPTS[lowered] is CONCEPTS[name], (cue, lowered, name)


def test_tree_coverage_counts_every_level():
    u = "the chair next to the desk with the monitor on it"
    assert cues.coverage(GOOD, u) == 1          # near; "on" is above, not below
    on = _node("chair", None, _rel("near", _node("desk", None,
                                                 _rel("above", _node("monitor")))))
    assert cues.coverage(on, u) == 2
    assert cues.predicate_count(GOOD) == 2


def test_the_package_is_the_tree_parser():
    import grounding.parsing as P
    assert P.TREE_PROMPTS == {"tree_v1": T.SYSTEM_PROMPT_TREE_V1}
    assert P.DEFAULT_PROMPT == "tree_v1"
    for gone in ("PROMPTS", "clean_program", "lower_program", "lower_one"):
        assert not hasattr(P, gone), gone


# ------------------------------------------------------------ llama.cpp

def test_llamacpp_sends_grammar(monkeypatch):
    from grounding.parsing.llamacpp_backend import LlamaCppServer
    bodies = []

    def fake(self, path, body=None):
        bodies.append(body)
        return {"content": T.dumps(GOOD), "tokens_predicted": 3}

    monkeypatch.setattr(LlamaCppServer, "_request", fake)
    server = LlamaCppServer(workers=2)
    out = server.complete_many([("p", 1), ("q", 2)], max_tokens=8, temperature=0.7,
                               grammar=T.gbnf())
    assert [o.text for o in out] == [T.dumps(GOOD)] * 2
    assert all(b["grammar"] == T.gbnf() for b in bodies)
    server.complete("p", max_tokens=8, temperature=0.0, seed=None)
    assert "grammar" not in bodies[-1]


# ------------------------------------------------------------ transformers

@pytest.fixture(scope="module")
def tiny_lm():
    """A one-layer random Qwen2 over a byte-level tokenizer, all logits equal."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("xgrammar")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {c: i for i, c in enumerate(alphabet)}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(["<|im_end|>"])
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|im_end|>")
    tokenizer.padding_side = "left"
    config = transformers.Qwen2Config(
        vocab_size=len(tokenizer) + 5,            # padded logits, as in Qwen
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096,
        tie_word_embeddings=False)
    torch.manual_seed(0)
    model = transformers.Qwen2ForCausalLM(config).eval()
    with torch.no_grad():
        model.lm_head.weight.zero_()              # uniform over what the grammar allows
    return model, tokenizer


def test_hf_greedy_under_grammar_is_a_valid_tree(tiny_lm):
    from grounding.parsing.generation import Sampler, batch_generate
    model, tok = tiny_lm
    prompts = ["Sentence: the chair\n\nParsed JSON:", "x"]
    free = batch_generate(model, tok, prompts, max_tokens=40, sampler=Sampler())
    assert all(T.parse_generation(t)[2] != "ok" for t in free.texts)
    out = batch_generate(model, tok, prompts, max_tokens=200, sampler=Sampler(),
                         grammar=T.gbnf())
    for text in out.texts:
        tree, exe, outcome = T.parse_generation(text)
        assert outcome == "ok", text
        assert T.in_layout(tree)


def test_hf_sampling_under_grammar_stays_in_the_grammar(tiny_lm):
    import xgrammar as xgr
    from grounding.parsing.generation import Sampler, batch_generate
    model, tok = tiny_lm
    grammar = xgr.Grammar.from_ebnf(T.gbnf())
    out = batch_generate(model, tok, ["a", "bb", "ccc", "d"], max_tokens=400,
                         sampler=Sampler(temperature=1.0), seed=3, grammar=T.gbnf())
    finished = 0
    for text in out.texts:
        tree, exe, outcome = T.parse_generation(text)
        if outcome == "ok":
            finished += 1
            continue
        # cut off by max_tokens: still a prefix of the grammar
        matcher = xgr.GrammarMatcher(xgr.GrammarCompiler(
            xgr.TokenizerInfo.from_huggingface(tok, vocab_size=model.config.vocab_size)
        ).compile_grammar(grammar))
        assert matcher.accept_string(text), text
    assert finished >= 1


# ------------------------------------------------------------ rollouts

gr = pytest.importorskip("experiments.parser_adaptation.generate_rollouts")


def test_rollout_samples_keep_the_tree_and_its_outcome():
    row = {"row_idx": 1, "stimulus_id": "s", "scan_id": "scene", "target_id": "3",
           "instance_type": "chair", "utterance": "u"}
    rec = gr.row_record(row, [T.dumps(GOOD), T.dumps(SYNONYM), "nope"])
    assert [s["outcome"] for s in rec["samples"]] == ["ok", "relation_name", "no_json"]
    assert rec["samples"][0]["parsed"] == GOOD          # as emitted, nulls included
    assert rec["samples"][1]["parsed"] == SYNONYM       # rejected, not repaired


def _args(**kw):
    base = dict(input="in.csv", backend="llamacpp", teacher="qwen", model="m.gguf",
                prompt="tree_v1", n_samples=8, temp=0.7, max_tokens=320, seed=0,
                tokenizer="Qwen/Qwen3-30B-A3B", grammar=T.gbnf())
    base.update(kw)
    return argparse.Namespace(**base)


def test_rollout_resume_refuses_a_grammar_change(tmp_path):
    system = T.SYSTEM_PROMPT_TREE_V1
    meta_path = tmp_path / "r.jsonl.meta.json"
    with_grammar = gr.invocation(_args(), system)
    assert with_grammar["grammar"] == T.grammar_record(T.gbnf())
    gr.check_resume(meta_path, with_grammar, have_rows=False)
    gr.check_resume(meta_path, gr.invocation(_args(), system), have_rows=True)
    with pytest.raises(SystemExit, match="grammar"):
        gr.check_resume(meta_path, gr.invocation(_args(grammar=None), system), True)


def test_rollouts_refuse_openai_with_grammar(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["g", "--backend", "openai", "--prompt", "tree_v1"])
    with pytest.raises(SystemExit):
        gr.main()


# ------------------------------------------------ filter -> winners -> sets

class _Scenes:
    def drop(self, scan_id):
        pass


def _fake_score(scenes, scan_id, program):
    """Objects 1 (target) and 2. ``near`` anywhere in the tree picks the target."""
    assert "head" not in program and "color" not in program   # executable, nulls dropped
    names = T.relation_names(program)
    return [1, 2], ([0.9, 0.1] if "near" in names else [0.1, 0.9])


def _patch_engine(monkeypatch, module):
    monkeypatch.setattr(module.engine, "bind", lambda rep: None)
    monkeypatch.setattr(module.engine, "bound", lambda: "extended")
    monkeypatch.setattr(module, "Scenes", _Scenes)
    monkeypatch.setattr(module, "score", _fake_score)


@pytest.fixture
def tree_pool(tmp_path):
    samples = [T.dumps(GOOD), T.dumps(GOOD), T.dumps(SYNONYM), "no json",
               T.dumps(FAR), T.dumps(REORDERED)]
    row = {"row_idx": 7, "stimulus_id": "s", "scan_id": "scene0000_00",
           "target_id": "1", "instance_type": "chair",
           "utterance": "the chair next to the desk with the monitor on it"}
    rec = gr.row_record(row, samples)
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(json.dumps(rec) + "\n")
    meta = {"prompt": "tree_v1", "grammar": T.grammar_record(T.gbnf())}
    (tmp_path / "rollouts.jsonl.meta.json").write_text(json.dumps(meta))
    return rollouts


def test_filter_uses_the_grammar_check_not_clean_lower(tree_pool, tmp_path, monkeypatch):
    fr = pytest.importorskip("experiments.parser_adaptation.filter_rollouts")
    _patch_engine(monkeypatch, fr)
    assert "clean_program" not in open(fr.__file__).read()
    out = tmp_path / "keepers.jsonl"
    monkeypatch.setattr(sys, "argv", ["f", "--rollouts", str(tree_pool),
                                      "--output", str(out)])
    assert fr.main() == 0
    kept = [json.loads(l) for l in out.read_text().splitlines()]
    assert [k["program"] for k in kept] == [GOOD, REORDERED]   # trees as emitted
    stats = json.loads((tmp_path / "keepers.jsonl.stats.json").read_text())
    assert stats["prompt"] == "tree_v1" and stats["representation"] == "extended"
    assert stats["grammar"] == T.grammar_record(T.gbnf())
    assert stats["outcomes"]["invalid_tree"] == 1 and stats["outcomes"]["wrong_rank"] == 1
    assert stats["outcomes"]["duplicate"] == 1 and stats["outcomes"]["parse_null"] == 1
    t = stats
    assert t["invalid_by_reason"] == {"relation_name": 1}
    assert t["sample_outcomes"] == {"ok": 4, "relation_name": 1, "no_json": 1}
    assert t["sample_invalid_rate"] == pytest.approx(2 / 6)
    assert t["depth_of_kept"] == {"2": 1, "3": 1}


def test_filter_refuses_a_v4_pool_and_names_keepers(tree_pool, tmp_path, monkeypatch):
    fr = pytest.importorskip("experiments.parser_adaptation.filter_rollouts")
    _patch_engine(monkeypatch, fr)
    meta = tmp_path / "rollouts.jsonl.meta.json"
    meta.write_text(json.dumps({"prompt": "v4"}))
    monkeypatch.setattr(sys, "argv", ["f", "--rollouts", str(tree_pool), "--output",
                                      str(tmp_path / "k.jsonl")])
    with pytest.raises(SystemExit, match="not a tree prompt"):
        fr.main()
    assert fr.default_keepers("data/rollouts/nr3d_rollouts_qwen_coder_tree_v1.jsonl",
                              "extended") == \
        "data/distilled/nr3d_keepers_qwen_coder_tree_v1_ext.jsonl"


def test_winner_pool_keeps_the_tree_and_rejects_layout(tree_pool, tmp_path, monkeypatch):
    fr = pytest.importorskip("experiments.parser_adaptation.filter_rollouts")
    _patch_engine(monkeypatch, fr)
    keepers = tmp_path / "keepers.jsonl"
    monkeypatch.setattr(sys, "argv", ["f", "--rollouts", str(tree_pool),
                                      "--output", str(keepers)])
    fr.main()

    bwp = pytest.importorskip("experiments.parser_adaptation.build_winner_pool")
    _patch_engine(monkeypatch, bwp)
    monkeypatch.setattr(bwp, "difficulty_table", lambda path: {})
    winners = tmp_path / "winners.jsonl"
    monkeypatch.setattr(sys, "argv", ["w", "--keepers", str(keepers),
                                      "--output", str(winners)])
    assert bwp.main() == 0
    (w,) = [json.loads(l) for l in winners.read_text().splitlines()]
    assert w["program"] == GOOD and w["depth"] == 3
    assert w["size"] == 4 and w["predicate_count"] == 2 and w["coverage"] == 1
    stats = json.loads((tmp_path / "winners.jsonl.stats.json").read_text())
    assert stats["prompt"] == "tree_v1"
    assert stats["rejections"] == {"key_order": 1}
    assert stats["depth_of_winners"] == {"3": 1}

    # keepers of a v4 pool are refused
    k_stats = tmp_path / "keepers.jsonl.stats.json"
    k_stats.write_text(json.dumps({**json.loads(k_stats.read_text()), "prompt": "v4"}))
    with pytest.raises(SystemExit, match="not a tree prompt"):
        bwp.main()


def test_sft_rows_have_the_tree_as_target(tmp_path, monkeypatch):
    bus = pytest.importorskip("experiments.parser_adaptation.build_uniform_sets")
    split = tmp_path / "validation_split.json"
    split.write_text(json.dumps({"valid_prompt_ids": [0, 1], "covered_prompts": 12}))
    monkeypatch.setattr(bus, "VALID_PROMPTS_FILE", split)
    monkeypatch.setattr(bus, "SIZE_MATCHED_PROMPTS", 6)
    pool = tmp_path / "winners.jsonl"
    with pool.open("w") as f:
        for p in range(12):
            f.write(json.dumps({
                "row_idx": p, "scan_id": "scene0000_00", "utterance": f"chair {p}",
                "program": GOOD, "margin": 0.5, "winner_rank": 0, "source": "original",
                "is_hard": p % 3 == 0, "size": 4, "word_count": 2, "coverage": 1,
                "predicate_count": 2, "depth": 3}) + "\n")
    (tmp_path / "winners.jsonl.stats.json").write_text(json.dumps({"prompt": "tree_v1"}))
    out = tmp_path / "sets"
    monkeypatch.setattr(sys, "argv", ["b", "--pool", str(pool), "--out", str(out),
                                      "--only", bus.RANDOM_SIZE_MATCHED])
    assert bus.main() == 0
    rows = [json.loads(l) for l in
            (out / bus.RANDOM_SIZE_MATCHED / "train.jsonl").read_text().splitlines()]
    assert len(rows) == 12
    for r in rows:
        system, user, assistant = r["messages"]
        assert system["content"] == T.SYSTEM_PROMPT_TREE_V1
        assert user["content"] == f"Sentence: chair {r['row_idx']}\n\nParsed JSON:"
        assert assistant["content"] == T.dumps(GOOD)
        assert T.parse_generation(assistant["content"])[2] == "ok"
    stats = json.loads((out / bus.RANDOM_SIZE_MATCHED / "stats.json").read_text())
    assert stats["frozen_valid_prompts_uncovered"] == 0
    assert stats["mean_executable_predicates_per_row"] == 2
    assert stats["multi_constraint_rate_rows"] == 1.0

    # a pool of v4 programs is refused
    (tmp_path / "winners.jsonl.stats.json").write_text(json.dumps({"prompt": "v4"}))
    monkeypatch.setattr(sys, "argv", ["b", "--pool", str(pool), "--out", str(out),
                                      "--force"])
    with pytest.raises(SystemExit, match="tree_v1"):
        bus.main()


def test_train_schedule_only_needs_the_arms_it_runs(tmp_path):
    tsa = pytest.importorskip("experiments.parser_adaptation.train_sft_arms")
    arm = tsa.RANDOM_SIZE_MATCHED
    (tmp_path / arm).mkdir()
    (tmp_path / arm / "train.jsonl").write_text("{}\n" * 8)
    plan = tsa.schedule(tmp_path, 10, [arm])
    assert {a["name"] for a in plan} == {arm} and len(plan) == len(tsa.TRAIN_SEEDS)
    with pytest.raises(SystemExit, match="missing"):
        tsa.schedule(tmp_path, 10)


# ------------------------------------------------ parse_split / scoring

ps = pytest.importorskip("experiments.parser_adaptation.parse_split")


def test_tree_teacher_out_names():
    from pathlib import Path
    rows = Path("data/corpora/nr3d_test.csv")
    assert ps.teacher_out("qwen", rows) == \
        "data/parses/test/nr3d_test_teacher_qwen_tree_v1_ext.jsonl"
    assert ps.teacher_out("qwen", Path("data/corpora/nr3d_dev.csv"), "tree_v1", False) == \
        "data/parses/dev/nr3d_dev_teacher_qwen_tree_v1_nogrammar_ext.jsonl"


@pytest.mark.parametrize("argv,match", [
    (["--teacher", "gpt4o"], "no-grammar"),
    (["--style", "teacher"], "invalid choice"),
    (["--prompt", "v4"], "invalid choice"),
])
def test_parse_split_tree_argument_checks(argv, match, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["p", "--representation", "extended", *argv])
    with pytest.raises(SystemExit):
        ps.main()
    assert match in capsys.readouterr().err


def test_parse_split_meta_records_grammar():
    from grounding.paths import REPO_ROOT
    args = argparse.Namespace(limit=0, backend="llamacpp", model="m.gguf", adapter=None,
                              teacher="qwen", student=None, style="rollout",
                              max_tokens=320, batch_size=8, no_think=False,
                              tokenizer="Qwen/Qwen3-30B-A3B", seed=0)
    rows = REPO_ROOT / "experiments/parser_adaptation/validation_split.json"
    meta = ps.invocation(args, rows, "tree_v1", T.SYSTEM_PROMPT_TREE_V1, "extended",
                         3, T.gbnf())
    assert meta["grammar"] == T.grammar_record(T.gbnf())
    assert ps.invocation(args, rows, "v4", "S", "extended", 3)["grammar"] is None


def test_score_checkpoint_rescores_trees(tmp_path, monkeypatch):
    sc = pytest.importorskip("experiments.parser_adaptation.score_checkpoint")
    picks = [{"dataset_idx": i, "scan_id": "scene0000_00", "target_id": 1,
              "target_label": "chair", "utterance": "u", "is_hard": False}
             for i in range(3)]
    monkeypatch.setattr(sc, "pick_rows", lambda *a: [dict(p) for p in picks])
    _patch_engine(monkeypatch, sc)
    raws = [T.dumps(GOOD), T.dumps(FAR), T.dumps(SYNONYM)]
    parses = tmp_path / "parses.jsonl"
    parses.write_text("".join(json.dumps({"row_idx": i, "dataset_idx": i, "raw": r}) + "\n"
                              for i, r in enumerate(raws)))
    grammar = T.grammar_record(T.gbnf())
    (tmp_path / "parses.jsonl.meta.json").write_text(
        json.dumps({"prompt": "tree_v1", "grammar": grammar}))
    out = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["s", "--split", "dev", "--n", "0", "--rescore",
                                      str(parses), "--prompt", "tree_v1", "--out", str(out)])
    assert sc.main() == 0
    result = json.loads(out.read_text())
    assert result["summary"]["model_top1"] == 1
    assert result["summary"]["tree_outcomes"] == {"ok": 2, "relation_name": 1}
    assert result["grammar"] == grammar and result["prompt"] == "tree_v1"
    by_idx = {r["dataset_idx"]: r for r in result["rows"]}
    assert by_idx[2]["outcome"] == "relation_name" and by_idx[2]["model"] is False

    # generations of the v4 parser are not trees
    (tmp_path / "parses.jsonl.meta.json").write_text(json.dumps({"prompt": "v4"}))
    with pytest.raises(SystemExit, match="not a tree prompt"):
        sc.main()


# ------------------------------------------------------------------ ablation

def test_ablation_teacher_commands():
    from experiments.ablation import design as D
    cmd = D.teacher_parse_command("teacher_qwen")
    assert "--no-grammar" not in cmd
    assert "--no-grammar" in D.teacher_parse_command("teacher_qwen", constrained=False)
    assert D.PARSERS["teacher_qwen_coder"] == \
        "data/parses/test/nr3d_test_teacher_qwen_coder_tree_v1_ext.jsonl"
    assert D.PARSERS["student"] == "data/parses/test/nr3d_test_student_tree_v1_ext.jsonl"


def test_ablation_results_never_overwrite_v4(capsys):
    from experiments.ablation import cli
    args = cli.build_parser().parse_args(["plan", "--parsers", "teacher_qwen", "student"])
    assert cli.out_directory(args, "test").name == "test_tree_v1"
    free = cli.build_parser().parse_args(
        ["plan", "--parsers", "teacher_qwen", "student", "--no-grammar"])
    paths = cli.parses_paths(free)
    assert paths["teacher_qwen"].name == "nr3d_test_teacher_qwen_tree_v1_nogrammar_ext.jsonl"
    assert paths["student"].name == "nr3d_test_student_tree_v1_nogrammar_ext.jsonl"
    assert cli.out_directory(free, "test").name == "test_tree_v1_nogrammar"
    cli.cmd_plan(free)
    assert "--teacher qwen --rows data/corpora/nr3d_test.csv --no-grammar" in \
        capsys.readouterr().out
