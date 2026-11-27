"""The frozen validation split under a tree pool that need not cover every id."""

import json

import pytest

bus = pytest.importorskip("experiments.parser_adaptation.build_uniform_sets")


def winner(p, rank=0):
    return {"row_idx": p, "scan_id": "scene0000_00", "utterance": f"the chair {p}",
            "program": {"category": "chair", "color": None, "relations": []},
            "margin": 0.1 * (p % 5 + 1), "winner_rank": rank, "source": "original",
            "is_hard": p % 3 == 0, "size": 0, "word_count": 3, "coverage": 0,
            "predicate_count": 0}


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    path = tmp_path / "validation_split.json"
    path.write_text(json.dumps({"valid_prompt_ids": [0, 1, 99],
                                "covered_prompts": 13}))
    monkeypatch.setattr(bus, "VALID_PROMPTS_FILE", path)
    return path


def test_uncovered_validation_ids_are_reported_not_refused(frozen):
    assert bus.load_valid_prompts(list(range(12))) == {0, 1}
    assert bus.UNCOVERED_VALID == {99}
    assert bus.load_valid_prompts([0, 1, 99]) == {0, 1, 99}
    assert bus.UNCOVERED_VALID == set()


def test_random_size_matched_draws_the_fixed_size(frozen, monkeypatch, tmp_path):
    monkeypatch.setattr(bus, "SIZE_MATCHED_PROMPTS", 6)
    pool = {p: [winner(p)] for p in range(12)}
    stats = bus.build_random_size_matched(bus.RANDOM_SIZE_MATCHED, pool, "SYS", tmp_path)
    rows = [json.loads(l) for l in
            (tmp_path / bus.RANDOM_SIZE_MATCHED / "train.jsonl").read_text().splitlines()]
    prompts = {r["row_idx"] for r in rows}
    assert len(prompts) == 6 and len(rows) == 12      # 6 prompts x 2 rows
    assert not prompts & {0, 1, 99}                   # validation never trained on
    assert stats["frozen_valid_prompts_uncovered"] == 1
    assert "same training-set size" in stats["config"]["size_matched_to"]
    assert bus.RANDOM_SIZE_MATCHED == "sft_tree_ext_random_size_matched"
