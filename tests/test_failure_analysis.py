"""Failure categories, leave-one-out blame, cue bookkeeping, and the script."""
import csv
import json
import sys
from types import SimpleNamespace

import torch

from grounding.resolution import resolver
from grounding.resolution.failure_analysis import (classify, colour_words,
                                                   dropped_cues, program_cues)

LABELS = ["chair", "chair", "chair", "table", "armchair"]


def cat(vec, name="chair"):
    return {"kind": "cat", "name": name, "vec": torch.tensor(vec), "depth": 0}


def rel(name, vec):
    return {"kind": "rel", "name": name, "vec": torch.tensor(vec), "depth": 0}


CHAIR = [0.33, 0.33, 0.33, 1e-9, 0.01]


def run(top, target=0, pred=1, meta=None):
    appearance = torch.minimum(torch.ones(5), top[0]["vec"])
    score = appearance.clone()
    for f in top[1:]:
        score = score * f["vec"]
    rels = [f for f in top if f["kind"] == "rel"]
    meta = meta or [{"negated": False, "multiplicity": None} for _ in rels]
    return classify(target=target, pred=pred, labels=LABELS, score=score, top=top,
                    rel_meta=meta)


def test_class_categories():
    assert run([cat([1e-9, 1e-9, 1e-9, 1.0, 1e-9], "table")], pred=3)["category"] == "wrong_head"
    out = run([cat([0.3, 0.3, 0.3, 1e-9, 0.3])], pred=4)
    assert out["category"] == "other_class" and out["pred_label"] == "armchair"
    assert run([cat(CHAIR)])["category"] == "underspecified"
    assert run([cat(CHAIR), rel("near", [0.5, 0.5, 0.2, 0.1, 0.1])])["category"] == "tie"
    assert run([cat(CHAIR)], target=0, pred=0)["category"] == "underspecified"


def test_evidence_categories_and_decisive():
    wrong = run([cat(CHAIR), rel("left", [0.2, 0.9, 0.5, 0.1, 0.1]),
                 rel("near", [0.5, 0.5, 0.5, 0.1, 0.1])],
                meta=[{"negated": False, "multiplicity": 3},
                      {"negated": True, "multiplicity": 1}])
    assert wrong["category"] == "conditions_wrong"
    assert wrong["against"] == ["left"]
    left = wrong["factors"][0]
    assert left["decisive"] is False          # without it the chairs tie
    assert set(left["tags"]) == {"view_dependent", "ambiguous_anchor"}
    assert wrong["factors"][1]["tags"] == ["negated"]

    lost = run([cat(CHAIR), rel("near", [0.8, 0.4, 0.1, 0.1, 0.1]),
                rel("on", [0.3, 0.9, 0.1, 0.1, 0.1])])
    assert lost["category"] == "outweighed"
    on = next(f for f in lost["factors"] if f["name"] == "on")
    near = next(f for f in lost["factors"] if f["name"] == "near")
    assert on["decisive"] is True and near["decisive"] is False


def test_decisive_needs_a_strict_win():
    # Without "left" the chairs tie; the argmax would pick index 0 = target,
    # but a tie broken by index order is not credited
    out = run([cat(CHAIR), rel("left", [0.2, 0.9, 0.1, 0.1, 0.1])])
    assert out["factors"][0]["decisive"] is False
    out = run([cat(CHAIR), rel("left", [0.2, 0.9, 0.1, 0.1, 0.1]),
               rel("near", [0.9, 0.5, 0.5, 0.1, 0.1])])
    assert next(f for f in out["factors"] if f["name"] == "left")["decisive"] is True


def test_cues():
    program = {"category": "chair", "relations": [
        {"relation_name": "against wall", "objects": []},
        {"relation_name": "near", "objects": [{"category": "table", "relations": [
            {"relation_name": "larger", "objects": []}]}]}]}
    assert program_cues(program) == {"against_wall", "near", "large"}
    dropped = dropped_cues("the big chair against the wall near the table in the corner",
                           program)
    assert "at_corner" in dropped and "near" not in dropped and "large" not in dropped
    assert colour_words("The dark brown chair, by the door.") == ["brown", "dark"]


def test_script_end_to_end(tmp_path, monkeypatch):
    from experiments.error_analysis import categorise as script

    n = 6
    labels = ["chair", "chair", "chair", "table", "table", "armchair"]
    obj_ids = [10, 11, 12, 13, 14, 15]
    g = torch.Generator().manual_seed(1)

    def category(name):
        return torch.softmax(100 * torch.tensor([float(l == name) for l in labels]), 0)
    concepts = {"chair": category("chair"), "table": category("table"),
                "near": torch.rand(n, n, generator=g),
                "against wall": torch.rand(n, generator=g)}
    concepts["near"].fill_diagonal_(0)
    rel_num = {"near": 1, "against wall": 0}
    tree = SimpleNamespace(
        parse=lambda _s, prog, c, v: resolver.parse("", prog, c, v, rel_num=rel_num,
                                                    valid_relations=rel_num.keys()),
        device=torch.device("cpu"),
        get_all_relations=lambda p: {"near", "against wall"},
        get_all_categories=lambda p: {"chair", "table"},
        get_all_features=lambda *a: concepts)
    scene = {"inst_labels": labels, "obj_ids": obj_ids, "pred_locs": [0] * n}
    monkeypatch.setattr(script, "load_tree", lambda _r: tree)
    monkeypatch.setattr(script, "get_all_colors", lambda p: set())
    monkeypatch.setattr(script.engine, "Scenes", lambda: SimpleNamespace(
        load=lambda s: scene, drop=lambda s: None))

    def row(i, program, target, sentence="the chair"):
        return {"row_idx": i, "scan_id": "s", "target_id": target, "sentence": sentence,
                "json_obj": program}
    near_table = {"category": "chair", "relations": [
        {"relation_name": "near", "objects": [{"category": "table", "relations": []}]}]}
    rows = [row(0, None, 10),
            row(1, {"category": "chair", "relations": []}, 11, "the chair in the corner"),
            row(2, {"category": "table", "relations": []}, 10),
            row(3, near_table, 10), row(4, near_table, 11), row(5, near_table, 12)]
    parses = tmp_path / "p.jsonl"
    parses.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(sys, "argv", ["categorise", "--parses", str(parses),
                                      "--out", str(tmp_path / "out")])
    assert script.main() == 0
    report = json.loads((tmp_path / "out" / "p" / "report.json").read_text())
    assert report["n_rows"] == 6
    cats = report["categories"]
    assert cats["no_program"]["n"] == 1 and cats["wrong_head"]["n"] == 1
    assert cats["underspecified"]["n"] == 1
    # "in" is matched too: cue matching is lexical and counts generic prepositions
    assert ["at_corner", 1] in report["underspecified"]["top_unencoded_cues"]
    assert sum(v["n"] for v in cats.values()) == report["n_failures"]
    assert report["predicates"]["near"]["rows"] == 3
    with open(tmp_path / "out" / "p" / "failures.csv", encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == report["n_failures"]
