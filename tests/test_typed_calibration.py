"""Typed calibration: grouping, mappings, mass alignment and the control rebuild."""
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from grounding.resolution import resolver
from grounding.resolution.condition_types import (
    appearance_typed, calibrate, colour_affinity, condition_group,
    match_relations, negate, traced_with_masses)
from grounding.resolution.config import DEFAULT_RESOLVER
from grounding.resolution.scoring import normalize_concept

N = 6
REL_NUM = {"near": 1, "close": 1, "on": 1, "against wall": 0, "left": 1,
           "larger": 0, "between": 2}
ERRORS = (NotImplementedError, ValueError, RuntimeError, KeyError, IndexError)


def scene(seed=0):
    g = torch.Generator().manual_seed(seed)
    labels = torch.tensor([0, 0, 0, 1, 1, 2])          # chair x3, table x2, lamp
    def category(k):
        return torch.softmax(100 * (labels == k).float(), dim=0)
    concepts = {"chair": category(0), "table": category(1), "lamp": category(2),
                "color:red": torch.rand(N, generator=g),
                "against wall": torch.rand(N, generator=g),
                "larger": torch.rand(N, generator=g),
                "between": torch.rand(N, N, N, generator=g)}
    for name in ("near", "close", "on", "left"):
        m = torch.rand(N, N, generator=g)
        m.fill_diagonal_(0)
        concepts[name] = m
    tree = SimpleNamespace(parse=lambda _scan, prog, c, v: resolver.parse(
        "", prog, c, v, rel_num=REL_NUM, valid_relations=REL_NUM.keys()))
    return tree, concepts, torch.arange(N)


def rel(name, *anchors, rank=None, negative=False):
    return {"relation_name": name, "objects": list(anchors), "rank": rank,
            "negative": negative}


def node(category, *relations, color=None):
    return {"category": category, "color": color, "relations": list(relations)}


PROGRAM = node("chair",
               rel("near", node("table", rel("against wall"))),
               rel("on", node("lamp")),
               rel("against wall", negative=True),
               rel("close", node("table"), rank=1),
               rel("left", node("table"), rank=2),
               rel("between", node("table"), node("lamp")),
               rel("unknown predicate"),
               color="red")


def test_groups():
    assert condition_group("near") == "relative"
    assert condition_group("near", {"rank": 1}) == "exclusive"
    assert condition_group("near", {"rank": True}) == "relative"
    assert condition_group("left|ranked") == "exclusive"
    assert condition_group("closest") == "exclusive"
    assert condition_group("isolated") == "exclusive"
    assert condition_group("on") == "absolute"
    assert condition_group("against wall") == "absolute"
    assert condition_group("on", {"rank": 2}) == "exclusive"


def test_engine_mappings_are_the_engines():
    raw = torch.rand(7)
    for mode in ("zscore_sigmoid", "zscore"):
        expected = normalize_concept(raw, DEFAULT_RESOLVER.updated(norm_mode=mode))
        assert torch.equal(calibrate(mode, raw), expected)
    flat = torch.full((5,), 0.3)
    assert torch.equal(calibrate("zscore_sigmoid", flat), torch.sigmoid(flat))


def test_raw_mapping_and_negation():
    raw = torch.tensor([0.0, 0.5, 2.0])
    out = calibrate("raw", raw, torch.tensor(2.0), floor=1e-3)
    assert torch.allclose(out, torch.tensor([1e-3, 0.25, 1.0]))
    assert torch.allclose(negate(out, "raw"), torch.tensor([0.999, 0.75, 1e-3]))
    assert torch.equal(calibrate("raw", raw, None, floor=0.1),
                       torch.tensor([0.1, 0.5, 1.0]))
    v = torch.tensor([0.2, 0.9])
    assert torch.equal(negate(v, "zscore"), v.max() - v)


def test_colour_category():
    category = torch.tensor([0.5, 0.5, 1e-9, 1e-9])
    colour = torch.tensor([0.2, 0.6, 0.9, 0.1])
    assert torch.equal(colour_affinity(colour, category, "raw"), colour)
    out = colour_affinity(colour, category, "category")
    # Members (0, 1) sit at -1 and +1 standard deviations
    assert torch.allclose(out[:2], torch.sigmoid(torch.tensor([-1.0, 1.0])), atol=1e-5)
    lone = torch.tensor([1.0, 1e-9, 1e-9, 1e-9])
    assert torch.equal(colour_affinity(colour, lone, "category"), colour)


def test_trace_masses_and_control_rebuild():
    tree, concepts, valid = scene()
    score, top, masses = traced_with_masses(tree, PROGRAM, concepts, valid, ERRORS)
    factors = [f for f in top if f["kind"] == "rel"]
    names = [f["name"] for f in factors]
    assert names == ["near", "on", "against wall", "close", "left|ranked", "between"]
    relations = match_relations(factors, PROGRAM)
    assert [r["relation_name"] for r in relations] == [
        "near", "on", "against wall", "close", "left", "between"]

    def ground(program):
        return tree.parse("", program, concepts, valid)

    expected = [ground(node("table", rel("against wall"))).sum(),
                ground(node("lamp")).sum(), None, ground(node("table")).sum(), None,
                ground(node("table")).sum() * ground(node("lamp")).sum()]
    for got, want in zip(masses, expected):
        if want is None:
            assert got is None
        else:
            assert torch.allclose(got, want)

    # The shipped arm rebuilt from the trace is the engine's score
    rebuilt = appearance_typed(top, DEFAULT_RESOLVER.color_mode, "raw")
    for f, r in zip(factors, relations):
        vec = calibrate(DEFAULT_RESOLVER.norm_mode, f["raw"])
        if r.get("negative") is True:
            vec = negate(vec, DEFAULT_RESOLVER.norm_mode)
        rebuilt = rebuilt * vec
    assert torch.allclose(rebuilt, score)
    groups = [condition_group(f["name"], r) for f, r in zip(factors, relations)]
    assert groups == ["relative", "absolute", "absolute", "exclusive", "exclusive",
                      "exclusive"]


def test_unmatched_factor_raises():
    with pytest.raises(ValueError):
        match_relations([{"name": "near", "key": "rel:near#1"}], node("chair"))


def test_script_end_to_end(tmp_path, monkeypatch):
    from experiments.calibration import typed_calibration as script

    tree, concepts, valid = scene()
    rows = [{"json_obj": PROGRAM, "target_id": 10 + t, "row_idx": t, "scan_id": "s"}
            for t in range(3)]
    rows.append({"json_obj": node("chair", rel("near", node("table")),
                                  rel("on", node("lamp"))),
                 "target_id": 11, "row_idx": 9, "scan_id": "s"})
    monkeypatch.setattr(script, "load_tree", lambda _rep: tree)
    monkeypatch.setattr(script, "load_rows", lambda _p: rows)
    monkeypatch.setattr(script, "engine", SimpleNamespace(Scenes=lambda: None))
    monkeypatch.setattr(script, "scene_batches", lambda *_a: iter(
        [(rows, concepts, valid, [10 + i for i in range(N)])]))
    monkeypatch.setattr(sys, "argv", ["typed_calibration", "--out", str(tmp_path),
                                      "--arm", "x=relative:released,colour:category"])
    assert script.main() == 0
    [path] = tmp_path.rglob("*.json")
    report = json.loads(path.read_text())
    assert report["control"]["agree"] == report["n_rows"] == 4
    assert set(report["overall"]["all rows"]) >= {"n", "shipped", "typed", "x"}
    assert report["coverage"]["absolute"] == {"on": 4, "against wall": 3}
    assert "has colour" in report["strata"]
