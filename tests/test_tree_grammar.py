"""Tree track: the grammar, the post-hoc check and the prompt agree with each other
and with the extended vocabulary, and deep trees execute."""

from __future__ import annotations

import copy
import json
import random
import re

import pytest

from grounding.parsing import tree as T


# ---------------------------------------------------------------- vocabulary

def test_relations_are_extended_vocabulary_names_with_matching_arity():
    from grounding.representation.pipeline import rel_num
    for name, spec in T.RELATIONS.items():
        assert name in rel_num, name
        # rel_num counts anchors the encoder relates; a binary encoder may be
        # used without objects (compared within the node's own category).
        expected = 2 if name in ("between", "middle") else spec.max_objects
        assert rel_num[name] == expected, name


def test_relations_cover_each_encoder_once():
    from grounding.representation.pipeline import CONCEPTS
    encoders = [CONCEPTS[n] for n in T.RELATIONS if n != "middle"]
    assert len(encoders) == len(set(encoders)), "two canonical names share an encoder"
    # every encoder of the extended vocabulary is reachable
    assert set(CONCEPTS.values()) == set(encoders)


def test_colours_are_scorable():
    from grounding.representation.encoders.color import _canonical
    for c in T.COLOURS:
        assert _canonical(c) == c, c


# -------------------------------------------------------------------- prompt

def test_prompt_lists_the_grammar_vocabulary():
    p = T.SYSTEM_PROMPT_TREE_V1
    unary = [n for n, s in T.RELATIONS.items() if s.max_objects == 0 and n != "middle"]
    binary = [n for n, s in T.RELATIONS.items() if (s.min_objects, s.max_objects) == (0, 1)]
    assert "a property of the node itself: " + ", ".join(unary) + "\n" in p
    assert "zero or one object: " + ", ".join(binary) + "." in p
    assert "color is one of: " + ", ".join(T.COLOURS) + ", or null." in p
    assert T.TREE_PROMPTS["tree_v1"] is p


@pytest.mark.parametrize("sentence,tree", T.TREE_EXAMPLES)
def test_prompt_examples_are_in_the_grammar(sentence, tree):
    assert f'Input: "{sentence}"\nOutput:\n{T.dumps(tree)}' in T.SYSTEM_PROMPT_TREE_V1
    T.to_executable(tree)


# ----------------------------------------------------------- post-hoc check

def _node(cat="chair", color=None, *rels):
    return {"category": cat, "color": color, "relations": list(rels)}


def _rel(name, *objs, rank=None, negative=False):
    return {"relation_name": name, "negative": negative, "rank": rank, "objects": list(objs)}


def test_to_executable_only_drops_nulls():
    t = _node("box", "red", _rel("above", _node("table", None, _rel("near", _node("window")))),
              _rel("larger", rank=1), _rel("left", rank=2, negative=True))
    assert T.to_executable(t) == {
        "category": "box", "color": "red", "relations": [
            {"relation_name": "above", "negative": False, "objects": [
                {"category": "table", "relations": [
                    {"relation_name": "near", "negative": False, "objects": [
                        {"category": "window", "relations": []}]}]}]},
            {"relation_name": "larger", "negative": False, "rank": 1, "objects": []},
            {"relation_name": "left", "negative": True, "rank": 2, "objects": []},
        ]}


BAD = {
    "relation_name": _node("chair", None, _rel("next to", _node("bed"))),
    "color": _node("chair", "grey"),
    "category": _node("Chair"),
    "rank_not_allowed": _node("chair", None, _rel("above", _node("bed"), rank=1)),
    "rank": _node("chair", None, _rel("left", rank=0)),
    "arity": _node("chair", None, _rel("between", _node("bed"))),
    "too_deep": _node("a", None, _rel("near", _node("b", None, _rel(
        "near", _node("c", None, _rel("near", _node("d"))))))),
    "node_keys": {"category": "chair", "relations": []},
    "relation_keys": _node("chair", None, {"relation_name": "near", "objects": []}),
    "negative": _node("chair", None, _rel("near", negative=None)),
}


@pytest.mark.parametrize("reason", sorted(BAD))
def test_to_executable_rejects_without_repair(reason):
    with pytest.raises(T.InvalidTree) as e:
        T.to_executable(BAD[reason])
    assert e.value.reason == reason


def test_bool_rank_is_rejected():
    with pytest.raises(T.InvalidTree):
        T.to_executable(_node("chair", None, _rel("left", rank=True)))


def test_parse_generation():
    good = T.dumps(T.TREE_EXAMPLES[0][1])
    obj, exe, outcome = T.parse_generation(good)
    assert outcome == "ok" and obj == T.TREE_EXAMPLES[0][1] and exe["category"] == "chair"
    # unconstrained output: prose around the object is skipped, the object is not edited
    assert T.parse_generation("Sure!\n```json\n" + good + "\n```")[2] == "ok"
    assert T.parse_generation("no json here") == (None, None, "no_json")
    bad = T.dumps(BAD["relation_name"])
    obj, exe, outcome = T.parse_generation(bad)
    assert exe is None and outcome == "relation_name" and obj == BAD["relation_name"]


# ------------------------------------------------------------------ grammar

def random_tree(rng: random.Random, depth: int = 1, max_depth: int = T.MAX_DEPTH) -> dict:
    words = ["chair", "table", "trash can", "kitchen cabinet door", "tv"]
    rels = []
    for _ in range(rng.randint(0, 3)):
        options = [n for n, s in T.RELATIONS.items()
                   if depth < max_depth or s.min_objects == 0]
        name = rng.choice(options)
        spec = T.RELATIONS[name]
        hi = 0 if depth == max_depth else spec.max_objects
        n_obj = rng.randint(spec.min_objects, hi)
        rels.append(_rel(name, *[random_tree(rng, depth + 1, max_depth) for _ in range(n_obj)],
                         rank=rng.choice([None, 1, 3]) if spec.ranked else None,
                         negative=rng.random() < 0.2))
    return _node(rng.choice(words), rng.choice([None, *T.COLOURS]), *rels)


def mutate(rng: random.Random, t: dict) -> dict:
    """One random edit that leaves the grammar (as far as to_executable can tell)."""
    t = copy.deepcopy(t)
    nodes = []

    def walk(n, d):
        nodes.append((n, d))
        for r in n["relations"]:
            for o in r["objects"]:
                walk(o, d + 1)
    walk(t, 1)
    node, d = rng.choice(nodes)
    kind = rng.choice(["name", "color", "rank", "arity", "category"])
    if kind == "color":
        node["color"] = "grey"
    elif kind == "category":
        node["category"] = "Chair2"
    else:
        r = _rel("near")
        node["relations"].append(r)
        if kind == "name":
            r["relation_name"] = "next to"
        elif kind == "rank":
            r["relation_name"], r["rank"] = "above", 2
        else:
            r["relation_name"] = "between"
            r["objects"] = [_node("bed")]
    return t



def _xgrammar_accepts():
    xgr = pytest.importorskip("xgrammar")
    grammar = xgr.Grammar.from_ebnf(T.gbnf())
    from xgrammar.testing import _is_grammar_accept_string
    return lambda s: _is_grammar_accept_string(grammar, s)


def test_grammar_and_check_agree_on_random_trees():
    accepts = _xgrammar_accepts()
    rng = random.Random(0)
    for _ in range(300):
        t = random_tree(rng)
        T.to_executable(t)
        text = T.dumps(t)
        assert accepts(text), text
        # the compact layout is in the grammar too
        assert accepts(json.dumps(t, separators=(",", ":"))), text
        bad = mutate(rng, t)
        with pytest.raises(T.InvalidTree):
            T.to_executable(bad)
        assert not accepts(T.dumps(bad)), T.dumps(bad)


def test_grammar_rejects_layout_it_does_not_allow():
    accepts = _xgrammar_accepts()
    t = T.TREE_EXAMPLES[0][1]
    assert not accepts(json.dumps(t, indent=1))            # no newlines
    assert not accepts(T.dumps(t) + " ")                   # nothing after the object
    reordered = {"color": None, "category": "chair", "relations": []}
    assert not accepts(T.dumps(reordered))                 # fixed key order


# ---------------------------------------------------------------- execution

def test_depth_three_tree_executes_and_its_third_level_matters():
    """An anchor's own relation changes which anchor is meant, and so the target."""
    torch = pytest.importorskip("torch")
    from grounding.representation.pipeline import ALL_VALID_RELATIONS, rel_num
    from grounding.resolution.resolver import parse, DEVICE

    # objects: 0 chair_a, 1 chair_b, 2 desk_a, 3 desk_b, 4 monitor (on desk_b)
    cat = {
        "chair": torch.tensor([1., 1., 0., 0., 0.]),
        "desk": torch.tensor([0., 0., 1., 1., 0.]),
        "monitor": torch.tensor([0., 0., 0., 0., 1.]),
    }
    near = torch.zeros(5, 5)
    near[0, 2] = near[2, 0] = 1.0   # chair_a near desk_a
    near[1, 3] = near[3, 1] = 1.0   # chair_b near desk_b
    below = torch.zeros(5, 5)
    below[3, 4] = 1.0               # desk_b below monitor
    concepts = {**{k: v.to(DEVICE) for k, v in cat.items()},
                "near": near.to(DEVICE), "below": below.to(DEVICE)}
    valid = torch.arange(5)

    flat = T.to_executable(_node("chair", None, _rel("near", _node("desk"))))
    deep = T.to_executable(_node("chair", None, _rel(
        "near", _node("desk", None, _rel("below", _node("monitor"))))))
    s_flat = parse("s", flat, concepts, valid, rel_num=rel_num,
                   valid_relations=ALL_VALID_RELATIONS)
    s_deep = parse("s", deep, concepts, valid, rel_num=rel_num,
                   valid_relations=ALL_VALID_RELATIONS)
    assert torch.isclose(s_flat[0], s_flat[1])          # two-level tree can't tell
    assert int(torch.argmax(s_deep)) == 1               # three levels pick chair_b
