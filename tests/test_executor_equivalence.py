"""The thesis resolver, run with LaSP's arithmetic on LaSP's features, is LaSP.

This is what makes the executor factor of the ablation a one-change contrast:
the baseline tree's ``--executor thesis`` with ``softmax`` must reproduce the
upstream ``parse`` on every program, and with ``zscore_sigmoid`` it must not.
"""

import copy
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("openai")
pytest.importorskip("transformers")

lasp = pytest.importorskip("src.eval.eval_nr3d")

N_OBJECTS = 7
CATEGORIES = ["chair", "table", "lamp", "door"]
UNKNOWN = ["isolated", "smallest", "lowest", "made_up"]   # not in LaSP's table


def features(rng: random.Random):
    gen = torch.Generator().manual_seed(rng.randrange(1 << 30))
    feats = {c: torch.rand(N_OBJECTS, generator=gen) for c in CATEGORIES}
    for name, arity in lasp.rel_num.items():
        shape = (N_OBJECTS,) * (arity + 1)
        feats[name] = torch.rand(*shape, generator=gen)
    return feats


def program(rng: random.Random, depth: int = 0) -> dict:
    node = {"category": rng.choice(CATEGORIES), "relations": []}
    if rng.random() < 0.4:
        node["color"] = rng.choice(["red", "brown"])
    if depth < 2:
        for _ in range(rng.randint(0, 3)):
            name = rng.choice(list(lasp.rel_num) + UNKNOWN)
            arity = lasp.rel_num.get(name, 1)
            n_obj = rng.choice([0, 1, 2]) if arity == 2 else rng.choice([0, 1])
            rel = {"relation_name": name,
                   "objects": [program(rng, depth + 1) for _ in range(n_obj)]}
            if rng.random() < 0.2:
                rel["negative"] = True
            if rng.random() < 0.3:
                rel["rank"] = rng.choice([1, 2, 3])
            node["relations"].append(rel)
    return node


def run(scorer, json_obj, feats):
    valid = torch.arange(N_OBJECTS)
    return scorer("scene0000_00", copy.deepcopy(json_obj), feats, valid)


@pytest.mark.parametrize("seed", range(200))
def test_thesis_resolver_with_softmax_is_lasp(seed):
    rng = random.Random(seed)
    feats, prog = features(rng), program(rng)
    thesis, config = lasp.thesis_parse("softmax")
    assert config.use_rank is False and config.combine == "sum"
    expected = run(lasp.parse, prog, feats)
    got = run(thesis, prog, feats)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-7)


def test_thesis_calibration_changes_scores():
    """The executor did change: zscore_sigmoid gives different scores."""
    rng = random.Random(1)
    thesis, _ = lasp.thesis_parse("zscore_sigmoid")
    differs = 0
    for _ in range(50):
        feats, prog = features(rng), program(rng)
        if any(r["relation_name"] in lasp.ALL_VALID_RELATIONS
               for r in prog["relations"]):
            differs += not torch.allclose(run(thesis, prog, feats),
                                          run(lasp.parse, prog, feats))
    assert differs > 0


def test_lasp_features_carry_no_colour():
    """Colour drops out of the lasp representation because no key exists."""
    rng = random.Random(3)
    feats = features(rng)
    prog = {"category": "chair", "color": "red", "relations": []}
    thesis, _ = lasp.thesis_parse("zscore_sigmoid")
    torch.testing.assert_close(run(thesis, prog, feats), run(lasp.parse, prog, feats))
