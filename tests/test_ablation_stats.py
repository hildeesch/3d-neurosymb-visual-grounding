import itertools
import json

import numpy as np
import pytest

from experiments.ablation import stats as S
from experiments.ablation.design import Cell
from grounding.resolution.statistics import mcnemar


@pytest.mark.parametrize("wins,losses", [(0, 0), (3, 0), (12, 5), (40, 61), (250, 190)])
def test_two_cell_signflip_is_exact_mcnemar(wins, losses):
    d = np.array([1] * wins + [-1] * losses + [0] * 30, dtype=float)
    assert S.signflip_p(d) == pytest.approx(mcnemar(wins, losses), rel=1e-9)


@pytest.mark.parametrize("seed", range(5))
def test_signflip_matches_enumeration(seed):
    rng = np.random.default_rng(seed)
    d = rng.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=12)
    nz = d[d != 0]
    observed = abs(nz.sum())
    hits = sum(abs(np.dot(signs, nz)) >= observed - 1e-9
               for signs in itertools.product([-1, 1], repeat=nz.size))
    assert S.signflip_p(d) == pytest.approx(hits / 2 ** nz.size)


def test_holm():
    assert S.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_bootstrap_ci_brackets_mean():
    d = np.array([1] * 60 + [-1] * 20 + [0] * 920, dtype=float)
    lo, hi = S.bootstrap_ci(d, n_boot=2000, seed=0)
    assert lo < d.mean() < hi


def _design_cells():
    return [Cell("student", r, e) for r in ("extended", "lasp")
            for e in ("thesis", "lasp")]


def test_factorial_weights_2x2():
    design = {"representation": ("extended", "lasp"), "executor": ("thesis", "lasp")}
    cells = _design_cells()
    inter = S.factorial_weights(design, cells, ("representation", "executor"))
    assert {c.name: float(w) for c, w in inter.items()} == {
        "student__extended__thesis": 1.0, "student__extended__lasp": -1.0,
        "student__lasp__thesis": -1.0, "student__lasp__lasp": 1.0}
    main = S.factorial_weights(design, cells, ("representation",))
    assert sorted(float(w) for w in main.values()) == [-0.5, -0.5, 0.5, 0.5]


def test_factorial_effects_recover_known_did():
    # rows 0-99: everything right; rows 100-199: only the full cell is right,
    # i.e. representation and executor each help only together.
    cells = _design_cells()
    y = {c: np.zeros(1000, bool) for c in cells}
    for c in cells:
        y[c][:100] = True
    y[cells[0]][100:200] = True
    design = {"representation": ("extended", "lasp"), "executor": ("thesis", "lasp")}
    effects = {e["effect"]: e for e in S.factorial_effects(design, y, n_boot=500)}
    assert effects["representation"]["estimate_pp"] == pytest.approx(5.0)
    assert effects["executor"]["estimate_pp"] == pytest.approx(5.0)
    assert effects["representation x executor"]["estimate_pp"] == pytest.approx(10.0)
    assert effects["representation x executor"]["p_exact"] < 1e-20


def test_load_correct_counts_absent_rows_as_misses(tmp_path):
    dump = tmp_path / "d.jsonl"
    dump.write_text("\n".join(json.dumps({"dataset_idx": i, "correct": i % 2 == 0})
                              for i in (0, 1, 2)) + "\n")
    assert S.load_correct(dump, [0, 1, 2, 3]).tolist() == [True, False, True, False]
    with pytest.raises(ValueError):
        S.load_correct(dump, [0, 1])
