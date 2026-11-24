"""Typed calibration: each condition is calibrated according to its semantic type.

The shipped resolver maps every condition with the same z-score + sigmoid. Here
each top-level condition is first assigned to one of three groups, and each
group gets its own mapping:

  exclusive  superlatives, ordinals and other predicates with one intended
             referent ("closest", "largest", "second from the left",
             "between", "isolated"). More is always better, so a log-linear
             (softmax-shaped) contribution fits.
  relative   graded predicates read against the scene, with many candidates
             able to satisfy them ("near", "left", "taller"). These keep the
             shipped z-score + sigmoid.
  absolute   threshold/contact predicates with a physical meaning ("on",
             "under", "inside", "against wall", "on the floor", "corner").
             These get a fixed scale instead of a scene-relative one: the
             encoder's own [0, 1] value. For anchored relations it is first
             divided by the anchor mass, so that it is an expected relation
             value over the anchor candidates rather than a sum.

Colour is not a condition: it multiplies the category evidence with its raw
score. It can either stay raw or be standardised among the candidates of the
requested category ("the dark cabinet" read as "the darker cabinet").

The grouping is fixed by the predicate's meaning, not tuned on accuracy, so
that one dev pool can both choose and report it without fitting the table to it.

Under the product rule a constant scale per condition does not change the
argmax, so mixing softmax-scaled and sigmoid-scaled affinities is safe. It is
not safe under the mean rules, which is why only the product is used here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from grounding.representation.encoders.ordinal import membership
from grounding.resolution import scoring, tracing
from grounding.resolution.config import DEFAULT_RESOLVER

__all__ = ["ABSOLUTE_NAMES", "EXCLUSIVE_NAMES", "GROUPS", "MAPPINGS",
           "COLOUR_MAPPINGS", "condition_group", "match_relations",
           "calibrate", "negate", "colour_affinity", "appearance_typed",
           "traced_with_masses"]

GROUPS = ("exclusive", "relative", "absolute")
MAPPINGS = ("zscore_sigmoid", "zscore", "released", "raw")
COLOUR_MAPPINGS = ("raw", "category")

#: Predicate names (after lowering) that pick out one referent by themselves.
#: Any relation with a rank, including rank 1 ("closest to X" = "close", rank 1),
#: is exclusive as well.
EXCLUSIVE_NAMES = frozenset({
    "closest", "farthest", "furthest", "highest", "lowest", "smallest",
    "largest", "middle", "center", "between",
    "isolated", "isolation", "alone", "by itself", "lone",
})

#: Predicate names with a contact or containment reading: true or false in
#: physical terms, and several candidates may satisfy them at once.
ABSOLUTE_NAMES = frozenset({
    "on", "on top", "on the top", "above", "below", "under", "beneath",
    "in", "inside", "within", "attached",
    "against wall", "on the floor", "corner",
})

#: The resolver's own epsilon and constant-vector bypass (scoring.normalize_concept).
_ZS_EPS = 1e-6


def _has_rank(relation):
    rank = None if relation is None else relation.get("rank")
    return isinstance(rank, int) and not isinstance(rank, bool) and rank >= 1


def condition_group(label, relation=None):
    """Group of one traced relation factor; `label` is its trace name."""
    name = label.split("|")[0]
    if label.endswith("|ranked") or _has_rank(relation) or name in EXCLUSIVE_NAMES:
        return "exclusive"
    if name in ABSOLUTE_NAMES:
        return "absolute"
    return "relative"


def match_relations(factors, program):
    """The program relation behind each top-level relation factor, in order.

    The resolver visits ``program["relations"]`` in order and traces one factor
    per relation it can execute, so the factors are an ordered subsequence.
    """
    relations = list(program.get("relations") or [])
    out, i = [], 0
    for f in factors:
        name = f["name"].split("|")[0]
        while i < len(relations) and relations[i].get("relation_name") != name:
            i += 1
        if i == len(relations):
            raise ValueError(f"factor {f['key']} has no relation in the program")
        out.append(relations[i])
        i += 1
    return out


def calibrate(mapping, raw, mass=None, *, floor=1e-3, config=DEFAULT_RESOLVER):
    """One condition's affinities from its raw scores under `mapping`."""
    if mapping in ("zscore_sigmoid", "zscore"):
        # The engine's own arithmetic, so the control arm reproduces it exactly
        return scoring.normalize_concept(raw, config.updated(norm_mode=mapping))
    if mapping == "released":
        return F.softmax(raw, dim=0)
    if mapping == "raw":
        x = raw if mass is None or float(mass) <= _ZS_EPS else raw / mass
        return x.clamp(min=floor, max=1.0)
    raise ValueError(mapping)


def negate(vec, mapping, *, floor=1e-3):
    """Negation as the resolver applies it, or 1 - x on the absolute scale."""
    if mapping == "raw":
        return (1.0 - vec).clamp(min=floor, max=1.0)
    return vec.max() - vec


def colour_affinity(colour, category, mapping, *, threshold=0.5):
    """Colour factor: raw, or standardised among the category's candidates.

    With fewer than two category members, or no spread among them, there is
    nothing to compare against and the raw score is kept.
    """
    if mapping == "raw":
        return colour
    if mapping != "category":
        raise ValueError(mapping)
    members = membership(category) > threshold
    if int(members.sum()) < 2:
        return colour
    values = colour[members]
    sd = values.std(unbiased=False)
    if float(sd) < _ZS_EPS:
        return colour
    return torch.sigmoid((colour - values.mean()) / (sd + _ZS_EPS))


def appearance_typed(top, color_mode, colour_mapping):
    """``stage2.appearance_of`` with the colour factor mapped by `colour_mapping`."""
    out = category = None
    for f in top:
        if f["kind"] == "cat":
            category = f["vec"]
            out = torch.minimum(torch.ones_like(f["vec"]), f["vec"])
        elif f["kind"] == "color" and out is not None:
            value = colour_affinity(f["vec"], category, colour_mapping)
            out = out * value if color_mode == "product" else torch.minimum(out, value)
    return out


class _PositionedBinds(list):
    """Records, with each bind, how many factors had been traced before it."""

    def append(self, item):
        item["at"] = len(tracing._TRACE)
        super().append(item)


def _bind_mass(anchors, combine):
    if combine == "sum":
        mass = anchors[0].sum()
        for a in anchors[1:]:
            mass = mass * a.sum()
    else:
        mass = anchors[0].max()
        for a in anchors[1:]:
            mass = mass * a.max()
    return mass


def traced_with_masses(tree, program, concepts, valid, errors, config=DEFAULT_RESOLVER):
    """Score one program: (score or None, top-level trace, rel-factor anchor masses).

    The masses list is aligned with the top-level ``rel`` factors; an entry is
    None when that relation was not bound to anchors (unary encoders, and
    ranked relations whose raw score is already a rank score).
    """
    binds = _PositionedBinds()
    trace = tracing.start_trace(binds=binds)
    try:
        score = tree.parse("", program, concepts, valid)
    except errors:
        score = None
    finally:
        tracing.stop_trace()
    top_idx = [i for i, f in enumerate(trace) if f["depth"] == 0]
    rel_idx = [i for i in top_idx if trace[i]["kind"] == "rel"]
    mass_at = {}
    for b in binds:
        if b["depth"] != 0:
            continue
        # A relation's anchors are resolved, then bound, then its factor is traced
        owner = next((i for i in rel_idx if i >= b["at"]), None)
        if owner is not None and owner not in mass_at:
            mass_at[owner] = _bind_mass(b["anchors"], config.combine)
    masses = [None if trace[i]["name"].endswith("|ranked") else mass_at.get(i)
              for i in rel_idx]
    return score, [trace[i] for i in top_idx], masses
