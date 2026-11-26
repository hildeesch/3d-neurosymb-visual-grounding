"""Sort grounding failures into pipeline-ordered categories.

A failure is a row whose predicted object is not the annotated target. Each one
gets ONE primary category, the first that applies in pipeline order, so the
categories add up to the number of failures:

  no_program         the parser produced no executable program
  execution_error    the program raised during execution
  no_category        the head is a place word (corner, middle, room, center):
                     the resolver returns a flat score
  wrong_head         the target is not a member of the head category the program
                     asks for: the parser named the wrong object class, or CLIP
                     maps the head to another label
  other_class        the target's class is requested, but an object of ANOTHER
                     class won (near-synonym labels share category evidence)
  underspecified     same-class distractor won and the program has no condition
                     (no relation, no colour) to tell them apart
  tie                conditions exist but give the target and the winner the same
                     score; the argmax broke the tie against the target
  conditions_wrong   no condition prefers the target; at least one prefers the
                     winner (the evidence itself points at the wrong object)
  outweighed         some conditions prefer the target, others prefer the
                     winner, and the product chose the winner
  unexplained        none of the above (a guard; expected to stay empty)

Within the last three, each condition that prefers the winner is listed with
tags (view-dependent predicate, ambiguous anchor, ranked, negated, colour), and
it is "decisive" when leaving only that factor out makes the target the strict
winner (a tie does not count, since the argmax would break it by index).

Membership in a category is peak-relative category evidence above 0.5, the same
rule the resolver uses for anchors and ordinals.
"""

from __future__ import annotations

import torch

from grounding.parsing.cues import cue_set
from grounding.representation.encoders.ordinal import membership

__all__ = ["CATEGORIES", "EVIDENCE_CATEGORIES", "VIEW_DEPENDENT", "classify",
           "program_cues", "dropped_cues", "colour_words"]

CATEGORIES = ("no_program", "execution_error", "no_category", "wrong_head",
              "other_class", "underspecified", "tie", "conditions_wrong",
              "outweighed", "unexplained")
EVIDENCE_CATEGORIES = ("tie", "conditions_wrong", "outweighed")

#: Predicates whose truth depends on where the speaker stands.
VIEW_DEPENDENT = frozenset({"left", "right", "front", "back", "behind", "facing"})

MEMBER_THRESH = 0.5
_REL_TOL = 1e-6

#: Lowered relation names -> the v4 cue namespace of ``parsing.cues``.
_LOWERED_TO_CUE = {
    "larger": "large", "bigger": "large", "largest": "large", "longer": "large",
    "smaller": "small", "shorter": "small", "smallest": "small",
    "higher": "high", "taller": "high", "highest": "high", "upper": "high",
    "lower": "low", "lowest": "low",
    "on the floor": "on_the_floor", "against wall": "against_wall",
    "corner": "at_corner", "isolation": "isolated", "alone": "isolated",
    "by itself": "isolated", "lone": "isolated",
    "closer": "close", "closest": "close", "farthest": "far", "furthest": "far",
    "on the top": "on top", "cross": "across",
}

_COLOUR_WORDS = ("black", "white", "gray", "grey", "brown", "red", "orange",
                 "yellow", "green", "blue", "purple", "pink", "beige", "tan",
                 "silver", "metal", "metallic", "wooden", "wood", "dark", "light",
                 "bright")


def _walk(node):
    if not isinstance(node, dict):
        return
    yield node
    for relation in node.get("relations") or []:
        for child in relation.get("objects") or relation.get("anchors") or []:
            yield from _walk(child)


def program_cues(program):
    """Cue concepts a lowered program encodes, at every level."""
    out = set()
    for node in _walk(program):
        for relation in node.get("relations") or []:
            name = str(relation.get("relation_name"))
            out.add(_LOWERED_TO_CUE.get(name, name))
    return out


def dropped_cues(sentence, program):
    """Cue concepts the sentence names that the program does not encode."""
    return sorted(cue_set(sentence or "") - program_cues(program or {}))


def colour_words(sentence):
    words = set((sentence or "").lower().replace(",", " ").replace(".", " ").split())
    return sorted(words & set(_COLOUR_WORDS))


def _prefers(t, p):
    scale = max(abs(t), abs(p), 1e-12)
    if p - t > _REL_TOL * scale:
        return "winner"
    if t - p > _REL_TOL * scale:
        return "target"
    return "neutral"


def classify(*, target, pred, labels, score, top, rel_meta):
    """Primary category and evidence detail for one failure.

    target, pred  candidate indices (positions in the scene's object list)
    labels        semantic label per candidate
    score         the engine's final score vector
    top           depth-0 trace factors (kind, name, vec), as ``tracing`` records
    rel_meta      per depth-0 ``rel`` factor: {"negated": bool, "multiplicity": int|None}
    """
    cat = next((f for f in top if f["kind"] == "cat"), None)
    if cat is None:
        return {"category": "no_category", "factors": [], "against": []}
    member = membership(cat["vec"])
    detail = {"head": cat["name"], "target_label": labels[target],
              "pred_label": labels[pred],
              "target_membership": round(float(member[target]), 4)}
    if float(member[target]) < MEMBER_THRESH:
        return {"category": "wrong_head", **detail, "factors": [], "against": []}
    if labels[pred] != labels[target]:
        return {"category": "other_class", **detail, "factors": [], "against": []}

    evidence = [f for f in top if f["kind"] in ("rel", "color")]
    if not evidence:
        return {"category": "underspecified", **detail, "factors": [], "against": []}

    appearance = torch.minimum(torch.ones_like(cat["vec"]), cat["vec"])
    rel_iter = iter(rel_meta)
    factors = []
    for f in evidence:
        meta = next(rel_iter) if f["kind"] == "rel" else {}
        name = f["name"].split("|")[0]
        t, p = float(f["vec"][target]), float(f["vec"][pred])
        tags = []
        if f["kind"] == "color":
            tags.append("colour")
        if name in VIEW_DEPENDENT:
            tags.append("view_dependent")
        if f["name"].endswith("|ranked"):
            tags.append("ranked")
        if meta.get("negated"):
            tags.append("negated")
        if (meta.get("multiplicity") or 0) >= 2:
            tags.append("ambiguous_anchor")
        factors.append({"kind": f["kind"], "name": f["name"], "target": round(t, 6),
                        "winner": round(p, 6), "prefers": _prefers(t, p),
                        "tags": tags, "vec": f["vec"]})

    # Leave-one-out: would the target win without this factor?
    for i, fac in enumerate(factors):
        s = appearance.clone()
        for j, other in enumerate(factors):
            if j != i:
                s = s * other["vec"]
        others = torch.cat([s[:target], s[target + 1:]])
        fac["decisive"] = (fac["prefers"] == "winner"
                           and bool((s[target] > others).all()))
    for fac in factors:
        del fac["vec"]

    against = [f for f in factors if f["prefers"] == "winner"]
    pro = [f for f in factors if f["prefers"] == "target"]
    if _prefers(float(score[target]), float(score[pred])) == "neutral":
        category = "tie"
    elif against and not pro:
        category = "conditions_wrong"
    elif against:
        category = "outweighed"
    else:
        # Nothing favours the winner yet it scored higher; should not happen
        # for same-label candidates, whose category evidence is identical
        category = "unexplained"
    return {"category": category, **detail, "factors": factors,
            "against": [f["name"] for f in against]}
