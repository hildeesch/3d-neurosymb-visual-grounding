"""Tree parser track: the model writes the executable program tree directly.

The v4 track asks for a flat JSON, repairs it with ``clean_program`` (regex
rules on the sentence) and turns it into a tree with ``lower_program`` (alias
table, depth cap). This track replaces all three with one fixed grammar:

* the prompt asks for the executable schema itself (``category`` / ``color`` /
  ``relations`` -> ``relation_name`` / ``negative`` / ``rank`` / ``objects``),
  with anchors as full nodes, so anchors can carry their own relations;
* relation names and colours come from a closed, canonical vocabulary (one
  name per encoder meaning, no synonyms), and each name fixes how many objects
  it takes;
* :func:`gbnf` renders that grammar for constrained decoding (llama.cpp
  ``grammar`` field, xgrammar for Transformers), and :func:`to_executable`
  checks the same grammar after the fact. It rejects; it never repairs.

Nothing here reads the sentence. Whatever the model emits inside the grammar is
what gets executed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from grounding.parsing.parse import extract_json_objects

__all__ = [
    "RelationSpec", "RELATIONS", "COLOURS", "MAX_DEPTH", "TREE_PROMPTS",
    "SYSTEM_PROMPT_TREE_V1", "InvalidTree", "gbnf", "to_executable",
    "parse_generation", "dumps", "depth_of",
]


@dataclass(frozen=True)
class RelationSpec:
    """How a relation may appear in the tree.

    ``min_objects``/``max_objects`` bound the ``objects`` list; ``ranked`` says
    whether ``rank`` may be an integer (otherwise it must be null).
    """

    min_objects: int
    max_objects: int
    ranked: bool


# One canonical name per encoder meaning. Every name is a key of the extended
# vocabulary (grounding.representation.pipeline.CONCEPTS) with the matching
# anchor count; tests/test_tree_grammar.py checks this.
RELATIONS: dict[str, RelationSpec] = {
    # Properties of the node itself (unary encoders, no anchors).
    "larger": RelationSpec(0, 0, True),
    "smaller": RelationSpec(0, 0, True),
    "higher": RelationSpec(0, 0, True),
    "lower": RelationSpec(0, 0, True),
    "against wall": RelationSpec(0, 0, True),
    "on the floor": RelationSpec(0, 0, False),
    "corner": RelationSpec(0, 0, False),
    "isolated": RelationSpec(0, 0, False),
    # Binary encoders. One object: relative to that anchor. None: relative to
    # the other objects of the node's own category ("the left chair").
    "near": RelationSpec(0, 1, True),
    "far": RelationSpec(0, 1, True),
    "above": RelationSpec(0, 1, False),
    "below": RelationSpec(0, 1, False),
    "left": RelationSpec(0, 1, True),
    "right": RelationSpec(0, 1, True),
    "front": RelationSpec(0, 1, True),
    "behind": RelationSpec(0, 1, True),
    # Ternary encoder (Between). "between" names both anchors; "middle" uses
    # the node's own category for both ("the middle chair").
    "between": RelationSpec(2, 2, False),
    "middle": RelationSpec(0, 0, False),
}

# Colour slot vocabulary: the v4 list, all scorable by encoders/color.py.
COLOURS: tuple[str, ...] = (
    "black", "white", "gray", "brown", "red", "orange", "yellow", "green",
    "blue", "purple", "pink", "beige", "tan", "wooden", "metal", "silver",
    "dark", "light",
)

#: Node levels: target (1), its anchors (2), their anchors (3).
MAX_DEPTH = 3
#: Categories: 1-4 lowercase words. Bounded so a constrained sampler cannot run on.
_CATEGORY = re.compile(r"[a-z]+(?: [a-z]+){0,3}")
_MAX_RANK = 9

_NODE_KEYS = ("category", "color", "relations")
_RELATION_KEYS = ("relation_name", "negative", "rank", "objects")


def _names(pred) -> list[str]:
    return [n for n, s in RELATIONS.items() if pred(s)]


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

def _node(category, color=None, *relations):
    return {"category": category, "color": color, "relations": list(relations)}


def _rel(name, *objects, rank=None, negative=False):
    return {"relation_name": name, "negative": negative, "rank": rank,
            "objects": list(objects)}


#: (sentence, tree) few-shot pairs. The first three are the v4 examples in tree
#: form; the fourth shows an anchor with its own relation.
TREE_EXAMPLES: tuple[tuple[str, dict], ...] = (
    ("the second chair from the left next to the bed",
     _node("chair", None, _rel("left", rank=2), _rel("near", _node("bed")))),
    ("the lone black table against the wall",
     _node("table", "black", _rel("isolated"), _rel("against wall"))),
    ("the metal shelf at the end of the counter",
     _node("shelf", "metal", _rel("near", _node("counter")))),
    ("the chair farthest from the desk with the monitor on it",
     _node("chair", None,
           _rel("far", _node("desk", None, _rel("below", _node("monitor"))), rank=1))),
)


# Frozen literal, like the v2-v5 prompts: editing RELATIONS/COLOURS must not
# silently change cached supervision. tests/test_tree_grammar.py checks that
# the lists and examples below still agree with the grammar.
SYSTEM_PROMPT_TREE_V1 = """You convert one natural-language 3D referring expression into a JSON program tree.

Respond with ONLY a single JSON object on one line - no prose, no markdown fences.

Every object in the tree is a node:
{"category": "<object category, singular, lowercase>", "color": <color or null>, "relations": [<relation>, ...]}
Every relation is:
{"relation_name": "<name>", "negative": <true or false>, "rank": <1-9 or null>, "objects": [<node>, ...]}

The root node is the target object. An object a relation refers to (an anchor) is itself a node, so it carries its own color and relations. Attach every phrase to the object it describes: "the box on the table that is by the window" -> box above table, and that table near window. Use at most three levels: target, anchor, anchor of the anchor.

Relation names, by how many objects they take:
- no objects, a property of the node itself: larger, smaller, higher, lower, against wall, on the floor, corner, isolated
- zero or one object: near, far, above, below, left, right, front, behind. With one object the relation is to that anchor ("the chair left of the desk"). With no objects it compares the node with the other objects of its own category ("the left chair").
- exactly two objects: between
- no objects: middle ("the middle chair", "in the middle of the room")

Rules:
- Emit ONLY what the sentence states. Never add a relation or color that is not there. Prefer the minimal tree.
- Use only the relation names above. Map other wording to the closest one: next to / beside / by / close to / in / inside / within / attached / around / facing / with -> near; on / on top of / over -> above; under / beneath / underneath -> below; opposite / across from / away from -> far; in front of -> front; back / at the back of -> behind; big / long / wide -> larger; small / short / narrow -> smaller; tall / high / upper / top -> higher; low / bottom -> lower; lone / alone / by itself -> isolated.
- "wall", "floor", "ceiling" and "room" are not objects and never a node: "against / by / along the wall" -> against wall; "on the floor" -> on the floor; "in the corner" -> corner; "in the middle of the room" -> middle.
- Never use a position word ("bottom", "top", "end", "middle", "side", "row", "part") or a sub-part as a category: "the foot of the bed" -> near, object bed. "bottom"/"lowest" -> lower and "top"/"highest" -> higher go on the TARGET, and the relation is kept: "the box on the bottom shelf" -> box with lower, and box above shelf.
- color is one of: black, white, gray, brown, red, orange, yellow, green, blue, purple, pink, beige, tan, wooden, metal, silver, dark, light, or null. "dark", "light", "wooden" and "metal" are colors here. Put the color on the node it describes: "the chair by the black table" -> the table node has color black.
- rank is for ordinals and superlatives only, 1-based: "the largest table" -> larger rank 1; "second from the left" -> left rank 2; "closest to the door" -> near rank 1 with object door; "farthest from X" -> far rank 1 with object X. Otherwise rank is null. between and middle always have rank null.
- negative is true only when the sentence negates that relation: "the box that is not on the shelf" -> above, negative true, object shelf.
- Categories are singular real scene objects ("chairs" -> "chair").

Example 1:
Input: "the second chair from the left next to the bed"
Output:
{"category": "chair", "color": null, "relations": [{"relation_name": "left", "negative": false, "rank": 2, "objects": []}, {"relation_name": "near", "negative": false, "rank": null, "objects": [{"category": "bed", "color": null, "relations": []}]}]}

Example 2:
Input: "the lone black table against the wall"
Output:
{"category": "table", "color": "black", "relations": [{"relation_name": "isolated", "negative": false, "rank": null, "objects": []}, {"relation_name": "against wall", "negative": false, "rank": null, "objects": []}]}

Example 3:
Input: "the metal shelf at the end of the counter"
Output:
{"category": "shelf", "color": "metal", "relations": [{"relation_name": "near", "negative": false, "rank": null, "objects": [{"category": "counter", "color": null, "relations": []}]}]}

Example 4:
Input: "the chair farthest from the desk with the monitor on it"
Output:
{"category": "chair", "color": null, "relations": [{"relation_name": "far", "negative": false, "rank": 1, "objects": [{"category": "desk", "color": null, "relations": [{"relation_name": "below", "negative": false, "rank": null, "objects": [{"category": "monitor", "color": null, "relations": []}]}]}]}]}
"""

#: Prompt registry for this track. Kept separate from parsing.prompt.PROMPTS so
#: the frozen v2-v5 literals and their cached supervision are untouched.
TREE_PROMPTS = {"tree_v1": SYSTEM_PROMPT_TREE_V1}


# --------------------------------------------------------------------------
# Grammar (GBNF; also accepted by xgrammar's EBNF parser)
# --------------------------------------------------------------------------

def _lit(s: str) -> str:
    """A GBNF string literal for the JSON text ``s``."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _alt(values) -> str:
    return " | ".join(_lit(json.dumps(v)) for v in values)


def gbnf(max_depth: int = MAX_DEPTH) -> str:
    """The tree grammar as GBNF, root = target node, nodes up to ``max_depth``.

    Layout is fixed: keys in schema order, one line, an optional single space
    after ``:`` and ``,``. Nodes at ``max_depth`` can only take relations
    without objects.
    """
    if max_depth < 1:
        raise ValueError("max_depth must be >= 1")

    def rel_rule(d: int, name: str, spec: RelationSpec) -> str | None:
        last = d == max_depth
        if last and spec.min_objects > 0:
            return None
        hi = 0 if last else spec.max_objects
        child = f"node{d + 1}"
        if hi == 0:
            objects = '"[]"'
        else:
            items = " ".join([child] + [f'"," ws {child}'] * (spec.min_objects - 1))
            if spec.min_objects == 0:
                items = f"( {child} )?"
            objects = f'"[" {items} "]"'
        rank = "rank" if spec.ranked else '"null"'
        return (f'"{{" "\\"relation_name\\":" ws {_lit(json.dumps(name))} '
                f'"," ws "\\"negative\\":" ws bool "," ws "\\"rank\\":" ws {rank} '
                f'"," ws "\\"objects\\":" ws {objects} "}}"')

    lines = ["root ::= node1"]
    for d in range(1, max_depth + 1):
        lines.append(
            f'node{d} ::= "{{" "\\"category\\":" ws category "," ws "\\"color\\":" ws color '
            f'"," ws "\\"relations\\":" ws "[" ( rel{d} ( "," ws rel{d} )* )? "]" "}}"')
        alts = []
        for name, spec in RELATIONS.items():
            body = rel_rule(d, name, spec)
            if body is None:
                continue
            rule = f"rel{d}-{re.sub(r'[^a-z]+', '-', name)}"
            lines.append(f"{rule} ::= {body}")
            alts.append(rule)
        lines.append(f"rel{d} ::= " + " | ".join(alts))
    lines += [
        'category ::= "\\"" word ( " " word )? ( " " word )? ( " " word )? "\\""',
        "word ::= [a-z]+",
        f"color ::= \"null\" | {_alt(COLOURS)}",
        'bool ::= "true" | "false"',
        f'rank ::= "null" | [1-{_MAX_RANK}]',
        'ws ::= " "?',
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Post-hoc check: the same grammar, on parsed JSON
# --------------------------------------------------------------------------

class InvalidTree(ValueError):
    """A program outside the grammar. ``reason`` is a short stable code."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def _check_node(node: Any, depth: int, max_depth: int) -> dict:
    if not isinstance(node, dict) or set(node) != set(_NODE_KEYS):
        raise InvalidTree("node_keys", repr(sorted(node) if isinstance(node, dict) else node))
    category = node["category"]
    if not isinstance(category, str) or not _CATEGORY.fullmatch(category):
        raise InvalidTree("category", repr(category))
    color = node["color"]
    if color is not None and color not in COLOURS:
        raise InvalidTree("color", repr(color))
    rels = node["relations"]
    if not isinstance(rels, list):
        raise InvalidTree("relations_type")
    out: dict[str, Any] = {"category": category}
    if color is not None:
        out["color"] = color
    out["relations"] = [_check_relation(r, depth, max_depth) for r in rels]
    return out


def _check_relation(rel: Any, depth: int, max_depth: int) -> dict:
    if not isinstance(rel, dict) or set(rel) != set(_RELATION_KEYS):
        raise InvalidTree("relation_keys", repr(sorted(rel) if isinstance(rel, dict) else rel))
    name = rel["relation_name"]
    spec = RELATIONS.get(name)
    if spec is None:
        raise InvalidTree("relation_name", repr(name))
    if not isinstance(rel["negative"], bool):
        raise InvalidTree("negative", repr(rel["negative"]))
    rank = rel["rank"]
    if rank is not None:
        if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= _MAX_RANK:
            raise InvalidTree("rank", repr(rank))
        if not spec.ranked:
            raise InvalidTree("rank_not_allowed", name)
    objects = rel["objects"]
    if not isinstance(objects, list):
        raise InvalidTree("objects_type", name)
    hi = 0 if depth == max_depth else spec.max_objects
    if not spec.min_objects <= len(objects) <= hi:
        reason = "too_deep" if depth == max_depth and objects else "arity"
        raise InvalidTree(reason, f"{name} with {len(objects)} objects at depth {depth}")
    out: dict[str, Any] = {"relation_name": name, "negative": rel["negative"]}
    if rank is not None:
        out["rank"] = rank
    out["objects"] = [_check_node(o, depth + 1, max_depth) for o in objects]
    return out


def to_executable(tree: Any, max_depth: int = MAX_DEPTH) -> dict:
    """Check ``tree`` against the grammar and return the program the resolver runs.

    The only change is dropping null ``color``/``rank`` (the resolver reads
    absence). Anything outside the grammar raises :class:`InvalidTree`.
    """
    return _check_node(tree, 1, max_depth)


def parse_generation(text: str, max_depth: int = MAX_DEPTH) -> tuple[dict | None, dict | None, str]:
    """``(tree, executable, outcome)`` for one generation.

    ``tree`` is the model's JSON as emitted (the SFT target), ``executable`` the
    checked program, ``outcome`` ``"ok"`` or an :class:`InvalidTree` reason
    (``"no_json"`` when no object could be read). Grammar-constrained output is
    the JSON text itself; unconstrained output may have prose around it, which
    is skipped, but the object is never edited.
    """
    text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = extract_json_objects(text)
    if not isinstance(obj, dict):
        return None, None, "no_json"
    try:
        return obj, to_executable(obj, max_depth), "ok"
    except InvalidTree as e:
        return obj, None, e.reason


def dumps(tree: dict) -> str:
    """The one-line layout the prompt examples and SFT targets use."""
    return json.dumps(tree, ensure_ascii=False)


def depth_of(node: dict) -> int:
    """Node levels in an executable or model tree (target alone = 1)."""
    child = [depth_of(o) for r in node.get("relations") or [] for o in r.get("objects") or []]
    return 1 + max(child, default=0)
