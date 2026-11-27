"""Cue coverage for parser winner selection.

The cue inventory (what a sentence explicitly states) combines the previous
parser's declared v4 predicates, its lowering aliases and cleaning triggers;
it is the same scanner the v4 winner pools and the error analysis used.
Matching is leftmost, longest-first and non-overlapping. A tree is credited by
tree relation name (``TREE_NAME_OF_CUE``), at every level."""
from __future__ import annotations

import re

from grounding.parsing.cleaning import _ATTR_TRIGGER
from grounding.parsing.lowering import _PREDICATE_ALIAS
from grounding.parsing.tree import relation_names

__all__ = ["cue_set", "cue_count", "TREE_NAME_OF_CUE", "expressed", "coverage",
           "predicate_count"]

#: The predicates the v4 system prompt declares, in its own spelling.
V4_PREDICATES = (
    "beside", "near", "next to", "within", "in", "inside", "close", "far",
    "opposite", "across", "surrounded by", "around", "attached", "above",
    "on top", "on", "below", "under", "beneath", "left", "right", "front",
    "behind", "back", "facing", "between", "middle", "center",
)

#: Inflections and multiword spellings the alias table does not carry because
#: lowering never sees them -- they are surface English, not model output.
_EXTRA_SURFACE: dict[str, tuple[str, ...]] = {
    "close": ("closest", "closer", "close to", "closest to", "nearest"),
    "far": ("farthest", "furthest", "far from", "away from", "far away from"),
    "front": ("in front of",),
    "behind": ("in back of", "at the back of"),
    "on top": ("on top of", "atop"),
    "next to": ("adjacent to", "adjacent"),
    "attached": ("touching", "connected to", "connected"),
    "above": ("over",),
    "under": ("underneath",),
    "center": ("centre", "centered", "centred"),
    "beside": ("by the side of",),
    "surrounded by": ("surrounded",),
    "facing": ("faces", "faced"),
    "left": ("leftmost", "left of", "to the left of", "left hand side"),
    "right": ("rightmost", "right of", "to the right of", "right hand side"),
}

#: ``isolated`` is the one v4 attribute ``_ATTR_TRIGGER`` has no row for -- the
#: baseline arm cannot execute it, so cleaning never gated it. Taken from the
#: isolation cue table the lexical probe measures with.
_ISOLATION = (
    r"\bby it ?self\b", r"\bby them ?selves\b", r"\bon it'?s own\b",
    r"\bon their own\b", r"\b(?:standing |sitting |sits |stands |all )?alone\b",
    r"\blone(?:ly|some)?\b", r"\bsolitary\b", r"\bsolo\b", r"\bisolated\b",
    r"\bin isolation\b",
    r"\baway from (?:the |all (?:the )?)?(?:other|rest|remaining)",
    r"\b(?:separated|separate|apart|set apart|removed|detached)\s+from\b",
    r"\bset apart\b", r"\boff to (?:the )?(?:side|itself)\b",
    r"\bnothing (?:is )?(?:around|next to|beside|near)\b",
    r"\bnot (?:near|next to|beside|touching|close to) any\b",
)


def _uncapture(pattern: str) -> str:
    """Make every group non-capturing, so group index identifies the concept."""
    return re.sub(r"\((?!\?)", "(?:", pattern)


def _build() -> tuple[re.Pattern, list[str]]:
    alts: list[tuple[str, str]] = []

    for pred in V4_PREDICATES:
        forms = {pred, *_EXTRA_SURFACE.get(pred, ())}
        forms |= {a for a, p in _PREDICATE_ALIAS.items() if p == pred}
        for f in forms:
            alts.append((pred, r"\b" + re.escape(f).replace(r"\ ", r"\s+") + r"\b"))

    for name, pat in _ATTR_TRIGGER.items():
        alts.append((name, _uncapture(pat)))
    for pat in _ISOLATION:
        alts.append(("isolated", pat))

    # Longest literal first, so a specific phrase consumes the span before the
    # bare preposition inside it can.
    alts.sort(key=lambda ap: (-len(ap[1]), ap[0]))
    concepts = [c for c, _ in alts]
    return re.compile("|".join(f"({p})" for _, p in alts)), concepts


_SCANNER, _CONCEPTS = _build()


def cue_set(utterance: str) -> frozenset[str]:
    """v4 concepts the sentence explicitly names, by their v4 spelling."""
    found: set[str] = set()
    for m in _SCANNER.finditer(utterance.lower()):
        for i, g in enumerate(m.groups()):
            if g is not None:
                found.add(_CONCEPTS[i])
                break
    return frozenset(found)


def cue_count(utterance: str) -> int:
    return len(cue_set(utterance))


# --------------------------------------------------------------------------
# Trees
# --------------------------------------------------------------------------

#: Each cue concept -> the one tree relation name (grounding.parsing.tree.
#: RELATIONS) for the same encoder. The tree vocabulary has one name per
#: encoder, so several v4 spellings land on one name; the mapping is the tree
#: prompt's own "map other wording" rule. tests/test_tree_parser.py checks it
#: against the extended vocabulary's encoders and against the scanner.
TREE_NAME_OF_CUE: dict[str, str] = {
    **dict.fromkeys(("beside", "near", "next to", "within", "in", "inside",
                     "close", "surrounded by", "around", "attached", "facing"),
                    "near"),
    **dict.fromkeys(("far", "opposite", "across"), "far"),
    **dict.fromkeys(("above", "on top", "on"), "above"),
    **dict.fromkeys(("below", "under", "beneath"), "below"),
    "left": "left", "right": "right", "front": "front",
    "behind": "behind", "back": "behind",
    "between": "between", "middle": "middle", "center": "middle",
    "large": "larger", "small": "smaller", "high": "higher", "low": "lower",
    "on_the_floor": "on the floor", "against_wall": "against wall",
    "at_corner": "corner", "isolated": "isolated",
}


def expressed(tree: dict) -> frozenset[str]:
    """Relation names a tree encodes, at every level.

    Unlike a flat v4 program, a tree states an anchor's own relations on the
    anchor ("the desk with the monitor on it"), so the cue a sentence names may
    sit below the root.
    """
    return frozenset(relation_names(tree))


def coverage(tree: dict, utterance: str) -> int:
    """How many explicitly stated cues the tree says, counted by tree name."""
    stated = {TREE_NAME_OF_CUE[c] for c in cue_set(utterance)}
    return len(stated & expressed(tree))


def predicate_count(tree: dict) -> int:
    """Relations the tree executes, at every level."""
    return len(relation_names(tree))
