"""The parser: a grammar-constrained program tree, checked, never repaired.

``tree`` holds the prompt, the grammar (GBNF, for llama.cpp and xgrammar) and
``to_executable``, the post-hoc check. ``cleaning`` and ``lowering`` belong to
the previous (v4) parser; they stay only because ``cues`` builds its cue
scanner from their tables and the error analysis of the v4 results reads them.
The parser pipeline does not call them.
"""

from grounding.parsing.fields import (
    nodes,
    relation_objects,
    relations,
)
from grounding.parsing.parse import (
    REPRESENTATIONS,
    extract_json_objects,
    load_rel_num,
    normalise_representation,
    representation_tree,
)
from grounding.parsing.tree import (
    DEFAULT_PROMPT,
    TREE_PROMPTS,
    InvalidTree,
    dumps,
    grammar_for,
    grammar_record,
    is_tree_prompt,
    parse_generation,
    to_executable,
    user_turn,
)
from grounding.parsing.schema import (
    ExecutableProgram,
    flat_program_of,
    is_executable,
    is_flat,
)

__all__ = [
    "REPRESENTATIONS",
    "DEFAULT_PROMPT",
    "TREE_PROMPTS",
    "InvalidTree",
    "ExecutableProgram",
    "dumps",
    "extract_json_objects",
    "flat_program_of",
    "grammar_for",
    "grammar_record",
    "is_executable",
    "is_flat",
    "is_tree_prompt",
    "load_rel_num",
    "normalise_representation",
    "nodes",
    "parse_generation",
    "relation_objects",
    "relations",
    "representation_tree",
    "to_executable",
    "user_turn",
]
