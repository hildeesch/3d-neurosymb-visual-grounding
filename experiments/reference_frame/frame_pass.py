"""The LLM pass that extracts an observer frame from an utterance."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from experiments.reference_frame.rows import REPO, load_jsonl
from grounding.parsing.generation import template_prompt
from grounding.paths import resolve_under_root
from grounding.resolution.artifacts import git_commit as current_git_commit, sha256


ALLOWED_POSES = {"standing", "sitting", "lying", None}


SYSTEM_PROMPT = r"""
Extract ONLY observer/reference-frame information explicitly stated in a
3D referring expression. Do not solve the grounding task.

Return exactly one JSON object:
{
  "has_viewpoint": true|false,
  "frame_type": "none"|"toward_only"|"from_only"|"from_toward"|
                "away_only"|"away_toward"|"at_only"|"at_toward"|
                "egocentric_only",
  "from_anchor": string|null,
  "toward_anchor": string|null,
  "away_from_anchor": string|null,
  "at_anchor": string|null,
  "body_pose": "standing"|"sitting"|"lying"|null,
  "observer_deictic": true|false,
  "evidence": string|null
}

Only extract information describing the OBSERVER:
- "facing/looking at X" -> toward_anchor = X
- "from X" / "entering through X" -> from_anchor = X
- "back to X" -> away_from_anchor = X
- "standing/sitting/lying at X" -> at_anchor = X
- explicit "your left/right", "behind you", etc. -> observer_deictic = true

Do NOT treat an object's own position or orientation as observer viewpoint.
"The chair facing the hallway" and "the pillow on the right" contain no
observer frame.

Do not infer missing orientation. "Standing in front of X" gives an observer
position, not toward_anchor = X. Preserve phrases such as "front of the
windows" or "foot of the bed" as part of the anchor.

Use the shortest exact phrase supporting the extraction as evidence.
If no explicit observer cue exists, output has_viewpoint=false, frame_type=none
and null anchors.

Examples:

Sentence: "Facing the carts, the carts on the left."
Output:
{"has_viewpoint":true,"frame_type":"toward_only","from_anchor":null,
"toward_anchor":"carts","away_from_anchor":null,"at_anchor":null,
"body_pose":null,"observer_deictic":false,"evidence":"Facing the carts"}

Sentence: "Standing in front of the toilet, choose the long horizontal bar to the right."
Output:
{"has_viewpoint":true,"frame_type":"at_only","from_anchor":null,
"toward_anchor":null,"away_from_anchor":null,
"at_anchor":"front of the toilet","body_pose":"standing",
"observer_deictic":false,"evidence":"Standing in front of the toilet"}

Sentence: "The cabinet hidden behind the black desk chair that's facing the hallway."
Output:
{"has_viewpoint":false,"frame_type":"none","from_anchor":null,
"toward_anchor":null,"away_from_anchor":null,"at_anchor":null,
"body_pose":null,"observer_deictic":false,"evidence":null}
""".strip()


EMPTY_FRAME = {
    "has_viewpoint": False,
    "frame_type": "none",
    "from_anchor": None,
    "toward_anchor": None,
    "away_from_anchor": None,
    "at_anchor": None,
    "body_pose": None,
    "observer_deictic": False,
    "evidence": None,
}


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def strip_thinking(text: str) -> str:
    """
    If a model still emits <think>...</think>, discard that section before
    looking for JSON. If thinking was truncated and there is no closing tag,
    leave the text unchanged so the caller gets a useful parsing error.
    """
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def extract_json_object(text: str) -> dict:
    """
    Tolerate code fences or a small amount of extra text, but require one valid
    JSON object. We intentionally do not try to interpret natural-language
    reasoning as structured output.
    """
    original = text
    text = strip_thinking(strip_code_fences(text))

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object in model output: {original!r}")

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                obj = json.loads(candidate)
                if not isinstance(obj, dict):
                    raise ValueError("Model output JSON was not an object.")
                return obj

    raise ValueError(f"Unclosed JSON object in model output: {original!r}")


def infer_frame_type(x: dict) -> str:
    has_from = bool(x.get("from_anchor"))
    has_toward = bool(x.get("toward_anchor"))
    has_away = bool(x.get("away_from_anchor"))
    has_at = bool(x.get("at_anchor"))
    has_deictic = bool(x.get("observer_deictic"))

    if has_away and has_toward:
        return "away_toward"
    if has_from and has_toward:
        return "from_toward"
    if has_at and has_toward:
        return "at_toward"
    if has_away:
        return "away_only"
    if has_from:
        return "from_only"
    if has_toward:
        return "toward_only"
    if has_at:
        return "at_only"
    if has_deictic:
        return "egocentric_only"
    return "none"


def canonicalize(obj: dict, sentence: str = "") -> dict:
    out = {
        "has_viewpoint": bool(obj.get("has_viewpoint", False)),
        "frame_type": obj.get("frame_type", "none"),
        "from_anchor": obj.get("from_anchor"),
        "toward_anchor": obj.get("toward_anchor"),
        "away_from_anchor": obj.get("away_from_anchor"),
        "at_anchor": obj.get("at_anchor"),
        "body_pose": obj.get("body_pose"),
        "observer_deictic": bool(obj.get("observer_deictic", False)),
        "evidence": obj.get("evidence"),
    }

    for key in (
        "from_anchor",
        "toward_anchor",
        "away_from_anchor",
        "at_anchor",
        "evidence",
    ):
        value = out[key]
        if isinstance(value, str):
            value = value.strip()
            out[key] = value or None
        elif value is not None:
            out[key] = str(value).strip() or None

    pose = out["body_pose"]
    if isinstance(pose, str):
        pose = pose.strip().lower()
    if pose not in ALLOWED_POSES:
        pose = None
    out["body_pose"] = pose

    s = sentence or ""

    # Hard guard from the 100-row manual validation: plain left/right is not egocentric
    explicit_deictic = bool(
        re.search(
            r"\b(?:your|to\s+your|on\s+your)\s+"
            r"(?:far\s+)?(?:left|right)\b",
            s,
            re.I,
        )
    )

    if out["observer_deictic"] and not explicit_deictic:
        out["observer_deictic"] = False

    has_anchor = any(
        out.get(k)
        for k in (
            "from_anchor",
            "toward_anchor",
            "away_from_anchor",
            "at_anchor",
        )
    )

    if not explicit_deictic and not has_anchor:
        out["observer_deictic"] = False

    # Conservative pronoun resolution for viewpoint anchors
    if isinstance(out.get("toward_anchor"), str):
        pron = out["toward_anchor"].strip().lower()
        if pron in {"them", "it", "those"}:
            prefix = s
            m_face = re.search(
                r"\b(?:when|if|while)?\s*(?:you\s+(?:are\s+)?)?"
                r"(?:face|facing|look(?:ing)?\s+at|stare|staring\s+at)\b",
                s,
                re.I,
            )
            if m_face:
                prefix = s[:m_face.start()]

            # Prefer an explicit plural noun phrase before the viewpoint cue
            plural_phrases = re.findall(
                r"\b(?:the|these|those|of\s+the)\s+"
                r"(?:\d+\s+|two\s+|three\s+|four\s+|five\s+)?"
                r"([A-Za-z][A-Za-z _-]*?s)\b",
                prefix,
                re.I,
            )
            if plural_phrases:
                cand = plural_phrases[-1].strip(" ,.")
                if cand.lower() not in {"this", "its", "is", "was"}:
                    out["toward_anchor"] = cand

    # Derive redundant fields from populated slots for consistency
    out["frame_type"] = infer_frame_type(out)
    out["has_viewpoint"] = out["frame_type"] != "none"

    return out


def diff_summary(row: dict, new: dict, had_error: bool) -> dict:
    """What the extractor recovered on one row, as flags worth eyeballing.

    There is nothing to diff against any more: the v4 parser emits no viewpoint
    slot, so this describes the extractor's own output and does not compare it
    to a shipped field.
    """
    recovered_slots = [
        name
        for name in (
            "from_anchor",
            "toward_anchor",
            "away_from_anchor",
            "at_anchor",
        )
        if new.get(name)
    ]

    flags = []

    if had_error:
        flags.append("new_extractor_error")

    if len(recovered_slots) >= 2:
        flags.append("richer_multi_anchor_frame")

    if new["frame_type"] == "egocentric_only":
        flags.append("observer_pose_required")

    return {
        "frame_type": new["frame_type"],
        "slots": recovered_slots,
        "flags": flags,
    }


def build_prompt(tokenizer, sentence: str, enable_thinking: bool) -> str:
    # /no_think is harmless elsewhere and guards Qwen3 versions without enable_thinking
    user_content = (
        f"Sentence: {json.dumps(sentence, ensure_ascii=False)}"
        + ("" if enable_thinking else "\n/no_think")
    )
    if not hasattr(tokenizer, "apply_chat_template"):
        return SYSTEM_PROMPT + "\n\n" + user_content + "\nOutput:"
    return template_prompt(tokenizer, SYSTEM_PROMPT, user_content, tokenize=False,
                           enable_thinking=enable_thinking, fallback_suffix="")


def make_generator(
    model_id: str,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
    *,
    seed: int | None = 0,
    quantize: str = "none",
    device: str | None = None,
    dtype: str | None = None,
):
    try:
        from grounding.parsing.generation import batch_generate, load_model
        import torch  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "torch/transformers are not installed in this environment. Use the "
            "same environment as the local Qwen parser (pip install -e "
            "'.[training]')."
        ) from e

    model, tokenizer, sampler = load_model(
        model_id, temperature=temperature, seed=seed, quantize=quantize,
        device=device, dtype=dtype)

    def run(sentence: str) -> str:
        prompt = build_prompt(tokenizer, sentence, enable_thinking)
        return batch_generate(model, tokenizer, [prompt], max_tokens=max_tokens,
                              sampler=sampler).texts[0]

    return run

def parse_invocation(args, rows, git_commit=None) -> dict:
    """Frame-generation inputs and settings for ``<output>.meta.json``."""
    src = resolve_under_root(args.input)
    if git_commit is None:
        git_commit = current_git_commit()
    return {
        "producer": "python -m experiments.reference_frame parse",
        "git_commit": git_commit,
        "command": " ".join(sys.argv),
        "rows": {"path": str(src.relative_to(REPO)) if src.is_relative_to(REPO) else str(src),
                 "sha256": sha256(src),
                 "n": len(rows), "limit": args.limit},
        "model": args.model,
        "prompt": "reference_frame.build_prompt (frame schema), frozen in this file",
        # Recorded so a consumer can prove a frame pass came from THIS prompt
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "backend": "hf",
        "decoding": {"sampler": "greedy" if args.temperature == 0 else "sampled",
                     "temperature": args.temperature, "max_tokens": args.max_tokens,
                     "thinking": bool(args.thinking),
                     "seed": getattr(args, "seed", None),
                     "quantize": getattr(args, "quantize", "none")},
    }


def cmd_parse(args) -> int:
    rows = load_jsonl(args.input)
    args.output = resolve_under_root(args.output)
    meta_path = Path(str(args.output) + ".meta.json")

    if args.limit is not None:
        rows = rows[: args.limit]

    if not rows:
        raise SystemExit("No rows selected.")

    run_model = make_generator(
        model_id=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        enable_thinking=args.thinking,
        seed=args.seed,
        quantize=args.quantize,
        device=args.device,
        dtype=args.dtype,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    type_counts = Counter()
    flag_counts = Counter()
    failures = 0
    printed = 0

    with args.output.open("w") as out_f:
        for n, row in enumerate(rows, 1):
            sentence = row.get("sentence", "")
            raw = None
            error = None

            try:
                raw = run_model(sentence)
                frame = canonicalize(extract_json_object(raw), sentence=sentence)
            except Exception as e:
                failures += 1
                error = f"{type(e).__name__}: {e}"
                frame = dict(EMPTY_FRAME)

                print(f"\nERROR [{n}] {sentence}")
                print(error)
                if raw:
                    print("RAW:", raw)

            diff = diff_summary(row, frame, had_error=error is not None)

            type_counts[frame["frame_type"]] += 1
            flag_counts.update(diff["flags"])

            enriched = dict(row)
            enriched["viewpoint_frame"] = frame
            enriched["viewpoint_diff"] = diff
            enriched["viewpoint_raw_output"] = raw

            if error is not None:
                enriched["viewpoint_parse_error"] = error

            out_f.write(json.dumps(enriched, ensure_ascii=False) + "\n")

            should_print = (
                printed < args.print_changes
                and (
                    diff["flags"]
                    or frame["frame_type"] != "none"
                )
            )
            if should_print:
                printed += 1
                print(f"\n[{n}] {sentence}")
                print("FRAME:", json.dumps(frame, ensure_ascii=False))
                if diff["flags"]:
                    print("FLAGS:", ", ".join(diff["flags"]))
                else:
                    print("FLAGS: none")

    meta = parse_invocation(args, rows)
    import hashlib as _h
    meta["output_sha256"] = _h.sha256(args.output.read_bytes()).hexdigest()
    meta["written"] = {"rows": len(rows), "failures": failures}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print("\n=== Summary ===")
    print(f"Rows:     {len(rows)}")
    print(f"Failures: {failures} ({failures / len(rows):.1%})")

    print("\nFrame types:")
    for key, value in type_counts.most_common():
        print(f"  {key:18s} {value}")

    if flag_counts:
        print("\nExtraction flags:")
        for key, value in flag_counts.most_common():
            print(f"  {key:48s} {value}")

    print(f"\nWrote: {args.output}")
    return 0
