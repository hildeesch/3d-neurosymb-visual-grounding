"""Parse CSV or JSONL rows into executable program trees.

The parser writes the tree directly under the tree system prompt, decoded under
the tree grammar (llama.cpp ``grammar`` field, xgrammar on hf). Each row stores
``json_obj = to_executable(tree)`` -- null when the tree is outside the grammar,
never repaired -- plus the model's tree (``program``) and the check's
``outcome``. ``--no-grammar`` decodes the same prompt unconstrained (the grammar
ablation; ``_nogrammar`` in default names); the OpenAI backend needs it.
Rebuild features from the output before evaluation; existing rows are skipped
unless ``--fresh`` is set.

``--backend hf`` (default) runs a local Hugging Face model through PyTorch, with
an optional PEFT LoRA ``--adapter``. ``--backend openai`` sends the same system
prompt and user turn to an OpenAI chat model (greedy, T=0). ``--backend
llamacpp`` sends the identically rendered prompt to a running llama.cpp
``llama-server`` (a GGUF model such as Qwen3-30B-A3B), greedy.

Two styles render the same system prompt and user turn: ``sft`` as plain
ChatML (what the LoRA students were trained on) and ``rollout`` through the
model's chat template with thinking off (what the teachers saw when they
generated the rollouts).

``--teacher qwen|qwen_coder|gpt4o`` is the semantic-parsing ablation: the
untrained teacher parses the rows in place of the LoRA student, with the
``rollout`` style and greedy decoding, so the ablation changes the model and
nothing else. ``qwen`` (Qwen3-30B-A3B) and ``qwen_coder``
(Qwen3-Coder-30B-A3B-Instruct) default to the llama.cpp backend (the 16 GB
route; pass ``--backend hf`` for Transformers), ``gpt4o`` to the OpenAI API. On
llama.cpp the prompt is rendered with the served teacher's own chat template; a
GGUF that is not the requested teacher is refused.

``--student qwen3|coder`` selects the base model of a LoRA student
(Qwen3-1.7B or Qwen2.5-Coder-1.5B-Instruct) for ``--style sft``; pass the
adapter trained on that base with ``--adapter``. ``--model`` still overrides.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path


from grounding.paths import REPO_ROOT
from grounding.parsing import (REPRESENTATIONS, load_rel_num,
                               normalise_representation)
from grounding.parsing.tree import (DEFAULT_PROMPT, TREE_PROMPTS,
                                    add_grammar_args, grammar_for,
                                    grammar_record, parse_generation,
                                    relation_names, user_turn)
from grounding.parsing.generation import (add_model_args, batch_generate,
                                          encode_chat, free_memory, load_model,
                                          template_prompt)
from grounding.parsing.generation import chat_text
from grounding.parsing.llamacpp_backend import (add_llamacpp_args,
                                                resolve_template_tokenizer)
from grounding.parsing.models import DEFAULT_STUDENT, STUDENTS
from grounding.parsing.models import TEACHERS as TEACHER_MODELS
from grounding.parsing.openai_backend import DEFAULT_OPENAI_MODEL
from grounding.resolution.artifacts import git_commit, sha256

#: style -> rendered through the model's chat template?
STYLES = {
    # The student's SFT conditioning: plain ChatML, as the training rows
    "sft": False,
    # The teacher's rollout conditioning (generate_rollouts.user_message is the
    # same user turn), rendered with the chat template as the rollouts were
    "rollout": True,
}

#: --teacher presets for the semantic-parsing ablation. backend: default
#: backend; backends: the ones that serve this teacher; model: default model per
#: backend (None = the served GGUF's own name); max_tokens: the rollouts' budget.
TEACHERS = {
    **{key: {"backend": "llamacpp", "backends": ("llamacpp", "hf"),
             "model": {"hf": t.hf, "llamacpp": None},
             "max_tokens": 320}
       for key, t in TEACHER_MODELS.items()},
    "gpt4o": {"backend": "openai", "backends": ("openai",),
              "model": {"openai": DEFAULT_OPENAI_MODEL},
              "max_tokens": 320},
}


def track_tag(prompt: str = DEFAULT_PROMPT, constrained: bool = True) -> str:
    """Prompt part of a default output name: e.g. tree_v1 or tree_v1_nogrammar."""
    return prompt + ("" if constrained else "_nogrammar")


def teacher_out(teacher: str, rows_path: Path, prompt: str = DEFAULT_PROMPT,
                constrained: bool = True) -> str:
    """Default output for a teacher parse of ``rows_path``."""
    stem = rows_path.stem          # nr3d_test, nr3d_dev, ...
    split = "dev" if "dev" in stem else "test"
    tag = track_tag(prompt, constrained)
    return f"data/parses/{split}/{stem}_teacher_{teacher}_{tag}_ext.jsonl"


def apply_teacher(args, ap) -> None:
    """Fill the --teacher (or --student) preset into flags the user left unset."""
    student = getattr(args, "student", None)
    if args.teacher is None:
        for name, default in (("style", "sft"), ("backend", "hf"),
                              ("max_tokens", 256)):
            if getattr(args, name) is None:
                setattr(args, name, default)
        if student is not None:
            if args.backend != "hf":
                ap.error("--student is a local Transformers model; use --backend hf")
            if args.model is None:
                args.model = STUDENTS[student].hf
        return
    if student is not None:
        ap.error("--student and --teacher are two different parsers; pass one")
    preset = TEACHERS[args.teacher]
    if args.adapter:
        ap.error("--teacher is the untrained teacher; it takes no --adapter")
    if args.style not in (None, "rollout"):
        ap.error(f"--teacher uses --style rollout, not {args.style!r}: any "
                 "other style changes the prompt as well as the model")
    args.style = "rollout"
    args.backend = args.backend or preset["backend"]
    if args.backend not in preset["backends"]:
        ap.error(f"--teacher {args.teacher} runs on {'/'.join(preset['backends'])}, "
                 f"not --backend {args.backend}")
    if args.model is None:
        args.model = preset["model"].get(args.backend)
    if args.max_tokens is None:
        args.max_tokens = preset["max_tokens"]


def load_rows(path: Path, limit: int = 0) -> tuple[list[dict], bool]:
    """Rows to parse, and whether they carry a dataset_idx worth writing out."""
    rows = []
    if path.suffix == ".csv":
        with open(path, newline="") as f:
            for idx, r in enumerate(csv.DictReader(f)):
                rows.append({
                    "dataset_idx": idx,
                    "row_idx": idx,
                    "scan_id": r["scan_id"],
                    "target_id": int(r["target_id"]),
                    "target_label": r["instance_type"],
                    "sentence": r["utterance"],
                })
        keyed = True
    else:
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                rows.append({
                    "row_idx": r["row_idx"],
                    "scan_id": r["scan_id"],
                    "target_id": int(r["target_id"]),
                    "target_label": r.get("target_label"),
                    # probe files say "sentence", corpora say "utterance"; both are inputs
                    "sentence": r.get("sentence") or r["utterance"],
                })
        # A probe row_idx indexes the TRAIN csv, so dataset_idx would look up an unrelated split
        keyed = False
    return (rows[:limit] if limit else rows), keyed


def invocation(args, rows_path: Path, version: str, system: str,
               representation: str, n_rows: int, grammar: str | None = None) -> dict:
    """Everything that decides what a parse row looks like, recorded once.

    Written next to --out as `<out>.meta.json` so a frozen parse says how it was
    made. A resume is refused unless this record matches the one on disk: two
    invocations that differ in any decoding setting cannot share one file.
    """
    return {
        "producer": "experiments/parser_adaptation/parse_split.py",
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "rows": {"path": str(rows_path.relative_to(REPO_ROOT))
                 if rows_path.is_relative_to(REPO_ROOT) else str(rows_path),
                 "sha256": sha256(rows_path), "n": n_rows, "limit": args.limit},
        "backend": args.backend,
        "model": args.model, "adapter": args.adapter,
        "teacher": args.teacher,
        "student": getattr(args, "student", None),
        "representation": representation,
        "style": args.style, "prompt": version,
        "prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
        # None for v2-v5 prompts and --no-grammar; compared on resume
        "grammar": grammar_record(grammar),
        "decoding": {"sampler": "greedy", "temp": 0.0,
                     "max_tokens": args.max_tokens, "batch_size": args.batch_size,
                     "no_think": bool(args.no_think),
                     **({"quantize": args.quantize, "dtype": args.dtype}
                        if args.backend == "hf" else
                        {"template_tokenizer": args.tokenizer}
                        if args.backend == "llamacpp" else {"seed": args.seed})},
    }


def done_rows(path: Path) -> set:
    """row_idx values already written, so a resumed run skips them."""
    done = set()
    if not path.exists():
        return done
    for line in open(path):
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["row_idx"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--representation", dest="representation",
                    required=True, choices=list(REPRESENTATIONS),
                    help="whose rel_num vocabulary the parses lower against: "
                         "baseline | extended. Required, "
                         "not defaulted -- it decides what survives lowering")
    ap.add_argument("--rows", default="data/corpora/nr3d_test.csv",
                    help="utterances to parse: the test csv, or a probe jsonl")
    # Defaults describe the zero-shot 1.7B student; the full system supplies --adapter explicitly
    ap.add_argument("--out", default=None,
                    help="default: data/parses/test/nr3d_test_qwen3_1p7b_tree_v1"
                         "[_nogrammar]_ext.jsonl, or data/parses/<split>/<rows>_"
                         "teacher_<name>_tree_v1[_nogrammar]_ext.jsonl with --teacher")
    ap.add_argument("--teacher", choices=sorted(TEACHERS), default=None,
                    help="semantic-parsing ablation: parse with the untrained "
                         "teacher (qwen = Qwen3-30B-A3B, qwen_coder = "
                         "Qwen3-Coder-30B-A3B-Instruct, gpt4o = GPT-4o) under "
                         "the rollout prompt; see the module docstring")
    ap.add_argument("--student", choices=sorted(STUDENTS), default=None,
                    help="base model of a LoRA student (qwen3 = Qwen3-1.7B, "
                         "coder = Qwen2.5-Coder-1.5B-Instruct); sets --model. "
                         f"Default without --model: {STUDENTS[DEFAULT_STUDENT].hf}")
    ap.add_argument("--backend", choices=["hf", "llamacpp", "openai"], default=None,
                    help="default: hf, or the --teacher preset's backend. "
                         "hf: local PyTorch model; llamacpp: running llama-server "
                         "(GGUF, no adapters); openai: OpenAI chat API "
                         "(rollout style, --no-grammar)")
    ap.add_argument("--model", default=None,
                    help="default: Qwen/Qwen3-1.7B (hf), the served GGUF file "
                         f"name (llamacpp) or {DEFAULT_OPENAI_MODEL} (openai)")
    ap.add_argument("--adapter", default=None,
                    help="optional PEFT LoRA adapter directory (hf only); "
                         "convert old MLX adapters with "
                         "scripts/prepare/convert_mlx_adapter.py")
    ap.add_argument("--style", default=None, choices=sorted(STYLES),
                    help="sft: plain ChatML (students; default); rollout: the "
                         "chat template, thinking off (teachers; set by --teacher)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, choices=sorted(TREE_PROMPTS),
                    help="system prompt id (default: %(default)s)")
    # A MoE model reads more expert weights per step, so batch trades sparsity for memory
    ap.add_argument("--batch-size", type=int, default=8,
                    help="prompts per batch_generate call")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="default: 256, or the rollouts' 320 with --teacher")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--fresh", action="store_true",
                    help="overwrite --out instead of resuming into it")
    ap.add_argument("--no-think", action="store_true",
                    help="append ' /no_think' to each prompt. Required for a "
                         "RAW Qwen3 base; omit for SFT-derived models.")
    ap.add_argument("--seed", type=int, default=0,
                    help="openai: request seed (best-effort determinism)")
    ap.add_argument("--workers", type=int, default=8,
                    help="openai / llamacpp: concurrent requests (llamacpp: "
                         "match llama-server -np)")
    add_model_args(ap)
    add_llamacpp_args(ap)
    add_grammar_args(ap)
    args = ap.parse_args()
    apply_teacher(args, ap)
    server = None
    if args.backend == "llamacpp":
        if args.adapter:
            ap.error("--adapter applies to --backend hf only")
        from grounding.parsing.llamacpp_backend import LlamaCppServer

        server = LlamaCppServer(args.server_url, workers=args.workers)
        gguf = server.model_name(server.wait_ready())
        # A local teacher preset must be the model llama-server is serving,
        # rendered with that model's own chat template
        local_teacher = args.teacher if args.teacher in TEACHER_MODELS else None
        resolve_template_tokenizer(args, gguf, teacher=local_teacher)
        args.model = args.model or gguf
    args.model = args.model or (DEFAULT_OPENAI_MODEL if args.backend == "openai"
                                else STUDENTS[DEFAULT_STUDENT].hf)
    if args.backend == "openai":
        if args.style == "sft":
            ap.error("--backend openai needs --style rollout; the sft style is "
                     "the fine-tuned local student's format")
        if args.adapter or args.no_think:
            ap.error("--adapter and --no-think apply to --backend hf only")

    chat = STYLES[args.style]
    version = args.prompt
    system = TREE_PROMPTS[version]
    grammar = grammar_for(version, enabled=not args.no_grammar)
    if grammar is not None and args.backend == "openai":
        ap.error("the OpenAI API cannot decode under the tree grammar; pass "
                 "--no-grammar")

    rows_path = Path(args.rows)
    rows_path = rows_path if rows_path.is_absolute() else REPO_ROOT / rows_path
    if args.out is None:
        tag = track_tag(version, grammar is not None)
        args.out = (teacher_out(args.teacher, rows_path, version, grammar is not None)
                    if args.teacher else
                    f"data/parses/test/nr3d_test_qwen3_1p7b_{tag}_ext.jsonl")
    if rows_path.suffix == ".csv":
        assert "train" not in rows_path.name.lower(), (
            f"rows look like a train split: {args.rows!r}")
    representation = normalise_representation(args.representation)
    rel_num = load_rel_num(representation)

    rows, keyed = load_rows(rows_path, args.limit)
    out_path = Path(args.out)
    out_path = out_path if out_path.is_absolute() else REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = Path(str(out_path) + ".meta.json")
    if args.fresh:
        for p in (out_path, meta_path):
            if p.exists():
                p.unlink()

    meta = invocation(args, rows_path, version, system, representation, len(rows),
                      grammar)
    # A file with rows but no record of how they were made, or a record that
    # disagrees with this invocation, cannot be resumed into: the rows already
    # there and the rows about to be appended would be two populations under
    # one name. `--fresh` starts over; a different --out keeps both
    if out_path.exists() and done_rows(out_path):
        if not meta_path.exists():
            raise SystemExit(
                f"{out_path} has rows but no {meta_path.name}: its invocation is "
                "unknown, so this run cannot be shown to match it. Pass --fresh "
                "to start over, or a different --out.")
        recorded = json.loads(meta_path.read_text())
        drift = {k: (recorded.get(k), meta[k]) for k in
                 ("rows", "backend", "model", "adapter", "representation", "style",
                  "prompt", "prompt_sha256", "grammar", "decoding")
                 if recorded.get(k) != meta[k]}
        if drift:
            raise SystemExit(
                f"{out_path} was written by a different invocation; refusing "
                f"to mix rows. Differs in: {json.dumps(drift, indent=1)}\n"
                "Pass --fresh to start over, or a different --out.")
    elif not meta_path.exists() or args.fresh:
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    already = done_rows(out_path)
    todo = [r for r in rows if r["row_idx"] not in already]

    print(f"{len(rows)} utterances, {len(already)} already done, "
          f"{len(todo)} to do -> {out_path.name}")
    print(f"prompt {version} ({len(system)} chars), style {args.style} | "
          f"{len(rel_num)} relations in {representation}'s vocabulary")
    print(("decoding under the tree grammar" if grammar
           else "unconstrained decoding (--no-grammar)")
          + "; json_obj = to_executable(tree)")
    if args.adapter:
        print(f"adapter {args.adapter}")
    if not todo:
        print("Nothing to do.")
        return 0

    suffix = " /no_think" if args.no_think else ""

    def user_content(sentence: str) -> str:
        return user_turn(sentence) + suffix

    if args.backend == "openai":
        from grounding.parsing.openai_backend import OpenAIChat

        chat_api = OpenAIChat(args.model, workers=args.workers)
        print(f"Parsing with {args.model} via the OpenAI API (greedy)")

        def generate(chunk):
            results = chat_api.sample_many(
                system, [user_content(r["sentence"]) for r in chunk],
                seeds=[args.seed] * len(chunk), n=1, temperature=0.0,
                max_tokens=args.max_tokens)
            return [res.texts[0] for res in results]
    elif args.backend == "llamacpp":
        from grounding.parsing.llamacpp_backend import load_template_tokenizer

        tok = load_template_tokenizer(args.tokenizer)
        print(f"Parsing with {args.model} via llama-server at {args.server_url} "
              "(greedy)")

        def build_text(sentence: str) -> str:
            content = user_content(sentence)
            return (template_prompt(tok, system, content, tokenize=False) if chat
                    else chat_text(system, content))

        def generate(chunk):
            outs = server.complete_many([(build_text(r["sentence"]), None)
                                         for r in chunk],
                                        max_tokens=args.max_tokens, temperature=0.0,
                                        grammar=grammar)
            return [o.text for o in outs]
    else:
        print(f"Loading {args.model} ...")
        # Greedy decoding, so arms differ by prompt only
        model, tok, sampler = load_model(args.model, args.adapter,
                                         quantize=args.quantize,
                                         device=args.device, dtype=args.dtype)

        def build_prompt(sentence: str):
            content = user_content(sentence)
            return (template_prompt(tok, system, content) if chat
                    else encode_chat(tok, system, content))

        def generate(chunk):
            return batch_generate(
                model, tok, prompts=[build_prompt(r["sentence"]) for r in chunk],
                max_tokens=args.max_tokens, sampler=sampler, grammar=grammar).texts

    n, n_fail = len(todo), 0
    dropped: Counter = Counter()   # predicates the engine has no encoder for
    outcomes: Counter = Counter()  # ok / InvalidTree reason
    t0 = time.time()

    with open(out_path, "a") as out:
        for start in range(0, n, args.batch_size):
            chunk = todo[start:start + args.batch_size]
            for r, raw in zip(chunk, generate(chunk)):
                text = raw.split("<|im_end|>")[0]
                # The tree is executed as emitted, or not at all
                program, json_obj, outcome = parse_generation(text)
                outcomes[outcome] += 1
                if json_obj is not None:
                    # Names the representation has no encoder for are
                    # dropped at execution (e.g. under --representation baseline)
                    dropped.update(name for name in relation_names(json_obj)
                                   if name not in rel_num)
                if json_obj is None:
                    n_fail += 1
                # Unparsed rows still written, json_obj null, keeping the full denominator
                row = {"row_idx": r["row_idx"], "scan_id": r["scan_id"],
                       "target_id": r["target_id"],
                       "target_label": r["target_label"],
                       "sentence": r["sentence"], "json_obj": json_obj,
                       "raw": text, "program": program, "outcome": outcome}
                if keyed:
                    row["dataset_idx"] = r["dataset_idx"]
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

            done = min(start + args.batch_size, n)
            rate = done / max(time.time() - t0, 1e-9)
            print(f"[{done}/{n}] failures {n_fail}  {rate * 60:.1f} rows/min  "
                  f"ETA {(n - done) / rate / 60:.0f}m", flush=True)

    if args.backend == "hf":
        free_memory()

    meta["written"] = {"rows": len(done_rows(out_path)), "failures_this_run": n_fail,
                       "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    meta["written"]["outcomes_this_run"] = dict(outcomes.most_common())
    meta["output_sha256"] = sha256(out_path)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"\nWrote {out_path}  (+ {meta_path.name})")
    print(f"  {n - n_fail}/{n} utterances produced a runnable program "
          f"({n_fail} failures)")
    print("  outcomes: " + ", ".join(f"{k} {v}" for k, v in outcomes.most_common()))
    if dropped:
        total = sum(dropped.values())
        print(f"  {total} predicate mentions dropped — no encoder in "
              f"{representation} for these names:")
        for name, count in dropped.most_common(15):
            print(f"      {name:<20} {count}")
        print("  (each dropped mention is one constraint the engine never sees)")
    else:
        print("  every predicate emitted maps onto an encoder")
    try:
        out_display = out_path.relative_to(REPO_ROOT)
    except ValueError:
        out_display = out_path
    print(f"\nRebuild features before scoring:\n"
          f"  python scripts/evaluation/build_features.py --representation "
          f"{representation} --parses {out_display}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
