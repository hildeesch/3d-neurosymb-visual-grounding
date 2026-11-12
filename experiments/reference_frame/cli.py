#!/usr/bin/env python3
"""Reference-frame experiment commands.

Stages select and parse rows, join the frame pass to grounding parses, and
compare baseline/parsed/inverted arms. `select-doorway` and `doorway` apply the
same flow to entry language.

Frames only replace the left/right matrices in memory; they do not change the
grounding parse or its scored artifacts. This pipeline supports v4 only.

    python -m experiments.reference_frame <stage> --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.reference_frame import ablation, frame_pass, rows, selection
from grounding.parsing.generation import add_model_args
from grounding.resolution.config import (CONTROL_FLAGS, add_resolver_flags,
                                         resolver_from_args, resolver_suffix)


STAGES = {
    "select-rows": (selection.cmd_select_rows,
                    "pick frame-stating rows from RAW train utterances"),
    "parse": (frame_pass.cmd_parse, "LLM pass writing a viewpoint_frame per row"),
    "join": (selection.cmd_join, "join the v4 parse and the frame parse by row_idx"),
    "ablate": (ablation.cmd_ablate, "baseline / parsed / inverted grounding accuracy"),
    "select-doorway": (selection.cmd_select_doorway,
                       "pick the doorway-entry rows from RAW train utterances"),
    "doorway": (ablation.cmd_doorway, "the doorway-entry sign ablation"),
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True, metavar="stage")

    def stage(name):
        p = sub.add_parser(name, help=STAGES[name][1],
                           description=STAGES[name][1],
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        p.set_defaults(run=STAGES[name][0])
        return p

    p = stage("select-rows")
    p.add_argument("--out", default=str(rows.REFFRAME_ROWS))
    p.add_argument("--max-parse", type=int, default=400,
                   help="pre-parse budget: rows to select (0 = every "
                        "frame-stating row). Both 30B passes run over exactly "
                        "this file, so it is the cost of the diagnostic twice "
                        "over. The order is prefix-stable, so raising it adds "
                        "rows and moves none")

    p = stage("join")
    p.add_argument("--v4", default=str(rows.REFFRAME_V4),
                   help="the normal v4 grounding parse; supplies json_obj")
    p.add_argument("--frames", default=str(rows.REFFRAME_FRAMES),
                   help="the dedicated frame parse; supplies viewpoint_frame only")
    p.add_argument("--out", default=str(rows.REFFRAME_PROBE))
    p.add_argument("--limit", type=int, default=100)

    p = stage("parse")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--model", required=True,
                   help="local model path or Hugging Face model id "
                        "(loaded with transformers/PyTorch)")
    p.add_argument("--limit", type=int, default=None,
                   help="parse only the first N rows of the input, in file "
                        "order. This is a debugging convenience: the probe is "
                        "already the selection")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=220)
    p.add_argument("--print-changes", type=int, default=20)
    p.add_argument("--thinking", action="store_true",
                   help="enable Qwen3 thinking. Off by default: it truncates the JSON")
    p.add_argument("--seed", type=int, default=0,
                   help="sampling seed when --temperature > 0")
    add_model_args(p)

    p = add_resolver_flags(stage("ablate"), CONTROL_FLAGS)
    p.add_argument("--rows", default=ablation.ABLATE_DEFAULT_ROWS,
                   help="probe rows carrying a viewpoint_frame")
    p.add_argument("--out", default=ablation.ABLATE_DEFAULT_OUT)
    p.add_argument("--rows-out", default=ablation.ABLATE_DEFAULT_ROWS_OUT)
    p.add_argument("--features-cache", default=str(ablation.ABLATE_DEFAULT_CACHE))
    p.add_argument("--rebuild-features", action="store_true")

    p = stage("select-doorway")
    p.add_argument("--out", default=str(rows.DOORWAY_ROWS))
    p.add_argument("--max-parse", type=int, default=0,
                   help="pre-parse budget: rows to select (0 = every "
                        "entry-language frame row, which is what the diagnostic "
                        "uses). The order is prefix-stable, so raising it adds "
                        "rows and moves none")

    p = add_resolver_flags(stage("doorway"), CONTROL_FLAGS)
    p.add_argument("--rows", default=ablation.DOORWAY_DEFAULT_ROWS,
                   help="the 30B v4 parse of the select-doorway rows. Pointing "
                        "this at a test parse is allowed but makes the report "
                        "declare itself a test diagnostic")
    p.add_argument("--frames", default=str(rows.DOORWAY_FRAMES),
                   help="the frame pass over the select-doorway rows, written "
                        "by the SAME `parse` stage the general diagnostic uses. "
                        "`doorway` keeps only rows whose frame names a doorway "
                        "observer origin, and checks this file's sidecar "
                        "against the general pass's.")
    p.add_argument("--features-cache", default=str(ablation.DOORWAY_DEFAULT_CACHE),
                   help="per-scene feature cache for this diagnostic, built on "
                        "demand for the scans the selection touches. It is NOT "
                        "the shared extended tensor and build_features.py "
                        "cannot produce it: that path is test-split only")
    p.add_argument("--rebuild-features", action="store_true")
    p.add_argument("--out", default=ablation.DOORWAY_DEFAULT_OUT)
    p.add_argument("--rows-out", default=ablation.DOORWAY_DEFAULT_ROWS_OUT)
    p.add_argument("--no-write", action="store_true", help="print the report only")
    p.add_argument("--allow-room-centre-sign", action="store_true",
                   help="do not fail the control if wall endpoints are unavailable")

    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    # Only the two stages that score geometry carry the resolver flags
    if hasattr(args, "norm_mode"):
        args._resolver = resolver_from_args(args)
        args._suffix = resolver_suffix(args._resolver)
    return args.run(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
