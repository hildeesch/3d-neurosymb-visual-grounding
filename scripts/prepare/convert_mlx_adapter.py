#!/usr/bin/env python3
"""Convert an mlx-lm LoRA adapter directory into a PEFT adapter directory.

Lets adapters trained on Apple Silicon with ``mlx_lm lora`` (e.g. the
maintained ``sft_v4_ext_random_size_matched__seed2``) be loaded by the PyTorch
pipeline (``parse_split.py --adapter``, ``score_checkpoint.py --adapter``)
without retraining. Needs only ``safetensors``, ``torch`` and ``peft``.

mlx-lm stores ``<module>.lora_a`` (in, r) and ``<module>.lora_b`` (r, out) and
adds ``scale * x @ lora_a @ lora_b``. PEFT stores ``lora_A.weight`` (r, in) and
``lora_B.weight`` (out, r) and adds ``(alpha / r) * B(A(x))``, so the matrices
are transposed and ``lora_alpha = scale * r``.

    python scripts/prepare/convert_mlx_adapter.py \\
        artifacts/adapters/sft_v4_ext_random_size_matched__seed2 \\
        artifacts/adapters_peft/sft_v4_ext_random_size_matched__seed2

Periodic ``NNNNNNN_adapters.safetensors`` checkpoints become
``NNNNNNN_adapters/`` sub-directories (skip them with ``--final-only``).
Converted outputs should match MLX up to float rounding; verify on a few rows
with ``score_checkpoint.py`` before relying on them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PEFT_PREFIX = "base_model.model."


def convert_tensors(mlx_weights: dict, rank: int) -> tuple[dict, list[str], list[int]]:
    import torch

    out, targets, layers = {}, set(), set()
    pattern = re.compile(r"^(.*)\.lora_([ab])$")
    for key, value in mlx_weights.items():
        m = pattern.match(key)
        if not m:
            raise SystemExit(f"unexpected tensor {key!r}: only LoRA (not DoRA or "
                             "full fine-tuning) adapters are supported")
        module, which = m.groups()
        tensor = value.to(torch.float32).T.contiguous()
        if which == "a":
            if tensor.shape[0] != rank:
                raise SystemExit(f"{key}: rank {tensor.shape[0]} != config rank {rank}")
            out[f"{PEFT_PREFIX}{module}.lora_A.weight"] = tensor
        else:
            if tensor.shape[1] != rank:
                raise SystemExit(f"{key}: rank {tensor.shape[1]} != config rank {rank}")
            out[f"{PEFT_PREFIX}{module}.lora_B.weight"] = tensor
        targets.add(module.rsplit(".", 1)[-1])
        layer = re.search(r"\.layers\.(\d+)\.", f".{module}.")
        if layer:
            layers.add(int(layer.group(1)))
    return out, sorted(targets), sorted(layers)


def write_peft(dst: Path, weights: dict, mlx_cfg: dict, targets: list[str],
               layers: list[int], base_model: str) -> None:
    from peft import LoraConfig
    from safetensors.torch import save_file

    lora = mlx_cfg["lora_parameters"]
    rank = int(lora["rank"])
    dst.mkdir(parents=True, exist_ok=True)
    LoraConfig(r=rank, lora_alpha=float(lora["scale"]) * rank,
               lora_dropout=float(lora.get("dropout", 0.0)),
               target_modules=targets, layers_to_transform=layers,
               bias="none", task_type="CAUSAL_LM",
               base_model_name_or_path=base_model).save_pretrained(str(dst))
    save_file(weights, str(dst / "adapter_model.safetensors"),
              metadata={"format": "pt"})


def training_config(mlx_cfg: dict, src: Path, targets, layers) -> dict:
    """The run settings eval_parser_assistant_loss.py reads, from MLX's config."""
    keep = ("model", "data", "seed", "batch_size", "iters", "max_seq_length",
            "val_batches", "steps_per_eval", "steps_per_report", "save_every",
            "learning_rate", "num_layers", "lora_parameters", "mask_prompt",
            "grad_checkpoint", "optimizer", "fine_tune_type")
    cfg = {k: mlx_cfg[k] for k in keep if k in mlx_cfg}
    cfg.update({"trainer": "mlx_lm lora (converted by convert_mlx_adapter.py)",
                "converted_from": str(src), "lora_targets": targets,
                "lora_layers": layers})
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="mlx-lm adapter directory")
    ap.add_argument("dst", type=Path, help="PEFT adapter directory to create")
    ap.add_argument("--base-model", default=None,
                    help="Hugging Face id of the base model (default: the "
                         "'model' recorded in the MLX adapter_config.json)")
    ap.add_argument("--final-only", action="store_true",
                    help="skip the periodic NNNNNNN_adapters.safetensors files")
    args = ap.parse_args()

    from safetensors.torch import load_file

    cfg_path = args.src / "adapter_config.json"
    if not cfg_path.exists():
        sys.exit(f"{cfg_path} not found")
    mlx_cfg = json.loads(cfg_path.read_text())
    if mlx_cfg.get("fine_tune_type", "lora") != "lora":
        sys.exit(f"fine_tune_type {mlx_cfg['fine_tune_type']!r}: only lora converts")
    rank = int(mlx_cfg["lora_parameters"]["rank"])
    base_model = args.base_model or mlx_cfg.get("model")
    if not base_model:
        sys.exit("no base model recorded; pass --base-model")
    if "mlx-community" in base_model:
        print(f"note: {base_model} is an MLX conversion; pass --base-model with "
              "the matching Hugging Face id (e.g. Qwen/Qwen3-1.7B)", file=sys.stderr)

    jobs = [(args.src / "adapters.safetensors", args.dst)]
    if not args.final_only:
        for ckpt in sorted(args.src.glob("[0-9]*_adapters.safetensors")):
            jobs.append((ckpt, args.dst / ckpt.name.removesuffix(".safetensors")))

    for src_file, dst in jobs:
        if not src_file.exists():
            sys.exit(f"{src_file} not found")
        weights, targets, layers = convert_tensors(load_file(str(src_file)), rank)
        write_peft(dst, weights, mlx_cfg, targets, layers, base_model)
        cfg = training_config(mlx_cfg, args.src, targets, layers)
        cfg["model"] = base_model
        (dst / "training_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"{src_file.name} -> {dst}  ({len(weights)} tensors, "
              f"{len(targets)} module types, {len(layers)} layers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
