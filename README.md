# Symbolic 3D grounding on Nr3D

This project turns referring expressions into symbolic programs that identify
objects in ScanNet scenes. It extends LaSP with parser adaptation, colour,
isolation, wall proximity, ordinal features and score calibration. Reference-frame
interpretation is studied in a separate experiment.

**This branch (`tree-track`) replaces the v4 parser with a grammar-constrained
tree parser.** The parser writes the executable program tree directly (prompt
`tree_v1`), decoding is constrained to a fixed grammar, and `to_executable`
checks the same grammar afterwards -- it rejects and never repairs. The v4
`clean_program` / `lower_program` repair and lowering steps are not part of the
pipeline. See [The tree parser](#the-tree-parser). Every parser output on this
branch has `tree_v1` in its name and its own directory, so the v4 parser's
artifacts and results in the same working tree are never overwritten.

## Setup

Use Python 3.10 or later. Run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[evaluation]"
```

Parser generation and LoRA training run on PyTorch (Linux/CUDA, CPU fallback),
in a separate environment:

```bash
python -m venv .venv-train && source .venv-train/bin/activate
python -m pip install -e ".[training]"          # add ,quantization for 4/8-bit
```

The evaluation dependencies pin an older Transformers version and should not be
combined with the training extra. Checkpoint scoring also needs compatible
evaluation packages; that combined setup is not specified here. Dependency
versions are not a validated lock.

### Hardware

| Step | Model | Approx. memory |
| --- | --- | --- |
| Student parsing / scoring | Qwen3-1.7B (+ LoRA) | ~4 GB GPU (bf16), or CPU |
| Student LoRA training | Qwen3-1.7B, batch 4, `--grad-checkpoint` | ~6–8 GB GPU |
| Student parsing / training | Qwen2.5-Coder-1.5B-Instruct (`--student coder`) | as Qwen3-1.7B, slightly less |
| Teacher rollouts | Qwen3-30B-A3B, `--quantize 4bit` | ~18 GB GPU; does not load on a 16 GB GPU |
| Teacher rollouts | Qwen3-30B-A3B or Qwen3-Coder-30B-A3B-Instruct GGUF via `--backend llamacpp` | 16 GB GPU + ~16 GB free RAM (see below) |
| Teacher rollouts | `--backend openai` (GPT-4o) | none locally; needs `OPENAI_API_KEY` |

Check your GPU with `nvidia-smi`. All generators take `--device`, `--dtype`
and `--quantize`. With the default `--device auto`, scripts stop when no GPU
is usable instead of silently running on CPU; pass `--device cpu` to run on
CPU on purpose. If PyTorch warns that the NVIDIA driver is too old, install a
torch build for your driver's CUDA version (shown top right in `nvidia-smi`),
e.g. `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124`
for a CUDA 12.2 driver (cu118 if that still fails). Transformers 5.x needs
torch >= 2.6.

### Teacher on a 16 GB GPU: llama.cpp

Qwen3-30B-A3B at 4 bits (~18 GB) does not fit a 16 GB GPU through
Transformers. llama.cpp can run it there: `--n-cpu-moe N` keeps the expert
weights of N layers in system RAM, and because only ~3B parameters are active
per token this stays reasonably fast. (On Apple Silicon the original MLX run
fitted because GPU and CPU share one memory pool.)

1. **Get `llama-server`.** Either download a prebuilt Linux Vulkan release from
   https://github.com/ggml-org/llama.cpp/releases (no CUDA toolkit needed), or
   build with CUDA (needs a CUDA toolkit your driver supports, e.g. 12.2):
   ```bash
   git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
   cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j
   # binary: build/bin/llama-server
   ```
2. **Download the 4-bit GGUF** (~18.6 GB) into `models/` (git-ignored):
   ```bash
   hf download Qwen/Qwen3-30B-A3B-GGUF --include "*Q4_K_M*" --local-dir models/
   ```
3. **Start the server** in its own terminal. Raise `--n-cpu-moe` if it runs
   out of GPU memory, lower it while `nvidia-smi` shows room (lower = faster):
   ```bash
   llama-server -m models/Qwen3-30B-A3B-Q4_K_M.gguf -ngl 99 --n-cpu-moe 16 \
     -c 16384 -np 8 --port 8080
   ```
   `-np 8` serves 8 requests in parallel (one prompt's 8 samples); `-c` is
   shared between them, so each gets 2048 tokens, enough for prompt + output.
4. **Generate** (try `--limit 20` first to check speed and the share of
   samples outside the grammar):
   ```bash
   python -m experiments.parser_adaptation.generate_rollouts --backend llamacpp \
     --teacher qwen --limit 8750 --n-samples 8 --temp 0.7
   # -> data/rollouts/nr3d_rollouts_qwen_tree_v1.jsonl
   ```

The prompt is rendered with the Hugging Face tokenizer of the served model
(`Qwen/Qwen3-30B-A3B`, or `Qwen/Qwen3-Coder-30B-A3B-Instruct` for the coder
GGUF; only tokenizer files are downloaded), thinking off, identically to the
other backends, and sent as text to `/completion`. The tokenizer follows the
`--teacher` preset or the served GGUF's name; a mismatch between the two (or
with an explicit `--tokenizer`) is refused. Sampling
is pure temperature (llama.cpp's default top-k/top-p/min-p are switched off per
request) and every sample has its own seed. `parse_split.py --backend llamacpp`
does the same for greedy zero-shot parses. For the paper: this teacher is the
Q4_K_M GGUF quantisation, not the MLX 4-bit one, so it is the same model at a
comparable but not identical quantisation.

### Moving from the MLX version

The pipeline originally ran on Apple Silicon with MLX. It now uses PyTorch,
Transformers and PEFT; command-line flags are unchanged except where noted.

- Models are Hugging Face ids: `Qwen/Qwen3-30B-A3B` (with `--quantize 4bit`)
  replaces `mlx-community/Qwen3-30B-A3B-4bit`.
- Sampling is pure temperature sampling (top-p/top-k off), as MLX's sampler
  was, and is now seeded (`--seed`). Outputs are not token-identical to MLX runs.
- Old MLX adapters convert to PEFT without retraining:
  `python scripts/prepare/convert_mlx_adapter.py <mlx_adapter_dir> <peft_dir>`.
- `train_lora.py` replaces `mlx_lm lora` and takes the same flags and YAML. It
  uses mlx's Adam (no bias correction, as the thesis runs) and differs in a few
  details (loss over real tokens only, LoRA init draws); see its docstring.
  Retrained adapters will not reproduce the MLX numbers exactly.
- Parse and rollout files record `backend` in their `.meta.json`; resuming
  into a file written by the MLX version is refused.

Data, model weights, adapters and generated results are not included. You need:

- Aligned ReferIt3D scene exports under `data/raw/referit3d/scan_data/`, including
  points, instance locations, labels and colour mixtures.
- Annotations and split lists under `data/raw/`, including
  `data/raw/referit3d/annotations/`, and annotation CSVs under `data/corpora/`.
- CLIP weights for feature extraction and Qwen weights for parser generation.

Prepare the scene exports with external ScanNet/ReferIt3D tooling. Then create
local baseline data links and follow the benchmark builder's input options:

```bash
python scripts/prepare/init_tree.py
python scripts/prepare/build_benchmark.py --help
```

Use `init_tree.py --check` to check the links and `build_benchmark.py --verify`
to check existing caches. Neither supplies the dataset. Keep the frozen dev and
validation splits when comparing runs.

## Running experiments

With development parses available, build features and compare the two systems:

```bash
python scripts/evaluation/build_features.py \
  --parses data/parses/dev/nr3d_dev_teacher_qwen_tree_v1_ext.jsonl --split dev
python scripts/evaluation/compare_baseline.py \
  --parses data/parses/dev/nr3d_dev_teacher_qwen_tree_v1_ext.jsonl --split dev --totals parses \
  --norm-mode zscore_sigmoid --out-dir artifacts/results/representation/dev_comparison_tree_v1
```

Feature building replaces the cache for that system and split. Evaluate using
the same parses. `--totals parses` scores the supplied subset; `--totals official`
counts missing test parses as misses.

### The tree parser

The parser (`src/grounding/parsing/tree.py`) writes the executable schema
itself: `category` / `color` / `relations` -> `relation_name` / `negative` /
`rank` / `objects`, anchors as full nodes, so an anchor carries its own
relations, at most three levels. Relation names and colours come from a closed
vocabulary (one name per encoder, no synonyms, 18 colours), and each relation
fixes how many objects it takes. `tree.gbnf()` renders that grammar for
constrained decoding -- the llama.cpp `grammar` field for a teacher, an
xgrammar logits processor for a Transformers model -- and `to_executable`
checks it afterwards: a tree outside it is rejected with a reason
(`relation_name`, `arity`, `too_deep`, ...) and never repaired.

`--no-grammar` decodes the same prompt without the grammar (the grammar
ablation; outputs get `_nogrammar` in their names). The OpenAI API cannot
decode under a grammar, so a GPT-4o parser always needs `--no-grammar`. Every
`.meta.json` records the prompt id and the grammar's sha256; a resume under a
different prompt or grammar is refused.

Through the pipeline: rollout samples keep the model's tree and its outcome;
`filter_rollouts.py` executes `to_executable(tree)` and reports the invalid rate
per sample and per reason and the depth of the valid and kept trees;
`build_winner_pool.py` keeps the tree as emitted (rejecting a tree whose keys
are out of the grammar's order, `key_order`) and ranks winners by cue coverage
counted by tree relation name at every level; `build_uniform_sets.py` writes
`tree.dumps(program)` as the assistant turn; `parse_split.py` and
`score_checkpoint.py` store and score `json_obj = to_executable(tree)`.

The frozen validation split (by prompt id) still holds; ids a teacher's tree
pool does not cover are reported and never trained on. The size-matched arm
(`sft_tree_ext_random_size_matched`) draws 3908 prompts -- the v4 margin arm's
count -- so the tree students train on as many prompts as the v4 students did,
with the same LoRA YAML (`lora_config_sft_v4_ext.yaml`), 1000 updates and seeds
0-2.

Zero-shot parses (training environment, llama-server serving the teacher):

```bash
python -m experiments.parser_adaptation.parse_split --representation extended \
  --teacher qwen --rows data/corpora/nr3d_dev.jsonl           # [--no-grammar]
python -m experiments.parser_adaptation.score_checkpoint --split dev --n 0 \
  --rescore data/parses/dev/nr3d_dev_teacher_qwen_tree_v1_ext.jsonl \
  --out artifacts/results/parser_adaptation/dev/tree_v1/teacher_qwen.json
# GPT-4o (OPENAI_API_KEY set)
python -m experiments.parser_adaptation.parse_split --representation extended \
  --teacher gpt4o --no-grammar
```

### Teacher x student grid

Two teachers and two students, every pair trained the same way:

| | teacher `qwen`: Qwen3-30B-A3B | teacher `qwen_coder`: Qwen3-Coder-30B-A3B-Instruct |
| --- | --- | --- |
| student `qwen3`: Qwen3-1.7B | `student` (full system) | `student_from_qwen_coder` |
| student `coder`: Qwen2.5-Coder-1.5B-Instruct | `student_coder` | `student_coder_from_qwen_coder` |
| zero-shot | `teacher_qwen` | `teacher_qwen_coder` |

The coder teacher has the same 30B-total / 3B-active MoE layout, so it runs
through llama.cpp exactly like the Qwen3 teacher (Q4_K_M GGUF from
`unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF`, ~17.3 GB). Each teacher's rollouts
use the frozen 8750-prompt draw, T=0.7, 8 samples and 320 tokens under the
grammar. Every student is trained on the size-matched arm of its teacher's
winners, the seed is chosen on dev with `select_adapter.py`, and the test rows
are parsed greedily, with and without the grammar. All four pairs are trained
on this branch.

`experiments/parser_adaptation/teacher_student_grid.py` knows every path and
command. `plan` shows each step's status and command; `run` executes the
missing steps in this environment and skips finished outputs:

```bash
python -m experiments.parser_adaptation.teacher_student_grid plan --todo
# llama-server serving Qwen3-30B-A3B-Q4_K_M.gguf:
python -m experiments.parser_adaptation.teacher_student_grid run --teachers qwen \
    --steps rollouts zeroshot zeroshot_nogrammar
# llama-server serving Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf:
python -m experiments.parser_adaptation.teacher_student_grid run --teachers qwen_coder \
    --steps rollouts zeroshot zeroshot_nogrammar
# server stopped (training environment):
python -m experiments.parser_adaptation.teacher_student_grid run \
    --steps filter winners sets train dev select parse parse_nogrammar
```

Outputs per teacher `<t>` and pair `<s>_from_<t>`:
`data/rollouts/nr3d_rollouts_<t>_tree_v1.jsonl`,
`data/distilled/nr3d_{keepers,winners}_<t>_tree_v1_ext.jsonl`,
`data/training_sets_tree_v1_<t>/`, `artifacts/adapters_tree_v1_<s>_from_<t>/`,
`artifacts/results/parser_adaptation/dev/grid_tree_v1/<s>_from_<t>/` and
`data/parses/test/nr3d_test_<arm>_tree_v1[_nogrammar]_ext.jsonl`. A
`train_sft_arms.py` run refuses an adapter directory that holds another base
model's or training set's adapter.

### Ablations and interactions

`experiments/ablation/` scores every combination of three factors on the same
test rows, so each ablation is a paired contrast and each interaction a paired
difference-in-differences:

| Factor | Full system | Ablated level |
| --- | --- | --- |
| parser | LoRA student (Qwen3-1.7B <- Qwen3-30B-A3B) | the other three students of the teacher x student grid; untrained teachers `teacher_qwen`, `teacher_qwen_coder`, `teacher_gpt4o` (opt-in) |
| representation | `extended` encoders | `lasp`: upstream LaSP encoders and predicate table (no colour, isolation, thesis wall encoder, smallest/lowest, ordinal rank) |
| executor | `thesis` (`zscore_sigmoid`) | `lasp` (softmax) |

The resolver equals LaSP's `parse` only under LaSP's arithmetic; the frozen
`zscore_sigmoid` normalisation is a change, so it is its own factor
(`tests/test_executor_equivalence.py` checks both statements). The `lasp`
representation is run in the baseline tree, with `--executor thesis` for the
thesis-executor cell.

Teacher parses come from the untrained teacher under the rollout prompt and
grammar, with the same `to_executable` step as the students, so only the model
changes. They and the student arms come from the teacher x student grid above
(`parse_split.py --teacher gpt4o --no-grammar` for the opt-in GPT-4o arm).

Then, in the evaluation environment:

```bash
python -m experiments.ablation plan                 # cells, commands, missing inputs
python -m experiments.ablation run                  # builds features per parser arm, scores all cells
python -m experiments.ablation run --no-grammar     # the same on the parses decoded without the grammar
python -m experiments.ablation analyse              # re-run the statistics only
```

By default `run` scores the six local parser arms (the four grid students and
both Qwen teachers) at every representation and executor level: 24 cells.
`--parsers`, `--representations` and `--executors` restrict the design (for
example `--executors thesis` for a parser x representation 2x2); add
`teacher_gpt4o` to `--parsers` to include GPT-4o (always without the grammar).
Results go to `artifacts/results/ablation/<split>_tree_v1[_nogrammar]/` (the v4
parser's results in `artifacts/results/ablation/<split>/` are left alone):
per-cell dumps, `cells.json` (inputs and
digests), and `ablation.json`/`ablation.md` with a teacher x student accuracy
table per representation/executor cell, the cell table, every ablation against
the full system, and three families of factorial designs, each crossed with
representation and executor:

- grid: teacher[qwen|qwen_coder] x student[qwen3|coder] over the four trained
  students (the teacher and student main effects and their interaction);
- distillation: each student against its own teacher, zero-shot;
- teachers: `teacher_qwen` against each other teacher, zero-shot.

Effects are in accuracy points; +1 is the first level named for each factor
(the thesis model, or in a distillation design the student), so a negative
grid effect favours the code model. Tests are exact paired sign-flip tests
(exact McNemar for two cells), 95% CIs are seeded bootstraps over expressions,
and interaction p-values are also Holm-adjusted within each family.

The experiments are grouped by contribution:

| Folder under `experiments/` | Purpose |
| --- | --- |
| `parser_adaptation/` | Rollouts, training sets, LoRA training and adapter selection |
| `representation/` | Feature probes, ablations and sensitivity sweeps |
| `ablation/` | Parser (teacher x student grid) x representation x executor ablations and interactions |
| `calibration/` | Normalization, anchor binding and combination rules |
| `reference_frame/` | Observer-frame and doorway-entry experiments |

Each command has `--help`. For example:

```bash
python -m experiments.parser_adaptation.train_sft_arms --dry-run
python -m experiments.representation.probe_feature --help
python -m experiments.calibration.score_aggregation --help
python -m experiments.reference_frame --help
```

Training uses each teacher's R=8 tree pool and the IDs in
`experiments/parser_adaptation/validation_split.json`. Remove `--dry-run` to train; select adapters on dev data, then pass the
chosen adapter to `parse_split --style sft --adapter <path>`. Adapters are PEFT
directories; checkpoints are saved as `<adapter>/NNNNNNN_adapters/`.

Checkpoint selection credits every top tie; headline evaluation selects one
candidate with `torch.topk`. Preserve the scoring settings and row-level results
when comparing experiments.

## Results

The v4 parser's maintained run (on `main`) is
`sft_v4_ext_random_size_matched__seed2`. Its recorded test accuracy is
**64.28%** (4,811/7,485), versus **61.88%** (4,632/7,485) for the baseline using
the same parses. The tree parser's results are not recorded yet.

Local results live in `artifacts/results/`, grouped into `final/`, `baselines/`,
`parser_adaptation/`, `representation/`, `calibration/` and `reference_frame/`.
`final/manifest.json` identifies the maintained comparison, predictions and
significance report. Generated results are Git-ignored.

## Contributing

`src/grounding/` contains the parser, scene representation and resolver.
Inherited components live in `src/grounding/_vendor/lasp/`; `baselines/lasp/`
retains independent baseline scoring. Both include local loading adaptations.
Data preparation and evaluation commands live in `scripts/`.

Include regression coverage for behavior changes, and keep datasets, weights,
generated outputs and credentials out of commits. The project was developed with AI coding
assistance.

## License

Original contributions use the [MIT License](LICENSE). Inherited LaSP code retains
its copyright in [baselines/lasp/LICENSE](baselines/lasp/LICENSE) and
[src/grounding/_vendor/lasp/LICENSE](src/grounding/_vendor/lasp/LICENSE).
ScanNet and ReferIt3D data must be obtained under their own terms.
