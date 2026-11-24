"""Component ablations and factor interactions for the full grounding system.

Three factors, each with the thesis level and ablated levels:

  parser          student (LoRA Qwen3-1.7B <- Qwen3-30B-A3B) | the other
                  students of the teacher x student grid (Qwen2.5-Coder-1.5B
                  and/or the Qwen3-Coder-30B-A3B teacher) | zero-shot teachers
                  (teacher_qwen, teacher_qwen_coder, teacher_gpt4o)
  representation  extended (thesis encoders) | lasp (upstream encoders only)
  executor        thesis (zscore_sigmoid)    | lasp (softmax)

The parser arms' teacher and student models are factors of their own in the
grid design (teacher x student x representation x executor).

Every cell scores the same test rows, so ablations are paired contrasts and
interactions are paired difference-in-differences. See ``design`` for what each
level runs and ``stats`` for the tests.

    python -m experiments.ablation --help
"""
