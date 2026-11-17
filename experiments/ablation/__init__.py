"""Component ablations and factor interactions for the full grounding system.

Three factors, each with the thesis level and an ablated level:

  parser          student (LoRA Qwen3-1.7B) | teacher_qwen | teacher_gpt4o
  representation  extended (thesis encoders) | lasp (upstream encoders only)
  executor        thesis (zscore_sigmoid)    | lasp (softmax)

Every cell scores the same test rows, so ablations are paired contrasts and
interactions are paired difference-in-differences. See ``design`` for what each
level runs and ``stats`` for the tests.

    python -m experiments.ablation --help
"""
