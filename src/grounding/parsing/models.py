"""The teacher and student models of the distillation grid, in one place.

Two teachers generate rollouts and parse zero-shot, and two students are
LoRA-trained on each teacher's winners, so every (student, teacher) pair exists:

    teachers  qwen        Qwen3-30B-A3B                  (the thesis teacher)
              qwen_coder  Qwen3-Coder-30B-A3B-Instruct   (code-specialised, same MoE layout)
    students  qwen3       Qwen3-1.7B                     (the thesis student)
              coder       Qwen2.5-Coder-1.5B-Instruct    (code-specialised)

Both teachers are 30B-total / ~3B-active mixture-of-experts models, so the
coder runs through the same llama.cpp route (Q4_K_M GGUF, ``--n-cpu-moe``) on a
16 GB GPU. Neither coder model has a thinking mode; their chat templates ignore
``enable_thinking``, so ``template_prompt`` renders them without a think block.
The Qwen3-30B-A3B template with thinking off adds an empty one. Each teacher's
prompt must therefore be rendered with its own tokenizer. ``template_tokenizer``
picks it from the served GGUF and refuses a mismatch.

Everything here is plain data (no torch / transformers import), so the
evaluation environment can import it too.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TeacherModel", "StudentModel", "TEACHERS", "STUDENTS",
           "DEFAULT_TEACHER", "DEFAULT_STUDENT", "teacher_for_gguf",
           "template_tokenizer"]


@dataclass(frozen=True)
class TeacherModel:
    key: str
    label: str
    hf: str                 # Hugging Face id (Transformers backend, tokenizer)
    gguf_repo: str          # where the Q4_K_M GGUF for llama-server comes from
    gguf_file: str
    #: lower-case substrings that identify this model in a GGUF file name.
    #: Checked most specific first, so "qwen3-coder-30b" wins over "qwen3-30b".
    gguf_match: tuple[str, ...]

    @property
    def tokenizer(self) -> str:
        """Tokenizer whose chat template renders this teacher's prompts."""
        return self.hf


@dataclass(frozen=True)
class StudentModel:
    key: str
    label: str
    hf: str


TEACHERS: dict[str, TeacherModel] = {
    "qwen": TeacherModel(
        key="qwen", label="Qwen3-30B-A3B", hf="Qwen/Qwen3-30B-A3B",
        gguf_repo="Qwen/Qwen3-30B-A3B-GGUF", gguf_file="Qwen3-30B-A3B-Q4_K_M.gguf",
        gguf_match=("qwen3-30b-a3b", "qwen3_30b_a3b")),
    "qwen_coder": TeacherModel(
        key="qwen_coder", label="Qwen3-Coder-30B-A3B-Instruct",
        hf="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        gguf_repo="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        gguf_file="Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        gguf_match=("qwen3-coder-30b-a3b", "qwen3_coder_30b_a3b")),
}

STUDENTS: dict[str, StudentModel] = {
    "qwen3": StudentModel(key="qwen3", label="Qwen3-1.7B", hf="Qwen/Qwen3-1.7B"),
    # The Instruct build, as Qwen3-1.7B is the post-trained (chat) checkpoint
    # too: both start from a model that already follows the ChatML layout the
    # SFT rows are written in.
    "coder": StudentModel(key="coder", label="Qwen2.5-Coder-1.5B-Instruct",
                          hf="Qwen/Qwen2.5-Coder-1.5B-Instruct"),
}

DEFAULT_TEACHER = "qwen"
DEFAULT_STUDENT = "qwen3"


def teacher_for_gguf(name: str | None) -> str | None:
    """The registered teacher a GGUF file name belongs to, or None if unknown."""
    if not name:
        return None
    low = str(name).lower().replace(" ", "")
    # Longest pattern first, so a later, more specific model is never shadowed
    candidates = sorted(((m, key) for key, t in TEACHERS.items() for m in t.gguf_match),
                        key=lambda mk: -len(mk[0]))
    for match, key in candidates:
        if match in low:
            return key
    return None


def template_tokenizer(explicit: str | None, *, teacher: str | None = None,
                       gguf: str | None = None) -> str:
    """The tokenizer that renders prompts for a llama-server teacher.

    Precedence: ``explicit`` (--tokenizer), else the ``teacher`` preset, else
    the teacher recognised from the served ``gguf`` name, else Qwen3-30B-A3B
    (the historical default). Raises ``ValueError`` when the served GGUF is a
    known teacher whose template differs from the one chosen: rendering the
    coder with the Qwen3 template (or the reverse) changes the prompt as well as
    the model.
    """
    served = teacher_for_gguf(gguf)
    if teacher is not None and served is not None and teacher != served:
        raise ValueError(
            f"--teacher {teacher} but llama-server is serving {gguf!r}, which is "
            f"{TEACHERS[served].label}. Restart llama-server with "
            f"{TEACHERS[teacher].gguf_file}.")
    chosen = (explicit or (TEACHERS[teacher].tokenizer if teacher else None)
              or (TEACHERS[served].tokenizer if served else None)
              or TEACHERS[DEFAULT_TEACHER].tokenizer)
    if served is not None and chosen != TEACHERS[served].tokenizer:
        raise ValueError(
            f"llama-server is serving {gguf!r} ({TEACHERS[served].label}) but the "
            f"prompt would be rendered with the {chosen} chat template. Pass "
            f"--tokenizer {TEACHERS[served].tokenizer} (or omit --tokenizer).")
    return chosen
