# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Q-Scorer is an ML research project for MLLM-based Image Quality Assessment (IQA). It builds on mPLUG-Owl2 (LLaMA-2-7B based) with a custom DeepMLP regression head and score tokens. There are no web services, databases, or Docker dependencies.

### Python environment

- **Python 3.10** is required — `torch==2.0.1` (pinned in `pyproject.toml`) has no wheels for Python 3.12+.
- The venv lives at `/workspace/.venv` (created via `uv venv --python 3.10`).
- Activate with `source /workspace/.venv/bin/activate`.
- `numpy<2` must be pinned because `scikit-learn==1.2.2` and other pinned packages were compiled against NumPy 1.x.
- `protobuf` is an unlisted dependency required by the LLaMA tokenizer in `transformers==4.36.1`.

### Running code

- Always set `PYTHONPATH=/workspace:$PYTHONPATH` before running scripts — the codebase uses `from src.xxx import ...` imports relative to the repo root.
- Alternatively, `pip install -e .` installs the `DeQA-Score` package, but PYTHONPATH is still needed for the `src.*` imports used throughout the code.
- Entry points: `scorer.py` (single-image scoring), `scripts/infer_lora.sh` (batch inference), `scripts/train_lora.sh` (LoRA fine-tuning), `scripts/eval_score.sh` (metric evaluation).

### GPU requirement

- All model inference and training require a CUDA GPU. The code hardcodes `device="cuda"` with no CPU fallback.
- The `bitsandbytes` warning about missing GPU support is benign in CPU-only environments — it doesn't block imports.
- Full model loading requires downloading ~14GB base weights from HuggingFace (`MAGAer13/mplug-owl2-llama2-7b`) and LoRA weights (`2kxx/Qscorer_lora_5_1`).

### Linting and testing

- No test suite or linting configuration exists in the repository.
- Use `ruff check --select=E,F src/ scorer.py` for basic linting (ruff is installed in the venv). Pre-existing lint issues are expected.

### Evaluation pipeline (CPU-compatible)

- `src/evaluate/cal_plcc_srcc.py` runs PLCC/SRCC metric computation on prediction JSONs and works without a GPU.
- The tokenizer, image processor (`CLIPImageProcessor`), and `DeepMLP` module all work on CPU for unit-level testing.
