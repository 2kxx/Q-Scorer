"""Merge a Q-Scorer LoRA checkpoint into the base model and save a
self-contained checkpoint that can be loaded with a single line:

    model = AutoModelForCausalLM.from_pretrained(
        out_dir,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.score([Image.open("img.jpg")])

Why this script exists
----------------------
The original training pipeline saves two things:
  * a PEFT LoRA adapter directory
  * a `non_lora_trainables.bin` file that, due to a bug in
    `get_peft_state_non_lora_maybe_zero_3`, contains the *entire* base model
    instead of just the few extra trainable tensors (deepmlp, lm_head,
    resized embed_tokens). That is why your saved LoRA folder is ~14 GB.

To load it back you need: base mPLUG-Owl2 weights + non_lora_trainables +
LoRA. This script does exactly that ONCE, merges the LoRA into the base,
and saves one consolidated checkpoint containing:
  * merged model weights (`pytorch_model*.safetensors`)
  * tokenizer files (with `<score1>`..`<score5>` already added)
  * `preprocessor_config.json` (CLIP image processor)
  * the custom modeling code copied side-by-side
  * `config.json` with `auto_map`, `score_token_id`, and `score_tokens`

After this, loading does not need the original 14 GB base anymore.

Usage
-----
    python scripts/merge_lora_save.py \
        --model-path checkpoints/Qscorer_lora_5_1 \
        --model-base /path/to/mplug-owl2-llama2-7b \
        --preprocessor-path ./preprocessor/ \
        --output-dir checkpoints/Qscorer_merged
"""

import argparse
import os
import shutil
import sys

import torch

# Make sure we can import the in-tree package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.builder import load_pretrained_model  # noqa: E402


SCORE_TOKENS = ["<score1>", "<score2>", "<score3>", "<score4>", "<score5>"]

# Files that must travel with the merged checkpoint so it can be loaded
# anywhere via `trust_remote_code=True`. All of these now use only relative
# imports between siblings, so they are drop-in.
_MODEL_FILES = [
    "configuration_mplug_owl2.py",
    "modeling_mplug_owl2.py",
    "modeling_llama2.py",
    "modeling_attn_mask_utils.py",
    "visual_encoder.py",
    "deepmlp.py",
    "utils.py",
]


def _copy_remote_code(src_model_dir: str, out_dir: str) -> None:
    """Copy modeling .py files into the checkpoint dir for trust_remote_code."""
    for fname in _MODEL_FILES:
        src = os.path.join(src_model_dir, fname)
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"Expected model source file not found: {src}. "
                "Run this script from a checkout of the Q-Scorer repo."
            )
        shutil.copy2(src, os.path.join(out_dir, fname))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True,
                        help="Path to the trained LoRA directory (contains adapter_model.bin + non_lora_trainables.bin).")
    parser.add_argument("--model-base", required=True,
                        help="Path to the base mPLUG-Owl2 model (e.g. MAGAer13/mplug-owl2-llama2-7b checkpoint dir).")
    parser.add_argument("--preprocessor-path", required=True,
                        help="Path with the tokenizer + CLIP image processor (typically ./preprocessor/).")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for the merged self-contained checkpoint.")
    parser.add_argument("--torch-dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Dtype to save the merged weights in.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device used for the merge_and_unload step. CPU is fine but slow.")
    parser.add_argument("--no-remote-code", action="store_true",
                        help="Skip copying modeling .py files into the output dir. "
                             "Use this if you intend to load via the in-tree code (no trust_remote_code).")
    parser.add_argument("--safe-serialization", action="store_true", default=True,
                        help="Save as safetensors (default True).")
    args = parser.parse_args()

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.torch_dtype]

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print(f"Loading base + non_lora + LoRA, then merging via PEFT...")
    print(f"  model-base       : {args.model_base}")
    print(f"  model-path (LoRA): {args.model_path}")
    print(f"  preprocessor-path: {args.preprocessor_path}")
    print(f"  device           : {args.device}")
    print("=" * 80)

    # `load_pretrained_model` already does merge_and_unload internally for
    # "qscorer_lora", so the returned model has LoRA fully baked in.
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name="qscorer_lora",
        preprocessor_path=args.preprocessor_path,
        device=args.device,
    )

    # Make absolutely sure the IQA fields are persisted on the config before
    # we save. The original training code only put them on the model as an
    # instance attribute, so without this they would be lost on save+reload.
    score_token_id = tokenizer.convert_tokens_to_ids(SCORE_TOKENS)
    # Match the convention used everywhere else in the repo (reversed order).
    score_token_id_rev = list(score_token_id)
    score_token_id_rev.reverse()
    model.config.score_token_id = score_token_id_rev
    model.config.score_tokens = list(SCORE_TOKENS)
    model.config.vocab_size = len(tokenizer)
    # Wire HuggingFace AutoModel discovery so the one-line UX works.
    model.config.auto_map = {
        "AutoConfig": "configuration_mplug_owl2.MPLUGOwl2Config",
        "AutoModelForCausalLM": "modeling_mplug_owl2.MPLUGOwl2LlamaForCausalLM",
    }
    # Drop transient _name_or_path so it does not point at the original LoRA dir.
    if hasattr(model.config, "_name_or_path"):
        model.config._name_or_path = ""

    # Cast to the requested dtype for a compact on-disk artifact.
    model.to(dtype=dtype)

    print(f"Saving merged checkpoint to {args.output_dir} ...")
    model.save_pretrained(
        args.output_dir,
        safe_serialization=bool(args.safe_serialization),
    )
    tokenizer.save_pretrained(args.output_dir)
    # CLIP image processor saves as preprocessor_config.json.
    image_processor.save_pretrained(args.output_dir)

    if not args.no_remote_code:
        src_model_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "src", "model")
        )
        print(f"Copying modeling files from {src_model_dir} for trust_remote_code support...")
        _copy_remote_code(src_model_dir, args.output_dir)

    print("=" * 80)
    print("Done. Try loading with:")
    print()
    print("    from transformers import AutoModelForCausalLM")
    print("    import torch")
    print("    model = AutoModelForCausalLM.from_pretrained(")
    print(f"        '{args.output_dir}',")
    print("        trust_remote_code=True,")
    print("        attn_implementation='eager',")
    print(f"        torch_dtype=torch.{args.torch_dtype},")
    print("        device_map='auto',")
    print("    )")
    print("    from PIL import Image")
    print("    print(model.score([Image.open('fig/boat.jpg')]))")
    print("=" * 80)


if __name__ == "__main__":
    main()
