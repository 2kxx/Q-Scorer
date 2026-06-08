"""Installation-free Q-Scorer quick start, mirroring the DeQA-Score UX.

Run AFTER you produce a merged checkpoint via:

    python scripts/merge_lora_save.py \
        --model-path  checkpoints/Qscorer_lora_5_1 \
        --model-base  /path/to/mplug-owl2-llama2-7b \
        --preprocessor-path ./preprocessor/ \
        --output-dir  checkpoints/Qscorer_merged

Then either:

    python quick_start.py --model checkpoints/Qscorer_merged --img fig/boat.jpg

or, once you have uploaded the merged checkpoint to the Hugging Face Hub:

    python quick_start.py --model your-user/qscorer-merged --img https://.../img.jpg

No need to `pip install` this repo to score images with the merged
checkpoint: the modeling code travels inside the checkpoint folder and
is loaded via `trust_remote_code=True`.
"""

import argparse
from io import BytesIO

import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM


def _open_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith(("http://", "https://")):
        return Image.open(BytesIO(requests.get(path_or_url, stream=True).content))
    return Image.open(path_or_url)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="2kxx/Qscorer_merged",
                        help="Path to a local merged checkpoint dir, or a HuggingFace Hub repo id.")
    parser.add_argument("--img", action="append", required=True,
                        help="Path or URL to an image. Pass multiple --img flags to batch.")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-impl", default="eager",
                        help="Set to 'flash_attention_2' if flash-attn is installed for faster inference.")
    args = parser.parse_args()

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    print(f"Loading model from {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        attn_implementation=args.attn_impl,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
    )

    images = [_open_image(p) for p in args.img]
    print(f"Scoring {len(images)} image(s) ...")
    scores = model.score(images)
    for path, s in zip(args.img, scores):
        print(f"  {path}: {s:.4f}")


if __name__ == "__main__":
    main()
