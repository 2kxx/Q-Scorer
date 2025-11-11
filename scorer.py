from PIL import Image

import torch.nn as nn
import torch

from typing import List

from src.model.builder import load_pretrained_model

from src.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from src.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


class Scorer(nn.Module):
    def __init__(self, model_path="", model_base="", preprocessor_path="", device="cuda:0"):
        super().__init__()

        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path,
            model_base,
            "qscorer_lora",
            preprocessor_path=preprocessor_path,
            device=device
        )

        prompt = "USER: How would you rate the quality of this image?\n<|image|>\nASSISTANT: The quality of the image is"

        self.preferential_ids_ = [tokenizer.convert_tokens_to_ids(['<score5>', '<score4>', '<score3>', '<score2>', '<score1>'])]

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def forward(self, image: List[Image.Image]):
        image = [self.expand2square(img, tuple(int(x*255) for x in self.image_processor.image_mean)) for img in image]
        with torch.inference_mode():
            image_tensors = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().to(self.model.device)
            embedding = None
            current_input = self.input_ids.repeat(image_tensors.shape[0], 1)  # 扩展输入以匹配图像数量

            for i in range(2):
                output = self.model(
                    input_ids=current_input,
                    images=image_tensors,
                    output_hidden_states=True
                )
                logits = output["logits"][:, -1]  # 获取最后一个位置的 logits
                probs = torch.softmax(logits, dim=-1)
                if i == 1:
                    embedding = output["hidden_states"][:, -1, :]
                    output_logits = output["logits"][:, -1]

                vocab_ids = torch.argmax(probs, dim=-1)
                print(vocab_ids)
                current_input = torch.cat([current_input, vocab_ids.unsqueeze(1)], dim=1)

            scores = self.model.deepmlp(embedding)
            return scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--model-base", type=str)
    parser.add_argument("--preprocessor-path", type=str)
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--img_path", type=str)
    args = parser.parse_args()

    scorer = Scorer(
        model_path=args.model_path,
        model_base=args.model_base,
        preprocessor_path=args.preprocessor_path,
        device=args.device
    )

    print(scorer([Image.open(args.img_path)]).tolist())
