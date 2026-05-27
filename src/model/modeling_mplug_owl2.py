#    Copyright 2023 Haotian Liu & Qinghao Ye (Modified from LLaVA)
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import sys
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from transformers import (AutoConfig, AutoModelForCausalLM, LlamaForCausalLM,
                          LlamaModel)
from transformers.modeling_outputs import CausalLMOutputWithPast

# Use relative imports for every sibling module so this file works both as part
# of the `src.model` package and when copied next to its siblings into a
# self-contained checkpoint directory loaded via `trust_remote_code=True`.
from .configuration_mplug_owl2 import (MPLUGOwl2Config, MplugOwlVisionConfig,
                                       MplugOwlVisualAbstractorConfig)
from .deepmlp import DeepMLP
from .modeling_llama2 import replace_llama_modality_adaptive
from .utils import extend_list, find_prefix
from .visual_encoder import MplugOwlVisionModel, MplugOwlVisualAbstractorModel
# Transformers' `get_relative_imports` only walks direct relative imports of the
# entry-point module file. `.modeling_attn_mask_utils` is needed transitively by
# `.modeling_llama2`, so we reference it here to make it visible to the
# trust_remote_code dynamic loader. The alias is intentionally unused.
from .modeling_attn_mask_utils import (  # noqa: F401
    _prepare_4d_causal_attention_mask as _MPLUG_OWL2_TRANSITIVE_IMPORT,
)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<|image|>"
from icecream import ic


class MPLUGOwl2MetaModel:
    def __init__(self, config):
        super(MPLUGOwl2MetaModel, self).__init__(config)
        self.vision_model = MplugOwlVisionModel(
            MplugOwlVisionConfig(**config.visual_config["visual_model"])
        )
        self.visual_abstractor = MplugOwlVisualAbstractorModel(
            MplugOwlVisualAbstractorConfig(**config.visual_config["visual_abstractor"]),
            config.hidden_size,
        )

    def get_vision_tower(self):
        vision_model = getattr(self, "vision_model", None)
        if type(vision_model) is list:
            vision_model = vision_model[0]
        return vision_model

    def get_visual_abstractor(self):
        visual_abstractor = getattr(self, "visual_abstractor", None)
        if type(visual_abstractor) is list:
            visual_abstractor = visual_abstractor[0]
        return visual_abstractor


class MPLUGOwl2MetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def encode_images(self, images):
        image_features = self.get_model().vision_model(images).last_hidden_state
        image_features = (
            self.get_model()
            .visual_abstractor(encoder_hidden_states=image_features)
            .last_hidden_state
        )
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images
    ):
        if images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            multiway_indices = torch.zeros_like(input_ids).long().to(self.device)
            return (
                input_ids,
                multiway_indices,
                attention_mask,
                past_key_values,
                None,
                labels,
            )

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_modality_indicators = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(
                    cur_input_ids[:half_len]
                )
                cur_input_embeds_2 = self.get_model().embed_tokens(
                    cur_input_ids[half_len:]
                )
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2],
                    dim=0,
                )
                new_input_embeds.append(cur_input_embeds)

                cur_modality_indicators = (
                    torch.zeros(len(cur_input_embeds)).long().to(self.device)
                )
                new_modality_indicators.append(cur_modality_indicators)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            cur_modality_indicators = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                cur_new_input_embeds.append(
                    self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                )
                cur_new_input_embeds.append(cur_image_features)

                # Add modality indicator
                assert image_token_start == len(cur_input_ids[:image_token_start])
                cur_modality_indicators.append(
                    torch.zeros(len(cur_input_ids[:image_token_start])).long()
                )
                cur_modality_indicators.append(
                    torch.ones(len(cur_image_features)).long()
                )

                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=labels.device,
                            dtype=labels.dtype,
                        )
                    )
                    cur_labels = cur_labels[image_token_start + 1 :]
                cur_image_idx += 1
                cur_input_ids = cur_input_ids[image_token_start + 1 :]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(
                    self.get_model().embed_tokens(cur_input_ids)
                )
                cur_modality_indicators.append(torch.zeros(len(cur_input_ids)).long())
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [
                x.to(device=self.device) for x in cur_new_input_embeds
            ]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)

            # Modality
            cur_modality_indicators = [
                x.to(device=self.device) for x in cur_modality_indicators
            ]
            cur_modality_indicators = torch.cat(cur_modality_indicators, dim=0)
            new_modality_indicators.append(cur_modality_indicators)

            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            # Embedding
            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            # Modality
            new_modality_indicators_align = []
            for cur_modality_indicator in new_modality_indicators:
                cur_new_embed = torch.cat(
                    (
                        cur_modality_indicator,
                        torch.zeros(
                            max_len - cur_modality_indicator.shape[0],
                            dtype=cur_modality_indicator.dtype,
                            device=cur_modality_indicator.device,
                        ),
                    ),
                    dim=0,
                )
                new_modality_indicators_align.append(cur_new_embed)
            new_modality_indicators = torch.stack(new_modality_indicators_align, dim=0)

            # Label
            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (
                            cur_new_label,
                            torch.full(
                                (max_len - cur_new_label.shape[0],),
                                IGNORE_INDEX,
                                dtype=cur_new_label.dtype,
                                device=cur_new_label.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            # Attention Mask
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                    attention_mask, _new_labels, new_labels
                ):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],),
                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    cur_new_attention_mask = torch.cat(
                        (
                            new_attn_mask_pad_left,
                            cur_attention_mask,
                            new_attn_mask_pad_right,
                        ),
                        dim=0,
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            new_modality_indicators = torch.stack(new_modality_indicators, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (
                        attention_mask.shape[0],
                        new_input_embeds.shape[1] - input_ids.shape[1],
                    ),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    (new_attn_mask_pad_left, attention_mask), dim=1
                )
                assert attention_mask.shape == new_input_embeds.shape[:2]
        return (
            None,
            new_modality_indicators,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
            image_features
        )


class MPLUGOwl2LlamaModel(MPLUGOwl2MetaModel, LlamaModel):
    config_class = MPLUGOwl2Config

    def __init__(self, config: MPLUGOwl2Config):
        super(MPLUGOwl2LlamaModel, self).__init__(config)


class MPLUGOwl2LlamaForCausalLM(LlamaForCausalLM, MPLUGOwl2MetaForCausalLM):
    config_class = MPLUGOwl2Config

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MPLUGOwl2LlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # MLP regression head and IQA score tokens.
        # `deepmlp_hidden_dims` is read from config so the architecture is
        # persisted across save/load. Input dim ties to config.hidden_size to
        # avoid hard-coding 4096.
        deepmlp_hidden_dims = getattr(config, "deepmlp_hidden_dims", None) or [2048, 1024]
        self.deepmlp = DeepMLP(config.hidden_size, deepmlp_hidden_dims, 1)
        # Persist score_token_id from config so the merged checkpoint loaded
        # via `from_pretrained` already knows which vocab ids are score tokens.
        # Fixes the previous typo `score_tokenid` that left `score_token_id`
        # undefined and broke `forward_mlp` after a plain reload.
        self.score_token_id = list(getattr(config, "score_token_id", None) or [])

        # Initialize weights and apply final processing
        self.deepmlp._initialize_weights()
        self.post_init()


    def get_model(self):
        return self.model

    def forward(self, input_type=None, **kwargs):
        if input_type is None:
            return self.forward_mlp(**kwargs)
        elif input_type == "single":
            kwargs_desp = self.get_subitem(kwargs, task_type="description")
            kwargs_score = self.get_subitem(kwargs, task_type="score")
            loss_desp = 0
            if len(kwargs_desp["task_types"]) > 0:
                del kwargs_desp["task_types"]
                output_desp = self.forward_mlp(**kwargs_desp)
                loss_desp = output_desp.loss
            loss_score = 0
            if len(kwargs_score["task_types"]) > 0:
                del kwargs_score["task_types"]
                output_score = self.forward_mlp(**kwargs_score)
                loss_score = output_score.loss
            if dist.get_rank() == 0:
                loss_desp_item = loss_desp if type(loss_desp) == int else loss_desp.item()
                loss_score_item = loss_score if type(loss_score) == int else loss_score.item()
                print(
                    f"[loss (w/o weight) | "
                    f"description loss: {round(loss_desp_item, 6)}, "
                    f"score loss: {round(loss_score_item, 6)}]"
                )
            loss = self.config.weight_desp * loss_desp + self.config.weight_next_token * loss_score
            return CausalLMOutputWithPast(loss=loss)
        else:
            raise ValueError

    def forward_mlp(
            self,
            input_ids: torch.LongTensor = None,
            # modality_indicators: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            gt_scores: Optional[torch.FloatTensor] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        (
            input_ids,
            modality_indicators,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            image_features
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            modality_indicators=modality_indicators,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_embedding = hidden_states[..., :-1, :].contiguous()

            # Vectorized score loss.
            # The previous implementation ran two nested Python loops over
            # batch and sequence positions and called `self.deepmlp` once per
            # matching position (and used `i in self.score_token_id` on a
            # Python list, which is O(K) per position). For typical sequences
            # this is fine but it dominates step time when there are many
            # score-token positions. The mask-based version below is
            # numerically identical to the original: it computes the same
            # sum-of-squared-errors over all positions whose `shift_labels`
            # match any score token id, then divides by the batch size.
            if self.score_token_id:
                score_id_tensor = torch.as_tensor(
                    self.score_token_id,
                    device=shift_labels.device,
                    dtype=shift_labels.dtype,
                )
                mask = torch.isin(shift_labels, score_id_tensor)  # [B, T-1]
            else:
                mask = torch.zeros_like(shift_labels, dtype=torch.bool)

            if mask.any():
                selected_embeddings = shift_embedding[mask]                # [N, D]
                selected_scores = self.deepmlp(selected_embeddings)        # [N]
                selected_scores = torch.clamp(selected_scores, min=0.0, max=5.0)
                batch_idx = (
                    torch.arange(shift_labels.size(0), device=shift_labels.device)
                    .unsqueeze(1)
                    .expand_as(shift_labels)
                )
                selected_gt = gt_scores[batch_idx[mask]]                   # [N]
                loss_score = ((selected_scores - selected_gt) ** 2).sum()
            else:
                loss_score = torch.zeros((), device=shift_labels.device)

            loss_score = loss_score / shift_labels.size(0)

            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            # Per-batch loss logging is handled in `forward(input_type='single')`
            # at rank 0 only; the earlier raw `print` calls here fired on every
            # rank and every micro-batch, flooding training logs.
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if loss is not None and loss_score is not None:
            loss = loss + loss_score

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )

    def pad_and_concat(self,
                       tensor_a: torch.Tensor,
                       tensor_b: torch.Tensor,
                       pad_value: int) -> torch.Tensor:

        assert tensor_a.dim() == 2 and tensor_b.dim() == 2
        max_len = max(tensor_a.shape[1], tensor_b.shape[1])

        def pad(tensor):
            pad_len = max_len - tensor.shape[1]
            if pad_len > 0:
                return F.pad(tensor, (0, pad_len), value=pad_value)
            else:
                return tensor

        tensor_a = pad(tensor_a)
        tensor_b = pad(tensor_b)
        return torch.cat([tensor_a, tensor_b], dim=0)

    def get_subitem(self, item, task_type):
        for key in list(item.keys()):
            if item[key] is None:
                del item[key]

        subitem = {}
        for key in item:
            subitem[key] = []
        for idx in range(len(item["task_types"])):
            if item["task_types"][idx] == task_type:
                for key in item:
                    subitem[key].append(item[key][idx])

        batch_size = torch.tensor(len(subitem["task_types"])).cuda()
        world_size = dist.get_world_size()
        batch_size_allrank = [torch.tensor(0).cuda() for _ in range(world_size)]
        dist.barrier()
        dist.all_gather(batch_size_allrank, batch_size)
        batch_size_max = torch.stack(batch_size_allrank, dim=0).max().item()
        batch_size_min = torch.stack(batch_size_allrank, dim=0).min().item()

        for key in item:
            subitem[key] = extend_list(subitem[key], batch_size_max, batch_size_min)
            if torch.is_tensor(item[key]) and len(subitem[key]):
                subitem[key] = torch.stack(subitem[key], dim=0)
        return subitem

    # ------------------------------------------------------------------
    # One-line inference API
    # ------------------------------------------------------------------
    # `score()` mirrors the DeQA-Score style HF AutoModel UX:
    #
    #     model = AutoModelForCausalLM.from_pretrained(
    #         "user/qscorer-merged",
    #         trust_remote_code=True,
    #         attn_implementation="eager",
    #         torch_dtype=torch.float16,
    #         device_map="auto",
    #     )
    #     model.score([Image.open("img.jpg")])
    #
    # It lazily loads the tokenizer and image processor from the same
    # directory the model was loaded from, so the caller does not need to
    # install this repo. The prompt template, the special score tokens, and
    # the two-step decode loop match `scorer.py` / `src/evaluate/iqa_eval.py`.
    def _maybe_init_score_runtime(self, tokenizer=None, image_processor=None):
        if getattr(self, "_score_tokenizer", None) is None:
            if tokenizer is not None:
                self._score_tokenizer = tokenizer
            else:
                from transformers import AutoTokenizer
                self._score_tokenizer = AutoTokenizer.from_pretrained(
                    self.config._name_or_path, use_fast=False
                )
        if getattr(self, "_score_image_processor", None) is None:
            if image_processor is not None:
                self._score_image_processor = image_processor
            else:
                from transformers.models.clip.image_processing_clip import (
                    CLIPImageProcessor,
                )
                self._score_image_processor = CLIPImageProcessor.from_pretrained(
                    self.config._name_or_path
                )

    @staticmethod
    def _expand2square(pil_img, background_color):
        from PIL import Image as _PILImage
        width, height = pil_img.size
        if width == height:
            return pil_img
        if width > height:
            result = _PILImage.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        result = _PILImage.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

    @staticmethod
    def _tokenizer_image_token(prompt, tokenizer, image_token_index):
        chunks = [
            tokenizer(chunk).input_ids if len(chunk) > 0 else []
            for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)
        ]
        input_ids = []
        offset = 0
        if len(chunks) > 0 and len(chunks[0]) > 0 and chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(chunks[0][0])
        sep = [image_token_index] * (offset + 1)
        # Interleave chunks with the image-token separator.
        flat = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                flat.extend(sep)
            flat.extend(chunk[offset:] if i > 0 else chunk)
        input_ids = flat if input_ids == [] else input_ids + flat[1:]
        return torch.tensor(input_ids, dtype=torch.long)

    @torch.inference_mode()
    def score(self, images, tokenizer=None, image_processor=None):
        """Score image quality for a list of PIL images.

        Args:
            images: list of PIL.Image objects (one or more).
            tokenizer: optional. If omitted, loaded from the model directory.
            image_processor: optional. If omitted, loaded from the model directory.

        Returns:
            list[float] of quality scores in the trained scale (typically [0, 5]).
        """
        if not isinstance(images, (list, tuple)):
            images = [images]
        self._maybe_init_score_runtime(tokenizer, image_processor)
        tokenizer = self._score_tokenizer
        image_processor = self._score_image_processor

        prompt = (
            "USER: How would you rate the quality of this image?\n"
            f"{DEFAULT_IMAGE_TOKEN}\n"
            "ASSISTANT: The quality of the image is"
        )
        input_ids = self._tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX
        ).unsqueeze(0).to(self.device)

        bg = tuple(int(x * 255) for x in image_processor.image_mean)
        pil_imgs = [self._expand2square(img.convert("RGB"), bg) for img in images]
        pixel_values = image_processor.preprocess(pil_imgs, return_tensors="pt")[
            "pixel_values"
        ].to(device=self.device, dtype=self.dtype)

        current_input = input_ids.repeat(pixel_values.shape[0], 1)
        embedding = None
        # Two-step decode loop matches `scorer.py`: first step consumes the
        # prompt, second step exposes the hidden state immediately after
        # "The quality of the image is", which is what the regression head
        # was trained on.
        for step in range(2):
            output = self(
                input_ids=current_input,
                images=pixel_values,
                output_hidden_states=False,
            )
            logits = output["logits"][:, -1]
            if step == 1:
                # `output.hidden_states` here is the last-layer hidden state
                # (see forward_mlp), not the all-layers tuple.
                embedding = output["hidden_states"][:, -1, :]
            next_token = torch.argmax(logits, dim=-1)
            current_input = torch.cat([current_input, next_token.unsqueeze(1)], dim=1)

        scores = self.deepmlp(embedding)
        scores = torch.clamp(scores, min=0.0, max=5.0)
        return scores.float().cpu().reshape(-1).tolist()


# Register lazily so importing this module twice (e.g. once as part of the
# `src.model` package and once as a trust_remote_code dynamic module) does not
# raise "config already registered".
try:
    AutoConfig.register("mplug_owl2", MPLUGOwl2Config)
except (ValueError, KeyError):
    pass
try:
    AutoModelForCausalLM.register(MPLUGOwl2Config, MPLUGOwl2LlamaForCausalLM)
except (ValueError, KeyError):
    pass

replace_llama_modality_adaptive()
