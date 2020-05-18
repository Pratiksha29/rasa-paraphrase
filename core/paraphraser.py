# @credit: https://forum.rasa.com/t/paraphrasing-for-nlu-data-augmentation-experimental/27744?utm_source=linkedin
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import re
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, OpenAIGPTConfig

# NOTE: type checking is not fully supported: https://github.com/pytorch/pytorch/issues/16574


# Constants
MAX_LENGTH: int = int(10000)  # Hardcoded max length to avoid infinite loop
ALL_MODELS: tuple = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (GPT2Config, OpenAIGPTConfig)
    ),
    (),
)
MODEL_CLASSES: dict = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
}
PADDING_TEXT: str = ""

logger: logging.Logger = logging.getLogger()


@dataclass
class ModelParams:
    model_type: str
    model_name_or_path: str
    prompt: str
    padding_text: str
    xlm_lang: str
    length: int
    num_samples: int
    temperature: float
    repetition_penalty: float
    top_k: int
    top_p: float
    no_cuda: bool
    seed: int
    stop_token: str

    device: Any = "cpu"
    n_gpu: int = 0


def _gpu(params: ModelParams):
    params.device = torch.device(
        "cuda" if torch.cuda.is_available() and not params.no_cuda else "cpu"
    )
    params.n_gpu = torch.cuda.device_count()


def set_seed(params: ModelParams):
    _gpu(params)
    params.model_type = params.model_type.lower()
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    if params.n_gpu > 0:
        torch.cuda.manual_seed_all(params.seed)


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = -float("Inf"),
    stop_token_id: list = [],
):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """

    if stop_token_id:
        logits[:, stop_token_id] = filter_value
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(
    model,
    length: int,
    context: list,
    num_samples: int = 1,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    repetition_penalty: float = 1.0,
    is_xlnet: bool = False,
    is_xlm_mlm: bool = False,
    xlm_mask_token: str = None,
    xlm_lang: str = None,
    device: str = "cpu",
    stop_token_ids: list = [],
) -> torch.Tensor:

    new_context = torch.tensor(context, dtype=torch.long, device=device)
    new_context = new_context.unsqueeze(0).repeat(num_samples, 1)
    generated: torch.Tensor = new_context

    with torch.no_grad():
        for _ in trange(length):

            inputs: dict = {"input_ids": generated}
            if is_xlnet:
                # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
                input_ids: torch.Tensor = torch.cat(
                    (generated, torch.zeros((1, 1), dtype=torch.long, device=device)),
                    dim=1,
                )
                perm_mask: torch.Tensor = torch.zeros(
                    (1, input_ids.shape[1], input_ids.shape[1]),
                    dtype=torch.float,
                    device=device,
                )
                perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                target_mapping: torch.Tensor = torch.zeros(
                    (1, 1, input_ids.shape[1]), dtype=torch.float, device=device
                )
                target_mapping[0, 0, -1] = 1.0  # predict last token
                inputs = {
                    "input_ids": input_ids,
                    "perm_mask": perm_mask,
                    "target_mapping": target_mapping,
                }

            if is_xlm_mlm and xlm_mask_token:
                # XLM MLM models are direct models (predict same token, not next token)
                # => need one additional dummy token in the input (will be masked and guessed)
                input_ids = torch.cat(
                    (
                        generated,
                        torch.full(
                            (1, 1), xlm_mask_token, dtype=torch.long, device=device
                        ),
                    ),
                    dim=1,
                )
                inputs = {"input_ids": input_ids}

            if xlm_lang is not None:
                inputs["langs"] = torch.tensor(
                    [xlm_lang] * inputs["input_ids"].shape[1], device=device
                ).view(1, -1)

            outputs: tuple = model(
                **inputs
            )  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / (
                temperature if temperature > 0 else 1.0
            )

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty

            filtered_logits: torch.Tensor = top_k_top_p_filtering(
                next_token_logits,
                top_k=top_k,
                top_p=top_p,
                stop_token_id=stop_token_ids,
            )

            if temperature == 0:  # greedy sampling:
                next_token: torch.Tensor = torch.argmax(
                    filtered_logits, dim=-1
                ).unsqueeze(-1)
            else:
                next_token = torch.multinomial(
                    F.softmax(filtered_logits, dim=-1), num_samples=1
                )
            generated = torch.cat((generated, next_token), dim=1)

    return generated


class ParaphraseModel:
    def __init__(self, model_path: str, params: ModelParams):
        self.params = params
        self.meta = { 'model_path': model_path }

    def load_model(self, model_path: str):
        params = self.params
        model_class, tokenizer_class = MODEL_CLASSES[params.model_type]
        self.tokenizer = tokenizer_class.from_pretrained(model_path)
        self.model = model_class.from_pretrained(model_path)
        # a = self.model.to(params.device)

        if params.length < 0 and self.model.config.max_position_embeddings > 0:
            params.length = self.model.config.max_position_embeddings
        elif 0 < self.model.config.max_position_embeddings < params.length:
            params.length = (
                self.model.config.max_position_embeddings
            )  # No generation bigger than model size
        elif params.length < 0:
            params.length = MAX_LENGTH  # avoid infinite loop

    def get_paraphrases(self, prompt: str, num_samples: int, stop_words: str) -> list:
        params: ModelParams = self.params

        prompt += " >>>>>>>>"
        prompt = prompt.lower()

        stop_words_list: List[str] = stop_words.split(";")
        stop_words_list = [word.strip() for word in stop_words_list]

        added_stop_words: list = ["Ġ" + word for word in stop_words_list if word]
        all_stop_words: list = stop_words_list + added_stop_words

        raw_text: str = prompt

        all_preds: list = []
        if params.model_type in ["transfo-xl", "xlnet"]:
            # Models with memory likes to have a long prompt for short inputs.
            raw_text = (
                params.padding_text if params.padding_text else PADDING_TEXT
            ) + raw_text
        context_tokens = self.tokenizer.encode(raw_text, add_special_tokens=False)

        stop_token_ids: list = []
        for word in all_stop_words:
            token_ids = self.tokenizer.convert_tokens_to_ids([word])

            if 50256 in token_ids:
                token_ids.remove(50256)
            stop_token_ids += token_ids

        if params.model_type == "ctrl":
            if not any(
                context_tokens[0] == x for x in self.tokenizer.control_codes.values()
            ):
                logger.info(
                    "WARNING! You are not starting your generation from a control code so you won't get good results"
                )
        out: torch.Tensor = sample_sequence(
            model=self.model,
            context=context_tokens,
            num_samples=num_samples,
            length=params.length,
            temperature=params.temperature,
            top_k=params.top_k,
            top_p=params.top_p,
            repetition_penalty=params.repetition_penalty,
            device=params.device,
            stop_token_ids=stop_token_ids,
        )

        out_slice: List[Any] = out[:, len(context_tokens) :].tolist()
        for o in out_slice:
            text: str = self.tokenizer.decode(o, clean_up_tokenization_spaces=True)
            text = text[: text.find(params.stop_token) if params.stop_token else None]
            all_preds.append(text.split("<|endoftext|>")[0].strip())

        return all_preds
