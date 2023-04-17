import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPTNeoForCausalLM, GPT2Tokenizer
from transformers import set_seed

from config import settings
from .singleton import Singleton

device = "cuda" if torch.cuda.is_available() else "cpu"
device_id = 0 if torch.cuda.is_available() else -1
if settings.generator.device == "cpu":
    device = "cpu"
    device_id = -1


@Singleton
class Models(object):

    def __getattr__(self, item):
        if item in self.__dict__:
            return getattr(self, item)

        if item in ('microsoft_model', 'microsoft_tokenizer'):
            self.microsoft_model, self.microsoft_tokenizer = self.load_microsoft_model()

        if item in ('mj_model', 'mj_tokenizer'):
            self.mj_model, mj_tokenizer = self.load_model(
                model_name=settings.generator.mj_model,
                tokenizer_class=GPT2Tokenizer,
                model_class=GPT2LMHeadModel,
            )

        if item in ('gpt2_650k_model', 'gpt2_650k_tokenizer'):
            self.gpt2_650k_model, self.gpt2_650k_tokenizer = self.load_model(
                model_name=settings.generator.gpt2_650k_model,
                tokenizer_class=GPT2Tokenizer,
                model_class=GPT2LMHeadModel,
            )

        if item in ('gpt_neo_125m_model', 'gpt_neo_125m_tokenizer'):
            self.gpt_neo_125m_model, self.gpt_neo_125m_tokenizer = self.load_model(
                model_name=settings.generator.gpt2_650k_model,
                tokenizer_class=GPT2Tokenizer,
                model_class=GPTNeoForCausalLM,
            )
        return getattr(self, item)

    @classmethod
    def load_model(cls, model_name, tokenizer_class, model_class):
        tokenizer = tokenizer_class.from_pretrained(
            model_name,
            trust_remote_code=True,
            resume_download=True,
            local_files_only=settings.generator.local_files_only,
        )
        model = model_class.from_pretrained(
            model_name,
            trust_remote_code=True,
            resume_download=True,
            local_files_only=settings.generator.local_files_only,
            pad_token_id=tokenizer.eos_token_id
        ).to(device).eval()
        return model, tokenizer

    @classmethod
    def load_microsoft_model(cls):
        prompter_model = AutoModelForCausalLM.from_pretrained("microsoft/Promptist").eval()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return prompter_model, tokenizer


models = Models.instance()


def rand_length(min_length: int = 60, max_length: int = 90) -> int:
    if min_length > max_length:
        return max_length

    return random.randint(min_length, max_length)


@torch.no_grad()
def _generate_prompt(
        plain_text: str,
        min_length: int = 60,
        max_length: int = 90,
        num_beams: int = 8,
        num_return_sequences: int = 8,
        length_penalty: int = -1.0,
        model_name: str = 'microsoft',
) -> list:
    seed = random.randint(100, 1000000)
    set_seed(seed)
    model = getattr(models, f"{model_name}_model")
    tokenizer = getattr(models, f"{model_name}_tokenizer")
    if model is None or tokenizer is None:
        return []

    input_ids = tokenizer(
        plain_text.strip() + " Rephrase:",
        return_tensors="pt",
    ).input_ids

    eos_id = model.eos_token_id

    outputs = model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=rand_length(min_length, max_length),
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
        length_penalty=length_penalty
    )
    output_texts = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
    )
    result = []
    for output_text in output_texts:
        result.append(output_text.replace(plain_text + " Rephrase:", "").strip())

    return result


def get_valid_prompt(text: str) -> str:
    dot_split = text.split('.')[0]
    n_split = text.split('\n')[0]

    return {
        len(dot_split) < len(n_split): dot_split,
        len(n_split) > len(dot_split): n_split,
        len(n_split) == len(dot_split): dot_split
    }[True]


@torch.no_grad()
def generate_prompt(
        plain_text: str,
        min_length: int = 60,
        max_length: int = 90,
        num_beams: int = 8,
        num_return_sequences: int = 8,
        length_penalty: int = -1.0,
        model_name: str = 'microsoft',
):
    output = []
    for i in range(6):
        result_list = _generate_prompt(
            plain_text,
            min_length,
            max_length,
            num_beams,
            num_return_sequences,
            length_penalty,
            model_name
        )
        output += [
            get_valid_prompt(result['generated_text']) for result in result_list
        ]
        output = list(set(output))
        if len(output) >= num_return_sequences:
            break

    if model_name == 'mj':
        output = re.sub('[^ ]+\.[^ ]+', '', output)
        output = output.replace('<', '').replace('>', '')

    return "\n".join(output)
