import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline, set_seed
import random
import re
from .singleton import Singleton
from config import settings

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

        if item in ('mj_pipe',):
            self.mj_pipe = self.load_mj_pipe()

        if item in ('gpt2_650k_pipe',):
            self.gpt2_650k_pipe = self.load_gpt2_650k_pipe()

        if item in ('gpt_neo_125m_pipe',):
            self.gpt_neo_125m_pipe = self.load_gpt_neo_125m()
        return getattr(self, item)

    @classmethod
    def load_gpt_neo_125m(cls):
        return pipeline(
            'text-generation',
            model='DrishtiSharma/StableDiffusion-Prompt-Generator-GPT-Neo-125M',
            device=device_id,
            trust_remote_code=True,

        )

    @classmethod
    def load_gpt2_650k_pipe(cls):
        return pipeline(
            'text-generation',
            model='Ar4ikov/gpt2-650k-stable-diffusion-prompt-generator',
            device=device_id,
            trust_remote_code=True,

        )

    @classmethod
    def load_mj_pipe(cls):
        return pipeline(
            'text-generation',
            model='succinctly/text2image-prompt-generator',
            device=device_id,
            trust_remote_code=True,

        )

    @classmethod
    def load_microsoft_model(cls):
        prompter_model = AutoModelForCausalLM.from_pretrained("microsoft/Promptist").eval()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return prompter_model, tokenizer


models = Models.instance()


def rand_length(min_length: int = 60, max_length: int = 90) -> int:
    seed = random.randint(100, 1000000)
    set_seed(seed)
    if min_length > max_length:
        return max_length

    return random.randint(min_length, max_length)


@torch.no_grad()
def generate_prompt(
        plain_text,
        min_length=60,
        max_length=90,
        num_return_sequences=8,
        model_name='microsoft',
):
    if model_name == 'gpt2_650k':
        return generate_prompt_pipe(
            models.gpt2_650k_pipe,
            prompt=plain_text,
            min_length=min_length,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
        )
    elif model_name == 'gpt_neo_125m':
        return generate_prompt_pipe(
            models.gpt_neo_125m_pipe,
            prompt=plain_text,
            min_length=min_length,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
        )
    elif model_name == 'mj':
        return generate_prompt_mj(
            text_in_english=plain_text,
            num_return_sequences=num_return_sequences,
            min_length=min_length,
            max_length=max_length,
        )
    else:
        return generate_prompt_microsoft(
            plain_text=plain_text,
            min_length=min_length,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            num_beams=num_return_sequences,
        )


@torch.no_grad()
def generate_prompt_microsoft(
        plain_text,
        min_length=60,
        max_length=90,
        num_beams=8,
        num_return_sequences=8,
        length_penalty=-1.0
) -> str:
    input_ids = models.microsoft_tokenizer(
        plain_text.strip() + " Rephrase:",
        return_tensors="pt",

    ).input_ids

    eos_id = models.microsoft_tokenizer.eos_token_id

    outputs = models.microsoft_model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=rand_length(min_length, max_length),
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
        length_penalty=length_penalty
    )
    output_texts = models.microsoft_tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,

    )
    result = []
    for output_text in output_texts:
        result.append(output_text.replace(plain_text + " Rephrase:", "").strip())

    return "\n".join(result)


@torch.no_grad()
def generate_prompt_pipe(pipe, prompt: str, min_length=60, max_length: int = 255, num_return_sequences: int = 8) -> str:
    def get_valid_prompt(text: str) -> str:
        dot_split = text.split('.')[0]
        n_split = text.split('\n')[0]

        return {
            len(dot_split) < len(n_split): dot_split,
            len(n_split) > len(dot_split): n_split,
            len(n_split) == len(dot_split): dot_split
        }[True]

    output = []
    for _ in range(6):

        output += [
            get_valid_prompt(result['generated_text']) for result in
            pipe(
                prompt,
                max_new_tokens=rand_length(min_length, max_length),
                num_return_sequences=num_return_sequences,

            )
        ]
        output = list(set(output))
        if len(output) >= num_return_sequences:
            break

    # valid_prompt = get_valid_prompt(models.gpt2_650k_pipe(prompt, max_length=max_length)[0]['generated_text'])
    return "\n".join([o.strip() for o in output])


@torch.no_grad()
def generate_prompt_mj(text_in_english: str, num_return_sequences: int = 8, min_length=60, max_length=90) -> str:
    seed = random.randint(100, 1000000)
    set_seed(seed)

    result = ""
    for _ in range(6):
        sequences = models.mj_pipe(
            text_in_english,
            max_new_tokens=rand_length(min_length, max_length),
            num_return_sequences=num_return_sequences
        )
        list = []
        for sequence in sequences:
            line = sequence['generated_text'].strip()
            if line != text_in_english and len(line) > (len(text_in_english) + 4) and line.endswith(
                    (':', '-', '—')) is False:
                list.append(line)

        result = "\n".join(list)
        result = re.sub('[^ ]+\.[^ ]+', '', result)
        result = result.replace('<', '').replace('>', '')
        if result != '':
            break
    return result
    # return result, "\n".join(translate_en2zh(line) for line in result.split("\n") if len(line) > 0)
