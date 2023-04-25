import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline, set_seed
import random
import re
from .singleton import Singleton
from config import settings
from .models import ModelsBase

device = "cuda" if torch.cuda.is_available() else "cpu"
device_id = 0 if torch.cuda.is_available() else -1
if settings.generator.device == "cpu":
    device = "cpu"
    device_id = -1


@Singleton
class Models(ModelsBase):

    def __init__(self):
        super().__init__()

    def __getitem__(self, item):
        return super().__getitem__(item)

    def unload(self):
        return super().unload()

    def load(self, item):
        if item in ('microsoft_model', 'microsoft_tokenizer'):
            microsoft_model, microsoft_tokenizer = self.load_microsoft_model()
            self.register('microsoft_model', microsoft_model)
            self.register('microsoft_tokenizer', microsoft_tokenizer)

        if item in ('mj_pipe',):
            mj_pipe = self.load_mj_pipe()
            self.register('mj_pipe', mj_pipe)

        if item in ('gpt2_650k_pipe',):
            gpt2_650k_pipe = self.load_gpt2_650k_pipe()
            self.register('gpt2_650k_pipe', gpt2_650k_pipe)

        if item in ('gpt_neo_125m_pipe',):
            gpt_neo_125m_pipe = self.load_gpt_neo_125m()
            self.register('gpt_neo_125m_pipe', gpt_neo_125m_pipe)

    @classmethod
    def load_gpt_neo_125m(cls):
        return pipeline(
            'text-generation',
            model=settings.generator.gpt_neo_125m_model,
            tokenizer=settings.generator.gpt_neo_125m_model,
            device=device_id,
            trust_remote_code=True,

        )

    @classmethod
    def load_gpt2_650k_pipe(cls):
        return pipeline(
            'text-generation',
            model=settings.generator.gpt2_650k_model,
            device=device_id,
            trust_remote_code=True,

        )

    @classmethod
    def load_mj_pipe(cls):
        return pipeline(
            'text-generation',
            model=settings.generator.mj_model,
            device=device_id,
            trust_remote_code=True,

        )

    @classmethod
    def load_microsoft_model(cls):
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=settings.generator.microsoft_model,
            trust_remote_code=True,
            resume_download=True,
            local_files_only=settings.translate.local_files_only,
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            "gpt2",
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return model, tokenizer


models = Models.instance()


def rand_length(min_length: int = 60, max_length: int = 90) -> int:
    if min_length > max_length:
        return max_length

    return random.randint(min_length, max_length)


@torch.no_grad()
def _generate_prompt(
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
            num_return_sequences=1,
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
def generate_prompt(
        plain_text,
        min_length=60,
        max_length=90,
        num_return_sequences=8,
        model_name='microsoft',
):
    result = _generate_prompt(
        plain_text,
        min_length=min_length,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        model_name=model_name
    )
    if settings.generator.fix_sd_prompt:
        return [fix_sd_prompt(x) for x in result]
    return result


def fix_sd_prompt(text_input: str) -> str:
    text_input = re.sub(r",+", ",", text_input)
    text_input = re.sub(r"\.\.+", ".", text_input)
    text_input = re.sub(r"!!+", ".", text_input)
    text_input = re.sub(r"\|", ",", text_input)
    text_input = re.sub(r"<|>|\(|\)", "", text_input)
    return text_input


@torch.no_grad()
def generate_prompt_microsoft(
        plain_text,
        min_length=60,
        max_length=90,
        num_beams=8,
        num_return_sequences=8,
        length_penalty=-1.0
) -> list:
    input_ids = models.microsoft_tokenizer(
        plain_text.strip() + " Rephrase:",
        return_tensors="pt",
    ).to(device).input_ids

    eos_id = models.microsoft_tokenizer.eos_token_id

    outputs = models.microsoft_model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=rand_length(min_length, max_length),
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
        length_penalty=length_penalty,
    )
    output_texts = models.microsoft_tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,

    )
    result = []
    for output_text in output_texts:
        result.append(output_text.replace(plain_text + " Rephrase:", "").strip())
    result = list(set(result))

    if settings.generator.fix_sd_prompt:
        return [fix_sd_prompt(x) for x in result]

    return result


@torch.no_grad()
def generate_prompt_pipe(pipe, prompt: str, min_length=60, max_length: int = 255,
                         num_return_sequences: int = 8) -> list:
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
    return [o.strip() for o in output if len(o.strip()) > 0]


@torch.no_grad()
def generate_prompt_mj(text_in_english: str, num_return_sequences: int = 8, min_length=60, max_length=90) -> list:
    seed = random.randint(100, 1000000)
    set_seed(seed)

    result_list = []
    for _ in range(6):
        sequences = models.mj_pipe(
            text_in_english,
            max_new_tokens=rand_length(min_length, max_length),
            num_return_sequences=num_return_sequences
        )

        for sequence in sequences:
            line = sequence['generated_text'].strip()
            if line != text_in_english and len(line) > (len(text_in_english) + 4) and line.endswith(
                    (':', '-', '—')) is False:
                result_list.append(line)
        if len(result_list) >= num_return_sequences:
            break
    result_list = list(set(result_list))
    result_list = [re.sub(r'[^ ]+\.[^ ]+', '', r) for r in result_list if len(r) > 0]
    result_list = [r.replace('<', '').replace('>', '') for r in result_list if len(r) > 0]

    return result_list
    # return result, "\n".join(translate_en2zh(line) for line in result.split("\n") if len(line) > 0)
