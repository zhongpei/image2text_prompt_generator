import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from .singleton import Singleton
import re
import threading
from config import settings

device = "cuda" if torch.cuda.is_available() else "cpu"

if settings.translate.device == "cpu":
    device = "cpu"


@Singleton
class Models(object):

    def __getattr__(self, item):
        if item in self.__dict__:
            return getattr(self, item)

        if item in ('zh2en_model', 'zh2en_tokenizer',):
            self.zh2en_model, self.zh2en_tokenizer = self.load_model(settings.translate.zh2en_model)

        if item in ('en2zh_model', 'en2zh_tokenizer',):
            self.en2zh_model, self.en2zh_tokenizer = self.load_model(settings.translate.en2zh_model)

        return getattr(self, item)

    @classmethod
    def load_model(cls, model_name):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            resume_download=True,
            local_files_only=settings.translate.local_files_only,
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            resume_download=True,
            local_files_only=settings.translate.local_files_only,
        )
        return model, tokenizer


models = Models.instance()


@Singleton
class TranslateLocalCache(object):
    def __init__(self):

        if os.path.exists(settings.translate.cache_dir) and os.path.isdir(settings.translate.cache_dir):
            self.cache = self.load_cache(settings.translate.cache_dir)
        else:
            self.cache = {}

    @classmethod
    def get_cache_fns(cls, cache_dir):
        fns = []
        for root, dirs, files in os.walk(cache_dir, topdown=False):
            for name in files:
                lower_name = name.lower()
                if lower_name.endswith(".txt"):
                    fns.append(os.path.join(root, name))
        return fns

    @classmethod
    def load_cache(cls, cache_dir):
        cache = {}

        for cache_file in cls.get_cache_fns(cache_dir):
            with open(cache_file, "r", encoding='utf8') as f:
                lines = f.readlines()
                print(f"Loading cache from {cache_file} count: {len(lines)}")
                for line in lines:
                    if len(line.split("=")) != 2:
                        continue
                    key, value = line.split("=")
                    cache[key] = value
        return cache

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            key = args[0]
            if key in self.cache:
                return self.cache[key]
            else:
                result = func(*args, **kwargs)
                with threading.Lock():
                    self.cache[key] = result
                    return result

        return wrapper


translate_cache = TranslateLocalCache.instance()


def fix_text(text: str) -> str:
    text = text.strip()
    text_lines = [t.strip() for t in text.split("\n") if len(t.strip()) > 0]

    return "".join(text_lines)


@torch.no_grad()
def _zh2en(text: str, max_new_tokens: int = 512) -> str:
    text = fix_text(text)
    with torch.no_grad():
        encoded = models.zh2en_tokenizer([text], return_tensors="pt").to(models.zh2en_model.device)
        sequences = models.zh2en_model.generate(**encoded, max_new_tokens=max_new_tokens)
        return models.zh2en_tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]


@torch.no_grad()
def _en2zh(text: str, max_new_tokens: int = 512) -> str:
    text = fix_text(text)
    with torch.no_grad():
        encoded = models.en2zh_tokenizer([text], return_tensors="pt").to(models.zh2en_model.device)
        sequences = models.en2zh_model.generate(**encoded, max_new_tokens=max_new_tokens)
        return models.en2zh_tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]


def tags_en2zh(text: str, max_new_tokens: int = 512) -> str:
    output = translate_cache(_en2zh)(text, max_new_tokens)
    output = output.strip(",")
    output = re.sub(r",,+", ",", output)
    output = ",".join(list(set(output.strip(","))))
    return output


def en2zh(text: str, max_new_tokens: int = 512) -> str:
    return translate_cache(_en2zh)(text, max_new_tokens)


def zh2en(text: str, max_new_tokens: int = 512) -> str:
    return translate_cache(_zh2en)(text, max_new_tokens)


if __name__ == "__main__":
    input_text = "飞流直下三千尺，疑是银河落九天"
    en = zh2en(input_text)
    print(input_text, en)
    zh = en2zh(en)
    print(en, zh)
