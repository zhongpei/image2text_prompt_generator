from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from .singleton import Singleton

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
            self.zh2en_model, self.zh2en_tokenizer = self.load_zh2en_model()

        if item in ('en2zh_model', 'en2zh_tokenizer',):
            self.en2zh_model, self.en2zh_tokenizer = self.load_en2zh_model()

        if item in ('wenyanwen2modern_tokenizer', 'wenyanwen2modern_model',):
            self.wenyanwen2modern_tokenizer, self.wenyanwen2modern_model = self.load_wenyanwen2modern_model()

        return getattr(self, item)

    @classmethod
    def load_en2zh_model(cls):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            settings.translate.en2zh_model,
            trust_remote_code=True,
            resume_download=True,
            local_files_only=settings.translate.local_files_only,
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            settings.translate.en2zh_model,
            trust_remote_code=True,
            resume_download=True,
            local_files_only=settings.translate.local_files_only,
        )
        return model, tokenizer

    @classmethod
    def load_zh2en_model(cls):
        zh2en_model = AutoModelForSeq2SeqLM.from_pretrained(
            settings.translate.zh2en_model,
            trust_remote_code=True,
            resume_download=True,
            local_files_only=settings.translate.local_files_only,
        ).to(device).eval()
        zh2en_tokenizer = AutoTokenizer.from_pretrained(settings.translate.zh2en_model)

        return zh2en_model, zh2en_tokenizer,


models = Models.instance()


def fix_text(text: str) -> str:
    text = text.strip()
    text_lines = [t.strip() for t in text.split("\n") if len(t.strip()) > 0]

    return "".join(text_lines)


@torch.no_grad()
def zh2en(text: str, max_new_tokens: int = 512) -> str:
    text = fix_text(text)
    with torch.no_grad():
        encoded = models.zh2en_tokenizer([text], return_tensors="pt").to(models.zh2en_model.device)
        sequences = models.zh2en_model.generate(**encoded, max_new_tokens=max_new_tokens)
        return models.zh2en_tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]


@torch.no_grad()
def en2zh(text: str, max_new_tokens: int = 512) -> str:
    text = fix_text(text)
    with torch.no_grad():
        encoded = models.en2zh_tokenizer([text], return_tensors="pt").to(models.zh2en_model.device)
        sequences = models.en2zh_model.generate(**encoded, max_new_tokens=max_new_tokens)
        return models.en2zh_tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]


if __name__ == "__main__":
    input_text = "飞流直下三千尺，疑是银河落九天"
    en = zh2en(input_text)
    print(input_text, en)
    zh = en2zh(en)
    print(en, zh)
