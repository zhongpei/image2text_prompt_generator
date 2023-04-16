from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from .singleton import Singleton
from transformers import (
    EncoderDecoderModel,
    AutoTokenizer
)

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    def load_wenyanwen2modern_model(cls):
        PRETRAINED = "raynardj/wenyanwen-ancient-translate-to-modern"
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)
        model = EncoderDecoderModel.from_pretrained(PRETRAINED)
        return tokenizer, model

    @classmethod
    def load_en2zh_model(cls):
        en2zh_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh").eval()
        en2zh_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
        return en2zh_model, en2zh_tokenizer

    @classmethod
    def load_zh2en_model(cls):
        zh2en_model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-zh-en').eval()
        zh2en_tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-zh-en')

        return zh2en_model, zh2en_tokenizer,


models = Models.instance()


def wenyanwen2modern(text: str) -> str:
    tk_kwargs = dict(
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors='pt')

    inputs = models.wenyanwen2modern_tokenizer([text, ], **tk_kwargs)
    with torch.no_grad():
        return models.wenyanwen2modern_tokenizer.batch_decode(
            models.wenyanwen2modern_model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                num_beams=3,
                max_length=256,
                bos_token_id=101,
                eos_token_id=models.wenyanwen2modern_tokenizer.sep_token_id,
                pad_token_id=models.wenyanwen2modern_tokenizer.pad_token_id,
            ), skip_special_tokens=True)[0].replace(" ", "")


def zh2en(text: str) -> str:
    with torch.no_grad():
        encoded = models.zh2en_tokenizer([text], return_tensors="pt")
        sequences = models.zh2en_model.generate(**encoded, max_new_tokens=512)
        return models.zh2en_tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]


def en2zh(text: str) -> str:
    with torch.no_grad():
        encoded = models.en2zh_tokenizer([text], return_tensors="pt")
        sequences = models.en2zh_model.generate(**encoded, max_new_tokens=512)
        return models.en2zh_tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]


if __name__ == "__main__":
    input = "飞流直下三千尺，疑是银河落九天"
    input_m = wenyanwen2modern(input)
    en = zh2en(input_m)
    print(input, en)
    zh = en2zh(en)
    print(en, zh)
