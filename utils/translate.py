from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from .singleton import Singleton

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

        return getattr(self, item)

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


def zh2en(text):
    with torch.no_grad():
        encoded = models.zh2en_tokenizer([text], return_tensors="pt")
        sequences = models.zh2en_model.generate(**encoded)
        return models.zh2en_tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]


def en2zh(text):
    with torch.no_grad():
        encoded = models.en2zh_tokenizer([text], return_tensors="pt")
        sequences = models.en2zh_model.generate(**encoded)
        return models.en2zh_tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]


if __name__ == "__main__":
    input = "青春不能回头，所以青春没有终点。 ——《火影忍者》"
    en = zh2en(input)
    print(input, en)
    zh = en2zh(en)
    print(en, zh)
