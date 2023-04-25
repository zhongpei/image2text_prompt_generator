from __future__ import annotations

import PIL.Image
import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor

from . import dbimutils
from .singleton import Singleton
import os
import torch
from clip_interrogator import Config, Interrogator
from .models import ModelsBase
from config import settings

device = "cuda" if torch.cuda.is_available() else "cpu"


@Singleton
class Models(ModelsBase):
    # WD14 models
    SWIN_MODEL_REPO = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
    CONV_MODEL_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
    CONV2_MODEL_REPO = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
    VIT_MODEL_REPO = "SmilingWolf/wd-v1-4-vit-tagger-v2"

    MODEL_FILENAME = "model.onnx"
    LABEL_FILENAME = "selected_tags.csv"

    # CLIP models
    VIT_H_14_MODEL_REPO = "ViT-H-14/laion2b_s32b_b79k"  # Stable Diffusion 2.X
    VIT_L_14_MODEL_REPO = "ViT-L-14/openai"  # Stable Diffusion 1.X

    def __init__(self):
        super().__init__()

    @classmethod
    def load_clip_model(cls, model_repo):
        config = Config()
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config.blip_offload = False if torch.cuda.is_available() else True
        config.chunk_size = 2048
        config.flavor_intermediate_count = 512
        config.blip_num_beams = 64
        config.clip_model_name = model_repo

        ci = Interrogator(config)
        return ci

    def load(self, item: str) -> None:
        print(f"Loading {item}...")
        if item in ('clip_vit_h_14_model',):
            clip_vit_h_14_model = self.load_clip_model(self.VIT_H_14_MODEL_REPO)
            self.register('clip_vit_h_14_model', clip_vit_h_14_model)

        if item in ('clip_vit_l_14_model',):
            clip_vit_l_14_model = self.load_clip_model(self.VIT_L_14_MODEL_REPO)
            self.register('clip_vit_l_14_model', clip_vit_l_14_model)

        if item in ('swinv2_model',):
            swinv2_model = self.load_model(self.SWIN_MODEL_REPO, self.MODEL_FILENAME)
            self.register('swinv2_model', swinv2_model)
        if item in ('convnext_model',):
            convnext_model = self.load_model(self.CONV_MODEL_REPO, self.MODEL_FILENAME)
            self.register('convnext_model', convnext_model)
        if item in ('vit_model',):
            vit_model = self.load_model(self.VIT_MODEL_REPO, self.MODEL_FILENAME)
            self.register('vit_model', vit_model)
        if item in ('convnextv2_model',):
            convnextv2_model = self.load_model(self.CONV2_MODEL_REPO, self.MODEL_FILENAME)
            self.register('convnextv2_model', convnextv2_model)

        if item in ('git_model', 'git_processor'):
            git_model, git_processor = self.load_git_model()
            self.register('git_model', git_model)
            self.register('git_processor', git_processor)

        if item in ('tag_names', 'rating_indexes', 'general_indexes', 'character_indexes'):
            tag_names, rating_indexes, general_indexes, character_indexes = self.load_w14_labels()
            self.register('tag_names', tag_names)
            self.register('rating_indexes', rating_indexes)
            self.register('general_indexes', general_indexes)
            self.register('character_indexes', character_indexes)

    @classmethod
    def load_git_model(cls):

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=settings.git.model if settings.git.model else "microsoft/git-large-coco",


        )
        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=settings.git.model if settings.git.model else "microsoft/git-large-coco"
        )

        return model, processor

    @staticmethod
    def load_model(model_repo: str, model_filename: str) -> rt.InferenceSession:
        model_name = model_repo.split('/')[-1]

        local_dir = os.path.join(settings.wd14.model_dir, model_name)

        if os.path.exists(local_dir) and settings.wd14.local_files_only:
            path = os.path.join(local_dir, model_filename)
            model = rt.InferenceSession(path)
            return model

        path = huggingface_hub.hf_hub_download(
            model_repo,
            model_filename,
        )

        model = rt.InferenceSession(path)
        return model

    @classmethod
    def load_w14_labels(cls) -> list[str]:
        path = huggingface_hub.hf_hub_download(
            cls.CONV2_MODEL_REPO, cls.LABEL_FILENAME
        )
        df = pd.read_csv(path)

        tag_names = df["name"].tolist()
        rating_indexes = list(np.where(df["category"] == 9)[0])
        general_indexes = list(np.where(df["category"] == 0)[0])
        character_indexes = list(np.where(df["category"] == 4)[0])
        return [tag_names, rating_indexes, general_indexes, character_indexes]


models = Models.instance()


@torch.no_grad()
def clip_image2text(image: PIL.Image.Image, mode_type: str = 'best', model_name: str = 'vit_h_14') -> str:
    image = image.convert('RGB')
    model = getattr(models, f'clip_{model_name}_model')
    if mode_type == 'classic':
        prompt = model.interrogate_classic(image)
    elif mode_type == 'fast':
        prompt = model.interrogate_fast(image)
    elif mode_type == 'negative':
        prompt = model.interrogate_negative(image)
    else:
        prompt = model.interrogate(image)  # default to best
    if device == 'cuda':
        torch.cuda.empty_cache()
    return prompt


@torch.no_grad()
def git_image2text(input_image: PIL.Image.Image, max_length: int = 50) -> str:
    image = input_image.convert('RGB')
    pixel_values = models.git_processor(images=image, return_tensors="pt").to(device).pixel_values

    generated_ids = models.git_model.to(device).generate(pixel_values=pixel_values, max_length=max_length)
    generated_caption = models.git_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption


@torch.no_grad()
def w14_image2text(
        image: PIL.Image.Image,
        model_name: str,
        general_threshold: float,
        character_threshold: float,

):
    tag_names: list[str] = models.tag_names
    rating_indexes: list[np.int64] = models.rating_indexes
    general_indexes: list[np.int64] = models.general_indexes
    character_indexes: list[np.int64] = models.character_indexes
    model_name = "{}_model".format(model_name.lower())
    model = getattr(models, model_name)

    _, height, width, _ = model.get_inputs()[0].shape

    # Alpha to white
    image = image.convert("RGBA")
    new_image = PIL.Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    image = new_image.convert("RGB")
    image = np.asarray(image)

    # PIL RGB to OpenCV BGR
    image = image[:, :, ::-1]

    image = dbimutils.make_square(image, height)
    image = dbimutils.smart_resize(image, height)
    image = image.astype(np.float32)
    image = np.expand_dims(image, 0)

    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input_name: image})[0]

    labels = list(zip(tag_names, probs[0].astype(float)))

    # First 4 labels are actually ratings: pick one with argmax
    ratings_names = [labels[i] for i in rating_indexes]
    rating = dict(ratings_names)

    # Then we have general tags: pick any where prediction confidence > threshold
    general_names = [labels[i] for i in general_indexes]
    general_res = [x for x in general_names if x[1] > general_threshold]
    general_res = dict(general_res)

    # Everything else is characters: pick any where prediction confidence > threshold
    character_names = [labels[i] for i in character_indexes]
    character_res = [x for x in character_names if x[1] > character_threshold]
    character_res = dict(character_res)

    b = dict(sorted(general_res.items(), key=lambda item: item[1], reverse=True))
    a = (
        ", ".join(list(b.keys()))
        .replace("_", " ")
        .replace("(", "\(")
        .replace(")", "\)")
    )
    c = ", ".join(list(b.keys()))
    d = " ".join(list(b.keys()))

    return a, c, d, rating, character_res, general_res
