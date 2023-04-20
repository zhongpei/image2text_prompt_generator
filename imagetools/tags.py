from typing import List, Dict, Tuple

from utils.translate import tags_en2zh as translate_en2zh
import os
from utils.image2text import clip_image2text, w14_image2text
import random
import PIL.Image


def get_tag_from_file(fn: str) -> List[str]:
    if not os.path.exists(fn):
        return []
    with open(fn, "r") as f:
        data = f.read().split(",")
        tags = [d.strip().replace("_", " ") for d in data if len(d.strip()) > 0]
    print(f'Found {len(tags)} tags in {fn}')
    return tags


def get_tag_files(input_dir: str) -> Dict[str, List[str]]:
    tag_files = {}
    for root, dirs, files in os.walk(input_dir, topdown=False):
        for name in files:
            lower_name = name.lower()
            if lower_name.endswith(".txt"):
                print('Processing', os.path.join(root, name))
                tag_files.update(
                    {os.path.join(root, name): get_tag_from_file(os.path.join(root, name))}
                )
    return tag_files


def get_image_tags_fns(input_dir: str) -> List[Tuple[str, str]]:
    image_files = []
    image_tags_files = []
    for root, dirs, files in os.walk(input_dir, topdown=False):
        for name in files:
            lower_name = name.lower()
            if lower_name.endswith(".jpeg") or lower_name.endswith(".jpg") or lower_name.endswith(".png"):
                image_files.append(os.path.join(root, name))

    for image_fn in image_files:
        image_tags_files.append(
            (
                image_fn,
                os.path.splitext(image_fn)[0] + '.txt'
            )
        )

    return image_tags_files


def insert_tag2file(new_tags: str, fn: str, tags_pos: str):
    tags = get_tag_from_file(fn)

    new_tags = [t.strip().replace("_", " ") for t in new_tags.split(",") if len(t.strip()) > 0]
    if tags_pos == "top":
        tags = new_tags + tags
    elif tags_pos == "bottom":
        tags = tags + new_tags
    elif tags_pos == "center":
        tags = tags[:len(tags) // 2] + new_tags + tags[len(tags) // 2:]
    elif tags_pos == "random":
        rlen = random.randint(0, len(tags))
        tags = tags[:rlen] + new_tags + tags[rlen:]
    elif tags_pos == "cover":
        tags = new_tags
    else:
        raise ValueError(f"Unknown mode {tags_pos}")
    tags = [t.strip().replace("_", " ") for t in tags if len(t.strip()) > 0]
    with open(fn, "w+") as f:
        f.write(",".join(tags))


def gen_wd14_tags_files(
        input_dir: str,
        tags_pos: str,
        model_name: str,
        general_threshold: float,
        character_threshold: float,
        w14_tags_max_count: int,
) -> str:
    tag_files = get_image_tags_fns(input_dir)
    output = []
    for image_fn, tags_fn in tag_files:
        with PIL.Image.open(image_fn) as image:
            tags, c, d, _, _, _ = w14_image2text(
                image=image,
                model_name=model_name,
                general_threshold=general_threshold,
                character_threshold=character_threshold
            )

            tags = ",".join(tags.split(",")[:w14_tags_max_count])
            print(f'{image_fn} -> {tags}')
            insert_tag2file(new_tags=tags, tags_pos=tags_pos, fn=tags_fn)
            output.append(tags_fn)
    return "\n".join(output)


def gen_clip_tags_files(input_dir: str, tags_pos: str, clip_mode_type: str, clip_model_name: str) -> str:
    tag_files = get_image_tags_fns(input_dir)
    output = []
    for image_fn, tags_fn in tag_files:
        with PIL.Image.open(image_fn) as image:
            tags = clip_image2text(image=image, mode_type=clip_mode_type, model_name=clip_model_name)
            insert_tag2file(new_tags=tags, tags_pos=tags_pos, fn=tags_fn)
            output.append(tags_fn)
    return "\n".join(output)


def load_translated_tags(input_dir: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        return [], []

    tag_files = get_tag_files(input_dir)
    tags_count = {}

    for _, v in tag_files.items():
        for t in v:
            tags_count.setdefault(t, 0)
            tags_count[t] += 1

    tags_list = sorted(tags_count.keys())
    tags_map = translate_tags(
        sorted(tags_count.keys())
    )
    return [(t, tags_count[t]) for t in tags_list], [(tags_map[t], tags_count[t]) for t in tags_list]


def translate_tags(tags: List[str]) -> Dict[str, str]:
    """Translate a tag to a human readable string."""
    tags_zh = [translate_en2zh(tag) for tag in tags]
    return dict(zip(tags, tags_zh))
