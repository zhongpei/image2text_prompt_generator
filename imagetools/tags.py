from typing import List, Dict

from utils.translate import en2zh as translate_en2zh
import os


def get_tag_files(input_dir):
    def get_tag_from_file(fn):
        with open(fn, "r") as f:
            data = f.read().split(",")
            tags = [d for d in data if d.strip() if len(d.strip()) > 0]
        return tags

    tag_files = {}
    for root, dirs, files in os.walk(input_dir, topdown=False):
        for name in files:
            lower_name = name.lower()
            if lower_name.endswith(".txt"):
                tag_files.update(
                    {os.path.join(root, name): get_tag_from_file(os.path.join(root, name))}
                )
    return tag_files


def load_translated_tags(input_dir):
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        return {}
    tag_files = get_tag_files(input_dir)
    tags = {}
    for _, v in tag_files.items():
        tags.update(translate_tags(v))
    return tags


def translate_tags(tags: List[str]) -> Dict[str, str]:
    """Translate a tag to a human readable string."""
    tags_zh = [translate_en2zh(tag) for tag in tags]
    return dict(zip(tags, tags_zh))
