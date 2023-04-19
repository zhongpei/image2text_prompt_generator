from typing import List, Dict, Tuple

from utils.translate import en2zh as translate_en2zh
import os


def get_tag_from_file(fn: str) -> List[str]:
    with open(fn, "r") as f:
        data = f.read().split(",")
        tags = [d.strip().replace("_", " ") for d in data if len(d.strip()) > 0]
    return tags


def get_tag_files(input_dir: str) -> Dict[str, List[str]]:
    tag_files = {}
    for root, dirs, files in os.walk(input_dir, topdown=False):
        for name in files:
            lower_name = name.lower()
            if lower_name.endswith(".txt"):
                tag_files.update(
                    {os.path.join(root, name): get_tag_from_file(os.path.join(root, name))}
                )
    return tag_files


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
