from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import os
import subprocess


def clone():
    subprocess.run(["git", "clone", "https://github.com/wfjsw/danbooru-diffusion-prompt-builder"])


def get_files(tags_dir):
    fns = []
    for root, dirs, files in os.walk(tags_dir, topdown=False):
        for name in files:
            lower_name = name.lower()
            if lower_name.endswith(".yaml"):
                fns.append(os.path.join(root, name))
    return fns


def parse_files(fns):
    if not os.path.exists("./translate_cache"):
        os.mkdir("./translate_cache")
    if not os.path.exists(os.path.join("./translate_cache", "tags")):
        os.mkdir(os.path.join("./translate_cache", "tags"))

    if not os.path.exists(os.path.join("./translate_cache", "tags_zh2en")):
        os.mkdir(os.path.join("./translate_cache", "tags_zh2en"))
    all_tags_fn = open(os.path.join('./translate_cache', 'zh-CN.txt'), 'w+', encoding='utf8')
    for fn in fns:
        stream = open(fn, 'r')
        data = load(stream, Loader=Loader)
        tag_type, tags = parse_tag(data)

        tag_type = tag_type.replace('/', '_').replace('\\', '_')
        for name, tag in tags:
            all_tags_fn.write(f'{name}={tag}\n')

        output_file = os.path.join("./translate_cache", "tags", f"{tag_type}.txt")
        with open(output_file, 'w+', encoding='utf8') as f:
            for name, tag in tags:
                f.write(f'{name}={tag}\n')

        output_file = os.path.join("./translate_cache", "tags_zh2en", f"{tag_type}.txt")
        with open(output_file, 'w+', encoding='utf8') as f:
            for name, tag in tags:
                f.write(f'{tag}={name}\n')
    all_tags_fn.close()


def parse_tag(data):
    tag_type = data['name']
    tags = []
    for name, item in data['content'].items():
        print(name, item['name'], tag_type)
        tags.append((name.strip(), item['name'].strip()))
    return tag_type, tags


fns = get_files('./danbooru-diffusion-prompt-builder/data/tags')
tags = parse_files(fns)
