import subprocess
import click
import os
import platform
from transparent_background import Remover
from transparent_background.utils import ImageLoader
from PIL import Image


def get_image_files(image_path):
    image_files = []
    for root, dirs, files in os.walk(image_path, topdown=False):
        for name in files:
            lower_name = name.lower()
            if lower_name.endswith(".jpg") or lower_name.endswith(".png"):
                image_files.append(os.path.join(root, name))
    return image_files


@click.command()
@click.option('--image_path', default='download_images', help='Path to the image')
@click.option('--output_path', default='out_images', help='Path to the output ')
@click.option('--background_type', '-t', default='green', help='rgba map green blur overlay')
@click.option('--fast', '-f', is_flag=True, default=False, help='Fast mode')
@click.option('--jit', '-j', is_flag=True, default=False, help='Jit mode')
def remove_cmd(image_path, output_path, background_type, fast, jit):
    if platform.system() == 'Darwin':
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        fns = get_image_files(image_path)
        for fn in fns:
            remove_background(fn, output_path, background_type, fast, jit)
    else:
        remove_background(image_path, output_path, background_type, fast, jit)


def remove_background(
        image_path,
        output_path,
        background_type,
        fast=False,
        jit=False,
        ckpt=None,
):
    # Remove the background of the images
    # video_path: path to the source
    # output_path: path to the output
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    if ckpt is not None:
        if not os.path.isfile(ckpt):
            ckpt = None
    print("ckpt: {} fast: {}".format(ckpt, fast))
    remover = Remover(fast=fast, jit=jit, ckpt=ckpt)

    for img, name in ImageLoader(image_path):
        print("Processing: {}".format(name))
        out = remover.process(img, background_type)
        outname = '{}'.format(os.path.splitext(name)[0])
        Image.fromarray(out).save(os.path.join(output_path, '{}.png'.format(outname)))


if __name__ == '__main__':
    remove_cmd()
