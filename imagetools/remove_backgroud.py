import subprocess
import click
import os
import platform


def get_image_files(image_path):
    image_files = []
    for root, dirs, files in os.walk(image_path, topdown=False):
        for name in files:
            if name.endswith(".jpg") or name.endswith(".png"):
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


from config import settings


def remove_background(image_path, output_path, background_type, fast=False, jit=False):
    # Remove the background of the images
    # video_path: path to the source
    # output_path: path to the output
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    if os.path.exists(settings.image_tools.transparent_background_path):
        bin = settings.image_tools.transparent_background_path
    elif os.path.exists(os.path.join("venv", "Scripts", "transparent-background.exe")):
        bin = os.path.join("venv", "Scripts", "transparent-background.exe")
    elif os.path.exists(os.path.join("venv", "bin", "transparent-background")):
        bin = os.path.join("venv", "bin", "transparent-background")
    else:
        bin = "transparent-background"

    cmd = "{bin} --source {image_path} --dest {output_path}  --type {background_type}".format(
        bin=bin,
        image_path=image_path,
        output_path=output_path,
        background_type=background_type,
    )
    if fast:
        cmd += ' --fast'
    if jit:
        cmd += ' --jit'
    if platform.system() == "Drawin":
        cmd = 'PYTORCH_ENABLE_MPS_FALLBACK=1 ' + cmd
    env = os.environ.copy()
    env.update(
        {
            'PYTORCH_ENABLE_MPS_FALLBACK': '1'
        }
    )
    print(cmd)
    subprocess.call(cmd, shell=True, env=env)


if __name__ == '__main__':
    remove_cmd()
