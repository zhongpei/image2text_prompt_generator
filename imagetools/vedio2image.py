import subprocess
import click
import cv2
import os
import glob
import shutil
import numpy as np
import math
import re
import logging
from remove_backgroud import remove_background

_kernel = None

logger = logging.Logger(__name__)


@click.command()
@click.option("--movie_file", "-i", help="movie file", prompt="movie file")
@click.option("--image_path", default="./output_images", help="")
@click.option("--mask_path", default="./output_masks", help="")
@click.option("--fps", help="1/4 (every 4 seconds 1 frame)")
def video2image(movie_file, image_path, fps, mask_path):
    if not os.path.exists(image_path):
        os.mkdir(image_path)

    if not os.path.exists(mask_path):
        os.mkdir(mask_path)

    output_path = os.path.join(image_path, "%05d.png")

    if fps is not None:
        fps = "fps={}".format(fps)
    else:
        fps = ""
    cmd = "ffmpeg -ss 00:00:00  -y -i  {movie_file} {fps} -f image2 -c:v png {image_path}".format(
        movie_file=movie_file,
        fps=fps,
        image_path=output_path
    )
    print(cmd)
    subprocess.call(cmd, shell=True)
    # create_mask_transparent_background(image_path, mask_path)
    # keys = analyze_key_frames(png_dir=image_path,mask_dir=mask_path)


def mean_pixel_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return the mean average distance in pixel values between `left` and `right`.
    Both `left and `right` should be 2 dimensional 8-bit images of the same shape.
    """
    assert len(left.shape) == 2 and len(right.shape) == 2
    assert left.shape == right.shape
    num_pixels: float = float(left.shape[0] * left.shape[1])
    return (np.sum(np.abs(left.astype(np.int32) - right.astype(np.int32))) / num_pixels)


def create_mask_transparent_background(
        input_dir,
        output_dir,
        tb_use_fast_mode=False,
        tb_use_jit=False,
        st1_mask_threshold=1
):
    bin_path = os.path.join("venv", "Scripts")
    bin_path = os.path.join(bin_path, "transparent-background")

    remove_background(image_path=input_dir, output_path=output_dir, background_type="map", fast=tb_use_fast_mode,
                      jit=tb_use_jit)

    mask_imgs = glob.glob(os.path.join(output_dir, "*.png"))

    for m in mask_imgs:
        img = cv2.imread(m)
        img[img < int(255 * st1_mask_threshold)] = 0
        cv2.imwrite(m, img)

    p = re.compile(r'([0-9]+)_[a-z]*\.png')

    for mask in mask_imgs:
        base_name = os.path.basename(mask)
        m = p.fullmatch(base_name)
        if m:
            os.rename(mask, os.path.join(output_dir, m.group(1) + ".png"))


def estimated_kernel_size(frame_width: int, frame_height: int) -> int:
    """Estimate kernel size based on video resolution."""
    size: int = 4 + round(math.sqrt(frame_width * frame_height) / 192)
    if size % 2 == 0:
        size += 1
    return


def _detect_edges(lum: np.ndarray) -> np.ndarray:
    global _kernel
    """Detect edges using the luma channel of a frame.
    Arguments:
        lum: 2D 8-bit image representing the luma channel of a frame.
    Returns:
        2D 8-bit image of the same size as the input, where pixels with values of 255
        represent edges, and all other pixels are 0.
    """
    # Initialize kernel.
    if _kernel is None:
        kernel_size = estimated_kernel_size(lum.shape[1], lum.shape[0])
        _kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Estimate levels for thresholding.
    sigma: float = 1.0 / 3.0
    median = np.median(lum)
    low = int(max(0, (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))

    # Calculate edges using Canny algorithm, and reduce noise by dilating the edges.
    # This increases edge overlap leading to improved robustness against noise and slow
    # camera movement. Note that very large kernel sizes can negatively affect accuracy.
    edges = cv2.Canny(lum, low, high)
    return cv2.dilate(edges, _kernel)


def detect_edges(img_path, mask_path, is_invert_mask):
    im = cv2.imread(img_path)
    if mask_path:
        mask = cv2.imread(mask_path)[:, :, 0]
        mask = mask[:, :, np.newaxis]
        im = im * ((mask == 0) if is_invert_mask else (mask > 0))
    #        im = im * (mask/255)
    #        im = im.astype(np.uint8)
    #        cv2.imwrite( os.path.join( os.path.dirname(mask_path) , "tmp.png" ) , im)

    hue, sat, lum = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
    return _detect_edges(lum)


def get_mask_path_of_img(img_path, mask_dir):
    img_basename = os.path.basename(img_path)
    mask_path = os.path.join(mask_dir, img_basename)
    return mask_path if os.path.isfile(mask_path) else None


def analyze_key_frames(png_dir, mask_dir, th=8.5, min_gap=10, max_gap=300, add_last_frame=True, is_invert_mask=False):
    keys = []

    frames = sorted(glob.glob(os.path.join(png_dir, "[0-9]*.png")))

    key_frame = frames[0]
    keys.append(int(os.path.splitext(os.path.basename(key_frame))[0]))
    key_edges = detect_edges(key_frame, get_mask_path_of_img(key_frame, mask_dir), is_invert_mask)
    gap = 0

    for frame in frames:
        gap += 1
        if gap < min_gap:
            continue

        edges = detect_edges(frame, get_mask_path_of_img(frame, mask_dir), is_invert_mask)

        delta = mean_pixel_distance(edges, key_edges)

        _th = th * (max_gap - gap) / max_gap

        if _th < delta:
            basename_without_ext = os.path.splitext(os.path.basename(frame))[0]
            keys.append(int(basename_without_ext))
            key_frame = frame
            key_edges = edges
            gap = 0

    if add_last_frame:
        basename_without_ext = os.path.splitext(os.path.basename(frames[-1]))[0]
        last_frame = int(basename_without_ext)
        if not last_frame in keys:
            keys.append(last_frame)

    return keys


if __name__ == "__main__":
    video2image()
