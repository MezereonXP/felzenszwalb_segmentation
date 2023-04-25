import click
from felzenszwalb_segmentation import segment
import numpy as np
from glob import glob
import os
from loguru import logger
from PIL import Image


@click.command()
@click.option("--path", help="Path to the images")
@click.option("--save_path", help="Path to the saved images")
def run(path, save_path):
    if not os.path.exists(path):
        logger.error("Path does not exist")
        exit(0)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    images = glob(os.path.join(path, "*.jpg"))
    logger.info(f"Found {len(images)} images")

    for image in images:
        logger.info(f"Processing {image}")
        img = np.array(Image.open(image))
        segmented_img = segment(img, sigma=0.5, k=20, min_size=20)
        segmented_img = segmented_img.astype(np.uint8)
        logger.info(f"Saving to {save_path}")
        Image.fromarray(segmented_img).save(os.path.join(save_path, os.path.basename(image)))


if __name__ == "__main__":
    run()