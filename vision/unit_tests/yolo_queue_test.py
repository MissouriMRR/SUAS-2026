"""A unit test for running image inference using the PhotoQueue class"""

import asyncio
import logging
import os
import sys

from vision.yolov9.model import ObjectDetection
from vision.yolov9.queue import PhotoQueue


async def test_queue(
    all_images: list[str], test_early_stop: bool = False
) -> dict[str, ObjectDetection]:
    """
    Test the PhotoQueue class by adding images to the queue and running inference on them.

    Parameters
    ----------
    all_images : list[str]
        A list of image paths to be added to the queue.
    test_early_stop : bool, default=False
        Whether to stop the queue early for testing purposes, by default False

    Returns
    -------
    dict[str, ObjectDetection]
        A dictionary mapping image paths to their corresponding best ObjectDetection results.
    """
    initial_size: int = len(all_images)
    queue: PhotoQueue = PhotoQueue(True)
    logging.info("Starting queue with %d images", initial_size)
    for image_path in all_images:
        await queue.add_photo(image_path)

    await queue.start_queue()
    while not queue.queue.empty():
        if test_early_stop and queue.queue.qsize() < initial_size / 2:
            break
        await asyncio.sleep(10)

    return await queue.end_queue()


def get_all_images(path: str, limit: int = 10) -> list[str]:
    """Get all images in a directory.

    Parameters
    ----------
    path : str
        The directory to search for images.
    limit : int, default=10
        The maximum number of images to return.

    Returns
    -------
    list[str]
        A list of image paths.
    """
    photos: list[str] = [os.path.join(path, f) for f in os.listdir(path)]
    return photos[:limit]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    if len(sys.argv) < 2:
        logging.info("Need to provide a directory")
        sys.exit(1)
    directory: str = sys.argv[1]
    images: list[str] = get_all_images(directory)
    logging.info(asyncio.run(test_queue(images)))
