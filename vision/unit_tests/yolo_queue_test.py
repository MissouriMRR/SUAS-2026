"""A unit test for running image inference using the PhotoQueue class"""

import asyncio
import itertools
import logging
import os
import sys
from typing import Iterable

from vision.pipeline.standard_pipeline import filter_objects
from vision.yolo.model import ObjectDetection
from vision.yolo.queue import PhotoQueue


async def run_queue(
    images: Iterable[str], test_early_stop: bool = False
) -> dict[str, ObjectDetection]:
    """
    Test the PhotoQueue class by adding images to the queue and running inference on them.

    Parameters
    ----------
    images : Iterable[str]
        An iterable of image paths to be added to the queue.
    test_early_stop : bool, default=False
        Whether to stop the queue early for testing purposes, by default False

    Returns
    -------
    dict[str, ObjectDetection]
        A dictionary mapping image paths to their corresponding best ObjectDetection results.
    """
    initial_size: int = 0
    queue: PhotoQueue = PhotoQueue(True)
    for image_path in images:
        initial_size += 1
        await queue.add_photo(image_path)

    logging.info("%d images loaded into the queue. Starting queue...", initial_size)
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
    return list(itertools.islice((os.path.join(path, f.name) for f in os.scandir(path)), limit))


async def test_queue() -> None:
    """Runs the YOLO queue test. Prints each filtered detection."""
    directory: str = sys.argv[1]
    all_images: list[str] = get_all_images(directory)
    results: dict[str, ObjectDetection] = await run_queue(all_images)
    filtered: dict[str, ObjectDetection] = filter_objects(results, True, 0.2)
    for result in filtered.values():
        logging.info(
            "Detected %s at (%d, %d), (%d, %d) Cfd: %.2f Img: %s",
            result.category,
            result.bbox[0],
            result.bbox[1],
            result.bbox[2],
            result.bbox[3],
            result.confidence,
            result.image,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    if len(sys.argv) < 2:
        logging.info("Need to provide a directory")
        sys.exit(1)
    asyncio.run(test_queue())
