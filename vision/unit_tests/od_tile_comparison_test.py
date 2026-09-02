"""A unit test for running image inference using the ObjectDetectionDriver class"""

import asyncio
import itertools
import logging
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import numpy as np

from vision.object_detection import ObjectDetection, ObjectDetectionDriver

logger = logging.getLogger(__name__)


async def run_driver(
    driver: ObjectDetectionDriver, images: Iterable[str]
) -> list[ObjectDetection]:
    """
    Test the ObjectDetectionDriver class by adding images to it and running inference on them.
    This is mostly copied from vision_pipeline, and can serve as a decent reference to how that
    will work.

    Parameters
    ----------
    images : Iterable[str]
        An iterable of image paths to be added to the driver.

    Returns
    -------
    list[ObjectDetection]
        A list of ObjectDetection results.
    """
    await driver.start()

    initial_size: int = 0
    inference_tasks: list[asyncio.Task[None]] = []
    for image_path in images:
        initial_size += 1
        inference_tasks.append(asyncio.create_task(driver.add_image(image_path)))
        await asyncio.sleep(2)

    logger.info(
        "%d images added to the driver. Waiting for inference to finish...",
        initial_size,
    )
    task_results = await asyncio.gather(*inference_tasks, return_exceptions=True)
    for result in task_results:
        if isinstance(result, BaseException):
            logger.error("Image failed to process on all providers: %s", result)

    return await driver.end()


def get_all_images(path: str, limit: int | None = None) -> list[str]:
    """Get all images in a directory.

    Parameters
    ----------
    path : str
        The directory to search for images.
    limit : int | None, default=None
        The maximum number of images to return, or None to return all of them.

    Returns
    -------
    list[str]
        A list of image paths.
    """
    all_paths = (os.path.join(path, f.name) for f in os.scandir(path))
    if limit is None:
        return list(all_paths)
    return list(itertools.islice(all_paths, limit))


async def tile_setting_test() -> None:
    """
    Runs a inference test on given images using three drivers with different tile settings:
        1. 1280 tiles
        2. 640 tiles
        3. No tiling
    Used to compare how tile settings impact inference results.
    """
    image_path: Path = Path(sys.argv[1])
    all_images: list[str] = (
        [str(image_path)] if image_path.is_file() else get_all_images(str(image_path))
    )

    driver_1280_tiles = ObjectDetectionDriver(tile_images=True, tile_size=1280)
    driver_640_tiles = ObjectDetectionDriver(tile_images=True, tile_size=640)
    driver_no_tiling = ObjectDetectionDriver(tile_images=False)
    drivers: list[tuple[ObjectDetectionDriver, str]] = [
        (driver_1280_tiles, "1280px tiling"),
        (driver_640_tiles, "640px tiling"),
        (driver_no_tiling, "no tiling"),
    ]
    results: list[list[ObjectDetection]] = []

    for driver, title in drivers:
        await driver.start()
        results.append(await run_driver(driver, all_images))

    for i, (driver, title) in enumerate(drivers):
        logger.info(f"--- Results for {title} ---")
        for result in results[i]:
            x1, y1, x2, y2 = cast(
                tuple[int, int, int, int], result.bbox.astype(np.int32).tolist()
            )
            logger.info(
                "Detected %s at (%d, %d), (%d, %d) Cfd: %.2f Img: %s",
                result.category,
                x1,
                y1,
                x2,
                y2,
                result.confidence,
                result.image,
            )
        logger.info("-----------------------------")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    if len(sys.argv) < 2:
        logger.info("Need to provide a file/directory")
        sys.exit(1)
    asyncio.run(tile_setting_test())
