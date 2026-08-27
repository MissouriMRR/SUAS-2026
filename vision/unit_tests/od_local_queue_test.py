"""A unit test for running image inference using the PhotoQueue class"""

import asyncio
import itertools
import logging
import os
import sys
from collections.abc import Iterable
from typing import cast

import numpy as np

import vision.pipeline.standard_pipeline as std_obj
from state_machine.flight_settings import FlightSettings
from vision.common.constants import ODLCDict
from vision.common.localized_detection import LocalizedDetection
from vision.object_detection import ObjectDetection
from vision.object_detection.providers.local.queue import PhotoQueue
from vision.pipeline import pipeline_utils
from vision.vision_pipeline import filter_detections

logger = logging.getLogger(__name__)


async def run_queue(
    images: Iterable[str], test_early_stop: bool = False
) -> list[ObjectDetection]:
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
    list[ObjectDetection]
        A list of ObjectDetection results.
    """
    initial_size: int = 0
    queue: PhotoQueue = PhotoQueue(True)
    for image_path in images:
        initial_size += 1
        await queue.add_photo(image_path)

    logger.info("%d images loaded into the queue. Starting queue...", initial_size)
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
    return list(
        itertools.islice((os.path.join(path, f.name) for f in os.scandir(path)), limit)
    )


async def test_queue(camera_data_path: str | None = None) -> None:
    """Runs the YOLO queue test. Prints each filtered detection.

    Parameters
    ----------
    camera_data_path : str | None, default=None
        A file storing photo metadata, used for proximity detection.
        If none the proximity check is skipped.
    """
    directory: str = sys.argv[1]
    all_images: list[str] = get_all_images(directory, 10)
    results: list[ObjectDetection] = await run_queue(all_images)
    if camera_data_path is not None:
        logger.info("Filtering detections with camera data")
        image_parameters = pipeline_utils.read_parameter_json(camera_data_path)
        results = filter_detections(results, image_parameters)
        localized_detections: list[LocalizedDetection] = []
        for detection in results:
            localized = pipeline_utils.localize_detection(
                detection, image_parameters[detection.image.split("/")[-1]]
            )
            if localized is not None:
                localized_detections.append(localized)

        odlc_dict: ODLCDict = std_obj.create_odlc_dict(
            localized_detections, FlightSettings.from_mission_config()
        )
        pipeline_utils.output_odlc_json("flight/data/output.json", odlc_dict)
    for result in results:
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    if len(sys.argv) < 2:
        logger.info("Need to provide a directory")
        sys.exit(1)
    asyncio.run(test_queue("flight/data/camera.json"))
