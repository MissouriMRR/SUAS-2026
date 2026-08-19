"""A unit test for running image inference using the ObjectDetectionDriver class"""

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
from vision.common.bounding_box import BoundingBox
from vision.common.constants import ODLCDict
from vision.object_detection import ObjectDetection, ObjectDetectionDriver
from vision.pipeline import pipeline_utils
from vision.vision_pipeline import filter_detections


async def run_driver(images: Iterable[str]) -> list[ObjectDetection]:
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
    driver: ObjectDetectionDriver = ObjectDetectionDriver()
    await driver.start()

    initial_size: int = 0
    inference_tasks: list[asyncio.Task[None]] = []
    for image_path in images:
        initial_size += 1
        inference_tasks.append(asyncio.create_task(driver.add_image(image_path)))

    logging.info("%d images added to the driver. Waiting for inference to finish...", initial_size)
    task_results = await asyncio.gather(*inference_tasks, return_exceptions=True)
    for result in task_results:
        if isinstance(result, BaseException):
            logging.error("Image failed to process on all providers: %s", result)

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


async def test_driver(camera_data_path: str | None = None) -> None:
    """
    Runs the driver test on every image in the given directory. Prints each filtered detection.
    This is mostly copied from vision_pipeline, and can serve as a decent reference to how that
    will work.

    Parameters
    ----------
    camera_data_path : str | None, default=None
        A file storing photo metadata, used for proximity detection.
        If none the proximity check is skipped.
    """
    directory: str = sys.argv[1]
    all_images: list[str] = get_all_images(directory)
    results: list[ObjectDetection] = await run_driver(all_images)
    if camera_data_path is not None:
        logging.info("Filtering detections with camera data")
        image_parameters = pipeline_utils.read_parameter_json(camera_data_path)
        results = filter_detections(results, image_parameters)
        bounding_boxes: list[BoundingBox] = [
            pipeline_utils.detection_to_bbox(
                detection, image_parameters[detection.image.split("/")[-1]]
            )
            for detection in results
        ]

        odlc_dict: ODLCDict = std_obj.create_odlc_dict(
            bounding_boxes, FlightSettings.from_mission_config()
        )
        pipeline_utils.output_odlc_json("flight/data/output.json", odlc_dict)
    for result in results:
        x1, y1, x2, y2 = cast(tuple[int, int, int, int], result.bbox.astype(np.int32).tolist())
        logging.info(
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
        logging.info("Need to provide a directory")
        sys.exit(1)
    asyncio.run(test_driver("flight/data/camera.json"))
