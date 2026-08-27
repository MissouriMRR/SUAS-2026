"""Runs the necessary Vision code during the ODLC stage of competition"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import vision.common.constants as consts
from vision.object_detection import ObjectDetection, ObjectDetectionDriver
from vision.pipeline import odlc_utils, pipeline_utils

if TYPE_CHECKING:
    from state_machine.flight_settings import FlightSettings

logger = logging.getLogger(__name__)


async def odlc_pipeline(
    flight_settings: FlightSettings,
    camera_data_path: str,
    capture_status: asyncio.Event,
    output_path: str,
) -> None:
    """
    Finds all standard objects in each image in the input folder

    Parameters
    ----------
    flight_settings: FlightSettings
        The flight settings
    camera_data_path: str
        The path to the json file containing the CameraParameters entries
    capture_status: asyncio.Event
        An event that is set when all images have been taken
    output_path: str
        The json file name and path to save the data in
    """

    # Start object detection driver
    driver = ObjectDetectionDriver()
    await driver.start()

    # List of filenames for images already completed to prevent repeating work
    completed_images: list[str] = []

    # Dictionary storing all of the photo metadata (location, altitude, etc.)
    image_parameters: dict[str, consts.CameraParameters] = (
        pipeline_utils.read_parameter_json(camera_data_path)
    )

    # List of image inference tasks
    inference_tasks: list[asyncio.Task[None]] = []

    # Wait for and process unfinished images until no more images are being taken
    while not capture_status.is_set() or (
        set(image_parameters.keys()) - set(completed_images)
    ):
        # Wait to check the file instead of spamming it
        await asyncio.sleep(1)

        image_parameters = pipeline_utils.read_parameter_json(camera_data_path)

        # Loop through all images in the json - if it hasn't been processed, process it
        for image_path in image_parameters:
            if image_path not in completed_images:
                logger.info("Processing image: %s", image_path)
                full_image_path: str = f"images/{image_path}"

                # Save the image path as completed so it isn't processed again
                completed_images.append(image_path)

                # Add the image to the queue
                process_coro = driver.add_image(full_image_path)
                inference_tasks.append(asyncio.create_task(process_coro))

    # Wait for tasks to finish
    task_results = await asyncio.gather(*inference_tasks, return_exceptions=True)
    for result in task_results:
        if isinstance(result, BaseException):
            logger.error("Image failed to process on all providers: %s", result)

    # End the queue, get results
    detected_objects: list[ObjectDetection] = await driver.end()

    # Load in the json containing the camera data
    image_parameters = pipeline_utils.read_parameter_json(camera_data_path)

    # Filter all detections to the best for each class
    filtered_objects = odlc_utils.filter_detections(detected_objects, image_parameters)

    odlc_dict: consts.ODLCDict = odlc_utils.create_odlc_dict(
        filtered_objects, flight_settings
    )
    logger.info("%d ODLCs found: %s", len(odlc_dict), odlc_dict)
    odlc_utils.output_odlc_json(output_path, odlc_dict)
    flight_settings.yolo_status.set()
