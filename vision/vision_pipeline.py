"""Runs the necessary Vision code during the flyover stage of competition"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import vision.common.constants as consts
import vision.pipeline.pipeline_utils as pipe_utils
import vision.pipeline.standard_pipeline as std_obj
from vision.common.bounding_box import BoundingBox
from vision.object_detection.model import ObjectDetection
from vision.object_detection.queue import PhotoQueue

if TYPE_CHECKING:
    from state_machine.flight_settings import FlightSettings


def filter_detections(
    detections: list[ObjectDetection],
    image_parameters: dict[str, consts.CameraParameters],
) -> list[ObjectDetection]:
    """
    Filters all the detections to the best for each of the two classes (tent and mannequin).
    """
    deduped: list[tuple[ObjectDetection, BoundingBox]] = std_obj.proximity_check(
        detections, image_parameters
    )

    # First occurrence of each class in the deduped list is the highest confidence one
    # Since it is sorted in proximity_check
    best_per_class: dict[str, ObjectDetection] = {}
    for detection, _ in deduped:
        if detection.category not in best_per_class:
            best_per_class[detection.category] = detection

    return list(best_per_class.values())


async def flyover_pipeline(
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

    # Load model and queue
    queue: PhotoQueue = PhotoQueue(True)

    # List of filenames for images already completed to prevent repeating work
    completed_images: list[str] = []

    # Dictionary storing all of the photo metadata (location, altitude, etc.)
    image_parameters: dict[str, consts.CameraParameters] = pipe_utils.read_parameter_json(
        camera_data_path
    )

    # Start the queue runner
    await queue.start_queue()

    # Wait for and process unfinished images until no more images are being taken
    while not capture_status.is_set() or (set(image_parameters.keys()) - set(completed_images)):
        # Wait to check the file instead of spamming it
        await asyncio.sleep(1)

        image_parameters = pipe_utils.read_parameter_json(camera_data_path)

        # Loop through all images in the json - if it hasn't been processed, process it
        for image_path in image_parameters.keys():
            if image_path not in completed_images:
                logging.info("Processing image: %s", image_path)
                full_image_path: str = f"images/{image_path}"

                # Save the image path as completed so it isn't processed again
                completed_images.append(image_path)

                # Add the image to the queue
                await queue.add_photo(full_image_path)

    # End the queue, get results
    detected_objects: list[ObjectDetection] = await queue.end_queue()

    # Load in the json containing the camera data
    image_parameters = pipe_utils.read_parameter_json(camera_data_path)

    # Filter all detections to the best for each class
    detected_objects = filter_detections(detected_objects, image_parameters)

    # Convert detections to bounding boxes with geographic information
    bounding_boxes: list[BoundingBox] = [
        pipe_utils.detection_to_bbox(detection, image_parameters[detection.image.split("/")[-1]])
        for detection in detected_objects
    ]

    odlc_dict: consts.ODLCDict = std_obj.create_odlc_dict(bounding_boxes, flight_settings)
    logging.info("%d ODLCs found: %s", len(odlc_dict), odlc_dict)
    pipe_utils.output_odlc_json(output_path, odlc_dict)
    flight_settings.yolo_status.set()
