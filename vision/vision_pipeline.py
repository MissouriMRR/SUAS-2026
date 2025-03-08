"""Runs the necessary Vision code during the flyover stage of competition"""

import logging
from ctypes import c_bool
from multiprocessing.sharedctypes import SynchronizedBase  # pylint: disable=unused-import

import asyncio

import vision.common.constants as consts

from vision.common.bounding_box import BoundingBox

import vision.pipeline.standard_pipeline as std_obj
import vision.pipeline.pipeline_utils as pipe_utils

from vision.yolov9.queue import PhotoQueue


async def flyover_pipeline(
    camera_data_path: str, capture_status: "SynchronizedBase[c_bool]", output_path: str
) -> None:
    """
    Finds all standard objects in each image in the input folder

    Parameters
    ----------
    camera_data_path: str
        The path to the json file containing the CameraParameters entries
    capture_status: SynchronizedBase[c_bool]
        A text file containing True if all images have been taken and False otherwise
    output_path: str
        The json file name and path to save the data in
    """

    # Load model and queue
    queue = PhotoQueue(True)

    # List of filenames for images already completed to prevent repeating work
    completed_images: list[str] = []

    # Dictionary storing all of the photo metadata (location, altitude, etc.)
    image_parameters: dict[str, consts.CameraParameters] = pipe_utils.read_parameter_json(
        camera_data_path
    )

    # Start the queue runner
    await queue.start_queue()

    # Wait for and process unfinished images until no more images are being taken
    all_images_taken: c_bool = c_bool(False)
    while not all_images_taken:
        # Wait to check the file instead of spamming it
        await asyncio.sleep(1)

        # Check if all images have been taken
        all_images_taken = capture_status.value  # type: ignore

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
    detected_objects: consts.DetectionList = await queue.end_queue()

    # Load in the json containing the camera data
    image_parameters = pipe_utils.read_parameter_json(camera_data_path)

    # Filter the detections if there are more than 4
    if len(detected_objects) > 4:
        detected_objects = std_obj.filter_objects(detected_objects)

    # Convert the final detections to BoundingBoxes
    bounding_boxes: list[BoundingBox] = []
    for detection in detected_objects.values():
        image_name = detection.image.split("/")[-1]
        parameters: consts.CameraParameters = image_parameters[image_name]
        bounding_boxes.append(pipe_utils.detection_to_bbox(detection, parameters))

    odlc_dict: consts.ODLCDict = std_obj.create_odlc_dict(bounding_boxes)
    logging.info("%d ODLCs found: %s", len(detected_objects), odlc_dict)
    pipe_utils.output_odlc_json(output_path, odlc_dict)
