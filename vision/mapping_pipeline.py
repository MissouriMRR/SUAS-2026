import logging
from typing import Callable
from ctypes import c_bool
from multiprocessing.sharedctypes import SynchronizedBase  # pylint: disable=unused-import

import asyncio
import cv2

from mapping.map import Map

import vision.common.constants as consts
import vision.pipeline.pipeline_utils as pipe_utils

# Disable duplicate code checking because the flyover pipeline is similar
# pylint: disable=duplicate-code
async def airdrop_pipeline(camera_data_path: str, state_path: str, output_path: str) -> None:

    # List of filenames for images already completed to prevent repeating work
    completed_images: list[str] = []
    
    map: Map = Map()

    # Wait for and process unfinished images until no more images are being taken
    all_images_taken: c_bool = c_bool(False)
    while not all_images_taken:
        # Wait to check the file instead of spamming it
        await asyncio.sleep(1)

        # Check if all images have been taken
        all_images_taken = capture_status.value  # type: ignore

        # Load in the json containing the camera data
        image_parameters: dict[str, consts.CameraParameters] = pipe_utils.read_parameter_json(
            camera_data_path
        )

        # Loop through all images in the json - if it hasn't been processed, process it
        for image_path in image_parameters.keys():
            if image_path not in completed_images:
                logging.info("Processing image: %s", image_path)
                full_image_path: str = f"images/{image_path}"
                # Save the image path as completed so it isn't processed again
                completed_images.append(image_path)

                # Load the image to process
                image: consts.Image = cv2.imread(full_image_path)

                # Get the camera parameters from the loaded parameter file
                camera_parameters: consts.CameraParameters = image_parameters[image_path]
                
                map.add_img(image, camera_parameters)