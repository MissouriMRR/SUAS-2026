"""Checks if the camera object is connected correctly and can save pictures to the images folder."""

import asyncio
from datetime import datetime
import logging
import os

import cv2
from siyi_sdk.siyi_sdk import SIYISDK
from siyi_sdk.stream import SIYIRTSP


async def test_capture_image(photo_num: int = 5) -> None:
    """Test the capture_photo method of the Camera class.

    Parameters
    ----------
    photo_num : int, optional
        The number of photos to take, by default 1
    """
    logging.info("Connecting to camera...")
    camera: SIYISDK = SIYISDK()
    if not camera.connect(max_retries=10):
        logging.error("Failed to initialize camera. Exiting...")
        return

    try:
        camera_name: str = camera.getCameraTypeString()
        stream: SIYIRTSP = SIYIRTSP(
            rtsp_url="rtsp://192.168.144.25:8554/main.264",
            debug=False,
            cam_name=camera_name,
        )
    except Exception as ex:
        logging.error("Failed to initialize camera. Exiting...")
        logging.error(ex)
        return

    logging.info("Camera connected. Initializing capture...")
    session_id: int = 0
    if os.path.exists(f"{os.getcwd()}/images/"):
        for file in os.listdir(f"{os.getcwd()}/images/"):
            if file.startswith(f"{datetime.now().strftime('%Y%m%d')}"):
                if int(file.split("_")[1]) >= session_id:
                    session_id = int(file.split("_")[1]) + 1

    image_id: int = 1
    logging.info("Capturing images...")

    # If the images folder doesn't exist, we can't save images.
    # So we have to make sure the images folder exists.
    path: str = f"{os.getcwd()}/images/"
    os.makedirs(path, mode=0o777, exist_ok=True)

    while image_id <= photo_num:
        photo_name: str = f"{datetime.now().strftime('%Y%m%d')}_{session_id}_{image_id:04d}.jpg"
        target_name: str = f"{path}{photo_name}"
        cv2.imwrite(target_name, stream.getFrame())
        logging.info("Image #%d is being saved to %s", image_id, target_name)
        image_id += 1
        await asyncio.sleep(1)
        continue


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_capture_image())
