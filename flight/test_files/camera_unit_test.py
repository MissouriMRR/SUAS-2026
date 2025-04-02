"""Checks if the camera object is connected correctly and can save pictures to the images folder."""

import asyncio
import logging
import os

from flight.camera import CameraIRL


async def test_capture_image(photo_count: int = 5) -> None:
    """Test the capture_photo method of the Camera class.

    Parameters
    ----------
    photo_count : int, default 5
        The number of photos to take.
    """
    camera: CameraIRL = CameraIRL()
    logging.info("Current camera attitude: %s", camera.camera.getAttitude())
    logging.info("Capturing images...")

    # If the images folder doesn't exist, we can't save images.
    # So we have to make sure the images folder exists.
    path: str = f"{os.getcwd()}/images/"
    os.makedirs(path, mode=0o777, exist_ok=True)

    while camera.image_id < photo_count:
        await camera.capture_photo()
        await asyncio.sleep(1)

    camera.camera.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_capture_image())
