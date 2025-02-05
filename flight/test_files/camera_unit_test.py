"""Checks if the camera object is connected correctly and can save pictures to the images folder."""

import asyncio
import logging
import os

from flight.camera import Camera


async def test_capture_image(photo_count: int = 5) -> None:
    """Test the capture_photo method of the Camera class.

    Parameters
    ----------
    photo_count : int, optional
        The number of photos to take, by default 1
    """
    camera: Camera = Camera()

    logging.info("Capturing images...")

    # If the images folder doesn't exist, we can't save images.
    # So we have to make sure the images folder exists.
    path: str = f"{os.getcwd()}/images/"
    os.makedirs(path, mode=0o777, exist_ok=True)

    while camera.image_id < photo_count:
        await camera.capture_photo()
        await asyncio.sleep(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_capture_image())
