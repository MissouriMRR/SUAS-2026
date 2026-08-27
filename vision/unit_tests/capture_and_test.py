"""
This file contains a test that captures an image
using the gimbal and runs YOLO inference on it
"""

import asyncio
import logging
import os

import cv2
import numpy as np
import numpy.typing as npt

from flight.camera import CameraIRL
from vision.object_detection import ObjectDetection
from vision.object_detection.providers import LocalInferenceProvider


async def test_capture_and_test() -> None:
    """Capture an image using the gimbal and run YOLO inference on it"""
    camera = CameraIRL()
    camera.camera.requestSetAngles(0, 0)
    await asyncio.sleep(1)
    logging.info("Current camera attitude: %s", camera.camera.getAttitude())
    logging.info("Capturing images...")

    # If the images folder doesn't exist, we can't save images.
    # So we have to make sure the images folder exists.
    path: str = f"{os.getcwd()}/images/"
    os.makedirs(path, mode=0o777, exist_ok=True)
    image_path, _ = await camera.capture_photo()
    await asyncio.sleep(1)
    image: cv2.typing.MatLike | None = cv2.imread(image_path)
    if image is None:
        logging.error("%s is not an image", image_path)
        return
    cv2.imshow("Captured Image", image)

    # Run inference on the image
    yolo: LocalInferenceProvider = LocalInferenceProvider()
    await yolo.start()
    await yolo.add_image(image_path)
    results: list[ObjectDetection] = await yolo.end()

    for detection in results:
        conv: npt.NDArray[np.int32] = detection.bbox.astype(np.int32)
        logging.info(
            "Detected %s at (%d, %d), (%d, %d) Cfd: %.3f",
            detection.category,
            conv[0],
            conv[1],
            conv[2],
            conv[3],
            detection.confidence,
        )
        image = cv2.rectangle(
            image,
            (conv[0], conv[1]),
            (conv[2], conv[3]),
            (0, 255, 0),
            2,
        )
    image = cv2.resize(image, (1280, 720))
    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    camera.camera.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_capture_and_test())
