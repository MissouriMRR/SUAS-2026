"""A unit test for running singular images on the YOLO model"""

import asyncio
import logging
import os
import sys

import cv2
import numpy as np
import numpy.typing as npt

from vision.yolo.model import ObjectDetection, YOLO


async def test_image(image_path: str) -> None:
    """Test a single image using the YOLO model and show the results

    Parameters
    ----------
    image_path : str
        Path to the image to be processed
    """
    yolo: YOLO = YOLO(log_results=True)
    results: list[ObjectDetection] = await yolo.process_image(image_path)
    image: cv2.typing.MatLike = cv2.imread(image_path)
    for detection in results:
        conv: npt.NDArray[np.int32] = detection.bbox.astype(np.int32)
        cv2.rectangle(
            image,
            (conv[0], conv[1]),
            (conv[2], conv[3]),
            (0, 255, 0),
            2,
        )
        # Add caption with class
        cv2.putText(
            image,
            detection.category,
            (conv[0], conv[3] + 15),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 255, 0),
        )
    image = cv2.resize(image, (1280, 720))
    cv2.imshow("YOLO Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


async def run_all_images(image_paths: list[str]) -> None:
    """Run all images in the given list

    Parameters
    ----------
    image_paths : list[str]
        List of image paths to be processed
    """
    for image_path in image_paths:
        if os.path.isfile(image_path):
            logging.info("Processing file: %s", image_path)
            await test_image(image_path)
        else:
            logging.error("File not found: %s", image_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        logging.error("No image paths provided")
        sys.exit(1)
    file_names = sys.argv[1:]
    asyncio.run(run_all_images(file_names))
