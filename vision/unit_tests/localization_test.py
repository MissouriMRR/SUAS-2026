"""
Runs vision.pipeline.localization.localize_detection() over a hand-labeled
dataset and reports how far each localized coordinate lands from the object's
real coordinate.

The labeled data lives in `vision/unit_tests/data/localization_test_data.json`.
Each entry names a folder under `datasets/` and an image inside it; the camera
parameters for that image are read from the folder's `camera.json`.

This test is report-only: it logs the per-detection and aggregate error so the
localization accuracy of a collected dataset can be eyeballed. A detection is
only counted as a failure when localize_detection() cannot produce a coordinate
at all.
"""

import json
import logging
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict

import cv2
import numpy as np

import vision.common.constants as consts
from flight.waypoint.calculate_distance import calculate_distance
from vision.common.localized_detection import LocalizedDetection
from vision.object_detection import ObjectDetection
from vision.pipeline.localization import localize_detection

logger = logging.getLogger(__name__)

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
TEST_DATA_PATH: Path = Path(__file__).parent / "data" / "localization_test_data.json"

# The layout of the report table - a left-aligned object name, then the error
# and the localized coordinate right-aligned in their own columns
TABLE_ROW: str = "%-32s%12s%34s"
TABLE_RULE: str = "-" * 78


class DetectionData(TypedDict):
    """
    One labeled object as written in the test data JSON.

    Attributes
    ----------
    label : NotRequired[str]
        A human-readable name for the object, defaults to the image filename.
    category : NotRequired[str]
        The category the object would be detected as, defaults to "object".
    bbox : list[int]
        The bounding box corners as [x1, y1, x2, y2], where (x1, y1) is the
        top-left corner and (x2, y2) is the bottom-right corner.
    confidence : NotRequired[float]
        The detection confidence, defaults to 1.0 since the box is hand-labeled.
    real_coordinates : consts.Location
        The surveyed (latitude, longitude) of the object on the ground.
    """

    label: NotRequired[str]
    category: NotRequired[str]
    bbox: list[int]
    confidence: NotRequired[float]
    real_coordinates: consts.Location


class ImageData(TypedDict):
    """
    One labeled image as written in the test data JSON.

    Attributes
    ----------
    dataset : str
        The name of the folder under the dataset root holding the image.
    image : str
        The image's filename, which is also its key in the folder's camera.json.
    detections : list[DetectionData]
        Every labeled object in the image.
    """

    dataset: str
    image: str
    detections: list[DetectionData]


class TestData(TypedDict):
    """
    The contents of the test data JSON.

    Attributes
    ----------
    dataset_root : NotRequired[str]
        The folder holding the dataset folders, relative to the repository
        root, defaults to "datasets".
    images : list[ImageData]
        Every labeled image to run through localize_detection().
    """

    dataset_root: NotRequired[str]
    images: list[ImageData]


@dataclass
class LabeledDetection:
    """
    One hand-labeled object in one image, along with everything needed to
    localize it and score the result.

    Attributes
    ----------
    label : str
        A human-readable name for the object, used in the logged report.
    detection : ObjectDetection
        The detection handed to localize_detection().
    parameters : consts.CameraParameters
        The camera.json entry for the image the object was found in.
    real_coordinates : consts.Location
        The surveyed (latitude, longitude) of the object on the ground.
    """

    label: str
    detection: ObjectDetection
    parameters: consts.CameraParameters
    real_coordinates: consts.Location


def load_labeled_detections(data_path: Path = TEST_DATA_PATH) -> list[LabeledDetection]:
    """
    Reads the labeled test data JSON and builds an ObjectDetection plus the
    matching CameraParameters for every labeled object in it.

    Parameters
    ----------
    data_path : Path, default=TEST_DATA_PATH
        The path to the labeled test data JSON.

    Returns
    -------
    labeled_detections : list[LabeledDetection]
        Every labeled object across every image listed in the JSON.
    """
    with open(data_path, encoding="utf-8") as data_file:
        test_data: TestData = json.load(data_file)

    dataset_root: Path = REPO_ROOT / test_data.get("dataset_root", "datasets")

    labeled_detections: list[LabeledDetection] = []

    image_entry: ImageData
    for image_entry in test_data["images"]:
        dataset_dir: Path = dataset_root / image_entry["dataset"]
        image_name: str = image_entry["image"]
        image_path: Path = dataset_dir / image_name

        if not image_path.is_file():
            raise FileNotFoundError(f"Labeled image not found: {image_path}")

        # camera.json is keyed by image filename, matching the format written
        # out alongside the images when the dataset is captured
        with open(dataset_dir / "camera.json", encoding="utf-8") as camera_file:
            camera_data: dict[str, consts.CameraParameters] = json.load(camera_file)

        if image_name not in camera_data:
            raise KeyError(
                f"{image_name} has no entry in {dataset_dir / 'camera.json'}"
            )

        parameters: consts.CameraParameters = camera_data[image_name]

        # The image is read only for its shape - localize_detection() needs the
        # real pixel dimensions to place the bounding box in the camera's view
        raw_image: cv2.typing.MatLike | None = cv2.imread(str(image_path))
        if raw_image is None:
            raise ValueError(f"{image_path} could not be read as an image")

        height: int
        width: int
        height, width = raw_image.shape[:2]
        image_shape: consts.ImageShape = (height, width, 3)

        detection_entry: DetectionData
        for detection_entry in image_entry["detections"]:
            labeled_detections.append(
                LabeledDetection(
                    label=detection_entry.get("label", image_name),
                    detection=ObjectDetection(
                        image=str(image_path),
                        category=detection_entry.get("category", "object"),
                        bbox=np.array(detection_entry["bbox"], dtype=np.int64),
                        # Note: we're not using confidence in the localization
                        # test data at the moment. This will always be 1.0
                        confidence=detection_entry.get("confidence", 1.0),
                        shape=image_shape,
                    ),
                    parameters=parameters,
                    real_coordinates=detection_entry["real_coordinates"],
                )
            )

    return labeled_detections


def test_localize_detection(data_path: Path = TEST_DATA_PATH) -> list[float]:
    """
    Localizes every labeled detection and logs its error in meters, then logs
    the mean, median, min and max error across the whole dataset.

    Any detection that localize_detection() returns None for is logged as an
    error and left out of the statistics, since it has no coordinate to score.

    Parameters
    ----------
    data_path : Path, default=TEST_DATA_PATH
        The path to the labeled test data JSON.

    Returns
    -------
    errors_m : list[float]
        The error in meters of every detection that could be localized.
    """
    labeled_detections: list[LabeledDetection] = load_labeled_detections(data_path)
    if not labeled_detections:
        logger.error("No labeled detections found in %s", data_path)
        return []

    errors_m: list[float] = []

    logger.info(TABLE_ROW, "object", "error (m)", "localized (lat, lon)")
    logger.info(TABLE_RULE)

    labeled: LabeledDetection
    for labeled in labeled_detections:
        localized: LocalizedDetection | None = localize_detection(
            labeled.detection, labeled.parameters
        )

        if localized is None:
            logger.error(
                "No ground intersect for %s in %s",
                labeled.label,
                labeled.detection.image,
            )
            continue

        # Both altitudes are zero, so this is the ground distance between the
        # localized point and the surveyed point
        error_m: float = calculate_distance(
            localized.latitude,
            localized.longitude,
            0,
            labeled.real_coordinates["latitude"],
            labeled.real_coordinates["longitude"],
            0,
        )
        errors_m.append(error_m)

        logger.info(
            TABLE_ROW,
            labeled.label,
            f"{error_m:.2f}",
            f"({localized.latitude:.7f}, {localized.longitude:.7f})",
        )

    if not errors_m:
        logger.error("Nothing could be localized, no statistics to report")
        return errors_m

    logger.info(TABLE_RULE)
    logger.info("Detections localized: %d / %d", len(errors_m), len(labeled_detections))
    logger.info("Mean error:   %.2f m", statistics.fmean(errors_m))
    logger.info("Median error: %.2f m", statistics.median(errors_m))
    logger.info("Min error:    %.2f m", min(errors_m))
    logger.info("Max error:    %.2f m", max(errors_m))

    return errors_m


if __name__ == "__main__":
    # Bare message format so the report table lines up in the terminal
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # An alternate labeled data JSON can be passed in, otherwise the one in
    # vision/unit_tests/data is used
    test_localize_detection(Path(sys.argv[1]) if len(sys.argv) > 1 else TEST_DATA_PATH)
