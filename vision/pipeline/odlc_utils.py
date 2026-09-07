"""Functions that perform standard object detection, localization, and classification"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import vision.common.constants as consts
from flight.extract_gps import GPSData, extract_gps
from vision.common.constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DETECTIONS_OUTPUT_PATH,
    DEFAULT_REVIEWER_OUTPUT_PATH,
)
from vision.common.localized_detection import LocalizedDetection
from vision.object_detection import ObjectDetection
from vision.object_detection.providers.base import ObjectDetectionDict
from vision.pipeline import localization

if TYPE_CHECKING:
    from state_machine.flight_settings import FlightSettings

# Minimum number of pixels in a detection axis to be considered
MIN_DETECTION_SIZE: int = 2

logger = logging.getLogger(__name__)


def filter_detections(
    detections: list[ObjectDetection],
    image_parameters: dict[str, consts.CameraParameters],
    flight_settings: FlightSettings,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[LocalizedDetection]:
    """
    Filters all the detections to the best for each of the two classes (tent and mannequin).
    """
    logger.debug(f"Filtering {len(detections)} detections...")
    gps_data: GPSData = extract_gps(flight_settings.mission_data_path)

    # Remove detections that are too small
    detections = [
        detection
        for detection in detections
        if (
            detection.width >= MIN_DETECTION_SIZE
            and detection.height >= MIN_DETECTION_SIZE
        )
    ]
    logger.debug("Filtered to %d detections that are large enough.", len(detections))

    # Check if the detections are within the object boundary
    localized = localization.boundary_check(detections, image_parameters, gps_data)
    logger.debug(
        "Filtered to %d detections within the object boundary.", len(localized)
    )

    # Filter the detections to only include those with high enough confidence
    localized = [d for d in localized if d.confidence > confidence_threshold]
    logger.debug(
        "Filtered to %d detections with confidence > %f.",
        len(localized),
        confidence_threshold,
    )

    localized = localization.proximity_check(localized)
    logger.debug("Filtered to %d detections after proximity check.", len(localized))

    # Sort all detections by confidence in descending order
    localized.sort(key=lambda d: d.confidence, reverse=True)

    # First occurrence of each class in the localized list is the highest confidence one
    best_per_class: dict[str, LocalizedDetection] = {}
    for detection in localized:
        if detection.category not in best_per_class:
            best_per_class[detection.category] = detection

    return list(best_per_class.values())


def create_review_JSON(
    detections: list[ObjectDetection],
    output_path: Path = DEFAULT_DETECTIONS_OUTPUT_PATH,
) -> None:
    with open(output_path, "w") as f:
        json.dump([d.as_dict() for d in detections], f)


def import_review_JSON(
    path: Path = DEFAULT_REVIEWER_OUTPUT_PATH,
) -> list[ObjectDetection]:
    """
    Reads the detections accepted in the reviewer back in.

    Parameters
    ----------
    path : Path
        The JSON file the reviewer saved the accepted detections to.

    Returns
    -------
    detections : list[ObjectDetection]
        The accepted detections.
    """
    with open(path) as f:
        data: list[ObjectDetectionDict] = json.load(f)

    detections: list[ObjectDetection] = []
    for entry in data:
        # The reviewer stores bbox corners as floats and shape as a JSON list
        height, width = entry["shape"][0], entry["shape"][1]
        detections.append(
            ObjectDetection(
                image=entry["image"],
                category=entry["category"],
                bbox=np.array(entry["bbox"], dtype=np.int64),
                confidence=entry["confidence"],
                shape=(height, width),
            )
        )
    return detections


async def run_reviewer(
    detections: list[ObjectDetection],
    detections_path: Path = DEFAULT_DETECTIONS_OUTPUT_PATH,
    output_path: Path = DEFAULT_REVIEWER_OUTPUT_PATH,
) -> list[ObjectDetection]:
    """
    Opens the reviewer on the given detections and waits for the user to close it.

    The app runs as a subprocess because Qt has to own the main thread, which
    would block this event loop for as long as the review takes.

    Parameters
    ----------
    detections : list[ObjectDetection]
        The detections to review.
    detections_path : Path
        The JSON file the reviewer reads the detections from.
    output_path : Path
        The JSON file the reviewer writes the accepted detections to.

    Returns
    -------
    reviewed : list[ObjectDetection]
        The accepted detections, or all of the given detections if the reviewer
        was closed without saving.
    """
    create_review_JSON(detections, detections_path)

    # Clear the previous run's output so a discarded review can be detected
    output_path.unlink(missing_ok=True)

    # Importing opencv points QT_QPA_PLATFORM_PLUGIN_PATH at its own bundled Qt
    # plugins, which the reviewer would inherit and load instead of PySide6's
    environment = os.environ.copy()
    environment.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

    logger.info(f"Opening the reviewer with {len(detections)} detections...")
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "vision.reviewer",
        "--detection-data",
        str(detections_path),
        env=environment,
    )
    return_code = await process.wait()
    if return_code != 0:
        logger.error(f"Reviewer exited with code {return_code}")

    if not output_path.is_file():
        logger.warning("Reviewer closed without saving, keeping all detections")
        return detections

    reviewed = import_review_JSON(output_path)
    logger.info(f"Reviewer accepted {len(reviewed)} of {len(detections)} detections")
    return reviewed


def create_odlc_dict(
    localized_detections: Iterable[LocalizedDetection],
) -> consts.ODLCDict:
    """
    Creates the ODLCDict dictionary from a list of localized detections.

    Parameters
    ----------
    localized_detections : Iterable[LocalizedDetection]
        An iterable of the sightings of each object.

    Returns
    -------
    odlc_dict : consts.ODLCDict
        The dictionary of ODLCs matching the output format.
    """
    odlc_dict: consts.ODLCDict = {}
    for detection in localized_detections:
        odlc_dict[detection.category] = {
            "latitude": detection.latitude,
            "longitude": detection.longitude,
        }

    return odlc_dict


def output_odlc_json(output_path: str, odlc_dict: consts.ODLCDict) -> None:
    """
    Saves the ODLC_Dict to a file

    Parameters
    ----------
    output_path: str
        The json file name and path to save the data in
    odlc_dict: consts.ODLC_Dict
        The dictionary of ODLCs matched with bottles
    """

    with open(output_path, "w", encoding="UTF-8") as file:
        json.dump(odlc_dict, file, indent=4)
