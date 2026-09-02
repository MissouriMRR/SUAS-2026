"""Functions that perform standard object detection, localization, and classification"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

import vision.common.constants as consts
from flight.extract_gps import GPSData, extract_gps
from vision.common.localized_detection import LocalizedDetection
from vision.object_detection import ObjectDetection
from vision.pipeline import localization

if TYPE_CHECKING:
    from state_machine.flight_settings import FlightSettings

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.8

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
