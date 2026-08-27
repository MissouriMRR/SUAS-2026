"""Functions that perform standard object detection, localization, and classification"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import TYPE_CHECKING

import utm

import vision.common.constants as consts
from flight.extract_gps import GPSData, extract_gps
from flight.waypoint.geometry import Point
from vision.common.localized_detection import LocalizedDetection
from vision.object_detection import ObjectDetection
from vision.pipeline import localization

if TYPE_CHECKING:
    from state_machine.flight_settings import FlightSettings

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.8


def filter_detections(
    detections: list[ObjectDetection],
    image_parameters: dict[str, consts.CameraParameters],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[LocalizedDetection]:
    """
    Filters all the detections to the best for each of the two classes (tent and mannequin).
    """
    # Filter the detections to only include those with high enough confidence
    detections = [d for d in detections if d.confidence > confidence_threshold]

    deduped: list[LocalizedDetection] = localization.proximity_check(
        detections, image_parameters
    )

    # Sort all detections by confidence in descending order
    deduped.sort(key=lambda d: d.confidence, reverse=True)

    # First occurrence of each class in the deduped list is the highest confidence one
    best_per_class: dict[str, LocalizedDetection] = {}
    for detection in deduped:
        if detection.category not in best_per_class:
            best_per_class[detection.category] = detection

    return list(best_per_class.values())


def create_odlc_dict(
    localized_detections: Iterable[LocalizedDetection], flight_settings: FlightSettings
) -> consts.ODLCDict:
    """
    Creates the ODLCDict dictionary from a list of localized detections.
    Discards detections whose center is not inside the airdrop boundary.

    Parameters
    ----------
    localized_detections : Iterable[LocalizedDetection]
        An iterable of the sightings of each object.
    flight_settings : FlightSettings
        The flight settings.
        Used to get the airdrop boundary.

    Returns
    -------
    odlc_dict : consts.ODLCDict
        The dictionary of ODLCs matching the output format.
    """

    # Get ODLC boundary
    gps_data: GPSData = extract_gps(flight_settings.mission_data_path)
    odlc_boundary: list[Point] = [
        Point(odlc_boundary_point.easting, odlc_boundary_point.northing)
        for odlc_boundary_point in gps_data["object_boundary_utm"]
    ]
    zone_number: int = gps_data["object_boundary_utm"][0].zone_number
    zone_letter: str = gps_data["object_boundary_utm"][0].zone_letter

    odlc_dict: consts.ODLCDict = {}

    detection: LocalizedDetection
    for detection in localized_detections:
        # Check if in bounds
        easting: float
        northing: float
        easting, northing, _, _ = utm.from_latlon(
            detection.latitude,
            detection.longitude,
            force_zone_number=zone_number,
            force_zone_letter=zone_letter,
        )
        if not Point(easting, northing).is_inside_shape(odlc_boundary):
            continue

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
