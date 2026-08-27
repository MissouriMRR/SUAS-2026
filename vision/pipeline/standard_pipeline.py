"""Functions that perform standard object detection, localization, and classification"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import utm

import vision.common.constants as consts
import vision.pipeline.pipeline_utils as pipe_utils
from flight.extract_gps import GPSData, extract_gps
from flight.waypoint.calculate_distance import calculate_distance
from flight.waypoint.geometry import Point
from vision.common.localized_detection import LocalizedDetection
from vision.object_detection import ObjectDetection

if TYPE_CHECKING:
    from state_machine.flight_settings import FlightSettings


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


def proximity_check(
    detections: list[ObjectDetection],
    parameters: dict[str, consts.CameraParameters],
    min_distance: float = 7.0,
) -> list[tuple[ObjectDetection, LocalizedDetection]]:
    """
    Checks for detections that are too close to each other, and
    removes the one with lower confidence.
    The ruleset states that objects must be at least 50 feet apart.

    Parameters
    ----------
    detections : list[ObjectDetection]
        A list with all object detections.
    parameters : dict[str, consts.CameraParameters]
        A dictionary with the image parameters for every
        captured image.
    min_distance : float, optional
        The minimum distance between objects in METERS, by default 7.0

    Returns
    -------
    filtered_detections : list[tuple[ObjectDetection, LocalizedDetection]]
        All detections that are at least `min_distance` apart, along with
        their localized detections.
    """
    filtered_detections: list[tuple[ObjectDetection, LocalizedDetection]] = []
    # If we sort the detections by confidence, we can stop as soon as we find a collision
    detections.sort(key=lambda entry: entry.confidence, reverse=True)

    for detection in detections:
        image_name: str = detection.image.split("/")[-1]
        image_parameters: consts.CameraParameters = parameters[image_name]
        localized: LocalizedDetection | None = pipe_utils.localize_detection(
            detection, image_parameters
        )
        if localized is None:
            continue

        collided: bool = False
        existing_localized: LocalizedDetection
        for existing_detection, existing_localized in filtered_detections:
            if existing_detection.category != detection.category:
                continue

            distance: float = calculate_distance(
                localized.latitude,
                localized.longitude,
                0,
                existing_localized.latitude,
                existing_localized.longitude,
                0,
            )
            if distance < min_distance:
                collided = True
                break

        if not collided:
            filtered_detections.append((detection, localized))

    return filtered_detections
