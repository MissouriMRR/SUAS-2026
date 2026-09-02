"""Contains all pipeline utility functions that relate to localization of object detections."""

from __future__ import annotations

import logging
import math

import utm

import vision.common.constants as consts
from flight.extract_gps import GPSData
from flight.waypoint.calculate_distance import calculate_distance
from flight.waypoint.geometry import Point
from vision.common.localized_detection import LocalizedDetection
from vision.deskew.camera_distances import get_coordinates
from vision.object_detection import ObjectDetection

logger = logging.getLogger(__name__)


def localize_detection(
    detection: ObjectDetection, parameters: consts.CameraParameters
) -> LocalizedDetection | None:
    """
    Converts an ObjectDetection to a LocalizedDetection by deskewing its
    center pixel into a latitude/longitude.

    Parameters
    ----------
    detection: ObjectDetection
        The object detection to convert
    parameters: consts.CameraParameters
        The details of how and where the photo was taken

    Returns
    -------
    localized: LocalizedDetection | None
        The detection with its ground latitude/longitude, or None if the
        object's center pixel had no valid ground intersect.
    """
    coordinates: tuple[float, float] | None = get_coordinates(
        detection.get_center_coord(), detection.shape, parameters
    )

    if coordinates is None:
        return None

    latitude, longitude = coordinates

    return LocalizedDetection(
        image=detection.image,
        category=detection.category,
        bbox=detection.bbox,
        confidence=detection.confidence,
        shape=detection.shape,
        latitude=latitude,
        longitude=longitude,
    )


def get_center_offset(detection: ObjectDetection) -> float:
    """
    Gets the pixel distance between a detection's bounding box center and
    the center of the image it was detected in. A smaller offset means the
    object was closer to directly beneath the drone when the photo was
    taken, which correlates to a more accurate deskewed location.

    Parameters
    ----------
    detection: ObjectDetection
        The object detection to measure.

    Returns
    -------
    offset : float
        The pixel distance from the image center to the detection's
        bounding box center.
    """
    height, width = detection.shape[:2]
    image_center_x, image_center_y = width / 2, height / 2

    detection_x, detection_y = detection.get_center_coord()

    return math.dist((detection_x, detection_y), (image_center_x, image_center_y))


def boundary_check(
    detections: list[ObjectDetection],
    parameters: dict[str, consts.CameraParameters],
    gps_data: GPSData,
) -> list[LocalizedDetection]:
    """
    Removes all detections that are not within the object boundary specified in the
    flight data file.

    Parameters
    ----------
    detections : list[ObjectDetection]
        The list of detections to filter.
    parameters : dict[str, consts.CameraParameters]
        The camera parameters for each image.
    gps_data : GPSData
        The GPS data for the flight.

    Returns
    -------
    list[LocalizedDetection]
        The filtered list of detections.
    """
    # Get ODLC boundary
    odlc_boundary: list[Point] = [
        Point(odlc_boundary_point.easting, odlc_boundary_point.northing)
        for odlc_boundary_point in gps_data["object_boundary_utm"]
    ]
    zone_number: int = gps_data["object_boundary_utm"][0].zone_number
    zone_letter: str = gps_data["object_boundary_utm"][0].zone_letter

    detection: ObjectDetection
    filtered_detections: list[LocalizedDetection] = []
    for detection in detections:
        # Convert to LocalizedDetection
        image_name: str = detection.image.split("/")[-1]
        image_parameters: consts.CameraParameters = parameters[image_name]
        localized: LocalizedDetection | None = localize_detection(
            detection, image_parameters
        )
        if localized is None:
            continue

        # Check if in bounds
        easting: float
        northing: float
        easting, northing, _, _ = utm.from_latlon(
            localized.latitude,
            localized.longitude,
            force_zone_number=zone_number,
            force_zone_letter=zone_letter,
        )

        if not Point(easting, northing).is_inside_shape(odlc_boundary):
            logger.debug(
                f"Detection {detection} is outside the ODLC boundary, dropping"
            )
            continue

        filtered_detections.append(localized)
    return filtered_detections


def proximity_check(
    detections: list[LocalizedDetection],
    min_distance: float = 30.0,  # Objects will be at least 30 meters apart, as per RN Discord
) -> list[LocalizedDetection]:
    """
    Checks for detections that are too close to each other, and keeps the
    one closest to the center of its image, since that detection has the
    most accurate deskewed location.
    The ruleset states that objects must be at least 50 feet apart.

    Parameters
    ----------
    detections : list[LocalizedDetection]
        A list with all object detections.
    min_distance : float, optional
        The minimum distance between objects in METERS, by default 30.0

    Returns
    -------
    filtered_detections : list[LocalizedDetection]
        All detections that are at least `min_distance` apart, along with
        their localized detections.
    """
    filtered_detections: list[LocalizedDetection] = []

    # If we sort the detections by center offset ascending, the first
    # detection encountered in any cluster of nearby detections is
    # guaranteed to be the one with the lowest offset, so we can stop
    # comparing against a cluster as soon as we find a collision
    detections.sort(key=get_center_offset)

    for detection in detections:
        collided: bool = False
        existing_detection: LocalizedDetection
        for existing_detection in filtered_detections:
            if existing_detection.category != detection.category:
                continue

            distance: float = calculate_distance(
                detection.latitude,
                detection.longitude,
                0,
                existing_detection.latitude,
                existing_detection.longitude,
                0,
            )
            if distance < min_distance:
                collided = True
                break

        if not collided:
            filtered_detections.append(detection)

    return filtered_detections
