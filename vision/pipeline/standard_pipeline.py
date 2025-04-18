"""Functions that perform standard object detection, localization, and classification"""

import heapq
from typing import Iterable, TypeAlias

import utm

from flight.extract_gps import extract_gps, GPSData
from flight.waypoint.geometry import Point

from state_machine.flight_settings import FlightSettings

import vision.common.constants as consts

from vision.common.bounding_box import BoundingBox

from vision.standard_object.odlc_contour_detection import fetch_shape_contours
from vision.standard_object.odlc_classify_shape import process_shapes

import vision.pipeline.pipeline_utils as pipe_utils
from vision.yolo.model import ObjectDetection

ContourHierarchyList: TypeAlias = list[tuple[tuple[consts.Contour, ...], consts.Hierarchy]]

# We only care about the object classes that will actually appear in the competition
# You can see which are which here: https://github.com/WongKinYiu/yolov9/blob/main/data/coco.yaml
CLASS_PRIORITIES: dict[str, float] = {
    "person": 0.5,
    "car": 1.0,
    "motorcycle": 1.0,
    "airplane": 1.0,
    "bus": 1.0,
    "boat": 1.0,
    "stop sign": 1.0,
    "umbrella": 1.0,
    "suitcase": 1.0,
    "skis": 1.0,
    "snowboard": 1.0,
    "sports ball": 1.0,
    "baseball bat": 1.0,
    "tennis racket": 1.0,
    "bed": 1.0,
}


def find_standard_objects(
    original_image: consts.Image,
    camera_parameters: consts.CameraParameters,
    image_path: str,
) -> list[BoundingBox]:
    """
    Finds all bounding boxes of standard objects in an image

    Parameters
    ----------
    original_image: Image
        The image to find shapes in
    camera_parameters: CameraParameters
        The details of how and where the photo was taken
    image_path: str
        The path for the image the bounding box is from

    Returns
    -------
    found_odlcs: list[BoundingBox]
        The list of bounding boxes of detected standard objects
    """

    found_odlcs: list[BoundingBox] = []
    contours: list[consts.Contour] = fetch_shape_contours(original_image, True, "contours.jpg")
    shapes: list[BoundingBox] = process_shapes(contours)
    shape: BoundingBox
    for shape in shapes:
        # Set the shape attributes by reference. If successful, keep the shape
        if pipe_utils.set_generic_attributes(shape, original_image.shape, camera_parameters):
            found_odlcs.append(shape)

    return found_odlcs


def create_odlc_dict(
    bounding_boxes: Iterable[BoundingBox], flight_settings: FlightSettings
) -> consts.ODLCDict:
    """
    Creates the ODLCDict dictionary from a list of shape bounding boxes.
    Discards bounding boxes whose center is not inside the airdrop boundary.

    Parameters
    ----------
    bounding_boxes : Iterable[BoundingBox]
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
        for odlc_boundary_point in gps_data["odlc_boundary_utm"]
    ]
    zone_number: int = gps_data["odlc_boundary_utm"][0].zone_number
    zone_letter: str = gps_data["odlc_boundary_utm"][0].zone_letter

    odlc_dict: consts.ODLCDict = {}

    bbox: BoundingBox
    for bbox in bounding_boxes:
        # Check if in bounds
        easting: float
        northing: float
        easting, northing, _, _ = utm.from_latlon(
            bbox.get_attribute("latitude"),
            bbox.get_attribute("longitude"),
            force_zone_number=zone_number,
            force_zone_letter=zone_letter,
        )
        if not Point(easting, northing).is_inside_shape(odlc_boundary):
            continue

        odlc_dict[str(index)] = {
            "latitude": bbox.center_lat_lon[0],
            "longitude": bbox.center_lat_lon[1],
        }
    return odlc_dict


def filter_objects(
    detections: dict[str, ObjectDetection],
    expand_categories: bool = False,
    buffer: float = 0.0,
    output_count: int = 4,
) -> dict[str, ObjectDetection]:
    """
    Filters out objects to the best 4 detections.
    Only needs to be called if there are more than 4 detections.
    You can enable expand_categories to allow categories that are not in the competition
    set, and use buffer to set a higher priority for categories that are.
    The buffer will be added to the confidence value of the categories in the set.

    Parameters
    ----------
    detections : dict[str, ObjectDetection]
        The dictionary of detections to filter
    expand_categories : bool, default False
        Whether to include categories not in the competition set
    buffer : float, default 0.0
        The value to add to confidence values of categories in the competition set
    output_count : int, default 4
        The maximum number of detections to output

    Returns
    -------
    filtered_detections: dict[str, ObjectDetection]
        The dictionary of filtered detections
    """
    best_detections: list[tuple[float, str, ObjectDetection]] = []  # min heap

    # Get the entries with the highest confidence
    category: str
    detection: ObjectDetection
    for category, detection in detections.items():
        confidence: float = detection.confidence
        if category in CLASS_PRIORITIES:
            confidence += buffer * CLASS_PRIORITIES[category]
        elif not expand_categories:
            continue

        heap_entry: tuple[float, str, ObjectDetection] = confidence, category, detection
        if len(best_detections) == output_count:
            heapq.heappushpop(best_detections, heap_entry)
        else:
            heapq.heappush(best_detections, heap_entry)

    # Sort by confidence from highest to lowest (dicts maintain insertion order)
    best_detections.sort(key=lambda entry: -entry[0])
    return {entry[1]: entry[2] for entry in best_detections}
