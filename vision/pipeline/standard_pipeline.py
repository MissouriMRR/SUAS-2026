"""Functions that perform standard object detection, localization, and classification"""

from typing import TypeAlias

import vision.common.constants as consts

from vision.common.crop import crop_image
from vision.common.bounding_box import BoundingBox
from vision.common.odlc_characteristics import ODLCColor

from vision.standard_object.odlc_contour_detection import fetch_shape_contours
from vision.standard_object.odlc_classify_shape import process_shapes
from vision.standard_object.odlc_text_detection import get_odlc_text
from vision.standard_object.odlc_colors import find_colors

import vision.pipeline.pipeline_utils as pipe_utils
from vision.yolov9.model import ObjectDetection

ContourHeirarchyList: TypeAlias = list[tuple[tuple[consts.Contour, ...], consts.Hierarchy]]

# The various thresholds to run the image processing at
PROCESSING_THRESHOLDS: list[tuple[int, int]] = [
    (0, 50),
    (25, 150),
    (50, 250),
    (75, 350),
]


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
        if set_shape_attributes(shape, original_image) and pipe_utils.set_generic_attributes(
            shape, image_path, original_image.shape, camera_parameters
        ):
            found_odlcs.append(shape)

    return found_odlcs


def set_shape_attributes(
    shape: BoundingBox,
    original_image: consts.Image,
) -> bool:
    """
    Gets the attributes of a shape returned from process_shapes()
    Modifies `shape` in place

    Parameters
    ----------
    shape: BoundingBox
        The bounding box of the shape. Attribute "shape" must be set
    original_image: Image
        The image used to get the details for each shape

    Returns
    -------
    attributes_found: bool
        Returns true if all attributes were successfully found
    """

    if shape.get_attribute("shape") is None:
        return False

    odlc_img: consts.Image = crop_image(original_image, shape)

    text_bounding: BoundingBox = get_odlc_text(odlc_img)

    shape_color: ODLCColor
    text_color: ODLCColor

    if not text_bounding.get_attribute("text"):
        # No text was found, we can only get the shape color
        _, shape_color = find_colors(odlc_img)
        shape.set_attribute("shape_color", shape_color)
    else:
        # Text found, we can try to look for both colors
        shape.set_attribute("text", text_bounding.get_attribute("text"))
        text_img: consts.Image = crop_image(odlc_img, text_bounding)
        shape_color, text_color = find_colors(text_img)

        shape.set_attribute("shape_color", shape_color)
        shape.set_attribute("text_color", text_color)

    return True


def create_odlc_dict(bounding_boxes: list[BoundingBox]) -> consts.ODLCDict:
    """
    Creates the ODLC_Dict dictionary from a list of shape bounding boxes

    Parameters
    ----------
    sorted_odlcs: list[BoundingBox]
        The list of sightings of each object, matched to bottles

    Returns
    -------
    odlc_dict: consts.ODLC_Dict
        The dictionary of ODLCs matching the output format
    """

    odlc_dict: consts.ODLCDict = {}

    for bbox in bounding_boxes:
        odlc_dict[bbox.obj_type] = {
            "latitude": bbox.get_attribute("latitude"),
            "longitude": bbox.get_attribute("longitude"),
        }

    return odlc_dict


def filter_objects(
    detections: dict[str, ObjectDetection],
) -> dict[str, ObjectDetection]:
    """
    Filters out objects to the best 4 detections

    Parameters
    ----------
    detections: dict[str, ObjectDetection]
        The dictionary of detections to filter

    Returns
    -------
    filtered_detections: dict[str, ObjectDetection]
        The dictionary of filtered detections
    """
    # This might seem strange, but we want to avoid false judge detections
    if "person" in detections:
        detections.pop("person")

    sorted_detections = sorted(detections.items(), key=lambda x: x[1].confidence, reverse=True)

    return dict(sorted_detections[:4])
