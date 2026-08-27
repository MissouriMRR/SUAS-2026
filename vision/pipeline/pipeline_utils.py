"""Pipeline functions not specific to either standard or emergent object"""

import json

import vision.common.constants as consts
from vision.common.localized_detection import LocalizedDetection
from vision.deskew.camera_distances import get_coordinates
from vision.object_detection import ObjectDetection


def read_parameter_json(json_path: str) -> dict[str, consts.CameraParameters]:
    """
    Will read in the data from the given json file and return it as a python dict.

    Parameters
    ----------
    json_path : str
        The path of a valid json file, assumed to have data in the same format as return type.

    Returns
    -------
    data : dict[str, CameraParameters]
        The python dict version of the data from the given json file.
    """

    with open(json_path, encoding="utf-8") as jfile:
        data: dict[str, consts.CameraParameters] = json.load(jfile)

    return data


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
