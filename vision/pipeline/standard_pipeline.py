"""Functions that perform standard object detection, localization, and classification"""

from typing import TypeAlias

import numpy as np

from nptyping import NDArray, Shape, UInt8, Float32
import vision.common.constants as consts

from vision.common.crop import crop_image
from vision.competition_inputs.bottle_reader import BottleData
from vision.common.bounding_box import BoundingBox
from vision.common.odlc_characteristics import ODLCColor

from vision.standard_object.odlc_contour_detection import fetch_shape_contours
from vision.standard_object.odlc_classify_shape import process_shapes
from vision.standard_object.odlc_text_detection import get_odlc_text
from vision.standard_object.odlc_colors import find_colors

import vision.pipeline.pipeline_utils as pipe_utils

ContourHeirarchyList: TypeAlias = list[tuple[tuple[consts.Contour, ...], consts.Hierarchy]]

# The various thresholds to run the image processing at
PROCESSING_THRESHOLDS: list[tuple[int, int]] = [(0, 50), (25, 150), (50, 250), (75, 350)]


def create_odlc_dict(sorted_odlcs: list[list[BoundingBox]]) -> consts.ODLCDict:
    """
    Creates the ODLC_Dict dictionary from a list of shape bounding boxes

    Parameters
    ----------
    sorted_odlcs: list[list[BoundingBox]]
        The list of sightings of each object, matched to bottles

    Returns
    -------
    odlc_dict: consts.ODLC_Dict
        The dictionary of ODLCs matching the output format
    """

    odlc_dict: consts.ODLCDict = {}

    i: int
    bottle: list[BoundingBox]
    for i, bottle in enumerate(sorted_odlcs):
        coords_list: list[tuple[float, float]] = []

        shape: BoundingBox
        for shape in bottle:
            coords_list.append((shape.center_lat_lon[0], shape.center_lat_lon[1]))

        if len(bottle) > 0:
            coords_array: NDArray[Shape["*, 2"], Float32] = np.array(coords_list)

            average_coord: NDArray[Shape["2"], Float32] = np.average(coords_array, axis=0)

            odlc_dict[str(i)] = {"latitude": average_coord[0], "longitude": average_coord[1]}

    return odlc_dict
