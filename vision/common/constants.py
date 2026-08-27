"""Constant variables and common type aliases for Vision"""

from typing import Annotated, TypedDict

import numpy as np
from numpy.typing import NDArray

# The shape annotations below are documentation only, no way to enforce
type Image = Annotated[NDArray[np.uint8], "(height, width, 3)"]

type Point = Annotated[NDArray[np.float64], "(2,)"]
type Vector = Annotated[NDArray[np.float64], "(3,)"]

# Format of Corners is: (top left, top right, bottom right, bottom left), or
#     1--2
#     |  |
#     4--3
type Corners = Annotated[NDArray[np.float64], "(4, 2)"]

type ImageShape = tuple[int, int] | tuple[int, int, int]


class Location(TypedDict):
    """
    A saved coordinate location - primarily used for ODLCs

    Attributes
    ----------
    latitude: float
        The latitude in degrees of the location
    longitude: float
        The longitude in degrees of the location
    """

    latitude: float
    longitude: float


# str is the index of the water bottle to drop as a string (because json)
# Location is the coordinates of the standard object once found
type ODLCDict = dict[str, Location]

CameraConfig = TypedDict(
    "CameraConfig",
    {
        "Default": dict[str, float],
        "Airsim": dict[str, float],
        "airsim_flag": bool,
    },
)


class CameraParameters(TypedDict):
    """
    The details on how and where a photo was taken

    Attributes
    ----------
    rotation_deg: list[float]
        The rotation of the drone/camera
    drone_coordinates: list[float]
        The coordinates of the drone in degrees of (latitude, longitude)
    altitude: float
        The altitude of the drone in meters
    """

    rotation_deg: list[float]
    drone_coordinates: list[float]
    altitude: float


# The rotation offset of the camera to the drone. The offset is applied
#   in vision.vector_utils.pixel_intersect()
# In degrees of [roll, pitch, yaw]
# Set to [0.0, -90.0, 0.0] when the camera is facing directly downwards
ROTATION_OFFSET: list[float] = [0.0, -90.0, 90.0]


# Airdrop dataclasses
class AirdropConfig(TypedDict):
    """
    The configuration for an airdrop

    Attributes
    ----------
    servo: int
        The servo number to use for the airdrop
    loaded: bool
        Whether the airdrop is loaded or not
    """

    servo: int
    loaded: bool


AirdropStatus = dict[str, AirdropConfig]
