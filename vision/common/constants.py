"""Constant variables and common type aliases for Vision"""

from typing import TypeAlias, TypedDict
from nptyping import NDArray, Shape, UInt8, Float64, IntC, Bool8

# ordinary three channel image
Image: TypeAlias = NDArray[Shape["*, *, 3"], UInt8]

# image for maps, uses float to store accurate depth data in 4th channel
MapImage: TypeAlias = NDArray[Shape["*, *, 4"], Float64]

# single channel image type
ScImage: TypeAlias = NDArray[Shape["*, *"], UInt8]

# single channel image of booleans
Mask: TypeAlias = NDArray[Shape["*, *"], Bool8]

Point: TypeAlias = NDArray[Shape["2"], Float64]
Vector: TypeAlias = NDArray[Shape["3"], Float64]

# Point representing a pixel index
ImgPoint: TypeAlias = NDArray[Shape["2"], IntC]

# return types for cv2.findContours() -> tuple[tuple[Contour, ...], Hierarchy]
Contour: TypeAlias = NDArray[Shape["*, 1, 2"], IntC]
Hierarchy: TypeAlias = NDArray[Shape["1, *, 4"], IntC]

# Format of Corners is: (top left, top right, bottom right, bottom left), or
#     1--2
#     |  |
#     4--3
Corners: TypeAlias = NDArray[Shape["4, 2"], Float64]


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
ODLCDict: TypeAlias = dict[str, Location]

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

FEET_PER_METER: float = 3.28084