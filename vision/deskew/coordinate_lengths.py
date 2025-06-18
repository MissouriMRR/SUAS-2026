"""Functions for calculating coordinate degree lengths"""

import numpy as np
from vision.common.constants import FEET_PER_METER, Point, CameraParameters
from vision.deskew import vector_utils


def latitude_length(latitude_deg: float) -> float:
    """
    Returns the distance in feet of one degree of latitude at a particular latitude

    Parameters
    ---------
    latitude_deg : float
        The latitude in degrees

    Returns
    -------
    latitude_length
        The length of a degree of latitude in feet at the given latitude

    References
    ----------
    https://en.wikipedia.org/wiki/Geographic_coordinate_system#Length_of_a_degree
    """

    # Convert to radians for trig functions
    latitude_rad: float = np.deg2rad(latitude_deg)

    # Formula is adapted from the referenced Wikipedia page
    distance: float = (
        111132.92
        - 559.82 * np.cos(2 * latitude_rad)
        + 1.175 * np.cos(4 * latitude_rad)
        - 0.0023 * np.cos(6 * latitude_rad)
    ) * FEET_PER_METER

    return distance


def longitude_length(latitude_deg: float) -> float:
    """
    Calculates the distance in feet of one degree of longitude at that latitude

    Parameters
    ---------
    latitude_deg : float
        The latitude in degrees

    Returns
    -------
    longitude_length
        The length of a degree of longitude in feet at the given latitude

    References
    ----------
    https://en.wikipedia.org/wiki/Geographic_coordinate_system#Length_of_a_degree
    """

    # Convert degrees to radians for trig functions
    latitude_rad: float = np.deg2rad(latitude_deg)

    # Formula is adapted from the referenced Wikipedia page
    distance: float = (
        111412.84 * np.cos(latitude_rad)
        - 93.5 * np.cos(3 * latitude_rad)
        + 0.118 * np.cos(5 * latitude_rad)
    ) * FEET_PER_METER

    return distance

def get_coordinates(
    pixel: tuple[int, int],
    image_shape: tuple[int, int, int] | tuple[int, int],
    camera_parameters: CameraParameters,
) -> tuple[float, float] | None:
    """
    Calculates the coordinates of the given pixel.
    Returns None if there is no valid intersect.

    Parameters
    ----------
    pixel: tuple[int, int]
        The coordinates of the pixel in [X, Y] form
    image_shape : tuple[int, int, int] | tuple[int, int]
        The shape of the image (returned by `image.shape` when image is a numpy image array)
    camera_parameters: CameraParameters
        The details on how and where the photo was taken
        rotation_deg: list[float]
            The rotation of the drone in degrees. The constant ROTATION_OFFSET of the
            camera, stored in constants.py, will be applied first
        drone_coordinates: list[float]
            The coordinates of the drone in degrees of (latitude, longitude)
        altitude: float
            The altitude of the drone in meters

    Returns
    -------
    pixel_coordinates : tuple[float, float] | None
        The (latitude, longitude) coordinates of the pixel in degrees.
        Equal to None if there is no valid intersect.
    """

    # both should be 0 due to latitude

    # Calculate the latitude and longitude lengths (in feet)
    latitude_length_lat: float = latitude_length(
        camera_parameters["drone_coordinates"][0]
    )
    longitude_length_long: float = longitude_length(
        camera_parameters["drone_coordinates"][0]
    )

    print("coordinates", latitude_length_lat, longitude_length_long)

    altitude_m: float = camera_parameters["altitude"]

    # Find the pixel's intersect with the ground to get the location relative to the drone
    intersect: Point | None = vector_utils.pixel_intersect(
        pixel,
        image_shape,
        camera_parameters["rotation_deg"],
        altitude_m,
    )

    if intersect is None:
        print(f"pixel {pixel}")
        print(f"image_shape {image_shape}")
        print(f"camera_parameters" + str(camera_parameters["rotation_deg"]))
        print(f"altitude_m {altitude_m}")
        return

    # Invert the X axis so that the longitude is correct
    intersect[1] *= -1
    print("i need this", intersect)

    # Convert the location to latitude and longitude and add it to the drone's coordinates
    pixel_lat: float = camera_parameters["drone_coordinates"][0] + (intersect[0]*FEET_PER_METER) / latitude_length_lat
    pixel_lon: float = camera_parameters["drone_coordinates"][1] + (intersect[1]*FEET_PER_METER) / longitude_length_long

    return pixel_lat, pixel_lon