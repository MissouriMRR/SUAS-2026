"""
Contains the extract_gps() function for extracting data out of
the provided waypoint data JSON file for the SUAS competition.
"""

import argparse
import json
from typing import Any, NamedTuple

import utm
from typing_extensions import TypedDict

from vision.common.constants import Location


# Initialize namedtuples to store latitude/longitude/altitude data for provided points
class Waypoint(NamedTuple):
    """
    NamedTuple storing the data for a single waypoint.

    Attributes
    ----------
    latitude : float
        The latitude of the waypoint.
    longitude : float
        The longitude of the waypoint.
    altitude : float
        The altitude of the waypoint, in meters
    """

    latitude: float
    longitude: float
    altitude: float


class WaypointUtm(NamedTuple):
    """
    NamedTuple storing the data for a single waypoint using UTM coordinates.

    Attributes
    ----------
    easting : float
        The easting of the waypoint.
    northing : float
        The northing of the waypoint.
    zone_number : int
        The zone number of the waypoint.
    zone_letter : str
        The zone letter of the waypoint.
    altitude : float
        The altitude of the waypoint.
    """

    easting: float
    northing: float
    zone_number: int
    zone_letter: str
    altitude: float


class BoundaryPoint(NamedTuple):
    """
    NamedTuple storing the data for a single boundary point.

    Attributes
    ----------
    latitude : float
        The latitude of the boundary point.
    longitude : float
        The longitude of the boundary point.
    """

    latitude: float
    longitude: float


class BoundaryPointUtm(NamedTuple):
    """
    NamedTuple storing the data for a single boundary point using UTM coordinates.

    Attributes
    ----------
    easting : float
        The easting of the boundary point.
    northing : float
        The northing of the boundary point.
    zone_number : int
        The zone number of the boundary point.
    zone_letter : str
        The zone letter of the boundary point.
    """

    easting: float
    northing: float
    zone_number: int
    zone_letter: str


# Initialize GPSData object for sending all data from the file in a single dict
GPSData = TypedDict(
    "GPSData",
    {
        "waypoints": list[Waypoint],
        "waypoints_utm": list[WaypointUtm],
        "boundary_points": list[BoundaryPoint],
        "boundary_points_utm": list[BoundaryPointUtm],
        "object_boundary": list[BoundaryPoint],
        "object_boundary_utm": list[BoundaryPointUtm],
        "altitude_limits": list[float],
        "scan_altitude": float,
        "scan_heading": float,
        "default_airdrop_points": list[Location],
        "airdrop_altitude": float,
    },
)


def format_waypoints(
    json_data: dict[str, Any], fz_num: int, fz_letter: str
) -> tuple[list[Waypoint], list[WaypointUtm]]:
    """Store the lat/lon/altitude for each point into the Waypoints/BoundaryPoint namedtuple
    Appends each point into a list to be able to packed into the output

    Parameters
    ----------
    json_data : dict[str, Any]
        The JSON data from the file
    fz_num : int
        Forced zone number
    fz_letter : str
        Forced zone letter

    Returns
    -------
    tuple[list[Waypoint], list[WaypointUtm]]
        waypoints : list[Waypoint]
            The waypoints in lat/lon/alt format
        waypoints_utm : list[WaypointUtm]
            The waypoints in UTM format
    """
    waypoints: list[Waypoint] = []
    waypoints_utm: list[WaypointUtm] = []
    waypoint: dict[str, float]
    for waypoint in json_data["waypoints"]:
        latitude: float = waypoint["latitude"]
        longitude: float = waypoint["longitude"]
        altitude: float = waypoint["altitude"]

        waypoints.append(Waypoint(latitude, longitude, altitude))
        easting, northing, zone_number, zone_letter = utm.from_latlon(
            latitude, longitude, fz_num, fz_letter
        )
        assert zone_letter is not None
        full_waypoint_utm: WaypointUtm = WaypointUtm(
            easting, northing, zone_number, zone_letter, altitude
        )
        waypoints_utm.append(full_waypoint_utm)

    return waypoints, waypoints_utm


def extract_gps(path: str) -> GPSData:
    """
    Returns the waypoints, boundary points, and altitude limits from a waypoint data file.

    Parameters
    ----------
    path : str
        File path to the waypoint data JSON file.

    Returns
    -------
    GPSData : TypedDict[
            list[Waypoint[float, float, float]],
            list[WaypointUtm[float, float, int, str, float]],
            list[BoundaryPoint[float, float]],
            list[BoundaryPointUtm[float, float, int, str]],
            list[BoundaryPoint[float, float]],
            list[BoundaryPointUtm[float, float, int, str]],
            list[int, int, int],
        ]
        The data in the waypoint data file
        waypoints : list[Waypoint[float, float, float]]
            Waypoint : Waypoint[float, float, float]
                latitude : float
                    The latitude of the waypoint, in degrees.
                longitude : float
                    The longitude of the waypoint, in degrees.
                altitude : float
                    The altitude of the waypoint, in meters.
        waypoints_utm : list[WaypointUtm[float, float, int, str, float]]
            WaypointUtm : WaypointUtm[float, float, int, str, float]
                easting : float
                    The easting of the waypoint, in meters.
                northing : float
                    The northing of the waypoint., in meters
                zone_number : int
                    The zone number of the waypoint.
                zone_letter : str
                    The zone letter of the waypoint.
                altitude : float
                    The altitude of the waypoint, in meters.
        boundary_points : list[BoundaryPoint[float, float]]
            BoundaryPoint : BoundaryPoint[float, float]
                latitude : float
                    The latitude of the boundary point, in degrees.
                longitude : float
                    The longitude of the boundary point, in degrees.
        boundary_points_utm : list[BoundaryPointUtm[float, float, int, str]]
            BoundaryPointUtm : BoundaryPointUtm[float, float, int, str]
                easting : float
                    The easting of the boundary point, in meters.
                northing : float
                    The northing of the boundary point, in meters.
                zone_number : int
                    The zone number of the boundary point.
                zone_letter : str
                    The zone letter of the boundary point.
        object_boundary : list[BoundaryPoint[float, float]]
            BoundaryPoint : BoundaryPoint[float, float]
                latitude : float
                    The latitude of the object boundary point, in degrees.
                longitude : float
                    The longitude of the object boundary point, in degrees.
        object_boundary_utm : list[BoundaryPointUtm[float, float, int, str]]
            BoundaryPointUtm : BoundaryPointUtm[float, float, int, str]
                easting : float
                    The easting of the object boundary point, in meters.
                northing : float
                    The northing of the object boundary point, in meters.
                zone_number : int
                    The zone number of the object boundary point.
                zone_letter : str
                    The zone letter of the object boundary point.
        altitude_limits : list[float, float]
            altitude_min : float
                The minimum altitude that the drone must fly at all times, in meters.
            altitude_max : float
                The maximum altitude that the drone must fly at all times, in meters.
        scan_altitude : float
            The altitude to fly at while scanning the object area, in meters.
        scan_heading : float
            The heading to fly at while scanning the object area, in degrees.
        default_airdrop_points : list[Location]
            The coordinates to perform airdrops at if no objects are detected.
            Location : Location[float, float]
                latitude : float
                    The latitude of the waypoint, in degrees.
                longitude : float
                    The longitude of the waypoint, in degrees.
        airdrop_altitude : float
            The altitude to fly at while performing an airdrop, in meters.

    Raises
    ------
    KeyError
        If the structure of the JSON is incorrect.
    ValueError
        If there are invalid values in the JSON.
    """

    # Load the JSON file as a Python dict to be able to easily access the data
    with open(path, encoding="UTF-8") as data_file:
        json_data: dict[str, Any] = json.load(data_file)

    # Initialize lists to store waypoints & boundary points
    waypoints: list[Waypoint] = []
    waypoints_utm: list[WaypointUtm] = []
    boundary_points: list[BoundaryPoint] = []
    boundary_points_utm: list[BoundaryPointUtm] = []
    object_boundary: list[BoundaryPoint] = []
    object_boundary_utm: list[BoundaryPointUtm] = []

    # Get forced UTM zone number and zone letter
    forced_zone_number: int
    _, _, forced_zone_number, forced_zone_letter = utm.from_latlon(
        json_data["flyzones"]["boundaryPoints"][0]["latitude"],
        json_data["flyzones"]["boundaryPoints"][0]["longitude"],
    )
    assert forced_zone_letter is not None

    waypoints, waypoints_utm = format_waypoints(
        json_data, forced_zone_number, forced_zone_letter
    )

    boundary_point: dict[str, float]
    full_boundary_point_utm: BoundaryPointUtm
    for boundary_point in json_data["flyzones"]["boundaryPoints"]:
        latitude = boundary_point["latitude"]
        longitude = boundary_point["longitude"]

        boundary_points.append(BoundaryPoint(latitude, longitude))
        easting, northing, zone_number, zone_letter = utm.from_latlon(
            latitude, longitude, forced_zone_number, forced_zone_letter
        )
        assert zone_letter is not None
        full_boundary_point_utm = BoundaryPointUtm(
            easting, northing, zone_number, zone_letter
        )
        boundary_points_utm.append(full_boundary_point_utm)

    for boundary_point in json_data["flyzones"]["searchBoundary"]:
        latitude = boundary_point["latitude"]
        longitude = boundary_point["longitude"]

        object_boundary.append(BoundaryPoint(latitude, longitude))
        easting, northing, zone_number, zone_letter = utm.from_latlon(
            latitude, longitude, forced_zone_number, forced_zone_letter
        )
        assert zone_letter is not None
        full_boundary_point_utm = BoundaryPointUtm(
            easting, northing, zone_number, zone_letter
        )
        object_boundary_utm.append(full_boundary_point_utm)

    if len(object_boundary) != 4:
        raise ValueError("the object boundary must have exactly 4 points")

    # Package all data into the GPSData TypedDict to be exported
    waypoint_data: GPSData = {
        "waypoints": waypoints,
        "waypoints_utm": waypoints_utm,
        "boundary_points": boundary_points,
        "boundary_points_utm": boundary_points_utm,
        "object_boundary": object_boundary,
        "object_boundary_utm": object_boundary_utm,
        "altitude_limits": [
            json_data["flyzones"]["altitudeMin"],
            json_data["flyzones"]["altitudeMax"],
        ],
        "scan_altitude": json_data["scanAltitude"],
        "scan_heading": json_data["scanHeading"],
        "default_airdrop_points": json_data["defaultAirdropPoints"],
        "airdrop_altitude": json_data["airdropAltitude"],
    }
    return waypoint_data


# If run on its own, use the default data location
if __name__ == "__main__":
    # Read file to be used as the data file using the -file argument
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-file")
    args: argparse.Namespace = parser.parse_args()

    extract_gps(vars(args)["file"])
