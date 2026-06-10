"""
A unit test for the waypoint state.

Functions
---------
in_bounds(boundary, latitude, longitude, altitude)
    Checks if a given point is within a specified boundary.

waypoint_check(drone, flight_settings)
    Verifies if a drone reaches each waypoint in a predefined path.

run_test(flight_settings)
    Initializes the state machine and starts the waypoint check.
"""

import asyncio
import logging
import time
from typing import Final

import dronekit

from flight.extract_gps import BoundaryPoint, GPSData, extract_gps
from flight.extract_gps import Waypoint as Waylist
from flight.waypoint.calculate_distance import calculate_distance
from state_machine.drone import Drone
from state_machine.flight_settings import FlightSettings
from state_machine.state_machine import StateMachine
from state_machine.states.start import Start

# 3.28084 feet per meter
CLOSE_THRESHOLD: Final[float] = (
    15 / 3.28084
)  # How close the drone should get to each waypoint, in meters


def in_bounds(
    boundary: list[BoundaryPoint],
    latitude: float,
    longitude: float,
    altitude: float,
    min_altitude: float,
    max_altitude: float,
) -> bool:
    """
    Determines if a point specified by latitude, longitude, and altitude
    is inside a given boundary.

    Parameters
    ----------
    boundary : list of BoundaryPoint
        A list of boundary points defining a closed polygonal area.
    latitude : float
        The latitude of the point to check.
    longitude : float
        The longitude of the point to check.
    altitude : float
        The altitude of the point to check, in meters.
    min_altitude : float
        The minimum allowed altitude, in meters.
    max_altitude : float
        The maximum allowed altitude, in meters.

    Returns
    -------
    bool
        True if the point is inside the boundary, False otherwise.
    """
    if not min_altitude <= altitude <= max_altitude:
        return False
    num: int = len(boundary)
    j: int = num - 1
    inside: bool = False

    for i in range(num):
        lat_i: float = boundary[i][0]
        long_i: float = boundary[i][1]
        lat_j: float = boundary[j][0]
        long_j: float = boundary[j][1]

        if ((long_i > longitude) != (long_j > longitude)) and (
            latitude
            < (lat_j - lat_i) * (longitude - long_i) / (long_j - long_i) + lat_i
        ):
            inside = not inside

        j = i

    return inside


async def waypoint_check(drone: Drone, flight_settings: FlightSettings) -> None:
    """
    Checks if the drone reaches each waypoint in a list and remains
    within the specified boundary during its flight.

    Parameters
    ----------
    drone : Drone
        The drone object from the flight manager.
    flight_settings : FlightSettings
        The flight settings to use.
    """
    gps_dict: GPSData = extract_gps(flight_settings.mission_data_path)
    waypoints: list[Waylist] = gps_dict["waypoints"]
    boundary: list[BoundaryPoint] = gps_dict["boundary_points"]
    min_altitude: float = gps_dict["altitude_limits"][0]
    max_altitude: float = gps_dict["altitude_limits"][1]

    # Ensure that the flight manager code starts and sets the correct address.
    # 5 seconds is probably far longer than necessary.
    # Anyway, the drone will probably not have finished taking off after only
    # 5 seconds, so it doesn't matter.
    await asyncio.sleep(5.0)

    # connect to the drone
    while not drone.is_connected:
        await asyncio.sleep(1)

    previously_out_of_bounds: bool = False
    previous_log_time: float = time.perf_counter()  # time.perf_counter() is monotonic
    for waypoint_num, waypoint in enumerate(waypoints):
        while True:
            location: dronekit.LocationGlobalRelative = (
                drone.vehicle.location.global_relative_frame
            )

            # continuously checks current latitude, longitude and altitude of the drone
            drone_lat: float = location.lat
            drone_lon: float = location.lon
            drone_alt: float = location.alt

            # checks if drone's location is within boundary
            if not in_bounds(
                boundary, drone_lat, drone_lon, drone_alt, min_altitude, max_altitude
            ):
                if not previously_out_of_bounds:
                    logging.info("(Waypoint State Test) Out of bounds!")
                    previously_out_of_bounds = True
            else:
                if previously_out_of_bounds:
                    logging.info("(Waypoint State Test) Re-entered bounds.")
                    previously_out_of_bounds = False

            distance_to_waypoint: float = calculate_distance(
                drone_lat, drone_lon, drone_alt, *waypoint
            )

            # accurately checks if location is reached
            if distance_to_waypoint < CLOSE_THRESHOLD:
                break

            curr_time: float = time.perf_counter()
            if curr_time - previous_log_time >= 1.0:
                logging.info(
                    "(Waypoint State Test) %f m to waypoint", distance_to_waypoint
                )
                previous_log_time = curr_time

            await asyncio.sleep(0.1)

        logging.info("(Waypoint State Test) Waypoint %d reached.", waypoint_num)


async def run_test(flight_settings: FlightSettings) -> None:
    """
    Initialize and run the flight manager and waypoint check for testing
    the state machine in either simulated or real-world mode.

    Parameters
    ----------
    flight_settings : FlightSettings
        The flight settings to use.
    """
    drone: Drone = Drone()
    drone.use_settings(flight_settings.sim_mode)

    drone.odlc_scan = False
    await drone.connect_drone()

    state_task: asyncio.Task[None] = asyncio.ensure_future(
        StateMachine(Start(drone, flight_settings), drone, flight_settings).run()
    )
    await waypoint_check(drone, flight_settings)

    while not state_task.done():
        await asyncio.sleep(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_test(FlightSettings.from_mission_config()))
