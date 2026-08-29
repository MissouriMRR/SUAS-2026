"""
Tests the waypoint move_to() function.
"""

import asyncio
import logging
import sys

from flight import extract_gps
from flight.waypoint.goto import move_to
from state_machine.drone import Drone
from state_machine.flight_settings import FlightSettings

# Defining altitude and speed
MOVE_TO_TEST_ALTITUDE: int = 12
MOVE_TO_TEST_SPEED: int = 20


async def run(flight_settings: FlightSettings) -> None:
    """
    This function is a driver to test the goto function and runs through the
    given waypoints in the lats and longs lists at the altitude of 100.
    Makes the drone move to each location in the lats and longs arrays at the altitude of 100.

    Parameters
    ----------
    flight_settings : FlightSettings
        The flight settings to use.

    Notes
    -----
    Currently has 3 values in each the Lats and Longs array and code is looped
    and will stay in that loop until the drone has reached each of locations
    specified by the latitude and longitude and
    continues to run until forced disconnect
    """

    # Put all latitudes, longitudes and altitudes into separate arrays
    lats: list[float] = []
    longs: list[float] = []
    altitudes: list[float] = []

    waypoint_data = extract_gps.extract_gps(flight_settings.mission_data_path)
    waypoints = waypoint_data["waypoints"]

    waypoint: tuple[float, float, float]
    for waypoint in waypoints:
        lats.append(waypoint.latitude)
        longs.append(waypoint.longitude)
        altitudes.append(waypoint.altitude)

    # create a drone object
    drone: Drone = Drone()
    drone.use_settings(flight_settings.sim_mode)
    await drone.connect_drone()

    # initilize drone configurations
    drone.vehicle.airspeed = MOVE_TO_TEST_SPEED

    await drone.arm()

    await drone.takeoff(MOVE_TO_TEST_ALTITUDE)

    # move to each waypoint in mission
    point: int
    for point in range(len(lats)):
        await move_to(drone.vehicle, lats[point], longs[point], 100)

    # return home
    logging.info("Last waypoint reached")
    await drone.return_to_launch(100)
    print("Staying connected, press Ctrl-C to exit")

    # infinite loop till forced disconnect
    while True:
        await asyncio.sleep(1)


# Runs through the code until it has looped through each element of
#  the Lats and Longs array and the drone has arrived at each of them
if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run(FlightSettings.from_mission_config()))
    except KeyboardInterrupt:
        logging.info("CTRL+C: Program ended")
        sys.exit(0)
