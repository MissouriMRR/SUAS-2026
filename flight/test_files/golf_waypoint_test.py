"""
Tests moving to waypoints over the golf course.
"""

import asyncio
import logging
import sys

from flight.waypoint.goto import move_to
from state_machine.drone import Drone
from state_machine.flight_settings import FlightSettings

WAYPOINT_TOLERANCE: int = 6  #


# duplicate code disabled for testing function
# pylint: disable=duplicate-code
async def run(flight_settings: FlightSettings) -> None:
    """
    This function is a driver to test the move_to function and runs through the
    given waypoints in the lats and longs lists at the altitude of 15 m.

    Parameters
    ----------
    flight_settings : FlightSettings
        The flight settings to use.

    Notes
    -----
    Currently has 4 values in each the Lats and Longs array and code is looped
    and will stay in that loop until the drone has reached each of locations
    specified by the latitude and longitude and continues to run until forced disconnect
    """
    # Put all latitudes, longitudes and altitudes into separate arrays
    lats: list[float] = [37.948658, 37.948200, 37.948358, 37.948800]
    longs: list[float] = [-91.784431, -91.783406, -91.783253, -91.784169]

    # create a drone object
    drone: Drone = Drone()
    if flight_settings.sim_flag:
        drone.use_sim_settings()
    else:
        drone.use_real_settings()

    await drone.connect_drone()

    # initilize drone configurations
    drone.vehicle.airspeed = 10

    await drone.arm()

    await drone.takeoff(15)

    # move to each waypoint in mission
    for i in range(2):
        logging.info("Starting loop %s", i)
        for point in range(len(lats)):
            await move_to(drone.vehicle, lats[point], longs[point], 15)

    # return home
    logging.info("Last waypoint reached")
    await drone.return_to_launch()
    print("Staying connected, press Ctrl-C to exit")

    # infinite loop till forced disconnect
    while True:
        await asyncio.sleep(1)


# Runs through the code until it has looped through each element of
# the Lats and Longs array and the drone has arrived at each of them
if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run(FlightSettings.from_mission_config()))
    except KeyboardInterrupt:
        print("Program ended")
        sys.exit(0)
