"""
Test for taking off, holding position for 5 seconds, then landing.
"""

import asyncio
import logging
import sys

from state_machine.drone import Drone
from state_machine.flight_settings import FlightSettings


# duplicate code disabled for testing function
# pylint: disable=duplicate-code
async def run(flight_settings: FlightSettings) -> None:
    """
    This function is a driver to test if the drone can take off to an altitude of 15 m
    and then land.

    Parameters
    ----------
    flight_settings : FlightSettings
        The flight settings to use.

    Notes
    -----
    15m = 49.2126ft
    """

    # create a drone object
    drone: Drone = Drone()
    drone.use_settings(flight_settings.sim_mode)

    await drone.connect_drone()

    # initilize drone configurations
    drone.vehicle.airspeed = 10

    await drone.arm()

    await drone.takeoff(15)

    # wait in air for 5 seconds once at correct height
    logging.info("Reached takeoff altitude. Holding position for 5 seconds")
    await asyncio.sleep(5)

    # return home
    await drone.return_to_launch()
    print("Landed. Staying connected, press Ctrl-C to exit")

    # infinite loop till forced disconnect
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run(FlightSettings.from_mission_config()))
    except KeyboardInterrupt:
        print("Program ended")
        sys.exit(0)
