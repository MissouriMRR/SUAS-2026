"""Does a simple connection test to make sure the computer can connect to the drone."""

import asyncio
import logging

from state_machine.drone import Drone
from state_machine.flight_settings import FlightSettings


async def run_test(flight_settings: FlightSettings) -> None:
    """
    Run the state machine.

    Parameters
    ----------
    flight_settings : FlightSettings
        The flight settings to use.
    """
    drone: Drone = Drone()
    drone.use_settings(flight_settings.sim_mode)
    await drone.connect_drone()

    # connect to the drone
    logging.info("Waiting for drone to connect...")
    while not drone.is_connected:
        await asyncio.sleep(1)

    logging.info("Drone discovered!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_test(FlightSettings.from_mission_config()))
