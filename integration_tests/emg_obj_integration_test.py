"""Runs emergent object code for testing purposes"""

import asyncio
import logging

from state_machine.flight_manager import FlightManager
from state_machine.flight_settings import FlightSettings


async def run_test(flight_settings: FlightSettings) -> None:
    """
    Initialize and run the flight manager for the emergent object
    integration test.

    Parameters
    ----------
    flight_settings : FlightSettings
        The flight settings to use.
    """
    # Output logging info to stdout
    logging.basicConfig(filename="/dev/stdout", level=logging.INFO)

    flight_manager: FlightManager = FlightManager()
    flight_settings.skip_waypoint = True
    flight_settings.standard_object_count = 0
    await flight_manager.run_manager(flight_settings)


if __name__ == "__main__":
    asyncio.run(run_test(FlightSettings.from_mission_config()))
