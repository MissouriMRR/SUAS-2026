"""This module tests the mapping state."""

import asyncio
import logging

from state_machine.drone import Drone
from state_machine.flight_settings import FlightSettings
from state_machine.state_machine import StateMachine
from state_machine.states.start import Start


async def run_test(flight_settings: FlightSettings) -> None:
    """
    Initialize and run the flight manager and waypoint check for testing
    the state machine in either simulated or real-world mode.

    Parameters
    ----------
    flight_settings : FlightSettings
        The flight settings to use.
    """

    flight_settings.skip_waypoint = True
    flight_settings.skip_odlc_and_airdrop = True

    drone: Drone = Drone()
    drone.use_settings(flight_settings.sim_mode)
    await drone.connect_drone()

    state_task: asyncio.Task[None] = asyncio.ensure_future(
        StateMachine(Start(drone, flight_settings), drone, flight_settings).run()
    )

    while not state_task.done():
        await asyncio.sleep(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_test(FlightSettings.from_mission_config()))
