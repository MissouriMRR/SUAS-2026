"""File to test the kill switch functionality of the state machine."""

import asyncio
import logging

from state_machine.drone import Drone
from state_machine.flight_manager import FlightManager
from state_machine.flight_settings import FlightSettings


async def run_flight_code(flight_settings: FlightSettings) -> None:
    """Run flight code to hold the drone in mid air and log the flight mode.

    Parameters
    ----------
    flight_settings : FlightSettings
        The flight settings to use.
    """
    logging.info("Starting state machine")
    drone: Drone = Drone()
    if flight_settings.sim_flag:
        drone.use_sim_settings()
    else:
        drone.use_real_settings()
    await drone.connect_drone()

    await drone.arm()

    await drone.takeoff(12)

    await asyncio.sleep(5)
    logging.info("Holding position. Test the kill switch now.")

    while True:
        await asyncio.sleep(1)


async def start_flight() -> None:
    """Start the flight code in async."""
    await run_flight_code(FlightSettings.from_mission_config())


async def start_kill_switch(flight_task: asyncio.Task[None]) -> None:
    """Start the kill switch in async.

    Parameters
    ----------
    flight_process : Process
        The process running the flight code.
    """
    for countdown in range(20, 0, -1):
        logging.info("Activating the kill switch in %d...", countdown)
        await asyncio.sleep(1)

    flight_manager: FlightManager = FlightManager()
    if FlightSettings.from_mission_config().sim_flag:
        flight_manager.drone.use_sim_settings()
    else:
        flight_manager.drone.use_real_settings()

    await flight_manager.kill_switch(flight_task)


async def start_test() -> None:
    """Start the unit test."""
    logging.basicConfig(level=logging.INFO)

    flight_task: asyncio.Task[None] = asyncio.ensure_future(start_flight())
    kill_switch_task: asyncio.Task[None] = asyncio.ensure_future(start_kill_switch(flight_task))

    while not kill_switch_task.done():
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(start_test())
