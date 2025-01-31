"""File to test the kill switch functionality of the state machine."""

import asyncio
import logging

from state_machine.flight_manager import FlightManager
from state_machine.drone import Drone


async def run_flight_code() -> None:
    """Run flight code to hold the drone in mid air and log the flight mode."""
    logging.info("Starting state machine")
    drone: Drone = Drone()
    drone.use_sim_settings()
    await drone.connect_drone()

    await drone.arm()

    await drone.takeoff(12)

    await asyncio.sleep(5)
    logging.info("Holding position. Test the kill switch now.")

    while True:
        await asyncio.sleep(1)


async def start_1() -> None:
    """Start the flight code in async."""
    asyncio.run(run_flight_code())


async def start_2(flight_process: asyncio.Task[None]) -> None:
    """Start the kill switch in async.

    Args:
        flight_process (Process): The process running the flight code.
    """
    flight_manager: FlightManager = FlightManager()
    flight_manager.drone.use_sim_settings()
    asyncio.run(FlightManager().kill_switch(flight_process))


async def start_test() -> None:
    """Start the unit test."""
    logging.basicConfig(level=logging.INFO)

    flight_manager_task: asyncio.Task[None] = asyncio.ensure_future(start_1())
    asyncio.ensure_future(start_2(flight_manager_task))


if __name__ == "__main__":
    asyncio.run(start_test())
