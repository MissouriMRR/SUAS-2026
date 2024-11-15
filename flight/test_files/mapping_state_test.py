"""This module tests the mapping state."""

import asyncio
import logging
import sys

from state_machine.drone import Drone
from state_machine.flight_settings import FlightSettings
from state_machine.state_machine import StateMachine
from state_machine.states.start import Start


async def run_test(_sim: bool) -> None:  # Temporary fix for unused variable
    """
    Initialize and run the flight manager and waypoint check for testing
    the state machine in either simulated or real-world mode.

    Parameters
    ----------
    _sim : bool
        Specifies whether to run the state machine in simulation mode.
    """
    # Output logging info to stdout
    logging.basicConfig(filename="/dev/stdout", level=logging.INFO)

    path_data_path: str = "flight/data/golf_data.json"

    drone: Drone = Drone()
    if _sim:
        drone.use_sim_settings()
    else:
        drone.use_real_settings()

    flight_settings: FlightSettings = FlightSettings(
        sim_flag=_sim,
        skip_waypoint=True,
        skip_odlc_and_airdrop=True,
        path_data_path=path_data_path,
    )
    await drone.connect_drone()

    state_task: asyncio.Task[None] = asyncio.ensure_future(
        StateMachine(Start(drone, flight_settings), drone, flight_settings).run()
    )

    while not state_task.done():
        await asyncio.sleep(1)


if __name__ == "__main__":
    print("Pass argument --sim to enable the simulation flag.")
    print()
    asyncio.run(run_test("--sim" in sys.argv))
