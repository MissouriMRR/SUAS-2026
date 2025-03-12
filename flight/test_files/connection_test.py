"""Does a simple connection test to make sure the computer can connect to the drone."""

import asyncio
import logging
import sys

from state_machine.drone import Drone

SIM_ADDR: str = "udp:127.0.0.1:14550"  # Address to connect to the ardupilot simulator
AIRSIM_ADDR: str = "tcp:127.0.0.1:5762"  # Address to connect to the airsim simulator
CONTROLLER_ADDR: str = "serial:///dev/ttyFTDI:921600"  # Address to connect to a pixhawk board


async def run_test(sim: bool, airsim: bool) -> None:
    """
    Run the state machine.

    Parameters
    ----------
    sim : bool
        Whether to run the state machine in ardupilot simulation mode.
    airsim : bool
        Whether to run the state machine in airsim simulation mode.
    """
    drone: Drone = Drone()
    if sim:
        drone.use_sim_settings()
    elif airsim:
        drone.use_airsim_settings()
    else:
        drone.use_real_settings()

    await drone.connect_drone()

    # connect to the drone
    logging.info("Waiting for drone to connect...")
    while not drone.is_connected:
        await asyncio.sleep(1)

    logging.info("Drone discovered!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_test("--sim" in sys.argv, "--airsim" in sys.argv))
