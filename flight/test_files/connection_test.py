"""Does a simple connection test to make sure the computer can connect to the drone."""

import asyncio
import logging
import sys

from state_machine.drone import Drone

SIM_ADDR: str = "udp://:14540"  # Address to connect to the ardupilot simulator
AIRSIM_ADDR: str = "udp://:14030"  # Address to connect to the airsim simulator
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
    else:
        drone.use_real_settings()
    if sim:
        address: str = SIM_ADDR
    elif airsim:
        address = AIRSIM_ADDR
    else:
        address = CONTROLLER_ADDR
    drone: Drone = Drone(address)
    await drone.connect_drone()

    # connect to the drone
    logging.info("Waiting for drone to connect...")
    while not drone.is_connected:
        await asyncio.sleep(1)

    logging.info("Drone discovered!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_test("--sim" in sys.argv, "--airsim" in sys.argv))
