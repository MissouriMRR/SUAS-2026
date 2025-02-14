"""
File for the airdrop unit test
"""

import asyncio
import logging

from state_machine.flight_settings import FlightSettings
from state_machine.state_machine import StateMachine
from state_machine.states import Airdrop
from state_machine.drone import Drone


async def run() -> None:
    """
    Runs the Airdrop unit test
    """

    # create a drone object
    drone: Drone = Drone()
    drone.use_sim_settings()
    await drone.connect_drone()

    # initilize drone configurations
    drone.vehicle.airspeed = 20

    await drone.arm()

    await drone.takeoff(12)

    flight_settings: FlightSettings = FlightSettings(path_data_path="flight/data/golf_data.json")

    logging.info("starting airdrop")

    await airdrop_run(drone, flight_settings)

    try:
        # tell machine to sleep to prevent constant polling, preventing battery drain
        await asyncio.sleep(1)

        logging.info("Done!")
    except KeyboardInterrupt:
        logging.critical("Keyboard interrupt detected. Killing state machine and landing drone.")
    finally:
        print("Done")


async def airdrop_run(drone: Drone, flight_settings: FlightSettings) -> None:
    """
    Starts airdrop state of statemachine

    Parameters
    ----------
    drone: Drone
        drone class that includes drone object

    flight_settings: FlightSettings
        settings for flight to be passed into the statemachine
    """
    drone.odlc_scan = False
    await StateMachine(Airdrop(drone, flight_settings), drone, flight_settings).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())
