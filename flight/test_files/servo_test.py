"""Contains a function for testing the servos on the drone."""

import asyncio
import logging

from state_machine.drone import Drone
from state_machine.flight_settings import SimMode


async def run_servo_test(servo_id: int) -> None:
    """
    Run the servo test, opening and closing the specified servo.

    Parameters
    ----------
    servo_id : int
        The id of the servo to test. This should be from 1 to 4,
        and matches with the AUX port that the servo is connected to.
    """
    drone: Drone = Drone()
    drone.use_settings(SimMode.REAL)
    await drone.connect_drone()

    # connect to the drone
    logging.info("Waiting for drone to connect...")
    while not drone.is_connected:
        await asyncio.sleep(1)

    logging.info("Drone discovered!")

    await drone.open_servo(servo_id)
    await asyncio.sleep(5)
    await drone.close_servo(servo_id)
    return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_servo_test(1))
