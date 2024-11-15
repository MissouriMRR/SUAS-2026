"""
Test for taking off, holding position for 5 seconds, then landing.
"""

import asyncio
import logging
import sys

from dronekit import VehicleMode

from state_machine.drone import Drone


# duplicate code disabled for testing function
# pylint: disable=duplicate-code
async def run(sim: bool) -> None:
    """
    This function is a driver to test the goto function and runs through the
    given waypoints in the lats and longs lists at the altitude of 100.
    Makes the drone move to each location in the lats and longs arrays
    at the altitude of 100 and

    Notes
    -----
    15m = 49.2126ft
    """

    # create a drone object
    drone: Drone = Drone()
    if sim:
        drone.use_sim_settings()
    else:
        drone.use_real_settings()

    logging.info("Waiting for drone to connect...")
    await drone.connect_drone()
    logging.info("Drone discovered!")

    # initilize drone configurations
    drone.vehicle.airspeed = 10

    # connect to the drone
    logging.info("Waiting for pre-arm checks to pass...")
    while not drone.vehicle.is_armable:
        await asyncio.sleep(0.5)

    logging.info("-- Arming")
    drone.vehicle.mode = VehicleMode("GUIDED")
    drone.vehicle.armed = True
    while drone.vehicle.mode.name != "GUIDED" or not drone.vehicle.armed:
        await asyncio.sleep(0.5)

    logging.info("-- Taking off")
    drone.vehicle.simple_takeoff(15)

    # wait for drone to take off
    while drone.vehicle.location.global_relative_frame.alt < 14.9:
        await asyncio.sleep(1)

    # wait in air for 5 seconds once at correct height
    logging.info("Reached takeoff altitude. Holding position for 5 seconds")
    await asyncio.sleep(5)

    # return home
    logging.info("Returning to home")
    drone.vehicle.mode = VehicleMode("RTL")
    while drone.vehicle.mode.name != "RTL":
        await asyncio.sleep(0.5)
    while drone.vehicle.system_status.state != "STANDBY":
        await asyncio.sleep(0.5)
    print("Landed. Staying connected, press Ctrl-C to exit")

    # infinite loop till forced disconnect
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run("--sim" in sys.argv))
    except KeyboardInterrupt:
        print("Program ended")
        sys.exit(0)
