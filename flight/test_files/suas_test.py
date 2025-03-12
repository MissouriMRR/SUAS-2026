"""
File for test way point path for SUAS 3 miles in length
"""

import asyncio
import sys


SIM_ADDR: str = "udp://:14540"
AIRSIM_ADDR: str = "tcp:127.0.0.1:5762"
CON_ADDR: str = "serial:///dev/ttyFTDI:921600"


from flight.waypoint.goto import move_to
from state_machine.drone import Drone


async def run() -> None:
    """
    run simple waypoint flight path
    """

    # create a drone object
    drone: Drone = Drone()
    drone.use_sim_settings()
    await drone.connect_drone()

    # initilize drone configurations
    drone.vehicle.airspeed = 30

    await drone.arm()

    await drone.takeoff(12)

    obj_altitude: float = 12
    points: list[tuple[float, float]] = [
        (38.31413, -76.54352),
        (38.31629, -76.55587),
        (38.31611, -76.55126),
        (38.31712, -76.55102),
        (38.31560, -76.54838),
        (38.31413, -76.54352),
        (38.31629, -76.55587),
        (38.31413, -76.54352),
        (38.31466, -76.54665),
    ]

    point: tuple[float, float]
    for point in points:
        await move_to(drone.vehicle, point[0], point[1], obj_altitude)

    # return home
    await drone.return_to_launch()
    print("Staying connected, press Ctrl-C to exit")

    # infinite loop till forced disconnect
    while True:
        await asyncio.sleep(1)


# Runs through the code until it has looped through each element of
# the Lats and Longs array and the drone has arrived at each of them
if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run())
    except KeyboardInterrupt:
        print("Program ended")
        sys.exit(0)
