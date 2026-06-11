"""
A test that connects to a drone, arms it, takes off, and prints
the flight time every second until the drone disarms along with
the total flight time.
"""

import asyncio
import logging

from state_machine.drone import Drone
from state_machine.flight_settings import FlightSettings
from state_machine.state_machine import StateMachine
from state_machine.states.start import Start


async def flight_time_check(drone: Drone) -> None:
    """
    Monitors and logs the flight time once per second while the drone is armed.
    Waits for the drone to connect and arm before starting, then exits once
    the drone disarms.

    Parameters
    ----------
    drone : Drone
        The drone object from the flight manager.
    """
    logger = logging.getLogger("flight_time_check")

    while not drone.is_connected:
        await asyncio.sleep(0.1)

    while not drone.vehicle.armed:
        await asyncio.sleep(0.1)

    logger.info("Drone armed. Tracking flight time.")

    while drone.vehicle.armed:
        logger.info(
            "Flight time: %.1fs",
            drone.flight_time,
        )
        await asyncio.sleep(1.0)

    logger.info(
        "Drone disarmed. Flight time should now be 0.0s: %.1fs",
        drone.flight_time,
    )
    assert drone.flight_time == 0.0
    logger.info("Total flight time: %.1fs", drone.last_flight_time)


async def run_test(flight_settings: FlightSettings) -> None:
    """
    Initialize and run the state machine and flight time check.

    Parameters
    ----------
    flight_settings : FlightSettings
        The flight settings to use.
    """
    drone: Drone = Drone()
    drone.use_settings(flight_settings.sim_mode)

    # Skip all states, to go from start -> takeoff -> land
    drone.odlc_scan = False
    flight_settings.skip_waypoint = True
    flight_settings.skip_odlc_and_airdrop = True

    await drone.connect_drone()

    state_task: asyncio.Task[None] = asyncio.ensure_future(
        StateMachine(Start(drone, flight_settings), drone, flight_settings).run()
    )
    await flight_time_check(drone)

    while not state_task.done():
        await asyncio.sleep(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_test(FlightSettings.from_mission_config()))
