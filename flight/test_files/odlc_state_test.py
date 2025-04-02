"""A unit test for the ODLC state."""

import asyncio
import json
import logging

from state_machine.drone import Drone
from state_machine.state_machine import StateMachine
from state_machine.states import Start
from state_machine.flight_settings import FlightSettings


async def run_test(flight_settings: FlightSettings) -> None:
    """
    Tests the the ODLC state in the State Machine. Runs through an example run of the ODLC
    and its functions, testing if the drone can find 5 waypoints, as well as checking to make
    sure the JSON file is output correctly.

    Parameters
    ----------
    flight_settings : FlightSettings
        The flight settings to use.
    """
    logging.basicConfig(level=logging.INFO)

    flight_settings.skip_waypoint = True
    flight_settings.standard_object_count = 5

    drone: Drone = Drone()
    drone.use_settings(flight_settings.sim_mode)
    await drone.connect_drone()
    state_task: asyncio.Task[None] = asyncio.ensure_future(
        StateMachine(Start(drone, flight_settings), drone, flight_settings).run()
    )

    activated_odlcs: int = 0
    while activated_odlcs != flight_settings.standard_object_count:
        try:
            with open("flight/data/output.json", "r", encoding="UTF-8") as file:
                output_data: str = json.load(file)

            activated_odlcs = len(output_data)

            if activated_odlcs == 5:
                print("All 5 ODLCs were found.")
            else:
                print(f"{activated_odlcs} ODLCs found.")

        except FileNotFoundError:
            print("Output JSON file not found.")
        except json.JSONDecodeError as json_error:
            print(f"Error loading JSON file: {json_error}")

        await asyncio.sleep(10)
    while state_task.done() is False:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(run_test(FlightSettings.from_mission_config()))
