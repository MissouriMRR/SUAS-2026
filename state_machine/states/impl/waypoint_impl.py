"""Implement the behavior of the Waypoint state."""

# pylint: disable=too-many-locals,too-many-statements

import asyncio
import logging
import traceback
from typing import Final

import dronekit

from flight.extract_gps import BoundaryPointUtm
from flight.extract_gps import GPSData, extract_gps
from flight.extract_gps import WaypointUtm
from flight.waypoint.missions import WaypointMission
from state_machine.state_tracker import (
    update_drone,
    update_flight_settings,
    update_state,
)
from state_machine.states.airdrop import Airdrop
from state_machine.states.odlc import ODLC
from state_machine.states.state import State
from state_machine.states.waypoint import Waypoint

WAYPOINT_AIR_SPEED: Final[float] = 25.0  # in meters/second
WAYPOINT_MAX_LAPS: Final[int] = 10  # Taken from SUAS Rule 3.2.2
MISSION_POLL_INTERVAL: Final[float] = 0.1  # in seconds


async def run(self: Waypoint) -> State:
    """
    Run method implementation for the Waypoint state.

    This method instructs the drone to navigate to a specified waypoint and
    transitions to the Airdrop or ODLC State.

    Returns
    -------
    Airdrop : State
        The next state after successfully reaching the specified waypoint and
        initiating the Airdrop process.
    ODLC : State
        The next state after successfully reaching the specified waypoint and
        initiating the ODLC process.
    """

    try:
        if not self.flight_settings.skip_waypoint:
            await waypoint_logic(self)
            logging.info(
                "Waypoint state completed after %d lap(s). Currently at a flight time of %d:%05.2f",
                self.flight_settings.waypoint_laps_run,
                int(self.drone.flight_time // 60),
                self.drone.flight_time % 60,
            )

        return (ODLC if self.drone.odlc_scan else Airdrop)(self.drone, self.flight_settings)

    except asyncio.CancelledError as ex:
        logging.error("Waypoint state canceled")
        traceback.print_exc()
        raise ex
    finally:
        pass


async def ask_to_continue() -> bool:
    """
    Prompts the user to continue with another waypoint lap.

    The blocking input() call runs in a worker thread so the event loop
    (and the waypoint tracking loop) keeps running while waiting for an
    answer.
    """
    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    logging.info("Queue up another waypoint lap? (y/n)")
    while True:
        choice: str = (await loop.run_in_executor(None, input)).strip().lower()
        if choice in ("y", "n"):
            return choice == "y"
        logging.info("Invalid choice. Please enter 'y' or 'n'.")


async def waypoint_logic(self: Waypoint) -> None:
    """
    Run the logic for the waypoint state.

    Builds the route for a lap (routing around the flight boundary as needed) as
    a sequence of ArduPilot mission commands, uploads it to the vehicle, and
    switches to AUTO mode so that ArduPilot handles optimal mission traversal
    (corner cutting, S-Curves, etc.)

    While a lap is flying, once it's nearly done this asks whether to queue up
    another lap. If so, the next lap's commands are appended to the
    current mission and re-uploaded, so that ArduPilot is able to splice between
    the laps.

    Parameters
    ----------
    self : Waypoint
        The waypoint state object.
    """
    update_state("Waypoint")
    update_drone(self.drone)
    update_flight_settings(self.flight_settings)
    logging.info("Waypoint state running")

    # Extract GPS data from the mission data path
    gps_dict: GPSData = extract_gps(self.flight_settings.mission_data_path)
    waypoints_utm: list[WaypointUtm] = gps_dict["waypoints_utm"]
    waypoints_per_lap: int = len(waypoints_utm)

    boundary_points: list[BoundaryPointUtm] = gps_dict["boundary_points_utm"]

    # Initialize the waypoint mission
    mission: WaypointMission = WaypointMission(self.drone.vehicle, waypoints_utm, boundary_points)

    # Upload initial lap and set to AUTO mode
    laps_queued: int = 1
    mission.add_lap()
    mission.upload()

    self.drone.vehicle.airspeed = WAYPOINT_AIR_SPEED
    self.drone.vehicle.mode = dronekit.VehicleMode("AUTO")
    while self.drone.vehicle.mode.name != "AUTO":
        await asyncio.sleep(0.1)
    logging.info("Uploaded waypoint lap %d", laps_queued)

    mission_finalized: bool = False  # whether the mission is done accepting laps
    decided_next_lap: bool = False  # whether the next lap has been decided
    prompt_task: asyncio.Task[bool] | None = None  # task for asking the user to continue
    waypoint_num: int = 0  # real waypoints reached so far, across all laps
    while not (mission_finalized and waypoint_num >= laps_queued * waypoints_per_lap):
        position_in_lap: int = waypoint_num % waypoints_per_lap
        current_lap: int = waypoint_num // waypoints_per_lap + 1

        # Ask for next lap early so that there's time to upload the next lap before
        # the drone finishes the current one
        # If the user hasn't already been sent prompt this lap,
        # and we're at the second to last waypoint, create prompt_task
        if (
            not decided_next_lap
            and current_lap == laps_queued
            and position_in_lap == max(waypoints_per_lap - 2, 0)
        ):
            decided_next_lap = True  # Set to true right way so prompt is only sent once per lap

            # "Finalize" right away so that reaching the last waypoint is
            # observable no matter how late the answer comes
            mission.finalize()
            if laps_queued < WAYPOINT_MAX_LAPS:
                prompt_task = asyncio.create_task(ask_to_continue())
            else:
                mission_finalized = True

        if prompt_task is not None and prompt_task.done():
            wants_another_lap: bool = prompt_task.result()
            prompt_task = None
            if wants_another_lap and waypoint_num < laps_queued * waypoints_per_lap:
                # Remove final dummy command, add lap, and upload
                mission.unfinalize()
                mission.add_lap()
                mission.upload()
                laps_queued += 1
                self.flight_settings.waypoint_laps_run = laps_queued
                logging.info("Uploaded waypoint lap %d", laps_queued)
            else:
                if wants_another_lap:
                    # If the last waypoint has already been reached, adding
                    # another lap to the mission won't automatically continue
                    # the mission with the next lap, consider it too late.
                    # This is an edgecase, but would rather be safe and
                    # consider waypoints over rather than softlock
                    logging.info("Last waypoint reached; too late to add another lap; skipping lap")
                mission_finalized = True

        elif prompt_task is not None and waypoint_num >= laps_queued * waypoints_per_lap:
            # The lap finished before the answer received, so it's too late to
            # splice in another one; stop waiting for an answer
            logging.info("Final waypoint reached with no answer; ending mission")
            _ = prompt_task.cancel()
            prompt_task = None
            mission_finalized = True

        # Log waypoint num, lap num, and distance to waypoint after each waypoint hit
        radius: float = mission.distance_to_waypoint(waypoint_num)
        logging.debug(
            "Distance to waypoint %d: %.1f m / %.1f ft", waypoint_num, radius, radius * 3.28084
        )
        if mission.waypoints_reached() > waypoint_num:
            # This means the drone reached the waypoint it was going towards
            flight_time: float = self.drone.flight_time
            logging.info(
                "Reached waypoint %d of lap %d | mission time %d:%05.2f | radius %.1f m / %.1f ft",
                position_in_lap + 1,
                current_lap,
                int(flight_time // 60),
                flight_time % 60,
                radius,
                radius * 3.28084,
            )
            waypoint_num += 1
            decided_next_lap = False
            continue
        await asyncio.sleep(MISSION_POLL_INTERVAL)

    self.flight_settings.waypoint_laps_run = laps_queued

    # Hand control back to GUIDED mode for subsequent states
    self.drone.vehicle.mode = dronekit.VehicleMode("GUIDED")
    while self.drone.vehicle.mode.name != "GUIDED":
        await asyncio.sleep(0.1)


# Set the run_callable attribute of the Waypoint class to the run function
Waypoint.run_callable = run
