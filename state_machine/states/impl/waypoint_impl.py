"""Implement the behavior of the Waypoint state."""

# pylint: disable=too-many-locals,too-many-statements

import asyncio
import logging
import traceback
from typing import Final

import dronekit
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from flight.extract_gps import BoundaryPointUtm, GPSData, WaypointUtm, extract_gps
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

        return (ODLC if self.drone.odlc_scan else Airdrop)(
            self.drone, self.flight_settings
        )

    except asyncio.CancelledError:
        logging.error("Waypoint state canceled")
        traceback.print_exc()
        raise
    finally:
        pass


async def ask_to_continue(
    mission: WaypointMission,
    waypoints_per_lap: int,
) -> None:
    """
    Prompt for additional waypoint laps and splice them into the running
    mission. Laps can be added at any time throughout the lap.

    Parameters
    ----------
    mission : WaypointMission
        The mission laps are appended to. This coroutine is its only writer;
        the tracking loop only reads from it.
    waypoints_per_lap : int
        The number of real waypoints in a single lap.
    """
    session: PromptSession[str] = PromptSession()
    # patch_stdout() keeps log lines written to stdout from overwriting
    with patch_stdout():
        while True:
            reply: str = await session.prompt_async(
                "Queue up another waypoint lap? (y/n)\n"
            )
            answer: str = reply.strip().lower()

            if answer not in ("y", "n"):
                logging.info("Invalid choice. Please enter 'y' or 'n'.")
                continue

            if answer == "n":
                logging.info("No more laps queued; ending after lap %d", mission.laps)
                return

            if mission.waypoints_reached() >= mission.laps * waypoints_per_lap:
                # The drone is already flying to the dummy end command, so a lap
                # appended now wouldn't be picked up automatically.
                logging.info(
                    "Last waypoint reached; too late to add another lap, skipping lap"
                )
                return

            # Unfinalize, add a lap, and re-finalize to re-upload the mission
            mission.unfinalize()
            mission.add_lap()
            mission.finalize()  # appends the dummy end command and uploads
            logging.info("Uploaded waypoint lap %d", mission.laps)

            if mission.laps >= WAYPOINT_MAX_LAPS:
                logging.info("Reached the %d lap limit", WAYPOINT_MAX_LAPS)
                return


async def waypoint_logic(self: Waypoint) -> None:
    """
    Run the logic for the waypoint state.

    Builds the route for a lap (routing around the flight boundary as needed) as
    a sequence of ArduPilot mission commands, uploads it to the vehicle, and
    switches to AUTO mode so that ArduPilot handles optimal mission traversal
    (corner cutting, S-Curves, etc.)

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
    mission: WaypointMission = WaypointMission(
        self.drone.vehicle, waypoints_utm, boundary_points
    )

    # Upload initial lap and set to AUTO mode
    mission.add_lap()
    mission.finalize()  # appends the dummy end command and uploads

    self.drone.vehicle.airspeed = WAYPOINT_AIR_SPEED
    self.drone.vehicle.mode = dronekit.VehicleMode("AUTO")
    while self.drone.vehicle.mode.name != "AUTO":
        await asyncio.sleep(0.1)
    logging.info("Uploaded waypoint lap %d", mission.laps)

    # Runs alongside the mission for its whole duration, so laps can be queued
    # at any time. Once it's done, no more laps are coming.
    prompt_task: asyncio.Task[None] = asyncio.create_task(
        ask_to_continue(mission, waypoints_per_lap)
    )

    waypoint_num: int = 0  # real waypoints reached so far, across all laps
    # While the prompt task is still running and the drone hasn't reached the end of the mission
    while not (prompt_task.done() and waypoint_num >= mission.laps * waypoints_per_lap):
        position_in_lap: int = waypoint_num % waypoints_per_lap
        current_lap: int = waypoint_num // waypoints_per_lap + 1
        self.flight_settings.waypoint_laps_run = current_lap - 1

        if not prompt_task.done() and waypoint_num >= mission.laps * waypoints_per_lap:
            # The last waypoint was reached before an answer came in, so
            # it's too late to splice in another lap, stop the prompt task
            logging.info("Final waypoint reached with no answer, ending mission")
            _ = prompt_task.cancel()

        # Log waypoint num, lap num, and distance to waypoint after each waypoint hit
        radius: float = mission.distance_to_waypoint(waypoint_num)
        logging.debug(
            "Distance to waypoint %d, lap %d: %.1f m / %.1f ft",
            position_in_lap + 1,
            current_lap,
            radius,
            radius * 3.28084,
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
            continue
        await asyncio.sleep(MISSION_POLL_INTERVAL)

    # Cancel the prompt task just in case it is still open
    _ = prompt_task.cancel()

    if not prompt_task.cancelled():
        # Prompt task died on its own
        prompt_error: BaseException | None = prompt_task.exception()
        if prompt_error is not None:
            logging.error("Lap prompt failed: %r", prompt_error)

    self.flight_settings.waypoint_laps_run = mission.laps

    # Hand control back to GUIDED mode for subsequent states
    self.drone.vehicle.mode = dronekit.VehicleMode("GUIDED")
    while self.drone.vehicle.mode.name != "GUIDED":
        await asyncio.sleep(0.1)


# Set the run_callable attribute of the Waypoint class to the run function
Waypoint.run_callable = run
