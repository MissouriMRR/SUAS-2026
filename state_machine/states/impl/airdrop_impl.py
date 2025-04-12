"""Implements the behavior of the Airdrop state."""

import asyncio
import logging
import json

from flight.extract_gps import extract_gps
from flight.waypoint.goto import move_to

from state_machine.state_tracker import (
    update_state,
    update_drone,
    update_flight_settings,
)

from state_machine.drone import Drone
from state_machine.flight_settings import FlightSettings, SimMode

from state_machine.states.airdrop import Airdrop
from state_machine.states.mapping import Mapping
from state_machine.states.state import State
from state_machine.states.waypoint import Waypoint


async def run(self: Airdrop) -> State:
    """
    Implements the run method for the Airdrop state.

    Returns
    -------
    Waypoint : State
        The next state after the drone has successfully completed the Airdrop.

    Notes
    -----
    This method is responsible for initiating the Airdrop process of the drone and transitioning
    it back to the Waypoint state.
    """

    if self.flight_settings.skip_odlc_and_airdrop:
        return Mapping(self.drone, self.flight_settings)

    try:
        update_state("Airdrop")
        update_drone(self.drone)
        update_flight_settings(self.flight_settings)
        logging.info("Airdrop state running")

        with open("flight/data/output.json", encoding="utf8") as output:
            drop_locations: dict[str, dict[str, int]] = json.load(output)

        with open("flight/data/bottles.json", encoding="utf8") as output:
            cylinders: dict[str, dict[str, int | bool]] = json.load(output)

        logging.info("Moving to drop location")

        # Track attempted locations
        attempted_locations: set[str] = set()
        try:
            with open("flight/data/attempted_drops.json", encoding="utf8") as file:
                attempted_locations = set(json.load(file))
        except (FileNotFoundError, json.JSONDecodeError):
            with open("flight/data/attempted_drops.json", "w", encoding="utf8") as file:
                json.dump(list(attempted_locations), file)

        # Find if there is a loaded cylinder
        cylinder_num: str = ""
        for cylinder in cylinders:
            if cylinders[cylinder]["Loaded"]:
                cylinder_num = cylinder
                break
            if cylinder_num == "":
                cylinder_num = cylinder
        else:
            logging.warning("No beacons are loaded?")
            return Mapping(self.drone, self.flight_settings)

        dropped: bool = await attempt_drop(
            self.drone,
            self.flight_settings,
            drop_locations,
            cylinders,
            attempted_locations,
            cylinder_num,
            self.flight_settings.mission_data_path,
        )

        with open("flight/data/bottles.json", "w", encoding="utf8") as output:
            json.dump(cylinders, output)

        if not dropped:
            return Airdrop(self.drone, self.flight_settings)
        continue_run: bool = False

        for cylinder in cylinders:
            if (cylinders[cylinder])["Loaded"]:
                continue_run = True

        if continue_run:
            return Waypoint(self.drone, self.flight_settings)
        return Mapping(self.drone, self.flight_settings)

    except asyncio.CancelledError as ex:
        logging.error("Airdrop state canceled")
        raise ex
    finally:
        pass


async def attempt_drop(
    drone: Drone,
    flight_settings: FlightSettings,
    drop_locations: dict[str, dict[str, int]],
    cylinders: dict[str, dict[str, int | bool]],
    attempted_locations: set[str],
    cylinder_num: str,
    path: str,
    retry_mode: bool = False,
) -> bool:
    """
    Attempts to perform a drop at the next available location.

    Parameters
    ----------
    drone : Drone
        The drone object to control
    flight_settings : FlightSettings
        The flight settings object
    drop_locations : dict
        Dictionary of available drop locations
    cylinders : dict
        Dictionary of cylinder states
    attempted_locations : set
        Set of previously attempted drop locations
    cylinder_num : str
        The cylinder number to use for the drop
    path : str
        the path to where our waypoint/flight locations are stored
    retry_mode : bool
        Whether we're in retry mode (attempting previously visited locations)

    Returns
    -------
    bool
        True if drop was successful, False otherwise
    """
    try:
        # Find next available drop location
        available_locations = set(drop_locations.keys()) - attempted_locations
        location_id: str
        drop_loc: dict[str, int]

        if available_locations:
            # Get the next location ID (using min for consistent ordering)
            location_id = min(available_locations)
            drop_loc = drop_locations[location_id]
        else:
            logging.info("All locations attempted, entering retry mode")
            retry_mode = True
            # Pick the first location to retry
            location_id = min(drop_locations.keys())
            drop_loc = drop_locations[location_id]

        airdrop_altitude: float = extract_gps(path)["airdrop_altitude"]

        await move_to(drone.vehicle, drop_loc["latitude"], drop_loc["longitude"], airdrop_altitude)

        if retry_mode:
            logging.info(
                "Attempting drop at previously visited location %s",
                location_id,
            )
        else:
            logging.info(
                "Starting drop at fresh location %s",
                location_id,
            )

        if flight_settings.sim_mode is SimMode.REAL:
            await drone.open_servo((cylinders[cylinder_num])["Servo"])

        (cylinders[cylinder_num])["Loaded"] = False

        # Record attempted location
        attempted_locations.add(location_id)
        with open("flight/data/attempted_drops.json", "w", encoding="utf8") as file:
            json.dump(list(attempted_locations), file)

        await asyncio.sleep(15)
        logging.info("-- Airdrop done!")
        return True

    except KeyError:
        logging.warning("Drop location %s was not found. Skipping.", location_id)
        return False


# Setting the run_callable attribute of the Airdrop class to the run function
Airdrop.run_callable = run
