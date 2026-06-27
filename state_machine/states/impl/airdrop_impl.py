"""Implements the behavior of the Airdrop state."""

import asyncio
import json
import logging
import math
from typing import cast

import utm

from flight.extract_gps import extract_gps
from flight.waypoint.goto import move_to
from state_machine.drone import Drone
from state_machine.flight_settings import FlightSettings, SimMode
from state_machine.state_tracker import (
    update_drone,
    update_flight_settings,
    update_state,
)
from state_machine.states.airdrop import Airdrop
from state_machine.states.land import Land
from state_machine.states.state import State
from vision.common.constants import AirdropConfig, AirdropStatus, Location, ODLCDict

# The altitude to go up to while staying at the airdrop point
# This could allow stuck beacons to wiggle out
# 75 ft -> 23 m
WIGGLE_ALTITUDE: float = 23.0


async def run(self: Airdrop) -> State:
    """
    Implements the run method for the Airdrop state.

    Returns
    -------
    Waypoint | Land : State
        Goes to the Waypoint state if there is another airdrop to complete,
        or Land if there are no more airdrops to complete.

    Notes
    -----
    This method is responsible for initiating the Airdrop process of the drone and transitioning
    it back to the Waypoint state.
    """

    if self.flight_settings.skip_odlc_and_airdrop:
        return Land(self.drone, self.flight_settings)

    try:
        update_state("Airdrop")
        update_drone(self.drone)
        update_flight_settings(self.flight_settings)
        logging.info("Airdrop state running")

        with open("flight/data/output.json", encoding="utf8") as output:
            drop_locations: ODLCDict = cast(ODLCDict, json.load(output))
            if not drop_locations:
                logging.error("No drop locations found, loading fallback locations")
                fallback_locations: list[Location] = extract_gps(
                    self.flight_settings.mission_data_path
                )["default_airdrop_points"]
                for i, location in enumerate(fallback_locations):
                    drop_locations[f"fallback_{i}"] = location

        with open("flight/data/airdrops.json", encoding="utf8") as output:
            airdrops: AirdropStatus = cast(AirdropStatus, json.load(output))

        logging.info("Moving to drop location")

        # Find if there is a loaded airdrop
        airdrop_to_use: str = ""
        airdrop: str
        config: AirdropConfig
        for airdrop, config in airdrops.items():
            if config["loaded"]:
                airdrop_to_use = airdrop
                break

        if airdrop_to_use == "":
            logging.warning("No beacons are loaded.")
            return Land(self.drone, self.flight_settings)

        dropped: bool = await attempt_drop(
            self.drone,
            self.flight_settings,
            drop_locations,
            airdrops,
            airdrop_to_use,
            self.flight_settings.mission_data_path,
            self.flight_settings.mean_wind_speed,
            self.flight_settings.mean_wind_direction,
        )

        # Write new data back out to airdrops.json
        with open("flight/data/airdrops.json", "w", encoding="utf8") as file:
            json.dump(airdrops, file)

        if not dropped:
            return Airdrop(self.drone, self.flight_settings)
        continue_run: bool = False

        for airdrop_config in airdrops.values():
            if airdrop_config["loaded"]:
                continue_run = True

        if continue_run:
            return Airdrop(self.drone, self.flight_settings)
        return Land(self.drone, self.flight_settings)

    except asyncio.CancelledError as ex:
        logging.error("Airdrop state canceled")
        raise ex
    finally:
        pass


# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
async def attempt_drop(
    drone: Drone,
    flight_settings: FlightSettings,
    drop_locations: ODLCDict,
    airdrop_config: AirdropStatus,
    airdrop_to_use: str,
    path: str,
    mean_wind_speed: float,
    mean_wind_direction: float,
) -> bool:
    """
    Attempts to perform a drop at the given location.

    Parameters
    ----------
    drone : Drone
        The drone object to control
    flight_settings : FlightSettings
        The flight settings object
    drop_locations : ODLCDict
        Dictionary of available drop locations
    airdrop_config : AirdropStatus
        The airdrop status object
    airdrop_to_use : str
        The airdrop to use
    path : str
        The path to the airdrop data
    mean_wind_speed : float
        The mean wind speed
    mean_wind_direction : float
        The mean wind direction

    Returns
    -------
    bool
        True if drop was successful, False otherwise
    """
    try:
        drop_loc: Location = drop_locations[airdrop_to_use]

        airdrop_altitude: float = extract_gps(path)["airdrop_altitude"]

        wind_offset: float = calculate_airdrop_wind_offset(mean_wind_speed, airdrop_altitude)

        easting: float
        northing: float
        zone_number: int
        zone_letter: str
        easting, northing, zone_number, zone_letter = utm.from_latlon(
            drop_loc["latitude"], drop_loc["longitude"]
        )

        easting += wind_offset * -math.sin(math.radians(mean_wind_direction))
        northing += wind_offset * math.cos(math.radians(mean_wind_direction))

        drop_lat: float
        drop_lon: float
        drop_lat, drop_lon = utm.to_latlon(easting, northing, zone_number, zone_letter)

        await move_to(drone.vehicle, drop_lat, drop_lon, airdrop_altitude)

        logging.info(
            "Starting drop at fresh location %s",
            airdrop_to_use,
        )

        if flight_settings.sim_mode is SimMode.REAL:
            await drone.open_servo((airdrop_config[airdrop_to_use])["servo"])

        airdrop_config[airdrop_to_use]["loaded"] = False

        await asyncio.sleep(12)

        # Attempt to wiggle out stuck beacons
        logging.info("Moving up to 75 ft / 23 meters...")
        await move_to(drone.vehicle, drop_lat, drop_lon, WIGGLE_ALTITUDE)
        await asyncio.sleep(3)

        logging.info("-- Airdrop done!")
        return True

    except KeyError:
        logging.warning("Drop location %s was not found. Skipping.", airdrop_to_use)
        return False


def calculate_airdrop_wind_offset(wind_speed: float, drop_altitude: float) -> float:
    """
    Calculates the wind offset for dropping the payload.

    Parameters
    ----------
    wind_speed : float
        The wind speed, in meters per second.
    drop_altitude : float
        The altitude of the drop, in meters.

    Returns
    -------
    float
        The offset in the direction of the wind to add to the drop location.
    """
    if wind_speed <= 0.0:
        return 0.0

    payload_mass: float = 0.155  # kg
    gravity_acceleration: float = 9.81  # m/s^2
    parachute_area: float = 0.25  # cross-sectional area in m^2
    parachute_drag_coefficient: float = 1.4  # coefficient of drag
    air_density: float = 1.225  # kg/m^3
    parachute_closed_duration: float = 1.0  # seconds

    vertical_velocity_parachute_open: float = math.sqrt(
        2.0
        * payload_mass
        * gravity_acceleration
        / (air_density * parachute_drag_coefficient * parachute_area)
    )  # m/s

    freefall_distance: float = 0.5 * gravity_acceleration * parachute_closed_duration**2  # meters

    parachute_open_duration: float = (
        drop_altitude - freefall_distance
    ) / vertical_velocity_parachute_open  # seconds
    drop_duration: float = parachute_closed_duration + parachute_open_duration  # seconds

    offset: float = -wind_speed * (
        drop_duration + 0.2 * (math.exp(-5.0 * drop_duration) - 1.0)
    )  # meters

    offset *= 0.9  # should improve accuracy slightly

    return offset


# Setting the run_callable attribute of the Airdrop class to the run function
Airdrop.run_callable = run
