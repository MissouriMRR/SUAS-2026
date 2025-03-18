"""Implements the behavior of the Mapping state."""

# pylint: disable=too-many-locals

import asyncio
import json
import logging
import math
from typing import Final

import utm

from flight.camera import CameraIRL
from flight.extract_gps import extract_gps, GPSData
from flight.extract_gps import BoundaryPointUtm
from flight.waypoint.goto import move_to

from state_machine.state_tracker import (
    update_state,
    update_drone,
    update_flight_settings,
)
from state_machine.states.land import Land
from state_machine.states.mapping import Mapping
from state_machine.states.state import State
from vision.common.constants import CameraConfig

# These should be moved to a constants file
MAPPING_ALTITUDE: Final[float] = 30  # meters
HORIZONTAL_PHOTO_SPACING: Final[float] = 15  # meters
VERTICAL_PHOTO_SPACING: Final[float] = 15  # meters


async def run(self: Mapping) -> State:
    """
    Implements the run method for the Mapping state.

    This method captures photos of the mapping area and then transitions to the ODLC state.

    Returns
    -------
    ODLC : State
        The next state after the drone has successfully landed.
    """

    with open("vision/common/camera_config.json", encoding="ascii", mode="r+") as file:
        camera_config: CameraConfig = json.load(file)
        if self.flight_settings.airsim_flag:
            camera_config["airsim_flag"] = True
        else:
            camera_config["airsim_flag"] = False
        file.seek(0)
        file.truncate()
        json.dump(camera_config, file)

    try:
        update_state("Mapping")
        update_drone(self.drone)
        update_flight_settings(self.flight_settings)

        logging.info("Mapping")

        gps_dict: GPSData = extract_gps(self.flight_settings.path_data_path)
        mapping_boundary_utm: list[BoundaryPointUtm] = gps_dict["mapping_boundary_utm"]

        # The mapping area should be roughly rectangular with 4 vertices
        # We are going to travel in line segments parallel to the long edges

        is_long_edge_first: bool = math.hypot(
            mapping_boundary_utm[1].easting - mapping_boundary_utm[0].easting,
            mapping_boundary_utm[1].northing - mapping_boundary_utm[0].northing,
        ) >= math.hypot(
            mapping_boundary_utm[2].easting - mapping_boundary_utm[0].easting,
            mapping_boundary_utm[2].northing - mapping_boundary_utm[0].northing,
        )

        # Ensure the first edge is long
        if not is_long_edge_first:
            mapping_boundary_utm[1], mapping_boundary_utm[3] = (
                mapping_boundary_utm[3],
                mapping_boundary_utm[1],
            )

        # Take the max because it's better to take too many photos than too few
        short_edge_length = max(
            math.hypot(
                mapping_boundary_utm[3].easting - mapping_boundary_utm[0].easting,
                mapping_boundary_utm[3].northing - mapping_boundary_utm[0].northing,
            ),
            math.hypot(
                mapping_boundary_utm[2].easting - mapping_boundary_utm[1].easting,
                mapping_boundary_utm[2].northing - mapping_boundary_utm[1].northing,
            ),
        )

        # Get average direction of short edges
        step_count: int = math.ceil(short_edge_length / VERTICAL_PHOTO_SPACING)

        utm_zone_number = mapping_boundary_utm[0].zone_number
        utm_zone_letter = mapping_boundary_utm[0].zone_letter
        camera: CameraIRL = CameraIRL()
        reverse_direction: bool = False
        for i in range(step_count + 1):
            lerp_t = i / step_count

            # Linearly interpolate along the short edges
            start_easting: float = (1 - lerp_t) * mapping_boundary_utm[
                0
            ].easting + lerp_t * mapping_boundary_utm[3].easting
            start_northing: float = (1 - lerp_t) * mapping_boundary_utm[
                0
            ].northing + lerp_t * mapping_boundary_utm[3].northing
            end_easting: float = (1 - lerp_t) * mapping_boundary_utm[
                1
            ].easting + lerp_t * mapping_boundary_utm[2].easting
            end_northing: float = (1 - lerp_t) * mapping_boundary_utm[
                1
            ].northing + lerp_t * mapping_boundary_utm[2].northing

            # Move in a serpentine pattern
            if reverse_direction:
                start_easting, end_easting = end_easting, start_easting
                start_northing, end_northing = end_northing, start_northing

            lat: float
            lon: float
            lat, lon = utm.to_latlon(
                start_easting, start_northing, utm_zone_number, utm_zone_letter
            )
            await move_to(self.drone.vehicle, lat, lon, MAPPING_ALTITUDE)

            lat, lon = utm.to_latlon(end_easting, end_northing, utm_zone_number, utm_zone_letter)
            await camera.mapping_move_to(
                self.drone.vehicle,
                lat,
                lon,
                MAPPING_ALTITUDE,
                HORIZONTAL_PHOTO_SPACING,
            )

            reverse_direction = not reverse_direction

        camera.camera.disconnect()

        logging.info("Mapping state complete.")
    except asyncio.CancelledError as ex:
        logging.error("Mapping state canceled")
        raise ex

    return Land(self.drone, self.flight_settings)


# Setting the run_callable attribute of the Mapping class to the run function
Mapping.run_callable = run
