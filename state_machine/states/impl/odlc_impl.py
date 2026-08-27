"""Implements the behavior of the ODLC state."""

import asyncio
import logging
import math
import traceback
from pathlib import Path
from typing import Final

import utm

from flight.camera import CameraAirSim, CameraIRL
from flight.extract_gps import BoundaryPointUtm, GPSData, extract_gps
from flight.waypoint.goto import move_to
from state_machine.flight_settings import SimMode
from state_machine.state_tracker import (
    update_drone,
    update_flight_settings,
    update_state,
)
from state_machine.states.mapping import Mapping
from state_machine.states.odlc import ODLC
from state_machine.states.state import State
from vision.common import camera_config
from vision.odlc_pipeline import odlc_pipeline

HORIZONTAL_PHOTO_SPACING: Final[float] = 15  # meters
VERTICAL_PHOTO_SPACING: Final[float] = 15  # meters


async def run(self: ODLC) -> State:
    """
    Implements the run method for the ODLC state.

    This method initiates the ODLC scanning process of the drone, takes pictures and transfers
    picture data to the vision code, and then transitions to the Mapping state.

    Parameters
    ----------
    self : ODLC
        The current instance of the ODLC state.

    Returns
    -------
    Mapping : State
        The next state after the drone has successfully scanned the ODLC area.

    Raises
    ------
    asyncio.CancelledError
        If the execution of the ODLC state is canceled.
    """

    camera_config.update_sim_mode(self.flight_settings.sim_mode)

    if self.flight_settings.skip_odlc_and_airdrop:
        return Mapping(self.drone, self.flight_settings)

    try:
        update_state("ODLC")
        update_drone(self.drone)
        update_flight_settings(self.flight_settings)
        logging.info("ODLC state running")

        capture_status: asyncio.Event = asyncio.Event()

        asyncio.ensure_future(vision_odlc_logic(self, capture_status))

        flight_task: asyncio.Task[None] = asyncio.ensure_future(
            fly_scanning_pattern(self, capture_status)
        )

        logging.info("Starting check for flight task completion")

        while not flight_task.done():
            await asyncio.sleep(0.25)

        logging.info("ODLC flight scan complete. State completing...")
        flight_task.cancel()
    except asyncio.CancelledError:
        logging.error("ODLC state canceled")
        traceback.print_exc()
        raise

    return Mapping(self.drone, self.flight_settings)


async def fly_scanning_pattern(self: ODLC, capture_status: asyncio.Event) -> None:
    """
    This will fly the drone in a zig-zag pattern, scanning the entire search area.

    Parameters
    ----------
    self : ODLC
        The ODLC state object.
    capture_status : asyncio.Event
        An event that is set when the drone has successfully captured all images.
    """

    gps_dict: GPSData = extract_gps(self.flight_settings.mission_data_path)
    object_boundary_utm: list[BoundaryPointUtm] = gps_dict["object_boundary_utm"]

    # The mapping area should be roughly rectangular with 4 vertices
    # We are going to travel in line segments parallel to the long edges

    is_long_edge_first: bool = math.hypot(
        object_boundary_utm[1].easting - object_boundary_utm[0].easting,
        object_boundary_utm[1].northing - object_boundary_utm[0].northing,
    ) >= math.hypot(
        object_boundary_utm[2].easting - object_boundary_utm[0].easting,
        object_boundary_utm[2].northing - object_boundary_utm[0].northing,
    )

    # Ensure the first edge is long
    if not is_long_edge_first:
        object_boundary_utm[1], object_boundary_utm[3] = (
            object_boundary_utm[3],
            object_boundary_utm[1],
        )

    # Take the max because it's better to take too many photos than too few
    short_edge_length = max(
        math.hypot(
            object_boundary_utm[3].easting - object_boundary_utm[0].easting,
            object_boundary_utm[3].northing - object_boundary_utm[0].northing,
        ),
        math.hypot(
            object_boundary_utm[2].easting - object_boundary_utm[1].easting,
            object_boundary_utm[2].northing - object_boundary_utm[1].northing,
        ),
    )

    # Get average direction of short edges
    step_count: int = math.ceil(short_edge_length / VERTICAL_PHOTO_SPACING)

    utm_zone_number = object_boundary_utm[0].zone_number
    utm_zone_letter = object_boundary_utm[0].zone_letter
    if self.flight_settings.sim_mode is SimMode.REAL:
        camera: CameraIRL | CameraAirSim | None = CameraIRL()
    elif self.flight_settings.sim_mode is SimMode.AIRSIM:
        camera = CameraAirSim()
    else:
        camera = None
    reverse_direction: bool = False
    for i in range(step_count + 1):
        lerp_t = i / step_count

        # Linearly interpolate along the short edges
        start_easting: float = (1 - lerp_t) * object_boundary_utm[
            0
        ].easting + lerp_t * object_boundary_utm[3].easting
        start_northing: float = (1 - lerp_t) * object_boundary_utm[
            0
        ].northing + lerp_t * object_boundary_utm[3].northing
        end_easting: float = (1 - lerp_t) * object_boundary_utm[
            1
        ].easting + lerp_t * object_boundary_utm[2].easting
        end_northing: float = (1 - lerp_t) * object_boundary_utm[
            1
        ].northing + lerp_t * object_boundary_utm[2].northing

        # Move in a serpentine pattern
        if reverse_direction:
            start_easting, end_easting = end_easting, start_easting
            start_northing, end_northing = end_northing, start_northing

        lat: float
        lon: float
        lat, lon = utm.to_latlon(
            start_easting, start_northing, utm_zone_number, utm_zone_letter
        )
        await move_to(self.drone.vehicle, lat, lon, gps_dict["scan_altitude"])

        lat, lon = utm.to_latlon(
            end_easting, end_northing, utm_zone_number, utm_zone_letter
        )

        if camera is not None:
            await camera.scanning_move_to(
                self.drone.vehicle,
                lat,
                lon,
                gps_dict["scan_altitude"],
                HORIZONTAL_PHOTO_SPACING,
            )

        reverse_direction = not reverse_direction

    if camera is not None:
        camera.disconnect()
    capture_status.set()
    logging.info("Scan complete")


async def vision_odlc_logic(self: ODLC, capture_status: asyncio.Event) -> None:
    """
    Implements the vision logic for the run method of the ODLC state.

    Parameters
    ----------
    self : ODLC
        The ODLC state object.
    capture_status : asyncio.Event
        An event that is set when the drone has successfully captured all images.

    Returns
    -------
    Airdrop : State
        The next state after the drone has successfully scanned the ODLC area.

    Notes
    -----
    This method is responsible for initiating the ODLC scanning process of the drone
    and transitioning it to the Airdrop state.
    """
    camera_data_filename: str = "flight/data/camera.json"

    # Wait until camera.json exists
    logging.info("Waiting for %s to exist", camera_data_filename)
    while not Path(camera_data_filename).is_file():
        await asyncio.sleep(1)
    logging.info("Camera data file found.")

    await odlc_pipeline(
        self.flight_settings,
        "flight/data/camera.json",
        capture_status,
        "flight/data/output.json",
    )


# Setting the run_callable attribute of the ODLC class to the run function
ODLC.run_callable = run
