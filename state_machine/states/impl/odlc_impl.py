"""Implements the behavior of the ODLC state."""

import asyncio
import json
import logging
from pathlib import Path
import traceback

from flight.camera import Camera

from flight.extract_gps import extract_gps, GPSData
from flight.waypoint.goto import move_to
from state_machine.state_tracker import (
    update_state,
    update_drone,
    update_flight_settings,
)
from state_machine.states.airdrop import Airdrop
from state_machine.states.mapping import Mapping
from state_machine.states.odlc import ODLC
from state_machine.states.state import State

from vision.vision_pipeline import flyover_pipeline
from vision.common.constants import CameraConfig


async def run(self: ODLC) -> State:
    """
    Implements the run method for the ODLC state.

    This method initiates the ODLC scanning process of the drone, takes pictures and transfers
    picture data to the vision code, and then transitions to the Airdrop state.

    Parameters
    ----------
    self : ODLC
        The current instance of the ODLC state.

    Returns
    -------
    Airdrop : State
        The next state after the drone has successfully scanned the ODLC area.

    Raises
    ------
    asyncio.CancelledError
        If the execution of the ODLC state is canceled.
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

    if self.flight_settings.skip_odlc_and_airdrop:
        return Mapping(self.drone, self.flight_settings)

    try:
        update_state("ODLC")
        update_drone(self.drone)
        update_flight_settings(self.flight_settings)
        logging.info("ODLC state running")

        capture_status = asyncio.Event()

        vision_task: asyncio.Task[None] = asyncio.ensure_future(vision_odlc_logic(capture_status))

        flight_task: asyncio.Task[None] = asyncio.ensure_future(find_odlcs(self, capture_status))

        logging.info("Starting check for task completion")

        while not vision_task.done():
            await asyncio.sleep(0.25)

        while not flight_task.done():
            await asyncio.sleep(0.25)

        logging.info("ODLC scan complete. State completing...")
        vision_task.cancel()
        flight_task.cancel()
    except asyncio.CancelledError as ex:
        logging.error("ODLC state canceled")
        traceback.print_exc()
        raise ex

    return Airdrop(self.drone, self.flight_settings)


async def find_odlcs(self: ODLC, capture_status: asyncio.Event) -> None:
    """
    Implements the run method for the ODLC state.

    Parameters
    ----------
    capture_status : asyncio.Event
        An event that is set when the drone has successfully captured all images.

    Returns
    -------
    Airdrop : State
        The next state after the drone has successfully scanned the ODLC area.

    Notes
    -----
    This method is responsible for initiating the ODLC scanning process of the drone.
    """

    # Initialize the camera
    if not self.flight_settings.sim_flag:
        camera: Camera | None = Camera()
    else:
        camera = None

    gps_data: GPSData = extract_gps(self.flight_settings.path_data_path)

    logging.info("Starting odlc zone flyover")

    # traverses the 3 waypoints starting at the midpoint on left to midpoint on the right
    # then to the top left corner at the rectangle
    point: int
    for point in range(len(gps_data["odlc_waypoints"])):
        take_photos: bool = True

        logging.info("Moving to ODLC scan point %d", point)

        if camera:
            await camera.odlc_move_to(
                self.drone.vehicle,
                gps_data["odlc_waypoints"][point].latitude,
                gps_data["odlc_waypoints"][point].longitude,
                gps_data["odlc_altitude"],
                take_photos,
            )
        else:
            await move_to(
                self.drone.vehicle,
                gps_data["odlc_waypoints"][point].latitude,
                gps_data["odlc_waypoints"][point].longitude,
                gps_data["odlc_altitude"],
            )

    if camera:
        camera.camera.disconnect()
    capture_status.set()
    self.drone.odlc_scan = False
    logging.info("ODLC scan complete")


async def vision_odlc_logic(capture_status: asyncio.Event) -> None:
    """
    Implements the run method for the ODLC state.

    Parameters
    ----------
    capture_status : asyncio.Event
        An event that is set when the drone has successfully captured all images.
    flight_settings : FlightSettings
        Settings for this flight.

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

    await flyover_pipeline("flight/data/camera.json", capture_status, "flight/data/output.json")


# Setting the run_callable attribute of the ODLC class to the run function
ODLC.run_callable = run
