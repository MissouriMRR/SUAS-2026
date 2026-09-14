"""Implements the behavior of the Mapping state."""

# pylint: disable=too-many-locals

import asyncio
import logging
from pathlib import Path

from state_machine.drone import Drone
from state_machine.states.airdrop import Airdrop
from state_machine.states.mapping import Mapping
from state_machine.states.state import State
from vision.common import camera_config
from vision.mapping_pipeline import mapping_pipeline


async def run(self: Mapping) -> State:
    """
    Implements the run method for the Mapping state.

    This method captures photos of the mapping area and then transitions to the Airdrop state.
    If YOLO inference from the ODLC state is still running, the code will wait for it to finish
    before transitioning to the Airdrop state.

    Returns
    -------
    Airdrop : State
        The next state after the drone has successfully landed.
    """

    camera_config.update_sim_mode(self.flight_settings.sim_mode)

    try:
        self.record_progress()

        logging.info("Mapping")

        asyncio.ensure_future(
            vision_mapping_logic(
                self.drone,
                self,
                self.flight_settings.map_output_path,
                self.flight_settings.odm_ip,
                self.flight_settings.odm_port,
            )
        )

        logging.info("Mapping task scheduled")
    except asyncio.CancelledError:
        logging.error("Mapping state canceled")
        raise

    # Need to wait for YOLO processing to finish, or we may not have objects to drop at
    if (
        not self.flight_settings.yolo_status.is_set()
        and not self.flight_settings.skip_odlc_and_airdrop
    ):
        logging.info("Waiting for YOLO processing to finish...")
        await self.flight_settings.yolo_status.wait()
        logging.info("YOLO processing finished.")

    return Airdrop(self.drone, self.flight_settings)


async def vision_mapping_logic(
    drone: Drone, _: Mapping, map_output_path: str, odm_ip: str, odm_port: int
) -> None:
    """
    Implements the vision logic for the Mapping state.

    Parameters
    ----------
    drone : Drone
        The drone object, used to keep logging mission time while mapping
        processing continues after the state machine itself has exited.
    _ : Mapping
        The Mapping state object.
    map_output_path : str
        The path to save the map output.
    odm_ip : str
        The hostname or IP address of the ODM node used to generate the map.
    odm_port : int
        The port of the ODM node used to generate the map.
    """

    camera_data_filename: str = "flight/data/camera.json"

    logging.info("Waiting for %s to exist", camera_data_filename)
    while not Path(camera_data_filename).is_file():
        await asyncio.sleep(1)
    logging.info("Mapping camera data file found.")

    mapping_task: asyncio.Task[None] = asyncio.ensure_future(
        mapping_pipeline(
            camera_data_filename,
            "images",
            odm_ip,
            odm_port,
            map_output_path,
        )
    )

    while not mapping_task.done():
        mission_time: float = (
            drone.flight_time if drone.flight_start_time is not None else drone.last_flight_time
        )
        logging.info(
            "Mapping still processing. Mission time: %d:%05.2f",
            int(mission_time // 60),
            mission_time % 60,
        )
        await asyncio.sleep(5)

    await mapping_task


# Setting the run_callable attribute of the Mapping class to the run function
Mapping.run_callable = run
