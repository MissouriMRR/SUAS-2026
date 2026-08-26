"""Implements the behavior of the Mapping state."""

# pylint: disable=too-many-locals

import asyncio
import logging
from pathlib import Path

from state_machine.state_tracker import (
    update_drone,
    update_flight_settings,
    update_state,
)
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
        update_state("Mapping")
        update_drone(self.drone)
        update_flight_settings(self.flight_settings)

        logging.info("Mapping")

        asyncio.ensure_future(
            vision_mapping_logic(self, self.flight_settings.map_output_path)
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


async def vision_mapping_logic(_: Mapping, map_output_path: str) -> None:
    """
    Implements the vision logic for the Mapping state.

    Parameters
    ----------
    _ : Mapping
        The Mapping state object.
    map_output_path : str
        The path to save the map output.
    """

    camera_data_filename: str = "flight/data/camera.json"

    logging.info("Waiting for %s to exist", camera_data_filename)
    while not Path(camera_data_filename).is_file():
        await asyncio.sleep(1)
    logging.info("Mapping camera data file found.")

    await mapping_pipeline(camera_data_filename, "images", map_output_path)


# Setting the run_callable attribute of the Mapping class to the run function
Mapping.run_callable = run
