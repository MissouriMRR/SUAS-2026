"""Contains functions to interface with camera_config.json."""

import json

from state_machine.flight_settings import SimMode
from vision.common.constants import CameraConfig


def update_sim_mode(sim_mode: SimMode) -> None:
    """Update camera_config.json based on the sim mode passed in.

    Parameters
    ----------
    sim_mode : SimMode
        The sim mode to update camera_config.json with.
    """
    with open("vision/common/camera_config.json", "r", encoding="ascii") as file:
        camera_config: CameraConfig = json.load(file)
        camera_config["airsim_flag"] = sim_mode is SimMode.AIRSIM

    with open("vision/common/camera_config.json", "w", encoding="ascii") as file:
        json.dump(camera_config, file)
