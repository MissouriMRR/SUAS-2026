"""Gets the mission configuration."""

import json
from typing import TextIO, TypedDict


class MissionConfig(TypedDict):
    """
    A configuration for a flight mission.

    Attributes
    ----------
    run_title : str
        The name for the current flight operation
    run_description : str
        A small description for the current flight
    sim_flag : bool
        A flag representing if the connected drone is a simulation
    mission_data_path : str
        The path to the JSON file containing the boundary and waypoint data.
    skip_waypoint : bool
        Whether to skip the waypoint state.
    skip_odlc_and_airdrop : bool
        Whether to skip the ODLC and airdrop states.
    standard_object_count : int
        The number of standard objects to attempt to find.
    """

    run_title: str
    run_description: str
    sim_flag: bool
    mission_data_path: str
    skip_waypoint: bool
    skip_odlc_and_airdrop: bool
    standard_object_count: int


def get_mission_config() -> MissionConfig:
    """
    Get the mission configuration from mission_config.json

    Returns
    -------
    MissionConfig
        The mission configuration.
    """
    config_file: TextIO
    with open("mission_config.json", "r", encoding="utf-8") as config_file:
        return json.load(config_file)
