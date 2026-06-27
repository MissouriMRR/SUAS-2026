"""Gets the mission configuration."""

import json
from typing import TextIO, TypedDict


class SimModeConfig(TypedDict):
    """
    A configuration containing settings specific to each sim mode.

    Attributes
    ----------
    mission_data_path : str
        The path to the JSON file containing the boundary and waypoint data.
    standard_object_count : int
        The number of standard objects to attempt to find.
    """

    mission_data_path: str
    standard_object_count: int


class WindConfig(TypedDict):
    """
    A configuration containing manually entered wind speed and direction.

    Attributes
    ----------
    mean_wind_speed : float
        The mean wind speed, in meters per second.
    mean_wind_direction : float
        The mean wind direction, in degrees.
        A value of 0 represents north, and 90 represents west.
    """

    mean_wind_speed: float
    mean_wind_direction: float


class MissionConfig(TypedDict):
    """
    A configuration for a flight mission.

    Attributes
    ----------
    run_title : str
        The name for the current flight operation.
    run_description : str
        A small description for the current flight.
    real_mode_config : SimModeConfig
        Settings to use when running in real mode.
    sim_mode_config : SimModeConfig
        Settings to use when running in real mode.
    airsim_mode_config : SimModeConfig
        Settings to use when running in real mode.
    skip_waypoint : bool
        Whether to skip the waypoint state.
    skip_odlc_and_airdrop : bool
        Whether to skip the ODLC and airdrop states.
    simple_takeoff : bool
        Sets if flight will use a simple vertical takeoff.
    wind : WindConfig
        Manually entered information on the wind.
    map_output_path: str
        Path for where the map should be saved
    """

    run_title: str
    run_description: str
    real_mode_config: SimModeConfig
    sim_mode_config: SimModeConfig
    airsim_mode_config: SimModeConfig
    skip_waypoint: bool
    skip_odlc_and_airdrop: bool
    simple_takeoff: bool
    wind: WindConfig
    map_output_path : str


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
