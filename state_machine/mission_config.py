import json
from typing import TextIO, TypedDict


class MissionConfig(TypedDict):
    run_title: str
    run_description: str
    sim_flag: bool
    mission_data_path: str
    skip_waypoint: bool
    skip_odlc_and_airdrop: bool
    standard_object_count: int


def get_mission_config() -> MissionConfig:
    config_file: TextIO
    with open("mission_config.json", "r", encoding="utf-8") as config_file:
        return json.load(config_file)
