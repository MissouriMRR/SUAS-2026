"""Tests the mapping pipeline, using photos and data from the standard paths."""

import asyncio
import logging

from state_machine.flight_settings import FlightSettings
from vision.mapping_pipeline import mapping_pipeline

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    flight_settings : FlightSettings = FlightSettings.from_mission_config()
    asyncio.run(
        mapping_pipeline(
            "flight/data/camera.json",
            "images",
            flight_settings.odm_ip,
            flight_settings.odm_port,
            "vision/mapping/results",
        )
    )
