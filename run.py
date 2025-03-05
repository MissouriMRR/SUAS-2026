#!/usr/bin/env python3
"""
Main runnable file for the codebase

If running for competition, make sure that the following is set:
- Bottle data in vision/competition_inputs/bottle_data.json
- Waypoints in flight/data/waypoint_data.json
"""

import asyncio
import logging
import sys
from state_machine.flight_manager import FlightManager
from vision.common.constants import SENSOR_WIDTH, SENSOR_HEIGHT

if __name__ == "__main__":
    # Run multiprocessing function
    try:
        SIM_FLAG: bool = False
        AIRSIM_FLAG: bool = False
        logging.basicConfig(level=logging.INFO)
        logging.info("Starting processes")
        flight_manager: FlightManager = FlightManager()
        if "-s" in sys.argv:
            SIM_FLAG = True
        elif "-a" in sys.argv:
            AIRSIM_FLAG = True
            SENSOR_HEIGHT = 10
            SENSOR_WIDTH = 10

        asyncio.run(flight_manager.run_manager(SIM_FLAG, AIRSIM_FLAG))
    finally:
        logging.info("Done!")
