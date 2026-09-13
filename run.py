#!/usr/bin/env python3
"""
Main runnable file for the codebase

If running for competition, make sure that the following is set:
- Bottle data in vision/competition_inputs/bottle_data.json
- Waypoints in flight/data/waypoint_data.json

Pass --resume to continue from the mission progress recorded in
state_machine/progress.json
"""

import asyncio
import logging
from typing import Annotated

import typer

from state_machine.flight_manager import FlightManager
from state_machine.flight_settings import FlightSettings

app = typer.Typer()


@app.command()
def run(
    resume: Annotated[
        bool, typer.Option(help="Resume mission from progress.json")
    ] = False,
    _sim: Annotated[
        bool, typer.Option("-s", "--sim", help="Run in simulation mode")
    ] = False,
    _airsim: Annotated[
        bool, typer.Option("-a", "--airsim", help="Run in AirSim mode")
    ] = False,
) -> None:
    try:
        logging.basicConfig(level=logging.INFO)
        logging.info("Starting processes")
        flight_manager: FlightManager = FlightManager()
        asyncio.run(
            flight_manager.run_manager(FlightSettings.from_mission_config(), resume)
        )
    finally:
        logging.info("Done!")


if __name__ == "__main__":
    app()
