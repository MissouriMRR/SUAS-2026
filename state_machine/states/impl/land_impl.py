"""Implements the behavior of the Land state."""

import asyncio
import logging

from flight.extract_gps import extract_gps
from state_machine.state_tracker import (
    update_drone,
    update_flight_settings,
    update_state,
)
from state_machine.states.land import Land


async def run(self: Land) -> None:
    """
    Implements the run method for the Land state.

    This method initiates the landing process of the drone and transitions to the Start state.

    Returns
    -------
    Start : State
        The next state after the drone has successfully landed.

    Notes
    -----
    This method is responsible for initiating the landing process of the drone and transitioning
    it back to the Start state, preparing for a new flight.

    """
    try:
        update_state("Land")
        update_drone(self.drone)
        update_flight_settings(self.flight_settings)
        logging.info("Land state running")

        # Get minimum altitude before landing
        gps_dict = extract_gps(self.flight_settings.mission_data_path)
        min_alt = gps_dict["altitude_limits"][0]

        # Instruct the drone to land
        self.drone.vehicle.airspeed = 20
        await self.drone.return_to_launch(min_alt + 5)

        # Wait for the drone to disarm
        while self.drone.vehicle.armed:
            await asyncio.sleep(0.1)

        logging.info("Land state complete.")
        return
    except asyncio.CancelledError:
        logging.error("Land state canceled")
        raise


# Setting the run_callable attribute of the Land class to the run function
Land.run_callable = run
