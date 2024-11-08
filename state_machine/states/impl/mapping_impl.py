"""Implements the behavior of the Mapping state."""

import asyncio
import logging

import dronekit

from state_machine.state_tracker import (
    update_state,
    update_drone,
    update_flight_settings,
)
from state_machine.states.mapping import Mapping


async def run(self: Mapping) -> None:
    """
    Implements the run method for the Mapping state.

    This method captures photos of the mapping area and then transitions to the ODLC state.

    Returns
    -------
    ODLC : State
        The next state after the drone has successfully landed.
    """
    try:
        update_state("Mapping")
        update_drone(self.drone)
        update_flight_settings(self.flight_settings)

        logging.info("Mapping")

        # TODO: Implement the mapping state
        raise NotImplementedError("Mapping state not implemented yet")

        logging.info("Mapping state complete.")
    except asyncio.CancelledError as ex:
        logging.error("Mapping state canceled")
        raise ex


# Setting the run_callable attribute of the Mapping class to the run function
Mapping.run_callable = run
