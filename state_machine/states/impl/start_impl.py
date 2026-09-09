"""Implements the behavior of the Start state."""

import asyncio
import logging

from state_machine.state_tracker import (
    update_drone,
    update_flight_settings,
    update_state,
)
from state_machine.states.start import Start
from state_machine.states.state import State
from state_machine.states.takeoff import Takeoff

logger = logging.getLogger(__name__)


async def run(self: Start) -> State:
    """
    Implements the run method for the Start state.

    This method establishes a connection with the drone, waits for the drone to be
    discovered, ensures the drone has a global position estimate, arms the drone,
    and transitions to the Takeoff state.

    Returns
    -------
    Takeoff : State
        The next state after the Start state has successfully run.

    Notes
    -----
    This method is responsible for initializing the drone and transitioning it to the
    Takeoff state, which is the next step in the state machine.

    """
    try:
        update_state("Start")
        update_drone(self.drone)
        update_flight_settings(self.flight_settings)
        logger.info("Start state running")

        await self.drone.connect_drone()

        logger.info("Waiting for drone to be armable")
        while not self.drone.vehicle.is_armable:
            await asyncio.sleep(0.5)

        if not self.flight_settings.skip_waypoint:
            self.drone.init_waypoint_mission(self.flight_settings)

        message_1: str = "Waiting for user input to continue... "
        message_2: str = "(press enter when ready) "
        input(f"\x1b[38;2;255;255;0m{message_1}\x1b[3m{message_2}\x1b[0m")

        await self.drone.arm()

        logger.info("Start state complete")
        return Takeoff(self.drone, self.flight_settings)
    except asyncio.CancelledError:
        logger.error("Start state canceled")
        raise
    finally:
        pass


# Setting the run_callable attribute of the Start class to the run function
Start.run_callable = run
