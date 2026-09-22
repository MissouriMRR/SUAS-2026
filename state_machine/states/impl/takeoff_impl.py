"""Implements the behavior of the Takeoff state."""

import asyncio
import logging
from typing import Final

from flight.extract_gps import extract_gps
from state_machine.states.airdrop import Airdrop
from state_machine.states.mapping import Mapping
from state_machine.states.odlc import ODLC
from state_machine.states.state import State
from state_machine.states.takeoff import Takeoff
from state_machine.states.waypoint import Waypoint

RESUME_STATES: Final[dict[str, type[State]]] = {
    "Waypoint": Waypoint,
    "ODLC": ODLC,
    "Mapping": Mapping,
    "Airdrop": Airdrop,
}

logger = logging.getLogger(__name__)


async def run(self: Takeoff) -> State:
    """
    Implements the run method for the Takeoff state.

    This method initiates the drone takeoff process and transitions to the Waypoint
    state, or, if the mission is being resumed, to whichever state the mission
    left off at.

    Returns
    -------
    State
        The Waypoint state, or the state a resumed mission is continuing from.

    Raises
    ------
    asyncio.CancelledError
        If the execution of the Takeoff state is canceled.

    Notes
    -----
    This method is responsible for taking off the drone and transitioning it to the
    Waypoint state, which represents the navigation phase to reach a specified waypoint.
    Takeoff is where a resumed mission rejoins the state machine, because the drone
    has to get back in the air before it can continue from wherever it left off.

    """
    try:
        logger.info("Takeoff state running")

        # Set takeoff altitude to the minimum allowed altitude, plus one meter
        takeoff_altitude: float = (
            extract_gps(self.flight_settings.mission_data_path)["altitude_limits"][0]
            + 1.0
        )
        await self.drone.takeoff(takeoff_altitude)

        return next_state_type(self.drone.progress.state)(
            self.drone, self.flight_settings
        )
    except asyncio.CancelledError:
        logger.error("Takeoff state canceled")
        raise
    finally:
        pass


def next_state_type(resume_state: str | None) -> type[State]:
    """
    Get the state to fly after takeoff.

    Parameters
    ----------
    resume_state : str | None
        The name of the state a resumed mission is continuing from, or None when
        the mission is being flown from the beginning.

    Returns
    -------
    type[State]
        The state class to transition into. Falls back to Waypoint when no
        resume state was given, or when the name given isn't one a mission can
        be resumed from.
    """
    if resume_state is None:
        return Waypoint

    state_type: type[State] | None = RESUME_STATES.get(resume_state)
    if state_type is None:
        logger.warning(
            "Cannot resume into unknown state %s, continuing from Waypoint instead.",
            resume_state,
        )
        return Waypoint

    logger.info("Resuming mission from the %s state.", resume_state)
    return state_type


# Setting the run_callable attribute of the Takeoff class to the run function
Takeoff.run_callable = run

__all__ = ["Takeoff"]
