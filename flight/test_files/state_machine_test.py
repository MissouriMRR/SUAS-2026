"""Runs the state machine and kill switch in separate processes in order to test them."""

import asyncio
from state_machine.flight_manager import FlightManager
from state_machine.flight_settings import FlightSettings


async def run_test(flight_settings: FlightSettings) -> None:
    """
    Run the state machine.

    Parameters
    ----------
    flight_settings : FlightSettings
        The flight settings to use.
    """
    await FlightManager().run_manager(flight_settings)


if __name__ == "__main__":
    asyncio.run(run_test(FlightSettings.from_mission_config()))
