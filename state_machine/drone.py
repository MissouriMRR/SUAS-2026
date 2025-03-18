"""Defines the Drone class for the state machine."""

import asyncio
import json
import logging

import dronekit

from flight.waypoint.calculate_distance import calculate_distance
from state_machine.flight_settings import SimMode


class Drone:
    """
    A drone for the state machine to control.
    This class is a wrapper around the dronekit Vehicle class,
    and will be passed around to each state.
    Data can be stored in this class to be shared between states.

    Attributes
    ----------
    address : str
        The address used to connect to the drone.
    baud : int | None
        The baud rate, or None to use the default.
    is_connected
    odlc_scan : bool
        A boolean to tell if the odlc zone needs to be scanned, used the
        first run and if odlc needs to be scanned any other time
    vehicle
    _vehicle : dronekit.Vehicle | None
        The Dronekit Vehicle object that controls the drone, or None if a connection
        hasn't been made yet.

    Methods
    -------
    __init__(connection_string: str) -> None
        Initialize a new Drone object, but do not connect to a drone.
    arm(self) -> Awaitable[none]
        Arm the drone.
    close(self) -> Awaitable[none]
        Close the owned DroneKit Vehicle object.
    connect_drone(self) -> Awaitable[None]
        Connect to a drone.
    is_connected(self) -> bool
        Checks if a drone has been connected to.
    takeoff(self, takeoff_alt: float) -> Awaitable[None]
        Takeoff vertically to the passed altitude.
    use_settings(self, sim_mode: SimMode) -> None
        Modify the connection settings based on the given simulation mode.
    vehicle(self) -> dronekit.Vehicle
        Get the Dronekit Vehicle object owned by this Drone object.
    """

    def __init__(self, address: str = "", baud: int | None = None) -> None:
        """
        Initialize a new Drone object, but do not connect to a drone.

        Parameters
        ----------
        address : str, default ""
            The address of the drone to connect to when the `connect_drone()`
            method is called.
        baud : int, default None
            The baud rate, or None to use the default.
        """
        self._vehicle: dronekit.Vehicle | None = None
        self.address: str = address
        self.baud: int | None = baud
        self.odlc_scan: bool = True

        with open("flight/data/attempted_drops.json", "w", encoding="utf8") as file:
            json.dump({}, file)

    @property
    def is_connected(self) -> bool:
        """Checks if a drone has been connected to.

        Returns
        -------
        bool
            Whether this Drone object has connected to a drone.
        """
        return self._vehicle is not None

    @property
    def vehicle(self) -> dronekit.Vehicle:
        """Get the DroneKit Vehicle object owned by this Drone object.

        Returns
        -------
        dronekit.Vehicle
            The Vehicle object owned by this Drone object.

        Raises
        ------
        AttributeError
            If a connection hasn't been made yet.
        """
        vehicle: dronekit.Vehicle | None = self._vehicle
        if vehicle is None:
            raise RuntimeError("we haven't connected to the drone yet")
        return vehicle

    async def connect_drone(self) -> None:
        """Connect to a drone. This operation is idempotent.

        Raises
        ------
        RuntimeError
            If no connection address has been set.
        """
        if self.is_connected:
            return

        if len(self.address) == 0:
            raise RuntimeError("no connection address specified")

        logging.info("Waiting for drone to connect...")
        self._vehicle = (
            dronekit.connect(self.address, wait_ready=True)
            if self.baud is None
            else dronekit.connect(self.address, wait_ready=True, baud=self.baud)
        )
        logging.info("Drone discovered!")

    def remove_arming_check(self) -> None:
        """

        For use with airsim

        """
        self.vehicle.parameters["ARMING_CHECK"] = 0

    async def arm(self) -> None:
        """
        Arm the drone
        """

        logging.info("Waiting for vehicle to intialize...")
        while not self.vehicle.is_armable:
            # Vehicle is not ready to accept code
            await asyncio.sleep(0.5)

        self.vehicle.mode = dronekit.VehicleMode("GUIDED")
        self.vehicle.armed = True

        # Confirm vehicle is properly armed
        logging.info("Waiting for arming...")
        while not self.vehicle.armed or self.vehicle.mode.name != "GUIDED":
            await asyncio.sleep(0.5)

    async def takeoff(self, takeoff_alt: float) -> None:
        """
        Takeoff vertically to the passed altitude

        Parameters
        ----------
        takeoff_alt: float
            Altitude to reach in meters
        """
        logging.info("Using takeoff altitude of %f m", takeoff_alt)
        self.vehicle.simple_takeoff(takeoff_alt + 1.5)  # Add 5ft for margin of error

        # Verify vehicle reaches target altitude
        while self.vehicle.location.global_relative_frame.alt < takeoff_alt:
            await asyncio.sleep(0.5)
        logging.info("Reached target altitude (%f m).", takeoff_alt)

    async def return_to_launch(self) -> None:
        """
        Method to move vehicle above home location, then descend vertically
        """
        home_loc = dronekit.LocationGlobalRelative(
            self.vehicle.home_location.lat, self.vehicle.home_location.lon, 23
        )  # Min alt should be in constants file
        self.vehicle.simple_goto(home_loc)
        logging.info("Moving to home lat/lon...")
        while (
            calculate_distance(
                self.vehicle.location.global_relative_frame.lat,
                self.vehicle.location.global_relative_frame.lon,
                self.vehicle.location.global_relative_frame.alt,
                home_loc.lat,
                home_loc.lon,
                home_loc.alt,
            )
            > 1
        ):  # Get within 1 meter above home location
            await asyncio.sleep(0.5)
        self.vehicle.mode = dronekit.VehicleMode("RTL")
        logging.info("Descending...")
        while (
            self.vehicle.location.global_relative_frame.alt > 0.2
        ):  # Ensure drone gets within 8in above ground
            await asyncio.sleep(0.5)
        logging.info("Reached ground.")

    async def close(self) -> None:
        """Close the owned DroneKit Vehicle object."""
        self.vehicle.close()

    def use_settings(self, sim_mode: SimMode) -> None:
        """Modify the connection settings based on the given simulation mode.

        Parameters
        ----------
        sim_mode : SimMode
            The simulation mode.

        Raises
        ------
        ValueError
            If `sim_mode` is not a valid SimMode.
        """
        match sim_mode:
            case SimMode.REAL:
                self.address = "/dev/ttyFTDI"
                self.baud = 921600
            case SimMode.SIM:
                self.address = "tcp:127.0.0.1:5762"
                self.baud = None
            case SimMode.AIRSIM:
                self.address = "127.0.0.1:14030"
                self.baud = None
            case _:
                raise ValueError("invalid sim mode")
