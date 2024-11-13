"""A class that contains all the needed camera functionality for the drone."""

# pylint: disable=too-many-locals
# Not worth to deal with this with the time crunch we are in

import asyncio
from datetime import datetime
import json
import logging
import math
import os
from typing import TypedDict

import dronekit
import gphoto2

from flight.waypoint.goto import move_to
from flight.waypoint.calculate_distance import calculate_distance

WAYPOINT_TOLERANCE: int = 1  # in meters


class PhotoInfo(TypedDict):
    """Basic information about a photo."""

    focal_length: float
    """The focal length, in millimeters."""

    rotation_deg: tuple[float, float, float]
    """The roll, pitch, and yaw, in degrees."""

    drone_coordinates: tuple[float, float]
    """The latitude and longitude, in degrees"""

    altitude_f: float
    """The altitude, in meters."""


class Camera:
    """
    Initialize a new Camera object to control the Sony RX100-VII camera on the drone

    Attributes
    ----------
    camera : gphoto2.Camera
        The gphoto2 camera object.
    session_id : int
        The session id for the current session.
        This will start at 0 the first time pictures are taken on a given day.
        Will then increment by 1 for each session on a given day.
    image_id : int
        The image id for the current image.
        Starts at 0 and increments by 1 for each image taken.

    Methods
    -------
    capture_photo(path: str = f"{os.getcwd()}/images/")
        Capture a photo and save it to the specified path.
        The default path is the images folder in the current working directory.
        The file name will be the file format attribute.
        Returns the file name and the file path.
    mapping_move_to(
        drone: dronekit.Vehicle,
        latitude: float,
        longitude: float,
        altitude: float,
        interval: float,
        heading: float,
    )
        Move the drone to the specified latitude, longitude, and altitude.
        Takes photos along the way.
    odlc_move_to(
        drone: dronekit.Vehicle,
        latitude: float,
        longitude: float,
        altitude: float,
        take_photos: bool,
        heading: float,
    )
        Move the drone to the specified latitude, longitude, and altitude.
        Takes a photo at the end if take_photos is True.
    _get_photo_information(drone: dronekit.Vehicle)
        Get the current camera information to associate with a photo.
    """

    def __init__(self) -> None:
        self.camera: gphoto2.Camera = gphoto2.Camera()
        self.camera.init()

        self.session_id: int = 0
        if os.path.exists(f"{os.getcwd()}/images/"):
            for file in os.listdir(f"{os.getcwd()}/images/"):
                if file.startswith(f"{datetime.now().strftime('%Y%m%d')}"):
                    if int(file.split("_")[1]) >= self.session_id:
                        self.session_id = int(file.split("_")[1]) + 1

        self.image_id: int = 0

        logging.info("Camera initialized")

    async def capture_photo(self, path: str = f"{os.getcwd()}/images/") -> tuple[str, str]:
        """
        Capture a photo and save it to the specified path.

        Parameters
        ----------
        path : str, optional
            The path to save the image to, by default f"{os.getcwd()}/images/"

        Returns
        -------
        tuple[str, str]
            The file name and the file path.
        """
        # If the images folder doesn't exist, we can't save images.
        # So we have to make sure the images folder exists.
        os.makedirs(path, mode=0o777, exist_ok=True)

        file_path = self.camera.capture(gphoto2.GP_CAPTURE_IMAGE)
        while True:
            event_type, _event_data = self.camera.wait_for_event(100)
            if event_type == gphoto2.GP_EVENT_CAPTURE_COMPLETE:
                photo_name: str = (
                    f"{datetime.now().strftime('%Y%m%d')}_{self.session_id}_{self.image_id:04d}.jpg"
                )

                cam_file = gphoto2.check_result(
                    gphoto2.gp_camera_file_get(
                        self.camera,
                        file_path.folder,
                        file_path.name,
                        gphoto2.GP_FILE_TYPE_NORMAL,
                    )
                )
                target_name: str = f"{path}{photo_name}"
                cam_file.save(target_name)
                self.image_id += 1
                logging.info("Image is being saved to %s", target_name)
                return target_name, photo_name

    async def mapping_move_to(
        self,
        drone: dronekit.Vehicle,
        latitude: float,
        longitude: float,
        altitude: float,
        interval: float,
        heading: float = 0,
    ) -> None:
        """
        Moves to the drone to the requested waypoint while taking photos for the mapping state.

        Parameters
        ----------
        drone : dronekit.Vehicle
            The drone object with the camera.
        latitude : float
            The requested latitude to move to, in degrees.
        longitude : float
            The requested longitude to move to, in degrees.
        altitude : float
            The requested altitude to go to, in meters.
        interval : float
            The interval, in meters, at which to take photos.
        heading : float, default 0
            The yaw in which the camera should point, in degrees (0 is north, 90 is west).
        """
        info: dict[str, PhotoInfo] = {}

        drone.gimbal.rotate(
            -drone.attitude.pitch - 90, 0, heading  # pitch is relative to the drone
        )
        await asyncio.sleep(1.0)

        photo_info: PhotoInfo = await self._get_photo_info(drone)
        file_path: str
        _, file_path = await self.capture_photo(f"{os.getcwd()}/mapping_images/")
        point: dict[str, PhotoInfo] = {file_path: photo_info}
        info.update(point)

        goto_task: asyncio.Task[None] = asyncio.ensure_future(
            move_to(
                drone,
                latitude,
                longitude,
                altitude,
                airspeed=5.0,
                tolerance=WAYPOINT_TOLERANCE,
            )
        )

        start_pos: dronekit.LocationGlobalRelative = drone.location.global_relative_frame

        start_lat: float = start_pos.lat
        start_lon: float = start_pos.lon
        start_alt: float = start_pos.alt

        next_interval_count: int = 1
        while not goto_task.done():
            # Keep gimbal pointed straight down
            drone.gimbal.rotate(
                -drone.attitude.pitch - 90, 0, heading  # pitch is relative to the drone
            )

            position: dronekit.LocationGlobalRelative = drone.location.global_relative_frame

            drone_lat: float = position.lat
            drone_long: float = position.lon
            drone_alt: float = position.alt

            distance: float = calculate_distance(
                drone_lat, drone_long, drone_alt, start_lat, start_lon, start_alt
            )

            if distance >= next_interval_count * interval:
                next_interval_count += 1
                photo_info = await self._get_photo_info(drone)
                _, file_path = await self.capture_photo(f"{os.getcwd()}/mapping_images/")
                point = {file_path: photo_info}
                info.update(point)

            await asyncio.sleep(0.25)

        drone.gimbal.rotate(
            -drone.attitude.pitch - 90, 0, heading  # pitch is relative to the drone
        )
        await asyncio.sleep(1.0)

        photo_info = await self._get_photo_info(drone)
        _, file_path = await self.capture_photo(f"{os.getcwd()}/mapping_images/")
        point = {file_path: photo_info}
        info.update(point)

        current_photos: dict[str, PhotoInfo] = {}
        if os.path.exists("flight/data/mapping_photos.json"):
            with open("flight/data/mapping_photos.json", "r", encoding="utf8") as current_data:
                try:
                    current_photos = json.load(current_data)
                except json.JSONDecodeError:
                    pass

        with open("flight/data/mapping_photos.json", "w", encoding="ascii") as camera:
            json.dump(current_photos | info, camera)

    async def odlc_move_to(
        self,
        drone: dronekit.Vehicle,
        latitude: float,
        longitude: float,
        altitude: float,
        take_photos: bool,
        heading: float = 0,
    ) -> None:
        """
        This function takes in a latitude, longitude and altitude and autonomously
        moves the drone to that waypoint. It will take a photo at the end if passed
        True in take_photos and add the point and name of photo to a json.

        Parameters
        ----------
        drone : dronekit.Vehicle
            The drone object with the camera.
        latitude : float
            The requested latitude to move to, in degrees.
        longitude : float
            The requested longitude to move to, in degrees.
        altitude : float
            The requested altitude to go to, in meters.
        take_photos : bool
            Whether to take a photo once the waypoint has been reached.
        heading : float, default 0
            The yaw in which the camera should point, in degrees (0 is north, 90 is west).
        """
        info: dict[str, PhotoInfo] = {}

        await move_to(
            drone,
            latitude,
            longitude,
            altitude,
            airspeed=5.0,
            tolerance=WAYPOINT_TOLERANCE,
        )

        if take_photos:
            # Point the gimbal straight down
            drone.gimbal.rotate(
                -drone.gimbal.pitch - 90,  # pitch is relative to the drone
                drone.gimbal.roll,
                heading,
            )

        await asyncio.sleep(2)

        if not take_photos:
            return

        photo_info: PhotoInfo = await self._get_photo_info(drone)
        file_path: str
        _, file_path = await self.capture_photo()
        point: dict[str, PhotoInfo] = {file_path: photo_info}
        info.update(point)

        current_photos: dict[str, PhotoInfo] = {}
        if os.path.exists("flight/data/camera.json"):
            with open("flight/data/camera.json", "r", encoding="utf8") as current_data:
                try:
                    current_photos = json.load(current_data)
                except json.JSONDecodeError:
                    pass

        with open("flight/data/camera.json", "w", encoding="ascii") as camera:
            json.dump(current_photos | info, camera)

    async def _get_photo_info(self, drone: dronekit.Vehicle) -> PhotoInfo:
        """
        Gets the current camera information based on a drone's position.

        Parameters
        ----------
        drone : dronekit.Vehicle
            The drone whose position will be stored in the result.

        Returns
        -------
        PhotoInfo
            Camera information to be associated with a photo.
        """
        location: dronekit.LocationGlobalRelative = drone.location.global_relative_frame

        attitude: dronekit.Attitude = drone.attitude
        roll_deg: float = math.degrees(attitude.roll)
        pitch_deg: float = math.degrees(attitude.pitch)
        yaw_deg: float = math.degrees(attitude.yaw)

        return {
            "focal_length": 24,
            "rotation_deg": (
                roll_deg,
                pitch_deg,
                yaw_deg,
            ),
            "drone_coordinates": (location.lat, location.lon),
            "altitude_f": location.alt,
        }
