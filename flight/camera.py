"""A class that contains all the needed camera functionality for the drone."""

# pylint: disable=too-many-locals
import asyncio
import json
import logging
import math
import os
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, override

import aiofiles  # type: ignore
import aiohttp
import airsim
import dronekit
from siyi_sdk.siyi_sdk import SIYISDK

from flight.waypoint.calculate_distance import calculate_distance
from flight.waypoint.goto import move_to
from vision.common.constants import CameraParameters

WAYPOINT_TOLERANCE: int = 1  # in meters
logger = logging.getLogger(__name__)


class Camera(ABC):
    """
    Initialize a new Camera object to control the SIYI A8 Mini gimbal camera on the drone

    Attributes
    ----------
    camera : SIYISDK
        The object that controls the camera.
    session_id : int
        The session id for the current session.
        This will start at 0 the first time pictures are taken on a given day.
        Will then increment by 1 for each session on a given day.
    image_id : int
        The image id for the current image.
        Starts at 0 and increments by 1 for each image taken.
    base_api_url : str
        The base API URL for the camera

    Methods
    -------
    capture_photo(path: str = f"{os.getcwd()}/images/")
        Capture a photo and save it to the specified path.
        The default path is the images folder in the current working directory.
        The file name will be the file format attribute.
        Returns the file name and the file path.
    scanning_move_to(
        drone: dronekit.Vehicle,
        latitude: float,
        longitude: float,
        altitude: float,
        interval: float
    )
        Move the drone to the specified latitude, longitude, and altitude.
        Takes photos along the way.
    _get_camera_parameters(drone: dronekit.Vehicle)
        Get the current camera information to associate with a photo.
    """

    def __init__(self) -> None:
        self.image_id: int = 0
        self.session_id: int = 0

    @abstractmethod
    async def capture_photo(
        self, path: str = f"{os.getcwd()}/images/"
    ) -> tuple[str, str] | None:
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
        return None

    @abstractmethod
    async def scanning_move_to(
        self,
        drone: dronekit.Vehicle,
        latitude: float,
        longitude: float,
        altitude: float,
        interval: float,
    ) -> None:
        """
        Moves to the drone to the requested waypoint while taking photos for the ODLC and Mapping states.

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
        """

    @abstractmethod
    async def _get_camera_parameters(
        self, drone: dronekit.Vehicle
    ) -> CameraParameters | None:
        """
        Gets the current camera information based on a drone's position_pitch.

        Parameters
        ----------
        drone : dronekit.Vehicle
            The drone whose position will be stored in the result.

        Returns
        -------
        CameraParameters
            Camera information to be associated with a photo.
        """
        return None

    def disconnect(self) -> None:
        """
        Disconnects the IRL Camera.
        """


class CameraIRL(Camera):
    """
    Initialize a new Camera object to control the SIYI A8 Mini gimbal camera on the drone

    Attributes
    ----------
    camera : SIYISDK
        The object that controls the camera.
    session_id : int
        The session id for the current session.
        This will start at 0 the first time pictures are taken on a given day.
        Will then increment by 1 for each session on a given day.
    image_id : int
        The image id for the current image.
        Starts at 0 and increments by 1 for each image taken.
    base_api_url : str
        The base API URL for the camera

    Methods
    -------
    capture_photo(path: str = f"{os.getcwd()}/images/")
        Capture a photo and save it to the specified path.
        The default path is the images folder in the current working directory.
        The file name will be the file format attribute.
        Returns the file name and the file path.
    scanning_move_to(
        drone: dronekit.Vehicle,
        latitude: float,
        longitude: float,
        altitude: float,
        interval: float
    )
        Move the drone to the specified latitude, longitude, and altitude.
        Takes photos along the way.
    _get_camera_parameters(drone: dronekit.Vehicle)
        Get the current camera information to associate with a photo.
    """

    def __init__(self) -> None:
        super().__init__()
        logger.info("Connecting to camera...")
        self.camera: SIYISDK = SIYISDK()
        if not self.camera.connect(maxRetries=10):
            logger.error("Failed to connect to the camera")

        self.session_id: int = 0
        if os.path.exists(f"{os.getcwd()}/images/"):
            for file in os.listdir(f"{os.getcwd()}/images/"):
                if (
                    file.startswith(
                        f"{datetime.now(UTC).astimezone().strftime('%Y%m%d')}"
                    )
                    and int(file.split("_")[1]) >= self.session_id
                ):
                    self.session_id = int(file.split("_")[1]) + 1

        self.image_id: int = 0
        self.base_api_url: str = "http://192.168.144.25:82//cgi-bin/media.cgi"

        # Set gimbal to point straight down for taking pictures
        if not self.camera.requestSetAngles(0, -90):
            logger.error("Failed to set gimbal angle")

        logger.info("IRL Camera initialized")

    @override
    async def capture_photo(
        self, path: str = f"{os.getcwd()}/images/"
    ) -> tuple[str, str]:
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

        took_photo: bool = self.camera.requestPhoto()
        if not took_photo:
            logger.error("Failed to take photo")
            raise ValueError("Failed to take photo")

        # Need to wait a bit after sending the capture command to allow the
        # gimbal to process/save the photo, else we run into a race condition
        await asyncio.sleep(0.5)

        # Retrieve the image from the gimbal SD card
        session: aiohttp.ClientSession
        async with aiohttp.ClientSession() as session:
            json_data: dict[str, Any]

            async with session.get(
                f"{self.base_api_url}/api/v1/getmediacount?media_type=0&path=101SIYI_IMG"
            ) as response:
                json_data = await response.json()
                if response.status != 200:
                    logger.error(
                        "Failed to get media count. Response data: %s", json_data
                    )
                    raise ValueError("Failed to get media count")
                media_count: int = json_data["data"]["count"]

            # Have to request for all images in the directory due to a SIYI firmware bug
            async with session.get(
                f"{self.base_api_url}/api/v1/getmedialist?"
                f"media_type=0&path=101SIYI_IMG&start=0&count={media_count}"
            ) as response:
                json_data = await response.json()
                if response.status != 200:
                    logger.error(
                        "Failed to get media list. Response data: %s", json_data
                    )
                    raise ValueError("Failed to get media list")
                try:
                    media_path: str = json_data["data"]["list"][-1]["url"]
                except IndexError as exc:
                    logger.error(
                        "Failed to get media path. Response data: %s", json_data
                    )
                    raise ValueError("Failed to get media path") from exc

            async with session.get(media_path) as response:
                binary_data: bytes = await response.read()
                if response.status != 200:
                    logger.error("Failed to get media. Response data: %s", binary_data)
                    raise ValueError("Failed to get media")
                photo_name: str = f"{datetime.now(UTC).astimezone().strftime('%Y%m%d')}_{self.session_id}_{self.image_id:04d}.jpg"
                target_name: str = f"{path}{photo_name}"

                async with aiofiles.open(target_name, "wb") as file:
                    await file.write(binary_data)

                logger.info(
                    "Image #%d is being saved to %s", self.image_id, target_name
                )
                self.image_id += 1
                return target_name, photo_name

    @override
    async def scanning_move_to(
        self,
        drone: dronekit.Vehicle,
        latitude: float,
        longitude: float,
        altitude: float,
        interval: float,
    ) -> None:
        """
        Moves to the drone to the requested waypoint while taking photos for the ODLC and Mapping states.

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
        info: dict[str, CameraParameters] = {}

        self.camera.requestSetAngles(0, -90)  # Point straight down if it isn't already

        camera_parameters: CameraParameters = await self._get_camera_parameters(drone)
        file_path: str
        _, file_path = await self.capture_photo(f"{os.getcwd()}/images/")
        point: dict[str, CameraParameters] = {file_path: camera_parameters}
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

        start_pos: dronekit.LocationGlobalRelative = (
            drone.location.global_relative_frame
        )
        assert (
            start_pos.alt is not None
        )  # throw if altitude is not present for any reason

        start_lat: float = start_pos.lat
        start_lon: float = start_pos.lon
        start_alt: float = start_pos.alt

        next_interval_count: int = 1
        while not goto_task.done():
            position: dronekit.LocationGlobalRelative = (
                drone.location.global_relative_frame
            )
            assert position.alt is not None

            drone_lat: float = position.lat
            drone_long: float = position.lon
            drone_alt: float = position.alt

            distance: float = calculate_distance(
                drone_lat, drone_long, drone_alt, start_lat, start_lon, start_alt
            )

            if distance >= next_interval_count * interval:
                next_interval_count += 1
                camera_parameters = await self._get_camera_parameters(drone)
                _, file_path = await self.capture_photo(f"{os.getcwd()}/images/")
                point = {file_path: camera_parameters}
                info.update(point)

            await asyncio.sleep(0.25)
        await asyncio.sleep(1.0)

        camera_parameters = await self._get_camera_parameters(drone)
        _, file_path = await self.capture_photo(f"{os.getcwd()}/images/")
        point = {file_path: camera_parameters}
        info.update(point)

        current_photos: dict[str, CameraParameters] = {}
        if os.path.exists("flight/data/camera.json"):
            async with aiofiles.open(
                "flight/data/camera.json", "r", encoding="utf8"
            ) as current_data:
                try:
                    current_photos = json.loads(await current_data.read())
                except json.JSONDecodeError:
                    pass

        async with aiofiles.open(
            "flight/data/camera.json", "w", encoding="ascii"
        ) as camera:
            await camera.write(json.dumps(current_photos | info))

    @override
    async def _get_camera_parameters(self, drone: dronekit.Vehicle) -> CameraParameters:
        """
        Gets the current camera information based on a drone's position_pitch.

        Parameters
        ----------
        drone : dronekit.Vehicle
            The drone whose position will be stored in the result.

        Returns
        -------
        CameraParameters
            Camera information to be associated with a photo.
        """
        location: dronekit.LocationGlobalRelative = drone.location.global_relative_frame
        assert location.alt is not None

        gimbal_attitude = self.camera.getAttitude()

        attitude: dronekit.Attitude = drone.attitude
        roll_deg: float = gimbal_attitude[2] # Gimbal has its own imu outputs so we do not need to use the drone's roll and pitch 
        pitch_deg: float = gimbal_attitude[1]
        yaw_deg: float = math.degrees(attitude.yaw) - gimbal_attitude[0] #Gimbal yaw angle decreases as it rotates clockwise

        return CameraParameters(
            rotation_deg=[roll_deg, pitch_deg, yaw_deg],
            drone_coordinates=[location.lat, location.lon],
            altitude=location.alt,
        )

    @override
    def disconnect(self) -> None:
        """
        Disconnects the IRL Camera.
        """
        self.camera.disconnect()


class CameraAirSim(Camera):
    """
    Initialize a new Camera object to control a simulated airsim camera

    Attributes
    ----------
    client : airsim.MultirotorClient
        The simulated drone.
    session_id : int
        The session id for the current session.
        This will start at 0 the first time pictures are taken on a given day.
        Will then increment by 1 for each session on a given day.
    image_id : int
        The image id for the current image.
        Starts at 0 and increments by 1 for each image taken.
    base_api_url : str
        The base API URL for the camera


    Methods
    -------
    capture_photo(path: str = f"{os.getcwd()}/images/")
        Capture a photo and save it to the specified path.
        The default path is the images folder in the current working directory.
        The file name will be the file format attribute.
        Returns the file name and the file path.
    scanning_move_to(
        drone: dronekit.Vehicle,
        latitude: float,
        longitude: float,
        altitude: float,
        interval: float
    )
        Move the drone to the specified latitude, longitude, and altitude.
        Takes photos along the way if take_photos is True.
    _get_camera_parameters(drone: dronekit.Vehicle)
        Get the current camera information to associate with a photo.
    """

    def __init__(self) -> None:
        super().__init__()

        self.client: airsim.MultirotorClient = airsim.MultirotorClient()

        self.session_id: int = 0
        if os.path.exists(f"{os.getcwd()}/images/"):
            for file in os.listdir(f"{os.getcwd()}/images/"):
                if (
                    file.startswith(
                        f"{datetime.now(UTC).astimezone().strftime('%Y%m%d')}"
                    )
                    and int(file.split("_")[1]) >= self.session_id
                ):
                    self.session_id = int(file.split("_")[1]) + 1

        self.image_id: int = 0

        logger.info("Airsim Camera initialized")

    @override
    async def capture_photo(
        self, path: str = f"{os.getcwd()}/images/"
    ) -> tuple[str, str]:
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
        os.makedirs(path, mode=0o777, exist_ok=True)
        cam_file = self.client.simGetImage("bottom_center", airsim.ImageType.Scene)
        if cam_file is None:
            raise ValueError("Failed to get image from AirSim")
        photo_name: str = f"{datetime.now(UTC).astimezone().strftime('%Y%m%d')}_{self.session_id}_{self.image_id:04d}.jpg"
        target_name: str = f"{path}{photo_name}"
        async with aiofiles.open(target_name, mode="wb") as file:
            await file.write(cam_file)

        self.image_id += 1
        logger.info("Image #%d is being saved to %s", self.image_id, target_name)
        return target_name, photo_name

    @override
    async def scanning_move_to(
        self,
        drone: dronekit.Vehicle,
        latitude: float,
        longitude: float,
        altitude: float,
        interval: float,
    ) -> None:
        """
        Moves to the drone to the requested waypoint while taking photos for the ODLC and Mapping states.

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
        info: dict[str, CameraParameters] = {}

        camera_parameters: CameraParameters = await self._get_camera_parameters(drone)
        file_path: str
        _, file_path = await self.capture_photo(f"{os.getcwd()}/images/")
        point: dict[str, CameraParameters] = {file_path: camera_parameters}
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

        start_pos: dronekit.LocationGlobalRelative = (
            drone.location.global_relative_frame
        )
        assert start_pos.alt is not None

        start_lat: float = start_pos.lat
        start_lon: float = start_pos.lon
        start_alt: float = start_pos.alt

        next_interval_count: int = 1
        while not goto_task.done():
            position: dronekit.LocationGlobalRelative = (
                drone.location.global_relative_frame
            )
            assert position.alt is not None

            drone_lat: float = position.lat
            drone_long: float = position.lon
            drone_alt: float = position.alt

            distance: float = calculate_distance(
                drone_lat, drone_long, drone_alt, start_lat, start_lon, start_alt
            )

            if distance >= next_interval_count * interval:
                next_interval_count += 1
                camera_parameters = await self._get_camera_parameters(drone)
                _, file_path = await self.capture_photo(f"{os.getcwd()}/images/")
                point = {file_path: camera_parameters}
                info.update(point)

            await asyncio.sleep(0.25)
        await asyncio.sleep(1.0)

        camera_parameters = await self._get_camera_parameters(drone)
        _, file_path = await self.capture_photo(f"{os.getcwd()}/images/")
        point = {file_path: camera_parameters}
        info.update(point)

        current_photos: dict[str, CameraParameters] = {}
        if os.path.exists("flight/data/camera.json"):
            async with aiofiles.open(
                "flight/data/camera.json", "r", encoding="utf8"
            ) as current_data:
                try:
                    current_photos = json.loads(await current_data.read())
                except json.JSONDecodeError:
                    pass

        async with aiofiles.open(
            "flight/data/camera.json", "w", encoding="ascii"
        ) as camera:
            await camera.write(json.dumps(current_photos | info))

    @override
    async def _get_camera_parameters(self, drone: dronekit.Vehicle) -> CameraParameters:
        """
        Gets the current camera information based on a drone's position_pitch.

        Parameters
        ----------
        drone : dronekit.Vehicle
            The drone whose position will be stored in the result.

        Returns
        -------
        CameraParameters
            Camera information to be associated with a photo.
        """
        location: dronekit.LocationGlobalRelative = drone.location.global_relative_frame
        assert location.alt is not None

        attitude: dronekit.Attitude = drone.attitude
        roll_deg: float = math.degrees(attitude.roll)
        pitch_deg: float = math.degrees(attitude.pitch)
        yaw_deg: float = math.degrees(attitude.yaw)

        return CameraParameters(
            rotation_deg=[roll_deg, pitch_deg, yaw_deg],
            drone_coordinates=[location.lat, location.lon],
            altitude=location.alt,
        )
