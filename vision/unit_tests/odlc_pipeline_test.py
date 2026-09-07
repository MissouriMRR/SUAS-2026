"""A unit test for running the full ODLC pipeline, from images to the ODLC JSON"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path

from state_machine.flight_settings import FlightSettings
from vision.common.constants import CameraParameters, ODLCDict
from vision.odlc_pipeline import odlc_pipeline
from vision.pipeline import pipeline_utils

logger = logging.getLogger(__name__)

DEFAULT_CAMERA_DATA_PATH: str = "flight/data/camera.json"

# The pipeline looks for every image named in the camera data in this folder
IMAGE_DIR: Path = Path("images")


def filter_camera_data(camera_data_path: str, output_path: Path) -> int:
    """
    Creates a copy of the camera data file that only includes data for images
    that are present in the image folder.

    Parameters
    ----------
    camera_data_path : str
        The JSON file containing the CameraParameters of every captured image.
    output_path : Path
        The JSON file to write the available entries to.

    Returns
    -------
    available : int
        The number of post-filter images
    """
    parameters: dict[str, CameraParameters] = pipeline_utils.read_parameter_json(
        camera_data_path
    )
    available: dict[str, CameraParameters] = {
        image: image_parameters
        for image, image_parameters in parameters.items()
        if (IMAGE_DIR / image).is_file()
    }
    with open(output_path, "w", encoding="UTF-8") as file:
        json.dump(available, file)
    return len(available)


async def test_pipeline(camera_data_path: str = DEFAULT_CAMERA_DATA_PATH) -> None:
    """
    Runs the ODLC pipeline over the images already in the image folder, as if
    the drone had just finished capturing them.

    Parameters
    ----------
    camera_data_path : str
        The JSON file containing the CameraParameters of every captured image.
    """
    flight_settings: FlightSettings = FlightSettings.from_mission_config()

    with tempfile.TemporaryDirectory() as directory:
        available_data_path: Path = Path(directory) / "camera.json"
        output_path: Path = Path(directory) / "odlc.json"

        image_count: int = filter_camera_data(camera_data_path, available_data_path)
        if image_count == 0:
            logger.error(
                "None of the images in %s are in %s/", camera_data_path, IMAGE_DIR
            )
            return

        # Instantly set capture status
        capture_status: asyncio.Event = asyncio.Event()
        capture_status.set()

        logger.info("Running the ODLC pipeline on %d image(s)...", image_count)
        await odlc_pipeline(
            flight_settings,
            str(available_data_path),
            capture_status,
            str(output_path),
        )

        if not output_path.is_file():
            logger.error("The pipeline did not write an ODLC file")
            return
        odlc_dict: ODLCDict = json.loads(output_path.read_text(encoding="UTF-8"))

    logger.info("Pipeline found %d ODLC(s): %s", len(odlc_dict), odlc_dict)
    if not flight_settings.yolo_status.is_set():
        logger.error("The pipeline did not set the yolo status event")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(test_pipeline())
