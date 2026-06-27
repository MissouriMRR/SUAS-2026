"""Runs the necessary code for the Mapping component of the competition"""

import asyncio
import logging
import zipfile
from datetime import date
from pathlib import Path

from pyodm import Node, Task
from pyodm.exceptions import NodeConnectionError
from pyodm.types import TaskInfo, TaskStatus

import vision.pipeline.pipeline_utils as pipe_utils
from vision.common.constants import CameraParameters

logger = logging.getLogger(__name__)


async def wait_for_task_completion(
    task: Task, interval: int = 3, max_retries: int = 5, retry_timeout: int = 5
) -> None:
    """
    An async implementation of Task.wait_for_completion(), as the original one
    will block the event loop. Will return when the TaskStatus is COMPLETED, CANCELED or FAILED.

    Parameters
    ----------
    task: Task
        The task to wait for completion
    interval: int, optional
        The interval to wait between retries (default is 3 seconds)
    max_retries: int, optional
        The maximum number of retries (default is 5)
    retry_timeout: int, optional
        The timeout between retries (default is 5 seconds)
    """
    retry: int = 0
    while True:
        try:
            info: TaskInfo = task.info()
        except NodeConnectionError as e:
            if retry < max_retries:
                retry += 1
                await asyncio.sleep(retry * retry_timeout)
                continue
            raise e

        if info.status in [
            TaskStatus.COMPLETED,
            TaskStatus.CANCELED,
            TaskStatus.FAILED,
        ]:
            break

        await asyncio.sleep(interval)

    if info.status in [TaskStatus.FAILED, TaskStatus.CANCELED]:
        raise ConnectionError(f"Task failed or was canceled: {info.status}")


async def mapping_pipeline(
    camera_data_path: str,
    image_dir: str,
    capture_status: asyncio.Event,
    output_path: str = "vision/mapping/results",
) -> None:
    """
    Runs the code that generates the map from a folder of photos

    Parameters
    ----------
    camera_data_path: str
        The path to the json file containing the CameraParameters
    image_dir: str
        The path with all the images
    capture_status: asyncio.Event
        Once all the images are taken, capture status is set
        to true (used to determine when to generate the map)
    output_path: str
        The json file name and path to save the data in
    """

    # Check capture status to see if all images have been taken
    while not capture_status.is_set():
        await asyncio.sleep(1)

    logger.info("Capture status set, generating map...")

    # Read in the camera JSON data file and write formatted data to the geo.txt (needed for ODM)
    geo_path: Path = Path(image_dir) / "geo.txt"

    metadata: dict[str, CameraParameters] = pipe_utils.read_parameter_json(camera_data_path)

    # Write geo.txt with data from camera json
    lines = ["EPSG:4326"]

    for image_name, entry in metadata.items():
        lat, lon = entry["drone_coordinates"]
        alt = entry.get("altitude", 0)

        # Assumes rotation_deg is already [yaw, pitch, roll].
        rot = entry.get("rotation_deg", [0, 0, 0])
        yaw, pitch, roll = rot[0], rot[1], rot[2]

        lines.append(f"{image_name} {lon} {lat} {alt} {yaw} {pitch} {roll}")

    geo_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", geo_path)

    # Set up connection to ODM node
    node = Node("localhost", 3000)
    logger.info(
        "Node info: mem=%d/%d, engine=%s, version=%s",
        node.info().available_memory,
        node.info().total_memory,
        node.info().engine,
        node.info().version,
    )
    image_files = sorted(str(path) for path in Path(image_dir).glob("*.jpg")) + [str(geo_path)]

    # Create the ODM task (tweak settings here as needed)
    task = node.create_task(
        image_files,
        {
            "fast-orthophoto": True,
            "skip-3dmodel": True,
            "dsm": False,
            "dtm": False,
            "skip-report": True,
            "tiles": False,
            "cog": False,
            "orthophoto-png": True,
            # uncomment these if we need more speed
            # "feature-quality": "medium",
            # "min-num-features": 4000,
            # "matcher-neighbors": 8,
            # "resize-to": 2048,
        },
    )
    logger.info("Task UUID: %s", task.uuid)

    await wait_for_task_completion(task)
    try:
        # Verify output director exists and get ZIP file from ODM
        output = Path(output_path)
        output.mkdir(parents=True, exist_ok=True)
        zip_path = Path(task.download_zip(str(output)))

        # Extract map png or jpg to output
        today = date.today().strftime("%Y-%m-%d")
        with zipfile.ZipFile(zip_path) as zf:
            image_exts = (".png", ".jpg")
            for name in zf.namelist():
                if name.startswith("odm_orthophoto/") and name.endswith(image_exts):
                    ext = Path(name).suffix
                    dest_name = f"MissouriSandT_{today}_map{ext}"
                    (output / dest_name).write_bytes(zf.read(name))
        zip_path.unlink()
        logger.info("Downloaded orthophoto to %s", output_path)

        # Save Map to USB drive (pass in usb drive path up top)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(e)
