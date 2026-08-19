"""
Contains helper functions to call the inference_node API
https://github.com/MissouriMRR/inference_node
"""

import logging
import os
from typing import Any

import aiofiles
import aiohttp
import numpy as np

from vision.object_detection.providers.base import ObjectDetection

logger = logging.getLogger(__name__)

# Default host/port the inference_node API listens on, see inference_node/main.py
# IF YOU ARE NOT ABLE TO REACH THE JETSON:
# Use `ip addr` to find the Ethernet interface name
# Run `ip addr add 192.168.50.1/24 dev <eth_inferface_name> to assign an IP
# Ping `192.168.50.2`. You should get a response from the Jetson
DEFAULT_HOST = "192.168.50.2"
DEFAULT_PORT = 8642
DEFAULT_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"

# Timeouts, in seconds, for the two endpoints exposed by inference_node
HEALTH_TIMEOUT = 5
DETECT_TIMEOUT = 30


async def check_health(session: aiohttp.ClientSession, base_url: str = DEFAULT_URL) -> bool:
    """
    Call the /health endpoint of an inference_node instance to see if it is up
    and usable.

    Parameters
    ----------
    session : aiohttp.ClientSession
        The session to make the request with.
    base_url : str
        The base URL of the inference_node instance, e.g. "http://192.168.1.50:8642"

    Returns
    -------
    bool
        True if the node responded with a healthy status, False otherwise.
    """
    try:
        async with session.get(
            f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=HEALTH_TIMEOUT)
        ) as response:
            if response.status != 200:
                logger.error("inference_node health check failed with status %d", response.status)
                return False
            data: dict[str, Any] = await response.json()
            return data.get("status") == "running"
    except aiohttp.ClientError as exc:
        logger.error("inference_node health check failed: %s", exc)
        return False


async def detect_image(
    session: aiohttp.ClientSession, image_path: str, base_url: str = DEFAULT_URL
) -> list[ObjectDetection]:
    """
    Upload a single image to the /detect endpoint of an inference_node instance,
    then parse the response into a list of ObjectDetection instances.

    Parameters
    ----------
    session : aiohttp.ClientSession
        The session to make the request with.
    base_url : str
        The base URL of the inference_node instance, e.g. "http://192.168.1.50:8642"
    image_path : str
        The path to the image file to run inference on.

    Returns
    -------
    list[ObjectDetection]
        The detections found in the image, empty if none were found.

    Raises
    ------
    ValueError
        If the inference_node instance fails to process the image.
    """
    async with aiofiles.open(image_path, "rb") as file:
        image_bytes: bytes = await file.read()

    form = aiohttp.FormData()
    form.add_field(
        "image",
        image_bytes,
        filename=os.path.basename(image_path),
        content_type="image/jpeg",
    )

    async with session.post(
        f"{base_url}/detect", data=form, timeout=aiohttp.ClientTimeout(total=DETECT_TIMEOUT)
    ) as response:
        body: dict[str, Any] = await response.json()
        if response.status != 200:
            logger.error("Inference failed for %s: %s", image_path, body)
            raise ValueError(f"inference_node failed to process {image_path}")

    shape: tuple[int, int] = tuple(body["shape"])  # (height, width)

    return [
        ObjectDetection(
            image_path,
            detection["category"],
            np.array(detection["bbox"], dtype=np.int64),
            detection["confidence"],
            shape,
        )
        for detection in body["detections"]
    ]
