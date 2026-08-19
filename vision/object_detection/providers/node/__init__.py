"""Defines the NodeInferenceProvider class."""

import logging

from typing_extensions import final, override

import aiohttp

from vision.object_detection.providers.base import InferenceProvider, ObjectDetection
from vision.object_detection.providers.node.node import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    check_health,
    detect_image,
)

logger = logging.getLogger(__name__)


@final
class NodeInferenceProvider(InferenceProvider):
    """
    A provider to run object detection inference using `inference_node`,
    intended for running inference on an NVIDIA Jetson device.
    More info: https://github.com/MissouriMRR/inference_node
    """

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        """Set up internal variables and preload resources"""
        super().__init__()
        self.base_url: str = f"http://{host}:{port}"
        self.session: aiohttp.ClientSession | None = None
        self.results: list[ObjectDetection] = []

    @override
    async def start(self) -> None:
        """Call the /health endpoint to make sure the node is up"""
        self.session = aiohttp.ClientSession()
        if not await check_health(self.session, self.base_url):
            await self.session.close()
            self.session = None
            raise ConnectionError(f"inference_node at {self.base_url} is not reachable/healthy")

    @override
    async def add_image(self, image_path: str) -> None:
        """
        Run inference on the given image by calling the /detect endpoint with the image
        """
        if self.session is None:
            raise RuntimeError("NodeInferenceProvider.start() must be called before add_image()")
        self.results.extend(await detect_image(self.session, image_path, self.base_url))

    @override
    async def end(self) -> list[ObjectDetection]:
        """Return the results of the inference"""
        if self.session is not None:
            await self.session.close()
            self.session = None
        return self.results
