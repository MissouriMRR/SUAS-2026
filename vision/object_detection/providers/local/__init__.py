"""Defines the LocalInferenceProvider class."""

from typing_extensions import final, override
from vision.object_detection.providers.base import InferenceProvider, ObjectDetection
from vision.object_detection.providers.local.queue import PhotoQueue


@final
class LocalInferenceProvider(InferenceProvider):
    """
    A provider to run object detection inference using the local machine.
    """

    def __init__(self, max_runners: int = 3):
        """Set up internal variables and preload resources"""
        super().__init__()
        self.max_runners = max_runners
        self.queue: PhotoQueue = PhotoQueue(show_results=True)

    @override
    async def start(self) -> None:
        """Start the inference provider"""
        await self.queue.start_queue(self.max_runners)

    @override
    async def add_image(self, image_path: str) -> None:
        """Run inference on the given image"""
        await self.queue.add_photo(image_path)

    @override
    async def end(self) -> list[ObjectDetection]:
        """Return the results of the inference"""
        return await self.queue.end_queue()
