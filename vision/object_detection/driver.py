"""
Contains the object detection driver, which will process
images and return object detections using various providers.
"""

import logging
from typing import final

from vision.object_detection.providers import (
    InferenceProvider,
    LocalInferenceProvider,
    NodeInferenceProvider,
    ObjectDetection,
)

logger = logging.getLogger(__name__)

# All possible providers are listed here, in order of priority
PROVIDERS: list[type[InferenceProvider]] = [
    NodeInferenceProvider,
    LocalInferenceProvider,
]


@final
class ObjectDetectionDriver:
    """
    Driver that will use all available providers to run object
    detection on provided images.

    Attributes
    ----------
    _provider_types: list[type[InferenceProvider]]
        All provider types to try and use for inference.
    _providers: list[InferenceProviders]
        All provider instances available to use, ordered by priority.

    Methods
    -------
    start() -> None
        Attempts to start all _provider_types. Adds them to _providers.
    add_photo(image_path: str)
        Adds a photo to perform inference on. Throws exception if no
        providers are able to run.
    end() -> list[ObjectDetection]
        Ends all providers safely, combining all of their results together
        and returns the full list.
    """

    def __init__(self, providers: list[type[InferenceProvider]] | None = None):
        self._provider_types = providers if providers is not None else list(PROVIDERS)
        self._providers: list[InferenceProvider] = []

    async def start(self) -> None:
        """
        Start all providers, keeping only the ones that start successfully.
        """
        for provider_type in self._provider_types:
            try:
                provider = provider_type()
                await provider.start()
            except Exception:
                logger.warning(
                    "Provider %s failed to start, skipping", provider_type.__name__
                )
                continue
            self._providers.append(provider)

    async def add_image(self, image_path: str) -> None:
        """
        Adds a new image to be processed.

        This should be called in a way where you can fire the task
        to be scheduled and check on the failures later, so you can
        add images as they are captured from the drone (most likely
        using asyncio)
        """
        for provider in self._providers:
            try:
                await provider.add_image(image_path)
                return
            except Exception:
                logger.warning(
                    "Provider %s failed on %s, trying next provider",
                    type(provider).__name__,
                    image_path,
                )
        logger.error("All providers failed to process %s", image_path)
        raise RuntimeError(f"All providers failed to process {image_path}")

    async def end(self) -> list[ObjectDetection]:
        """
        End all providers and combine their results. Only call when
        add_image() has been called for all images and all the tasks
        have finished.
        """
        results: list[ObjectDetection] = []
        for provider in self._providers:
            results.extend(await provider.end())
        return results
