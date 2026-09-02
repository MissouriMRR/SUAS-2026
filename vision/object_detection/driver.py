"""
Contains the object detection driver, which will process
images and return object detections using various providers.
"""

import asyncio
import logging
import tempfile
from typing import final

from vision.object_detection.providers import (
    InferenceProvider,
    LocalInferenceProvider,
    NodeInferenceProvider,
    ObjectDetection,
)
from vision.object_detection.tiling import (
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_SIZE,
    Tile,
    is_clipped_at_seam,
    merge_detections,
    restore_detection,
    write_tiles,
)

logger = logging.getLogger(__name__)

# All possible providers are listed here, in order of priority
PROVIDERS: list[type[InferenceProvider]] = [
    NodeInferenceProvider,
    LocalInferenceProvider,
]

# The most requests that can be made to the inference provider at once
# Important for NodeInferenceProvider as http requests could get dropped
DEFAULT_MAX_REQUESTS: int = 8


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
    _tile_images: bool
        Whether images should be tiled before inference.
    _tile_size: int
        The size of each square tile, in pixels of the source image.
    _tile_overlap: float
        The fraction of a tile shared with the tile beside it. Must be between 0 and 0.95.
    _iou_threshold: float
        The overlap above which two detections of the same category are
        treated as the same object when merging tile results.
    _tile_dir: tempfile.TemporaryDirectory[str] | None
        The temporary directory holding every tile written so far, or None
        if tiling is off or the driver has not been started.
    _tiles: dict[str, Tile]
        Maps the path of each tile to where it came from, used to move
        detections back into the source image's coordinate space.

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

    def __init__(
        self,
        providers: list[type[InferenceProvider]] | None = None,
        tile_images: bool = False,
        tile_size: int = DEFAULT_TILE_SIZE,
        tile_overlap: float = DEFAULT_TILE_OVERLAP,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ):
        self._provider_types = providers if providers is not None else list(PROVIDERS)
        self._providers: list[InferenceProvider] = []
        self._tile_images: bool = tile_images
        self._tile_size: int = tile_size
        self._tile_overlap: float = tile_overlap
        self._iou_threshold: float = iou_threshold
        self._tile_dir: tempfile.TemporaryDirectory[str] | None = None
        self._tiles: dict[str, Tile] = {}
        self._req_semaphore = asyncio.Semaphore(DEFAULT_MAX_REQUESTS)

    async def start(self) -> None:
        """
        Start all providers, keeping only the ones that start successfully.
        """
        if self._tile_images:
            # Create temp dir for this session
            self._tile_dir = tempfile.TemporaryDirectory(prefix="odlc_tiles_")

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

        Raises
        ------
        TileError
            If the image could not be cut into tiles.
        RuntimeError
            If every tile of the image failed on every provider.
        """
        if self._tile_dir is None:
            # No tiling, run providers on the full image
            await self._run_providers(image_path)
            return

        # Reading and re-encoding a 4K image is slow, don't want to block
        tiles: list[Tile] = await asyncio.to_thread(
            write_tiles,
            image_path,
            self._tile_dir.name,
            self._tile_size,
            self._tile_overlap,
        )
        for tile in tiles:
            self._tiles[tile.path] = tile

        # Call _run_providers on every tile, wait for all to complete
        results = await asyncio.gather(
            *(self._run_providers(tile.path) for tile in tiles),
            return_exceptions=True,
        )

        failures: int = sum(isinstance(result, BaseException) for result in results)
        if failures > (len(results) // 2):
            # Failed on majority of tiles, throw error
            raise RuntimeError(
                f"{failures} of {len(results)} tiles of {image_path} failed on all providers"
            )
        elif failures > 0:
            # Pass but notify that some tiles failed
            logger.error(
                "%d of %d tiles of %s failed on all providers",
                failures,
                len(results),
                image_path,
            )

    async def _run_providers(self, image_path: str) -> None:
        """
        Runs a single image through the providers, in priority order, until
        one of them accepts it.

        Raises
        ------
        RuntimeError
            If every provider failed to process the image.
        """
        async with self._req_semaphore:
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

        Detections found in tiles are moved back into the coordinate space of
        the image they were cut from, and duplicates of the same object found
        in overlapping tiles are merged.
        """
        # Retrieve results from all providers
        results: list[ObjectDetection] = []
        for provider in self._providers:
            results.extend(await provider.end())

        if self._tile_dir is None:
            # Didn't tile images, so results are already in image coordinates
            return results

        try:
            restored: list[ObjectDetection] = []
            for detection in results:
                tile: Tile | None = self._tiles.get(detection.image)
                if tile is None:
                    # Not from a tile, so it is already in image coordinates
                    restored.append(detection)
                    continue

                if is_clipped_at_seam(detection, tile):
                    continue

                # Return to image coordinates
                restored.append(restore_detection(detection, tile))

            # Return merged detections
            return merge_detections(restored, self._iou_threshold)
        finally:
            # Clean up tile directory even if an error occurs
            tile_dir = self._tile_dir
            self._tile_dir = None
            self._tiles = {}
            await asyncio.to_thread(tile_dir.cleanup)
