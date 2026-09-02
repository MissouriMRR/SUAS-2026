"""
Splits large images into smaller tiles to be run through inference providers,
and merges the detections found in those tiles back into the coordinate space
of the image they were cut from.
"""

from __future__ import annotations

import logging
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from vision.common.constants import Image, ImageShape
from vision.object_detection.providers.base import ObjectDetection

logger = logging.getLogger(__name__)

# The size in pixels of the square crops taken from images.
# Should match the model input size
DEFAULT_TILE_SIZE: int = 640

# The fraction of a tile that overlaps with the tile beside it.
DEFAULT_TILE_OVERLAP: float = 0.3

# The intersection over union above which two detections of the same category
# in the same image are treated as the same object when merging tile results.
DEFAULT_IOU_THRESHOLD: float = 0.5

# How close, in pixels, a bounding box edge has to be to a tile edge before it
# is treated as having been cut off by it.
SEAM_MARGIN: int = 2


class TileError(Exception):
    """Raised when an image cannot be read or its tiles cannot be written."""


@dataclass(frozen=True)
class Tile:
    """
    A single square tile of an image, written to its own file.
    """

    path: str
    """The path to the file the crop was written to."""
    source_path: str
    """The path to the image the crop was taken from."""
    shape: ImageShape
    """The (height, width) shape of the crop itself."""
    source_shape: ImageShape
    """The (height, width) shape of the image the crop was taken from."""
    x_offset: int
    """The x pixel coordinate of the crop's left edge in the source image."""
    y_offset: int
    """The y pixel coordinate of the crop's top edge in the source image."""


def tile_origins(length: int, tile_size: int, overlap: float) -> list[int]:
    """
    Gets the start coordinates of every tile along one axis of an image.
    This is used to loop over start pixels for tiling.

    Parameters
    ----------
    length : int
        The length of the axis, in pixels.
    tile_size : int
        The size of a tile, in pixels.
    overlap : float
        The fraction of a tile shared with the tile beside it, from 0 to 0.95.

    Returns
    -------
    origins : list[int]
        The start coordinate of each tile, ascending.
    """
    # If length is less than tile_size, there will only be one tile
    if length <= tile_size:
        return [0]

    # Bound overlap so tiles always overlap and never leave gaps between them
    overlap = max(0.0, min(overlap, 0.95))

    # Calculate step size based on overlap
    step: int = max(1, int(tile_size * (1.0 - overlap)))

    origins: list[int] = list(range(0, length - tile_size + 1, step))

    # Ensure the last origin is exactly the edge of the image - tile size
    last_origin: int = length - tile_size
    if origins[-1] != last_origin:
        origins.append(last_origin)

    return origins


def write_tiles(
    image_path: str,
    tile_dir: str,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: float = DEFAULT_TILE_OVERLAP,
) -> list[Tile]:
    """
    Cuts an image into overlapping tiles and writes each one to its own file.

    Parameters
    ----------
    image_path : str
        The path to the image to cut up.
    tile_dir : str
        The directory to write the tiles into. A uniquely named subdirectory
        is created inside it to hold this image's tiles.
    tile_size : int, default=DEFAULT_TILE_SIZE
        The size of each square tile, in pixels of the source image.
    overlap : float, default=DEFAULT_TILE_OVERLAP
        The fraction of a tile shared with the tile beside it, from 0 to 0.95.

    Returns
    -------
    tiles : list[Tile]
        Every tile the image was cut into, together with where it came from.

    Raises
    ------
    TileError
        If the image cannot be read, or a tile cannot be written to disk.
    """
    raw_image: cv2.typing.MatLike | None = cv2.imread(image_path)
    if raw_image is None:
        raise TileError(f"{image_path} could not be read as an image")
    image: Image = raw_image.astype(np.uint8)

    height: int
    width: int
    height, width = image.shape[:2]
    source_shape: ImageShape = (height, width)

    # Create a temporary directory for the tiles of sharing images
    image_name: str = Path(image_path).stem
    output_dir: str = tempfile.mkdtemp(prefix=f"{image_name}_", dir=tile_dir)

    tiles: list[Tile] = []
    # Iterate over each tile origin
    for y_offset in tile_origins(height, tile_size, overlap):
        for x_offset in tile_origins(width, tile_size, overlap):
            # Extract the crop from the image
            crop: Image = image[
                y_offset : y_offset + tile_size, x_offset : x_offset + tile_size
            ]
            tile_path: str = str(Path(output_dir, f"x{x_offset}_y{y_offset}.jpg"))
            # Write the crop to the tile path
            if not cv2.imwrite(tile_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                raise TileError(f"Failed to write tile {tile_path} of {image_path}")

            # A crop is tile_size square except on an image smaller than a tile,
            # so its shape is recorded rather than assumed
            crop_height: int
            crop_width: int
            crop_height, crop_width = crop.shape[:2]
            crop_shape: ImageShape = (crop_height, crop_width)

            tiles.append(
                Tile(
                    path=tile_path,
                    source_path=image_path,
                    shape=crop_shape,
                    source_shape=source_shape,
                    x_offset=x_offset,
                    y_offset=y_offset,
                )
            )

    logger.debug(
        "Split %s (%dx%d) into %d %dpx tiles",
        image_path,
        width,
        height,
        len(tiles),
        tile_size,
    )
    return tiles


def is_clipped_at_seam(detection: ObjectDetection, tile: Tile) -> bool:
    """
    Checks whether a detection is cut off by a tile edge that falls inside the
    source image.

    We prefer to keep full detections over truncated ones for more accurate
    bbox middle points.

    Parameters
    ----------
    detection : ObjectDetection
        A detection with a bounding box relative to the tile.
    tile : Tile
        The tile the detection was found in.

    Returns
    -------
    clipped : bool
        True if the box touches a tile edge that is not also an image edge.
    """
    tile_height: int = tile.shape[0]
    tile_width: int = tile.shape[1]
    source_height: int = tile.source_shape[0]
    source_width: int = tile.source_shape[1]

    box: npt.NDArray[np.int64] = np.rint(detection.bbox).astype(np.int64)
    x1: int = int(box[0])
    y1: int = int(box[1])
    x2: int = int(box[2])
    y2: int = int(box[3])

    # An edge only counts as a seam if there is more of the image beyond it,
    # taken here as the left, top, right and bottom edges of the tile in turn
    seams: list[bool] = [
        tile.x_offset > 0 and x1 <= SEAM_MARGIN,  # x on left edge, crop is not leftmost
        tile.y_offset > 0 and y1 <= SEAM_MARGIN,  # y on top edge, crop is not topmost
        tile.x_offset + tile_width < source_width
        and x2 >= tile_width - 1 - SEAM_MARGIN,  # x on right, crop is not rightmost
        tile.y_offset + tile_height < source_height
        and y2 >= tile_height - 1 - SEAM_MARGIN,  # y on bottom, crop is not bottommost
    ]

    # If any of conditions are True, return True
    return any(seams)


def restore_detection(detection: ObjectDetection, tile: Tile) -> ObjectDetection:
    """
    Convert tile detection back to the coordinate space of the source image.

    Parameters
    ----------
    detection : ObjectDetection
        A detection with a bounding box relative to the tile.
    tile : Tile
        The tile the detection was found in.

    Returns
    -------
    restored : ObjectDetection
        The same detection, relative to the source image.
    """
    height: int = tile.source_shape[0]
    width: int = tile.source_shape[1]

    # Initial x/y using the tile offset
    offsets: npt.NDArray[np.int64] = np.array(
        [tile.x_offset, tile.y_offset, tile.x_offset, tile.y_offset], dtype=np.int64
    )
    # Max possible x/y values (one less than the image size)
    limits: npt.NDArray[np.int64] = np.array(
        [width - 1, height - 1, width - 1, height - 1], dtype=np.int64
    )
    # Add bbox to tile offset
    shifted: npt.NDArray[np.int64] = detection.bbox + offsets
    # Clip to image bounds
    bbox: npt.NDArray[np.int64] = np.clip(shifted, 0, limits)

    return ObjectDetection(
        tile.source_path,
        detection.category,
        bbox,
        detection.confidence,
        tile.source_shape,
    )


def _intersection_over_union(
    first: npt.NDArray[np.int64], second: npt.NDArray[np.int64]
) -> float:
    """
    Gets the intersection over union (IoU) of two bounding boxes.

    Parameters
    ----------
    first, second : npt.NDArray[np.int64]
        The bounding boxes to compare, each as [x1, y1, x2, y2].

    Returns
    -------
    iou : float
        The overlapping area of the boxes over their combined area.
    """
    overlap_width = float(min(first[2], second[2]) - max(first[0], second[0]))
    overlap_height = float(min(first[3], second[3]) - max(first[1], second[1]))
    if overlap_width <= 0 or overlap_height <= 0:
        return 0.0

    intersection = overlap_width * overlap_height
    first_area = float((first[2] - first[0]) * (first[3] - first[1]))
    second_area = float((second[2] - second[0]) * (second[3] - second[1]))
    union = first_area + second_area - intersection
    if union <= 0:
        return 0.0

    return intersection / union


def merge_detections(
    detections: list[ObjectDetection],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> list[ObjectDetection]:
    """
    Collapses detections of the same object found in more than one tile due
    to overlapping regions.

    Parameters
    ----------
    detections : list[ObjectDetection]
        Detections that have already been restored to image coordinates.
    iou_threshold : float, default=DEFAULT_IOU_THRESHOLD
        The IoU above which two detections of the same category are treated
        as the same object.

    Returns
    -------
    merged : list[ObjectDetection]
        The detections with duplicates removed.
    """
    # Dict of lists for each image/category group
    grouped: defaultdict[tuple[str, str], list[ObjectDetection]] = defaultdict(list)
    for detection in detections:
        grouped[(detection.image, detection.category)].append(detection)

    merged: list[ObjectDetection] = []
    for group in grouped.values():
        # Take the best confidence detection for each group
        group.sort(key=lambda detection: detection.confidence, reverse=True)

        kept: list[ObjectDetection] = []
        for detection in group:
            # Skip detections that collide with existing ones
            duplicate: bool = any(
                _intersection_over_union(detection.bbox, existing.bbox) > iou_threshold
                for existing in kept
            )
            if not duplicate:
                kept.append(detection)

        merged.extend(kept)

    return merged
