"""Data model backing the object detection reviewer app."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypedDict

from vision.common.constants import (
    DEFAULT_CAMERA_DATA_PATH,
    DEFAULT_DETECTIONS_OUTPUT_PATH,
    DEFAULT_REVIEWER_OUTPUT_PATH,
    CameraParameters,
)

logger = logging.getLogger(__name__)

# Absolute path to ensure that Qt can load the images
SUAS_ROOT: Path = Path(__file__).resolve().parents[2]
IMAGE_ROOT: Path = SUAS_ROOT / "images"

# Possible categories for detections
CATEGORIES: tuple[str, ...] = ("person", "tent")

# Manual detections should outweigh model detections, so they are given
# a confidence above 1.0
MANUAL_CONFIDENCE: float = 1.001


class ReviewStatus(Enum):
    """The status of a detection in the reviewer."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ReviewDetectionDict(TypedDict):
    """The JSON representation of detections being imported"""

    image: str
    category: str
    bbox: list[float]
    confidence: float
    shape: list[int]


@dataclass(eq=False)
class ReviewDetection:
    """
    A single detection displayed in the reviewer.

    Attributes
    ----------
    image : str
        The path to the image file, relative to the images root.
    category : str
        The category (or class name) of the object detection.
    bbox : tuple[float, float, float, float]
        The bounding box of the object detection, as (x1, y1, x2, y2).
    confidence : float
        The confidence score of the object detection.
    shape : tuple[int, int]
        The (height, width) of the image the detection came from.
    status : ReviewStatus
        The reviewer status of the detection.
    """

    image: str
    category: str
    bbox: tuple[float, float, float, float]
    confidence: float
    shape: tuple[int, int]
    status: ReviewStatus = ReviewStatus.PENDING

    @classmethod
    def from_dict(cls, data: ReviewDetectionDict) -> ReviewDetection:
        """Creates a detection from one entry of the detections JSON file."""
        x1, y1, x2, y2 = (float(value) for value in data["bbox"])
        height, width = (int(value) for value in data["shape"][:2])

        # Need to resolve the image path relative to the SUAS root
        image_path = str(SUAS_ROOT / data["image"])

        return cls(
            image=image_path,
            category=data["category"],
            bbox=(x1, y1, x2, y2),
            confidence=float(data["confidence"]),
            shape=(height, width),
        )

    def as_dict(self) -> ReviewDetectionDict:
        """Converts the detection back into its JSON representation."""
        return {
            "image": self.image,
            "category": self.category,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "shape": [self.shape[0], self.shape[1]],
        }

    @property
    def width(self) -> float:
        """The width of the bounding box."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        """The height of the bounding box."""
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        """The area of the bounding box."""
        return self.width * self.height


class ReviewSession:
    """
    The set of detections loaded into the reviewer, grouped by image.

    Attributes
    ----------
    detections : list[ReviewDetection]
        Every loaded detection, in file order.
    source : Path
        The JSON file the detections were loaded from.
    images : list[str]
        Paths to all images captured, whether or not they have detections.
    image_root : Path
        The directory image paths are resolved against.
    """

    def __init__(
        self,
        detections: list[ReviewDetection],
        source: Path,
        images: list[str],
        image_root: Path,
    ) -> None:
        self.detections: list[ReviewDetection] = detections
        self.source: Path = source
        self._image_root: Path = image_root
        self._by_image: dict[str, list[ReviewDetection]] = {}
        # Add each detection to the image index under its base image
        for detection in detections:
            self._by_image.setdefault(detection.image, []).append(detection)
        self._images: list[str] = images
        print(self._by_image)
        print(self._images)

    @classmethod
    def load(
        cls,
        detections_data_path: Path = DEFAULT_DETECTIONS_OUTPUT_PATH,
        camera_data_path: Path = DEFAULT_CAMERA_DATA_PATH,
        image_root: Path = IMAGE_ROOT,
    ) -> ReviewSession:
        """
        Creates a new ReviewSession from a detections JSON file written by `create_review_JSON()`.

        Parameters
        ----------
        detections_data_path : Path
            The detections JSON file to load.
        camera_data_path : Path
            The camera data JSON file naming every captured image.
        image_root : Path
            The directory to resolve the image paths against.

        Returns
        -------
        session : ReviewSession
        """
        with open(detections_data_path) as file:
            raw: list[ReviewDetectionDict] = json.load(file)
        detections = [ReviewDetection.from_dict(entry) for entry in raw]
        logger.info(f"Loaded {len(detections)} detections from {detections_data_path}")

        with open(camera_data_path) as file:
            camera_data: dict[str, CameraParameters] = json.load(file)
        logger.info(f"Loaded {len(camera_data)} images from {camera_data_path}")

        # Only keep the images that actually exist
        image_paths: list[str] = [
            str(image_root / image)
            for image in camera_data
            if (image_root / image).is_file()
        ]

        return cls(detections, Path(detections_data_path), image_paths, image_root)

    def add_detection(
        self,
        image: str,
        center: tuple[float, float],
        size: float,
        category: str,
        shape: tuple[int, int],
    ) -> ReviewDetection:
        """
        Adds a manual detection.

        Parameters
        ----------
        image : str
            The path of the image the detection belongs to.
        center : tuple[float, float]
            The (x, y) point in the image the box is centered on.
        size : float
            Size of the bbox in pixels.
        category : str
            The category to label the new detection as.
        shape : tuple[int, int]
            The (height, width) of the image, used to keep the box in bounds.

        Returns
        -------
        detection : ReviewDetection
            The detection that was added to the session.
        """
        height, width = shape
        half = size / 2.0

        # Bound the box to the image so it doesn't go out of bounds
        # Shift instead of cutting off bbox so it doesn't get filtered out
        x = min(max(center[0], half), max(float(width) - half, half))
        y = min(max(center[1], half), max(float(height) - half, half))

        detection = ReviewDetection(
            image=image,
            category=category,
            bbox=(x - half, y - half, x + half, y + half),
            confidence=MANUAL_CONFIDENCE,
            shape=shape,
            status=ReviewStatus.ACCEPTED,
        )
        # Add to detection lists so it is saved
        self.detections.append(detection)
        self._by_image.setdefault(image, []).append(detection)
        logger.info(f"Added a manual {category} detection at {detection.bbox}")
        return detection

    def save(self, path: Path = DEFAULT_REVIEWER_OUTPUT_PATH):
        """Writes the accepted detections back to JSON.

        Parameters
        ----------
        path : Path
            The path to save the JSON file to.
        """
        valid_detections = [
            detection.as_dict()
            for detection in self.detections
            if detection.status == ReviewStatus.ACCEPTED
        ]
        with open(path, "w") as file:
            json.dump(valid_detections, file)
        logger.info(f"Saved {len(valid_detections)} detections to {path}")

    @property
    def images(self) -> list[str]:
        """The image paths that have detections, in the order they were captured."""
        return list(self._by_image)

    @property
    def all_images(self) -> list[str]:
        """Every captured image path, including the ones without detections."""
        return list(self._images)

    def for_image(self, image: str) -> list[ReviewDetection]:
        """The detections belonging to one image, highest confidence first."""
        return sorted(
            self._by_image.get(image, []), key=lambda d: d.confidence, reverse=True
        )

    def counts(self) -> dict[ReviewStatus, int]:
        """The number of detections for each review status."""
        counts = dict.fromkeys(ReviewStatus, 0)
        for detection in self.detections:
            counts[detection.status] += 1
        return counts

    def resolve_image(self, image: str) -> Path | None:
        """
        Returns the full path to the provided image file.

        Parameters
        ----------
        image : str
            The image path as stored in the detection.

        Returns
        -------
        path : Path | None
            The existing file, or None if it could not be found.
        """
        image_path = self._image_root / image
        found = image_path if image_path.is_file() else None
        if found is None:
            logger.warning(f"Could not find image {image_path}")
        return found
