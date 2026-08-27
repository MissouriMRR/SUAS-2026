"""Sets up the base InferenceProvider class."""

from dataclasses import dataclass
from typing import override

import numpy as np
import numpy.typing as npt

from vision.common.constants import ImageShape


@dataclass(eq=False)
class ObjectDetection:
    """
    Stores the info of an object detection.

    Attributes
    ----------
    image : str
        The path to the image file.
    category : str
        The category (or class name) of the object detection.
    bbox : npt.NDArray[np.int64]
        The bounding box of the object detection, as [x1, y1, x2, y2].
    confidence : float
        The confidence score of the object detection.
    shape : ImageShape
        The shape of the image of the object detection from numpy.
    """

    image: str
    """The path to the image file."""
    category: str
    """The category (or class name) of the object detection."""
    bbox: npt.NDArray[np.int64]
    """The bounding box of the object detection, as [x1, y1, x2, y2]."""
    confidence: float
    """The confidence score of the object detection."""
    shape: ImageShape
    """The shape of the image of the object detection from numpy."""

    @override
    def __repr__(self) -> str:
        return f"{self.image} @ {self.bbox}: {self.confidence}"

    def get_center_coord(self) -> tuple[int, int]:
        """
        Gets the pixel coordinate at the center of the bounding box.

        Returns
        -------
        center : tuple[int, int]
            The (x, y) pixel coordinate at the center of the bounding box.
        """
        x1, y1, x2, y2 = self.bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    @property
    def top_left(self) -> tuple[int, int]:
        """The (x, y) pixel coordinate of the top-left corner of the bounding box."""
        return int(self.bbox[0]), int(self.bbox[1])

    @property
    def top_right(self) -> tuple[int, int]:
        """The (x, y) pixel coordinate of the top-right corner of the bounding box."""
        return int(self.bbox[2]), int(self.bbox[1])

    @property
    def bottom_left(self) -> tuple[int, int]:
        """The (x, y) pixel coordinate of the bottom-left corner of the bounding box."""
        return int(self.bbox[0]), int(self.bbox[3])

    def get_x_extremes(self) -> tuple[int, int]:
        """
        Gets the minimum and maximum x values of the bounding box.

        Returns
        -------
        min_x, max_x : tuple[int, int]
        """
        return int(self.bbox[0]), int(self.bbox[2])

    def get_y_extremes(self) -> tuple[int, int]:
        """
        Gets the minimum and maximum y values of the bounding box.

        Returns
        -------
        min_y, max_y : tuple[int, int]
        """
        return int(self.bbox[1]), int(self.bbox[3])


class InferenceProvider:
    """
    A provider to run object detection inference.
    """

    def __init__(self) -> None:
        """Set up internal variables and preload resources"""

    async def start(self) -> None:
        """Start the inference provider"""
        raise NotImplementedError

    async def add_image(self, _image_path: str) -> None:
        """Run inference on the given image"""
        raise NotImplementedError

    async def end(self) -> list[ObjectDetection]:
        """Return the results of the inferences"""
        raise NotImplementedError
