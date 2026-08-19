"""Sets up the base InferenceProvider class."""

from typing_extensions import override
import numpy as np
import numpy.typing as npt

from vision.common.constants import ImageShape


class ObjectDetection:
    """
    A class that stores the info of an object detection.

    Parameters
    ----------
    image_path : str
        The path to the image file.
    category : str
        The category (or class name) of the object detection.
    bbox : npt.NDArray[np.int64]
        The bounding box of the object detection.
    confidence : float
        The confidence score of the object detection.
    shape : tuple[int, ...]
        The shape of the image of the object detection from numpy.

    Methods
    -------
    __repr__()
        Returns a string representation of the ObjectDetection instance.
    category()
        Returns the category of the object detection.
    bbox()
        Returns the bounding box of the object detection.
    confidence()
        Returns the confidence score of the object detection.
    image()
        Returns the image path of the object detection.
    shape()
        Returns the shape of the image of the object detection.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        image_path: str,
        category: str,
        bbox: npt.NDArray[np.int64],
        confidence: float,
        shape: ImageShape,
    ):
        self._image_path: str = image_path
        self._category: str = category
        self._bbox: npt.NDArray[np.int64] = bbox
        self._confidence: float = confidence
        self._shape: ImageShape = shape

    @override
    def __repr__(self) -> str:
        return f"{self.image} @ {self.bbox}: {self.confidence}"

    @property
    def category(self) -> str:
        """
        Returns the category (or class name) of the object detection.

        Returns
        -------
        str
            The category (or class name) of the object detection.
        """
        return self._category

    @property
    def bbox(self) -> npt.NDArray[np.int64]:
        """
        Returns the bounding box of the object detection.

        Returns
        -------
        npt.NDArray[np.float32]
            The bounding box of the object detection.
        """
        return self._bbox

    @property
    def confidence(self) -> float:
        """
        Returns the confidence of the object detection.

        Returns
        -------
        float
            The confidence of the object detection.
        """
        return self._confidence

    @confidence.setter
    def confidence(self, value: float) -> None:
        """
        Sets the confidence of the object detection.

        Parameters
        ----------
        value : float
            The new confidence value.
        """
        self._confidence = value

    @property
    def image(self) -> str:
        """
        Returns the path to the image of the object detection.

        Returns
        -------
        str
            The path to the image of the object detection.
        """
        return self._image_path

    @property
    def shape(self) -> ImageShape:
        """
        Returns the shape of the image of the object detection.

        Returns
        -------
        ImageShape
            The shape of the image of the object detection.
        """
        return self._shape


class InferenceProvider:
    """
    A provider to run object detection inference.
    """

    def __init__(self) -> None:
        """Set up internal variables and preload resources"""

    async def start(self) -> None:
        """Start the inference provider"""
        raise NotImplementedError

    async def add_image(self, image_path: str) -> None:
        """Run inference on the given image"""
        raise NotImplementedError

    async def end(self) -> list[ObjectDetection]:
        """Return the results of the inferences"""
        raise NotImplementedError
