"""This file contains the class that implements and runs the YOLOv9 on images we capture."""

import asyncio
import functools
import logging
import os
from typing_extensions import Self

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime

from vision.common.constants import Image, ImageShape


# We only care about the object classes that will actually appear in the competition
# You can see which are which here: https://github.com/WongKinYiu/yolov9/blob/main/data/coco.yaml
CLASS_NAMES: dict[int, str] = {
    0: "person",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    8: "boat",
    11: "stop sign",
    25: "umbrella",
    28: "suitcase",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    34: "baseball bat",
    38: "tennis racket",
    59: "bed",
}

# The confidence needed to accept a prediction from the YOLO model
CONFIDENCE_THRESHOLD: float = 0.5


class ImageTrigger:
    """
    A class used for ONNX async running to let the
    code know that the image has completed processing.

    Parameters
    ----------
    file_name : str
        The name of the file that triggered the image processing.

    Methods
    -------
    completed()
        Marks the image as having completed processing.
    results
        Returns/sets the results of the ONNX/YOLO model.
    """

    def __init__(self, file_name: str):
        self._name: str = file_name
        self._results: list[npt.NDArray[np.float32]] | None = None
        self.trigger: bool = False

    def completed(self) -> None:
        """Marks the image as having completed processing."""
        self.trigger = True

    @property
    def results(self) -> list[npt.NDArray[np.float32]] | None:
        """Returns the results of the ONNX/YOLO model.

        Returns
        -------
        list[npt.NDArray[np.float32]] | None
            The results of the ONNX/YOLO model.
        """
        return self._results

    @results.setter
    def results(self, results: list[npt.NDArray[np.float32]]) -> None:
        """Sets the results of the ONNX/YOLO model.

        Parameters
        ----------
        results : list[npt.NDArray[np.float32]]
            The results of the ONNX/YOLO model.
        """
        self._results = results


@functools.total_ordering
class ObjectDetection:
    """
    A class that stores the info of an object detection.

    Parameters
    ----------
    image_path : str
        The path to the image file.
    category : str
        The category (or class name) of the object detection.
    bbox : npt.NDArray[np.float32]
        The bounding box of the object detection.
    confidence : float
        The confidence score of the object detection.
    shape : tuple[int, ...]
        The shape of the image of the object detection from numpy.

    Methods
    -------
    __repr__()
        Returns a string representation of the ObjectDetection instance.
    __lt__(other: Self)
        Compares two ObjectDetection instances based on their confidence scores.
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

    def __init__(
        self,
        image_path: str,
        category: str,
        bbox: npt.NDArray[np.float32],
        confidence: float,
        shape: ImageShape,
    ):
        self._image_path: str = image_path
        self._category: str = category
        self._bbox: npt.NDArray[np.float32] = bbox
        self._confidence: float = confidence
        self._shape: ImageShape = shape

    def __repr__(self) -> str:
        return f"{self.image} @ {self.bbox}: {self.confidence}"

    def __lt__(self, other: Self) -> bool:
        """Compares two ObjectDetection instances based on their confidence scores.

        Parameters
        ----------
        other : ObjectDetection
            The other ObjectDetection instance to compare with.

        Returns
        -------
        bool
            True if has a lower confidence score than the other instance, False otherwise.
        """
        if self.category != other.category:
            raise NotImplementedError("These two categories do not match")
        return self.confidence < other.confidence

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
    def bbox(self) -> npt.NDArray[np.float32]:
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


class YOLOv9:
    """
    This class implements and runs the YOLOv9 model on images we capture.

    Parameters
    ----------
    model_path : str, default="yolov9-m-converted.onnx"
        Path to the YOLOv9 model file, by default "yolov9-m-converted.onnx"
    log_results : bool, default=False
        Whether to log the results of the object detection, by default False

    Raises
    ------
    FileNotFoundError
        If the model file is not found in the specified path

    Methods
    -------
    _convert_image(image: cv2.typing.MatLike)
        Converts the image to a model-readable format.
    _filter_output(output: list[np.ndarray], image_size: tuple[int, int])
        Converts the output array of the model to the best predictions of the model.
    process_image(image_path: str)
        Processes the image using the YOLOv9 model and returns the best predictions.
    """

    def __init__(
        self, model_path: str = "yolov9-m-converted.onnx", log_results: bool = False
    ) -> None:
        # All models should be put in the models folder.
        full_model_path: str = os.path.join(os.path.dirname(__file__), "models", model_path)
        if not os.path.exists(full_model_path):
            raise FileNotFoundError(
                f"Model file {full_model_path} not found. Check the documentation \
                website for a link to download the models, \
                and place them in the vision/yolov9/models folder."
            )

        opt_session = onnxruntime.SessionOptions()
        opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        providers: list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.onnx_session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            full_model_path, sess_options=opt_session, providers=providers
        )
        logging.info("YOLOv9 model loaded: %s", model_path)
        self.model_output: list[onnxruntime.NodeArg] = self.onnx_session.get_outputs()
        self.output_names: list[str] = [
            self.model_output[i].name for i in range(len(self.model_output))
        ]
        self.input_shape: list[int] = self.onnx_session.get_inputs()[0].shape
        self.input_height: int = self.input_shape[2]
        self.input_width: int = self.input_shape[3]
        self.log_results: bool = log_results

    def _convert_image(self, image: Image) -> npt.NDArray[np.float32]:
        """
        Convert an image read through cv2 to a shaped array for model input.

        Parameters
        ----------
        image : Image
            The image to convert.

        Returns
        -------
        npt.NDArray[np.float32]
            The converted image, scaled from 0 to 1, and to the input size of the model.
        """
        # Convert the cv2 image to RGB and resize it to the input size of the model
        image_rgb: cv2.typing.MatLike = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized: cv2.typing.MatLike = cv2.resize(image_rgb, (self.input_width, self.input_height))

        # Scale input pixel value to 0 to 1
        input_image: npt.NDArray[np.float32] = resized / 255.0
        input_image = input_image.transpose(2, 0, 1)
        input_tensor: npt.NDArray[np.float32] = input_image[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def _filter_output(
        self, output: list[npt.NDArray[np.float32]], image_size: tuple[int, int]
    ) -> dict[int, tuple[npt.NDArray[np.float32], float]]:
        """
        Converts the output array of the model to a dictionary of the best predictions.

        Parameters
        ----------
        output : list[npt.NDArray[np.float32]]
            The output array from the model.
        image_size : tuple[int, int]
            The pixel size of the input image. (W, H)

        Returns
        -------
        dict[int, tuple[npt.NDArray[np.float32], float]]
            A dictionary of the best predictions.
        """
        # The output is formatted in 84 rows
        # The first 4 are the x, y, width, and height of the detection
        # The next 80 are the confidence of the detection being each class
        squeezed: npt.NDArray[np.float32] = np.squeeze(output)

        # Look for the max confidence, removing the first 4 elements because its the bbox
        confidences: npt.NDArray[np.float32] = np.max(squeezed[4:, :], axis=0)

        # Filter out the detections with low confidence
        inferences: npt.NDArray[np.float32] = squeezed[:, confidences > CONFIDENCE_THRESHOLD]
        scores: npt.NDArray[np.float32] = confidences[confidences > CONFIDENCE_THRESHOLD]

        # Get the column index of the best confidence score
        classes: npt.NDArray[np.float32] = np.argmax(inferences[4:, :], axis=0)

        # Get just the bounding box info, transpose
        boxes: npt.NDArray[np.float32] = inferences[:4, :].T

        # We only care about the classes that will appear in competition, choose the best one
        best_guesses: dict[int, tuple[npt.NDArray[np.float32], float]] = {}
        for i in range(len(classes)):
            if classes[i] not in CLASS_NAMES:
                continue
            if classes[i] in best_guesses:
                if scores[i] > best_guesses[classes[i]][1]:
                    best_guesses[classes[i]] = (boxes[i], scores[i])
            else:
                best_guesses[classes[i]] = (boxes[i], scores[i])

        # These bboxes are based on the model input image size, convert back to og image size
        for _, (box, _) in best_guesses.items():
            # Fix scaling
            box[0] = box[0] / self.input_width * image_size[0]  # X
            box[1] = box[1] / self.input_height * image_size[1]  # Y
            box[2] = box[2] / self.input_width * image_size[0]  # W
            box[3] = box[3] / self.input_height * image_size[1]  # H

            # Convert to 2 (x, y) coordinates
            box[0] -= box[2] / 2  # X1
            box[1] -= box[3] / 2  # Y1
            box[2] += box[0]  # X2
            box[3] += box[1]  # Y2

        return best_guesses

    async def process_image(self, image_path: str) -> list[ObjectDetection]:
        """
        Process an image in the model and return all detected objects and their info.

        Parameters
        ----------
        image_path : str
            Path to the image to be processed.

        Returns
        -------
        list[ObjectDetection]
            List of detected objects and their best prediction info.
        """

        def _onnx_callback(
            results: list[npt.NDArray[np.float32]], trigger: ImageTrigger, error: str
        ) -> None:
            """
            Callback that ONNX runs in a separate thread.

            Parameters
            ----------
            results : list[npt.NDArray[np.float32]]
                List of detected objects and their best prediction info.
            trigger : ImageTrigger
                Trigger object to signal completion.
            error : str
                Error message if any.

            Raises
            ------
            ValueError
                If an error occurs during processing.
            """
            if error:
                raise ValueError(f"Error occurred: {error}")
            trigger.results = results
            trigger.completed()

        image: Image = cv2.imread(image_path).astype(np.uint8)
        height: int
        width: int
        height, width = image.shape[:2]
        processed_image: npt.NDArray[np.float32] = self._convert_image(image)
        trigger: ImageTrigger = ImageTrigger(image_path)
        logging.info("Processing image: %s", image_path)
        self.onnx_session.run_async(
            self.output_names,
            {"images": processed_image},
            _onnx_callback,
            trigger,
        )
        while not trigger.trigger:
            await asyncio.sleep(0.1)

        if not trigger.results:
            raise ValueError("No results found")

        # Results are ready to process
        results: dict[int, tuple[npt.NDArray[np.float32], float]] = self._filter_output(
            trigger.results, (width, height)
        )
        detections: list[ObjectDetection] = [
            ObjectDetection(
                image_path,
                CLASS_NAMES[category],
                box,
                confidence,
                (height, width),
            )
            for (category, (box, confidence)) in results.items()
        ]

        return detections
