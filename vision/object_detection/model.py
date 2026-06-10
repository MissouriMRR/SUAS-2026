"""This file contains the class that implements and runs the model on images we capture."""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime
from cv2.typing import MatLike

from vision.common.constants import Image, ImageShape

# The confidence needed to accept a prediction from the model
CONFIDENCE_THRESHOLD: float = 0.8

# Class mappings for the model outputs
CLASS_MAPPINGS = {1: "person", 2: "tent"}


@dataclass
class ModelOutput:
    """A dataclass representing the output of the object detection model."""

    bboxs: npt.NDArray[np.float32]
    labels: npt.NDArray[np.int64]
    scores: npt.NDArray[np.float32]

    def __init__(self, output: list[npt.NDArray[np.float32 | np.int64]]) -> None:
        self.bboxs = output[0].astype(np.float32)
        self.labels = output[1].astype(np.int64)
        self.scores = output[2].astype(np.float32)


class ImageTrigger:
    """
    A class used for ONNX async running to let the
    code know that the image has completed processing.

    Parameters
    ----------
    file_name : str
        The name of the file that triggered the image processing.

    Attributes
    ----------
    _name : str
        The name of the file that triggered the image processing.
    _results : ModelOutput | None
        The results output from the ONNX/YOLO model.
    trigger : bool
        Indicates whether the image has completed processing.

    Methods
    -------
    completed()
        Marks the image as having completed processing.
    results
        Returns/sets the results of the ONNX/YOLO model.
    """

    def __init__(self, file_name: str):
        self._name: str = file_name
        self._results: ModelOutput | None = None
        self.trigger: bool = False

    def completed(self) -> None:
        """Marks the image as having completed processing."""
        self.trigger = True

    @property
    def results(self) -> ModelOutput | None:
        """Returns the results of the ONNX model.

        Returns
        -------
        ModelOutput | None
            The results of the ONNX model.
        """
        return self._results

    @results.setter
    def results(self, results: ModelOutput) -> None:
        """Sets the results of the ONNX model.

        Parameters
        ----------
        results : ModelOutput
            The results of the ONNX model.
        """
        self._results = results


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


class ObjectDetectionModel:
    """
    This class implements and runs object detection models on images we capture.

    Parameters
    ----------
    model_path : str, default="yolov9-m-converted.onnx"
        Path to the model ONNX file, by default "yolov9-m-converted.onnx"
    log_results : bool, default=False
        Whether to log the results of the object detection, by default False

    Attributes
    ----------
    onnx_session : onnxruntime.InferenceSession
        The inference session with the needed settings for
        inferencing the YOLO model.
    model_output : list[onnxruntime.NodeArg]
        A list of outputs generated by the model.
    output_names : list[str]
        A list of the names of the outputs generated by the model.
    input_shape : list[int]
        The shape of the input image expected by the model.
    input_height : int
        The height of the input image expected by the model.
    input_width : int
        The width of the input image expected by the model.
    log_results : bool
        Whether to log the results of the object detection.

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
        Processes the image using the model and returns the best predictions.
    """

    def __init__(
        self,
        model_path: str = "mannequin_tent_cnn_resnet50.onnx",
        log_results: bool = False,
    ) -> None:
        # All models should be put in the models folder.
        full_model_path: str = os.path.join(os.path.dirname(__file__), "models", model_path)
        if not os.path.exists(full_model_path):
            raise FileNotFoundError(
                f"Model file {full_model_path} not found. Check the documentation \
                website for a link to download the models, \
                and place them in the models folder."
            )

        # Set up options and providers. We want to use CUDA if available, otherwise CPU,
        # and ORT_DISABLE_ALL was required in the past to run through WSL
        opt_session = onnxruntime.SessionOptions()
        opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        providers: list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.onnx_session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            full_model_path, sess_options=opt_session, providers=providers
        )
        logging.info("Model loaded: %s", model_path)
        self.model_output: Sequence[onnxruntime.NodeArg] = self.onnx_session.get_outputs()
        # We get output names so that we can return all model outputs on callback
        self.output_names: list[str] = [
            self.model_output[i].name for i in range(len(self.model_output))
        ]
        self.input_shape: list[int] = self.onnx_session.get_inputs()[0].shape
        self.input_height: int = self.input_shape[1]
        self.input_width: int = self.input_shape[2]
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
        image_rgb: MatLike = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized: MatLike = cv2.resize(image_rgb, (self.input_width, self.input_height))

        # Transpose from (height, width, channels) to (channels, height, width)
        input_image: MatLike = resized.transpose(2, 0, 1).astype(np.float32)
        # Scale input pixel value to 0 to 1
        input_image = input_image / 255.0
        return input_image

    def _filter_output(
        self, output: ModelOutput, image_shape: tuple[int, int]
    ) -> list[tuple[npt.NDArray[np.int64], str, np.float32]]:
        """
        Converts the output of the model to a list of the best predictions for each class.

        Parameters
        ----------
        output : ModelOutput
            The output from the model.
        image_shape : tuple[int, int]
            The shape of the original image.

        Returns
        -------
        list[tuple[npt.NDArray[np.int64], str, np.float32]]
            A list of the best predictions for each class.
        """
        # Zip all results together before filtering
        # Labels and scores are expanded to make it possible to concatenate them with bboxs
        results: npt.NDArray[np.float32] = np.concatenate(
            [
                output.bboxs,
                np.expand_dims(output.labels, axis=1).astype(np.float32),
                np.expand_dims(output.scores, axis=1),
            ],
            axis=1,
        )

        # Get indices of best results for each class and filter to those results
        # [5:] is to take the score column only, skip the bbox columns
        # Sort all detections by score descending
        sorted_indices: npt.NDArray[np.int64] = np.argsort(results[:, 5])[::-1]

        # First occurrence of each class in the sorted order = best score for that class
        _, best_indices = np.unique(results[sorted_indices, 4], return_index=True)

        # Take the results from best_indices and keep the rest of the data associated with them
        results = results[sorted_indices[best_indices], :]

        # Filter the results to only include those with high confidence
        results = results[results[:, -1] > CONFIDENCE_THRESHOLD]

        # These bboxes are based on the model input image size, convert back to og image size
        x1: np.float32
        y1: np.float32
        x2: np.float32
        y2: np.float32
        class_num: np.float32
        confidence: np.float32
        final_results: list[tuple[npt.NDArray[np.int64], str, np.float32]] = []
        for x1, y1, x2, y2, class_num, confidence in results:
            # Fix scaling
            width_mul: np.float32 = np.float32(image_shape[0]) / self.input_width
            height_mul: np.float32 = np.float32(image_shape[1]) / self.input_height

            # Convert to 2 (x, y) coordinates
            x1 = x1 * width_mul  # X1
            y1 = y1 * height_mul  # Y1
            x2 = x2 * width_mul  # X2
            y2 = y2 * height_mul  # Y2
            box: npt.NDArray[np.int64] = np.rint(np.array([x1, y1, x2, y2]))

            final_results.append((box, CLASS_MAPPINGS[int(class_num)], confidence))

        # Log results if desired
        if self.log_results:
            logging.info(
                "Found %d detections with confidence above %.2f:",
                len(final_results),
                CONFIDENCE_THRESHOLD,
            )
            for box, class_name, confidence in final_results:
                logging.info(
                    "Detected %s at (%d, %d), (%d, %d) Cfd: %.2f",
                    class_name,
                    box[0],
                    box[1],
                    box[2],
                    box[3],
                    confidence,
                )

        return final_results

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
            List of detected objects and their prediction info.
        """

        def _onnx_callback(
            results: list[npt.NDArray[np.float32 | np.int64]],
            trigger: ImageTrigger,
            error: str,
        ) -> None:
            """
            Callback that ONNX runs in a separate thread.

            Parameters
            ----------
            results : list[npt.NDArray[np.float32 | np.int64]]
                List of detected objects and their prediction info.
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
                logging.error(error)
                raise ValueError(f"Error occurred: {error}")
            trigger.results = ModelOutput(results)
            trigger.completed()

        raw_image: cv2.typing.MatLike | None = cv2.imread(image_path)
        if raw_image is None:
            logging.error("%s is not an image, skipping", image_path)
            return []
        image: Image = raw_image.astype(np.uint8)
        logging.info("Processing image: %s", image_path)
        height: int
        width: int
        height, width = image.shape[:2]
        processed_image: npt.NDArray[np.float32] = self._convert_image(image)

        trigger: ImageTrigger = ImageTrigger(image_path)
        self.onnx_session.run_async(
            self.output_names,
            {"image": processed_image},
            _onnx_callback,
            trigger,
        )
        while not trigger.trigger:
            await asyncio.sleep(0.1)

        if not trigger.results:
            raise ValueError("No results found")

        # Consolidate results to best predictions for each category
        # Also convert to x,y coords in the original image
        results = self._filter_output(trigger.results, (width, height))

        detections: list[ObjectDetection] = [
            ObjectDetection(
                image_path,
                category,
                box,
                confidence.item(),
                (height, width),
            )
            for box, category, confidence in results
        ]

        return detections
