"""This file contains the class that implements and runs the YOLO model on images we capture."""

import asyncio
import logging
import os

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime

from vision.common.constants import Image, ImageShape

# The confidence needed to accept a prediction from the YOLO model
CONFIDENCE_THRESHOLD: float = 0.3

# Class names from the COCO dataset: https://github.com/WongKinYiu/yolov9/blob/main/data/coco.yaml
ALL_COCO_CLASSES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


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
    _results : list[npt.NDArray[np.float32]] | None
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


class YOLO:
    """
    This class implements and runs YOLO models on images we capture.

    Parameters
    ----------
    model_path : str, default="yolov9-m-converted.onnx"
        Path to the YOLO model file, by default "yolov9-m-converted.onnx"
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
        Processes the image using the YOLO model and returns the best predictions.
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
                and place them in the vision/yolo/models folder."
            )

        opt_session = onnxruntime.SessionOptions()
        opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        providers: list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.onnx_session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            full_model_path, sess_options=opt_session, providers=providers
        )
        logging.info("YOLO model loaded: %s", model_path)
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

        # Choose the best guess for each class
        best_guesses: dict[int, tuple[npt.NDArray[np.float32], float]] = {}
        for i in range(len(classes)):
            if classes[i] in best_guesses:
                if scores[i] > best_guesses[classes[i]][1]:
                    best_guesses[classes[i]] = (boxes[i], scores[i])
            else:
                best_guesses[classes[i]] = (boxes[i], scores[i])

        # These bboxes are based on the model input image size, convert back to og image size
        box: npt.NDArray[np.float32]
        for _, (box, _) in best_guesses.items():
            # Fix scaling
            x_mid: np.float32 = box[0] / self.input_width * image_size[0]  # X
            y_mid: np.float32 = box[1] / self.input_height * image_size[1]  # Y
            width: np.float32 = box[2] / self.input_width * image_size[0]
            height: np.float32 = box[3] / self.input_height * image_size[1]

            # Convert to 2 (x, y) coordinates
            box[0] = x_mid - (width / 2)  # X1
            box[1] = y_mid - (height / 2)  # Y1
            box[2] = x_mid + (width / 2)  # X2
            box[3] = y_mid + (height / 2)  # Y2

        # Log results if desired
        if self.log_results:
            logging.info(
                "Found %d detections with confidence above %.2f:",
                len(scores),
                CONFIDENCE_THRESHOLD,
            )
            class_num: int
            confidence: float
            for class_num, (box, confidence) in best_guesses.items():
                logging.info(
                    "Detected %s at (%d, %d), (%d, %d) Cfd: %.2f",
                    ALL_COCO_CLASSES[class_num],
                    box[0],
                    box[1],
                    box[2],
                    box[3],
                    confidence,
                )

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
                logging.error(error)
                raise ValueError(f"Error occurred: {error}")
            trigger.results = results
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
            {"images": processed_image},
            _onnx_callback,
            trigger,
        )
        while not trigger.trigger:
            await asyncio.sleep(0.1)

        if not trigger.results:
            raise ValueError("No results found")

        # Consolidate results to best predictions
        # Also convert to x,y coords in the original image
        results = self._filter_output(trigger.results, (width, height))

        detections: list[ObjectDetection] = [
            ObjectDetection(
                image_path,
                ALL_COCO_CLASSES[category],
                box,
                confidence,
                (height, width),
            )
            for (category, (box, confidence)) in results.items()
        ]

        return detections
