"""Contains the PhotoQueue class, which is used to manage a queue of photos for object detection."""

import asyncio
import collections
import logging
from collections.abc import Iterable
from typing import TypeVar, cast, override

import cv2
import numpy as np

from vision.object_detection.providers.base import ObjectDetection
from vision.object_detection.providers.local.model import ObjectDetectionModel

QueueItem = TypeVar("QueueItem")


class ImageReadError(Exception):
    """Raised when an image cannot be read for inference."""


class QueueCancelled(Exception):
    """Exception raised when the queue is cancelled."""


class CancellableQueue(asyncio.Queue[QueueItem]):
    """
    A subclass of the asyncio Queue that adds an event to cancel the queue.

    Attributes
    ----------
    cancelled: bool
        Whether the queue has been cancelled.

    Methods
    -------
    empty(self) -> bool
        Returns True if the queue is empty. Raises QueueCancelled if cancelled and the queue empty.
    cancel(self) -> None
        Sets self._cancelled to True which will raise QueueCancelled on get().
    """

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize)
        self._cancelled: bool = False
        self._queue: collections.deque[QueueItem]
        self._getters: collections.deque[asyncio.Future[None]]

    @override
    def _get(self) -> QueueItem:
        """
        Get an item from the queue.
        Overrides the _get method of the asyncio.Queue class.

        Returns
        -------
        QueueItem
            The next item from the queue

        Raises
        ------
        QueueCancelled
            If the queue is empty and cancelled.
        """
        if self.empty() and self._cancelled:
            raise QueueCancelled()
        return self._queue.popleft()

    async def cancel(self) -> None:
        """
        Sets self._cancelled to True which will raise QueueCancelled on get().
        """
        self._cancelled = True
        await self.join()
        while self._getters:
            getter = self._getters.popleft()
            getter.set_exception(QueueCancelled())


class PhotoQueue:
    """
    Manages a queue of photos for object detection, and runners for running the model inference.

    Attributes
    ----------
    model: YOLO
        The YOLO model used for inference.
    queue: CancellableQueue
        The queue of photos to be processed.
    runners: list[asyncio.Task[None]]
        The list of runners that are running the model inference on the photos.
    results: dict[str, ObjectDetection]
        The dictionary of results for each object class.
    show_results: bool
        Whether to show the results of the object detection.

    Methods
    -------
    add_photo(photo_path: str)
        Adds a photo to the queue.
    photo_runner(num: int)
        A runner that will check for photos in the queue, run inference on them, and store results.
    start_queue(max_runners: int = 3)
        Starts the photo runners.
    end_queue()
        Stops the photo runners, and returns the results.
    """

    def __init__(self, show_results: bool = False):
        self.model: ObjectDetectionModel = ObjectDetectionModel()
        self.queue: CancellableQueue[str] = CancellableQueue()
        self.runners: list[asyncio.Task[None]] = []
        self.results: list[ObjectDetection] = []
        self.show_results: bool = show_results

    async def _draw_results(
        self, photo: str, results: Iterable[ObjectDetection]
    ) -> cv2.typing.MatLike:
        """
        Draw detected object bboxes on the image.

        Parameters
        ----------
        photo : str
            Path to the image file.
        results : Iterable[ObjectDetection]
            Iterable of detected objects.

        Returns
        -------
        cv2.typing.MatLike
            Image with bounding boxes drawn.
        """
        image = cv2.imread(photo)
        if image is None:
            raise ImageReadError(f"Failed to read image: {photo}")
        for detection in results:
            x1, y1, x2, y2 = cast(
                tuple[int, int, int, int], detection.bbox.astype(np.int32).tolist()
            )
            logging.info(
                "Detected %s at (%d, %d), (%d, %d)",
                detection.category,
                x1,
                y1,
                x2,
                y2,
            )
            image = cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )
            # Add caption with class
            _ = cv2.putText(
                image,
                detection.category,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_TRIPLEX,
                (x2 - x1) / 250,
                (0, 255, 0),
            )
        return image

    def _print_results(self) -> None:
        """
        Prints out all of the stored results, including bbox, confidence, and image path.
        """
        for result in self.results:
            x1, y1, x2, y2 = cast(
                tuple[int, int, int, int], result.bbox.astype(np.int32).tolist()
            )
            logging.info(
                "Detected %s at (%d, %d), (%d, %d) Cfd: %.2f Img: %s",
                result.category,
                x1,
                y1,
                x2,
                y2,
                result.confidence,
                result.image,
            )

    async def add_photo(self, photo_path: str) -> None:
        """
        Add a photo to the queue for processing.

        Parameters
        ----------
        photo_path : str
            The path to the photo to be added to the queue.
        """
        self.queue.put_nowait(photo_path)

    async def photo_runner(self, num: int) -> None:
        """
        Runs the photo processing loop.

        Parameters
        ----------
        num : int
            The id of the runner.
        """
        window_title: str = f"Runner {num} Results"
        if self.show_results:
            cv2.namedWindow(window_title, cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(window_title, 1280, 720)

        while True:
            # We want to cancel the task if we are done capturing images,
            # but not until the queue is empty
            # queue_task = asyncio.create_task(self.queue.get())
            # event_task = asyncio.create_task(self.done_capturing.wait())

            # Wait for either one to complete
            try:
                image: str = await self.queue.get()
            except QueueCancelled:
                logging.debug("Runner %d cancelled", num)
                break
            logging.debug("Runner %d processing image %s", num, image)
            results = await self.model.process_image(image)

            # Add each result to the results list
            self.results.extend(results)

            if self.show_results:
                try:
                    annotated_image = await self._draw_results(image, results)
                    cv2.imshow(window_title, annotated_image)
                    _ = cv2.waitKey(1)
                except ImageReadError:
                    logging.error("Failed to show results for image: %s", image)
            self._print_results()
            self.queue.task_done()
            logging.debug("Runner %d finished processing image %s", num, image)

        if self.show_results:
            cv2.destroyWindow(f"Runner {num} Results")

    async def start_queue(self, max_runners: int = 3) -> None:
        """
        Starts the queue of runners to process images.

        Parameters
        ----------
        max_runners : int, default=3
            The maximum number of runners to start, by default 3
        """
        for i in range(max_runners):
            runner = asyncio.create_task(self.photo_runner(i + 1))
            self.runners.append(runner)

    async def end_queue(self) -> list[ObjectDetection]:
        """
        Ends the tasks of runners and returns the results.

        Returns
        -------
        list[ObjectDetection]
            The results of the object detection
        """
        # Cancel the queue to stop runners once the queue is empty
        await self.queue.cancel()
        if self.runners:
            _ = await asyncio.wait(
                self.runners, return_when=asyncio.ALL_COMPLETED, timeout=15
            )

        cv2.destroyAllWindows()

        return self.results
