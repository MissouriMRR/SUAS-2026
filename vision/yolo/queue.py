"""Contains the PhotoQueue class, which is used to manage a queue of photos for object detection."""

import asyncio
import collections
import logging
from typing import TypeVar, Iterable

import cv2
import numpy as np

from vision.yolo.model import YOLO, ObjectDetection

T = TypeVar("T")


class QueueCancelled(Exception):
    """Exception raised when the queue is cancelled."""


class CancellableQueue(asyncio.Queue[T]):
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
        self._queue: collections.deque[T]
        self._getters: collections.deque[asyncio.Future[None]]

    def _get(self) -> T:
        """
        Get an item from the queue.
        Overrides the _get method of the asyncio.Queue class.

        Returns
        -------
        T
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
        self.model = YOLO()
        self.queue: CancellableQueue[str] = CancellableQueue()
        self.runners: list[asyncio.Task[None]] = []
        self.results: dict[str, ObjectDetection] = {}
        self.show_results = show_results

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
        for detection in results:
            conv = detection.bbox.astype(np.int32)
            logging.info(
                "Detected %s at (%d, %d), (%d, %d)",
                detection.category,
                conv[0],
                conv[1],
                conv[2],
                conv[3],
            )
            image = cv2.rectangle(
                image,
                (conv[0], conv[1]),
                (conv[2], conv[3]),
                (0, 255, 0),
                2,
            )
        return image

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
        if self.show_results:
            cv2.namedWindow(f"Runner {num} Results")

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
            result: ObjectDetection
            for result in results:
                if result.category in self.results:
                    if result.confidence > self.results[result.category].confidence:
                        self.results[result.category] = result
                else:
                    self.results[result.category] = result
            if self.show_results:
                cv2.imshow(f"Runner {num} Results", await self._draw_results(image, results))
                cv2.waitKey(1)
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

    async def end_queue(self) -> dict[str, ObjectDetection]:
        """
        Ends the tasks of runners and returns the results.

        Returns
        -------
        dict[str, ObjectDetection]
            The results of the object detection
        """
        # Cancel the queue to stop runners once the queue is empty
        await self.queue.cancel()
        if self.runners:
            await asyncio.wait(self.runners, return_when=asyncio.ALL_COMPLETED, timeout=15)

        cv2.destroyAllWindows()

        return self.results
