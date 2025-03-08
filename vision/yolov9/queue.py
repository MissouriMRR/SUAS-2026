"""Contains the PhotoQueue class, which is used to manage a queue of photos for object detection."""

import asyncio
import logging

import cv2
import numpy as np

from vision.yolov9.model import YOLOv9, ObjectDetection


class PhotoQueue:
    """
    Manages a queue of photos for object detection, and runners for running the model inference.

    Attributes
    ----------
    model: YOLOv9
        The YOLOv9 model used for inference.
    queue: asyncio.Queue[str]
        The queue of photos to be processed.
    runners: list[asyncio.Task[None]]
        The list of runners that are running the model inference on the photos.
    results: dict[str, ObjectDetection]
        The dictionary of results for each object class.
    done_capturing: asyncio.Event
        The event that is set when all photos have been captured, telling the runners to stop.
    show_results: bool
        Whether to show the results of the object detection.

    Methods
    -------
    _draw_results(photo: str, results: list[ObjectDetection])
        Draws the bboxes of the object detection on the photo.
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
        self.model = YOLOv9()
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.runners: list[asyncio.Task[None]] = []
        self.results: dict[str, ObjectDetection] = {}
        self.done_capturing: asyncio.Event = asyncio.Event()
        self.show_results = show_results

    async def _draw_results(self, photo: str, results: list[ObjectDetection]) -> cv2.typing.MatLike:
        """
        Draw detected object bboxes on the image.

        Parameters
        ----------
        photo : str
            Path to the image file.
        results : list[ObjectDetection]
            List of detected objects.

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
            queue_task = asyncio.create_task(self.queue.get())
            event_task = asyncio.create_task(self.done_capturing.wait())

            # Wait for either one to complete
            _, pending = await asyncio.wait(
                [queue_task, event_task], return_when=asyncio.FIRST_COMPLETED
            )

            if self.done_capturing.is_set():
                # wait for all the other tasks to finish just in case
                for task in pending:
                    task.cancel()
                await asyncio.wait(pending)
                break

            image: str = queue_task.result()
            logging.debug("Runner %d processing image %s", num, image)
            results = await self.model.process_image(image)
            for result in results:
                if result.category in self.results:
                    if result > self.results[result.category]:
                        self.results[result.category] = result
                else:
                    self.results[result.category] = result
            if self.show_results:
                cv2.imshow(f"Runner {num} Results", await self._draw_results(image, results))
                cv2.waitKey(1)
            self.queue.task_done()

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
        # Set event to tell runners to stop once they don't have any images
        self.done_capturing.set()

        # Let runners finish going through the queue
        await asyncio.gather(*self.runners)

        return self.results
