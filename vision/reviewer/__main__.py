"""Entry point for the object detection reviewer app.

Run with `uv run -m vision.reviewer [detections.json] [--images IMAGE_ROOT]`.
"""

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from PySide6.QtWidgets import QApplication

from vision.common.constants import DEFAULT_DETECTIONS_OUTPUT_PATH
from vision.reviewer.main_window import ReviewWindow
from vision.reviewer.models import (
    SUAS_ROOT,
    ReviewSession,
)

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)


@app.command()
def main(
    detection_data: Annotated[
        Path,
        typer.Option(
            help="path to detections JSON file, output from create_review_JSON()"
        ),
    ] = DEFAULT_DETECTIONS_OUTPUT_PATH,
    images: Annotated[
        Path,
        typer.Option(help="path to the folder containing the images"),
    ] = SUAS_ROOT,
) -> int:
    """Loads a detections JSON file and opens the reviewer window."""
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Loading detections from {detection_data}, images from {images}")

    session = ReviewSession.load(detection_data, images)
    if not session.detections:
        logger.error(f"No detections in {detection_data}, check your data file")
        return 1

    app = QApplication(sys.argv[:1])
    window = ReviewWindow(session)
    window.show()
    return app.exec()


if __name__ == "__main__":
    app()
