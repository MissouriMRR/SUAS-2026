"""Tests the mapping pipeline, using photos and data from the standard paths."""

import asyncio
import logging

from vision.mapping_pipeline import mapping_pipeline

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    capture_status = asyncio.Event()
    capture_status.set()  # Signal that all images are captured

    asyncio.run(
        mapping_pipeline(
            "vision/mapping/datasets/camera.json",
            "vision/mapping/datasets/ODLC-Flight-2",
            capture_status,
            "vision/mapping/results",
        )
    )
