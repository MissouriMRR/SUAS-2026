"""Tests the mapping pipeline, using photos and data from the standard paths."""

import asyncio
import logging

from vision.mapping_pipeline import mapping_pipeline

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    asyncio.run(
        mapping_pipeline(
            "flight/data/camera.json",
            "images",
            "localhost",
            3000,
            "vision/mapping/results",
        )
    )
