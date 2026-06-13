"""Runs the necessary code for the Mapping component of the competition"""

import json
from pyodm import Node
import os
import logging
from ctypes import c_bool
from multiprocessing.sharedctypes import SynchronizedBase  # pylint: disable=unused-import

import asyncio
from pathlib import Path
import time


async def mapping_pipeline(
    camera_data_path: str,
    image_dir: str,
    capture_status: asyncio.Event,
    state_path: str,
    output_path: str = "vision/mapping/results",
) -> None:
    """
    Runs the code that generates the map from a folder of photos

    Parameters
    ----------
    camera_data_path: str
        The path to the json file containing the CameraParameters
    image_dir: str
        The path with all the images
    capture_status: asyncio.Event
        Once all the images are taken, capture status is set to true (used to determine when to generate the map)
    state_path: str
        A text file containing True if all images have been taken and False otherwise
    output_path: str
        The json file name and path to save the data in
    """

    # Check capture status to see if all images have been taken
    while not capture_status.is_set():
        await asyncio.sleep(1)

    # Read in the camera JSON data file and write formatted data to the geo.txt (needed for ODM)
    geo_path: Path = Path(image_dir) / "geo.txt"  # TODO verify this creates the file

    with open(camera_data_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Write geo.txt with data from camera json
    lines = ["EPSG:4326"]

    for image_name, entry in metadata.items():
        lat, lon = entry["drone_coordinates"]
        alt = entry.get("altitude", 0)

        # Assumes rotation_deg is already [yaw, pitch, roll].
        rot = entry.get("rotation_deg", [0, 0, 0])
        yaw, pitch, roll = rot[0], rot[1], rot[2]

        lines.append(f"{image_name} {lon} {lat} {alt} {yaw} {pitch} {roll}")

    geo_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {geo_path}")

    # Set up connection to ODM node
    n = Node("localhost", 3000)
    print("Node info:", n.info())
    image_files = sorted(str(path) for path in Path(image_dir).glob("*.jpg")) + [str(geo_path)]

    # Create the ODM task (tweak settings here as needed)
    task = n.create_task(
        image_files,
        {
            "dsm": False,
            "fast-orthophoto": True,
            "skip-3dmodel": True,
        },
    )
    print("Task UUID:", task.uuid)

    _ = task.wait_for_completion()
    try:
        out = os.listdir(task.download_assets(output_path))
    except Exception as e:
        print(e)
        out = ""
    print(out)
