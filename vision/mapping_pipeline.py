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


# Disable duplicate code checking because the flyover pipeline is similar
# pylint: disable=duplicate-code
async def mapping_pipeline(
    camera_data_path: str, image_dir: str, state_path: str, output_path: str
) -> None:
    """
    Runs the code that generates the map from a folder of photos

    Parameters
    ----------
    camera_data_path: str
        The path to the json file containing the CameraParameters entries
    state_path: str
        A text file containing True if all images have been taken and False otherwise
    output_path: str
        The json file name and path to save the data in
    """

    # Wait for and process unfinished images until no more images are being taken
    all_images_taken: c_bool = c_bool(True) # TODO may need to modify to false
    first_check = True
    images_needed_filler_value = 20 # TODO modify this. We need to find a way to determine when all images have been taken. 

    while not all_images_taken:
        image_count = sum(1 for entry in os.scandir(image_dir) if entry.is_file())
        if(image_count == images_needed_filler_value):
            all_images_taken = True
        # Wait to check the file instead of spamming it
        if not first_check:
            await asyncio.sleep(1)

        first_check = False
        # NOTE
        # Check if all images have been taken
        # all_images_taken = capture_status.value  # type: ignore TODO integrate capture_status after basic setup

    # # Load in the json containing the camera data
    # image_parameters: dict[str, consts.CameraParameters] = pipe_utils.read_parameter_json(
    #     camera_data_path
    # )

    # Read in the camera JSON data file and write formatted data to the geo.txt (needed for ODM)
    geo_path : Path = Path(image_dir) / "geo.txt" # TODO verify this

    with open(camera_data_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

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
    n = Node("localhost", 3000)

    print("Node info:", n.info())
    geo_file = Path(image_dir) / "geo.txt"
    image_files = sorted(str(path) for path in Path(image_dir).glob("*.jpg")) + [
        str(geo_file)
    ]

    task = n.create_task(
        image_files,
        {
            "dsm": False,
            "fast-orthophoto": True,
            "skip-3dmodel": True,
        },
    )
    print("Task UUID:", task.uuid)

    # for _ in range(12):
    #     info = task.info()
    #     print("status:", info.status)
    #     try:
    #         print("last 10 lines:", task.output(-10))
    #     except Exception as e:
    #         print("could not read output:", e)
    #     time.sleep(5)

    _ = task.wait_for_completion()
    try:
        out = os.listdir(task.download_assets("vision/mapping/results"))
    except Exception as e:
        print(e)
        out = ""
    print(out)