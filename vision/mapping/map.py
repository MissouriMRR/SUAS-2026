from pathlib import Path
from pyodm import Node
import os
import time
import json


# Process camera.json meta data to create geo.txt


IMAGE_DIR = Path("vision/mapping/datasets/ODLC-Flight-2")
JSON_PATH = Path("vision/mapping/datasets/camera.json")
GEO_PATH = IMAGE_DIR / "geo.txt"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

lines = ["EPSG:4326"]

for image_name, entry in metadata.items():
    lat, lon = entry["drone_coordinates"]
    alt = entry.get("altitude", 0)

    # Assumes rotation_deg is already [yaw, pitch, roll].
    rot = entry.get("rotation_deg", [0, 0, 0])
    yaw, pitch, roll = rot[0], rot[1], rot[2]

    lines.append(f"{image_name} {lon} {lat} {alt} {yaw} {pitch} {roll}")

GEO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Wrote {GEO_PATH}")
n = Node("localhost", 3000)

print("Node info:", n.info())
geo_file = IMAGE_DIR / "geo.txt"
image_files = sorted(str(path) for path in IMAGE_DIR.glob("*.jpg")) + [str(geo_file)]

task = n.create_task(
    image_files,
    {
        "dsm": False,
        "fast-orthophoto": True,
        "skip-3dmodel": True,
    },
)
print("Task UUID:", task.uuid)

for _ in range(12):
    info = task.info()
    print("status:", info.status)
    try:
        print("last 10 lines:", task.output(-10))
    except Exception as e:
        print("could not read output:", e)
    time.sleep(5)

_ = task.wait_for_completion()
out = os.listdir(task.download_assets("vision/mapping/results"))
print(out)
