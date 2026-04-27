# NOTE don't remember what this was for

import json
from pathlib import Path

IMAGE_DIR = Path("/path/to/images")
JSON_PATH = Path("/path/to/metadata.json")
GEO_PATH = IMAGE_DIR / "geo.txt"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

lines = ["EPSG:4326"]

for image_name, entry in metadata.items():
    lat, lon = entry["drone_coordinates"]
    alt = entry.get("altitude", 0)

    # Assumes rotation_deg is already [yaw, pitch, roll].
    # If your source uses a different convention, adjust this mapping.
    rot = entry.get("rotation_deg", [0, 0, 0])
    yaw, pitch, roll = rot[0], rot[1], rot[2]

    lines.append(f"{image_name} {lon} {lat} {alt} {yaw} {pitch} {roll}")

GEO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Wrote {GEO_PATH}")
