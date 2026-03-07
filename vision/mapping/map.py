from pathlib import Path
from pyodm import Node
import os
import time

n = Node("localhost", 3000)

print("Node info:", n.info())

image_files = sorted(
    str(path) for path in Path("vision/mapping/datasets/ODLC-Flight-2").glob("*.jpg")
)

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
