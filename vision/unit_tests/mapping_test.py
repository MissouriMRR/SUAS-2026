from vision.mapping_pipeline import mapping_pipeline
import asyncio

if __name__ == "__main__":
    capture_status = asyncio.Event()
    capture_status.set()  # Signal that all images are captured

    print("AHH")
    asyncio.run(
        mapping_pipeline(
            "./vision/mapping/datasets/camera.json",
            "./vision/mapping/datasets/ODLC-Flight-2",
            capture_status,
            "",
            "vision/mapping/results",
        )
    )
