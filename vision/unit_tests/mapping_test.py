from vision.mapping_pipeline import mapping_pipeline
import asyncio 
if __name__ == "__main__":
    print("AHH")
    asyncio.run(mapping_pipeline("/SUAS-2025/flight/data/camera.json", "/SUAS-2025/images","",""))
