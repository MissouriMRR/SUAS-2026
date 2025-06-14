from vision.mapping_pipeline import mapping_pipeline
import asyncio 
if __name__ == "__main__":
    print("AHH")
    asyncio.run(mapping_pipeline("mapping_photos.json", "Mapping","",""))
