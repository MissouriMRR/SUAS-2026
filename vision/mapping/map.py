import numpy as np

class Map:
    def __init__(self, pixels_per_foot, center_coord) -> None:
        self.img = None
        self.pixels_per_foot = pixels_per_foot
        self.center_coord = center_coord