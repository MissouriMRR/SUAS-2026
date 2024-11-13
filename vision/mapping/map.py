import numpy as np
import vision.common.constants as consts

class Map:
    """
    An object representing a map
    Stores the map image itself, the scale of the map, and the bounds on the coordinates of the map
    
    Parameters
    ----------
    pixels_per_foot: float
        The scale of the map in pixel *edges* per foot - for example, a value of 2 means that one square foot will
        take up a square of 4 pixels
    img: consts.ImageWAlpha
        The map image. Alpha channel will be used to store resolution data
    coord_min: consts.Point
        The minimum lat, lon of the map
    coord_max: consts.Point
        The maximum lat, lon of the map
    """
    def __init__(self, pixels_per_foot) -> None:
        self.pixels_per_foot: float = pixels_per_foot
        self.img: consts.ImageWAlpha = None
        self.coord_min = None
        self.coord_max = None
        self.img_count = 0