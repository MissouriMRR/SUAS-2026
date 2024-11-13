import numpy as np
import cv2

import vision.common.constants as consts

from vision.mapping.map import Map
from vision.deskew import coordinate_lengths

def superimpose(map: Map, new_img: consts.ImageWAlpha, new_img_offset: consts.ImgPoint):
    # The amount of padding to add to the map in pixels
    # If any axis of new_img_offset is negative, the map must be padded in that direction
    map_padding = np.array([min(0, new_img_offset[0]), min(0, new_img_offset[1])])
    
    pass


def add_image(map_img: consts.Image, proj_img: consts.ImageWAlpha, center_coord: consts.Point):
    """
    Adds an image to a map
    
    Parameters
    ----------
    map:
        the map to update
    proj_img:
        the projected image, correctly scaled to match the map's scale
    center_coord:
        the coordinate in the center of the projected image, in the format (lat, lon)
    
    Returns
    -------
    updated_map:
        the updated map
    """
    
    coordinate_change = center_coord - map.center_coord
    
    # Calculate the latitude and longitude lengths (in meters)
    latitude_length: float = coordinate_lengths.latitude_length(
        map.center_coord[0]
    )
    longitude_length: float = coordinate_lengths.longitude_length(
        map.center_coord[0]
    )
    
    relative_coord_ft = np.array(
        [coordinate_change[0] * latitude_length, coordinate_change[1] * longitude_length]
    )
    
    pixel_offset = np.round(relative_coord_ft * map.pixels_per_foot)
    
    pass