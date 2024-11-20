import numpy as np
import vision.common.constants as consts

from vision.deskew import camera_distances
from vision.deskew import deskew
from vision.deskew import coordinate_lengths

from vision.mapping import overlaying

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
    
    
    def add_img(self, img, camera_parameters):
        img_corner_coords = camera_distances.corner_coords(img.shape, camera_parameters)
        
        img_min_coord = np.min(img_corner_coords, axis=1)
        img_max_coord = np.max(img_corner_coords, axis=1)
        
        projected_image = deskew.deskew(
            img, 
            camera_parameters["focal_length"],
            camera_parameters["rotation_deg"],
            scale=self.pixels_per_foot
        )
        
        if self.img_count == 0:
            self.img = projected_image
            self.coord_min = img_min_coord
            self.coord_max = img_max_coord
        else:
            self.add_projected_image(projected_image, img_corner_coords)
        
        return
    
    
    def add_projected_image(self, proj_img, img_corner_coords):
        # Calculate the "top left" and "bottom right" corners of the bounding box in GPS coordinate space
        # May not actually be top left and bottom right, but it doesn't really matter - we just care about the bounds
        img_min_coord = np.min(img_corner_coords, axis=1)
        img_max_coord = np.max(img_corner_coords, axis=1)
        
        # Calculate the center of the projected image
        center_coord = (img_min_coord + img_max_coord) / 2
        
        map_center = (self.coord_max + self.coord_min) / 2
        
        # The position of the projected image relative to the map in GPS coordinate space
        coordinate_change = center_coord - map_center
        
        # Calculate the latitude and longitude lengths (in meters)
        latitude_length: float = coordinate_lengths.latitude_length(
            self.center_coord[0]
        )
        longitude_length: float = coordinate_lengths.longitude_length(
            map.center_coord[0]
        )
        
        relative_coord_ft = np.array(
            [coordinate_change[0] * latitude_length, coordinate_change[1] * longitude_length]
        )
        
        pixel_offset = np.round(relative_coord_ft * self.pixels_per_foot)
        
        updated_map_img = overlaying.offset_overlay(proj_img, self.img, pixel_offset)
        
        updated_map_min = np.minimum(self.coord_min, img_min_coord)
        updated_map_max = np.maximum(self.coord_max, img_max_coord)
        
        self.coord_min = updated_map_min
        self.coord_max = updated_map_max
        self.img = updated_map_img
        
        return