import numpy as np
import vision.common.constants as consts
from PIL import Image
import matplotlib.pyplot as plt

from vision.deskew import camera_distances
from vision.deskew import deskew
from vision.deskew import coordinate_lengths
from vision.deskew import vector_utils

from vision.mapping import overlaying

import cv2
import math

class Map:
    """
    An object representing a map
    Stores the map image itself, the scale of the map, and the bounds on the coordinates of the map
    
    Parameters
    ----------
    pixels_per_foot: float
        The scale of the map in pixel *edges* per foot - for example, a value of 2 means that one square foot will
        take up a square of 4 pixels
    feather_width: float
        Roughly the width of the fade between adjacent images
    img: consts.MapImage
        The map image. Alpha channel will be used to store resolution data
    coord_min: consts.Point
        The minimum lat, lon of the map
    coord_max: consts.Point
        The maximum lat, lon of the map
    """
    def __init__(self, pixels_per_foot, feather_width = 10) -> None:
        self.pixels_per_foot: float = pixels_per_foot
        self.feather_width = feather_width
        
        self.img: consts.MapImage = None
        self.img_count = 0
        self.coord_min = None
        self.coord_max = None
    
    
    def prepare_image(self, img, camera_parameters):

        blank_channel = np.zeros((img.shape[0], img.shape[1]))
        blank_channel=blank_channel+1

        image_wdepth: consts.MapImage = np.dstack((img, blank_channel))

        projected_image_wdepth, _ = deskew.deskew(
            image_wdepth,
            camera_parameters["rotation_deg"],
            scale=self.pixels_per_foot * camera_parameters["altitude_f"] * consts.FEET_PER_METER
        )
        
        projected_image_wdepth = self.fill_distances(projected_image_wdepth, camera_parameters)

        return projected_image_wdepth
    

    def fill_distances(self, img: consts.Image , camera_parameters: consts.CameraParameters) -> consts.Image:
        distance = lambda x, y : np.linalg.norm(vector_utils.pixel_intersect(
                (x, y),
                img.shape,
                camera_parameters["rotation_deg"],
                height=self.pixels_per_foot * camera_parameters["altitude_f"]
            ))
        height: int = img.shape[0] -1
        width: int = img.shape[1] -1
        center_height: int = img.shape[0] //2
        center_width: int = img.shape[1] //2
        # has the pixel coordinates of the start and end indexes of each fourth of the image

        #calculating distance for each pixel takes way too long, so it will be simplified by making the points up to the center on top,bottom, and center linearly scaled
        # then filling the rest linearly

        #initialize corners and center distances
        img[0,0,3] = distance(0,0)
        img[0,width,3] = distance(width,0)
        img[height,width,3]= distance(width, height)
        img[height, 0] = distance(0, width)
        
        img[center_height, center_width,3] = distance(center_width, center_height)
        img[0, center_width,3] = distance(center_width, 0)
        img[center_height, 0] = distance(0, center_height)
        img[height, center_width,3] = distance(center_width, height)
        img[center_height, width,3] = distance(width, center_height)

        # start by filling top, bottom, and center
        for i in [0,center_height,height]:
            for pixel in range(1, width):
                if pixel < center_width:
                    img[i,pixel,3] = int(img[i,0,3]* pixel/(center_width) + img[i,center_width,3] * (1-pixel/(center_width)))
                if pixel > center_width:
                    #percentages are close enough not perfect
                    img[i,pixel,3] = int(img[i,center_width,3]* (pixel-center_width)/(center_width) + img[i,width,3] * (1-(pixel-center_width)/(center_width)))

        # fill the rest
        for column in range(0, img.shape[1]):
            for pixel in range(1,height):
                if pixel < center_height:
                    img[pixel,column,3] = int(img[0,column,3]* pixel/(center_height) + img[center_height,column,3] * (1-pixel/(center_height)))
                if pixel > center_height:
                    #percentages are close enough not perfect
                    img[pixel,column,3] = int(img[center_height,column,3]* (pixel-center_height)/(center_height) + img[height,column,3] * (1-(pixel-center_height)/(center_height)))
        
        return img
        

    
    def add_img(self, img, camera_parameters):
        
        img_corner_coords = camera_distances.corner_coords(img.shape, camera_parameters)
        
        
        projected_image_wdepth = self.prepare_image(img, camera_parameters)


        #projected_image_wdepth = np.delete(projected_image_wdepth, 3, 2)



        if self.img_count == 0:
            
            img_min_coord = np.min(img_corner_coords, axis=0)
            img_max_coord = np.max(img_corner_coords, axis=0)
            self.img = projected_image_wdepth
            self.coord_min = img_min_coord
            self.coord_max = img_max_coord
        else:
            self.add_projected_image(projected_image_wdepth, img_corner_coords)



        self.img_count += 1
        
        
        return
    
    
    def add_projected_image(self, proj_img, img_corner_coords):
        # Calculate the "top left" and "bottom right" corners of the bounding box in GPS coordinate space
        # May not actually be top left and bottom right, but it doesn't really matter - we just care about the bounds
        
        img_min_coord = np.min(img_corner_coords, axis=0)
        img_max_coord = np.max(img_corner_coords, axis=0)
        
        # Calculate the center of the projected image
        center_coord = (img_min_coord + img_max_coord) / 2
        
        map_center = (self.coord_max + self.coord_min) / 2
        
        # The position of the projected image relative to the map in GPS coordinate space
        coordinate_change = center_coord - map_center
        
        # Calculate the latitude and longitude lengths (in meters)
        latitude_length: float = coordinate_lengths.latitude_length(
            center_coord[0]
        )
        longitude_length: float = coordinate_lengths.longitude_length(
            center_coord[1]
        )
        
        relative_coord_ft = np.array(
            [coordinate_change[0] * latitude_length, coordinate_change[1] * longitude_length]
        )
        
        pixel_offset = np.round(relative_coord_ft * self.pixels_per_foot)
        
        updated_map_img = overlaying.offset_overlay(proj_img, self.img, pixel_offset, self.feather_width)
        
        self.coord_min = np.minimum(self.coord_min, img_min_coord)
        self.coord_max = np.maximum(self.coord_min, img_min_coord)
        self.img = updated_map_img
        
        return