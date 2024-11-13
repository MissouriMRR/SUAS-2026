import numpy as np
import cv2

import vision.common.constants as consts

from vision.mapping.map import Map
from vision.deskew import coordinate_lengths
from vision.deskew.deskew import deskew
from vision.deskew.camera_distances import corner_coords

from overlaying import add_to_img


def add_img(map: Map, img: consts.Image, camera_parameters: consts.CameraParameters):
    map.img_count += 1
    
    if map.img_count == 1:
        return add_first_img(map, img, camera_parameters)
    else:
        return add_nth_img(map, img, camera_parameters)


def add_first_img(map: Map, img: consts.Image, camera_parameters: consts.CameraParameters):
    """
    Adds an image to the map
    Only for the first image
    """
    img_corner_coords = corner_coords(img.shape, camera_parameters)
    
    img_min_coord = np.min(img_corner_coords, axis=1)
    img_max_coord = np.max(img_corner_coords, axis=1)
    
    projected_image = deskew(
        img, 
        camera_parameters["focal_length"],
        camera_parameters["rotation_deg"],
        scale=map.pixels_per_foot
    )
    
    map.img = projected_image
    map.coord_min = img_min_coord
    map.coord_max = img_max_coord
    
    return map
    


def add_nth_img(map: Map, img: consts.Image, camera_parameters: consts.CameraParameters):
    """
    Adds an image to the map
    Only for when the map has one image or more
    """
    
    img_corner_coords = corner_coords(img.shape, camera_parameters)
    
    projected_image = deskew(
        img, 
        camera_parameters["focal_length"],
        camera_parameters["rotation_deg"],
        scale=map.pixels_per_foot
    )
    
    map = add_projected_image(map, projected_image, img_corner_coords)
    
    return map


def add_projected_image(map: Map, proj_img: consts.ImageWAlpha, img_corner_coords):
    """
    Adds an image to a map
    """
    
    # Calculate the "top left" and "bottom right" corners of the bounding box in GPS coordinate space
    # May not actually be top left and bottom right, but it doesn't really matter - we just care about the bounds
    img_min_coord = np.min(img_corner_coords, axis=1)
    img_max_coord = np.max(img_corner_coords, axis=1)
    
    # Calculate the center of the projected image
    center_coord = (img_min_coord + img_max_coord) / 2
    
    map_center = (map.coord_max + map.coord_min) / 2
    
    # The position of the projected image relative to the map in GPS coordinate space
    coordinate_change = center_coord - map_center
    
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
    
    updated_map_img = add_to_img(proj_img, map.img, pixel_offset)
    
    updated_map_min = np.minimum(map.coord_min, img_min_coord)
    updated_map_max = np.maximum(map.coord_max, img_max_coord)
    
    map.coord_min = updated_map_min
    map.coord_max = updated_map_max
    map.img = updated_map_img
    
    return map