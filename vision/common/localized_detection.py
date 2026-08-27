"""
Contains the LocalizedDetection class, which extends ObjectDetection with the
latitude/longitude of the detected object, as computed by the deskew calculations in vision/deskew.
"""

from dataclasses import dataclass

from vision.object_detection import ObjectDetection


@dataclass()
class LocalizedDetection(ObjectDetection):
    """
    An ObjectDetection with the ground latitude/longitude of the object.

    Attributes
    ----------
    latitude : float
        The latitude in degrees of the object.
    longitude : float
        The longitude in degrees of the object.
    """

    latitude: float
    longitude: float
