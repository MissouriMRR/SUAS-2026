"""
Testing vision.common.crop.py
"""

import unittest
import numpy as np
import random as rand
from vision.common.crop import Image, BoundingBox, crop_image
from vision.common.bounding_box import ObjectType, Vertices


class TestTopLeft(unittest.TestCase):
    """
    Tests crop on top-left corner of image
    """

    # image
    img = np.ndarray(shape=(1080, 1920, 3))
    for row in range(img.shape[0]):
        for px in range(img.shape[1]):
            img[row, px] = [row, px, 0]

    # bounding box
    vertices: Vertices = ((0, 0), (99, 0), (99, 99), (0, 99))
    obj: ObjectType = ObjectType.STD_OBJECT
    bBox = BoundingBox(vertices, obj, None)

    # cropped image
    cropped: Image = crop_image(img, bBox)

    def test_size(self):
        """
        Test if crop returns image of correct size
        """
        self.assertEqual(self.cropped.shape, (99, 99, 3))

    def test_corner_values(self):
        """
        Test if correct portion of image was cropped by looking at corner values
        """
        self.assertTrue(all(self.cropped[0, 0] == [0, 0, 0]))
        self.assertTrue(all(self.cropped[0, self.cropped.shape[1] - 1] == [0, 98, 0]))
        self.assertTrue(
            all(
                self.cropped[self.cropped.shape[0] - 1, self.cropped.shape[1] - 1]
                == [98, 98, 0]
            )
        )
        self.assertTrue(all(self.cropped[self.cropped.shape[0] - 1, 0] == [98, 0, 0]))


class TestBottomRight(unittest.TestCase):
    """
    Tests crop on bottom-right corner of image
    """

    # image
    img = np.ndarray(shape=(1080, 1920, 3))
    for row in range(img.shape[0]):
        for px in range(img.shape[1]):
            img[row, px] = [row, px, 0]

    # bounding box
    vertices: Vertices = ((1820, 980), (1919, 980), (1919, 1079), (1820, 1079))
    obj: ObjectType = ObjectType.STD_OBJECT
    bBox = BoundingBox(vertices, obj, None)

    # cropped image
    cropped: Image = crop_image(img, bBox)

    def test_size(self):
        """
        Test if crop returns image of correct size
        """
        self.assertEqual(self.cropped.shape, (99, 99, 3))

    def test_corner_values(self):
        """
        Test if correct portion of image was cropped by looking at corner values
        """
        self.assertTrue(all(self.cropped[0, 0] == [980, 1820, 0]))
        self.assertTrue(
            all(self.cropped[0, self.cropped.shape[1] - 1] == [980, 1918, 0])
        )
        self.assertTrue(
            all(
                self.cropped[self.cropped.shape[0] - 1, self.cropped.shape[1] - 1]
                == [1078, 1918, 0]
            )
        )
        self.assertTrue(
            all(self.cropped[self.cropped.shape[0] - 1, 0] == [1078, 1820, 0])
        )


class TestCenter(unittest.TestCase):
    """
    Tests crop on center of image
    """

    # image
    img = np.ndarray(shape=(1080, 1920, 3))
    for row in range(img.shape[0]):
        for px in range(img.shape[1]):
            img[row, px] = [row, px, 0]

    # bounding box
    vertices: Vertices = ((910, 490), (1009, 490), (1009, 589), (910, 589))
    obj: ObjectType = ObjectType.STD_OBJECT
    bBox = BoundingBox(vertices, obj, None)

    # cropped image
    cropped: Image = crop_image(img, bBox)

    def test_size(self):
        """
        Test if crop returns image of correct size
        """
        self.assertEqual(self.cropped.shape, (99, 99, 3))

    def test_corner_values(self):
        """
        Test if correct portion of image was cropped by looking at corner values
        """
        self.assertTrue(all(self.cropped[0, 0] == [490, 910, 0]))
        self.assertTrue(
            all(self.cropped[0, self.cropped.shape[1] - 1] == [490, 1008, 0])
        )
        self.assertTrue(
            all(
                self.cropped[self.cropped.shape[0] - 1, self.cropped.shape[1] - 1]
                == [588, 1008, 0]
            )
        )
        self.assertTrue(
            all(self.cropped[self.cropped.shape[0] - 1, 0] == [588, 910, 0])
        )


class TestRandom(unittest.TestCase):
    """
    Tests crop of random dimensions on random location in the image
    """

    # image
    img = np.ndarray(shape=(1080, 1920, 3))
    for row in range(img.shape[0]):
        for px in range(img.shape[1]):
            img[row, px] = [row, px, 0]

    # cropped images and related data
    cropped: list[Image] = []
    cropInfo: list[dict[str:int]] = []

    # generate cropped images
    for i in range(10):
        # get random crop dimensions
        cropWidth = rand.randint(0, 1920)
        cropHeight = rand.randint(0, 1080)
        startX = rand.randint(0, 1920 - cropWidth)
        startY = rand.randint(0, 1080 - cropHeight)

        # bounding box
        vertices: Vertices = (
            (startX, startY),
            (startX + cropWidth, startY),
            (startX + cropWidth, startY + cropHeight),
            (startX, startY + cropHeight),
        )
        obj: ObjectType = ObjectType.STD_OBJECT
        bBox = BoundingBox(vertices, obj, None)

        # cropped image
        cropped.append(crop_image(img, bBox))
        cropInfo.append(
            {
                "cropWidth": cropWidth,
                "cropHeight": cropHeight,
                "startX": startX,
                "startY": startY,
            }
        )

    def test_size(self):
        """
        Test if crop returns image of correct size
        """
        for i, img in enumerate(self.cropped):
            self.assertEqual(
                img.shape,
                (self.cropInfo[i]["cropHeight"], self.cropInfo[i]["cropWidth"], 3),
            )

    def test_corner_values(self):
        """
        Test if correct portion of image was cropped by looking at corner values
        """
        for i, img in enumerate(self.cropped):
            self.assertTrue(
                all(
                    img[0, 0]
                    == [self.cropInfo[i]["startY"], self.cropInfo[i]["startX"], 0]
                )
            )
            self.assertTrue(
                all(
                    img[0, img.shape[1] - 1]
                    == [
                        self.cropInfo[i]["startY"],
                        self.cropInfo[i]["startX"] + self.cropInfo[i]["cropWidth"] - 1,
                        0,
                    ]
                )
            )
            self.assertTrue(
                all(
                    img[img.shape[0] - 1, img.shape[1] - 1]
                    == [
                        self.cropInfo[i]["startY"] + self.cropInfo[i]["cropHeight"] - 1,
                        self.cropInfo[i]["startX"] + self.cropInfo[i]["cropWidth"] - 1,
                        0,
                    ]
                )
            )
            self.assertTrue(
                all(
                    img[img.shape[0] - 1, 0]
                    == [
                        self.cropInfo[i]["startY"] + self.cropInfo[i]["cropHeight"] - 1,
                        self.cropInfo[i]["startX"],
                        0,
                    ]
                )
            )


if __name__ == "__main__":
    unittest.main()
