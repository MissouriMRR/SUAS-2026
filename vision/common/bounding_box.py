"""
Bounding box objects represent an area in an image and
are used to convey information between flight and vision processes.
"""

from typing import TypeAlias

import numpy as np

# A set of 4 coordinates that distinguish a region of an image.
# The order of the coordinates is (top-left, top-right, bottom-right, bottom-left).
Vertices: TypeAlias = tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]


def tlwh_to_vertices(tl_x: int, tl_y: int, width: int, height: int) -> Vertices:
    """
    Gets the vertices of a bounding box from a coordinate, width, and height.

    Parameters
    ----------
    tl_x : int
        the top-left x coordinate of the bounding box
    tl_y : int
        the top-left y coordinate of the bounding box
    width : int
        the width of the bounding box
    height : int
        the height of the bounding box

    Returns
    -------
    vertices : Vertices
        Denotes the 4 coordinates representing a box in an image.
        Vertices is a tuple of 4 coordinates.
        Each coordinate consists of a tuple 2 integers.
        Order is (top-left, top-right, bottom-right, bottom-left).
    """

    tl_coord: tuple[int, int] = (tl_x, tl_y)  # top left
    tr_coord: tuple[int, int] = (tl_x + width, tl_y)  # top right
    br_coord: tuple[int, int] = (tl_x + width, tl_y + height)  # bottom right
    bl_coord: tuple[int, int] = (tl_x, tl_y + height)  # bottom left

    return (tl_coord, tr_coord, br_coord, bl_coord)


class BoundingBox:
    """
    A set of 4 coordinates that distinguish a region of an image.
    The top-left coordinate, width, and height are used to create the
    BoundingBox.

    Parameters
    ----------
    top_left : tuple[int, int]
        The top-left coordinate of the BoundingBox.
    width : int
        The width of the BoundingBox.
    height : int
        The height of the BoundingBox.
    """

    def __init__(self, top_left: tuple[int, int], width: int, height: int) -> None:
        self.top_left: tuple[int, int] = top_left
        self.width: int = width
        self.height: int = height

        # Calculate the 4 vertices of the bounding box
        self.vertices: Vertices = tlwh_to_vertices(top_left[0], top_left[1], width, height)

    def get_x_vals(self) -> list[int]:  # Used by this class only
        """
        Gets the x values of the 4 coordinates.

        Returns
        -------
        x_vals : list[int]
            The 4 x values of the vertices.
        """

        x_vals: list[int] = [vert[0] for vert in self.vertices]
        return x_vals

    def get_y_vals(self) -> list[int]:  # Used by this class only
        """
        Gets the y values of the 4 coordinates.

        Returns
        -------
        y_vals : list[int]
            The 4 y values of the vertices.
        """

        y_vals: list[int] = [vert[1] for vert in self.vertices]
        return y_vals

    def get_x_extremes(self) -> tuple[int, int]:  # Currently Used
        """
        Gets the minimum and maximum x values of the BoundingBox

        Returns
        -------
        min_x, max_x : tuple[int, int]
            The minimum and maximum x values.
        """

        x_vals: list[int] = self.get_x_vals()
        min_x: int = np.amin(x_vals)
        max_x: int = np.amax(x_vals)

        return min_x, max_x

    def get_y_extremes(self) -> tuple[int, int]:  # Currently Used
        """
        Gets the minimum and maximum y values of the BoundingBox

        Returns
        -------
        min_y, max_y : tuple[int, int]
            The minimum and maximum y values.
        """

        y_vals: list[int] = self.get_y_vals()
        min_y: int = np.amin(y_vals)
        max_y: int = np.amax(y_vals)

        return min_y, max_y

    def get_x_avg(self) -> int:  # Used by this class only
        """
        Gets the average x coordinate of the bounding box.

        Returns
        -------
        average : int
            the average of the 4 coordinates' x-values
        """

        return int(np.mean(self.get_x_vals()))

    def get_y_avg(self) -> int:  # Used by this class only
        """
        Gets the average y coordinate of the bounding box.

        Returns
        -------
        average : int
            the average of the 4 coordinates' y-values
        """

        return int(np.mean(self.get_y_vals()))

    def get_center_coord(self) -> tuple[int, int]:  # Currently Used
        """
        Gets the coordinate of the center of the BoundingBox

        Returns
        -------
        center_pt : tuple[int, int]
            the coordinate point at the center of the bounding box
        """

        return (self.get_x_avg(), self.get_y_avg())

    def get_rotation_angle(self) -> float:  # Currently Not Used
        """
        Calculates the angle of rotation of the BoundingBox
        based on the top left and right coordinates.

        Returns
        -------
        angle : float
            The angle of rotation of the BoundingBox in degrees.
        """

        # Get the top left and right coordinates
        tl_x: int = self.vertices[0][0]  # Top left x
        tl_y: int = self.vertices[0][1]  # Top left y
        tr_x: int = self.vertices[1][0]  # Top right x
        tr_y: int = self.vertices[1][1]  # Top right y

        angle: float = 0
        if tr_x - tl_x == 0:  # Prevent division by 0
            angle = 90.0 if (tr_y - tl_y > 0) else -90.0
        else:
            angle = np.rad2deg(np.arctan((tr_y - tl_y) / (tr_x - tl_x)))

        return angle

    def get_width(self) -> int:  # Currently Used
        """
        Get the width of the BoundingBox.

        Returns
        -------
        width: int
            the width of the BoundingBox based on max and min x values.
        """

        # Get the min and max x values, then calculate the width by subtracting
        # the min from the max
        min_x: int
        max_x: int
        min_x, max_x = self.get_x_extremes()
        width: int = max_x - min_x

        return width

    def get_height(self) -> int:  # Currently Used
        """
        Get the height of the BoundingBox.

        Returns
        -------
        height: int
            the height of the BoundingBox based on max and min x values.
        """

        # Get the min and max y values, then calculate the height by
        # subtracting the min from the max
        min_y: int
        max_y: int
        min_y, max_y = self.get_y_extremes()
        height: int = max_y - min_y

        return height

    def get_width_height(self) -> tuple[int, int]:  # Currently Used
        """
        Gets the width and height of the BoundingBox.

        Returns
        -------
        (width, height) : tuple[int, int]
            the width and height of the bounding box
        """

        return self.get_width(), self.get_height()

    def get_tlwh(self) -> tuple[int, int, int, int]:  # Currently Not Used
        """
        Gets the BoundingBox formatted with top left coordinate, width, and height.

        Returns
        -------
        tlwh_coord : tuple[int, int, int, int]
            the bounding box in top left, width, height format

            tl_x : int
                the top-left x coordinate of the bounding box
            tl_y : int
                the top-left y coordinate of the bounding box
            width : int
                the width of the bounding box
            height : int
                the height of the bounding box
        """

        tl_x: int = self.vertices[0][0]
        tl_y: int = self.vertices[0][1]
        width: int = self.get_width()
        height: int = self.get_height()

        return tl_x, tl_y, width, height


# Driver for testing functionality of BoundingBox object
if __name__ == "__main__":

    test_top_left: tuple[int, int] = (0, 0)
    test_width: int = 39
    test_height: int = 50

    # constructor
    object_bounds = BoundingBox(top_left=test_top_left, width=test_width, height=test_height)

    # width, height
    print("Width:", object_bounds.get_width())
    print("Height:", object_bounds.get_height())
    print("Width and Height:", object_bounds.get_width_height())

    # values, extremes, average
    print()
    print("X values:", object_bounds.get_x_vals())
    print("Y values:", object_bounds.get_y_vals())
    print("X extremes:", object_bounds.get_x_extremes())
    print("Y extremes:", object_bounds.get_y_extremes())
    print("X average:", object_bounds.get_x_avg())
    print("Y average:", object_bounds.get_y_avg())

    # center and rotation
    print()
    print("Center coordinate:", object_bounds.get_center_coord())
    print("Rotation angle:", object_bounds.get_rotation_angle())
