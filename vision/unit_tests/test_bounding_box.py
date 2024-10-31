"""
Testing vision.common.bounding_box.py

Each test is ran 100 times with randomized values for each test

NOTE: Some code here could have been simplified. However, there is an issue
with pylint itself where it does not recognize type aliases or types themselves
are correct when being indexed in a list or tuple. Instead of passing the test,
pylint throws an "invalid-sequence-index" error even when no error is present.

See issue here:
    https://github.com/pylint-dev/pylint/issues/4083
"""

import unittest
from typing import TypeAlias, Any
import numpy as np

import vision.common.bounding_box as bb


Test_Vertices_Type_Alias: TypeAlias = tuple[
    tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]
]
"""
Test_Vertices_Type_Alias is a type alias for a tuple of 4 tuples, each
containing 2 integers

It is the same as the Vertices type alias in vision.common.bounding_box.py

This is only used to test the type alias in the test_type_aliases function
"""


class TestTypeAliases(unittest.TestCase):
    """
    Unit Test for Type Aliases
    """

    def test_type_aliases(self) -> None:
        """
        Asserts that the type aliases are correct
        """

        self.assertEqual(
            bb.Vertices,
            Test_Vertices_Type_Alias,
            msg="The Vertices type alias did not match the expected type",
        )


class TestTLWHToVerts(unittest.TestCase):
    """
    Unit Test for tlwh_to_vertices
    """

    def test_coordinate_types(self) -> None:
        """
        Asserts that the function returns a list of 4 tuples, each containing two floats
        """

        test_coordinates: bb.Vertices = bb.tlwh_to_vertices(20, 20, 10, 10)

        # Assert that the return value matches bb.Vertices
        self.assertEqual(type(test_coordinates), tuple)
        for coord in test_coordinates:
            self.assertEqual(type(coord), tuple, "The coordinate is not a tuple")
            self.assertEqual(len(coord), 2, "The coordinate does not contain 2 values")
            self.assertEqual(type(coord[0]), int, "The first value of the coordinate is not an int")
            self.assertEqual(
                type(coord[1]), int, "The second value of the coordinate is not an int"
            )

    def test_coordinate_values(self) -> None:
        """
        Asserts that the function performs the correct conversion using randomized values
        """

        # Template String for the fail message
        template_fail_msg: str = "Test {}/100 Failed: {} did not return the correct value"

        for i in range(100):

            # Randomize the values
            test_x = np.random.randint(0, 1000)
            test_y = np.random.randint(0, 1000)
            test_w = np.random.randint(0, 1000)
            test_h = np.random.randint(0, 1000)

            # Run the test
            test_coordinates: bb.Vertices = bb.tlwh_to_vertices(test_x, test_y, test_w, test_h)

            # Create a fail message on the event of a failed test
            coordinate_values_fail_msg: str = template_fail_msg.format(i + 1, "tlwh_to_vertices")

            # Assert that the return value is correct
            self.assertEqual(
                test_coordinates,
                (
                    (test_x, test_y),
                    (test_x + test_w, test_y),
                    (test_x + test_w, test_y + test_h),
                    (test_x, test_y + test_h),
                ),
                msg=coordinate_values_fail_msg,
            )


class TestObjectType(unittest.TestCase):
    """
    Unit Test for ObjectType
    """

    def test_object_type(self) -> None:
        """
        Asserts that the function returns a string
        """

        self.assertEqual(type(bb.ObjectType.STD_OBJECT.value), str, "STD_OBJECT is not a string")
        self.assertEqual(type(bb.ObjectType.EMG_OBJECT.value), str, "EMG_OBJECT is not a string")
        self.assertEqual(type(bb.ObjectType.TEXT.value), str, "TEXT is not a string")

    def test_object_values(self) -> None:
        """
        Asserts that the function returns the correct object type for each input
        """

        self.assertEqual(
            bb.ObjectType.STD_OBJECT.value, "std_object", "STD_OBJECT value is incorrect"
        )
        self.assertEqual(
            bb.ObjectType.EMG_OBJECT.value, "emg_object", "EMG_OBJECT value is incorrect"
        )
        self.assertEqual(bb.ObjectType.TEXT.value, "text", "TEXT value is incorrect")


class TestBoundingBox(unittest.TestCase):
    """
    Unit Test for BoundingBox
    """

    def gen_new_test_set(self) -> dict[str, Any]:
        """
        Generates a new BoundingBox object with randomized values and returns
        the BoundingBox object, the vertices, the object type, and the
        attributes in a list in that order.

        This is used for testing later on as we need to know the values before
        so we can compare them with what BoundingBox returns
        """

        # Create a new test set to return in the form of a dictionary
        # where each key corresponds to a value in the BoundingBox object
        # EX: Vertices would be the vertices of the BoundingBox object
        new_test_set: dict[str, Any] = {}

        # Randomize a set of vertices, the order of which isn't important for
        # this test as we are mainly testing the math itself
        test_vertices: bb.Vertices = (
            (np.random.randint(0, 1000), np.random.randint(0, 1000)),
            (np.random.randint(0, 1000), np.random.randint(0, 1000)),
            (np.random.randint(0, 1000), np.random.randint(0, 1000)),
            (np.random.randint(0, 1000), np.random.randint(0, 1000)),
        )

        ## Randomize the Object Type
        # Original code documented below, changed due to pylint issue

        # # Create a list to randomly select from
        # test_obj_type_list: list[bb.ObjectType] = [
        #     bb.ObjectType.STD_OBJECT,
        #     bb.ObjectType.EMG_OBJECT,
        #     bb.ObjectType.TEXT,
        # ]
        # test_obj_type: bb.ObjectType = test_obj_type_list[np.random.randint(0, 3)]

        # This is really roundabout and redundant, however, this passes the
        # pylint test
        test_obj_type_index: int = np.random.randint(0, 3)
        test_obj_type: bb.ObjectType
        if test_obj_type_index == 0:
            test_obj_type = bb.ObjectType.STD_OBJECT
        elif test_obj_type_index == 1:
            test_obj_type = bb.ObjectType.EMG_OBJECT
        else:
            test_obj_type = bb.ObjectType.TEXT

        # A list of strings to randomly select from for the attributes
        test_attributes_str: list[str] = [
            "Test",
            "Another_Test",
            "lowercase_test",
            "UPPERCASE_TEST",
            "1234567890",
        ]

        # Generate some random attributes
        test_attributes: dict[str, Any] = {
            "attrubute_int": np.random.randint(0, 1000),
            "attribute_str": np.random.choice(test_attributes_str),
        }

        # Create a test Bounding Box object
        test_bounding_box: bb.BoundingBox = bb.BoundingBox(
            vertices=test_vertices, obj_type=test_obj_type, attributes=test_attributes
        )

        # Put the randomized values into the dictionary
        new_test_set["Bounding_Box"] = test_bounding_box
        new_test_set["Vertices"] = test_vertices
        new_test_set["Object_Type"] = test_obj_type
        new_test_set["Attributes"] = test_attributes

        return new_test_set

    def gen_new_test_vertices(self) -> Test_Vertices_Type_Alias:
        """
        Generates a new set of vertices with randomized values and returns them
        """

        return (
            (np.random.randint(0, 1000), np.random.randint(0, 1000)),
            (np.random.randint(0, 1000), np.random.randint(0, 1000)),
            (np.random.randint(0, 1000), np.random.randint(0, 1000)),
            (np.random.randint(0, 1000), np.random.randint(0, 1000)),
        )

    def gen_new_test_obj_type(self) -> bb.ObjectType:
        """
        Generates a new object type with randomized values and returns it
        """

        ## Randomize the Object Type
        # Original code documented below, changed due to pylint issue

        # # Create a list to randomly select from
        # test_obj_type_list: list[bb.ObjectType] = [
        #     bb.ObjectType.STD_OBJECT,
        #     bb.ObjectType.EMG_OBJECT,
        #     bb.ObjectType.TEXT,
        # ]
        # test_obj_type: bb.ObjectType = test_obj_type_list[np.random.randint(0, 3)]

        # This is really roundabout and redundant, however, this passes the
        # pylint test
        test_obj_type_index: int = np.random.randint(0, 3)
        test_obj_type: bb.ObjectType
        if test_obj_type_index == 0:
            test_obj_type = bb.ObjectType.STD_OBJECT
        elif test_obj_type_index == 1:
            test_obj_type = bb.ObjectType.EMG_OBJECT
        else:
            test_obj_type = bb.ObjectType.TEXT

        return test_obj_type

    def gen_new_test_attributes(self) -> dict[Any, Any]:
        """
        Generates a new set of attributes with randomized values and returns them
        """

        # A list of strings to randomly select from for the attributes
        test_attributes_str: list[str] = [
            "Test",
            "Another_Test",
            "lowercase_test",
            "UPPERCASE_TEST",
            "1234567890",
        ]

        return {
            "attrubute_int": np.random.randint(0, 1000),
            "attribute_str": np.random.choice(test_attributes_str),
        }

    def test_bounding_box_repr(self) -> None:
        """
        Asserts that the __repr__ method works as intended
        """

        # Template String for the fail message
        template_fail_msg: str = "Test {}/100 Failed: {} did not return the correct value"

        # Run this test 100 times with randomized values
        for i in range(100):

            ## Create the required values for the test
            # New test set
            new_test_set: dict[str, Any] = self.gen_new_test_set()
            # Separate the values for easier access
            test_bounding_box: bb.BoundingBox = new_test_set["Bounding_Box"]
            test_vertices: Test_Vertices_Type_Alias = new_test_set["Vertices"]
            test_obj_type: bb.ObjectType = new_test_set["Object_Type"]

            # Create a fail message on the event of a failed test
            _repr_fail_msg: str = template_fail_msg.format(
                i + 1, "BoundingBox.__repr__ did not return a string"
            )

            # Assert that the __repr__ method returns a string type
            self.assertEqual(type(repr(test_bounding_box)), str, msg=_repr_fail_msg)

            # Create a fail message on the event of a failed test
            fail_msg: str = (
                f"Test {i}/100 Failed: BoundingBox.__repr__ returned an incorrect string"
            )

            # Assert that the __repr__ method returns the correct string
            self.assertEqual(
                repr(test_bounding_box),
                f"BoundingBox[{id(test_bounding_box)}, {test_obj_type}]: {test_vertices}",
                msg=fail_msg,
            )

            print(test_obj_type)

    def test_properties(self) -> None:
        """
        This test is used to test all properties of the BoundingBox class.

        These properties are:
        - vertices
        - obj_type
        - attributes

        Each of these have a getter and setter method; both of which are tested
        """

        # Template String for the fail message
        template_fail_msg: str = "Test {}/100 Failed: {} did not return the correct value"

        for i in range(100):

            ## Create the required values for the test
            # New test set
            new_test_set: dict[str, Any] = self.gen_new_test_set()
            # Separate the values for easier access
            test_bounding_box: bb.BoundingBox = new_test_set["Bounding_Box"]
            test_vertices: Test_Vertices_Type_Alias = new_test_set["Vertices"]
            test_obj_type: bb.ObjectType = new_test_set["Object_Type"]
            test_attributes: dict[Any, Any] = new_test_set["Attributes"]

            ## Test the vertices property
            # Create a fail message on the event of a failed test
            get_vertices_fail_msg: str = template_fail_msg.format(i + 1, "BoundingBox.vertices")

            # Assert that the getter method returns the correct value
            self.assertEqual(test_bounding_box.vertices, test_vertices, msg=get_vertices_fail_msg)

            # Randomize a new set of vertices
            new_vertices: Test_Vertices_Type_Alias = self.gen_new_test_vertices()

            # Set the new vertices
            test_bounding_box.vertices = new_vertices

            # Create a fail message on the event of a failed test
            set_vertices_fail_msg: str = template_fail_msg.format(i + 1, "BoundingBox.vertices")

            # Assert that the setter method sets the correct value
            self.assertEqual(test_bounding_box.vertices, new_vertices, msg=set_vertices_fail_msg)

            ## Test the obj_type property
            # Create a fail message on the event of a failed test
            get_obj_type_fail_msg: str = template_fail_msg.format(i + 1, "BoundingBox.obj_type")

            # Assert that the getter method returns the correct value
            self.assertEqual(test_bounding_box.obj_type, test_obj_type, msg=get_obj_type_fail_msg)

            # Randomize a new object type
            new_obj_type: bb.ObjectType = self.gen_new_test_obj_type()

            # Set the new object type
            test_bounding_box.obj_type = new_obj_type

            # Create a fail message on the event of a failed test
            set_obj_type_fail_msg: str = template_fail_msg.format(i + 1, "BoundingBox.obj_type")

            # Assert that the setter method sets the correct value
            self.assertEqual(test_bounding_box.obj_type, new_obj_type, msg=set_obj_type_fail_msg)

            ## Test the attributes property
            # Create a fail message on the event of a failed test
            get_attributes_fail_msg: str = template_fail_msg.format(i + 1, "BoundingBox.attributes")

            # Assert that the getter method returns the correct value
            self.assertEqual(
                test_bounding_box.attributes, test_attributes, msg=get_attributes_fail_msg
            )

            # Randomize a new set of attributes
            new_attributes: dict[Any, Any] = self.gen_new_test_attributes()

            # Set the new attributes
            test_bounding_box.attributes = new_attributes

            # Create a fail message on the event of a failed test
            set_attributes_fail_msg: str = template_fail_msg.format(i + 1, "BoundingBox.attributes")

            # Assert that the setter method sets the correct value
            self.assertEqual(
                test_bounding_box.attributes, new_attributes, msg=set_attributes_fail_msg
            )

    def test_attribute_sg(self) -> None:
        """
        Asserts that the set_attribute and get_attribute set and return the
        correct values
        """

        # Template String for the fail message
        template_fail_msg: str = "Test {}/100 Failed: {} did not return the correct value"

        for i in range(100):

            ## Create the required values for the test
            # New test set
            new_test_set: dict[str, Any] = self.gen_new_test_set()
            # Separate the values for easier access
            test_bounding_box: bb.BoundingBox = new_test_set["Bounding_Box"]
            test_attributes: dict[Any, Any] = new_test_set["Attributes"]

            ## Test the get_attribute method
            # Randomly select a key from the attributes
            test_key: str = np.random.choice(list(test_attributes.keys()))

            # Create a fail message on the event of a failed test
            get_attribute_fail_msg: str = template_fail_msg.format(
                i + 1, "BoundingBox.get_attribute"
            )

            # Assert that the get_attribute method returns the correct value
            self.assertEqual(
                test_bounding_box.get_attribute(test_key),
                test_attributes[test_key],
                msg=get_attribute_fail_msg,
            )

            ## Test the set_attribute method
            # Randomize a new value for the key
            new_value: int = np.random.randint(0, 1000)

            # Set the new value
            test_bounding_box.set_attribute(test_key, new_value)

            # Create a fail message on the event of a failed test
            set_attribute_fail_msg: str = (
                f"Test {i+1}/100 Failed: BoundingBox.set_attribute did not set the correct value"
            )

            # Assert that the set_attribute method sets the correct value
            self.assertEqual(
                test_bounding_box.get_attribute(test_key), new_value, msg=set_attribute_fail_msg
            )

    def test_x_methods(self) -> None:
        """
        Asserts that the following methods return the correct values:
        - get_x_values
        - get_x_extremes
        - get_x_avg
        """

        # Template String for the fail message
        template_fail_msg: str = "Test {}/100 Failed: {} did not return the correct value"

        for i in range(100):

            ## Create the required values for the test
            # New test set
            new_test_set: dict[str, Any] = self.gen_new_test_set()
            # Separate the values for easier access
            test_bounding_box: bb.BoundingBox = new_test_set["Bounding_Box"]
            test_vertices: Test_Vertices_Type_Alias = new_test_set["Vertices"]

            ## Test the get_x_values method
            # Create a fail message on the event of a failed test
            get_x_vals_fail_msg: str = template_fail_msg.format(i + 1, "BoundingBox.get_x_values")

            # Assert that the get_x_values method returns the correct value
            self.assertEqual(
                test_bounding_box.get_x_vals(),
                [vert[0] for vert in test_vertices],
                msg=get_x_vals_fail_msg,
            )

            ## Test the get_x_extremes method
            # Get a list of the x values
            test_x_values = test_bounding_box.get_x_vals()  # Already tested

            # Create a fail message on the event of a failed test
            get_x_extremes_fail_msg: str = template_fail_msg.format(
                i + 1, "BoundingBox.get_x_extremes"
            )

            # Assert that the get_x_extremes method returns the correct value
            self.assertEqual(
                test_bounding_box.get_x_extremes(),
                (min(test_x_values), max(test_x_values)),
                msg=get_x_extremes_fail_msg,
            )

            ## Test the get_x_avg method
            # Get the average of the x values
            test_x_avg = int(sum(test_x_values) / len(test_x_values))

            # Create a fail message on the event of a failed test
            get_x_avg_fail_msg: str = template_fail_msg.format(i + 1, "BoundingBox.get_x_avg")

            # Assert that the get_x_avg method returns the correct value
            self.assertEqual(test_bounding_box.get_x_avg(), test_x_avg, msg=get_x_avg_fail_msg)

    def test_y_methods(self) -> None:
        """
        Asserts that the following methods return the correct values:
        - get_y_values
        - get_y_extremes
        - get_y_avg
        """

        # Template String
        template_fail_msg: str = "Test {}/100 Failed: {} did not return the correct value"

        for i in range(100):

            ## Create the required values for the test
            # New test set
            new_test_set: dict[str, Any] = self.gen_new_test_set()
            # Separate the values for easier access
            test_bounding_box: bb.BoundingBox = new_test_set["Bounding_Box"]
            test_vertices: Test_Vertices_Type_Alias = new_test_set["Vertices"]

            ## Test the get_y_values method
            # Create a fail message on the event of a failed test
            get_y_values_fail_msg: str = template_fail_msg.format(i + 1, "BoundingBox.get_y_values")

            # Assert that the get_y_values method returns the correct value
            self.assertEqual(
                test_bounding_box.get_y_vals(),
                [vert[1] for vert in test_vertices],
                msg=get_y_values_fail_msg,
            )

            ## Test the get_y_extremes method
            # Get a list of the y values
            test_y_values = test_bounding_box.get_y_vals()  # Already tested

            # Create a fail message on the event of a failed test
            get_y_extremes_fail_msg: str = template_fail_msg.format(
                i + 1, "BoundingBox.get_y_extremes"
            )

            # Assert that the get_y_extremes method returns the correct value
            self.assertEqual(
                test_bounding_box.get_y_extremes(),
                (min(test_y_values), max(test_y_values)),
                msg=get_y_extremes_fail_msg,
            )

            ## Test the get_y_avg method
            # Get the average of the y values
            test_y_avg = int(sum(test_y_values) / len(test_y_values))

            # Create a fail message on the event of a failed test
            fail_msg: str = template_fail_msg.format(i + 1, "BoundingBox.get_y_avg")

            # Assert that the get_y_avg method returns the correct value
            self.assertEqual(test_bounding_box.get_y_avg(), test_y_avg, msg=fail_msg)

    def test_get_center_coord(self) -> None:
        """
        Asserts that the get_center_coord method returns the correct value
        """

        # Template String
        template_fail_msg: str = "Test {}/100 Failed: {} did not return the correct value"

        for i in range(100):

            ## Create the required values for the test
            # New test set
            new_test_set: dict[str, Any] = self.gen_new_test_set()
            # Separate the values for easier access
            test_bounding_box: bb.BoundingBox = new_test_set["Bounding_Box"]

            ## Test the get_center_coord method
            # Get the average of the x and y values
            test_x_avg = test_bounding_box.get_x_avg()  # Already tested
            test_y_avg = test_bounding_box.get_y_avg()  # Already tested

            # Create a fail message on the event of a failed test
            fail_msg: str = template_fail_msg.format(i + 1, "BoundingBox.get_center_coord")

            # Assert that the get_center_coord method returns the correct value
            self.assertEqual(
                test_bounding_box.get_center_coord(), (test_x_avg, test_y_avg), msg=fail_msg
            )

    def test_get_rotation_angle(self) -> None:
        """
        Asserts that the get_rotation_angle method returns the correct value
        """

        # Template String
        template_fail_msg: str = "Test {}/100 Failed: {} did not return the correct value"

        for i in range(100):

            ## Create the required values for the test
            # New test set
            new_test_set: dict[str, Any] = self.gen_new_test_set()
            # Separate the values for easier access
            test_bounding_box: bb.BoundingBox = new_test_set["Bounding_Box"]
            test_vertices: Test_Vertices_Type_Alias = new_test_set["Vertices"]

            ## Calculate the rotation angle
            # Get the x and y values for the top-left and top-right vertices
            tl_x: int = test_vertices[0][0]
            tl_y: int = test_vertices[0][1]
            tr_x: int = test_vertices[1][0]
            tr_y: int = test_vertices[1][1]

            # Calculate the angle
            if tr_x - tl_x == 0:  # This is to prevent division by zero
                angle = 90.0 if (tr_y - tl_y > 0) else -90.0
            else:
                angle = np.rad2deg(np.arctan((tr_y - tl_y) / (tr_x - tl_x)))

            # Create a fail message on the event of a failed test
            fail_msg: str = template_fail_msg.format(i + 1, "BoundingBox.get_rotation_angle")

            # Assert that the get_rotation_angle method returns the correct value
            self.assertEqual(test_bounding_box.get_rotation_angle(), angle, msg=fail_msg)

    def test_get_width(self) -> None:
        """
        Asserts the the get_width method returns the correct value
        """

        # Template String
        template_fail_msg: str = "Test {}/100 Failed: {} did not return the correct value"

        for i in range(100):

            ## Create the required values for the test
            # New test set
            new_test_set: dict[str, Any] = self.gen_new_test_set()
            # Separate the values for easier access
            test_bounding_box: bb.BoundingBox = new_test_set["Bounding_Box"]

            ## Calculate the width
            # Get the maximum and minimum x values
            min_x: int = min(test_bounding_box.get_x_vals())  # Already tested
            max_x: int = max(test_bounding_box.get_x_vals())  # Already tested

            # Calculate the width
            width: int = max_x - min_x

            # Create a fail message on the event of a failed test
            fail_msg: str = template_fail_msg.format(i + 1, "BoundingBox.get_width")

            # Assert that the get_width method returns the correct value
            self.assertEqual(test_bounding_box.get_width(), width, msg=fail_msg)

    def test_get_height(self) -> None:
        """
        Asserts that the get_height method returns the correct value
        """

        # Template String
        template_fail_msg: str = "Test {}/100 Failed: {} did not return the correct value"

        for i in range(100):

            ## Create the required values for the test
            # New test set
            new_test_set: dict[str, Any] = self.gen_new_test_set()
            # Separate the values for easier access
            test_bounding_box: bb.BoundingBox = new_test_set["Bounding_Box"]

            ## Calculate the height
            # Get the maximum and minimum y values
            min_y: int = min(test_bounding_box.get_y_vals())  # Already tested
            max_y: int = max(test_bounding_box.get_y_vals())  # Already tested

            # Calculate the height
            height: int = max_y - min_y

            # Create a fail message on the event of a failed test
            fail_msg: str = template_fail_msg.format(i + 1, "BoundingBox.get_height")

            # Assert that the get_height method returns the correct value
            self.assertEqual(test_bounding_box.get_height(), height, msg=fail_msg)

    def test_get_width_height(self) -> None:
        """
        Asserts that the get_width_height method returns the correct value
        """

        # Template String
        template_fail_msg: str = "Test {}/100 Failed: {} did not return the correct value"

        for i in range(100):
            ## Create the required values for the test
            # New test set
            new_test_set: dict[str, Any] = self.gen_new_test_set()
            # Separate the values for easier access
            test_bounding_box: bb.BoundingBox = new_test_set["Bounding_Box"]

            # Calculate the width and height
            width: int = test_bounding_box.get_width()  # Already tested
            height: int = test_bounding_box.get_height()  # Already tested

            # Create a fail message on the event of a failed test
            fail_msg: str = template_fail_msg.format(i + 1, "BoundingBox.get_width_height")

            # Assert that the get_width_height method returns the correct value
            self.assertEqual(test_bounding_box.get_width_height(), (width, height), msg=fail_msg)

    def test_tlwh(self) -> None:
        """
        Asserts that the get_tlwh method returns the correct value
        """

        # Template String
        template_fail_msg: str = "Test {}/100 Failed: {} did not return the correct value"

        for i in range(100):
            ## Create the required values for the test
            # New test set
            new_test_set: dict[str, Any] = self.gen_new_test_set()
            # Separate the values for easier access
            test_bounding_box: bb.BoundingBox = new_test_set["Bounding_Box"]
            test_vertices: Test_Vertices_Type_Alias = new_test_set["Vertices"]

            # Get the top-left vertex
            tl_vertex: tuple[int, int] = test_vertices[0]

            # Get the width and height
            width: int = test_bounding_box.get_width()  # Already tested
            height: int = test_bounding_box.get_height()  # Already tested

            # Create a fail message on the event of a failed test
            fail_msg: str = template_fail_msg.format(i + 1, "BoundingBox.get_tlwh")

            # Assert that the get_tlwh method returns the correct value
            self.assertEqual(
                test_bounding_box.get_tlwh(),
                (tl_vertex[0], tl_vertex[1], width, height),
                msg=fail_msg,
            )


if __name__ == "__main__":
    unittest.main()
