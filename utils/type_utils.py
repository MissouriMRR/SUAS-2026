"""Utility functions for types."""

from typing import TypeGuard


def assert_list_type[T](obj: object, item_type: type[T]) -> list[T]:
    """Assert that an object is a list with the specified item type.

    Parameters
    ----------
    obj : object
        The object to check and return.
    item_type : type[T]
        The type of items in the list. Due to type erasure, this must not be a generic type.

    Returns
    -------
    list[T]
        The original object `obj`, if it is a list whose items are all of the type given by
        `item_type`.

    Raises
    ------
    TypeError
        If `obj` is not a list whose items are all of the type given by `item_type`.
    """
    if check_list_type(obj, item_type):
        return obj
    raise TypeError(f"object is not of type 'list[{item_type}]'")


def check_list_type[T](obj: object, item_type: type[T]) -> TypeGuard[list[T]]:
    """Check if an object is a list with the specified item type.

    Parameters
    ----------
    obj : object
        The object to check.
    item_type : type[T]
        The type of items in the list. Due to type erasure, this must not be a generic type.

    Returns
    -------
    TypeGuard[list[T]]
        True if `obj` is a list whose items are all of the type given by `item_type`, otherwise
        False.
    """
    return isinstance(obj, list) and all(isinstance(item, item_type) for item in obj)
