"""Utility functions for types."""

from typing import Type, TypeGuard, TypeVar

T = TypeVar("T")


def assert_list_type(obj: object, item_type: Type[T]) -> list[T]:
    """Assert that an object is a list with the specified item type.

    Parameters
    ----------
    obj : object
        The object to check and return.
    item_type : Type[T]
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
    raise TypeError(f"object is not of type 'list[{item_type.__name__}]'")


def check_list_type(obj: object, item_type: Type[T]) -> TypeGuard[list[T]]:
    """Check if an object is a list with the specified item type.

    Parameters
    ----------
    obj : object
        The object to check.
    item_type : Type[T]
        The type of items in the list. Due to type erasure, this must not be a generic type.

    Returns
    -------
    TypeGuard[list[T]]
        True if `obj` is a list whose items are all of the type given by `item_type`, otherwise
        False.
    """
    if not isinstance(obj, list):
        return False

    for item in obj:
        if not isinstance(item, item_type):
            return False

    return True
