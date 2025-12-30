"""
Helper utility functions.

Common utilities for number formatting, data manipulation, and other shared operations.
"""

import numpy as np


def average(arr: np.ndarray) -> np.generic:
    """
    Find the most frequent value in an array (mode).

    Args:
        arr: Input numpy array

    Returns:
        The most frequent value in the array
    """
    unique, counts = np.unique(arr, return_counts=True)
    return unique[np.argmax(counts)]


def format_number_and_round_numpy(
    number: int | float | np.integer | np.floating,
) -> int | float:
    """
    Format and round a numpy number.

    Integer types are converted to Python int.
    Float types are rounded to 3 decimal places and converted to Python float.

    Args:
        number: Input number (numpy or Python numeric type)

    Returns:
        Formatted Python int or float

    Raises:
        ValueError: If input is not a valid numeric type
    """
    if isinstance(number, (np.integer, int)):
        return int(number)

    if isinstance(number, (np.floating, float)):
        return float(round(number, 3))

    raise ValueError(f"Invalid number type: {type(number)}")


def format_number_and_round(number: int | float) -> int | float:
    """
    Format and round a Python number.

    Integers are returned as-is.
    Floats that are whole numbers are converted to int.
    Other floats are rounded to 3 decimal places.

    Args:
        number: Input number (int or float)

    Returns:
        Formatted int or float

    Raises:
        ValueError: If input is not int or float
    """
    if not isinstance(number, (int, float)):
        raise ValueError(f"Invalid number type: {type(number)}")

    if isinstance(number, int):
        return number

    # float
    if number.is_integer():
        return int(number)

    return float(round(number, 3))
