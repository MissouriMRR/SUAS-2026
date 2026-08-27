"""General pipeline utility functions."""

import json

import vision.common.constants as consts


def read_parameter_json(json_path: str) -> dict[str, consts.CameraParameters]:
    """
    Will read in the data from the given json file and return it as a python dict.

    Parameters
    ----------
    json_path : str
        The path of a valid json file, assumed to have data in the same format as return type.

    Returns
    -------
    data : dict[str, CameraParameters]
        The python dict version of the data from the given json file.
    """

    with open(json_path, encoding="utf-8") as jfile:
        data: dict[str, consts.CameraParameters] = json.load(jfile)

    return data
