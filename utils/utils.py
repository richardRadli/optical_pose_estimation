import colorlog
import logging
import os
import re

from typing import List, Union

from datetime import datetime
from pathlib import Path


def create_dir(root_dir, timestamp):
    location = os.path.join(root_dir, f"{timestamp}")
    os.makedirs(location, exist_ok=True)
    return location


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- C R E A T E   T I M E S T A M P ------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def create_timestamp() -> str:
    """
    Creates a timestamp in the format of '%Y-%m-%d_%H-%M-%S', representing the current date and time.

    :return: The timestamp string.
    """

    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ F I L E   R E A D E R -----------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def file_reader(file_path: str, extension: str):
    """

    :param file_path:
    :param extension:
    :return:
    """

    return sorted([str(file) for file in Path(file_path).glob(f'*.{extension}')], key=numerical_sort)


def find_latest_file_in_latest_directory(path):
    dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if not dirs:
        raise ValueError(f'No directories at given path: {path}')

    dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_dir = dirs[0]
    files = [os.path.join(latest_dir, f) for f in os.listdir(latest_dir) if os.path.isfile(os.path.join(latest_dir, f))]

    if not files:
        raise ValueError(f'No files at given path: {latest_dir}')

    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_file = files[0]
    logging.info(f'Latest file: {latest_file}')

    return latest_file


def find_latest_subdir(directory):
    # Get a list of all subdirectories in the given directory
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    # Check if there are any subdirectories
    if not subdirectories:
        print(f"No subdirectories found in {directory}.")
        return None

    # Find the latest subdirectory based on the last modification time
    latest_subdir = max(subdirectories, key=lambda d: os.path.getmtime(os.path.join(directory, d)))

    return os.path.join(directory, latest_subdir)


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- N U M E R I C A L   S O R T --------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def numerical_sort(value: str) -> List[Union[str, int]]:
    """
    Sort numerical values in a string in a way that ensures numerical values are sorted correctly.

    :param value: The input string.
    :return: A list of strings and integers sorted by numerical value.
    """

    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def setup_logger():
    """
    Set up a colorized logger with the following log levels and colors:

    - DEBUG: Cyan
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Red on a white background

    Returns:
        The configured logger instance.
    """

    # Check if logger has already been set up
    logger = logging.getLogger()
    if logger.hasHandlers():
        return logger

    # Set up logging
    logger.setLevel(logging.INFO)

    # Create a colorized formatter
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        })

    # Create a console handler and add the formatter to it
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
