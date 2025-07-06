"""This module contains miscellaneous helper functions that don't fit into any other category."""

import yaml
from pathlib import Path
from typing import Any, Dict, Union


def create_folder_if_not_exists(folder_path: Path) -> None:
    """
    Creates a folder if it does not already exist.

    This function checks if the specified folder path exists. If it doesn't, it creates
    the folder along with any necessary parent directories. If the folder already exists,
    it simply prints a message indicating so.

    Args:
        folder_path (Path): The path of the folder to create.

    Returns:
        None

    Prints:
        A message indicating whether the folder was created or already existed.

    Raises:
        OSError: If there's an error creating the directory (e.g., insufficient permissions).
    """
    # Check if the path exists and is a directory
    if not folder_path.exists():
        # Create the directory (including any necessary parent directories)
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"Folder created: {folder_path.as_posix()}")
    else:
        print(f"Folder already exists: {folder_path.as_posix()}")


def get_src_folder_absolute_path():
    # Get the current file's directory
    current_file_path = Path(__file__).resolve()

    # Search for the 'src' folder upwards
    for parent in current_file_path.parents:
        if parent.name == "src":
            return parent

    raise RuntimeError("No 'src' folder found in the path hierarchy.")
