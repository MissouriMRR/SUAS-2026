"""
Used to keep track of mission progress and resuming from it later, saving and loading to the progress file
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict

DEFAULT_PROGRESS_PATH = Path("state_machine/progress.json")

RESUMABLE_STATES: Final[tuple[str, ...]] = ("Waypoint", "ODLC", "Mapping", "Airdrop")


class MissionProgressJSON(TypedDict):
    state: str | None
    waypoint_laps_complete: int
    image_capture_complete: bool


@dataclass
class MissionProgress:
    state: str | None = None
    waypoint_laps_complete: int = 0
    image_capture_complete: bool = False

    def to_dict(self) -> MissionProgressJSON:
        """
        Convert progress data into the dictionary written to the JSON file.

        Returns
        -------
        MissionProgressJSON
            The JSON representation of this progress.
        """
        return {
            "state": self.state,
            "waypoint_laps_complete": self.waypoint_laps_complete,
            "image_capture_complete": self.image_capture_complete,
        }

    @classmethod
    def load_default(cls) -> "MissionProgress":
        return cls(state=None, waypoint_laps_complete=0, image_capture_complete=False)

    @classmethod
    def load_progress_file(
        cls, file_path: Path = DEFAULT_PROGRESS_PATH
    ) -> "MissionProgress":
        """
        Reads the data from the state JSON file.

        Parameters
        ----------
        file_path : Path, optional
            The file path of the JSON file, by default "state_machine/progress.json"

        Returns
        -------
        MissionProgress
            The data from the JSON file
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data: MissionProgressJSON = json.load(file)
                return MissionProgress(**data)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {file_path} not found.")

    def update_progress_file(self, file_path: Path = DEFAULT_PROGRESS_PATH) -> None:
        """
        Updates the progress file to reflect the current mission progress.

        Parameters
        ----------
        file_path : Path, optional
            The file path of the JSON file, by default "state_machine/progress.json"
        """
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=4)

        logging.info("Progress updated in %s.", file_path)
