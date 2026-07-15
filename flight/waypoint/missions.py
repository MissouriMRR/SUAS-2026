"""Includes tools for creating and uploading waypoint missions."""

import logging
import math
from typing import Final

import utm
from dronekit import Command, CommandSequence, LocationGlobalRelative, Vehicle
from pymavlink import mavutil

from flight.extract_gps import BoundaryPointUtm, WaypointUtm
from flight.waypoint import pathfinding
from flight.waypoint.geometry import LineSegment, Point
from flight.waypoint.graph import GraphNode

BOUNDARY_SHRINKAGE: Final[float] = 5.0  # in meters
WAYPOINT_TOLERANCE: Final[float] = 29.0  # 100ft -> 29m


# pylint: disable=too-many-instance-attributes
class WaypointMission:
    """
    Class to handle parsing waypoint laps into Mission items, appending
    additional laps as needed, and uploading it to the drone.
    """

    def __init__(
        self,
        vehicle: Vehicle,
        waypoints: list[WaypointUtm],
        boundary: list[BoundaryPointUtm],
    ) -> None:
        self.vehicle: Vehicle = vehicle
        self.waypoints: list[WaypointUtm] = waypoints
        self._init_boundary(boundary)
        self.mission: list[Command] = []
        # Mission sequence number of each real waypoint (the ones from the
        # mission data), excluding intermediary boundary-avoidance points.
        self.waypoint_seqs: list[int] = []
        self.laps: int = 0
        self._finalized: bool = False

        self.command_sequence: CommandSequence = self.vehicle.commands
        # Download the vehicle's current mission (if any), once, so `clear()`
        # (called from `upload_mission()` every time) correctly preserves the real
        # home position at seq 0 instead of mistaking a leftover command from a
        # previous flight for it.
        self.command_sequence.download()
        self.command_sequence.wait_ready()
        self.command_sequence.next = 1  # 0 is home; 1 is the first queued waypoint

    def _init_boundary(self, boundary: list[BoundaryPointUtm]) -> None:
        # Deduplicate last boundary point if exists
        if (
            len(boundary) > 1
            and boundary[-1].easting == boundary[0].easting
            and boundary[-1].northing == boundary[0].northing
        ):
            _ = boundary.pop()

        boundary_vertices: list[Point] = []
        for point in boundary:
            boundary_vertices.append(Point(point.easting, point.northing))

        self.search_graph: list[GraphNode[Point, float]] = pathfinding.create_pathfinding_graph(
            boundary_vertices, BOUNDARY_SHRINKAGE
        )
        self.boundary: list[BoundaryPointUtm] = boundary
        self.zone_number: int = self.boundary[0].zone_number
        self.zone_letter: str = self.boundary[0].zone_letter

    def _get_drone_pos(self) -> tuple[Point, float]:
        drone_position: LocationGlobalRelative = self.vehicle.location.global_relative_frame
        drone_easting: float
        drone_northing: float
        drone_easting, drone_northing, _, _ = utm.from_latlon(
            drone_position.lat, drone_position.lon, self.zone_number, self.zone_letter
        )
        # Sometimes altitude can be unknown
        drone_alt: float = drone_position.alt if drone_position.alt is not None else 0.0
        return Point(drone_easting, drone_northing), drone_alt

    def _find_best_path(self, start_point: Point, end_point: Point) -> list[Point]:
        path: list[Point]
        try:
            path = list(
                pathfinding.shortest_path_between(start_point, end_point, self.search_graph)
            )
        except RuntimeError:
            # No path found, just use a direct path
            # (this should never happen)
            logging.warning("No path found, using direct path")
            path = [start_point, end_point]
        return path

    def add_lap(self) -> None:
        """
        Adds a lap of waypoints from self.waypoints to the stored mission,
        avoiding the boundary using intermediary points if needed.
        """
        last_point: Point
        last_altitude: float
        if len(self.mission) == 0:
            # First lap, get drone position
            last_point, last_altitude = self._get_drone_pos()
        else:
            # Subsequent laps are routed from the previous lap's final waypoint
            last_point = Point(self.waypoints[-1].easting, self.waypoints[-1].northing)
            last_altitude = self.waypoints[-1].altitude

        for waypoint in self.waypoints:
            # Find the best path to the next waypoint,
            # avoiding the boundary.
            path = self._find_best_path(last_point, Point(waypoint.easting, waypoint.northing))

            segments: list[LineSegment] = list(LineSegment.from_points([last_point] + path, False))
            path_length: float = sum(segment.length() for segment in segments)
            leg_start_altitude: float = last_altitude

            traveled: float = 0.0
            for index, point in enumerate(path):
                traveled += segments[index].length()
                is_final_point: bool = index == len(path) - 1

                # Gradually move toward the end altitude along the path
                altitude_fraction: float = traveled / path_length
                curr_altitude = (
                    waypoint.altitude
                    if is_final_point
                    else leg_start_altitude
                    + (waypoint.altitude - leg_start_altitude) * altitude_fraction
                )

                lat_deg, lon_deg = utm.to_latlon(
                    point.x, point.y, self.zone_number, self.zone_letter
                )
                self.mission.append(
                    Command(
                        0,  # target_sys
                        0,  # target_comp
                        0,  # sequence
                        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,  # Frame
                        mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,  # Command
                        0,  # Current
                        0,  # Autocontinue
                        0,  # Hold time
                        WAYPOINT_TOLERANCE,  # Acceptance radius, might be useless(?)
                        0,  # Pass radius (0 for straight through, >0 for curved path)
                        0,  # Yaw
                        lat_deg,  # Latitude
                        lon_deg,  # Longitude
                        curr_altitude,  # Altitude
                    )
                )
                if is_final_point:
                    # Home occupies seq 0, so the command just appended will be
                    # uploaded with seq == len(self.mission).
                    self.waypoint_seqs.append(len(self.mission))

            last_point = Point(waypoint.easting, waypoint.northing)
            last_altitude = waypoint.altitude
        self.laps += 1

    def finalize(self) -> None:
        """
        Append a dummy end command by duplicating the final waypoint and
        upload the mission. Without it, the vehicle's mission index never
        advances past the final real waypoint, so wouldn't be able to detect
        when the mission is complete.
        """
        if self._finalized:
            return
        self.mission.append(self.mission[-1])
        self._finalized = True
        self.upload()

    def unfinalize(self) -> None:
        """
        Remove the dummy end command added by finalize() so that more
        laps can be appended. Does not re-upload the mission as laps should
        be added after.
        """
        if not self._finalized:
            return
        _ = self.mission.pop()
        self._finalized = False

    def waypoints_reached(self) -> int:
        """
        Count how many real waypoints (no intermediaries) the drone has reached so far,
        across all laps, based on which mission command the vehicle is currently flying
        to.

        Returns
        -------
        int
            The number of real waypoints reached since the mission started.
        """
        next_seq: int = self.command_sequence.next
        return sum(1 for seq in self.waypoint_seqs if next_seq > seq)

    def distance_to_waypoint(self, waypoint_num: int) -> float:
        """
        Get the drone's current 3D distance in meters from a waypoint.

        Parameters
        ----------
        waypoint_num : int
            The overall zero-based waypoint number since the start of the
            mission, counting across laps as counted by waypoints_reached().

        Returns
        -------
        float
            The distance in meters from the drone to the waypoint.
        """
        waypoint: WaypointUtm = self.waypoints[waypoint_num % len(self.waypoints)]
        drone_point, drone_alt = self._get_drone_pos()
        return math.hypot(
            drone_point.x - waypoint.easting,
            drone_point.y - waypoint.northing,
            drone_alt - waypoint.altitude,
        )

    def upload(self) -> None:
        """
        Upload the current stored mission to the vehicle.
        """

        self.command_sequence.clear()
        for command in self.mission:
            self.command_sequence.add(command)
        self.command_sequence.upload()
