"""
Defines a function to find the shortest path between two waypoints that
stays within the mission flight boundary.
"""

import heapq
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

from flight.waypoint.geometry import LineSegment, Point
from flight.waypoint.graph import GraphNode

type Node = GraphNode[Point, float]

logger = logging.getLogger(__name__)


@dataclass
class PathfindingGraph:
    """
    Stores nodes for path searching and the two boundaries
    used for safe pathfinding.
    """

    nodes: list[Node]
    safe_boundary: list[Point]
    outer_boundary: list[Point]


class _SearchNode(NamedTuple):
    """
    Contains a node to visit. Used when searching a graph.

    Attributes
    ----------
    priority : float
        The priority of the node to search. A lower value means a higher
        priority.
    distance_so_far : float
        The distance traveled to get to this node.
    visitor : Node
        The node that is visiting.
    node : Node
        The node to visit.
    """

    priority: float
    distance_so_far: float
    visitor: Node | None
    node: Node


def _shrink_line_segment(line_segment: LineSegment) -> LineSegment:
    """
    Shrink a line segment slightly. To be used to prevent two line segments
    sharing an endpoint from being considered as intersecting.

    Parameters
    ----------
    line_segment : LineSegment
        The line segment to shrink.

    Returns
    -------
    LineSegment
        A new line segment with the endpoints moved inward very slightly.
    """
    direction: Point = line_segment.p_2 - line_segment.p_1
    direction /= direction.distance_from_origin()
    return LineSegment(
        line_segment.p_1 + 1e-3 * direction, line_segment.p_2 - 1e-3 * direction
    )


def _visitors(start_node: Node, goal_node: Node) -> Iterable[Point]:
    """
    Yield the positions of the visitors in a path to a goal node, with the
    goal node being last.

    Parameters
    ----------
    start_node : Node
        The start node in a search.
    goal_node : Node
        The goal node in a search.

    Yields
    -------
    Point
        The positions of the visitors in a path to the goal node, from the
        start to (and including) the goal node.

    Raises
    ------
    RuntimeError
        If the path to the goal node is not valid. In normal usage, this should
        never occur.
    """
    points: list[Point] = []
    visitor: Node | None = goal_node
    while visitor is not start_node:
        if visitor is None:
            raise RuntimeError("no path was found to the destination")

        points.append(visitor.value)
        visitor = visitor.visitor

    points.reverse()
    yield from points


def _search(
    search_queue: list[_SearchNode],
    start_node: Node,
    goal_node: Node,
    boundary_line_segments: Iterable[LineSegment],
) -> bool:
    """
    Search for the goal node in the graph. Mutates the search queue passed
    in and the visitor attribute of the graph nodes.

    Parameters
    ----------
    search_queue : list[_SearchNode]
        A min priority queue used to search for the goal node. Must already
        contain the starting nodes in the graph in the correct order.
    start_node:
        The start node.
    goal_node : Node
        The goal node.
    boundary_line_segments : Iterable[LineSegment]
        The line segments forming the flight area boundary.

    Returns
    -------
    bool
        Whether the search was successful.
    """

    while search_queue:
        search_node: _SearchNode = heapq.heappop(search_queue)
        curr_distance_so_far: float = search_node.distance_so_far
        visitor: Node | None = search_node.visitor
        node: Node = search_node.node

        if node.visitor is not None:
            continue

        if (visitor == start_node or node == goal_node) and visitor is not None:
            shrunk_straight_path: LineSegment = _shrink_line_segment(
                LineSegment(visitor.value, node.value)
            )
            if any(
                shrunk_straight_path.intersects(boundary_line_segment)
                for boundary_line_segment in boundary_line_segments
            ):
                continue

        node.visitor = visitor

        if node == goal_node:
            return True

        visitor = node
        for current_node, weight in node.edges.items():
            if current_node.visitor is not None:
                continue

            distance_so_far: float = curr_distance_so_far + weight
            heapq.heappush(
                search_queue,
                _SearchNode(
                    distance_so_far
                    + LineSegment(current_node.value, goal_node.value).length(),
                    distance_so_far,
                    visitor,
                    current_node,
                ),
            )

    return False


def find_safe_point(point: Point, boundary_segments: Iterable[LineSegment]) -> Point:
    """
    Find the closest point on the boundary to the given point.
    This is used when the point is inside the outer boundary but not inside
    the shrunk boundary. We want the drone to safely move inside the closest point
    of the "safe" boundary before doing any further pathfinding.

    Parameters
    ----------
    point : Point
        The point to find the closest boundary point to.
    boundary_segments : Iterable[LineSegment]
        The boundary segments to search for safe points in.

    Returns
    -------
    Point
        The closest point on the boundary to the given point.
    """
    closest_point = point
    min_length = float("inf")
    for segment in boundary_segments:
        candidate_point = segment.closest_point_to(point)
        length = LineSegment(point, candidate_point).length()

        if length < min_length:
            min_length = length
            closest_point = candidate_point

    # Nudge the point slightly further inside the boundary
    # Normalize direction vector before applying
    vector: Point = closest_point - point
    vector = vector / vector.distance_from_origin()
    return closest_point + vector


def shortest_path_between(
    src: Point, dst: Point, graph: PathfindingGraph
) -> Iterable[Point]:
    """
    Find the shortest path between two points given a graph with all possible
    paths between boundary points.

    Parameters
    ----------
    src : Point
        The point we're currently at.
    dst : Point
        The point to move to.
    graph : PathfindingGraph
        The pathfinding graph containing the boundary nodes and boundary line segments.

    Yields
    -------
    Point
        The next point to move to in order to get to the destination using
        the shorted path.

    Raises
    ------
    RuntimeError
        If no path was found to the destination. In normal usage, this should
        never occur.
    """
    boundary_nodes: list[Node] = list(graph.nodes)

    safe_boundary_line_segments: Iterable[LineSegment] = list(
        LineSegment.from_points(graph.safe_boundary, True)
    )

    outer_boundary_line_segments: Iterable[LineSegment] = list(
        LineSegment.from_points(graph.outer_boundary, True)
    )

    # If either points are outside the outer boundary, a safe path
    # is not possible and should throw
    if not src.is_inside_shape(graph.outer_boundary) or not dst.is_inside_shape(
        graph.outer_boundary
    ):
        raise RuntimeError(
            "src or dst is outside the outer boundary, impossible to find safe path"
        )

    # If the straight line path doesn't intersect the safe boundary,
    # that will obviously be the shortest path.
    straight_path: LineSegment = LineSegment(src, dst)
    if src.is_inside_shape(graph.safe_boundary) and dst.is_inside_shape(
        graph.safe_boundary
    ):
        if not any(
            straight_path.intersects(boundary_line_segment)
            for boundary_line_segment in safe_boundary_line_segments
        ):
            yield dst
            return
    else:
        # One of the points is not inside the safe boundary, but as
        # long as the path between the points is inside the outer boundary it
        # is fine
        if not any(
            straight_path.intersects(boundary_line_segment)
            for boundary_line_segment in outer_boundary_line_segments
        ):
            yield dst
            return

    start_node: Node = GraphNode(src)
    goal_node: Node = GraphNode(dst)
    search_queue: list[_SearchNode] = []

    # If either of the two provided points are outside the safe boundary,
    # we want to find the nearest point on the safe boundary to use as the start/goal
    safe_entry: Iterable[Point] = []
    safe_exit: Iterable[Point] = []
    if not src.is_inside_shape(graph.safe_boundary):
        logger.info("src is outside safe boundary, finding closest entry point")
        closest_entry = find_safe_point(src, safe_boundary_line_segments)
        safe_entry = [closest_entry]
        start_node = GraphNode(closest_entry)

    if not dst.is_inside_shape(graph.safe_boundary):
        logger.info("dst is outside safe boundary, finding closest exit point")
        closest_exit = find_safe_point(dst, safe_boundary_line_segments)
        # We want to return the real dst so it finishes with going straight to it
        # The search algo will deal with getting to the closest_exit
        safe_exit = [dst]
        goal_node = GraphNode(closest_exit)

    # We use the A* search algorithm once we are inside the safe boundary

    for boundary_node in boundary_nodes:
        boundary_node.visitor = None

        straight_path = LineSegment(boundary_node.value, goal_node.value)
        distance_to_goal: float = straight_path.length()
        boundary_node.connect(goal_node, distance_to_goal)

        straight_path = LineSegment(start_node.value, boundary_node.value)
        distance_from_start: float = straight_path.length()
        heapq.heappush(
            search_queue,
            _SearchNode(
                distance_from_start + distance_to_goal,
                distance_from_start,
                start_node,
                boundary_node,
            ),
        )

    success: bool = _search(
        search_queue, start_node, goal_node, safe_boundary_line_segments
    )

    for boundary_node in boundary_nodes:
        boundary_node.disconnect(goal_node)

    if success:
        # If safe_entry and safe_exit are empty since they aren't needed
        # they won't yield anything
        yield from safe_entry
        yield from _visitors(start_node, goal_node)
        yield from safe_exit
    else:
        raise RuntimeError("no path was found to the destination")


def create_pathfinding_graph(
    boundary: Iterable[Point], safety_margin: float
) -> PathfindingGraph:
    """
    Create a graph suitable to be used as the boundary graph when pathfinding.

    Parameters
    ----------
    boundary : Iterable[Point]
        The vertices of the boundary. They must be in order, but it does not
        matter whether they are in clockwise or counterclockwise order.
    safety_margin : float
        How far to move the boundary vertices inward. The units are the same
        as the units for `boundary`.

    Returns
    -------
    PathfindingGraph
        A pathfinding graph containing the boundary nodes and safe boundary line segments.
    """
    points: list[Point] = list(boundary)
    points_moved_inward: list[Point] = []
    for point, line_segment_1, line_segment_2 in zip(
        points,
        LineSegment.from_points([points[-1]] + points[:-1], True),
        LineSegment.from_points(points, True),
    ):
        # line_segment_1 and line_segment_2 are the line segments connecting
        #   this point to the adjacent points

        vec_1: Point = line_segment_1.p_1 - line_segment_1.p_2
        vec_2: Point = line_segment_2.p_2 - line_segment_2.p_1
        vec_1 /= vec_1.distance_from_origin()
        vec_2 /= vec_2.distance_from_origin()

        perp_vec_1: Point = Point(-vec_1.y, vec_1.x)

        inward_diff: Point = vec_1 + vec_2
        if inward_diff.distance_from_origin() < 1e-3:
            inward_diff = perp_vec_1
        inward_diff /= inward_diff.distance_from_origin()

        if not (point + 1e-3 * inward_diff).is_inside_shape(points):
            inward_diff *= -1.0

        # The length of inward_diff in the direction of perp_vec_1
        length_divisor: float = abs(
            inward_diff.dot(perp_vec_1) / perp_vec_1.distance_from_origin()
        )
        inward_diff /= length_divisor

        inward_diff *= safety_margin
        points_moved_inward.append(point + inward_diff)

    boundary_line_segments: list[LineSegment] = list(
        LineSegment.from_points(points_moved_inward, True)
    )

    # Rather inefficient
    # Thankfully, there shouldn't be too many boundary vertices
    nodes: list[Node] = [GraphNode(point) for point in points_moved_inward]
    for node_1 in nodes:
        for node_2 in nodes:
            if node_1 == node_2:
                continue

            straight_path: LineSegment = LineSegment(node_1.value, node_2.value)
            midpoint: Point = (straight_path.p_1 + straight_path.p_2) / 2

            direction: Point = straight_path.p_2 - straight_path.p_1
            direction /= direction.distance_from_origin()
            shrunk_straight_path: LineSegment = LineSegment(
                straight_path.p_1 + 1e-3 * direction,
                straight_path.p_2 - 1e-3 * direction,
            )

            if straight_path in boundary_line_segments or (
                midpoint.is_inside_shape(points_moved_inward)
                and not any(
                    shrunk_straight_path.intersects(boundary_line_segment)
                    for boundary_line_segment in boundary_line_segments
                )
            ):
                node_1.connect(node_2, straight_path.length())

    return PathfindingGraph(nodes, points_moved_inward, points)
