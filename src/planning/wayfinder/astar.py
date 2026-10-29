"""A* over the exported ``LaneGraph`` with length cost and a planar heuristic (E3-01).

The search uses only the directed edges present in the graph. The exported
topology (BL-A-02) contains lane edges and legal successor connections and no
lane-change edges, so v1 cannot plan a lane change: it never invents an edge
and never relaxes a transition that the exporter did not list.

Edge cost is the planar polyline length of the edge in metres. The heuristic
is the planar Euclidean distance between a node and the goal; because every
edge polyline is at least as long as its XY chord, the heuristic never
overestimates and A* returns a shortest path. Node elevation is kept on the
nodes but is not part of the cost in this planar v1.

Ties are broken deterministically by ``(f, g, node_id)``; the adjacency list is
sorted by end node ID; parents change only on a strictly shorter tentative
cost. The result for a given graph therefore does not depend on the order in
which nodes or edges were listed.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from autonomy.interfaces import GraphEdge, LaneGraph

__all__ = ["SearchResult", "edge_length_m", "euclidean_xy_m", "shortest_path"]


def edge_length_m(edge: GraphEdge) -> float:
    """Planar polyline length of one directed edge."""
    return sum(math.dist(a, b) for a, b in zip(edge.points_xy_m, edge.points_xy_m[1:]))


def euclidean_xy_m(a, b) -> float:
    """Admissible planar heuristic between two graph nodes."""
    return math.hypot(a.x_m - b.x_m, a.y_m - b.y_m)


@dataclass(frozen=True)
class SearchResult:
    """Outcome of one shortest-path query.

    ``reachable`` is False when no legal edge sequence links the two nodes; the
    path is then empty and ``length_m`` is ``inf``. This is a normal result, not
    an exception, so callers must check it before using the path. A query from
    a node to itself is reachable with an empty edge list and zero length.
    """

    start_id: str
    goal_id: str
    reachable: bool
    node_ids: tuple[str, ...]
    edges: tuple[GraphEdge, ...]
    length_m: float
    expanded: int

    def __post_init__(self) -> None:
        if self.reachable:
            if not self.node_ids or self.node_ids[0] != self.start_id or self.node_ids[-1] != self.goal_id:
                raise ValueError("reachable path must run from start to goal")
            if len(self.edges) != len(self.node_ids) - 1:
                raise ValueError("path edges must link consecutive nodes")
            for edge, a, b in zip(self.edges, self.node_ids, self.node_ids[1:]):
                if edge.start_id != a or edge.end_id != b:
                    raise ValueError("path edge does not join its nodes")
        elif self.node_ids or self.edges or self.length_m != math.inf:
            raise ValueError("unreachable result carries no path")


def _adjacency(graph: LaneGraph) -> dict[str, tuple[GraphEdge, ...]]:
    out: dict[str, list[GraphEdge]] = {node.node_id: [] for node in graph.nodes}
    for edge in graph.edges:
        out[edge.start_id].append(edge)
    return {key: tuple(sorted(edges, key=lambda e: e.end_id)) for key, edges in out.items()}


def shortest_path(graph: LaneGraph, start_id: str, goal_id: str) -> SearchResult:
    """Return the shortest legal path from ``start_id`` to ``goal_id``.

    Raises ``ValueError`` for an unknown node; returns an unreachable
    ``SearchResult`` when the goal cannot be reached along directed edges.
    """
    nodes = {node.node_id: node for node in graph.nodes}
    for name, key in (("start_id", start_id), ("goal_id", goal_id)):
        if key not in nodes:
            raise ValueError(f"{name} {key!r} is not a graph node")
    adjacency = _adjacency(graph)
    goal = nodes[goal_id]
    cost = {start_id: 0.0}
    parent: dict[str, GraphEdge] = {}
    closed: set[str] = set()
    frontier = [(euclidean_xy_m(nodes[start_id], goal), 0.0, start_id)]
    expanded = 0
    while frontier:
        _f, g, current = heapq.heappop(frontier)
        if current in closed or g > cost[current]:
            continue
        closed.add(current)
        expanded += 1
        if current == goal_id:
            break
        for edge in adjacency[current]:
            tentative = g + edge_length_m(edge)
            if edge.end_id in closed or tentative >= cost.get(edge.end_id, math.inf):
                continue
            cost[edge.end_id] = tentative
            parent[edge.end_id] = edge
            heapq.heappush(frontier, (tentative + euclidean_xy_m(nodes[edge.end_id], goal), tentative, edge.end_id))
    if goal_id not in closed:
        return SearchResult(start_id, goal_id, False, (), (), math.inf, expanded)
    edges = []
    node = goal_id
    while node != start_id:
        edge = parent[node]
        edges.append(edge)
        node = edge.start_id
    edges.reverse()
    node_ids = (start_id,) + tuple(edge.end_id for edge in edges)
    return SearchResult(start_id, goal_id, True, node_ids, tuple(edges), cost[goal_id], expanded)
