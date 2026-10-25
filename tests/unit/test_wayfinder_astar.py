"""E3-01: A* on small lane graphs matches a Dijkstra reference, including unreachable goals."""

import heapq
import math
import random

import pytest

from autonomy.interfaces import GraphEdge, GraphNode, LaneGraph
from planning.wayfinder.astar import SearchResult, edge_length_m, shortest_path


def dijkstra_reference(graph, start_id, goal_id):
    """Plain Dijkstra with the same (cost, node id) tie-break; returns (cost, node path)."""
    adjacency = {node.node_id: [] for node in graph.nodes}
    for edge in graph.edges:
        adjacency[edge.start_id].append(edge)
    distance = {start_id: 0.0}
    parent = {}
    done = set()
    heap = [(0.0, start_id)]
    while heap:
        cost, node = heapq.heappop(heap)
        if node in done:
            continue
        done.add(node)
        for edge in sorted(adjacency[node], key=lambda e: e.end_id):
            candidate = cost + edge_length_m(edge)
            if candidate < distance.get(edge.end_id, math.inf):
                distance[edge.end_id] = candidate
                parent[edge.end_id] = node
                heapq.heappush(heap, (candidate, edge.end_id))
    if goal_id not in done:
        return math.inf, ()
    path = [goal_id]
    while path[-1] != start_id:
        path.append(parent[path[-1]])
    return distance[goal_id], tuple(reversed(path))


def build(nodes, edges):
    """``nodes``: {id: (x, y)}; ``edges``: [(a, b)] straight or [(a, b, via)] with interior points."""
    graph_nodes = [GraphNode(key, x, y) for key, (x, y) in nodes.items()]
    graph_edges = []
    for spec in edges:
        a, b = spec[0], spec[1]
        via = tuple(spec[2]) if len(spec) > 2 else ()
        graph_edges.append(GraphEdge(a, b, (nodes[a],) + via + (nodes[b],), 5.0))
    return LaneGraph(graph_nodes, graph_edges)


def grid_graph(one_way=True):
    """3x3 grid; east and north edges only when one_way, plus a bent detour edge."""
    nodes = {f"n{x}{y}": (10.0 * x, 10.0 * y) for x in range(3) for y in range(3)}
    edges = []
    for x in range(3):
        for y in range(3):
            if x < 2:
                edges.append((f"n{x}{y}", f"n{x + 1}{y}"))
                if not one_way:
                    edges.append((f"n{x + 1}{y}", f"n{x}{y}"))
            if y < 2:
                edges.append((f"n{x}{y}", f"n{x}{y + 1}"))
                if not one_way:
                    edges.append((f"n{x}{y + 1}", f"n{x}{y}"))
    # A curved edge whose polyline is longer than its chord: heuristic must stay admissible.
    edges.append(("n00", "n22", ((5.0, 25.0), (25.0, 25.0))))
    return nodes, edges


@pytest.mark.parametrize("one_way", [True, False])
@pytest.mark.parametrize("goal", ["n22", "n20", "n02", "n11", "n00"])
def test_matches_dijkstra_cost_from_origin(one_way, goal):
    graph = build(*grid_graph(one_way))
    result = shortest_path(graph, "n00", goal)
    cost, path = dijkstra_reference(graph, "n00", goal)
    assert result.reachable
    assert result.length_m == pytest.approx(cost)
    assert sum(edge_length_m(e) for e in result.edges) == pytest.approx(result.length_m)
    assert result.node_ids[0] == "n00" and result.node_ids[-1] == goal
    assert len(result.node_ids) == len(path)


def test_curved_edge_is_not_chosen_over_shorter_grid_path():
    graph = build(*grid_graph())
    result = shortest_path(graph, "n00", "n22")
    # The bent direct edge measures 5+sqrt(5^2+25^2)+20+... > 40 m along the grid.
    assert edge_length_m(next(e for e in graph.edges if e.start_id == "n00" and e.end_id == "n22")) > 40
    assert result.length_m == pytest.approx(40.0)
    assert "n22" in result.node_ids and len(result.node_ids) == 5


def test_one_way_edges_make_return_unreachable_without_raising():
    graph = build(*grid_graph(one_way=True))
    forward = shortest_path(graph, "n00", "n22")
    back = shortest_path(graph, "n22", "n00")
    assert forward.reachable
    assert back == SearchResult("n22", "n00", False, (), (), math.inf, back.expanded)
    assert dijkstra_reference(graph, "n22", "n00") == (math.inf, ())
    assert back.expanded >= 1


def test_disconnected_component_is_unreachable_and_unknown_nodes_raise():
    nodes, edges = grid_graph()
    nodes["island"] = (100.0, 100.0)
    graph = build(nodes, edges)
    assert not shortest_path(graph, "n00", "island").reachable
    assert not shortest_path(graph, "island", "n00").reachable
    with pytest.raises(ValueError):
        shortest_path(graph, "n00", "ghost")
    with pytest.raises(ValueError):
        shortest_path(graph, "ghost", "n00")


def test_start_equals_goal_is_a_zero_length_path():
    graph = build(*grid_graph())
    result = shortest_path(graph, "n11", "n11")
    assert result.reachable and result.node_ids == ("n11",) and result.edges == () and result.length_m == 0


def test_tie_breaking_is_deterministic_under_listing_order():
    """Two equal-length paths exist on the grid; the choice must not depend on edge order."""
    nodes, edges = grid_graph(one_way=True)
    reference = shortest_path(build(nodes, edges), "n00", "n11")
    assert reference.length_m == pytest.approx(20.0)
    for seed in range(8):
        shuffled_edges = list(edges)
        random.Random(seed).shuffle(shuffled_edges)
        items = list(nodes.items())
        random.Random(seed + 100).shuffle(items)
        other = shortest_path(build(dict(items), shuffled_edges), "n00", "n11")
        assert other.node_ids == reference.node_ids
        assert other.length_m == pytest.approx(reference.length_m)
    # (f, g, node id) ordering: both "n01" and "n10" tie on f; the lower node id is expanded.
    assert reference.node_ids == ("n00", "n01", "n11")


def test_random_graphs_agree_with_dijkstra():
    rng = random.Random(5308)
    for _ in range(40):
        count = rng.randint(4, 12)
        nodes = {f"v{i}": (rng.uniform(-50, 50), rng.uniform(-50, 50)) for i in range(count)}
        edges = set()
        for _ in range(rng.randint(count, 3 * count)):
            a, b = rng.sample(sorted(nodes), 2)
            edges.add((a, b))
        graph = build(nodes, sorted(edges))
        for _ in range(5):
            start, goal = rng.sample(sorted(nodes), 2)
            result = shortest_path(graph, start, goal)
            cost, path = dijkstra_reference(graph, start, goal)
            assert result.reachable == (cost < math.inf)
            if result.reachable:
                assert result.length_m == pytest.approx(cost)
                assert result.node_ids[0] == start and result.node_ids[-1] == goal


def test_result_invariants_are_enforced():
    with pytest.raises(ValueError):
        SearchResult("a", "b", True, ("a",), (), 0.0, 1)
    with pytest.raises(ValueError):
        SearchResult("a", "b", False, ("a", "b"), (), math.inf, 1)
    with pytest.raises(ValueError):
        SearchResult("a", "b", False, (), (), 3.0, 1)
