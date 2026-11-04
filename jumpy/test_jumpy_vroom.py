"""Solve the models of MathOptVRP's tests from Python with the compiled Vroom
backend of JuMPy, built by JuMPy's julia/build_backend.py:

    pip install jumpy-opt dist/jumpy_opt_vroom-*.whl pytest
    pytest jumpy/test_jumpy_vroom.py
"""

import pytest

import jumpy.vroom as jp

# Customers 1 to 4 and the depot 5.
DISTANCES = [
    [0, 3, 4, 5, 2],
    [3, 0, 2, 6, 4],
    [4, 2, 0, 3, 5],
    [5, 6, 3, 0, 3],
    [2, 4, 5, 3, 0],
]
DEPOT = 5


def routes(model, nodes):
    """The customers of each truck: its column of nodes before the first 0."""
    n, k = nodes.shape
    result = []
    for j in range(k):
        route = []
        for i in range(n):
            customer = round(model.value(nodes[i, j]))
            if customer == 0:
                break
            route.append(customer)
        result.append(route)
    return result


def distance(route):
    stops = [DEPOT, *route, DEPOT]
    return sum(DISTANCES[a - 1][b - 1] for a, b in zip(stops, stops[1:]))


def test_vrp():
    model = jp.Model()
    n, k = 4, 2
    nodes = model.variables((n, k), jp.MathOptVRP.Partition(n, k))
    model.objective = jp.minimize(
        sum(
            jp.MathOptVRP.op_sum_distances(DISTANCES, jp.vcat(DEPOT, nodes[:, j], DEPOT))
            for j in range(k)
        )
    )
    assert model.optimize() == 1
    solution = routes(model, nodes)
    assert sorted(customer for route in solution for customer in route) == [1, 2, 3, 4]
    assert model.objective_value() == sum(distance(route) for route in solution) == 13


def test_cvrp():
    model = jp.Model()
    n, k = 4, 2
    nodes = model.variables((n, k), jp.MathOptVRP.Partition(n, k))
    for j in range(k):
        # Each customer loads 2 units and a truck carries at most 4.
        model.constraint(nodes[:, j], jp.MathOptVRP.Capacity([2] * n, 4))
    model.objective = jp.minimize(
        sum(
            jp.MathOptVRP.op_sum_distances(DISTANCES, jp.vcat(DEPOT, nodes[:, j], DEPOT))
            for j in range(k)
        )
    )
    assert model.optimize() == 1
    solution = routes(model, nodes)
    assert sorted(customer for route in solution for customer in route) == [1, 2, 3, 4]
    assert all(len(route) <= 2 for route in solution)
    # Without the capacity, a single truck visits all customers for 13.
    assert model.objective_value() == sum(distance(route) for route in solution) > 13


def test_tsp():
    model = jp.Model()
    tour = model.variables(4, jp.MathOptVRP.Permutation(4))
    model.objective = jp.minimize(
        jp.MathOptVRP.op_sum_distances(DISTANCES, jp.vcat(DEPOT, tour, DEPOT))
    )
    assert model.optimize() == 1
    route = [round(model.value(node)) for node in tour]
    assert sorted(route) == [1, 2, 3, 4]
    assert model.objective_value() == distance(route)


def test_vrptw():
    model = jp.Model()
    n, k = 4, 2
    t = model.variables(k, lower=0)
    nodes = model.variables((n, k), jp.MathOptVRP.Partition(n, k))
    # Two copies of the depot, 5 and 6, start and end each route.
    order = [0, 1, 2, 3, 4, 4]
    travel = [[DISTANCES[a][b] for b in order] for a in order]
    earliest = [0, 10, 0, 5, 0, 0]
    latest = [100, 100, 100, 100, 10**9, 10**9]
    service = [5, 5, 5, 5, 0, 0]
    windows = jp.MathOptVRP.TimeWindows["WITHOUT_START_TIME"]
    for j in range(k):
        model.constraint(
            jp.vcat(t[j], n + 1, nodes[:, j], n + 2),
            windows(travel, earliest, latest, service, n),
        )
    model.objective = jp.minimize(sum(t))
    assert model.optimize() == 1
    solution = routes(model, nodes)
    assert sorted(customer for route in solution for customer in route) == [1, 2, 3, 4]
    assert model.objective_value() == pytest.approx(sum(model.value(x) for x in t))


def test_registry():
    VRP = jp.MathOptVRP
    assert {"Capacity", "Partition", "PartitionPD", "Permutation", "TimeWindows"} <= set(
        dir(VRP)
    )
    assert "op_sum_distances" in dir(VRP)
    assert dir(jp.MOI) == ["GreaterThan"]
    with pytest.raises(AttributeError, match="not exposed"):
        VRP.RouteOrder
