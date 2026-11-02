# Traced in ordinary Julia before building the image with JuMPy's
# julia/build_backend.py. It builds and solves, through the C ABI as Python
# does, the VRP, TSP and VRPTW models of MathOptVRP's tests.
import JuMPy
import JuMPyVroom

const J = JuMPyVroom
const C = JuMPy.Client

# Customers 1:4 and the depot 5.
const DISTANCES = [0 3 4 5 2; 3 0 2 6 4; 4 2 0 3 5; 5 6 3 0 3; 2 4 5 3 0]

function routes(m, nodes, n, k)
    for j = 1:k, i = 1:n
        J.jumpy_value(m, C.index(J, m, nodes, i, j))
    end
end

# VRP: two trucks minimizing the total distance.
let m = J.jumpy_new_model(), n = 4, k = 2
    @assert J.jumpy_set_time_limit_sec(m, 5.0) == 0
    set = C.new_set(J, "MathOptVRP.Partition", n, k)
    nodes = C.constrained_variables(J, m, (n, k), set)
    J.jumpy_free_set(set)
    distances = C.array(J, m, DISTANCES)
    depot = C.constant(J, m, n + 1)
    objective = C.constant(J, m, 0)
    for j = 1:k
        route = C.concatenate(J, m, depot, C.index(J, m, nodes, 1:n, j), depot)
        cost = C.nonlinear(J, m, "sum_distances", distances, route)
        objective = C.apply(J, m, "+", objective, cost)
    end
    C.set_objective(J, m, 0, objective)
    @assert J.jumpy_optimize(m) == 1
    J.jumpy_objective_value(m)
    J.jumpy_has_values(m)
    routes(m, nodes, n, k)
    J.jumpy_free_model(m)
end

# TSP: a permutation of the customers, with float distances.
let m = J.jumpy_new_model(), n = 4
    set = C.new_set(J, "MathOptVRP.Permutation", n)
    tour = C.constrained_variables(J, m, (n,), set)
    J.jumpy_free_set(set)
    depot = C.constant(J, m, n + 1)
    route = C.concatenate(J, m, depot, tour, depot)
    distances = C.array(J, m, Float64.(DISTANCES))
    C.set_objective(J, m, 0, C.nonlinear(J, m, "sum_distances", distances, route))
    @assert J.jumpy_optimize(m) == 1
    for i = 1:n
        J.jumpy_value(m, C.index(J, m, tour, i))
    end
    J.jumpy_free_model(m)
end

# VRPTW: the total time of the routes, with two copies of the depot.
let m = J.jumpy_new_model(), n = 4, k = 2
    t = J.jumpy_variables(m, Clonglong(k))
    set = C.new_set(J, "MathOptVRP.Partition", n, k)
    nodes = C.constrained_variables(J, m, (n, k), set)
    J.jumpy_free_set(set)
    order = [1, 2, 3, 4, 5, 5]
    travel = Float64.(DISTANCES[order, order])
    earliest = [0.0, 10.0, 0.0, 5.0, 0.0, 0.0]
    latest = [100.0, 100.0, 100.0, 100.0, 1e9, 1e9]
    service = [5.0, 5.0, 5.0, 5.0, 0.0, 0.0]
    objective = C.constant(J, m, 0)
    for j = 1:k
        tj = C.variable(J, m, t, j)
        C.add_constraint(J, m, tj, "MathOptInterface.GreaterThan", 0.0)
        route = C.concatenate(
            J,
            m,
            tj,
            C.constant(J, m, n + 1),
            C.index(J, m, nodes, 1:n, j),
            C.constant(J, m, n + 2),
        )
        name = "MathOptVRP.TimeWindows{WITHOUT_START_TIME}"
        @assert C.add_constraint(J, m, route, name, travel, earliest, latest, service, n) !=
                C_NULL
        objective = C.apply(J, m, "+", objective, tj)
    end
    C.set_objective(J, m, 0, objective)
    @assert J.jumpy_optimize(m) == 1
    J.jumpy_objective_value(m)
    J.jumpy_termination_status(m)
    routes(m, nodes, n, k)
    for j = 1:k
        J.jumpy_value(m, C.variable(J, m, t, j))
    end
    J.jumpy_free_model(m)
end
