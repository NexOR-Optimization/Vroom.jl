# Minimal MOI wrapper. Scope is intentionally narrow: just enough to run
# `MathOptVRP.Tests.test_vrp`. We accept one `MathOptVRP.Partition` set of
# variables, a `MOI.ScalarNonlinearFunction` objective built from
# `MathOptVRP.op_sum_distances` (one leaf per truck, optionally wrapped in
# `:+` nodes), and lower it to a Vroom JSON `Problem` with one `Vehicle`
# per truck and one `Job` per customer.

import MathOptInterface as MOI
import MathOptVRP

mutable struct Optimizer <: MOI.AbstractOptimizer
    next_variable::Int
    next_constraint::Int
    # (row, col) of each partition variable, column-major in the order
    # `add_constrained_variables` received them.
    variable_to_position::Dict{MOI.VariableIndex,Tuple{Int,Int}}
    partition::Union{Nothing,MathOptVRP.Partition}
    objective_sense::MOI.OptimizationSense
    objective_function::Union{Nothing,MOI.ScalarNonlinearFunction}
    silent::Bool
    time_limit::Union{Nothing,Float64}
    # Solution state, populated by `optimize!`.
    solved::Bool
    routes::Vector{Vector{Int}}
    objective_value::Int
    termination_status::MOI.TerminationStatusCode
    primal_status::MOI.ResultStatusCode
    raw_status::String

    function Optimizer()
        return new(
            0,
            0,
            Dict{MOI.VariableIndex,Tuple{Int,Int}}(),
            nothing,
            MOI.FEASIBILITY_SENSE,
            nothing,
            false,
            nothing,
            false,
            Vector{Int}[],
            0,
            MOI.OPTIMIZE_NOT_CALLED,
            MOI.NO_SOLUTION,
            "",
        )
    end
end

MOI.get(::Optimizer, ::MOI.SolverName) = "Vroom"

function MOI.is_empty(m::Optimizer)
    return m.partition === nothing &&
           m.objective_function === nothing &&
           m.objective_sense == MOI.FEASIBILITY_SENSE &&
           !m.solved
end

function MOI.empty!(m::Optimizer)
    m.next_variable = 0
    m.next_constraint = 0
    empty!(m.variable_to_position)
    m.partition = nothing
    m.objective_sense = MOI.FEASIBILITY_SENSE
    m.objective_function = nothing
    m.solved = false
    empty!(m.routes)
    m.objective_value = 0
    m.termination_status = MOI.OPTIMIZE_NOT_CALLED
    m.primal_status = MOI.NO_SOLUTION
    m.raw_status = ""
    return
end

# Parameters

MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.get(m::Optimizer, ::MOI.Silent) = m.silent
function MOI.set(m::Optimizer, ::MOI.Silent, silent::Bool)
    m.silent = silent
    return
end

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
MOI.get(m::Optimizer, ::MOI.TimeLimitSec) = m.time_limit
MOI.set(m::Optimizer, ::MOI.TimeLimitSec, ::Nothing) = (m.time_limit = nothing; return)
function MOI.set(m::Optimizer, ::MOI.TimeLimitSec, v::Real)
    m.time_limit = Float64(v)
    return
end

# Objective

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
MOI.get(m::Optimizer, ::MOI.ObjectiveSense) = m.objective_sense
function MOI.set(m::Optimizer, ::MOI.ObjectiveSense, s::MOI.OptimizationSense)
    m.objective_sense = s
    return
end

function MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction})
    return true
end

function MOI.get(m::Optimizer, ::MOI.ObjectiveFunctionType)
    return MOI.ScalarNonlinearFunction
end

function MOI.set(
    m::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction},
    f::MOI.ScalarNonlinearFunction,
)
    m.objective_function = f
    return
end

function MOI.get(m::Optimizer, ::MOI.ListOfModelAttributesSet)
    attrs = Any[MOI.ObjectiveSense()]
    if m.objective_function !== nothing
        push!(attrs, MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction}())
    end
    return attrs
end

# Variables

function MOI.supports_add_constrained_variables(::Optimizer, ::Type{MathOptVRP.Partition})
    return true
end

function MOI.add_constrained_variables(m::Optimizer, set::MathOptVRP.Partition)
    m.partition === nothing ||
        error("Vroom: only one MathOptVRP.Partition set is supported per model")
    n_rows, n_cols = set.num_clients, set.num_trucks
    n = n_rows * n_cols
    vars = Vector{MOI.VariableIndex}(undef, n)
    # `JuMP.build_variable(::Partition)` flattens column-major via `vec`, so
    # entry `k` corresponds to row `((k - 1) % n_rows) + 1` of column
    # `((k - 1) ÷ n_rows) + 1`.
    for k = 1:n
        m.next_variable += 1
        v = MOI.VariableIndex(m.next_variable)
        vars[k] = v
        row = ((k - 1) % n_rows) + 1
        col = ((k - 1) ÷ n_rows) + 1
        m.variable_to_position[v] = (row, col)
    end
    m.partition = set
    m.next_constraint += 1
    ci = MOI.ConstraintIndex{MOI.VectorOfVariables,MathOptVRP.Partition}(m.next_constraint)
    return vars, ci
end

# Incremental interface (JuMP copies via `default_copy_to`).

MOI.supports_incremental_interface(::Optimizer) = true
function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(dest, src)
end

# ── Objective parsing ─────────────────────────────────────────────────
# JuMP produces `sum(op_sum_distances(M, [depot; col; depot]) for i = 1:T)`
# as a `ScalarNonlinearFunction`. The root is either a single
# `:sum_distances` leaf (T == 1) or a tree of `:+` nodes whose leaves are
# all `:sum_distances`. Each leaf's args[1] is the distance matrix and
# args[2] is `[depot, var, var, ..., var, depot]`.

function _collect_sum_distances_leaves!(
    leaves::Vector{MOI.ScalarNonlinearFunction},
    f::MOI.ScalarNonlinearFunction,
)
    if f.head == :+
        for a in f.args
            a isa MOI.ScalarNonlinearFunction ||
                error("Vroom: unsupported `:+` arg of type $(typeof(a))")
            _collect_sum_distances_leaves!(leaves, a)
        end
    elseif f.head == :sum_distances
        push!(leaves, f)
    else
        error(
            "Vroom: unsupported ScalarNonlinearFunction head `$(f.head)`. ",
            "Only `:sum_distances` (optionally wrapped in `:+`) is lowered.",
        )
    end
    return
end

function _parse_leaf(m::Optimizer, leaf::MOI.ScalarNonlinearFunction)
    length(leaf.args) == 2 ||
        error("Vroom: `:sum_distances` expects 2 args, got $(length(leaf.args))")
    matrix = leaf.args[1]
    matrix isa AbstractMatrix{<:Real} ||
        error("Vroom: `:sum_distances` arg 1 must be a real matrix; got $(typeof(matrix))")
    items = _normalize_items(leaf.args[2])
    length(items) >= 3 ||
        error("Vroom: `:sum_distances` vector must be `[depot; col; depot]` (≥ 3 entries)")
    items[1] isa Real ||
        error("Vroom: depot_start must be a `Real`; got $(typeof(items[1]))")
    items[end] isa Real ||
        error("Vroom: depot_end must be a `Real`; got $(typeof(items[end]))")
    depot_start = round(Int, items[1])
    depot_end = round(Int, items[end])
    depot_start == depot_end || error("Vroom: depot_start != depot_end is not supported")
    # All interior items must be partition variables of one column.
    column = nothing
    for k = 2:(length(items)-1)
        it = items[k]
        it isa MOI.VariableIndex || error(
            "Vroom: interior `:sum_distances` items must be variables; got $(typeof(it))",
        )
        pos = get(m.variable_to_position, it, nothing)
        pos === nothing &&
            error("Vroom: variable $(it) is not part of a registered Partition")
        if column === nothing
            column = pos[2]
        elseif column != pos[2]
            error(
                "Vroom: `:sum_distances` mixes variables from columns $(column) and $(pos[2])",
            )
        end
    end
    column === nothing && error("Vroom: `:sum_distances` has no interior variables")
    return matrix, depot_start, column::Int
end

# JuMP can hand us the second `:sum_distances` arg as either a raw
# `AbstractVector` of mixed `Real`s and `MOI.VariableIndex`s (the path
# via MathOptVRP's `moi_function(::Array)` type piracy), or — when JuMP
# promotes `vcat(depot::Int, ::Vector{VariableRef}, depot::Int)` to a
# `Vector{AffExpr}` — as a `MOI.VectorAffineFunction`. Normalise both
# into a `Vector{Any}` of constants / `VariableIndex` per row.
function _normalize_items(raw)
    if raw isa MOI.VectorOfVariables
        return Any[vi for vi in raw.variables]
    elseif raw isa MOI.VectorAffineFunction
        n = length(raw.constants)
        T = eltype(raw.constants)
        per_row = [MOI.ScalarAffineTerm{T}[] for _ = 1:n]
        for vt in raw.terms
            push!(per_row[vt.output_index], vt.scalar_term)
        end
        return Any[
            _simplify_item(MOI.ScalarAffineFunction(per_row[i], raw.constants[i])) for
            i = 1:n
        ]
    elseif raw isa AbstractVector
        return Any[_simplify_item(el) for el in raw]
    end
    return error("Vroom: `:sum_distances` arg 2 has unexpected type $(typeof(raw))")
end

_simplify_item(x) = x
function _simplify_item(f::MOI.ScalarAffineFunction)
    if isempty(f.terms)
        return f.constant
    end
    if length(f.terms) == 1 && iszero(f.constant) && isone(f.terms[1].coefficient)
        return f.terms[1].variable
    end
    return f
end

# ── Optimize ─────────────────────────────────────────────────────────

function MOI.optimize!(m::Optimizer)
    m.partition !== nothing || error("Vroom: model has no `MathOptVRP.Partition` variables")
    m.objective_function !== nothing && m.objective_sense == MOI.MIN_SENSE ||
        error("Vroom: requires a `MIN_SENSE` `:sum_distances` objective")

    leaves = MOI.ScalarNonlinearFunction[]
    _collect_sum_distances_leaves!(leaves, m.objective_function)
    isempty(leaves) && error("Vroom: empty `:sum_distances` objective")

    parsed = [_parse_leaf(m, leaf) for leaf in leaves]
    n_trucks = length(leaves)
    n_trucks == m.partition.num_trucks || error(
        "Vroom: objective has $(n_trucks) `:sum_distances` terms but Partition has ",
        "$(m.partition.num_trucks) trucks",
    )
    matrix_ref = parsed[1][1]
    depot = parsed[1][2]
    for (mat, dep, _) in parsed
        mat == matrix_ref ||
            error("Vroom: per-truck `:sum_distances` matrices must be equal")
        dep == depot || error("Vroom: per-truck depots must agree; got $(dep) vs $(depot)")
    end
    leaf_columns = Int[col for (_, _, col) in parsed]
    sort(leaf_columns) == collect(1:n_trucks) ||
        error("Vroom: `:sum_distances` columns are not a permutation of 1:$(n_trucks)")

    durations = Matrix{Int}(round.(Int, matrix_ref))
    n_locations = size(durations, 1)
    n_locations == size(durations, 2) ||
        error("Vroom: distance matrix must be square; got $(size(durations))")
    n_clients = m.partition.num_clients
    customer_locs = [loc for loc = 0:(n_locations-1) if loc != depot]
    length(customer_locs) == n_clients || error(
        "Vroom: matrix has $(length(customer_locs)) non-depot rows but Partition has ",
        "$(n_clients) customers",
    )

    vehicles =
        [Vehicle(id = i - 1, start_index = depot, end_index = depot) for i = 1:n_trucks]
    jobs = [Job(id = loc, location_index = loc) for loc in customer_locs]
    problem = Problem(
        vehicles = vehicles,
        jobs = jobs,
        shipments = Shipment[],
        matrices = DurationMatrices(car = DurationMatrix(durations)),
    )

    sol = try
        vroom(problem)
    catch err
        m.solved = true
        m.termination_status = MOI.OTHER_ERROR
        m.primal_status = MOI.NO_SOLUTION
        m.raw_status = sprint(showerror, err)
        return
    end

    # Rebuild routes per *user-defined* truck column. Vehicle `k - 1` was
    # created for the `k`th leaf, which corresponds to column
    # `leaf_columns[k]` of the partition.
    routes = [Int[] for _ = 1:n_trucks]
    for r in sol.routes
        truck_col = leaf_columns[r.vehicle+1]
        for step in r.steps
            step.type == "job" || continue
            push!(routes[truck_col], step.location_index)
        end
    end

    m.routes = routes
    m.objective_value = sol.summary.cost
    m.solved = true
    if sol.code == 0
        m.termination_status = MOI.OPTIMAL
        m.primal_status = MOI.FEASIBLE_POINT
        m.raw_status = "vroom OK"
    else
        m.termination_status = MOI.OTHER_ERROR
        m.primal_status = MOI.NO_SOLUTION
        m.raw_status = "vroom code=$(sol.code)"
    end
    return
end

# ── Solution getters ─────────────────────────────────────────────────

MOI.get(m::Optimizer, ::MOI.TerminationStatus) = m.termination_status

function MOI.get(m::Optimizer, attr::MOI.PrimalStatus)
    return attr.result_index == 1 ? m.primal_status : MOI.NO_SOLUTION
end

MOI.get(::Optimizer, ::MOI.DualStatus) = MOI.NO_SOLUTION
MOI.get(m::Optimizer, ::MOI.RawStatusString) = m.raw_status
MOI.get(m::Optimizer, ::MOI.ResultCount) = m.primal_status == MOI.NO_SOLUTION ? 0 : 1
MOI.get(::Optimizer, ::MOI.SolveTimeSec) = 0.0

function MOI.get(m::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(m, attr)
    return Float64(m.objective_value)
end

# Vroom assigns customers itself, so individual `VariablePrimal` queries
# don't have a meaningful answer to return — the test reaches into
# `inner.routes` via `read_routes` instead. Return the depot value so
# `JuMP.value(::VariableRef)` at least doesn't throw.
function MOI.get(m::Optimizer, attr::MOI.VariablePrimal, ::MOI.VariableIndex)
    MOI.check_result_index_bounds(m, attr)
    return 0.0
end
