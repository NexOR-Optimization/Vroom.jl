# Minimal MOI wrapper. Scope is intentionally narrow: just enough to run
# `MathOptVRP.Tests.test_vrp`, `test_tsp`, `test_vrppd` and `test_vrptw`.
# We accept one `MathOptVRP.Partition` or `MathOptVRP.PartitionPD` set of
# variables, either:
#   - a `MOI.ScalarNonlinearFunction` objective built from
#     `MathOptVRP.op_sum_distances` (one leaf per truck, optionally wrapped
#     in `:+` nodes), lowered to a Vroom JSON `Problem` with one `Vehicle`
#     per truck, one `Job` per plain customer/service, and one `Shipment`
#     per pickup/delivery pair; or
#   - a linear `sum(t)` objective over free `t[i] >= 0` variables plus one
#     `MathOptVRP.TimeWindows` constraint per truck, lowered to a `Problem`
#     with per-`Job` `time_windows`/`service`, reading each truck's total
#     time back off Vroom's `"end"` step `arrival`.

import MathOptInterface as MOI
import MathOptVRP

# Per-truck data parsed out of one `MathOptVRP.TimeWindows` constraint;
# see `MOI.add_constraint(::Optimizer, ::MOI.VectorAffineFunction, ::MathOptVRP.TimeWindows)`.
struct _TimeWindowsEntry
    t_var::MOI.VariableIndex
    depot_start::Int
    depot_end::Int
    set::MathOptVRP.TimeWindows
end

mutable struct Optimizer <: MOI.AbstractOptimizer
    next_variable::Int
    next_constraint::Int
    # (row, col) of each partition variable, column-major in the order
    # `add_constrained_variables` received them.
    variable_to_position::Dict{MOI.VariableIndex,Tuple{Int,Int}}
    partition::Union{Nothing,MathOptVRP.PartitionPD}
    objective_sense::MOI.OptimizationSense
    # `ScalarNonlinearFunction` is a `:sum_distances` objective (vrp)
    # `ScalarAffineFunction{Float64}` is a `sum(t)` objective
    # paired with `time_windows_by_column` (vrptw).
    objective_function::Union{
        Nothing,
        MOI.ScalarNonlinearFunction,
        MOI.ScalarAffineFunction{Float64},
    }
    # One `MathOptVRP.TimeWindows` constraint per truck column, keyed by
    # column; populated by `add_constraint`, consumed by `optimize!`.
    time_windows_by_column::Dict{Int,_TimeWindowsEntry}
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
            Dict{Int,_TimeWindowsEntry}(),
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
           isempty(m.time_windows_by_column) &&
           !m.solved
end

function MOI.empty!(m::Optimizer)
    m.next_variable = 0
    m.next_constraint = 0
    empty!(m.variable_to_position)
    m.partition = nothing
    m.objective_sense = MOI.FEASIBILITY_SENSE
    m.objective_function = nothing
    empty!(m.time_windows_by_column)
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

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
)
    return true
end

function MOI.get(m::Optimizer, ::MOI.ObjectiveFunctionType)
    m.objective_function isa MOI.ScalarAffineFunction{Float64} &&
        return MOI.ScalarAffineFunction{Float64}
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

function MOI.set(
    m::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
    f::MOI.ScalarAffineFunction{Float64},
)
    m.objective_function = f
    return
end

function MOI.get(m::Optimizer, ::MOI.ListOfModelAttributesSet)
    attrs = Any[MOI.ObjectiveSense()]
    if m.objective_function !== nothing
        push!(attrs, MOI.ObjectiveFunction{typeof(m.objective_function)}())
    end
    return attrs
end

# Variables
#
# `PartitionPD` is a `num_rows × num_trucks` matrix flattened column-major
# by `JuMP.build_variable`. Plain `Partition` models reach this constructor
# through MathOptVRP's `PartitionToPartitionPDBridge`, with zero pairs.
_partition_dims(set::MathOptVRP.PartitionPD) =
    (set.num_services + 2 * set.num_pickup_deliveries, set.num_trucks)

function MOI.supports_add_constrained_variables(::Optimizer, ::Type{MathOptVRP.PartitionPD})
    return true
end

function MOI.add_constrained_variables(m::Optimizer, set::MathOptVRP.PartitionPD)
    m.partition === nothing ||
        error("Vroom: only one MathOptVRP.PartitionPD set is supported per model")
    n_rows, n_cols = _partition_dims(set)
    n = n_rows * n_cols
    vars = Vector{MOI.VariableIndex}(undef, n)
    # `JuMP.build_variable(::PartitionPD)` flattens column-major via `vec`, so
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
    ci = MOI.ConstraintIndex{MOI.VectorOfVariables,typeof(set)}(m.next_constraint)
    return vars, ci
end

# `vrptw` declares free `t[i] >= 0` variables (one per truck,
# not part of the `Partition`) purely so `sum(t)` can serve as the
# objective; the bound itself carries no meaning to Vroom and is accepted
# as a no-op. `variable_to_position` intentionally has no entry for these,
# which is how the `TimeWindows` `MOI.add_constraint` method below tells a
# `t` variable apart from a `Partition` node variable.

function MOI.add_variable(m::Optimizer)
    m.next_variable += 1
    return MOI.VariableIndex(m.next_variable)
end

function MOI.supports_add_constrained_variable(
    ::Optimizer,
    ::Type{MOI.GreaterThan{Float64}},
)
    return true
end

function MOI.add_constrained_variable(m::Optimizer, ::MOI.GreaterThan{Float64})
    v = MOI.add_variable(m)
    m.next_constraint += 1
    ci = MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}}(m.next_constraint)
    return v, ci
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VariableIndex},
    ::Type{MOI.GreaterThan{Float64}},
)
    return true
end

function MOI.add_constraint(m::Optimizer, ::MOI.VariableIndex, ::MOI.GreaterThan{Float64})
    m.next_constraint += 1
    return MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}}(
        m.next_constraint,
    )
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
        error("Vroom: `:sum_distances` vector must be `[depot; col; depot]`")
    items[1] isa Real ||
        error("Vroom: depot_start must be a `Real`; got $(typeof(items[1]))")
    items[end] isa Real ||
        error("Vroom: depot_end must be a `Real`; got $(typeof(items[end]))")
    depot_start = round(Int, items[1])
    depot_end = round(Int, items[end])
    depot_start == depot_end || error("Vroom: depot_start != depot_end is not supported")
    depot = depot_start - 1 # MOI node values are one-based; Vroom is zero-based.
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
    return matrix, depot, column::Int
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

# A `PartitionPD` node is either a `Job` (service) or one half of a
# `Shipment` pickup/delivery pair. With zero pairs, every node is a service,
# which is exactly the plain `Partition` case produced by the bridge.

function _jobs_and_shipments(set::MathOptVRP.PartitionPD, customer_locs::Vector{Int})
    ns = set.num_services
    npd = set.num_pickup_deliveries
    jobs = [Job(id = loc, location_index = loc) for loc in customer_locs if loc < ns]
    shipments = [
        Shipment(
            pickup = ShipmentStep(id = ns + k - 1, location_index = ns + k - 1),
            delivery = ShipmentStep(
                id = ns + npd + k - 1,
                location_index = ns + npd + k - 1,
            ),
        ) for k = 1:npd
    ]
    return jobs, shipments
end

# ── TimeWindows constraint parsing ────────────────────────────────────
# `MathOptVRP.TimeWindows{WITHOUT_START_TIME}(travel, earliest, latest,
# service, num_items)` is applied, one per truck, to
# `[route_end; first_node; route...; last_node]` where `route_end` is a
# free variable (see `MOI.add_constrained_variable` above), `route` is one
# column of `Partition` variables (`num_items` of them), and `first_node`
# / `last_node` are one-based indices into `travel`/`earliest`/`latest`/
# `service` for two (possibly distinct) logical copies of the depot.
# `earliest`/`latest` are indexed by customer location, matching how
# `_jobs_and_shipments` sets `Job.id`.

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:Union{MOI.VectorOfVariables,MOI.VectorAffineFunction{Float64}}},
    ::Type{<:MathOptVRP.TimeWindows{MathOptVRP.WITHOUT_START_TIME}},
)
    return true
end

function MOI.add_constraint(
    m::Optimizer,
    f::Union{MOI.VectorOfVariables,MOI.VectorAffineFunction{Float64}},
    s::MathOptVRP.TimeWindows{MathOptVRP.WITHOUT_START_TIME},
)
    items = _normalize_items(f)
    length(items) == MOI.dimension(s) || error(
        "Vroom: TimeWindows constraint expects $(MOI.dimension(s)) entries, got ",
        "$(length(items))",
    )
    items[1] isa MOI.VariableIndex || error(
        "Vroom: TimeWindows entry 1 (`route_end`) must be a variable; got ",
        "$(typeof(items[1]))",
    )
    items[2] isa Real || error("Vroom: TimeWindows entry 2 (first_node) must be a `Real`")
    items[end] isa Real ||
        error("Vroom: TimeWindows last entry (last_node) must be a `Real`")
    # One-based indices into `s.travel`/`s.earliest`/`s.latest`/`s.service`;
    # zero-based below to match Vroom's location indexing.
    depot_start = round(Int, items[2]) - 1
    depot_end = round(Int, items[end]) - 1
    column = nothing
    for k = 3:(length(items)-1)
        it = items[k]
        it isa MOI.VariableIndex ||
            error("Vroom: TimeWindows node entries must be variables; got $(typeof(it))")
        pos = get(m.variable_to_position, it, nothing)
        pos === nothing &&
            error("Vroom: variable $(it) is not part of a registered Partition")
        if column === nothing
            column = pos[2]
        elseif column != pos[2]
            error("Vroom: TimeWindows mixes variables from columns $(column) and $(pos[2])")
        end
    end
    column === nothing &&
        error("Vroom: TimeWindows constraint has no interior node variables")
    haskey(m.time_windows_by_column, column) &&
        error("Vroom: only one TimeWindows constraint per truck column is supported")

    t_var = items[1]::MOI.VariableIndex
    m.time_windows_by_column[column] = _TimeWindowsEntry(t_var, depot_start, depot_end, s)
    m.next_constraint += 1
    return MOI.ConstraintIndex{typeof(f),typeof(s)}(m.next_constraint)
end

# `sum(t)` lowers to a `MOI.ScalarAffineFunction` with a unit-coefficient
# term per truck's `t` variable and a zero constant.
function _time_vars(f::MOI.ScalarAffineFunction{Float64})
    iszero(f.constant) || error("Vroom: VRPTW objective must have a zero constant")
    vars = MOI.VariableIndex[]
    for term in f.terms
        isone(term.coefficient) || error(
            "Vroom: VRPTW objective must be a plain `sum(t)` over unit-coefficient terms",
        )
        push!(vars, term.variable)
    end
    return vars
end

# ── Optimize ─────────────────────────────────────────────────────────

# Parses the `MOI.ScalarNonlinearFunction` `:sum_distances` objective.
# Vehicle `k - 1` was created for the `k`th objective leaf,
# which corresponds to column `vehicle_to_column[k]`.
function _lower_sum_distances(m::Optimizer)
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
    n_clients, _ = _partition_dims(m.partition)
    0 <= depot < n_locations || error("Vroom: depot index $(depot + 1) is out of bounds")
    customer_locs = [loc for loc = 0:(n_locations-1) if loc != depot]
    length(customer_locs) == n_clients || error(
        "Vroom: matrix has $(length(customer_locs)) non-depot rows but Partition has ",
        "$(n_clients) customers",
    )

    vehicles =
        [Vehicle(id = i - 1, start_index = depot, end_index = depot) for i = 1:n_trucks]
    jobs, shipments = _jobs_and_shipments(m.partition, customer_locs)
    return vehicles, jobs, shipments, durations, leaf_columns
end

# Parses the `MathOptVRP.TimeWindows` constraints + `sum(t)` objective
# into the same ingredient shape as `_lower_sum_distances`.
# Vehicles are built directly in column order, so `vehicle_to_column` is
# the identity — unlike the `:sum_distances` path, there's no leaf order
# to permute against.
function _lower_time_windows(m::Optimizer)
    m.objective_function isa MOI.ScalarAffineFunction{Float64} &&
    m.objective_sense == MOI.MIN_SENSE || error(
        "Vroom: TimeWindows constraints require a `MIN_SENSE` linear `sum(t)` objective",
    )
    n_trucks = m.partition.num_trucks
    sort(collect(keys(m.time_windows_by_column))) == collect(1:n_trucks) || error(
        "Vroom: expected one TimeWindows constraint for each of the $(n_trucks) truck ",
        "columns",
    )
    entries = [m.time_windows_by_column[col] for col = 1:n_trucks]

    obj_vars = _time_vars(m.objective_function)
    Set(obj_vars) == Set(e.t_var for e in entries) && length(obj_vars) == n_trucks || error(
        "Vroom: objective must be `sum(t)` over exactly the TimeWindows constraints' ",
        "time variables",
    )

    ref = entries[1].set
    depot_start = entries[1].depot_start
    depot_end = entries[1].depot_end
    for e in entries
        e.set.travel == ref.travel ||
            error("Vroom: per-truck TimeWindows travel matrices must be equal")
        e.set.earliest == ref.earliest ||
            error("Vroom: per-truck TimeWindows `earliest` must agree")
        e.set.latest == ref.latest ||
            error("Vroom: per-truck TimeWindows `latest` must agree")
        e.set.service == ref.service ||
            error("Vroom: per-truck TimeWindows `service` must agree")
        e.depot_start == depot_start ||
            error("Vroom: per-truck TimeWindows first_node (depot_start) must agree")
        e.depot_end == depot_end ||
            error("Vroom: per-truck TimeWindows last_node (depot_end) must agree")
    end

    durations = Matrix{Int}(round.(Int, ref.travel))
    n_locations = size(durations, 1)
    n_locations == size(durations, 2) ||
        error("Vroom: distance matrix must be square; got $(size(durations))")
    n_clients, _ = _partition_dims(m.partition)
    customer_locs =
        [loc for loc = 0:(n_locations-1) if loc != depot_start && loc != depot_end]
    length(customer_locs) == n_clients || error(
        "Vroom: matrix has $(length(customer_locs)) non-depot rows but Partition has ",
        "$(n_clients) customers",
    )
    length(ref.earliest) == n_locations ||
        error("Vroom: TimeWindows `earliest` length must match the travel matrix size")

    vehicles = [
        Vehicle(id = i - 1, start_index = depot_start, end_index = depot_end) for
        i = 1:n_trucks
    ]
    jobs = [
        Job(
            id = loc,
            location_index = loc,
            service = round(Int, ref.service[loc+1]),
            time_windows = [[
                round(Int, ref.earliest[loc+1]),
                round(Int, ref.latest[loc+1]),
            ]],
        ) for loc in customer_locs
    ]
    return vehicles, jobs, Shipment[], durations, collect(1:n_trucks)
end


# Shared solve mechanics: build the `Problem`, call `vroom`, and rebuild
# `m.routes` per *user-defined* truck column (via `vehicle_to_column`)
# plus each truck's total elapsed time (`truck_time`, off the `"end"`
# step's `arrival`). `compute_objective(sol, truck_time)` picks how a
# given path turns that into `m.objective_value` — `sol.summary.cost`
# (distance) for `:sum_distances`, `sum(truck_time)` for `TimeWindows`.
function _solve!(
    m::Optimizer,
    vehicles::Vector{Vehicle},
    jobs::Vector{Job},
    shipments::Vector{Shipment},
    durations::Matrix{Int},
    vehicle_to_column::Vector{Int},
    compute_objective::Function,
)
    n_trucks = length(vehicle_to_column)
    problem = Problem(
        vehicles = vehicles,
        jobs = jobs,
        shipments = shipments,
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

    routes = [Int[] for _ = 1:n_trucks]
    truck_time = zeros(Int, n_trucks)
    for r in sol.routes
        truck_col = vehicle_to_column[r.vehicle+1]
        for step in r.steps
            if step.type in ("job", "pickup", "delivery")
                push!(routes[truck_col], step.location_index + 1)
            elseif step.type == "end"
                truck_time[truck_col] = step.arrival
            end
        end
    end

    m.routes = routes
    m.objective_value = compute_objective(sol, truck_time)
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

function MOI.optimize!(m::Optimizer)
    m.partition !== nothing ||
        error("Vroom: model has no `MathOptVRP.PartitionPD` variables")
    if !isempty(m.time_windows_by_column)
        vehicles, jobs, shipments, durations, vehicle_to_column = _lower_time_windows(m)
        return _solve!(
            m,
            vehicles,
            jobs,
            shipments,
            durations,
            vehicle_to_column,
            (sol, truck_time) -> sum(truck_time),
        )
    end
    m.objective_function !== nothing && m.objective_sense == MOI.MIN_SENSE ||
        error("Vroom: requires a `MIN_SENSE` `:sum_distances` objective")
    vehicles, jobs, shipments, durations, vehicle_to_column = _lower_sum_distances(m)
    return _solve!(
        m,
        vehicles,
        jobs,
        shipments,
        durations,
        vehicle_to_column,
        (sol, _) -> sol.summary.cost,
    )
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

# Map each zero-padded Partition proxy to its position in Vroom's route.
function MOI.get(m::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(m, attr)
    row, column = m.variable_to_position[vi]
    route = m.routes[column]
    return Float64(row <= length(route) ? route[row] : 0)
end
