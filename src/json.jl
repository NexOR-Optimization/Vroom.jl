import LinearAlgebra
import JSON

# Model corresponding to what VROOM needs as JSON input when it is serialized
@kwdef struct Vehicle
    id::Int
    start_index::Int
    end_index::Int
    capacity::Vector{Int} = Int[]
end

function JSON.lower(v::Vehicle)
    nt = (id = v.id, start_index = v.start_index, end_index = v.end_index)
    isempty(v.capacity) && return nt
    return merge(nt, (capacity = v.capacity,))
end

@kwdef struct Job
    id::Int
    location_index::Int
    setup::Int = 0
    service::Int = 0
    time_windows::Vector{Vector{Int}} = Vector{Int}[]
    delivery::Vector{Int} = Int[]
    pickup::Vector{Int} = Int[]
end

# VROOM rejects an explicit empty `time_windows` array (it errors with
# "Invalid time_windows array for object"), so omit the key entirely when
# there are no windows (or amounts) rather than serializing `[]`.
function JSON.lower(job::Job)
    nt = (
        id = job.id,
        location_index = job.location_index,
        setup = job.setup,
        service = job.service,
    )
    isempty(job.time_windows) || (nt = merge(nt, (time_windows = job.time_windows,)))
    isempty(job.delivery) || (nt = merge(nt, (delivery = job.delivery,)))
    isempty(job.pickup) || (nt = merge(nt, (pickup = job.pickup,)))
    return nt
end

@kwdef struct ShipmentStep
    id::Int
    location_index::Int
    setup::Int = 0
    service::Int = 0
end

@kwdef struct Shipment
    amount::Vector{Int} = Int[]
    pickup::ShipmentStep
    delivery::ShipmentStep
end

# Only `Int` is supported unfortunately
struct DurationMatrix
    durations::Vector{Vector{Int}}
end

function DurationMatrix(durations::Matrix{Int})
    return DurationMatrix(eachrow(durations))
end

@kwdef struct DurationMatrices
    car::DurationMatrix
end

@kwdef struct Problem
    vehicles::Vector{Vehicle}
    jobs::Vector{Job}
    shipments::Vector{Shipment}
    matrices::DurationMatrices
end

function checked_c_convert(i, I)
    @assert i in I "Invalid index $i : not in $I"
    return i - 1
end

# Solution
struct Summary
    cost::Int
    routes::Int
    unassigned::Int
    setup::Int
    service::Int
    duration::Int
    waiting_time::Int
    priority::Int
    violations::Vector{Int}
end

struct Step
    type::String
    location_index::Int
    setup::Int
    service::Int
    waiting_time::Int
    arrival::Int
    duration::Int
    violations::Vector{Int}
end

struct Route
    vehicle::Int
    cost::Int
    setup::Int
    service::Int
    duration::Int
    waiting_time::Int
    priority::Int
    steps::Vector{Step}
    violations::Vector{Int}
end

struct Solution
    code::Int
    summary::Summary
    unassigned::Vector{Int}
    routes::Vector{Route}
end

# The functions of JLL packages are compiled with `infer = false`, which a
# trimmed (juliac) image cannot run. So build the command of `vroom_jll.vroom()`
# from the paths set by `vroom_jll.__init__` instead, prepending the library
# path of its dependencies (e.g. OpenSSL) as `JLLWrappers.adjust_ENV!` does.
function _vroom_command()
    libpath = vroom_jll.LIBPATH[]
    current = get(ENV, JLLWrappers.LIBPATH_env, "")
    if !isempty(current)
        libpath = string(libpath, JLLWrappers.pathsep, current)
    end
    return addenv(Cmd([vroom_jll.vroom_path]), JLLWrappers.LIBPATH_env => libpath)
end

# Pipe the problem to the executable and read the solution back, without
# temporary files. The standard error is captured for the error message,
# so that the redirections do not depend on those of the Julia process.
function vroom(model::Problem)
    input = IOBuffer(JSON.json(model))
    errors = IOBuffer()
    output = try
        read(pipeline(_vroom_command(); stdin = input, stderr = errors), String)
    catch err
        err isa ProcessFailedException || rethrow()
        error("vroom failed: ", String(take!(errors)))
    end
    return JSON.parse(output, Solution)
end
