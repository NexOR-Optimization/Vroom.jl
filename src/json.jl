import LinearAlgebra
import StructTypes

# Model corresponding to what VROOM needs as JSON input when it is serialized
@kwdef struct Vehicle
    id::Int
    start_index::Int
    end_index::Int
end

@kwdef struct Job
    id::Int
    location_index::Int
    setup::Int = 0
    service::Int = 0
end

@kwdef struct ShipmentStep
    id::Int
    location_index::Int
    setup::Int = 0
    service::Int = 0
end

@kwdef struct Shipment
    amount::Int
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

StructTypes.StructType(::Type{Summary}) = StructTypes.Struct()

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

StructTypes.StructType(::Type{Step}) = StructTypes.Struct()

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

StructTypes.StructType(::Type{Route}) = StructTypes.Struct()

struct Solution
    code::Int
    summary::Summary
    unassigned::Vector{Int}
    routes::Vector{Route}
end

StructTypes.StructType(::Type{Solution}) = StructTypes.Struct()

import JSON3

function vroom(model::Problem)
    display(model.vehicles)
    display(model.jobs)
    display(model.matrices)
    root_dir = joinpath(dirname(dirname(dirname(dirname(@__DIR__)))))
    vroom = joinpath(root_dir, "vroom", "bin", "vroom")
    solution = nothing
    mktemp() do input, input_io
        JSON3.write(input_io, model)
        flush(input_io)
        mktemp() do output, _
            run(`$vroom -i $input -o $output`)
            return solution = JSON3.read(read(output, String), Solution)
        end
    end
    return solution
end
