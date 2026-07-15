using JuMP
using Test
using Vroom
import MathOptInterface as MOI
using MathOptVRP

# The wrapper has already stored routes per partition column on the
# inner `Vroom.Optimizer`, so `read_routes` just reads them back.
function _vroom_read_routes(model, _nodes)
    return JuMP.unsafe_backend(model).routes
end

# `Vroom.Optimizer` only supports `MathOptVRP.Partition`, not `List`, so
# `test_tsp` (which uses `List`) needs `MathOptVRP.ListToPartitionBridge`
# registered on the optimizer it gets from JuMP. This factory
# instantiates a bridged `Vroom.Optimizer` and registers the extra bridge
# directly on it before handing it back.
function _vroom_tsp_optimizer()
    optimizer = MOI.instantiate(Vroom.Optimizer; with_bridge_type = Float64)
    MOI.Bridges.add_bridge(optimizer, MathOptVRP.ListToPartitionBridge{Float64})
    return optimizer
end

@testset "MathOptVRP.test_tsp" begin
    MathOptVRP.Tests.test_tsp(_vroom_tsp_optimizer; read_routes = _vroom_read_routes)
end

@testset "MathOptVRP.test_vrp" begin
    # Vroom only supports the VRP variant here, so we invoke that test
    # directly rather than running `MathOptVRP.Tests.runtests`.
    MathOptVRP.Tests.test_vrp(Vroom.Optimizer; read_routes = _vroom_read_routes)
end

@testset "MathOptVRP.test_vrppd" begin
    MathOptVRP.Tests.test_vrppd(Vroom.Optimizer; read_routes = _vroom_read_routes)
end
