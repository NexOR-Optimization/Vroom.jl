using JuMP
using Test
using Vroom
import MathOptInterface as MOI
using MathOptVRP

@testset "MathOptVRP" begin
    optimizer = MOI.instantiate(Vroom.Optimizer; with_bridge_type = Float64)
    MathOptVRP.Bridges.add_all_bridges(optimizer)
    MathOptVRP.Tests.test_tsp(optimizer)
    MathOptVRP.Tests.test_vrp(Vroom.Optimizer)
end

@testset "MathOptVRP.test_vrppd" begin
    MathOptVRP.Tests.test_vrppd(Vroom.Optimizer; read_routes = _vroom_read_routes)
end
