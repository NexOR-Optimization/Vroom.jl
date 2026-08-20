using JuMP
using Test
using Vroom
import MathOptInterface as MOI
using MathOptVRP

@testset "$test" for test in [
    MathOptVRP.test_trp,
    MathOptVRP.test_vrp,
]
    test(Vroom.Optimizer)
end

@testset "MathOptVRP.test_vrppd" begin
    MathOptVRP.Tests.test_vrppd(Vroom.Optimizer; read_routes = _vroom_read_routes)
end
