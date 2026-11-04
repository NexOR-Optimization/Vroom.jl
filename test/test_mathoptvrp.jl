using JuMP
using Test
using Vroom
import MathOptInterface as MOI
using MathOptVRP

@testset "$test" for test in [
    MathOptVRP.Tests.test_tsp,
    MathOptVRP.Tests.test_vrp,
    MathOptVRP.Tests.test_vrppd,
    MathOptVRP.Tests.test_vrptw,
    MathOptVRP.Tests.test_cvrp,
]
    test(Vroom.Optimizer)
end
