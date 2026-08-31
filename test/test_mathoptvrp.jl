using JuMP
using Test
using Vroom
import MathOptInterface as MOI
using MathOptVRP

@testset "$test" for test in [
    MathOptVRP.Tests.test_tsp,
    MathOptVRP.Tests.test_vrp,
    MathOptVRP.Tests.test_vrppd,
]
    test(Vroom.Optimizer)
end
