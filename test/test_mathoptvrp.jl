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


# A direct model has no bridges: Vroom supports `Partition` and `Permutation`.
@testset "direct model" begin
    M = [0 3 4 5 2; 3 0 2 6 4; 4 2 0 3 5; 5 6 3 0 3; 2 4 5 3 0]
    model = direct_model(Vroom.Optimizer())
    set_silent(model)
    @variable(model, nodes[1:4, 1:2] in MathOptVRP.Partition(4, 2))
    @objective(
        model,
        Min,
        sum(MathOptVRP.op_sum_distances(M, vcat(5, nodes[:, i], 5)) for i = 1:2),
    )
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) == 13
    tour = direct_model(Vroom.Optimizer())
    set_silent(tour)
    @variable(tour, x[1:4] in MathOptVRP.Permutation(4))
    @objective(tour, Min, MathOptVRP.op_sum_distances(M, [5; x; 5]))
    optimize!(tour)
    @test termination_status(tour) == MOI.OPTIMAL
    @test sort(round.(Int, value.(x))) == 1:4
    # The route end of a time window is the total time of its truck.
    timed = direct_model(Vroom.Optimizer())
    set_silent(timed)
    @variable(timed, t[1:2] >= 0)
    @variable(timed, y[1:4, 1:2] in MathOptVRP.Partition(4, 2))
    order = [1, 2, 3, 4, 5, 5]
    for i = 1:2
        @constraint(
            timed,
            [t[i]; 5; y[:, i]; 6] in MathOptVRP.TimeWindows{MathOptVRP.WITHOUT_START_TIME}(
                M[order, order],
                zeros(Int, 6),
                fill(100, 6),
                [1, 1, 1, 1, 0, 0],
                4,
            )
        )
    end
    @objective(timed, Min, sum(t))
    optimize!(timed)
    @test termination_status(timed) == MOI.OPTIMAL
    @test sum(value.(t)) ≈ objective_value(timed)
    @test all(>=(0), value.(t)) && maximum(value.(t)) > 0
end
