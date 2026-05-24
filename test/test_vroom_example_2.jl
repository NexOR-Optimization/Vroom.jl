using Test
import Vroom

# vroom/docs/example_2.json
function _test_sol(sol)
    @test sol.code == 0
    @test sol.summary.cost == 5461
    @test sol.summary.routes == 1
    @test sol.summary.unassigned == 0
    @test sol.summary.setup == 0
    @test sol.summary.service == 0
    @test sol.summary.duration == 5461
    @test sol.summary.waiting_time == 0
    @test sol.summary.priority == 0
    @test isempty(sol.summary.violations)
    @test isempty(sol.unassigned)
    route = sol.routes[]
    @test route.vehicle == 0
    @test route.cost == 5461
    @test route.setup == 0
    @test route.service == 0
    @test route.duration == 5461
    @test route.waiting_time == 0
    @test route.priority == 0
    @test length(route.steps) == 4
    @test [step.location_index for step in route.steps] == [0, 2, 3, 1]
    @test isempty(route.violations)
end

perm = [1, 4, 2, 3]
costs = [
    0 2104 197 1299
    2103 0 2255 3152
    197 2256 0 1102
    1299 3153 1102 0
][perm, perm]

# vroom/docs/example_2.json
@testset "Example 2 raw" begin
    problem = Vroom.Problem(
        vehicles = [Vroom.Vehicle(id = 0, start_index = 0, end_index = 1)],
        jobs = [
            Vroom.Job(id = 1414, location_index = 2),
            Vroom.Job(id = 1515, location_index = 3),
        ],
        shipments = Vroom.Shipment[],
        matrices = Vroom.DurationMatrices(car = Vroom.DurationMatrix(costs)),
    )
    _test_sol(Vroom.vroom(problem))
end

@testset "Example 2 from model" begin
    model = Vroom.Model(2, [1], [2], costs)
    problem = Vroom.Problem(model)
    _test_sol(Vroom.vroom(problem))
end;
