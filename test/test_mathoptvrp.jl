using JuMP
using Test
using Vroom
using MathOptVRP

# The wrapper has already stored routes per partition column on the
# inner `Vroom.Optimizer`, so `read_routes` just reads them back.
function _vroom_read_routes(model, _nodes)
    return JuMP.unsafe_backend(model).routes
end

@testset "MathOptVRP.test_vrp" begin
    # Vroom only supports the VRP variant here, so we invoke that test
    # directly rather than running `MathOptVRP.Tests.runtests`.
    MathOptVRP.Tests.test_vrp(Vroom.Optimizer; read_routes = _vroom_read_routes)
end
