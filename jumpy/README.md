# JuMPy backend of Vroom

This directory is the recipe of the compiled Vroom backend of
[JuMPy](https://github.com/NexOR-Optimization/JuMPy): a Python package
`jumpy-opt-vroom` providing `import jumpy.vroom`, which needs neither Julia nor
Python code specific to Vroom. It solves the routing models of
[MathOptVRP](https://github.com/NexOR-Optimization/MathOptVRP.jl) with the
`vroom` executable of vroom_jll:

```python
import jumpy.vroom as jp

VRP = jp.MathOptVRP
model = jp.Model()
nodes = model.variables((4, 2), VRP.Partition(4, 2))  # 4 customers, 2 trucks
model.objective = jp.minimize(sum(
    VRP.op_sum_distances(distances, jp.vcat(5, nodes[:, j], 5)) for j in range(2)
))
model.optimize()
```

- [`src/JuMPyVroom.jl`](src/JuMPyVroom.jl) defines the C entry points with
  `JuMPy.@entrypoints Vroom.Optimizer`. The sets of `jp.MathOptVRP` are those
  that `Vroom.Optimizer` supports, and its operators those of MathOptVRP.
- [`workload.jl`](workload.jl) solves routing models through these entry points;
  it is traced to find the methods that the compiled library must retain.
- [`roots.jl`](roots.jl) retains the methods that tracing cannot find.

Build the wheel with JuMPy's build script, which requires Julia, a C compiler
and JuliaC, and test it:

```bash
python path/to/JuMPy/julia/build_backend.py jumpy --wheel dist
pip install jumpy-opt dist/jumpy_opt_vroom-*.whl pytest
pytest jumpy/test_jumpy_vroom.py
```
