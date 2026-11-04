# Compilation roots of Vroom, registered in the image-build session after
# loading JuMPyVroom.
import Vroom

# JSON.jl reads the solution through an `invoke` of the generic `make` for this
# struct, which dispatch tracing does not record.
let JSON = Vroom.JSON, StructUtils = Vroom.JSON.StructUtils
    Base.Experimental.entrypoint(
        StructUtils.make,
        (
            JSON.JSONReadStyle{
                JSON.Object{String,Any},
                Nothing,
                StructUtils.DefaultStyle,
                Nothing,
            },
            Type{Vroom.Summary},
            JSON.LazyValue{String},
            NamedTuple{(),Tuple{}},
        ),
    )
end
