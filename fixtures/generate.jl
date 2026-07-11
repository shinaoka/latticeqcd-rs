using Gaugefields
using NPZ

const NC = 3
const BETA = 6.0
const VERSION = string(Base.pkgversion(Gaugefields))
const CHECKOUT = dirname(dirname(pathof(Gaugefields)))
const COMMIT = readchomp(`git -C $CHECKOUT rev-parse HEAD`)

function json_complex_arrays(io, links)
    print(io, "[")
    for mu in eachindex(links)
        mu > 1 && print(io, ",")
        print(io, "[")
        for (i, value) in enumerate(vec(links[mu].U))
            i > 1 && print(io, ",")
            print(io, "[", reinterpret(UInt64, real(value)), ",", reinterpret(UInt64, imag(value)), "]")
        end
        print(io, "]")
    end
    print(io, "]")
end

function generate(name, lattice, condition; reproducible=false)
    out = joinpath(@__DIR__, name)
    mkpath(out)
    args = reproducible ? (; condition, randomnumber="Reproducible") : (; condition)
    links = Initialize_Gaugefields(NC, 0, lattice...; args...)
    if reproducible
        # Gaugefields.jl deliberately resets StableRNG(123) for each direction.
        # Shift each direction along its matching lattice axis so the fixture
        # detects direction swaps while preserving every site-local SU(3) value.
        for mu in 1:4
            shifts = ntuple(axis -> axis == mu + 2 ? 1 : 0, 6)
            links[mu].U .= circshift(links[mu].U, shifts)
        end
    end
    for mu in 0:3
        NPZ.npzwrite(joinpath(out, "u$(mu).npy"), links[mu + 1].U)
    end
    open(joinpath(out, "metadata.json"), "w") do io
        print(io, "{\n  \"nc\": 3,\n")
        print(io, "  \"lattice\": [", join(lattice, ", "), "],\n")
        print(io, "  \"beta\": $BETA,\n")
        print(io, "  \"expected_observables\": {},\n")
        print(io, "  \"gaugefields_jl_version\": \"$VERSION\",\n")
        print(io, "  \"gaugefields_jl_commit\": \"$COMMIT\",\n")
        print(io, "  \"reference_bits\": ")
        json_complex_arrays(io, links)
        print(io, "\n}\n")
    end
end

generate("cold_1x1x1x1", (1, 1, 1, 1), "cold")
generate("random_2x2x2x2", (2, 2, 2, 2), "hot"; reproducible=true)
