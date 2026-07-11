using Gaugefields
using NPZ

const NC = 3
const LATTICE = (1, 1, 1, 1)
const BETA = 6.0
const OUT = joinpath(@__DIR__, "cold_1x1x1x1")

mkpath(OUT)
links = Initialize_Gaugefields(NC, 0, LATTICE..., condition="cold")
for mu in 0:3
    NPZ.npzwrite(joinpath(OUT, "u$(mu).npy"), links[mu + 1].U)
end

version = string(Base.pkgversion(Gaugefields))
checkout = dirname(dirname(pathof(Gaugefields)))
commit = readchomp(`git -C $checkout rev-parse HEAD`)
open(joinpath(OUT, "metadata.json"), "w") do io
    print(io, "{\n")
    print(io, "  \"nc\": 3,\n")
    print(io, "  \"lattice\": [1, 1, 1, 1],\n")
    print(io, "  \"beta\": 6.0,\n")
    print(io, "  \"expected_observables\": {\"plaquette\": 1.0},\n")
    print(io, "  \"gaugefields_jl_version\": \"$version\",\n")
    print(io, "  \"gaugefields_jl_commit\": \"$commit\"\n")
    print(io, "}\n")
end
