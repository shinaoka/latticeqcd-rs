using Gaugefields
using NPZ

length(ARGS) == 2 || error("usage: julia --startup-file=no --project=/path/to/Gaugefields.jl fixtures/check_ildg_readback.jl FILE.ildg FIXTURE_DIR")

const LATTICE = (2, 2, 2, 2)
const NC = 3
const GAUGEFIELDS_VERSION = "0.7.2"
const GAUGEFIELDS_COMMIT = "9e5719970770f4497405a856315c90bef7f74449"
const ildg_path = abspath(ARGS[1])
const fixture_dir = abspath(ARGS[2])

string(Base.pkgversion(Gaugefields)) == GAUGEFIELDS_VERSION ||
    error("expected Gaugefields.jl v$GAUGEFIELDS_VERSION")
checkout = dirname(dirname(pathof(Gaugefields)))
readchomp(`git -C $checkout rev-parse HEAD`) == GAUGEFIELDS_COMMIT ||
    error("expected Gaugefields.jl commit $GAUGEFIELDS_COMMIT")

ildg = Gaugefields.ILDG(ildg_path)
tmpdir = mktempdir()
try
    links = Gaugefields.Initialize_Gaugefields(NC, 0, LATTICE...; condition="cold")
    Gaugefields.load_gaugefield!(
        links,
        1,
        ildg,
        LATTICE,
        NC;
        tmpfilename=joinpath(tmpdir, "binary"),
    )
    for mu in 0:3
        actual = links[mu + 1].U
        expected = NPZ.npzread(joinpath(fixture_dir, "u$(mu).npy"))
        size(actual) == (3, 3, 2, 2, 2, 2) || error("wrong shape for direction $mu")
        reinterpret(UInt64, vec(actual)) == reinterpret(UInt64, vec(expected)) ||
            error("component mismatch for direction $mu")
    end
finally
    rm(tmpdir; recursive=true, force=true)
end

println("ILDG Julia readback: 4 directions, 144 ComplexF64 (288 real) components each, bit-exact")
