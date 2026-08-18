import Pkg
import Random

const FERMIONS_TASK_A_MODE = ARGS == ["fermions_task_a"]
const D1_MODE = isempty(ARGS) || ARGS == ["measurements_task_d1"]
const D2_MODE = isempty(ARGS) || ARGS == ["gradientflow_task_d2"]
const REFERENCE_MODE = D1_MODE || D2_MODE
if !(isempty(ARGS) || ARGS == ["reproducible_rng"] || ARGS == ["hmc_trajectory"] || ARGS == ["heatbath_statistics"] || ARGS == ["ildg"] || ARGS == ["wilsonloop_task_b"] || ARGS == ["stout_task_c"] || ARGS == ["measurements_task_d1"] || ARGS == ["gradientflow_task_d2"] || FERMIONS_TASK_A_MODE)
    error("usage: julia --startup-file=no fixtures/generate.jl [reproducible_rng|hmc_trajectory|heatbath_statistics|ildg|wilsonloop_task_b|stout_task_c|measurements_task_d1|gradientflow_task_d2|fermions_task_a]")
end

hex_word(value::UInt64) = "0x" * lpad(string(value, base=16), 16, '0')
json_string_array(values) = "[" * join([string(Char(34), value, Char(34)) for value in values], ", ") * "]"
json_number_array(values) = "[" * join(repr.(values), ", ") * "]"

function generate_reproducible_rng()
    out = joinpath(@__DIR__, "reproducible_rng")
    mkpath(out)
    state = (UInt64(1), UInt64(2), UInt64(3), UInt64(4))

    raw_rng = Random.Xoshiro(state...)
    raw = UInt64[]
    for _ in 1:10
        push!(raw, Random.rand(raw_rng, UInt64))
    end

    normal_rng = Random.Xoshiro(state...)
    normals = Float64[]
    for _ in 1:5
        raw_u1 = Random.rand(normal_rng, UInt64)
        raw_u2 = Random.rand(normal_rng, UInt64)
        u1 = (Float64(raw_u1 >>> 12) + 0.5) * 2.0^-52
        u2 = (Float64(raw_u2 >>> 12) + 0.5) * 2.0^-52
        radius = sqrt(-2.0 * log(u1))
        theta = 2π * u2
        push!(normals, radius * cos(theta))
        push!(normals, radius * sin(theta))
    end

    julia_commit = string(Base.GIT_VERSION_INFO.commit)
    raw_hex = hex_word.(raw)
    normal_bits = hex_word.(reinterpret.(UInt64, normals))
    q = Char(34)
    open(joinpath(out, "metadata.json"), "w") do io
        println(io, "{")
        println(io, "  ", q, "julia_version", q, ": ", q, Base.VERSION, q, ",")
        println(io, "  ", q, "julia_commit", q, ": ", q, julia_commit, q, ",")
        println(io, "  ", q, "julia_source", q, ": {", q, "url", q, ": ", q,
            "https://github.com/JuliaLang/julia/blob/$julia_commit/stdlib/Random/src/Xoshiro.jl", q,
            ", ", q, "revision", q, ": ", q, julia_commit, q, "},")
        println(io, "  ", q, "algorithm", q, ": ", q, "xoshiro256++", q, ",")
        println(io, "  ", q, "rand_xoshiro_version", q, ": ", q, "0.6.0", q, ",")
        println(io, "  ", q, "rand_xoshiro_source", q, ": ", q, "https://docs.rs/rand_xoshiro/0.6.0", q, ",")
        println(io, "  ", q, "state", q, ": [1, 2, 3, 4],")
        println(io, "  ", q, "state_word_order", q, ": ", q,
            "Julia (s0, s1, s2, s3), each word encoded little-endian", q, ",")
        println(io, "  ", q, "state_note", q, ": ", q,
            "Julia s4 is auxiliary splitmix/task-fork state and is not imported", q, ",")
        println(io, "  ", q, "raw_generation", q, ": ", q,
            "explicit scalar loop calling rand(rng, UInt64) once per word; no array or bulk generation", q, ",")
        println(io, "  ", q, "raw_outputs", q, ": ", json_string_array(raw_hex), ",")
        println(io, "  ", q, "uniform_formula", q, ": ", q,
            "u = (Float64(next_u64 >> 12) + 0.5) * 2^-52", q, ",")
        println(io, "  ", q, "box_muller", q, ": {", q, "u_order", q, ": ", q, "u1 then u2", q,
            ", ", q, "pair_order", q, ": ", q, "[r*cos(TAU*u2), r*sin(TAU*u2)]", q,
            ", ", q, "odd_fill_policy", q, ": ", q,
            "fill the cosine result and discard the final sine result", q, "},")
        println(io, "  ", q, "normal_values", q, ": [", join(repr.(normals), ", "), "],")
        println(io, "  ", q, "normal_bits", q, ": ", json_string_array(normal_bits), ",")
        println(io, "  ", q, "normal_comparison_tolerance", q, ": 1e-14")
        println(io, "}")
    end
end

if ARGS == ["reproducible_rng"]
    generate_reproducible_rng()
    exit()
end

const REQUESTED_CHECKOUT = get(ENV, "GAUGEFIELDS_JL_DIR", nothing)
isnothing(REQUESTED_CHECKOUT) && error("set GAUGEFIELDS_JL_DIR to a clean Gaugefields.jl checkout")
const WILSONLOOP_CHECKOUT = get(
    ENV,
    "WILSONLOOP_JL_DIR",
    joinpath(dirname(abspath(REQUESTED_CHECKOUT)), "Wilsonloop.jl"),
)
isdir(WILSONLOOP_CHECKOUT) || error("expected Wilsonloop.jl checkout at $WILSONLOOP_CHECKOUT")
const QCDMEASUREMENTS_CHECKOUT = get(ENV, "QCDMEASUREMENTS_JL_DIR", nothing)
if REFERENCE_MODE
    isempty(get(ENV, "LATTICEQCD_JULIA_PROJECT", "")) &&
        error("set LATTICEQCD_JULIA_PROJECT for the pinned reference project")
end
if D1_MODE
    isnothing(QCDMEASUREMENTS_CHECKOUT) &&
        error("set QCDMEASUREMENTS_JL_DIR for measurements_task_d1/default")
    isdir(QCDMEASUREMENTS_CHECKOUT) ||
        error("expected QCDMeasurements.jl checkout at $QCDMEASUREMENTS_CHECKOUT")
end
const JULIA_PROJECT = get(ENV, "LATTICEQCD_JULIA_PROJECT", "")
const ACTIVE_PROJECT = isempty(JULIA_PROJECT) ? REQUESTED_CHECKOUT : JULIA_PROJECT
const LATTICEDIRACOPERATORS_CHECKOUT = get(
    ENV,
    "LATTICEDIRACOPERATORS_JL_DIR",
    joinpath(dirname(abspath(REQUESTED_CHECKOUT)), "LatticeDiracOperators.jl"),
)
if FERMIONS_TASK_A_MODE
    isdir(LATTICEDIRACOPERATORS_CHECKOUT) ||
        error("expected LatticeDiracOperators.jl checkout at $LATTICEDIRACOPERATORS_CHECKOUT")
end
Pkg.activate(ACTIVE_PROJECT)
using Gaugefields
import Wilsonloop
if D1_MODE
    using QCDMeasurements
end
using NPZ
import Gaugefields.Temporalfields_module: get_temp
using LinearAlgebra
if FERMIONS_TASK_A_MODE
    @eval using LatticeDiracOperators
end

const NC = 3
const BETA = 6.0
const HMC_EPSILON = 0.5
const HMC_DT = 0.125
const HMC_BETA = 5.7
const HMC_STEP_SIZE = 0.01
const HMC_STEPS = 4
const HMC_STATE = (UInt64(1), UInt64(2), UInt64(3), UInt64(4))
const HMC_JULIA_COMMIT = "9e5719970770f4497405a856315c90bef7f74449"
const HEATBATH_JULIA_COMMIT = "9e5719970770f4497405a856315c90bef7f74449"
const HEATBATH_BETAS = (5.5, 5.7, 6.0)
const HEATBATH_SEEDS = (2026081801, 2026081802, 2026081803)
const HEATBATH_BURN_IN = 512
const HEATBATH_BLOCKS = 32
const HEATBATH_SWEEPS_PER_BLOCK = 32
const HEATBATH_MAX_ATTEMPTS = 100_000
const ILDG_JULIA_COMMIT = "9e5719970770f4497405a856315c90bef7f74449"
const ILDG_STABLE_RNG_SEED = 123
const ILDG_LATTICE = (2, 2, 2, 2)
const VERSION = string(Base.pkgversion(Gaugefields))
const CHECKOUT = dirname(dirname(pathof(Gaugefields)))
const COMMIT = readchomp(`git -C $CHECKOUT rev-parse HEAD`)
const DIRTY = read(`git -C $CHECKOUT status --porcelain --untracked-files=all`, String)
isempty(strip(DIRTY)) || error("refusing fixture provenance from dirty Gaugefields.jl checkout: $CHECKOUT")
const WILSONLOOP_VERSION = string(Base.pkgversion(Wilsonloop))
const WILSONLOOP_SOURCE = dirname(dirname(pathof(Wilsonloop)))
const WILSONLOOP_COMMIT = readchomp(`git -C $WILSONLOOP_CHECKOUT rev-parse HEAD`)
const WILSONLOOP_DIRTY = read(
    `git -C $WILSONLOOP_CHECKOUT status --porcelain --untracked-files=all`,
    String,
)
isempty(strip(WILSONLOOP_DIRTY)) || error("refusing fixture provenance from dirty Wilsonloop.jl checkout: $WILSONLOOP_CHECKOUT")
read(joinpath(WILSONLOOP_SOURCE, "src", "Wilsonloop.jl")) ==
    read(joinpath(WILSONLOOP_CHECKOUT, "src", "Wilsonloop.jl")) ||
    error("active Wilsonloop.jl source does not match the pinned checkout")
const QCDMEASUREMENTS_VERSION = D1_MODE ? string(Base.pkgversion(QCDMeasurements)) : ""
const QCDMEASUREMENTS_SOURCE = D1_MODE ? dirname(dirname(pathof(QCDMeasurements))) : ""
const QCDMEASUREMENTS_COMMIT = D1_MODE ? readchomp(`git -C $QCDMEASUREMENTS_CHECKOUT rev-parse HEAD`) : ""
const QCDMEASUREMENTS_DIRTY = D1_MODE ? read(
    `git -C $QCDMEASUREMENTS_CHECKOUT status --porcelain --untracked-files=all`,
    String,
) : ""
if D1_MODE
    isempty(strip(QCDMEASUREMENTS_DIRTY)) ||
        error("refusing fixture provenance from dirty QCDMeasurements.jl checkout: $QCDMEASUREMENTS_CHECKOUT")
    read(joinpath(QCDMEASUREMENTS_SOURCE, "src", "QCDMeasurements.jl")) ==
        read(joinpath(QCDMEASUREMENTS_CHECKOUT, "src", "QCDMeasurements.jl")) ||
        error("active QCDMeasurements.jl source does not match the pinned checkout")
end

if FERMIONS_TASK_A_MODE
    const LATTICEDIRACOPERATORS_VERSION = string(Base.pkgversion(LatticeDiracOperators))
    const LATTICEDIRACOPERATORS_SOURCE = dirname(dirname(pathof(LatticeDiracOperators)))
    const LATTICEDIRACOPERATORS_COMMIT = readchomp(
        `git -C $LATTICEDIRACOPERATORS_CHECKOUT rev-parse HEAD`,
    )
    const LATTICEDIRACOPERATORS_DIRTY = read(
        `git -C $LATTICEDIRACOPERATORS_CHECKOUT status --porcelain --untracked-files=all`,
        String,
    )
    isempty(strip(LATTICEDIRACOPERATORS_DIRTY)) ||
        error("refusing fixture provenance from dirty LatticeDiracOperators.jl checkout: $LATTICEDIRACOPERATORS_CHECKOUT")
    read(joinpath(LATTICEDIRACOPERATORS_SOURCE, "src", "LatticeDiracOperators.jl")) ==
        read(joinpath(LATTICEDIRACOPERATORS_CHECKOUT, "src", "LatticeDiracOperators.jl")) ||
        error("active LatticeDiracOperators.jl source does not match the pinned checkout")
end

function fermions_task_a_links(lattice)
    links = Initialize_Gaugefields(NC, 0, lattice...; condition="cold")
    nx, ny, nz, nt = lattice
    for direction in 1:4, it in 1:nt, iz in 1:nz, iy in 1:ny, ix in 1:nx
        a = 0.017 * direction + 0.031 * (ix - 1) - 0.013 * (iy - 1) +
            0.007 * (iz - 1) + 0.011 * (it - 1)
        b = -0.023 * direction + 0.019 * (ix - 1) + 0.005 * (iy - 1) -
            0.009 * (iz - 1) + 0.003 * (it - 1)
        links[direction].U[:, :, ix, iy, iz, it] .= ComplexF64[
            cis(a) 0 0
            0 cis(b) 0
            0 0 cis(-a - b)
        ]
    end
    return links
end

function fermions_task_a_input(lattice)
    nx, ny, nz, nt = lattice
    input = Array{ComplexF64}(undef, NC, nx, ny, nz, nt, 4)
    for spin in 1:4, it in 1:nt, iz in 1:nz, iy in 1:ny, ix in 1:nx, color in 1:NC
        site = (ix - 1) + nx * ((iy - 1) + ny * ((iz - 1) + nz * (it - 1)))
        real_part = 0.013 * (color + 3 * (spin - 1) + 12 * site)
        imag_part = -0.009 * (color + 2 * (color - 1) + (spin - 1) + 5 * site)
        input[color, ix, iy, iz, it, spin] = ComplexF64(real_part, imag_part)
    end
    return input
end

function fermions_task_a_field(links, input)
    source = Initialize_pseudofermion_fields(links[1], "Wilson"; nowing=true)
    source.f .= input
    return source
end

function generate_fermions_task_a()
    VERSION == "0.7.2" || error("expected Gaugefields.jl v0.7.2, found $VERSION")
    COMMIT == "9e5719970770f4497405a856315c90bef7f74449" ||
        error("expected Gaugefields.jl commit 9e5719970770f4497405a856315c90bef7f74449, found $COMMIT")
    LATTICEDIRACOPERATORS_VERSION == "0.6.4" ||
        error("expected LatticeDiracOperators.jl v0.6.4, found $LATTICEDIRACOPERATORS_VERSION")
    LATTICEDIRACOPERATORS_COMMIT == "bdef628184597815ba3e0cddf2536df767e78a02" ||
        error("expected LatticeDiracOperators.jl commit bdef628184597815ba3e0cddf2536df767e78a02, found $LATTICEDIRACOPERATORS_COMMIT")

    lattice = (2, 2, 2, 2)
    links = fermions_task_a_links(lattice)
    input = fermions_task_a_input(lattice)
    out = joinpath(@__DIR__, "fermions_task_a")
    mkpath(out)
    for direction in 1:4
        NPZ.npzwrite(joinpath(out, "u$(direction - 1).npy"), links[direction].U)
    end
    NPZ.npzwrite(joinpath(out, "input_julia.npy"), input)
    NPZ.npzwrite(joinpath(out, "input_rust.npy"), permutedims(input, (1, 6, 2, 3, 4, 5)))

    cases = ((name="periodic", boundary=[1, 1, 1, 1]), (name="antiperiodic", boundary=[1, 1, 1, -1]))
    for case in cases
        source = fermions_task_a_field(links, input)
        parameters = Dict{String,Any}(
            "Dirac_operator" => "Wilson",
            "κ" => 0.13,
            "r" => 1.0,
            "faster version" => false,
            "verbose_level" => 0,
            "boundarycondition" => Int8.(case.boundary),
        )
        dirac = Dirac_operator(links, source, parameters)
        result = similar(source)
        result_dagger = similar(source)
        result_normal = similar(source)
        mul!(result, dirac, source)
        mul!(result_dagger, adjoint(dirac), source)
        mul!(result_normal, DdagD_operator(links, source, parameters), source)
        for (label, field) in (("d", result), ("ddag", result_dagger), ("ddagd", result_normal))
            NPZ.npzwrite(joinpath(out, "$(label)_$(case.name)_julia.npy"), copy(field.f))
            NPZ.npzwrite(
                joinpath(out, "$(label)_$(case.name)_rust.npy"),
                permutedims(field.f, (1, 6, 2, 3, 4, 5)),
            )
        end
    end

    open(joinpath(out, "metadata.json"), "w") do io
        q = Char(34)
        print(io, "{\n")
        print(io, "  \"schema\": \"fermions_task_a.v1\",\n")
        print(io, "  \"lattice\": [2, 2, 2, 2],\n")
        print(io, "  \"nc\": 3,\n")
        print(io, "  \"components\": 4,\n")
        print(io, "  \"kappa\": 0.13,\n")
        print(io, "  \"r\": 1.0,\n")
        print(io, "  \"boundaries\": {\"periodic\": [1, 1, 1, 1], \"antiperiodic\": [1, 1, 1, -1]},\n")
        print(io, "  \"gaugefields_jl\": {\"package\": \"Gaugefields.jl\", \"version\": \"$VERSION\", \"commit\": \"$COMMIT\", \"clean\": true},\n")
        print(io, "  \"latticediracoperators_jl\": {\"package\": \"LatticeDiracOperators.jl\", \"version\": \"$LATTICEDIRACOPERATORS_VERSION\", \"commit\": \"$LATTICEDIRACOPERATORS_COMMIT\", \"clean\": true},\n")
        print(io, "  \"source_urls\": [\n")
        print(io, "    \"https://github.com/shinaoka/Gaugefields.jl/blob/$COMMIT/src/AbstractGaugefields.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/Gaugefields.jl/blob/$COMMIT/src/4D/nowing/gaugefields_4D_nowing.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/LatticeDiracOperators.jl/blob/$LATTICEDIRACOPERATORS_COMMIT/src/AbstractFermions_4D.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/LatticeDiracOperators.jl/blob/$LATTICEDIRACOPERATORS_COMMIT/src/Diracoperators.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/LatticeDiracOperators.jl/blob/$LATTICEDIRACOPERATORS_COMMIT/src/WilsonFermion/WilsonFermion.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/LatticeDiracOperators.jl/blob/$LATTICEDIRACOPERATORS_COMMIT/src/WilsonFermion/WilsonFermion_4D.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/LatticeDiracOperators.jl/blob/$LATTICEDIRACOPERATORS_COMMIT/src/WilsonFermion/WilsonFermion_4D_nowing.jl\"\n")
        print(io, "  ],\n")
        print(io, "  \"source_functions\": [\"Initialize_Gaugefields\", \"mk_gamma\", \"Wilson_Dirac_operator\", \"shift_fermion\", \"shifted_fermion!\", \"Wx!\", \"Wdagx_noclover!\", \"DdagD_Wilson_operator\", \"LinearAlgebra.mul!\", \"LinearAlgebra.dot\", \"mul_γ5x!\"],\n")
        print(io, "  \"entrypoint_map\": [\n")
        print(io, "    {\"julia\": \"Initialize_Gaugefields\", \"julia_source\": \"Gaugefields.jl/src/AbstractGaugefields.jl\", \"rust\": \"fermions_task_a_links + GaugeLinks::host_view\"},\n")
        print(io, "    {\"julia\": \"mk_gamma\", \"julia_source\": \"src/WilsonFermion/WilsonFermion.jl\", \"rust\": \"wilson.rs::GAMMA + project_spin\"},\n")
        print(io, "    {\"julia\": \"Wilson_Dirac_operator\", \"julia_source\": \"src/WilsonFermion/WilsonFermion.jl\", \"rust\": \"WilsonDirac::with_boundary\"},\n")
        print(io, "    {\"julia\": \"shift_fermion/shifted_fermion!\", \"julia_source\": \"src/WilsonFermion/WilsonFermion_4D_nowing.jl\", \"rust\": \"WilsonDirac::neighbor\"},\n")
        print(io, "    {\"julia\": \"Wx!\", \"julia_source\": \"src/WilsonFermion/WilsonFermion.jl\", \"rust\": \"FermionOperator for WilsonDirac::apply_into\"},\n")
        print(io, "    {\"julia\": \"Wdagx_noclover!\", \"julia_source\": \"src/WilsonFermion/WilsonFermion.jl\", \"rust\": \"FermionOperator for WilsonAdjoint::apply_into\"},\n")
        print(io, "    {\"julia\": \"DdagD_Wilson_operator + LinearAlgebra.mul!\", \"julia_source\": \"src/WilsonFermion/WilsonFermion.jl + src/Diracoperators.jl\", \"rust\": \"NormalOperator::apply_into\"},\n")
        print(io, "    {\"julia\": \"LinearAlgebra.dot\", \"julia_source\": \"src/AbstractFermions_4D.jl\", \"rust\": \"FermionField::inner_product\"},\n")
        print(io, "    {\"julia\": \"mul_γ5x!\", \"julia_source\": \"src/WilsonFermion/WilsonFermion_4D_nowing.jl\", \"rust\": \"FermionField::gamma5\"}\n")
        print(io, "  ],\n")
        print(io, "  \"construction\": \"explicit diagonal SU(3) links and spinor values from fixed formulas; no RNG or global random state\",\n")
        print(io, "  \"layout\": {\"julia_shape\": \"[3,NX,NY,NZ,NT,4]\", \"rust_shape\": \"[3,4,NX,NY,NZ,NT]\", \"julia_input\": \"ComplexF64 Fortran [color,x,y,z,t,spin]\", \"rust_input_and_outputs\": \"Complex64 column-major [color,spin,x,y,z,t]\", \"conversion\": \"permutedims(input, (1, 6, 2, 3, 4, 5)); the same explicit transpose is applied to every output\", \"permutation\": [1, 6, 2, 3, 4, 5], \"site_order\": \"x fastest\"},\n")
        print(io, "  \"gamma\": \"Euclidean chiral basis from mk_gamma; gamma5=diag(-1,-1,+1,+1)\",\n")
        print(io, "  \"files\": [\"u0.npy\", \"u1.npy\", \"u2.npy\", \"u3.npy\", \"input_julia.npy\", \"input_rust.npy\", \"d_periodic_julia.npy\", \"d_periodic_rust.npy\", \"ddag_periodic_julia.npy\", \"ddag_periodic_rust.npy\", \"ddagd_periodic_julia.npy\", \"ddagd_periodic_rust.npy\", \"d_antiperiodic_julia.npy\", \"d_antiperiodic_rust.npy\", \"ddag_antiperiodic_julia.npy\", \"ddag_antiperiodic_rust.npy\", \"ddagd_antiperiodic_julia.npy\", \"ddagd_antiperiodic_rust.npy\"],\n")
        print(io, "  \"comparison\": {\"component_max_abs_tolerance\": 2e-12, \"criterion\": \"maximum absolute complex-component residual over every color, component, and site\"},\n")
        print(io, "  \"generator\": {\"script\": \"fixtures/generate.jl\", \"mode\": \"fermions_task_a\", \"randomness\": \"none\"}\n")
        print(io, "}\n")
    end
end

if FERMIONS_TASK_A_MODE
    generate_fermions_task_a()
    exit()
end

function distinguish_reproducible_directions!(links)
    # Gaugefields.jl deliberately resets StableRNG(123) for each direction.
    # Shift each direction along its matching lattice axis so fixtures detect
    # direction swaps while preserving every site-local SU(3) value.
    for mu in 1:4
        shifts = ntuple(axis -> axis == mu + 2 ? 1 : 0, 6)
        links[mu].U .= circshift(links[mu].U, shifts)
    end
    return links
end

function ildg_be_bytes(value::UInt64)
    return UInt8[
        (value >> 56) & 0xff,
        (value >> 48) & 0xff,
        (value >> 40) & 0xff,
        (value >> 32) & 0xff,
        (value >> 24) & 0xff,
        (value >> 16) & 0xff,
        (value >> 8) & 0xff,
        value & 0xff,
    ]
end

function ildg_append_float64!(payload, value::Float64)
    append!(payload, ildg_be_bytes(reinterpret(UInt64, value)))
end

function ildg_write_header(io, flags::UInt16, payload_length::UInt64, record_type::String)
    bytes = codeunits(record_type)
    0 < length(bytes) < 128 || error("invalid ILDG record type")
    all(byte -> 0x20 <= byte <= 0x7e, bytes) || error("invalid ILDG record type")
    header = zeros(UInt8, 144)
    copyto!(header, 1, UInt8[0x45, 0x67, 0x89, 0xab])
    copyto!(header, 5, UInt8[0x00, 0x01])
    copyto!(header, 7, UInt8[(flags >> 8) & 0xff, flags & 0xff])
    copyto!(header, 9, ildg_be_bytes(payload_length))
    copyto!(header, 17, collect(bytes))
    write(io, header)
end

function ildg_write_record(io, flags::UInt16, record_type::String, payload::Vector{UInt8})
    ildg_write_header(io, flags, UInt64(length(payload)), record_type)
    write(io, payload)
    padding = mod(-length(payload), 8)
    padding == 0 || write(io, zeros(UInt8, padding))
end

function ildg_xml(lattice)
    return collect(codeunits("""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<ildgFormat xmlns=\"http://www.lqcd.org/ildg\">
  <version>1.0</version>
  <field>su3gauge</field>
  <precision>64</precision>
  <lx>$(lattice[1])</lx>
  <ly>$(lattice[2])</ly>
  <lz>$(lattice[3])</lz>
  <lt>$(lattice[4])</lt>
</ildgFormat>
"""))
end

function ildg_binary(links, lattice)
    nx, ny, nz, nt = lattice
    payload = UInt8[]
    sizehint!(payload, 4 * 3 * 3 * 2 * 8 * nx * ny * nz * nt)
    for it in 1:nt, iz in 1:nz, iy in 1:ny, ix in 1:nx, mu in 1:4
        for row in 1:3, column in 1:3
            value = links[mu].U[row, column, ix, iy, iz, it]
            ildg_append_float64!(payload, real(value))
            ildg_append_float64!(payload, imag(value))
        end
    end
    return payload
end

function generate_ildg_fixture()
    VERSION == "0.7.2" || error("expected Gaugefields.jl v0.7.2, found $VERSION")
    COMMIT == ILDG_JULIA_COMMIT || error("expected Gaugefields.jl commit $ILDG_JULIA_COMMIT, found $COMMIT")
    links = Initialize_Gaugefields(
        NC,
        0,
        ILDG_LATTICE...;
        condition="hot",
        randomnumber="Reproducible",
    )
    distinguish_reproducible_directions!(links)
    out = joinpath(@__DIR__, "ildg_task_a")
    mkpath(out)
    for mu in 1:4
        NPZ.npzwrite(joinpath(out, "u$(mu - 1).npy"), links[mu].U)
    end
    xml = ildg_xml(ILDG_LATTICE)
    binary = ildg_binary(links, ILDG_LATTICE)
    open(joinpath(out, "gauge.ildg"), "w") do io
        ildg_write_record(io, UInt16(0x8000), "ildg-format", xml)
        ildg_write_record(io, UInt16(0x4000), "ildg-binary-data", binary)
    end
    open(joinpath(out, "metadata.json"), "w") do io
        print(io, "{\n")
        print(io, "  \"schema\": \"ildg_task_a.v1\",\n")
        print(io, "  \"lattice\": [2, 2, 2, 2],\n")
        print(io, "  \"nc\": 3,\n")
        print(io, "  \"condition\": \"hot\",\n")
        print(io, "  \"randomnumber\": \"Reproducible\",\n")
        print(io, "  \"stable_rng_seed\": ", ILDG_STABLE_RNG_SEED, ",\n")
        print(io, "  \"direction_disambiguation\": \"direction mu is periodically shifted by +1 along lattice axis mu; preserves site-local SU(3) values\",\n")
        print(io, "  \"gaugefields_jl_version\": \"$VERSION\",\n")
        print(io, "  \"gaugefields_jl_commit\": \"$COMMIT\",\n")
        print(io, "  \"source_urls\": [\"https://github.com/shinaoka/Gaugefields.jl/blob/$COMMIT/src/output/ildg_format.jl\", \"https://github.com/shinaoka/Gaugefields.jl/blob/$COMMIT/src/4D/nowing/gaugefields_4D_nowing.jl\"],\n")
        print(io, "  \"source_functions\": [\"Initialize_Gaugefields\", \"save_binarydata layout\", \"load_binarydata! order\"],\n")
        print(io, "  \"writer\": \"independent manual LIME writer; no c-lime and no incomplete save_binarydata! implementation\",\n")
        print(io, "  \"lime\": {\"header_bytes\": 144, \"version\": 1, \"flags\": {\"format_mb\": true, \"binary_me\": true}, \"padding\": \"8-byte zero padding\"},\n")
        print(io, "  \"xml\": {\"version\": \"1.0\", \"field\": \"su3gauge\", \"precision\": 64, \"dimensions\": [\"lx\", \"ly\", \"lz\", \"lt\"]},\n")
        print(io, "  \"binary_layout\": \"big-endian IEEE Float64; t,z,y,x,mu,row,column,real/imag; Julia first color index is Rust row\",\n")
        print(io, "  \"files\": [\"gauge.ildg\", \"u0.npy\", \"u1.npy\", \"u2.npy\", \"u3.npy\"],\n")
        print(io, "  \"readback_script\": \"fixtures/check_ildg_readback.jl\",\n")
        print(io, "  \"readback\": \"pinned Gaugefields.jl load_gaugefield! with explicit dimensions; every component bit-exact against u*.npy\",\n")
        print(io, "  \"comparison\": {\"ildg_input\": \"component bit-exact\", \"field_max_abs_tolerance\": 0.0, \"scalar_tolerance\": 2e-12},\n")
        print(io, "  \"provenance_note\": \"Values come from the pinned Gaugefields.jl hot initializer; this independent manual standards-complete LIME framing is validated against Gaugefields.jl layout and does not claim a c-lime writer.\"\n")
        print(io, "}\n")
    end
end

if ARGS == ["ildg"]
    generate_ildg_fixture()
    exit()
end

const WILSONLOOP_TASK_B_JULIA_COMMIT = "e1a617fdedb19b785f89bdeb13c30e53b20743a7"
const WILSONLOOP_TASK_B_GAUGEFIELDS_COMMIT = "9e5719970770f4497405a856315c90bef7f74449"

function generate_wilsonloop_task_b()
    VERSION == "0.7.2" || error("expected Gaugefields.jl v0.7.2, found $VERSION")
    COMMIT == WILSONLOOP_TASK_B_GAUGEFIELDS_COMMIT ||
        error("expected Gaugefields.jl commit $WILSONLOOP_TASK_B_GAUGEFIELDS_COMMIT, found $COMMIT")
    WILSONLOOP_VERSION == "0.1.5" ||
        error("expected Wilsonloop.jl v0.1.5, found $WILSONLOOP_VERSION")
    WILSONLOOP_COMMIT == WILSONLOOP_TASK_B_JULIA_COMMIT ||
        error("expected Wilsonloop.jl commit $WILSONLOOP_TASK_B_JULIA_COMMIT, found $WILSONLOOP_COMMIT")

    lattice = (2, 2, 2, 2)
    links = Initialize_Gaugefields(NC, 0, lattice...; condition="hot", randomnumber="Reproducible")
    distinguish_reproducible_directions!(links)
    julia_plaquette_coefficient = 0.365
    julia_rectangle_coefficient = -0.155

    gauge_action = GaugeAction(links)
    for mu in 1:3, nu in (mu + 1):4
        plaquette = Wilsonloop.make_plaq(mu, nu)
        rectangle_nu_long = Wilsonloop.Wilsonline([(mu, 1), (nu, 2), (mu, -1), (nu, -2)])
        rectangle_mu_long = Wilsonloop.Wilsonline([(mu, 2), (nu, 1), (mu, -2), (nu, -1)])
        push!(gauge_action, julia_plaquette_coefficient, [plaquette, plaquette'])
        push!(gauge_action, julia_rectangle_coefficient, [rectangle_nu_long, rectangle_nu_long'])
        push!(gauge_action, julia_rectangle_coefficient, [rectangle_mu_long, rectangle_mu_long'])
    end

    force = initialize_TA_Gaugefields(links)
    out = joinpath(@__DIR__, "wilsonloop_task_b")
    mkpath(out)
    for mu in 1:4
        NPZ.npzwrite(joinpath(out, "u$(mu - 1).npy"), links[mu].U)
        derivative = similar(links[1])
        Gaugefields.calc_dSdUμ!(derivative, gauge_action, mu, links)
        NPZ.npzwrite(joinpath(out, "dsdu$(mu - 1).npy"), derivative.U)
        product = similar(links[1])
        mul!(product, links[mu], derivative)
        clear_U!(force[mu])
        Traceless_antihermitian_add!(force[mu], 1.0, product)
        NPZ.npzwrite(joinpath(out, "force_coeff$(mu - 1).npy"), force[mu].a)
    end

    open(joinpath(out, "metadata.json"), "w") do io
        print(io, "{\n")
        print(io, "  \"schema\": \"wilsonloop_task_b.v1\",\n")
        print(io, "  \"lattice\": [2, 2, 2, 2],\n")
        print(io, "  \"nc\": 3,\n")
        print(io, "  \"condition\": \"hot\",\n")
        print(io, "  \"randomnumber\": \"Reproducible\",\n")
        print(io, "  \"direction_disambiguation\": \"direction mu is periodically shifted by +1 along axis mu\",\n")
        print(io, "  \"gaugefields_jl\": {\"version\": \"$VERSION\", \"commit\": \"$COMMIT\", \"clean\": true},\n")
        print(io, "  \"wilsonloop_jl\": {\"version\": \"$WILSONLOOP_VERSION\", \"commit\": \"$WILSONLOOP_COMMIT\", \"clean\": true},\n")
        print(io, "  \"source_urls\": [\n")
        print(io, "    \"https://github.com/shinaoka/Gaugefields.jl/blob/$COMMIT/src/action/GaugeActions.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/Gaugefields.jl/blob/$COMMIT/src/4D/TA_gaugefields_4D_serial.jl\",\n")
        print(io, "    \"https://github.com/akio-tomiya/Wilsonloop.jl/blob/$WILSONLOOP_COMMIT/src/Wilsonloop.jl\"\n")
        print(io, "  ],\n")
        print(io, "  \"source_functions\": [\"GaugeAction\", \"calc_dSdUμ!\", \"Traceless_antihermitian_add!\", \"make_plaq\", \"Wilsonline\"],\n")
        print(io, "  \"planes\": [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]],\n")
        print(io, "  \"per_plane_terms\": [\n")
        print(io, "    {\"name\": \"plaquette\", \"steps_template\": [\"mu\", \"nu\", \"-mu\", \"-nu\"], \"julia_coefficient_f\": 0.365, \"rust_coefficient_c\": 0.73},\n")
        print(io, "    {\"name\": \"rectangle_nu_long\", \"steps_template\": [\"mu\", \"nu\", \"nu\", \"-mu\", \"-nu\", \"-nu\"], \"julia_coefficient_f\": -0.155, \"rust_coefficient_c\": -0.31},\n")
        print(io, "    {\"name\": \"rectangle_mu_long\", \"steps_template\": [\"mu\", \"mu\", \"nu\", \"-mu\", \"-mu\", \"-nu\"], \"julia_coefficient_f\": -0.155, \"rust_coefficient_c\": -0.31}\n")
        print(io, "  ],\n")
        print(io, "  \"expanded_rust_terms\": 18,\n")
        print(io, "  \"coefficient_mapping\": \"Rust c=2*f because Julia inserts f*W and f*W†; Rust evaluates c*sum_x Re tr(W)\",\n")
        print(io, "  \"force_mapping\": \"Julia calc_dSdU is holomorphic: each Rust occurrence uses c/2=f; for U -> exp((i/2)sum(v_a lambda_a)t)U, dS/dt=-sum(force_a v_a)\",\n")
        print(io, "  \"layout\": {\"links\": \"ComplexF64 Fortran [row,column,x,y,z,t]\", \"force\": \"Float64 Fortran [gell_mann_component,x,y,z,t]\", \"site_order\": \"x fastest\"},\n")
        print(io, "  \"files\": [\"u0.npy\", \"u1.npy\", \"u2.npy\", \"u3.npy\", \"dsdu0.npy\", \"dsdu1.npy\", \"dsdu2.npy\", \"dsdu3.npy\", \"force_coeff0.npy\", \"force_coeff1.npy\", \"force_coeff2.npy\", \"force_coeff3.npy\"],\n")
        print(io, "  \"comparison\": {\"force_component_tolerance\": 2e-12, \"derivative_component_tolerance\": 2e-12, \"criterion\": \"maximum absolute residual over every direction/site/component\"},\n")
        print(io, "  \"generator\": {\"script\": \"fixtures/generate.jl\", \"mode\": \"wilsonloop_task_b\", \"oracle_only\": true}\n")
        print(io, "}\n")
    end
end

if ARGS == ["wilsonloop_task_b"]
    generate_wilsonloop_task_b()
    exit()
end

const STOUT_TASK_C_GAUGEFIELDS_COMMIT = "9e5719970770f4497405a856315c90bef7f74449"
const STOUT_TASK_C_RHOS = (0.12, -0.07)
const STOUT_TASK_C_LATTICE = (2, 2, 2, 2)
const STOUT_TASK_C_TOLERANCE = 5e-12

function generate_stout_task_c()
    VERSION == "0.7.2" || error("expected Gaugefields.jl v0.7.2, found $VERSION")
    COMMIT == STOUT_TASK_C_GAUGEFIELDS_COMMIT ||
        error("expected Gaugefields.jl commit $STOUT_TASK_C_GAUGEFIELDS_COMMIT, found $COMMIT")
    lattice = STOUT_TASK_C_LATTICE
    links = Initialize_Gaugefields(NC, 0, lattice...; condition="hot", randomnumber="Reproducible")
    distinguish_reproducible_directions!(links)
    out = joinpath(@__DIR__, "stout_task_c")
    mkpath(out)
    for mu in 1:4
        NPZ.npzwrite(joinpath(out, "u$(mu - 1).npy"), links[mu].U)
    end

    labels = ("plus", "minus")
    for (rho, label) in zip(STOUT_TASK_C_RHOS, labels)
        # This is the pinned Gaugefields.jl one-layer, plaquette-only path:
        # STOUT_Layer -> CovNeuralnet -> calc_smearedU -> forward! -> calc_C!.
        layer = STOUT_Layer(["plaquette"], [rho], links)
        network = CovNeuralnet(links)
        push!(network, layer)
        smeared, _, _ = calc_smearedU(links, network)
        for mu in 1:4
            maximum(abs.(smeared[mu].U .- links[mu].U)) > 1e-8 ||
                error("stout output is trivial for rho=$rho, direction=$mu")
            NPZ.npzwrite(joinpath(out, "stout_$(label)$(mu - 1).npy"), smeared[mu].U)
        end
    end

    open(joinpath(out, "metadata.json"), "w") do io
        print(io, "{\n")
        print(io, "  \"schema\": \"stout_task_c.v1\",\n")
        print(io, "  \"lattice\": [2, 2, 2, 2],\n")
        print(io, "  \"nc\": 3,\n")
        print(io, "  \"condition\": \"hot\",\n")
        print(io, "  \"randomnumber\": \"Reproducible\",\n")
        print(io, "  \"direction_disambiguation\": \"direction mu is periodically shifted by +1 along axis mu\",\n")
        print(io, "  \"gaugefields_jl\": {\"package\": \"Gaugefields.jl\", \"version\": \"$VERSION\", \"commit\": \"$COMMIT\", \"clean\": true},\n")
        print(io, "  \"wilsonloop_jl\": {\"package\": \"Wilsonloop.jl\", \"version\": \"$WILSONLOOP_VERSION\", \"commit\": \"$WILSONLOOP_COMMIT\", \"clean\": true},\n")
        print(io, "  \"source_urls\": [\n")
        print(io, "    \"https://github.com/shinaoka/Gaugefields.jl/blob/$COMMIT/src/smearing/stout_fast.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/Gaugefields.jl/blob/$COMMIT/src/smearing/stout_dataset.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/Gaugefields.jl/blob/$COMMIT/src/smearing/Abstractsmearing.jl\",\n")
        print(io, "    \"https://github.com/akio-tomiya/Wilsonloop.jl/blob/$WILSONLOOP_COMMIT/src/Wilsonloop.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/Gaugefields.jl/blob/$COMMIT/test/HMCstout_test_nowing.jl\"\n")
        print(io, "  ],\n")
        print(io, "  \"source_functions\": [\"STOUT_Layer\", \"CovNeuralnet\", \"calc_smearedU\", \"apply_smearing_U\", \"apply_neuralnet!\", \"forward!\", \"calc_C!\", \"STOUT_dataset\", \"make_Cμ\", \"HMCstout_test_nowing\"],\n")
        print(io, "  \"rhos\": {\"positive\": 0.12, \"negative\": -0.07},\n")
        print(io, "  \"convention\": \"C_mu=rho*positive six-term plaquette staple; Omega=C_mu*U_mu†; Q=TA(Omega); Uprime=exp(Q)*U_mu\",\n")
        print(io, "  \"routine_order\": \"one isotropic plaquette-only layer, all four directions evaluated from the unchanged input before output storage\",\n")
        print(io, "  \"layout\": {\"links\": \"ComplexF64 Fortran [row,column,x,y,z,t]\", \"site_order\": \"x fastest\"},\n")
        print(io, "  \"files\": [\"u0.npy\", \"u1.npy\", \"u2.npy\", \"u3.npy\", \"stout_plus0.npy\", \"stout_plus1.npy\", \"stout_plus2.npy\", \"stout_plus3.npy\", \"stout_minus0.npy\", \"stout_minus1.npy\", \"stout_minus2.npy\", \"stout_minus3.npy\"],\n")
        print(io, "  \"comparison\": {\"field_max_abs_tolerance\": 5e-12, \"criterion\": \"maximum absolute complex-component residual over every direction/site/row/column\"},\n")
        print(io, "  \"update\": \"all links use one unchanged input snapshot\",\n")
        print(io, "  \"generator\": {\"script\": \"fixtures/generate.jl\", \"mode\": \"stout_task_c\", \"oracle_only\": true}\n")
        print(io, "}\n")
    end
end

if ARGS == ["stout_task_c"]
    generate_stout_task_c()
    exit()
end

const MEASUREMENTS_TASK_D1_GAUGEFIELDS_COMMIT = "9e5719970770f4497405a856315c90bef7f74449"
const MEASUREMENTS_TASK_D1_WILSONLOOP_COMMIT = "e1a617fdedb19b785f89bdeb13c30e53b20743a7"
const MEASUREMENTS_TASK_D1_QCDMEASUREMENTS_COMMIT = "9e04c37bbd68712cf7a749ae5aff10eb6aae4566"
const MEASUREMENTS_TASK_D1_LATTICE = (2, 2, 2, 2)
const MEASUREMENTS_TASK_D1_BETA = 5.7
const MEASUREMENTS_TASK_D1_SEED = HEATBATH_SEEDS[2]
const MEASUREMENTS_TASK_D1_TOLERANCE = 2e-12
const MEASUREMENTS_TASK_D1_RUST_STATE = (
    "0x2468ace113579bdf",
    "0x1111222233334445",
    "0x5555666677778889",
    "0x9999aaaabbbbcccd",
)

function d1_measurement_state(U)
    return (
        plaquette=Plaquette_measurement(U),
        polyakov=Polyakov_measurement(U),
        topology=Topological_charge_measurement(U; TC_methods=["clover"]),
    )
end

function d1_observe(measurements, U)
    polyakov = get_value(measure(measurements.polyakov, U))
    topology = get_value(measure(measurements.topology, U))
    return (
        plaquette=get_value(measure(measurements.plaquette, U)),
        polyakov=polyakov,
        q=topology["clover"],
    )
end

function d1_block_summary(blocks)
    mean_value = sum(blocks) / length(blocks)
    variance = sum((value - mean_value)^2 for value in blocks) / (length(blocks) - 1)
    return (
        block_means=blocks,
        mean=mean_value,
        variance=variance,
        standard_error=sqrt(variance / length(blocks)),
    )
end

function d1_heatbath_chain()
    Random.seed!(MEASUREMENTS_TASK_D1_SEED)
    U = Initialize_Gaugefields(NC, 0, MEASUREMENTS_TASK_D1_LATTICE...; condition="cold")
    h = Heatbath(U, MEASUREMENTS_TASK_D1_BETA)
    measurements = d1_measurement_state(U)

    for _ in 1:HEATBATH_BURN_IN
        heatbath!(U, h)
    end

    series = [Float64[] for _ in 1:6]
    for _ in 1:HEATBATH_BLOCKS
        block = [Float64[] for _ in 1:6]
        for _ in 1:HEATBATH_SWEEPS_PER_BLOCK
            heatbath!(U, h)
            observation = d1_observe(measurements, U)
            values = (
                observation.plaquette,
                real(observation.polyakov),
                imag(observation.polyakov),
                abs(observation.polyakov),
                observation.q,
                observation.q^2,
            )
            for (index, value) in enumerate(values)
                push!(block[index], value)
            end
        end
        for index in eachindex(series)
            push!(series[index], sum(block[index]) / HEATBATH_SWEEPS_PER_BLOCK)
        end
    end
    return map(d1_block_summary, series)
end

function d1_write_summary(io, name, summary, indent)
    q = Char(34)
    print(io, indent, q, name, q, ": {\"block_means\": ", json_number_array(summary.block_means))
    print(io, ", \"mean\": ", repr(summary.mean))
    print(io, ", \"variance\": ", repr(summary.variance))
    print(io, ", \"standard_error\": ", repr(summary.standard_error), "}")
end

function d1_write_chain(io, summaries)
    q = Char(34)
    print(io, "    {\n")
    print(io, "      \"beta\": ", repr(MEASUREMENTS_TASK_D1_BETA), ",\n")
    print(io, "      \"julia_seed\": ", MEASUREMENTS_TASK_D1_SEED, ",\n")
    print(io, "      \"measurements\": ", HEATBATH_BLOCKS * HEATBATH_SWEEPS_PER_BLOCK, ",\n")
    print(io, "      \"observables\": {\n")
    names = ("plaquette", "polyakov_real", "polyakov_imag", "polyakov_magnitude", "q", "q_squared")
    for (index, name) in enumerate(names)
        d1_write_summary(io, name, summaries[index], "        ")
        index == length(names) ? print(io, "\n") : print(io, ",\n")
    end
    print(io, "      }\n    }")
end

function d1_representative_path(steps)
    Wilsonloop.Wilsonline([(abs(step), sign(step)) for step in steps], Dim=4)
end

function d1_representative_path_matrix(links, steps)
    loop = d1_representative_path(steps)
    output = similar(links[1])
    temp1 = similar(links[1])
    temp2 = similar(links[1])
    Gaugefields.evaluate_gaugelinks!(output, loop, links, [temp1, temp2])
    return copy(output.U[:, :, 1, 1, 1, 1])
end

function generate_measurements_task_d1()
    VERSION == "0.7.2" || error("expected Gaugefields.jl v0.7.2, found $VERSION")
    COMMIT == MEASUREMENTS_TASK_D1_GAUGEFIELDS_COMMIT ||
        error("expected Gaugefields.jl commit $MEASUREMENTS_TASK_D1_GAUGEFIELDS_COMMIT, found $COMMIT")
    WILSONLOOP_VERSION == "0.1.5" ||
        error("expected Wilsonloop.jl v0.1.5, found $WILSONLOOP_VERSION")
    WILSONLOOP_COMMIT == MEASUREMENTS_TASK_D1_WILSONLOOP_COMMIT ||
        error("expected Wilsonloop.jl commit $MEASUREMENTS_TASK_D1_WILSONLOOP_COMMIT, found $WILSONLOOP_COMMIT")
    QCDMEASUREMENTS_VERSION == "0.2.13" ||
        error("expected QCDMeasurements.jl v0.2.13, found $QCDMEASUREMENTS_VERSION")
    QCDMEASUREMENTS_COMMIT == MEASUREMENTS_TASK_D1_QCDMEASUREMENTS_COMMIT ||
        error("expected QCDMeasurements.jl commit $MEASUREMENTS_TASK_D1_QCDMEASUREMENTS_COMMIT, found $QCDMEASUREMENTS_COMMIT")

    links = Initialize_Gaugefields(
        NC,
        0,
        MEASUREMENTS_TASK_D1_LATTICE...;
        condition="hot",
        randomnumber="Reproducible",
    )
    distinguish_reproducible_directions!(links)
    measurements = d1_measurement_state(links)
    observation = d1_observe(measurements, links)
    direct_polyakov = calculate_Polyakov_loop(links, similar(links[1]), similar(links[1]))
    abs(observation.polyakov - direct_polyakov) <= MEASUREMENTS_TASK_D1_TOLERANCE ||
        error("QCDMeasurements/Gaugefields Polyakov routines disagree")

    out = joinpath(@__DIR__, "measurements_task_d1")
    mkpath(out)
    for mu in 1:4
        NPZ.npzwrite(joinpath(out, "u$(mu - 1).npy"), links[mu].U)
    end

    representative_paths = (
        (name="forward", steps=[1]),
        (name="backward", steps=[-1]),
        (name="open", steps=[1, 2, -3]),
        (name="plaquette", steps=[1, 2, -1, -2]),
        (name="rectangle", steps=[1, 2, 2, -1, -2, -2]),
        (name="clover_right_bottom", steps=[-2, 1, 2, -1]),
    )
    for path in representative_paths
        value = d1_representative_path_matrix(links, path.steps)
        NPZ.npzwrite(joinpath(out, "path_$(path.name).npy"), value)
    end
    summaries = d1_heatbath_chain()

    open(joinpath(out, "metadata.json"), "w") do io
        q = Char(34)
        print(io, "{\n")
        print(io, "  \"nc\": 3,\n")
        print(io, "  \"lattice\": [2, 2, 2, 2],\n")
        print(io, "  \"beta\": ", repr(MEASUREMENTS_TASK_D1_BETA), ",\n")
        print(io, "  \"expected_observables\": {\n")
        print(io, "    \"schema\": \"measurements_task_d1.v1\",\n")
        print(io, "    \"scalar\": {\n")
        print(io, "      \"polyakov_loop\": {\"real\": ", repr(real(observation.polyakov)), ", \"imag\": ", repr(imag(observation.polyakov)), "},\n")
        print(io, "      \"clover_topological_charge\": ", repr(observation.q), ",\n")
        print(io, "      \"clover_topological_charge_squared\": ", repr(observation.q^2), "\n")
        print(io, "    },\n")
        print(io, "    \"representative_paths\": {\n")
        print(io, "      \"origin\": {\"rust_site_index\": 0, \"rust_coordinates\": [0, 0, 0, 0], \"julia_coordinates\": [1, 1, 1, 1]},\n")
        print(io, "      \"source_functions\": [\"Wilsonloop.Wilsonline\", \"Gaugefields.evaluate_gaugelinks!\"],\n")
        print(io, "      \"source_urls\": [\"https://github.com/akio-tomiya/Wilsonloop.jl/blob/", WILSONLOOP_COMMIT, "/src/Wilsonloop.jl\", \"https://github.com/shinaoka/Gaugefields.jl/blob/", COMMIT, "/src/AbstractGaugefields.jl\"],\n")
        print(io, "      \"layout\": {\"matrix\": \"ComplexF64/Complex64; first index is color row, second is color column; Julia [row,column] and Rust Mat3 column-major order\", \"artifact\": \"one NPY ComplexF64/Complex64 [3,3] Fortran-order matrix per path\", \"site\": \"links [row,column,x,y,z,t] with x fastest; output.U[:,:,1,1,1,1] at Julia coordinates [1,1,1,1]\"},\n")
        print(io, "      \"tolerance\": ", repr(MEASUREMENTS_TASK_D1_TOLERANCE), ",\n")
        print(io, "      \"paths\": [\n")
        for (index, path) in enumerate(representative_paths)
            print(io, "        {\"name\": \"", path.name, "\", \"steps\": ", json_number_array(path.steps), ", \"file\": \"path_", path.name, ".npy\"}")
            index == length(representative_paths) ? print(io, "\n") : print(io, ",\n")
        end
        print(io, "      ]\n")
        print(io, "    },\n")
        print(io, "    \"ensemble\": {\n")
        print(io, "      \"schema\": \"measurements_task_d1_ensemble.v1\",\n")
        print(io, "      \"chains\": [\n")
        d1_write_chain(io, summaries)
        print(io, "\n      ],\n")
        print(io, "      \"schedule\": {\"initial_condition\": \"cold\", \"burn_in_sweeps\": ", HEATBATH_BURN_IN,
            ", \"blocks\": ", HEATBATH_BLOCKS, ", \"sweeps_per_block\": ", HEATBATH_SWEEPS_PER_BLOCK,
            ", \"measurements\": ", HEATBATH_BLOCKS * HEATBATH_SWEEPS_PER_BLOCK,
            ", \"measurement\": \"after each measured heatbath! sweep\", \"block_statistic\": \"mean of consecutive per-sweep values\", \"standard_error\": \"sample_stddev(block_means) / sqrt(blocks)\", \"max_attempts\": ", HEATBATH_MAX_ATTEMPTS, "},\n")
        print(io, "      \"comparison\": {\"criterion\": \"abs(mean_rust - mean_julia) <= 6 * sqrt(se_rust^2 + se_julia^2)\", \"sigma_multiplier\": 6.0, \"q_squared_relative_ceiling\": 0.25, \"independent_streams\": true, \"bitwise_trajectory_parity\": false}\n")
        print(io, "    },\n")
        print(io, "    \"provenance\": {\n")
        print(io, "      \"gaugefields_jl\": {\"version\": ", q, VERSION, q, ", \"commit\": ", q, COMMIT, q,
            ", \"source_functions\": [\"Initialize_Gaugefields\", \"calculate_Polyakov_loop\", \"Heatbath\", \"heatbath!\", \"calculate_Plaquette\"], \"source_urls\": [\"https://github.com/shinaoka/Gaugefields.jl/blob/", COMMIT, "/src/AbstractGaugefields.jl\", \"https://github.com/shinaoka/Gaugefields.jl/blob/", COMMIT, "/src/heatbath/heatbathmodule.jl\"]},\n")
        print(io, "      \"wilsonloop_jl\": {\"version\": ", q, WILSONLOOP_VERSION, q, ", \"commit\": ", q, WILSONLOOP_COMMIT, q, ", \"source_url\": \"https://github.com/akio-tomiya/Wilsonloop.jl/blob/", WILSONLOOP_COMMIT, "/src/Wilsonloop.jl\"},\n")
        print(io, "      \"qcdmeasurements_jl\": {\"version\": ", q, QCDMEASUREMENTS_VERSION, q, ", \"commit\": ", q, QCDMEASUREMENTS_COMMIT, q, ", \"source_functions\": [\"Plaquette_measurement\", \"Polyakov_measurement\", \"Topological_charge_measurement\", \"measure\"], \"source_urls\": [\"https://github.com/akio-tomiya/QCDMeasurements.jl/blob/", QCDMEASUREMENTS_COMMIT, "/src/measurements/measure_plaquette.jl\", \"https://github.com/akio-tomiya/QCDMeasurements.jl/blob/", QCDMEASUREMENTS_COMMIT, "/src/measurements/measure_polyakov.jl\", \"https://github.com/akio-tomiya/QCDMeasurements.jl/blob/", QCDMEASUREMENTS_COMMIT, "/src/measurements/measure_topological_charge.jl\"]},\n")
        print(io, "      \"julia\": {\"version\": ", q, Base.VERSION, q, ", \"source_commit\": ", q, Base.GIT_VERSION_INFO.commit, q, "},\n")
        print(io, "      \"scalar_field\": {\"condition\": \"hot\", \"randomnumber\": \"Reproducible\", \"direction_disambiguation\": \"direction mu is periodically shifted by +1 along axis mu\", \"layout\": \"ComplexF64 Fortran [row,column,x,y,z,t], x fastest\"},\n")
        print(io, "      \"rust_rng_state_beta_5_7\": ", json_string_array(MEASUREMENTS_TASK_D1_RUST_STATE), ",\n")
        print(io, "      \"generator\": {\"script\": \"fixtures/generate.jl\", \"mode\": \"measurements_task_d1\", \"oracle_only\": true, \"scalar_tolerance\": ", repr(MEASUREMENTS_TASK_D1_TOLERANCE), "}\n")
        print(io, "    }\n")
        print(io, "  },\n")
        print(io, "  \"gaugefields_jl_version\": ", q, VERSION, q, ",\n")
        print(io, "  \"gaugefields_jl_commit\": ", q, COMMIT, q, "\n")
        print(io, "}\n")
    end
end

if ARGS == ["measurements_task_d1"]
    generate_measurements_task_d1()
    exit()
end

const GRADIENTFLOW_TASK_D2_GAUGEFIELDS_COMMIT = "9e5719970770f4497405a856315c90bef7f74449"
const GRADIENTFLOW_TASK_D2_WILSONLOOP_COMMIT = "e1a617fdedb19b785f89bdeb13c30e53b20743a7"
const GRADIENTFLOW_TASK_D2_LATTICE = (2, 2, 2, 2)
const GRADIENTFLOW_TASK_D2_STEP_SIZE = 0.01
const GRADIENTFLOW_TASK_D2_TOLERANCE = 5e-12

function gradientflow_task_d2_input()
    links = Initialize_Gaugefields(
        NC,
        0,
        GRADIENTFLOW_TASK_D2_LATTICE...;
        condition="hot",
        randomnumber="Reproducible",
    )
    return distinguish_reproducible_directions!(links)
end

function gradientflow_task_d2_wilson_action()
    loops = Wilsonloop.Wilsonline{4}[]
    for mu in 1:3, nu in (mu + 1):4
        push!(loops, Wilsonloop.make_plaq(mu, nu))
    end
    return [loops], [0.5]
end

function gradientflow_task_d2_mixed_action()
    plaquettes = Wilsonloop.Wilsonline{4}[]
    rectangles_nu = Wilsonloop.Wilsonline{4}[]
    rectangles_mu = Wilsonloop.Wilsonline{4}[]
    for mu in 1:3, nu in (mu + 1):4
        push!(plaquettes, Wilsonloop.make_plaq(mu, nu))
        push!(rectangles_nu, Wilsonloop.Wilsonline([(mu, 1), (nu, 2), (mu, -1), (nu, -2)]))
        push!(rectangles_mu, Wilsonloop.Wilsonline([(mu, 2), (nu, 1), (mu, -2), (nu, -1)]))
    end
    return [plaquettes, rectangles_nu, rectangles_mu], [0.365, -0.155, -0.155]
end

function gradientflow_task_d2_output(steps, loops, values)
    links = gradientflow_task_d2_input()
    flow = Gradientflow_general(
        links,
        loops,
        values;
        Nflow=steps,
        eps=GRADIENTFLOW_TASK_D2_STEP_SIZE,
    )
    flow!(links, flow)
    return links
end

function gradientflow_task_d2_write_links(out, prefix, links)
    for mu in 1:4
        NPZ.npzwrite(joinpath(out, "$(prefix)$(mu - 1).npy"), links[mu].U)
    end
end

function generate_gradientflow_task_d2()
    VERSION == "0.7.2" || error("expected Gaugefields.jl v0.7.2, found $VERSION")
    COMMIT == GRADIENTFLOW_TASK_D2_GAUGEFIELDS_COMMIT ||
        error("expected Gaugefields.jl commit $GRADIENTFLOW_TASK_D2_GAUGEFIELDS_COMMIT, found $COMMIT")
    WILSONLOOP_VERSION == "0.1.5" ||
        error("expected Wilsonloop.jl v0.1.5, found $WILSONLOOP_VERSION")
    WILSONLOOP_COMMIT == GRADIENTFLOW_TASK_D2_WILSONLOOP_COMMIT ||
        error("expected Wilsonloop.jl commit $GRADIENTFLOW_TASK_D2_WILSONLOOP_COMMIT, found $WILSONLOOP_COMMIT")

    out = joinpath(@__DIR__, "gradientflow_task_d2")
    mkpath(out)
    input = gradientflow_task_d2_input()
    gradientflow_task_d2_write_links(out, "u", input)

    wilson_loops, wilson_values = gradientflow_task_d2_wilson_action()
    wilson_one = gradientflow_task_d2_output(1, wilson_loops, wilson_values)
    wilson_loops, wilson_values = gradientflow_task_d2_wilson_action()
    wilson_four = gradientflow_task_d2_output(4, wilson_loops, wilson_values)
    mixed_loops, mixed_values = gradientflow_task_d2_mixed_action()
    mixed_one = gradientflow_task_d2_output(1, mixed_loops, mixed_values)

    for (prefix, flowed) in (("flow_one", wilson_one), ("flow_four", wilson_four), ("flow_mixed", mixed_one))
        for mu in 1:4
            maximum(abs.(flowed[mu].U .- input[mu].U)) > 1e-8 ||
                error("gradient flow output is trivial for $prefix, direction=$mu")
        end
        gradientflow_task_d2_write_links(out, prefix, flowed)
    end

    open(joinpath(out, "metadata.json"), "w") do io
        q = Char(34)
        print(io, "{\n")
        print(io, "  \"schema\": \"gradientflow_task_d2.v1\",\n")
        print(io, "  \"lattice\": [2, 2, 2, 2],\n")
        print(io, "  \"nc\": 3,\n")
        print(io, "  \"condition\": \"hot\",\n")
        print(io, "  \"randomnumber\": \"Reproducible\",\n")
        print(io, "  \"direction_disambiguation\": \"direction mu is periodically shifted by +1 along lattice axis mu\",\n")
        print(io, "  \"step_size\": ", repr(GRADIENTFLOW_TASK_D2_STEP_SIZE), ",\n")
        print(io, "  \"field_tolerance\": ", repr(GRADIENTFLOW_TASK_D2_TOLERANCE), ",\n")
        print(io, "  \"gaugefields_jl\": {\"package\": \"Gaugefields.jl\", \"version\": \"$VERSION\", \"commit\": \"$COMMIT\", \"clean\": true},\n")
        print(io, "  \"wilsonloop_jl\": {\"package\": \"Wilsonloop.jl\", \"version\": \"$WILSONLOOP_VERSION\", \"commit\": \"$WILSONLOOP_COMMIT\", \"clean\": true},\n")
        print(io, "  \"source_functions\": [\"Initialize_Gaugefields\", \"Wilsonline\", \"make_plaq\", \"Gradientflow_general\", \"GaugeAction\", \"F_update!\", \"flow!\", \"exp_aF_U!\"],\n")
        print(io, "  \"source_urls\": [\n")
        print(io, "    \"https://github.com/shinaoka/Gaugefields.jl/blob/$COMMIT/src/smearing/gradientflow.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/Gaugefields.jl/blob/$COMMIT/src/action/GaugeActions.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/Gaugefields.jl/blob/$COMMIT/src/AbstractGaugefields.jl\",\n")
        print(io, "    \"https://github.com/akio-tomiya/Wilsonloop.jl/blob/$WILSONLOOP_COMMIT/src/Wilsonloop.jl\"\n")
        print(io, "  ],\n")
        print(io, "  \"actions\": {\n")
        print(io, "    \"wilson\": {\"planes\": [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]], \"julia_f\": 0.5, \"rust_c\": 1.0, \"terms\": 6},\n")
        print(io, "    \"mixed\": {\"planes\": [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]], \"plaquette_julia_f\": 0.365, \"plaquette_rust_c\": 0.73, \"rectangle_julia_f\": -0.155, \"rectangle_rust_c\": -0.31, \"rectangles_per_plane\": 2, \"terms\": 18}\n")
        print(io, "  },\n")
        print(io, "  \"coefficient_mapping\": \"Rust c=2*f because Julia inserts f*W and f*W†; Rust evaluates c*sum_x Re tr(W)\",\n")
        print(io, "  \"force_mapping\": \"Julia calc_dSdU is holomorphic: Rust loop_action_force uses c/2=f per occurrence; dS/dt=-sum(force_a*v_a), and RK3 supplies the negative flow coefficients\",\n")
        print(io, "  \"routine_order\": \"F0; W1=exp(-eps/4 F0)U; F1; W2=exp(eps*(-8/9 F1+17/36 F0))W1; F2; U'=exp(eps*(-3/4 F2+8/9 F1-17/36 F0))W2\",\n")
        print(io, "  \"layout\": {\"links\": \"ComplexF64 Fortran [row,column,x,y,z,t]\", \"site_order\": \"x fastest\"},\n")
        print(io, "  \"files\": [\"u0.npy\", \"u1.npy\", \"u2.npy\", \"u3.npy\", \"flow_one0.npy\", \"flow_one1.npy\", \"flow_one2.npy\", \"flow_one3.npy\", \"flow_four0.npy\", \"flow_four1.npy\", \"flow_four2.npy\", \"flow_four3.npy\", \"flow_mixed0.npy\", \"flow_mixed1.npy\", \"flow_mixed2.npy\", \"flow_mixed3.npy\"],\n")
        print(io, "  \"comparison\": {\"field_max_abs_tolerance\": ", repr(GRADIENTFLOW_TASK_D2_TOLERANCE), ", \"criterion\": \"maximum absolute complex-component residual over every direction/site/row/column\", \"su3_residuals\": \"unitarity and determinant residuals\", \"plaquette\": \"normalized plaquette initial and final are reported by the Rust parity test\"},\n")
        print(io, "  \"generator\": {\"script\": \"fixtures/generate.jl\", \"mode\": \"gradientflow_task_d2\", \"oracle_only\": true, \"routine\": \"pinned Gaugefields.jl Gradientflow_general plus flow!; no Rust algorithm reimplementation\"}\n")
        print(io, "}\n")
    end
end

if ARGS == ["gradientflow_task_d2"]
    generate_gradientflow_task_d2()
    exit()
end

function hmc_open_unit(rng)
    raw = rand(rng, UInt64)
    return (Float64(raw >>> 12) + 0.5) * 2.0^-52
end

function hmc_normal_pair(rng)
    u1 = hmc_open_unit(rng)
    u2 = hmc_open_unit(rng)
    radius = sqrt(-2.0 * log(u1))
    theta = 2π * u2
    return radius * cos(theta), radius * sin(theta)
end

function hmc_fill_momentum!(momentum, rng)
    for field in momentum
        values = vec(field.a)
        index = 1
        while index <= length(values)
            first, second = hmc_normal_pair(rng)
            values[index] = first
            index += 1
            if index <= length(values)
                values[index] = second
                index += 1
            end
        end
    end
end

function hmc_action(U, gauge_action, momentum)
    nc = U[1].NC
    gauge = -evaluate_GaugeAction(gauge_action, U) / nc
    kinetic = momentum * momentum / 2
    return real(gauge + kinetic)
end

function hmc_u_update!(U, momentum, dt, temps)
    temp1, it_temp1 = get_temp(temps)
    temp2, it_temp2 = get_temp(temps)
    expU, it_expU = get_temp(temps)
    W, it_W = get_temp(temps)
    for mu in 1:4
        exptU!(expU, 0.5 * dt, momentum[mu], [temp1, temp2])
        mul!(W, expU, U[mu])
        substitute_U!(U[mu], W)
    end
    unused!(temps, it_temp1)
    unused!(temps, it_temp2)
    unused!(temps, it_expU)
    unused!(temps, it_W)
end

function hmc_p_update!(U, momentum, dt, gauge_action, temps)
    dSdU, it_dSdU = get_temp(temps)
    product, it_product = get_temp(temps)
    for mu in 1:4
        calc_dSdUμ!(dSdU, gauge_action, mu, U)
        mul!(product, U[mu], dSdU)
        Traceless_antihermitian_add!(momentum[mu], -dt / 3.0, product)
    end
    unused!(temps, it_dSdU)
    unused!(temps, it_product)
end

function hmc_trajectory!(U, momentum, gauge_action, step_size, steps, temps)
    for _ in 1:steps
        hmc_u_update!(U, momentum, step_size, temps)
        hmc_p_update!(U, momentum, step_size, gauge_action, temps)
        hmc_u_update!(U, momentum, step_size, temps)
    end
end

function generate_hmc_trajectory()
    VERSION == "0.7.2" || error("expected Gaugefields.jl v0.7.2, found $VERSION")
    COMMIT == HMC_JULIA_COMMIT || error("expected Gaugefields.jl commit $HMC_JULIA_COMMIT, found $COMMIT")
    lattice = (2, 2, 2, 2)
    U = Initialize_Gaugefields(NC, 0, lattice...; condition="cold")
    momentum = initialize_TA_Gaugefields(U)
    rng = Random.Xoshiro(HMC_STATE...)
    hmc_fill_momentum!(momentum, rng)
    initial_momentum = [copy(field.a) for field in momentum]

    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop, plaqloop')
    push!(gauge_action, HMC_BETA / 2, plaqloop)
    temps = Temporalfields(U[1]; num=10)
    initial_hamiltonian = hmc_action(U, gauge_action, momentum)

    proposed = similar(U)
    substitute_U!(proposed, U)
    hmc_trajectory!(proposed, momentum, gauge_action, HMC_STEP_SIZE, HMC_STEPS, temps)
    proposed_hamiltonian = hmc_action(proposed, gauge_action, momentum)
    delta_h = proposed_hamiltonian - initial_hamiltonian
    acceptance_probability = delta_h <= 0.0 ? 1.0 : exp(-delta_h)
    acceptance_uniform = hmc_open_unit(rng)
    accepted = acceptance_uniform <= acceptance_probability
    next_raw_word = rand(rng, UInt64)

    out = joinpath(@__DIR__, "hmc_trajectory")
    mkpath(out)
    for mu in 1:4
        NPZ.npzwrite(joinpath(out, "p_initial$(mu - 1).npy"), initial_momentum[mu])
        NPZ.npzwrite(joinpath(out, "p_final$(mu - 1).npy"), momentum[mu].a)
        NPZ.npzwrite(joinpath(out, "u_proposed$(mu - 1).npy"), proposed[mu].U)
    end
    open(joinpath(out, "metadata.json"), "w") do io
        print(io, "{\n")
        print(io, "  \"lattice\": [2, 2, 2, 2],\n")
        print(io, "  \"nc\": 3,\n")
        print(io, "  \"beta\": ", repr(HMC_BETA), ",\n")
        print(io, "  \"step_size\": ", repr(HMC_STEP_SIZE), ",\n")
        print(io, "  \"steps\": ", HMC_STEPS, ",\n")
        print(io, "  \"initial_rng_state\": [1, 2, 3, 4],\n")
        print(io, "  \"acceptance_uniform\": ", repr(acceptance_uniform), ",\n")
        print(io, "  \"acceptance_uniform_bits\": ", reinterpret(UInt64, acceptance_uniform), ",\n")
        print(io, "  \"next_raw_word\": ", next_raw_word, ",\n")
        print(io, "  \"initial_hamiltonian\": ", repr(initial_hamiltonian), ",\n")
        print(io, "  \"proposed_hamiltonian\": ", repr(proposed_hamiltonian), ",\n")
        print(io, "  \"delta_h\": ", repr(delta_h), ",\n")
        print(io, "  \"acceptance_probability\": ", repr(acceptance_probability), ",\n")
        print(io, "  \"accepted\": ", accepted, ",\n")
        print(io, "  \"array_order\": \"Fortran / Julia column-major; coefficient/site blocks are compact\",\n")
        print(io, "  \"momentum_files\": [\"p_initial0.npy\", \"p_initial1.npy\", \"p_initial2.npy\", \"p_initial3.npy\"],\n")
        print(io, "  \"final_momentum_files\": [\"p_final0.npy\", \"p_final1.npy\", \"p_final2.npy\", \"p_final3.npy\"],\n")
        print(io, "  \"proposed_link_files\": [\"u_proposed0.npy\", \"u_proposed1.npy\", \"u_proposed2.npy\", \"u_proposed3.npy\"],\n")
        print(io, "  \"open_unit_formula\": \"(Float64(next_u64 >> 12) + 0.5) * 2^-52\",\n")
        print(io, "  \"box_muller\": \"uncached pairs, u1 then u2, [r*cos(2*pi*u2), r*sin(2*pi*u2)]\",\n")
        print(io, "  \"trajectory\": \"U <- exp((dt/2)P)U; P <- P - (dt/3)gauge_force(U,beta); U <- exp((dt/2)P)U\",\n")
        print(io, "  \"gaugefields_jl_version\": \"$VERSION\",\n")
        print(io, "  \"gaugefields_jl_commit\": \"$COMMIT\",\n")
        print(io, "  \"source_paths\": {\"hmc\": \"test/HMC_test_nowing.jl\", \"ta\": \"src/TA_Gaugefields.jl; src/4D/TA_gaugefields_4D_serial.jl\"},\n")
        print(io, "  \"comparison_tolerance\": 2e-13,\n")
        print(io, "  \"hamiltonian_tolerance\": 2e-12,\n")
        print(io, "  \"provenance\": \"Generated through Gaugefields.jl's exported field, exptU!, calc_dSdUμ!, Traceless_antihermitian_add!, mul!, and substitute_U! operations, with the HMC_test_nowing temporary-field get_temp seam.\"\n")
        print(io, "}\n")
    end
end

if ARGS == ["hmc_trajectory"]
    generate_hmc_trajectory()
    exit()
end

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

function generate(name, lattice, condition; reproducible=false, write_shifts=false, write_observables=false)
    out = joinpath(@__DIR__, name)
    mkpath(out)
    args = reproducible ? (; condition, randomnumber="Reproducible") : (; condition)
    links = Initialize_Gaugefields(NC, 0, lattice...; args...)
    reproducible && distinguish_reproducible_directions!(links)
    plaquette_sum = calculate_Plaquette(links, similar(links[1]), similar(links[1]))
    normalized_plaquette = plaquette_sum / (6 * links[1].NV * links[1].NC)
    action = -(BETA / links[1].NC) * plaquette_sum
    for mu in 0:3
        NPZ.npzwrite(joinpath(out, "u$(mu).npy"), links[mu + 1].U)
    end
    if write_shifts
        for link_mu in 0:3, axis in 1:4, sign in (-1, 1)
            shifted = shift_U(links[link_mu + 1], sign * axis)
            label = sign == 1 ? "plus" : "minus"
            NPZ.npzwrite(joinpath(out, "u$(link_mu)_shift$(axis - 1)_$(label).npy"), copy(shifted.parent.Ushifted))
        end
    end
    if write_observables
        gauge_action = GaugeAction(links)
        plaqloop = make_loops_fromname("plaquette")
        append!(plaqloop, plaqloop')
        push!(gauge_action, BETA / 2, plaqloop)
        momenta = initialize_TA_Gaugefields(links)
        for mu in 1:4
            staple, temp = similar(links[1]), similar(links[1])
            Gaugefields.construct_staple!(staple, links, mu, temp)
            NPZ.npzwrite(joinpath(out, "measurement_staple$(mu - 1).npy"), staple.U)
            d = similar(links[1])
            Gaugefields.calc_dSdUμ!(d, gauge_action, mu, links)
            NPZ.npzwrite(joinpath(out, "dsdu$(mu - 1).npy"), d.U)
            product = similar(links[1])
            mul!(product, links[mu], d)
            clear_U!(momenta[mu])
            Traceless_antihermitian_add!(momenta[mu], 1.0, product)
            NPZ.npzwrite(joinpath(out, "force_coeff$(mu - 1).npy"), momenta[mu].a)
            clear_U!(momenta[mu])
            Traceless_antihermitian_add!(
                momenta[mu], -HMC_EPSILON * HMC_DT / NC, product)
            NPZ.npzwrite(joinpath(out, "momentum_delta$(mu - 1).npy"), momenta[mu].a)
        end
    end
    open(joinpath(out, "metadata.json"), "w") do io
        print(io, "{\n  \"nc\": 3,\n")
        print(io, "  \"lattice\": [", join(lattice, ", "), "],\n")
        print(io, "  \"beta\": $BETA,\n")
        print(io, "  \"expected_observables\": {\"plaquette_sum\": ", repr(plaquette_sum),
              ", \"normalized_plaquette\": ", repr(normalized_plaquette),
              ", \"wilson_action\": ", repr(action), "},\n")
        print(io, "  \"gaugefields_jl_version\": \"$VERSION\",\n")
        print(io, "  \"gaugefields_jl_commit\": \"$COMMIT\",\n")
        print(io, "  \"reference_bits\": ")
        json_complex_arrays(io, links)
        print(io, "\n}\n")
    end
end

function generate_exp_ta()
    out = joinpath(@__DIR__, "exp_ta")
    mkpath(out)
    cases = [
        (name="zero", coefficients=zeros(8), t=0.75, branch="zero"),
        (name="random_a", coefficients=[0.31, -0.27, 0.19, 0.41, -0.13, 0.23, -0.37, 0.29], t=0.7, branch="analytic"),
        (name="random_b", coefficients=[-0.17, 0.43, -0.11, 0.07, 0.33, -0.39, 0.21, -0.25], t=-0.45, branch="analytic"),
        (name="balanced_pair", coefficients=[1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], t=1.0, branch="fallback"),
        (name="balanced_octet", coefficients=[0.4, -0.3, 0.2, -0.1, -0.2, 0.1, -0.4, 0.3], t=0.6, branch="analytic"),
        (name="exact_degenerate", coefficients=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], t=0.5, branch="fallback"),
        (name="near_below", coefficients=[0.0, 0.0, 1.0, 1e-13, 0.0, 0.0, 0.0, 0.0], t=0.5, branch="fallback"),
        (name="near_above", coefficients=[0.0, 0.0, 1.0, 1e-11, 0.0, 0.0, 0.0, 0.0], t=0.5, branch="analytic"),
    ]
    links = Initialize_Gaugefields(NC, 0, 1, 1, 1, 1; condition="cold")
    momentum = initialize_TA_Gaugefields(links)[1]
    result = similar(links[1])
    temps = [similar(links[1]), similar(links[1])]
    expected = Array{ComplexF64}(undef, 3, 3, length(cases))
    for (index, case) in enumerate(cases)
        momentum.a[:, 1, 1, 1, 1] .= case.coefficients
        if startswith(case.name, "balanced_")
            c1, c2, c3, c4, c5, c6, c7, c8 = 0.5 .* case.t .* case.coefficients
            r3 = sqrt(3.0)
            v = ComplexF64[
                c3+c8/r3 c1-im*c2 c4-im*c5;
                c1+im*c2 -c3+c8/r3 c6-im*c7;
                c4+im*c5 c6+im*c7 -2c8/r3
            ]
            if case.name == "balanced_pair"
                args = Float64[]
                for row in 1:3, column in 1:3
                    push!(args, real(v[row, column]), imag(v[row, column]))
                end
                e = Gaugefields.AbstractGaugefields_module.exp_T4(args...)
                expected[:, :, index] .= ComplexF64[
                    e[1] e[2] e[3]; e[4] e[5] e[6]; e[7] e[8] e[9]
                ]
            else
                expected[:, :, index] .= exp(im * v)
            end
        else
            exptU!(result, case.t, momentum, temps)
            expected[:, :, index] .= result.U[:, :, 1, 1, 1, 1]
        end
    end
    NPZ.npzwrite(joinpath(out, "expected.npy"), expected)
    open(joinpath(out, "metadata.json"), "w") do io
        print(io, "{\n  \"gaugefields_jl_commit\": \"$COMMIT\",\n")
        print(io, "  \"gaugefields_jl_version\": \"$VERSION\",\n")
        print(io, "  \"source_function\": \"exptU!\",\n")
        print(io, "  \"source_file\": \"src/4D/TA_gaugefields_4D_serial.jl\",\n")
        print(io, "  \"fallback_predicate\": \"nrm2_k < 1e-24\",\n")
        print(io, "  \"balanced_oracle\": \"pinned exp_T4 fallback or LinearAlgebra.exp after the generator convention; guards exptU! csum cancellation\",\n")
        print(io, "  \"cases\": [\n")
        for (index, case) in enumerate(cases)
            index > 1 && print(io, ",\n")
            print(io, "    {\"name\": \"$(case.name)\", \"coefficients\": [")
            print(io, join(repr.(case.coefficients), ", "))
            print(io, "], \"t\": $(repr(case.t)), \"branch\": \"$(case.branch)\"}")
        end
        print(io, "\n  ]\n}\n")
    end
end

function generate_normalize_su3()
    out = joinpath(@__DIR__, "normalize_su3")
    mkpath(out)
    random_base = exp(im * ComplexF64[
        0.31 0.17-0.23im -0.11+0.07im;
        0.17+0.23im -0.19 0.29-0.13im;
        -0.11-0.07im 0.29+0.13im -0.12
    ])
    random_perturbation = ComplexF64[
        0.006+0.002im -0.004+0.003im 0.001-0.005im;
        -0.003-0.001im 0.005-0.002im 0.004+0.001im;
        0.002+0.004im -0.001+0.002im -0.006+0.003im
    ]
    cases = [
        (name="identity", matrix=Matrix{ComplexF64}(I, 3, 3)),
        (name="deterministic_random_perturbation", matrix=random_base + random_perturbation),
        (name="controlled_drift", matrix=ComplexF64[
            1.02+0.01im  0.02+0.01im -0.04+0.02im;
            0.03-0.02im  0.97-0.03im  0.01-0.02im;
           -0.01+0.04im  0.05+0.01im  1.01+0.03im
        ]),
    ]
    inputs = Array{ComplexF64}(undef, 3, 3, length(cases))
    expected = similar(inputs)
    field = Initialize_Gaugefields(NC, 0, 1, 1, 1, 1; condition="cold")[1]
    for (index, case) in enumerate(cases)
        inputs[:, :, index] .= case.matrix
        field.U[:, :, 1, 1, 1, 1] .= case.matrix
        Gaugefields.AbstractGaugefields_module.normalize_U!(field)
        expected[:, :, index] .= field.U[:, :, 1, 1, 1, 1]
    end
    NPZ.npzwrite(joinpath(out, "input.npy"), inputs)
    NPZ.npzwrite(joinpath(out, "expected.npy"), expected)
    open(joinpath(out, "metadata.json"), "w") do io
        print(io, "{\n  \"gaugefields_jl_commit\": \"$COMMIT\",\n")
        print(io, "  \"gaugefields_jl_version\": \"$VERSION\",\n")
        print(io, "  \"source_function\": \"normalize_U!\",\n")
        print(io, "  \"source_file\": \"src/4D/nowing/gaugefields_4D_nowing.jl\",\n")
        print(io, "  \"lattice\": [1, 1, 1, 1],\n")
        print(io, "  \"cases\": [", join(["\"$(case.name)\"" for case in cases], ", "), "]\n}\n")
    end
end

function heatbath_normalized_plaquette(U, temp1, temp2)
    return real(calculate_Plaquette(U, temp1, temp2)) / (6 * U[1].NV * U[1].NC)
end

function heatbath_chain(beta, seed)
    Random.seed!(seed)
    U = Initialize_Gaugefields(NC, 0, 2, 2, 2, 2; condition="cold")
    h = Heatbath(U, beta)
    temp1 = similar(U[1])
    temp2 = similar(U[1])

    for _ in 1:HEATBATH_BURN_IN
        heatbath!(U, h)
    end

    block_means = Float64[]
    for _ in 1:HEATBATH_BLOCKS
        block_sum = 0.0
        for _ in 1:HEATBATH_SWEEPS_PER_BLOCK
            heatbath!(U, h)
            block_sum += heatbath_normalized_plaquette(U, temp1, temp2)
        end
        push!(block_means, block_sum / HEATBATH_SWEEPS_PER_BLOCK)
    end
    mean = sum(block_means) / HEATBATH_BLOCKS
    sample_variance = sum((value - mean)^2 for value in block_means) / (HEATBATH_BLOCKS - 1)
    standard_error = sqrt(sample_variance / HEATBATH_BLOCKS)
    return block_means, mean, standard_error
end

function generate_heatbath_statistics()
    VERSION == "0.7.2" || error("expected Gaugefields.jl v0.7.2, found $VERSION")
    COMMIT == HEATBATH_JULIA_COMMIT || error("expected Gaugefields.jl commit $HEATBATH_JULIA_COMMIT, found $COMMIT")
    out = joinpath(@__DIR__, "heatbath_statistics")
    mkpath(out)
    chains = [heatbath_chain(beta, seed) for (beta, seed) in zip(HEATBATH_BETAS, HEATBATH_SEEDS)]

    open(joinpath(out, "metadata.json"), "w") do io
        print(io, "{\n")
        print(io, "  \"schema\": \"heatbath_statistics.v1\",\n")
        print(io, "  \"nc\": 3,\n")
        print(io, "  \"lattice\": [2, 2, 2, 2],\n")
        print(io, "  \"chains\": [\n")
        for (index, ((block_means, mean, standard_error), beta, seed)) in enumerate(zip(chains, HEATBATH_BETAS, HEATBATH_SEEDS))
            index > 1 && print(io, ",\n")
            print(io, "    {\n")
            print(io, "      \"beta\": ", repr(beta), ",\n")
            print(io, "      \"julia_seed\": ", seed, ",\n")
            print(io, "      \"measurements\": ", HEATBATH_BLOCKS * HEATBATH_SWEEPS_PER_BLOCK, ",\n")
            print(io, "      \"block_means\": ", json_number_array(block_means), ",\n")
            print(io, "      \"mean\": ", repr(mean), ",\n")
            print(io, "      \"standard_error\": ", repr(standard_error), "\n")
            print(io, "    }")
        end
        print(io, "\n  ],\n")
        print(io, "  \"schedule\": {\n")
        print(io, "    \"initial_condition\": \"cold\",\n")
        print(io, "    \"burn_in_sweeps\": ", HEATBATH_BURN_IN, ",\n")
        print(io, "    \"blocks\": ", HEATBATH_BLOCKS, ",\n")
        print(io, "    \"sweeps_per_block\": ", HEATBATH_SWEEPS_PER_BLOCK, ",\n")
        print(io, "    \"measurements\": ", HEATBATH_BLOCKS * HEATBATH_SWEEPS_PER_BLOCK, ",\n")
        print(io, "    \"measurement\": \"after each measured heatbath! sweep\",\n")
        print(io, "    \"block_statistic\": \"mean of consecutive plaquette measurements\",\n")
        print(io, "    \"standard_error\": \"sample_stddev(block_means) / sqrt(blocks)\",\n")
        print(io, "    \"max_attempts\": ", HEATBATH_MAX_ATTEMPTS, ",\n")
        print(io, "    \"sweep\": \"Gaugefields.jl Heatbath/heatbath! direction and even-odd schedule\"\n")
        print(io, "  },\n")
        print(io, "  \"comparison\": {\n")
        print(io, "    \"criterion\": \"abs(mean_rust - mean_julia) <= 6 * sqrt(se_rust^2 + se_julia^2)\",\n")
        print(io, "    \"sigma_multiplier\": 6.0,\n")
        print(io, "    \"rust_max_attempts\": ", HEATBATH_MAX_ATTEMPTS, ",\n")
        print(io, "    \"independent_streams\": true,\n")
        print(io, "    \"bitwise_trajectory_parity\": false\n")
        print(io, "  },\n")
        print(io, "  \"gaugefields_jl\": {\n")
        print(io, "    \"package\": \"Gaugefields.jl\",\n")
        print(io, "    \"version\": \"$VERSION\",\n")
        print(io, "    \"commit\": \"$COMMIT\",\n")
        print(io, "    \"clean_tracked_worktree\": true,\n")
        print(io, "    \"source_paths\": [\"src/heatbath/heatbathmodule.jl\", \"src/AbstractGaugefields.jl\", \"src/4D/nowing/gaugefields_4D_nowing.jl\"],\n")
        print(io, "    \"source_urls\": [\"https://github.com/shinaoka/Gaugefields.jl/blob/$COMMIT/src/heatbath/heatbathmodule.jl\", \"https://github.com/shinaoka/Gaugefields.jl/blob/$COMMIT/src/AbstractGaugefields.jl\", \"https://github.com/shinaoka/Gaugefields.jl/blob/$COMMIT/src/4D/nowing/gaugefields_4D_nowing.jl\"],\n")
        print(io, "    \"operations\": [\"Heatbath\", \"heatbath!\", \"calculate_Plaquette\"],\n")
        print(io, "    \"iteration_max\": \"Heatbath constructor default ITERATION_MAX=10^5\"\n")
        print(io, "  },\n")
        print(io, "  \"julia\": {\n")
        print(io, "    \"version\": \"", Base.VERSION, "\",\n")
        print(io, "    \"source_commit\": \"", Base.GIT_VERSION_INFO.commit, "\",\n")
        print(io, "    \"rng\": \"Random.seed!(seed) on Julia's default task-local RNG\"\n")
        print(io, "  },\n")
        print(io, "  \"normalization\": \"calculate_Plaquette(U, temp1, temp2) / (6 * NV * NC)\",\n")
        print(io, "  \"deliberate_corrections\": [\n")
        print(io, "    \"Rust omits the four unused preliminary draws before the Julia rejection loop\",\n")
        print(io, "    \"Rust normalizes projected SU(2) matrices by sqrt(abs2(alpha) + abs2(beta))\",\n")
        print(io, "    \"Rust rejects zero or singular staples and non-finite intermediates\",\n")
        print(io, "    \"Rust maps xoshiro words to open uniforms instead of permitting rand() == 0\",\n")
        print(io, "    \"Rust takes an explicit ReproducibleRng instead of Julia's global RNG\"\n")
        print(io, "  ],\n")
        print(io, "  \"generator\": {\n")
        print(io, "    \"script\": \"fixtures/generate.jl\",\n")
        print(io, "    \"mode\": \"heatbath_statistics\",\n")
        print(io, "    \"reimplementation\": false\n")
        print(io, "  }\n")
        print(io, "}\n")
    end
end

if ARGS == ["heatbath_statistics"]
    generate_heatbath_statistics()
    exit()
end

generate_reproducible_rng()
generate_hmc_trajectory()
generate("cold_1x1x1x1", (1, 1, 1, 1), "cold")
generate("random_2x2x2x2", (2, 2, 2, 2), "hot"; reproducible=true, write_observables=true)
generate("random_4x4x4x4", (4, 4, 4, 4), "hot"; reproducible=true, write_observables=true)
generate("shifts_3x2x4x5", (3, 2, 4, 5), "hot"; reproducible=true, write_shifts=true)
generate_exp_ta()
generate_normalize_su3()
generate_heatbath_statistics()
generate_ildg_fixture()
generate_wilsonloop_task_b()
generate_stout_task_c()
generate_measurements_task_d1()
generate_gradientflow_task_d2()
