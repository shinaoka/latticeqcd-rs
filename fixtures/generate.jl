import Pkg
import Random

const FERMIONS_TASK_A_MODE = ARGS == ["fermions_task_a"]
const FERMIONS_TASK_B_MODE = ARGS == ["fermions_task_b"]
const FERMIONS_TASK_C_MODE = ARGS == ["fermions_task_c"]
const FERMIONS_TASK_D_MODE = ARGS == ["fermions_task_d"]
const FERMIONS_TASK_E_MODE = ARGS == ["fermions_task_e"]
const FERMIONS_TASK_E_ENSEMBLE_MODE = ARGS == ["fermions_task_e_ensemble"]
const FERMIONS_MODE = FERMIONS_TASK_A_MODE || FERMIONS_TASK_B_MODE || FERMIONS_TASK_C_MODE || FERMIONS_TASK_D_MODE || FERMIONS_TASK_E_MODE || FERMIONS_TASK_E_ENSEMBLE_MODE
const D1_MODE = isempty(ARGS) || ARGS == ["measurements_task_d1"]
const D2_MODE = isempty(ARGS) || ARGS == ["gradientflow_task_d2"]
const REFERENCE_MODE = D1_MODE || D2_MODE
if !(isempty(ARGS) || ARGS == ["reproducible_rng"] || ARGS == ["hmc_trajectory"] || ARGS == ["heatbath_statistics"] || ARGS == ["ildg"] || ARGS == ["wilsonloop_task_b"] || ARGS == ["stout_task_c"] || ARGS == ["measurements_task_d1"] || ARGS == ["gradientflow_task_d2"] || FERMIONS_TASK_A_MODE || FERMIONS_TASK_B_MODE || FERMIONS_TASK_C_MODE || FERMIONS_TASK_D_MODE || FERMIONS_TASK_E_MODE || FERMIONS_TASK_E_ENSEMBLE_MODE)
    error("usage: julia --startup-file=no fixtures/generate.jl [reproducible_rng|hmc_trajectory|heatbath_statistics|ildg|wilsonloop_task_b|stout_task_c|measurements_task_d1|gradientflow_task_d2|fermions_task_a|fermions_task_b|fermions_task_c|fermions_task_d|fermions_task_e|fermions_task_e_ensemble]")
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
if FERMIONS_MODE
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
if FERMIONS_MODE
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

if FERMIONS_MODE
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

const FERMIONS_TASK_B_EPS = 1.0e-20
const FERMIONS_TASK_B_MAXSTEPS = 2_000

function fermions_task_b_rhs(lattice)
    nx, ny, nz, nt = lattice
    rhs = Array{ComplexF64}(undef, NC, nx, ny, nz, nt, 4)
    for spin in 1:4, it in 1:nt, iz in 1:nz, iy in 1:ny, ix in 1:nx, color in 1:NC
        site = (ix - 1) + nx * ((iy - 1) + ny * ((iz - 1) + nz * (it - 1)))
        flat = (color - 1) + NC * ((spin - 1) + 4 * site)
        rhs[color, ix, iy, iz, it, spin] = ComplexF64(
            0.017 * (flat + 1),
            -0.011 * (2 * flat + 3),
        )
    end
    return rhs
end

function fermions_task_b_guess(lattice, name)
    nx, ny, nz, nt = lattice
    guess = zeros(ComplexF64, NC, nx, ny, nz, nt, 4)
    name == "zero" && return guess
    name == "nonzero" || error("unknown Task B guess $name")
    for spin in 1:4, it in 1:nt, iz in 1:nz, iy in 1:ny, ix in 1:nx, color in 1:NC
        site = (ix - 1) + nx * ((iy - 1) + ny * ((iz - 1) + nz * (it - 1)))
        flat = (color - 1) + NC * ((spin - 1) + 4 * site)
        guess[color, ix, iy, iz, it, spin] = ComplexF64(-0.0009 * flat, 0.0013 * flat)
    end
    return guess
end

function fermions_task_b_field(links, values)
    field = Initialize_pseudofermion_fields(links[1], "Wilson"; nowing=true)
    field.f .= values
    return field
end

function fermions_task_b_true_residual_squared(operator, solution, rhs)
    applied = similar(rhs)
    mul!(applied, operator, solution)
    return sum(abs2, vec(rhs.f) .- vec(applied.f))
end

function fermions_task_b_cg_diagnostics(initial, operator, rhs)
    x = similar(initial)
    x.f .= initial.f
    res = similar(rhs)
    res.f .= rhs.f
    temp = similar(rhs)
    mul!(temp, operator, x)
    LatticeDiracOperators.Dirac_operators.add!(res, -1, temp)
    initial_residual_squared = real(dot(res, res))
    if initial_residual_squared < FERMIONS_TASK_B_EPS
        return (
            method="cg",
            iterations=0,
            recursive_residual_squared=initial_residual_squared,
            initial_residual_squared=initial_residual_squared,
            convergence_branch="initial_residual",
            restart_count=0,
        )
    end
    p = similar(res)
    p.f .= res.f
    q = similar(res)
    c1 = dot(p, p)
    for iterations in 1:FERMIONS_TASK_B_MAXSTEPS
        mul!(q, operator, p)
        alpha = c1 / dot(p, q)
        LatticeDiracOperators.Dirac_operators.add!(x, alpha, p)
        LatticeDiracOperators.Dirac_operators.add!(res, -alpha, q)
        c3 = dot(res, res)
        recursive_residual_squared = real(c3)
        if recursive_residual_squared < FERMIONS_TASK_B_EPS
            return (
                method="cg",
                iterations,
                recursive_residual_squared,
                initial_residual_squared,
                convergence_branch="updated_residual",
                restart_count=0,
            )
        end
        beta = c3 / c1
        c1 = c3
        LatticeDiracOperators.Dirac_operators.add!(beta, p, 1, res)
    end
    error("Task B Julia CG diagnostic replay exhausted")
end

function generate_fermions_task_b()
    VERSION == "0.7.2" || error("expected Gaugefields.jl v0.7.2, found $VERSION")
    COMMIT == "9e5719970770f4497405a856315c90bef7f74449" ||
        error("expected Gaugefields.jl commit 9e5719970770f4497405a856315c90bef7f74449")
    LATTICEDIRACOPERATORS_VERSION == "0.6.4" ||
        error("expected LatticeDiracOperators.jl v0.6.4, found $LATTICEDIRACOPERATORS_VERSION")
    LATTICEDIRACOPERATORS_COMMIT == "bdef628184597815ba3e0cddf2536df767e78a02" ||
        error("expected LatticeDiracOperators.jl commit bdef628184597815ba3e0cddf2536df767e78a02, found $LATTICEDIRACOPERATORS_COMMIT")

    lattice = (2, 2, 2, 2)
    links = fermions_task_a_links(lattice)
    rhs_values = fermions_task_b_rhs(lattice)
    guesses = Dict(name => fermions_task_b_guess(lattice, name) for name in ("zero", "nonzero"))
    parameters = Dict{String,Any}(
        "Dirac_operator" => "Wilson",
        "κ" => 0.13,
        "r" => 1.0,
        "faster version" => false,
        "verbose_level" => 0,
        "boundarycondition" => Int8[1, 1, 1, -1],
        "method_CG" => "cg",
        "eps_CG" => FERMIONS_TASK_B_EPS,
        "MaxCGstep" => FERMIONS_TASK_B_MAXSTEPS,
    )
    rhs = fermions_task_b_field(links, rhs_values)
    dirac = Dirac_operator(links, rhs, parameters)
    normal = DdagD_operator(links, rhs, parameters)
    out = joinpath(@__DIR__, "fermions_task_b")
    mkpath(out)
    for direction in 1:4
        NPZ.npzwrite(joinpath(out, "u$(direction - 1).npy"), links[direction].U)
    end
    NPZ.npzwrite(joinpath(out, "rhs_julia.npy"), rhs_values)
    NPZ.npzwrite(joinpath(out, "rhs_rust.npy"), permutedims(rhs_values, (1, 6, 2, 3, 4, 5)))
    for name in ("zero", "nonzero")
        NPZ.npzwrite(joinpath(out, "guess_$(name)_julia.npy"), guesses[name])
        NPZ.npzwrite(
            joinpath(out, "guess_$(name)_rust.npy"),
            permutedims(guesses[name], (1, 6, 2, 3, 4, 5)),
        )
    end

    cases = Dict{String,Any}()
    for (method, operator, solver) in (
        ("cg", normal, LatticeDiracOperators.Dirac_operators.cg),
        ("bicgstab", dirac, LatticeDiracOperators.Dirac_operators.bicgstab),
    )
        for guess_name in ("zero", "nonzero")
            case_name = "$(method)_$(guess_name)"
            initial = fermions_task_b_field(links, guesses[guess_name])
            solution = fermions_task_b_field(links, guesses[guess_name])
            if method == "cg"
                solver(
                    solution,
                    operator,
                    rhs;
                    eps=FERMIONS_TASK_B_EPS,
                    maxsteps=FERMIONS_TASK_B_MAXSTEPS,
                    verbose=Verbose_print(0),
                )
                diagnostics = fermions_task_b_cg_diagnostics(initial, operator, rhs)
            else
                diagnostics = solver(
                    solution,
                    operator,
                    rhs;
                    eps=FERMIONS_TASK_B_EPS,
                    maxsteps=FERMIONS_TASK_B_MAXSTEPS,
                    verbose=Verbose_print(0),
                )
            end
            true_residual_squared = fermions_task_b_true_residual_squared(operator, solution, rhs)
            initial_residual_squared = fermions_task_b_true_residual_squared(operator, initial, rhs)
            NPZ.npzwrite(joinpath(out, "$(case_name)_solution_julia.npy"), solution.f)
            NPZ.npzwrite(
                joinpath(out, "$(case_name)_solution_rust.npy"),
                permutedims(solution.f, (1, 6, 2, 3, 4, 5)),
            )
            cases[case_name] = (
                method=String(diagnostics.method),
                guess=guess_name,
                operator=method == "cg" ? "DdagD" : "D",
                iterations=diagnostics.iterations,
                recursive_residual_squared=diagnostics.recursive_residual_squared,
                initial_residual_squared=initial_residual_squared,
                true_residual_squared=true_residual_squared,
                tolerance=FERMIONS_TASK_B_EPS,
                maximum_iterations=FERMIONS_TASK_B_MAXSTEPS,
                restart_count=diagnostics.restart_count,
                convergence_branch=String(diagnostics.convergence_branch),
            )
        end
    end

    open(joinpath(out, "metadata.json"), "w") do io
        q = Char(34)
        print(io, "{\n")
        print(io, "  \"schema\": \"fermions_task_b.v1\",\n")
        print(io, "  \"lattice\": [2, 2, 2, 2],\n")
        print(io, "  \"nc\": 3,\n")
        print(io, "  \"components\": 4,\n")
        print(io, "  \"kappa\": 0.13,\n")
        print(io, "  \"r\": 1.0,\n")
        print(io, "  \"boundaries\": [1, 1, 1, -1],\n")
        print(io, "  \"solver_parameters\": {\"tolerance\": ", repr(FERMIONS_TASK_B_EPS),
            ", \"max_iterations\": ", FERMIONS_TASK_B_MAXSTEPS,
            ", \"julia_operator_keys\": [\"Dirac_operator\", \"κ\", \"r\", \"faster version\", \"verbose_level\", \"boundarycondition\", \"method_CG\", \"eps_CG\", \"MaxCGstep\"], \"julia_solver_keywords\": [\"eps\", \"maxsteps\", \"verbose\"]},\n")
        print(io, "  \"gaugefields_jl\": {\"package\": \"Gaugefields.jl\", \"version\": \"$VERSION\", \"commit\": \"$COMMIT\", \"clean\": true},\n")
        print(io, "  \"latticediracoperators_jl\": {\"package\": \"LatticeDiracOperators.jl\", \"version\": \"$LATTICEDIRACOPERATORS_VERSION\", \"commit\": \"$LATTICEDIRACOPERATORS_COMMIT\", \"clean\": true},\n")
        print(io, "  \"source_urls\": [\n")
        print(io, "    \"https://github.com/shinaoka/LatticeDiracOperators.jl/blob/$LATTICEDIRACOPERATORS_COMMIT/src/cgmethods.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/LatticeDiracOperators.jl/blob/$LATTICEDIRACOPERATORS_COMMIT/src/Diracoperators.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/LatticeDiracOperators.jl/blob/$LATTICEDIRACOPERATORS_COMMIT/src/WilsonFermion/WilsonFermion.jl\"\n")
        print(io, "  ],\n")
        print(io, "  \"source_functions\": [\"cg\", \"bicgstab\", \"DdagD_operator\", \"LinearAlgebra.mul!\", \"LinearAlgebra.dot\"],\n")
        print(io, "  \"entrypoint_map\": [\n")
        print(io, "    {\"julia\": \"Dirac_operators.cg\", \"julia_source\": \"src/cgmethods.jl:768-868\", \"rust\": \"conjugate_gradient\"},\n")
        print(io, "    {\"julia\": \"Dirac_operators.bicgstab\", \"julia_source\": \"src/cgmethods.jl:157-310\", \"rust\": \"bicgstab\"},\n")
        print(io, "    {\"julia\": \"DdagD_operator\", \"julia_source\": \"src/Diracoperators.jl:151-169\", \"rust\": \"NormalOperator\"},\n")
        print(io, "    {\"julia\": \"LinearAlgebra.mul!\", \"julia_source\": \"src/Diracoperators.jl:415-430\", \"rust\": \"FermionOperator::apply_into\"},\n")
        print(io, "    {\"julia\": \"LinearAlgebra.dot\", \"julia_source\": \"src/cgmethods.jl:20-48\", \"rust\": \"FermionField::inner_product + checked algebra\"}\n")
        print(io, "  ],\n")
        print(io, "  \"layout\": {\"julia_shape\": \"[3,NX,NY,NZ,NT,4]\", \"rust_shape\": \"[3,4,NX,NY,NZ,NT]\", \"conversion\": \"permutedims(array, (1, 6, 2, 3, 4, 5))\", \"permutation\": [1, 6, 2, 3, 4, 5], \"site_order\": \"x fastest\"},\n")
        print(io, "  \"construction\": \"explicit diagonal SU(3) links, rhs, and zero/nonzero guesses from fixed formulas; no RNG or global state\",\n")
        print(io, "  \"cases\": {")
        case_names = sort(collect(keys(cases)))
        for (index, case_name) in enumerate(case_names)
            index > 1 && print(io, ",")
            case = cases[case_name]
            print(io, "\n    ", q, case_name, q, ": {\"method\": ", q, case.method, q,
                ", \"guess\": ", q, case.guess, q, ", \"operator\": ", q, case.operator, q,
                ", \"iterations\": ", case.iterations,
                ", \"recursive_residual_squared\": ", repr(case.recursive_residual_squared),
                ", \"initial_residual_squared\": ", repr(case.initial_residual_squared),
                ", \"true_residual_squared\": ", repr(case.true_residual_squared),
                ", \"tolerance\": ", repr(case.tolerance),
                ", \"maximum_iterations\": ", case.maximum_iterations,
                ", \"restart_count\": ", case.restart_count,
                ", \"convergence_branch\": ", q, case.convergence_branch, q, "}")
        end
        print(io, "\n  },\n")
        print(io, "  \"files\": [\"u0.npy\", \"u1.npy\", \"u2.npy\", \"u3.npy\", \"rhs_julia.npy\", \"rhs_rust.npy\", \"guess_zero_julia.npy\", \"guess_zero_rust.npy\", \"guess_nonzero_julia.npy\", \"guess_nonzero_rust.npy\", \"cg_zero_solution_julia.npy\", \"cg_zero_solution_rust.npy\", \"cg_nonzero_solution_julia.npy\", \"cg_nonzero_solution_rust.npy\", \"bicgstab_zero_solution_julia.npy\", \"bicgstab_zero_solution_rust.npy\", \"bicgstab_nonzero_solution_julia.npy\", \"bicgstab_nonzero_solution_rust.npy\"],\n")
        print(io, "  \"comparison\": {\"solution_max_abs_tolerance\": 2e-11, \"rust_true_relative_residual_tolerance\": 1e-11, \"criterion\": \"fresh sum(abs2, b-A*x) independent of recursive residual\"},\n")
        print(io, "  \"generator\": {\"script\": \"fixtures/generate.jl\", \"mode\": \"fermions_task_b\", \"randomness\": \"none\"}\n")
        print(io, "}\n")
    end
end

if FERMIONS_TASK_B_MODE
    generate_fermions_task_b()
    exit()
end

const FERMIONS_TASK_C_EPS = 1.0e-20
const FERMIONS_TASK_C_MAXSTEPS = 2_000
const FERMIONS_TASK_C_BETA = 5.7
const FERMIONS_TASK_C_KAPPA = 0.13
const FERMIONS_TASK_C_STEP_SIZE = 0.002
const FERMIONS_TASK_C_STEPS = 2
const FERMIONS_TASK_C_ACCEPTANCE_STATE = (UInt64(0x434143434550545f), UInt64(17), UInt64(29), UInt64(43))

function fermions_task_c_xi(lattice)
    nx, ny, nz, nt = lattice
    xi = Array{ComplexF64}(undef, NC, nx, ny, nz, nt, 4)
    for spin in 1:4, it in 1:nt, iz in 1:nz, iy in 1:ny, ix in 1:nx, color in 1:NC
        site = (ix - 1) + nx * ((iy - 1) + ny * ((iz - 1) + nz * (it - 1)))
        flat = (color - 1) + NC * ((spin - 1) + 4 * site)
        xi[color, ix, iy, iz, it, spin] = ComplexF64(
            0.021 * (flat + 1) - 0.004 * (spin - 1),
            -0.015 * (flat + 2) + 0.003 * (color - 1),
        )
    end
    return xi
end

function fermions_task_c_momentum(links)
    p = initialize_TA_Gaugefields(links)
    nx, ny, nz, nt = links[1].NV == 0 ? (0, 0, 0, 0) : size(links[1].U)[3:6]
    for mu in 1:4, it in 1:nt, iz in 1:nz, iy in 1:ny, ix in 1:nx, a in 1:8
        site = (ix - 1) + nx * ((iy - 1) + ny * ((iz - 1) + nz * (it - 1)))
        p[mu].a[a, ix, iy, iz, it] =
            0.007 * (a + 2 * mu) + 0.0011 * site - 0.0007 * (a * mu)
    end
    return p
end

function fermions_task_c_u_update!(U, momentum, dt, temps)
    temp1, it_temp1 = get_temp(temps)
    temp2, it_temp2 = get_temp(temps)
    expU, it_expU = get_temp(temps)
    W, it_W = get_temp(temps)
    try
        for mu in 1:4
            exptU!(expU, 0.5 * dt, momentum[mu], [temp1, temp2])
            mul!(W, expU, U[mu])
            substitute_U!(U[mu], W)
        end
    finally
        unused!(temps, it_temp1)
        unused!(temps, it_temp2)
        unused!(temps, it_expU)
        unused!(temps, it_W)
    end
end

function fermions_task_c_gauge_p_update!(U, momentum, dt, gauge_action, temps)
    dSdU, it_dSdU = get_temp(temps)
    product, it_product = get_temp(temps)
    try
        for mu in 1:4
            calc_dSdUμ!(dSdU, gauge_action, mu, U)
            mul!(product, U[mu], dSdU)
            Traceless_antihermitian_add!(momentum[mu], -dt / 3.0, product)
        end
    finally
        unused!(temps, it_dSdU)
        unused!(temps, it_product)
    end
end

function fermions_task_c_fermion_p_update!(U, momentum, dt, fermi_action, phi)
    raw_force = [similar(U[1]) for _ in 1:4]
    calc_UdSfdU!(raw_force, fermi_action, U, phi)
    for mu in 1:4
        Traceless_antihermitian_add!(momentum[mu], -dt, raw_force[mu])
    end
end

function fermions_task_c_hamiltonian(U, gauge_action, momentum, fermion_action_value)
    nc = U[1].NC
    return real(-evaluate_GaugeAction(gauge_action, U) / nc + momentum * momentum / 2 + fermion_action_value)
end

function fermions_task_c_trajectory!(U, momentum, gauge_action, fermi_action, phi, dt, steps)
    temps = Temporalfields(U[1]; num=10)
    for _ in 1:steps
        fermions_task_c_u_update!(U, momentum, dt, temps)
        fermions_task_c_gauge_p_update!(U, momentum, dt, gauge_action, temps)
        fermions_task_c_fermion_p_update!(U, momentum, dt, fermi_action, phi)
        fermions_task_c_u_update!(U, momentum, dt, temps)
    end
end

function generate_fermions_task_c()
    VERSION == "0.7.2" || error("expected Gaugefields.jl v0.7.2, found $VERSION")
    COMMIT == "9e5719970770f4497405a856315c90bef7f74449" ||
        error("expected Gaugefields.jl commit 9e5719970770f4497405a856315c90bef7f74449")
    LATTICEDIRACOPERATORS_VERSION == "0.6.4" ||
        error("expected LatticeDiracOperators.jl v0.6.4, found $LATTICEDIRACOPERATORS_VERSION")
    LATTICEDIRACOPERATORS_COMMIT == "bdef628184597815ba3e0cddf2536df767e78a02" ||
        error("expected LatticeDiracOperators.jl commit bdef628184597815ba3e0cddf2536df767e78a02")

    lattice = (2, 2, 2, 2)
    links = fermions_task_a_links(lattice)
    xi_values = fermions_task_c_xi(lattice)
    xi = fermions_task_a_field(links, xi_values)
    parameters = Dict{String,Any}(
        "Dirac_operator" => "Wilson",
        "κ" => FERMIONS_TASK_C_KAPPA,
        "r" => 1.0,
        "faster version" => false,
        "verbose_level" => 0,
        "boundarycondition" => Int8[1, 1, 1, -1],
        "method_CG" => "bicg",
        "eps_CG" => FERMIONS_TASK_C_EPS,
        "MaxCGstep" => FERMIONS_TASK_C_MAXSTEPS,
    )
    dirac = Dirac_operator(links, xi, parameters)
    fermi_action = FermiAction(dirac, Dict())
    phi = similar(xi)
    sample_pseudofermions!(phi, links, fermi_action, xi)

    normal = DdagD_operator(links, xi, parameters)
    x = similar(xi)
    solve_DinvX!(x, normal, phi)
    y = similar(xi)
    mul!(y, dirac, x)
    action = real(dot(phi, x))
    raw_force = [similar(links[1]) for _ in 1:4]
    calc_UdSfdU!(raw_force, fermi_action, links, phi)
    force = initialize_TA_Gaugefields(links)
    for mu in 1:4
        Traceless_antihermitian_add!(force[mu], 1.0, raw_force[mu])
    end

    gauge_action = GaugeAction(links)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop, plaqloop')
    push!(gauge_action, FERMIONS_TASK_C_BETA / 2, plaqloop)
    initial_momentum = fermions_task_c_momentum(links)
    proposed = similar(links)
    substitute_U!(proposed, links)
    momentum = initialize_TA_Gaugefields(links)
    for mu in 1:4
        momentum[mu].a .= initial_momentum[mu].a
    end
    h_initial = fermions_task_c_hamiltonian(links, gauge_action, momentum, action)
    fermions_task_c_trajectory!(proposed, momentum, gauge_action, fermi_action, phi,
        FERMIONS_TASK_C_STEP_SIZE, FERMIONS_TASK_C_STEPS)
    h_proposed = fermions_task_c_hamiltonian(
        proposed,
        gauge_action,
        momentum,
        evaluate_FermiAction(fermi_action, proposed, phi),
    )
    delta_h = h_proposed - h_initial
    probability = delta_h <= 0.0 ? 1.0 : exp(-delta_h)
    acceptance_rng = Random.Xoshiro(FERMIONS_TASK_C_ACCEPTANCE_STATE...)
    acceptance_raw = rand(acceptance_rng, UInt64)
    acceptance_uniform = (Float64(acceptance_raw >>> 12) + 0.5) * 2.0^-52
    accepted = acceptance_uniform <= probability
    next_raw_word = rand(acceptance_rng, UInt64)

    out = joinpath(@__DIR__, "fermions_task_c")
    mkpath(out)
    for direction in 1:4
        NPZ.npzwrite(joinpath(out, "u$(direction - 1).npy"), links[direction].U)
        NPZ.npzwrite(joinpath(out, "force$(direction - 1).npy"), force[direction].a)
        NPZ.npzwrite(joinpath(out, "p_initial$(direction - 1).npy"), initial_momentum[direction].a)
        NPZ.npzwrite(joinpath(out, "p_final$(direction - 1).npy"), momentum[direction].a)
        NPZ.npzwrite(joinpath(out, "u_proposed$(direction - 1).npy"), proposed[direction].U)
    end
    for (name, field) in (("xi", xi), ("phi", phi), ("x", x), ("y", y))
        NPZ.npzwrite(joinpath(out, "$(name)_julia.npy"), copy(field.f))
        NPZ.npzwrite(joinpath(out, "$(name)_rust.npy"), permutedims(field.f, (1, 6, 2, 3, 4, 5)))
    end
    open(joinpath(out, "metadata.json"), "w") do io
        q = Char(34)
        print(io, "{\n")
        print(io, "  \"schema\": \"fermions_task_c.v1\",\n")
        print(io, "  \"lattice\": [2, 2, 2, 2],\n")
        print(io, "  \"nc\": 3,\n  \"components\": 4,\n")
        print(io, "  \"beta\": ", repr(FERMIONS_TASK_C_BETA), ",\n")
        print(io, "  \"kappa\": ", repr(FERMIONS_TASK_C_KAPPA), ",\n")
        print(io, "  \"r\": 1.0,\n  \"boundaries\": [1, 1, 1, -1],\n")
        print(io, "  \"solver_parameters\": {\"tolerance\": ", repr(FERMIONS_TASK_C_EPS),
            ", \"max_iterations\": ", FERMIONS_TASK_C_MAXSTEPS,
            ", \"julia_operator_keys\": [\"Dirac_operator\", \"κ\", \"r\", \"faster version\", \"verbose_level\", \"boundarycondition\", \"method_CG\", \"eps_CG\", \"MaxCGstep\"], \"julia_solver_keywords\": [\"eps\", \"maxsteps\", \"verbose\"]},\n")
        print(io, "  \"gaugefields_jl\": {\"package\": \"Gaugefields.jl\", \"version\": \"$VERSION\", \"commit\": \"$COMMIT\", \"clean\": true},\n")
        print(io, "  \"latticediracoperators_jl\": {\"package\": \"LatticeDiracOperators.jl\", \"version\": \"$LATTICEDIRACOPERATORS_VERSION\", \"commit\": \"$LATTICEDIRACOPERATORS_COMMIT\", \"clean\": true},\n")
        print(io, "  \"source_urls\": [\n")
        print(io, "    \"https://github.com/shinaoka/LatticeDiracOperators.jl/blob/$LATTICEDIRACOPERATORS_COMMIT/src/action/WilsonFermiAction.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/LatticeDiracOperators.jl/blob/$LATTICEDIRACOPERATORS_COMMIT/test/wilsonhmc.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/LatticeDiracOperators.jl/blob/$LATTICEDIRACOPERATORS_COMMIT/src/WilsonFermion/WilsonFermion.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/Gaugefields.jl/blob/$COMMIT/src/4D/TA_gaugefields_4D_serial.jl\"\n  ],\n")
        print(io, "  \"source_functions\": [\"sample_pseudofermions!\", \"evaluate_FermiAction\", \"calc_UdSfdU!\", \"calc_UdSfdU_fromX!\", \"MDstep!\", \"U_update!\", \"P_update!\", \"P_update_fermion!\", \"Traceless_antihermitian_add!\"],\n")
        print(io, "  \"entrypoint_map\": [\n")
        print(io, "    {\"julia\": \"sample_pseudofermions!\", \"julia_source\": \"src/action/WilsonFermiAction.jl:362-377\", \"rust\": \"WilsonFermiAction::sample_pseudofermion\"},\n")
        print(io, "    {\"julia\": \"evaluate_FermiAction\", \"julia_source\": \"src/action/WilsonFermiAction.jl:86-97\", \"rust\": \"WilsonFermiAction::evaluate\"},\n")
        print(io, "    {\"julia\": \"calc_UdSfdU!\", \"julia_source\": \"src/action/WilsonFermiAction.jl:99-136\", \"rust\": \"WilsonFermiAction::force\"},\n")
        print(io, "    {\"julia\": \"calc_UdSfdU_fromX!\", \"julia_source\": \"src/action/WilsonFermiAction.jl:138-234\", \"rust\": \"wilson_action.rs::force_from_x\"},\n")
        print(io, "    {\"julia\": \"MDstep!/U_update!/P_update!/P_update_fermion!\", \"julia_source\": \"test/wilsonhmc.jl:46-146\", \"rust\": \"wilson_hmc.rs::wilson_hmc_update\"},\n")
        print(io, "    {\"julia\": \"Traceless_antihermitian_add!\", \"julia_source\": \"https://github.com/shinaoka/Gaugefields.jl/blob/$COMMIT/src/4D/TA_gaugefields_4D_serial.jl:181-269\", \"rust\": \"Mat3::add_ta_coefficients\"}\n  ],\n")
        print(io, "  \"layout\": {\"julia_shape\": \"[3,NX,NY,NZ,NT,4]\", \"rust_shape\": \"[3,4,NX,NY,NZ,NT]\", \"conversion\": \"permutedims(array, (1, 6, 2, 3, 4, 5))\", \"permutation\": [1, 6, 2, 3, 4, 5], \"site_order\": \"x fastest\"},\n")
        print(io, "  \"construction\": \"explicit diagonal SU(3) links, fixed xi, phi, and coefficient-space momentum; no global RNG; the acceptance draw uses explicit Julia Xoshiro state\",\n")
        print(io, "  \"pseudofermion_refresh\": {\"flavors\": 2, \"formula\": \"phi = D† xi\", \"complex_normal_scale\": \"1/sqrt(2) per independent real and imaginary standard normal\", \"fixture_xi\": \"fixed array; sampler parity is checked separately in Rust\"},\n")
        print(io, "  \"force_convention\": {\"x\": \"(D†D)^-1 phi\", \"y\": \"D x\", \"raw_formula\": \"-kappa Pminus U Xplus outer Y + kappa X outer (Yplus† U† Pplus)\", \"wrapped_link_sign\": \"applied to both terms exactly once\", \"projection\": \"Gaugefields.jl Traceless_antihermitian_add!; A=(i/2) sum_a c_a lambda_a\", \"gauge_1_over_nc\": \"not applied here\"},\n")
        print(io, "  \"momentum_update_scaling\": {\"gauge\": \"-step_size/NC\", \"fermion\": \"-step_size\"},\n")
        print(io, "  \"action\": ", repr(action), ",\n")
        print(io, "  \"trajectory\": {\"step_size\": ", repr(FERMIONS_TASK_C_STEP_SIZE), ", \"steps\": ", FERMIONS_TASK_C_STEPS, ", \"initial_hamiltonian\": ", repr(h_initial), ", \"proposed_hamiltonian\": ", repr(h_proposed), ", \"delta_h\": ", repr(delta_h), ", \"acceptance_probability\": ", repr(probability), ", \"acceptance_rng_state\": [", join(string.(FERMIONS_TASK_C_ACCEPTANCE_STATE), ", "), "], \"acceptance_uniform\": ", repr(acceptance_uniform), ", \"acceptance_uniform_bits\": ", reinterpret(UInt64, acceptance_uniform), ", \"accepted\": ", accepted, ", \"next_raw_word\": ", next_raw_word, "},\n")
        print(io, "  \"comparison\": {\"field_max_abs_tolerance\": 2e-10, \"force_max_abs_tolerance\": 2e-10, \"action_tolerance\": 2e-10, \"force_finite_difference_tolerance\": 2e-7, \"finite_difference_epsilons\": [1e-3, 5e-4, 2.5e-4], \"finite_difference\": \"central U <- exp(epsilon*T_a)U; the final residual is tested with an O(epsilon^2) trend\", \"trajectory_tolerance\": 2e-10},\n")
        print(io, "  \"files\": [\"u0.npy\", \"u1.npy\", \"u2.npy\", \"u3.npy\", \"xi_julia.npy\", \"xi_rust.npy\", \"phi_julia.npy\", \"phi_rust.npy\", \"x_julia.npy\", \"x_rust.npy\", \"y_julia.npy\", \"y_rust.npy\", \"force0.npy\", \"force1.npy\", \"force2.npy\", \"force3.npy\", \"p_initial0.npy\", \"p_initial1.npy\", \"p_initial2.npy\", \"p_initial3.npy\", \"p_final0.npy\", \"p_final1.npy\", \"p_final2.npy\", \"p_final3.npy\", \"u_proposed0.npy\", \"u_proposed1.npy\", \"u_proposed2.npy\", \"u_proposed3.npy\"],\n")
        print(io, "  \"generator\": {\"script\": \"fixtures/generate.jl\", \"mode\": \"fermions_task_c\", \"randomness\": \"none for fixed fields and trajectory; explicit Xoshiro only for recorded acceptance draw\"}\n")
        print(io, "}\n")
    end
end

if FERMIONS_TASK_C_MODE
    generate_fermions_task_c()
    exit()
end

const FERMIONS_TASK_D_ABSOLUTE_SQUARED_TOLERANCE = 1.0e-24
const FERMIONS_TASK_D_OPERATOR_TOLERANCE = 2.0e-12
const FERMIONS_TASK_D_SHIFTED_TRUE_RELATIVE_TOLERANCE = 1.0e-11
const FERMIONS_TASK_D_MAXSTEPS = 2_000
const FERMIONS_TASK_D_MASS = 0.17
const FERMIONS_TASK_D_SHIFTS = (0.31, 0.0, 0.07)

function fermions_task_d_input(lattice)
    nx, ny, nz, nt = lattice
    input = Array{ComplexF64}(undef, NC, nx, ny, nz, nt, 1)
    for it in 1:nt, iz in 1:nz, iy in 1:ny, ix in 1:nx, color in 1:NC
        site = (ix - 1) + nx * ((iy - 1) + ny * ((iz - 1) + nz * (it - 1)))
        input[color, ix, iy, iz, it, 1] = ComplexF64(
            0.021 * (color + 2 * site) - 0.004 * (ix - 1),
            -0.013 * (2 * color + site + 1) + 0.003 * (iy - 1),
        )
    end
    return input
end

function fermions_task_d_rhs(lattice)
    nx, ny, nz, nt = lattice
    rhs = Array{ComplexF64}(undef, NC, nx, ny, nz, nt, 1)
    for it in 1:nt, iz in 1:nz, iy in 1:ny, ix in 1:nx, color in 1:NC
        site = (ix - 1) + nx * ((iy - 1) + ny * ((iz - 1) + nz * (it - 1)))
        flat = (color - 1) + NC * site
        rhs[color, ix, iy, iz, it, 1] = ComplexF64(
            0.017 * (flat + 1),
            -0.011 * (2 * flat + 3),
        )
    end
    return rhs
end

function fermions_task_d_field(links, values)
    field = LatticeDiracOperators.Dirac_operators.Initialize_StaggeredFermion(
        links[1]; nowing=true)
    field.f .= values
    return field
end

function fermions_task_d_eta(lattice)
    nx, ny, nz, nt = lattice
    eta = Array{Float64}(undef, 4, nx, ny, nz, nt)
    for it in 1:nt, iz in 1:nz, iy in 1:ny, ix in 1:nx
        x, y, z = ix - 1, iy - 1, iz - 1
        eta[1, ix, iy, iz, it] = 1.0
        eta[2, ix, iy, iz, it] = iseven(x) ? 1.0 : -1.0
        eta[3, ix, iy, iz, it] = iseven(x + y) ? 1.0 : -1.0
        eta[4, ix, iy, iz, it] = iseven(x + y + z) ? 1.0 : -1.0
    end
    return eta
end

function fermions_task_d_eta_impulses(lattice)
    nx, ny, nz, nt = lattice
    extents = (nx, ny, nz, nt)
    values = Array{Float64}(undef, 4, 2, 2)
    for direction in 1:4
        for side in 1:2
            source = zeros(Int, 4)
            source[direction] = side == 1 ? 0 : extents[direction] - 1
            x, y, z = source[1:3]
            phase = direction == 1 ? 1 : direction == 2 ? (iseven(x) ? 1 : -1) :
                direction == 3 ? (iseven(x + y) ? 1 : -1) :
                (iseven(x + y + z) ? 1 : -1)
            values[direction, side, 1] = phase
            values[direction, side, 2] = direction == 4 ? -1 : 1
        end
    end
    return values
end

function fermions_task_d_true_residual_squared(operator, solution, rhs, shift)
    applied = similar(rhs)
    mul!(applied, operator, solution)
    applied.f .+= shift .* solution.f
    return sum(abs2, vec(rhs.f) .- vec(applied.f))
end

function fermions_task_d_shifted_diagnostics(operator, rhs, shifts, shifted_solutions, eps, maxsteps)
    x = similar(rhs)
    x.f .= 0
    temp1 = similar(rhs)
    r = similar(rhs)
    p = similar(rhs)
    q = similar(rhs)
    r.f .= rhs.f
    mul!(temp1, operator, x)
    r.f .-= temp1.f
    p.f .= r.f
    initial = real(dot(r, r))
    recursive = fill(initial, length(shifts))
    iterations = fill(0, length(shifts))
    converged = fill(initial < eps, length(shifts))
    αm = 1.0
    βm = 0.0
    ρm = ones(ComplexF64, length(shifts))
    ρ0 = ones(ComplexF64, length(shifts))
    ρp = ones(ComplexF64, length(shifts))
    for iteration in 1:maxsteps
        mul!(q, operator, p)
        pAp = dot(p, q)
        rr = dot(r, r)
        αk = rr / pAp
        x.f .+= αk .* p.f
        r.f .-= αk .* q.f
        βk = dot(r, r) / rr
        p.f .= βk .* p.f .+ r.f
        for index in eachindex(shifts)
            converged[index] && continue
            ρkj = ρ0[index]
            ρkmj = ρm[index]
            ρp[index] = ρkj * ρkmj * αm /
                (ρkmj * αm * (1.0 + αk * shifts[index]) +
                 αk * βm * (ρkmj - ρkj))
            αkj = (ρp[index] / ρkj) * αk
            βkj = (ρp[index] / ρkj)^2 * βk
            estimate = abs(rr * abs(ρp[index])^2)
            recursive[index] = estimate
            if estimate < eps
                iterations[index] = iteration
                converged[index] = true
            end
        end
        ρm .= ρ0
        ρ0 .= ρp
        αm = αk
        βm = βk
        all(converged) && break
    end
    all(converged) || error("Task D Julia shifted-CG diagnostic replay exhausted")
    true_residuals = [
        fermions_task_d_true_residual_squared(operator, solution, rhs, shifts[index])
        for (index, solution) in enumerate(shifted_solutions)
    ]
    return initial, recursive, iterations, true_residuals
end

function generate_fermions_task_d()
    VERSION == "0.7.2" || error("expected Gaugefields.jl v0.7.2, found $VERSION")
    COMMIT == "9e5719970770f4497405a856315c90bef7f74449" ||
        error("expected Gaugefields.jl commit 9e5719970770f4497405a856315c90bef7f74449")
    LATTICEDIRACOPERATORS_VERSION == "0.6.4" ||
        error("expected LatticeDiracOperators.jl v0.6.4, found $LATTICEDIRACOPERATORS_VERSION")
    LATTICEDIRACOPERATORS_COMMIT == "bdef628184597815ba3e0cddf2536df767e78a02" ||
        error("expected LatticeDiracOperators.jl commit bdef628184597815ba3e0cddf2536df767e78a02")

    lattice = (2, 2, 2, 2)
    links = fermions_task_a_links(lattice)
    input_values = fermions_task_d_input(lattice)
    rhs_values = fermions_task_d_rhs(lattice)
    out = joinpath(@__DIR__, "fermions_task_d")
    mkpath(out)
    for direction in 1:4
        NPZ.npzwrite(joinpath(out, "u$(direction - 1).npy"), links[direction].U)
    end
    NPZ.npzwrite(joinpath(out, "input_julia.npy"), input_values)
    NPZ.npzwrite(joinpath(out, "input_rust.npy"), permutedims(input_values, (1, 6, 2, 3, 4, 5)))
    NPZ.npzwrite(joinpath(out, "rhs_julia.npy"), rhs_values)
    NPZ.npzwrite(joinpath(out, "rhs_rust.npy"), permutedims(rhs_values, (1, 6, 2, 3, 4, 5)))
    NPZ.npzwrite(joinpath(out, "eta.npy"), fermions_task_d_eta(lattice))
    NPZ.npzwrite(joinpath(out, "eta_impulses.npy"), fermions_task_d_eta_impulses(lattice))

    cases = ((name="periodic", boundary=[1, 1, 1, 1]),
        (name="default_antiperiodic", boundary=[1, 1, 1, -1]))
    for case in cases
        source = fermions_task_d_field(links, input_values)
        parameters = Dict{String,Any}(
            "Dirac_operator" => "Staggered",
            "mass" => FERMIONS_TASK_D_MASS,
            "verbose_level" => 0,
            "boundarycondition" => Int8.(case.boundary),
            "eps" => FERMIONS_TASK_D_ABSOLUTE_SQUARED_TOLERANCE,
            "MaxCGstep" => FERMIONS_TASK_D_MAXSTEPS,
            "method_CG" => "cg",
        )
        dirac = LatticeDiracOperators.Dirac_operators.Staggered_Dirac_operator(
            links, source, parameters)
        d = similar(source)
        ddag = similar(source)
        normal = similar(source)
        mul!(d, dirac, source)
        mul!(ddag, adjoint(dirac), source)
        mul!(
            normal,
            LatticeDiracOperators.Dirac_operators.DdagD_Staggered_operator(dirac),
            source,
        )
        k = similar(source)
        k.f .= 0.5 .* (d.f .- ddag.f)
        for (label, field) in (("d", d), ("ddag", ddag), ("k", k),
            ("normal_composition", normal), ("normal_closed", normal))
            NPZ.npzwrite(joinpath(out, "$(label)_$(case.name)_julia.npy"), copy(field.f))
            NPZ.npzwrite(
                joinpath(out, "$(label)_$(case.name)_rust.npy"),
                permutedims(field.f, (1, 6, 2, 3, 4, 5)),
            )
        end
    end

    source = fermions_task_d_field(links, input_values)
    rhs = fermions_task_d_field(links, rhs_values)
    parameters = Dict{String,Any}(
        "Dirac_operator" => "Staggered",
        "mass" => FERMIONS_TASK_D_MASS,
        "verbose_level" => 0,
        "boundarycondition" => Int8[1, 1, 1, -1],
        "eps" => FERMIONS_TASK_D_ABSOLUTE_SQUARED_TOLERANCE,
        "MaxCGstep" => FERMIONS_TASK_D_MAXSTEPS,
        "method_CG" => "cg",
    )
    dirac = LatticeDiracOperators.Dirac_operators.Staggered_Dirac_operator(
        links, source, parameters)
    normal = LatticeDiracOperators.Dirac_operators.DdagD_Staggered_operator(dirac)
    shifted_solutions = [similar(rhs) for _ in FERMIONS_TASK_D_SHIFTS]
    base = similar(rhs)
    base.f .= 0
    LatticeDiracOperators.Dirac_operators.shiftedcg(
        shifted_solutions,
        collect(FERMIONS_TASK_D_SHIFTS),
        base,
        normal,
        rhs;
        eps=FERMIONS_TASK_D_ABSOLUTE_SQUARED_TOLERANCE,
        maxsteps=FERMIONS_TASK_D_MAXSTEPS,
        verbose=Verbose_print(0),
    )
    for (index, solution) in enumerate(shifted_solutions)
        NPZ.npzwrite(joinpath(out, "shift$(index - 1)_julia.npy"), copy(solution.f))
        NPZ.npzwrite(
            joinpath(out, "shift$(index - 1)_rust.npy"),
            permutedims(solution.f, (1, 6, 2, 3, 4, 5)),
        )
    end
    initial, recursive, iterations, true_residuals = fermions_task_d_shifted_diagnostics(
        normal,
        rhs,
        collect(FERMIONS_TASK_D_SHIFTS),
        shifted_solutions,
        FERMIONS_TASK_D_ABSOLUTE_SQUARED_TOLERANCE,
        FERMIONS_TASK_D_MAXSTEPS,
    )

    open(joinpath(out, "shifted_reports.json"), "w") do io
        print(io, "{\n  \"initial_residual_squared\": ", repr(initial), ",\n")
        print(io, "  \"reports\": [\n")
        for index in eachindex(FERMIONS_TASK_D_SHIFTS)
            index > 1 && print(io, ",\n")
            print(io, "    {\"shift\": ", repr(FERMIONS_TASK_D_SHIFTS[index]),
                ", \"iterations\": ", iterations[index],
                ", \"recursive_residual_squared\": ", repr(recursive[index]),
                ", \"true_residual_squared\": ", repr(true_residuals[index]),
                ", \"absolute_squared_tolerance\": ", repr(FERMIONS_TASK_D_ABSOLUTE_SQUARED_TOLERANCE),
                ", \"maximum_iterations\": ", FERMIONS_TASK_D_MAXSTEPS,
                ", \"convergence_branch\": \"updated_residual\"}")
        end
        print(io, "\n  ]\n}\n")
    end

    open(joinpath(out, "metadata.json"), "w") do io
        q = Char(34)
        print(io, "{\n")
        print(io, "  \"schema\": \"fermions_task_d.v1\",\n")
        print(io, "  \"lattice\": [2, 2, 2, 2],\n")
        print(io, "  \"nc\": 3,\n  \"components\": 1,\n")
        print(io, "  \"mass\": ", repr(FERMIONS_TASK_D_MASS), ",\n")
        print(io, "  \"boundaries\": {\"periodic\": [1, 1, 1, 1], \"default_antiperiodic\": [1, 1, 1, -1]},\n")
        print(io, "  \"shifts\": ", json_number_array(collect(FERMIONS_TASK_D_SHIFTS)), ",\n")
        print(io, "  \"solver_parameters\": {\"absolute_squared_tolerance\": ", repr(FERMIONS_TASK_D_ABSOLUTE_SQUARED_TOLERANCE),
            ", \"max_iterations\": ", FERMIONS_TASK_D_MAXSTEPS,
            ", \"julia_solver_keywords\": [\"eps\", \"maxsteps\", \"verbose\"]},\n")
        print(io, "  \"gaugefields_jl\": {\"package\": \"Gaugefields.jl\", \"version\": \"$VERSION\", \"commit\": \"$COMMIT\", \"clean\": true},\n")
        print(io, "  \"latticediracoperators_jl\": {\"package\": \"LatticeDiracOperators.jl\", \"version\": \"$LATTICEDIRACOPERATORS_VERSION\", \"commit\": \"$LATTICEDIRACOPERATORS_COMMIT\", \"clean\": true},\n")
        print(io, "  \"source_urls\": [\n")
        print(io, "    \"https://github.com/shinaoka/LatticeDiracOperators.jl/blob/$LATTICEDIRACOPERATORS_COMMIT/src/StaggeredFermion/StaggeredFermion.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/LatticeDiracOperators.jl/blob/$LATTICEDIRACOPERATORS_COMMIT/src/StaggeredFermion/StaggeredFermion_4D_nowing.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/LatticeDiracOperators.jl/blob/$LATTICEDIRACOPERATORS_COMMIT/src/cgmethods.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/Gaugefields.jl/blob/$COMMIT/src/4D/nowing/gaugefields_4D_nowing.jl\"\n  ],\n")
        print(io, "  \"source_functions\": [\"Staggered_Dirac_operator\", \"Dx!\", \"shift_fermion\", \"shifted_fermion!\", \"staggered_U\", \"DdagD_Staggered_operator\", \"LinearAlgebra.mul!\", \"shiftedcg\"],\n")
        print(io, "  \"entrypoint_map\": [\n")
        print(io, "    {\"julia\": \"Staggered_Dirac_operator\", \"julia_source\": \"src/StaggeredFermion/StaggeredFermion.jl:25-78\", \"rust\": \"staggered.rs::StaggeredDirac::with_boundary\"},\n")
        print(io, "    {\"julia\": \"Dx!\", \"julia_source\": \"src/StaggeredFermion/StaggeredFermion_4D_nowing.jl:43-80\", \"rust\": \"staggered.rs::StaggeredDirac::apply_hopping_to_data\"},\n")
        print(io, "    {\"julia\": \"staggered_U\", \"julia_source\": \"Gaugefields.jl/src/4D/nowing/gaugefields_4D_nowing.jl:459-504\", \"rust\": \"staggered.rs::staggered_eta + Mat3::scaled\"},\n")
        print(io, "    {\"julia\": \"shift_fermion/shifted_fermion!\", \"julia_source\": \"src/StaggeredFermion/StaggeredFermion_4D_nowing.jl:99-198\", \"rust\": \"staggered.rs::StaggeredDirac::neighbor\"},\n")
        print(io, "    {\"julia\": \"LinearAlgebra.mul! + DdagD_Staggered_operator\", \"julia_source\": \"src/StaggeredFermion/StaggeredFermion.jl:166-243\", \"rust\": \"staggered.rs::StaggeredNormalOperator + StaggeredClosedNormalOperator\"},\n")
        print(io, "    {\"julia\": \"Dirac_operators.shiftedcg\", \"julia_source\": \"src/cgmethods.jl:872-968\", \"rust\": \"solvers.rs::multi_shift_cg\"}\n  ],\n")
        print(io, "  \"layout\": {\"julia_shape\": \"[3,NX,NY,NZ,NT,1]\", \"rust_shape\": \"[3,1,NX,NY,NZ,NT]\", \"conversion\": \"permutedims(array, (1, 6, 2, 3, 4, 5))\", \"permutation\": [1, 6, 2, 3, 4, 5], \"site_order\": \"x fastest\"},\n")
        print(io, "  \"eta\": {\"formula\": [\"eta_0=1\", \"eta_1=(-1)^x\", \"eta_2=(-1)^(x+y)\", \"eta_3=(-1)^(x+y+z)\"], \"coordinates\": \"zero-based\", \"files\": [\"eta.npy\", \"eta_impulses.npy\"], \"impulse_layout\": \"[direction, lower_or_upper_source, eta_or_default_boundary_sign]\", \"wrap_sign\": \"boundary sign applied once on each wrapped fermion hop\"},\n")
        print(io, "  \"construction\": \"explicit diagonal nontrivial SU(3) links, one-component input and rhs from fixed formulas; no RNG or global state\",\n")
        print(io, "  \"normal\": {\"composition\": \"Ddag(D(x))\", \"closed_form\": \"mass^2*x-K(K(x))\", \"antihermitian_identity\": \"Kdag=-K\"},\n")
        print(io, "  \"comparison\": {\"operator_max_abs_tolerance\": ", repr(FERMIONS_TASK_D_OPERATOR_TOLERANCE), ", \"antihermiticity_tolerance\": ", repr(FERMIONS_TASK_D_OPERATOR_TOLERANCE), ", \"normal_composition_tolerance\": ", repr(FERMIONS_TASK_D_OPERATOR_TOLERANCE), ", \"shifted_true_relative_residual_tolerance\": ", repr(FERMIONS_TASK_D_SHIFTED_TRUE_RELATIVE_TOLERANCE), ", \"criterion\": \"maximum absolute complex-component operator residual and fresh relative shifted residual\"},\n")
        print(io, "  \"files\": [\"u0.npy\", \"u1.npy\", \"u2.npy\", \"u3.npy\", \"input_julia.npy\", \"input_rust.npy\", \"rhs_julia.npy\", \"rhs_rust.npy\", \"eta.npy\", \"eta_impulses.npy\",\n")
        print(io, "    \"d_periodic_julia.npy\", \"d_periodic_rust.npy\", \"ddag_periodic_julia.npy\", \"ddag_periodic_rust.npy\", \"k_periodic_julia.npy\", \"k_periodic_rust.npy\", \"normal_composition_periodic_julia.npy\", \"normal_composition_periodic_rust.npy\", \"normal_closed_periodic_julia.npy\", \"normal_closed_periodic_rust.npy\",\n")
        print(io, "    \"d_default_antiperiodic_julia.npy\", \"d_default_antiperiodic_rust.npy\", \"ddag_default_antiperiodic_julia.npy\", \"ddag_default_antiperiodic_rust.npy\", \"k_default_antiperiodic_julia.npy\", \"k_default_antiperiodic_rust.npy\", \"normal_composition_default_antiperiodic_julia.npy\", \"normal_composition_default_antiperiodic_rust.npy\", \"normal_closed_default_antiperiodic_julia.npy\", \"normal_closed_default_antiperiodic_rust.npy\",\n")
        print(io, "    \"shift0_julia.npy\", \"shift0_rust.npy\", \"shift1_julia.npy\", \"shift1_rust.npy\", \"shift2_julia.npy\", \"shift2_rust.npy\", \"shifted_reports.json\"],\n")
        print(io, "  \"generator\": {\"script\": \"fixtures/generate.jl\", \"mode\": \"fermions_task_d\", \"randomness\": \"none\"}\n")
        print(io, "}\n")
    end
end

if FERMIONS_TASK_D_MODE
    generate_fermions_task_d()
    exit()
end

const FERMIONS_TASK_E_MASS = 0.17
const FERMIONS_TASK_E_XI_SCALE = 0.25
const FERMIONS_TASK_E_LAMBDA_LOW = 0.0004
const FERMIONS_TASK_E_LAMBDA_HIGH = 64.0
const FERMIONS_TASK_E_SOLVER_TOLERANCE = 1.0e-24
const FERMIONS_TASK_E_MAXSTEPS = 2_000
const FERMIONS_TASK_E_FORCE_TOLERANCE = 2.0e-9
const FERMIONS_TASK_E_FD_TOLERANCE = 5.0e-7
# Recorded by the deterministic Rust all-512 force check.  The first three
# points expose truncation; the last point is the first all-coefficient pass.
const FERMIONS_TASK_E_FD_EPSILONS = (0.32, 0.16, 0.08, 0.04)
const FERMIONS_TASK_E_FD_MAX_RESIDUALS = (
    8.434653210321642e-6,
    2.139177378187619e-6,
    5.605769951367093e-7,
    1.6563038083509257e-7,
)
const FERMIONS_TASK_E_FD_RATIOS = (3.9429424115674574, 3.816027765580949, 3.384505863660601)
const FERMIONS_TASK_E_FD_PASS_COUNTS = (291, 442, 510, 512)
const FERMIONS_TASK_E_FD_SELECTED_EPSILON = 0.04
const FERMIONS_TASK_E_GRID_POINTS = 4_097
const FERMIONS_TASK_E_STEP_SIZE = 0.001
const FERMIONS_TASK_E_STEPS = 2
const FERMIONS_TASK_E_ACCEPTANCE_STATE = (UInt64(4846228630232126559), UInt64(17), UInt64(29), UInt64(43))
function fermions_task_e_field(links, values)
    field = LatticeDiracOperators.Dirac_operators.Initialize_StaggeredFermion(
        links[1]; nowing=true)
    field.f .= values
    return field
end

function fermions_task_e_xi(lattice)
    nx, ny, nz, nt = lattice
    xi = Array{ComplexF64}(undef, NC, nx, ny, nz, nt, 1)
    for it in 1:nt, iz in 1:nz, iy in 1:ny, ix in 1:nx, color in 1:NC
        site = (ix - 1) + nx * ((iy - 1) + ny * ((iz - 1) + nz * (it - 1)))
        flat = (color - 1) + NC * site
        xi[color, ix, iy, iz, it, 1] = FERMIONS_TASK_E_XI_SCALE * ComplexF64(
            0.013 * (flat + 1) - 0.002 * (ix - 1),
            -0.009 * (flat + 2) + 0.001 * (iy - 1),
        )
    end
    return xi
end

function fermions_task_e_momentum(links)
    p = initialize_TA_Gaugefields(links)
    nx, ny, nz, nt = size(links[1].U)[3:6]
    for mu in 1:4, it in 1:nt, iz in 1:nz, iy in 1:ny, ix in 1:nx, a in 1:8
        site = (ix - 1) + nx * ((iy - 1) + ny * ((iz - 1) + nz * (it - 1)))
        p[mu].a[a, ix, iy, iz, it] =
            0.007 * (a + 2 * mu) + 0.0011 * site - 0.0007 * (a * mu)
    end
    return p
end

function fermions_task_e_apply(coeff, operator, rhs, eps, maxsteps)
    shifted = [similar(rhs) for _ in 1:coeff.n]
    base = similar(rhs)
    base.f .= 0
    LatticeDiracOperators.Dirac_operators.shiftedcg(
        shifted, coeff.β, base, operator, rhs;
        eps=eps, maxsteps=maxsteps, verbose=Verbose_print(0),
    )
    result = similar(rhs)
    result.f .= coeff.α0 .* rhs.f
    for j in 1:coeff.n
        result.f .+= coeff.α[j] .* shifted[j].f
    end
    initial, recursive, iterations, true_residuals = fermions_task_d_shifted_diagnostics(
        operator, rhs, coeff.β, shifted, eps, maxsteps)
    return result, shifted, (initial=initial, recursive=recursive,
        iterations=iterations, true_residuals=true_residuals)
end

function fermions_task_e_scalar_error(coeff, power)
    maximum(begin
        x = i == 0 ? FERMIONS_TASK_E_LAMBDA_LOW :
            i == FERMIONS_TASK_E_GRID_POINTS - 1 ? FERMIONS_TASK_E_LAMBDA_HIGH :
            FERMIONS_TASK_E_LAMBDA_LOW * exp(
                log(FERMIONS_TASK_E_LAMBDA_HIGH / FERMIONS_TASK_E_LAMBDA_LOW) *
                i / (FERMIONS_TASK_E_GRID_POINTS - 1))
        approx = coeff.α0 + sum(coeff.α[j] / (x + coeff.β[j]) for j in 1:coeff.n)
        abs(approx - x^power)
    end for i in 0:(FERMIONS_TASK_E_GRID_POINTS - 1))
end

function fermions_task_e_write_coefficients(io, name, coeff, power, role)
    q = Char(34)
    bits = UInt64[reinterpret(UInt64, coeff.α0); reinterpret.(UInt64, coeff.α); reinterpret.(UInt64, coeff.β)]
    print(io, "    ", q, name, q, ": {", q, "role", q, ": ", q, role, q,
        ", ", q, "power", q, ": ", repr(power), ", ", q, "degree", q, ": ", coeff.n,
        ", ", q, "alpha0", q, ": ", repr(coeff.α0), ", ", q, "alpha", q, ": ",
        json_number_array(coeff.α), ", ", q, "beta", q, ": ", json_number_array(coeff.β),
        ", ", q, "bits", q, ": ", json_string_array(hex_word.(bits)), "}")
end

function fermions_task_e_write_reports(io, reports)
    q = Char(34)
    print(io, "{\n")
    for (index, (name, report)) in enumerate(reports)
        index > 1 && print(io, ",\n")
        print(io, "  ", q, name, q, ": {", q, "initial_residual_squared", q, ": ", repr(report.initial),
            ", ", q, "reports", q, ": [")
        for j in 1:length(report.iterations)
            j > 1 && print(io, ", ")
            print(io, "{", q, "shift", q, ": ", repr(name == "refresh" ? LatticeDiracOperators.Rhmc.coeffs_18.β[j] :
                name == "action" ? LatticeDiracOperators.Rhmc.coeffs_m18.β[j] :
                LatticeDiracOperators.Rhmc.coeffs_m14_n10.β[j]),
                ", ", q, "iterations", q, ": ", report.iterations[j],
                ", ", q, "recursive_residual_squared", q, ": ", repr(report.recursive[j]),
                ", ", q, "true_residual_squared", q, ": ", repr(report.true_residuals[j]),
                ", ", q, "absolute_squared_tolerance", q, ": ", repr(FERMIONS_TASK_E_SOLVER_TOLERANCE),
                ", ", q, "maximum_iterations", q, ": ", FERMIONS_TASK_E_MAXSTEPS,
                ", ", q, "convergence_branch", q, ": ", q, "updated_residual", q, "}")
        end
        print(io, "]}")
    end
    print(io, "\n}\n")
end

function generate_fermions_task_e()
    VERSION == "0.7.2" || error("expected Gaugefields.jl v0.7.2, found $VERSION")
    COMMIT == "9e5719970770f4497405a856315c90bef7f74449" ||
        error("expected Gaugefields.jl commit 9e5719970770f4497405a856315c90bef7f74449")
    LATTICEDIRACOPERATORS_VERSION == "0.6.4" ||
        error("expected LatticeDiracOperators.jl v0.6.4, found $LATTICEDIRACOPERATORS_VERSION")
    LATTICEDIRACOPERATORS_COMMIT == "bdef628184597815ba3e0cddf2536df767e78a02" ||
        error("expected LatticeDiracOperators.jl commit bdef628184597815ba3e0cddf2536df767e78a02")

    lattice = (2, 2, 2, 2)
    links = fermions_task_a_links(lattice)
    xi_values = fermions_task_e_xi(lattice)
    xi = fermions_task_e_field(links, xi_values)
    parameters = Dict{String,Any}(
        "Dirac_operator" => "Staggered",
        "mass" => FERMIONS_TASK_E_MASS,
        "verbose_level" => 0,
        "boundarycondition" => Int8[1, 1, 1, -1],
        "eps" => FERMIONS_TASK_E_SOLVER_TOLERANCE,
        "MaxCGstep" => FERMIONS_TASK_E_MAXSTEPS,
        "method_CG" => "cg",
    )
    dirac = LatticeDiracOperators.Dirac_operators.Staggered_Dirac_operator(
        links, xi, parameters)
    normal = LatticeDiracOperators.Dirac_operators.DdagD_Staggered_operator(dirac)
    refresh_coeff = LatticeDiracOperators.Rhmc.coeffs_18
    action_coeff = LatticeDiracOperators.Rhmc.coeffs_m18
    force_coeff = LatticeDiracOperators.Rhmc.coeffs_m14_n10
    refresh_values, _, refresh_report = fermions_task_e_apply(
        refresh_coeff, normal, xi, FERMIONS_TASK_E_SOLVER_TOLERANCE, FERMIONS_TASK_E_MAXSTEPS)

    fermi_action = FermiAction(dirac, Dict("Nf" => 2))
    phi = similar(xi)
    sample_pseudofermions!(phi, links, fermi_action, xi)
    maximum(abs.(phi.f .- refresh_values.f)) <= 2.0e-11 ||
        error("Julia refresh disagrees with the pinned rational replay")
    action_x, _, action_report = fermions_task_e_apply(
        action_coeff, normal, phi, FERMIONS_TASK_E_SOLVER_TOLERANCE, FERMIONS_TASK_E_MAXSTEPS)
    action_value = real(dot(action_x, action_x))
    evaluated_action = evaluate_FermiAction(fermi_action, links, phi)
    abs(action_value - evaluated_action) <= 2.0e-10 ||
        error("Julia action replay disagrees with StaggeredFermiAction")

    force_x, force_shifted, force_report = fermions_task_e_apply(
        force_coeff, normal, phi, FERMIONS_TASK_E_SOLVER_TOLERANCE, FERMIONS_TASK_E_MAXSTEPS)
    force_y = [similar(phi) for _ in 1:force_coeff.n]
    for j in 1:force_coeff.n
        mul!(force_y[j], dirac, force_shifted[j])
    end
    raw_force = [similar(links[1]) for _ in 1:4]
    calc_UdSfdU!(raw_force, fermi_action, links, phi)
    force = initialize_TA_Gaugefields(links)
    for mu in 1:4
        Traceless_antihermitian_add!(force[mu], 1.0, raw_force[mu])
    end

    gauge_action = GaugeAction(links)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop, plaqloop')
    push!(gauge_action, 5.7 / 2, plaqloop)
    initial_momentum = fermions_task_e_momentum(links)
    proposed = similar(links)
    substitute_U!(proposed, links)
    momentum = initialize_TA_Gaugefields(links)
    for mu in 1:4
        momentum[mu].a .= initial_momentum[mu].a
    end
    h_initial = fermions_task_c_hamiltonian(links, gauge_action, momentum, evaluated_action)
    fermions_task_c_trajectory!(proposed, momentum, gauge_action, fermi_action, phi,
        FERMIONS_TASK_E_STEP_SIZE, FERMIONS_TASK_E_STEPS)
    proposed_action = evaluate_FermiAction(fermi_action, proposed, phi)
    h_proposed = fermions_task_c_hamiltonian(proposed, gauge_action, momentum, proposed_action)
    delta_h = h_proposed - h_initial
    probability = delta_h <= 0.0 ? 1.0 : exp(-delta_h)
    acceptance_rng = Random.Xoshiro(FERMIONS_TASK_E_ACCEPTANCE_STATE...)
    acceptance_raw = rand(acceptance_rng, UInt64)
    acceptance_uniform = (Float64(acceptance_raw >>> 12) + 0.5) * 2.0^-52
    accepted = acceptance_uniform <= probability
    next_raw_word = rand(acceptance_rng, UInt64)

    out = joinpath(@__DIR__, "fermions_task_e")
    isdir(out) && rm(out; recursive=true, force=true)
    mkpath(out)
    files = String[]
    function save_field(name, field)
        NPZ.npzwrite(joinpath(out, "$(name)_julia.npy"), copy(field.f))
        NPZ.npzwrite(joinpath(out, "$(name)_rust.npy"), permutedims(field.f, (1, 6, 2, 3, 4, 5)))
        push!(files, "$(name)_julia.npy")
        push!(files, "$(name)_rust.npy")
    end
    for direction in 1:4
        NPZ.npzwrite(joinpath(out, "u$(direction - 1).npy"), links[direction].U)
        push!(files, "u$(direction - 1).npy")
        NPZ.npzwrite(joinpath(out, "force$(direction - 1).npy"), force[direction].a)
        push!(files, "force$(direction - 1).npy")
        NPZ.npzwrite(joinpath(out, "p_initial$(direction - 1).npy"), initial_momentum[direction].a)
        push!(files, "p_initial$(direction - 1).npy")
        NPZ.npzwrite(joinpath(out, "p_final$(direction - 1).npy"), momentum[direction].a)
        push!(files, "p_final$(direction - 1).npy")
        NPZ.npzwrite(joinpath(out, "u_proposed$(direction - 1).npy"), proposed[direction].U)
        push!(files, "u_proposed$(direction - 1).npy")
    end
    save_field("xi", xi)
    save_field("phi", phi)
    save_field("action_x", action_x)
    for j in 1:force_coeff.n
        save_field("force_x$(j - 1)", force_shifted[j])
        save_field("force_y$(j - 1)", force_y[j])
    end
    open(joinpath(out, "rational_reports.json"), "w") do io
        fermions_task_e_write_reports(io, [("refresh", refresh_report), ("action", action_report), ("force", force_report)])
    end
    push!(files, "rational_reports.json")

    refresh_error = fermions_task_e_scalar_error(refresh_coeff, 1.0 / 8.0)
    action_error = fermions_task_e_scalar_error(action_coeff, -1.0 / 8.0)
    force_error = fermions_task_e_scalar_error(force_coeff, -1.0 / 4.0)
    sort!(files)
    q = Char(34)
    open(joinpath(out, "metadata.json"), "w") do io
        print(io, "{\n")
        print(io, "  \"schema\": \"fermions_task_e.v1\",\n")
        print(io, "  \"lattice\": [2, 2, 2, 2],\n  \"nc\": 3,\n  \"components\": 1,\n")
        print(io, "  \"nf\": 2,\n  \"mass\": ", repr(FERMIONS_TASK_E_MASS), ",\n  \"xi_scale\": ", repr(FERMIONS_TASK_E_XI_SCALE), ",\n")
        print(io, "  \"boundaries\": [1, 1, 1, -1],\n")
        print(io, "  \"spectral_bounds\": {\"claimed_lower\": ", repr(FERMIONS_TASK_E_LAMBDA_LOW),
            ", \"claimed_upper\": ", repr(FERMIONS_TASK_E_LAMBDA_HIGH),
            ", \"table_lower\": 0.0004, \"table_upper\": 64.0, \"caller_assertion\": true},\n")
        print(io, "  \"degrees\": {\"refresh\": 15, \"action\": 15, \"md_force\": 10},\n")
        print(io, "  \"solver_parameters\": {\"absolute_squared_tolerance\": ", repr(FERMIONS_TASK_E_SOLVER_TOLERANCE),
            ", \"max_iterations\": ", FERMIONS_TASK_E_MAXSTEPS,
            ", \"julia_solver_keywords\": [\"eps\", \"maxsteps\", \"verbose\"]},\n")
        print(io, "  \"rational_form\": \"R(M)b=alpha0*b+sum_j alpha_j*(M+beta_j I)^-1*b\",\n")
        print(io, "  \"coefficient_roles\": {\"refresh\": \"x^(+1/8) degree-15\", \"action\": \"x^(-1/8) degree-15\", \"md_force\": \"x^(-1/4) degree-10 inverse residues; alpha0 has no link derivative\"},\n")
        print(io, "  \"coefficient_tables\": {\n")
        fermions_task_e_write_coefficients(io, "refresh", refresh_coeff, 1.0 / 8.0, "private coeffs_18")
        print(io, ",\n")
        fermions_task_e_write_coefficients(io, "action", action_coeff, -1.0 / 8.0, "private coeffs_m18")
        print(io, ",\n")
        fermions_task_e_write_coefficients(io, "md_force", force_coeff, -1.0 / 4.0, "private coeffs_m14_n10")
        print(io, "\n  },\n")
        print(io, "  \"scalar_log_grid\": {\"points\": ", FERMIONS_TASK_E_GRID_POINTS,
            ", \"spacing\": \"lambda_low*exp(log(lambda_high/lambda_low)*i/(points-1)); endpoints exact\",
            \"max_abs_error\": {\"refresh\": ", repr(refresh_error),
            ", \"action\": ", repr(action_error), ", \"md_force\": ", repr(force_error),
            "}, \"powers\": {\"refresh\": 0.125, \"action\": -0.125, \"md_force\": -0.25}},\n")
        print(io, "  \"gaugefields_jl\": {\"package\": \"Gaugefields.jl\", \"version\": \"$VERSION\", \"commit\": \"$COMMIT\", \"clean\": true},\n")
        print(io, "  \"latticediracoperators_jl\": {\"package\": \"LatticeDiracOperators.jl\", \"version\": \"$LATTICEDIRACOPERATORS_VERSION\", \"commit\": \"$LATTICEDIRACOPERATORS_COMMIT\", \"clean\": true},\n")
        print(io, "  \"source_urls\": [\n")
        print(io, "    \"https://github.com/shinaoka/LatticeDiracOperators.jl/blob/$LATTICEDIRACOPERATORS_COMMIT/src/action/StaggeredFermiAction.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/LatticeDiracOperators.jl/blob/$LATTICEDIRACOPERATORS_COMMIT/src/rhmc/rhmc.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/LatticeDiracOperators.jl/blob/$LATTICEDIRACOPERATORS_COMMIT/src/StaggeredFermion/StaggeredFermion.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/LatticeDiracOperators.jl/blob/$LATTICEDIRACOPERATORS_COMMIT/src/StaggeredFermion/StaggeredFermion_4D_nowing.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/LatticeDiracOperators.jl/blob/$LATTICEDIRACOPERATORS_COMMIT/src/cgmethods.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/Gaugefields.jl/blob/$COMMIT/src/4D/TA_gaugefields_4D_serial.jl\"\n  ],\n")
        print(io, "  \"source_functions\": [\"StaggeredFermiAction\", \"sample_pseudofermions!\", \"evaluate_FermiAction\", \"calc_UdSfdU!\", \"calc_UdSfdU_fromX!\", \"RHMC\", \"shiftedcg\", \"Traceless_antihermitian_add!\", \"MDstep!\", \"U_update!\", \"P_update!\", \"P_update_fermion!\"],\n")
        print(io, "  \"entrypoint_map\": [\n")
        print(io, "    {\"julia\": \"sample_pseudofermions!\", \"julia_source\": \"src/action/StaggeredFermiAction.jl:176-276\", \"rust\": \"staggered_action.rs::StaggeredFermiAction::sample_pseudofermion\"},\n")
        print(io, "    {\"julia\": \"evaluate_FermiAction\", \"julia_source\": \"src/action/StaggeredFermiAction.jl:98-142\", \"rust\": \"staggered_action.rs::StaggeredFermiAction::evaluate\"},\n")
        print(io, "    {\"julia\": \"calc_UdSfdU!/calc_UdSfdU_fromX!\", \"julia_source\": \"src/action/StaggeredFermiAction.jl:278-422\", \"rust\": \"staggered_action.rs::StaggeredFermiAction::force/force_from_shifted_xy\"},\n")
        print(io, "    {\"julia\": \"RHMC\", \"julia_source\": \"src/rhmc/rhmc.jl:1-1294\", \"rust\": \"rhmc.rs::typed private coefficient roles\"},\n")
        print(io, "    {\"julia\": \"shiftedcg\", \"julia_source\": \"src/cgmethods.jl:872-968\", \"rust\": \"solvers.rs::multi_shift_cg\"},\n")
        print(io, "    {\"julia\": \"MDstep!/U_update!/P_update!/P_update_fermion!\", \"julia_source\": \"test/wilsonhmc.jl:46-146\", \"rust\": \"rhmc.rs::staggered_hmc_update/staggered_leapfrog_trajectory\"},\n")
        print(io, "    {\"julia\": \"Traceless_antihermitian_add!\", \"julia_source\": \"Gaugefields.jl/src/4D/TA_gaugefields_4D_serial.jl:181-269\", \"rust\": \"Mat3::add_ta_coefficients\"}\n  ],\n")
        print(io, "  \"layout\": {\"julia_shape\": \"[3,NX,NY,NZ,NT,1]\", \"rust_shape\": \"[3,1,NX,NY,NZ,NT]\", \"conversion\": \"permutedims(array, (1,6,2,3,4,5))\", \"permutation\": [1,6,2,3,4,5], \"force\": \"Float64 Fortran [gell_mann_component,x,y,z,t]\"},\n")
        print(io, "  \"refresh\": {\"flavors\": 2, \"formula\": \"phi=R_(+1/8)(M)xi\", \"complex_normal_scale\": \"1/sqrt(2) per independent real and imaginary standard normal\", \"xi\": \"explicit deterministic field\"},\n")
        print(io, "  \"action\": {\"formula\": \"||R_(-1/8)(M)phi||^2\", \"x_name\": \"X=R_(-1/8)(M)phi\", \"value\": ", repr(action_value), ", \"evaluated_value\": ", repr(evaluated_action), "},\n")
        print(io, "  \"force\": {\"x_name\": \"X_j=(M+beta_j I)^-1 phi\", \"y_name\": \"Y_j=D X_j\", \"outer_products\": 2, \"projection_count\": 1, \"formula\": \"0.5*alpha_j*((eta U Xplus) outer Ydag + X outer (eta U Yplus)dag)\", \"wrapped_boundary_sign\": \"applied once to each shifted plus field\", \"gauge_1_over_nc\": \"not applied here\"},\n")
        print(io, "  \"trajectory\": {\"beta\": 5.7, \"step_size\": ", repr(FERMIONS_TASK_E_STEP_SIZE),
            ", \"steps\": ", FERMIONS_TASK_E_STEPS, ", \"initial_hamiltonian\": ", repr(h_initial),
            ", \"proposed_hamiltonian\": ", repr(h_proposed), ", \"delta_h\": ", repr(delta_h),
            ", \"acceptance_probability\": ", repr(probability), ", \"accepted\": ", accepted,
            ", \"acceptance_rng_state\": [", join(string.(FERMIONS_TASK_E_ACCEPTANCE_STATE), ", "),
            "], \"acceptance_uniform\": ", repr(acceptance_uniform),
            ", \"acceptance_uniform_bits\": ", reinterpret(UInt64, acceptance_uniform),
            ", \"next_raw_word\": ", next_raw_word, "},\n")
        print(io, "  \"comparison\": {\"field_max_abs_tolerance\": 2.0e-9, \"action_tolerance\": 2.0e-9, \"force_tolerance\": ", repr(FERMIONS_TASK_E_FORCE_TOLERANCE),
            ", \"finite_difference_tolerance\": ", repr(FERMIONS_TASK_E_FD_TOLERANCE),
            ", \"finite_difference_epsilons\": ", json_number_array(FERMIONS_TASK_E_FD_EPSILONS),
            ", \"finite_difference_series\": {\"epsilons\": ", json_number_array(FERMIONS_TASK_E_FD_EPSILONS),
            ", \"max_residuals\": ", json_number_array(FERMIONS_TASK_E_FD_MAX_RESIDUALS),
            ", \"global_max_ratios\": ", json_number_array(FERMIONS_TASK_E_FD_RATIOS),
            ", \"pass_counts\": ", json_number_array(FERMIONS_TASK_E_FD_PASS_COUNTS),
            ", \"tolerance\": ", repr(FERMIONS_TASK_E_FD_TOLERANCE),
            ", \"selected_epsilon\": ", repr(FERMIONS_TASK_E_FD_SELECTED_EPSILON),
            ", \"selected_pass_count\": 512, \"coefficient_count\": 512",
            ", \"construction\": \"central U <- exp(epsilon*T_a)U; finite differences of StaggeredFermiAction::evaluate; global maxima over 4*16*8 coefficients\"}",
            ", \"reversibility_tolerance\": 5.0e-9, \"criterion\": \"maximum absolute payload residual; central force FD; reversibility\"},\n")
        print(io, "  \"files\": [", join([q * file * q for file in files], ", "), "],\n")
        print(io, "  \"generator\": {\"script\": \"fixtures/generate.jl\", \"mode\": \"fermions_task_e\", \"randomness\": \"none for fixed xi/phi/momentum; explicit Xoshiro only for acceptance word\"}\n")
        print(io, "}\n")
    end
end

if FERMIONS_TASK_E_MODE
    generate_fermions_task_e()
    exit()
end

const FERMIONS_TASK_E_ENSEMBLE_MASTER_SEED = 2026081901
const FERMIONS_TASK_E_ENSEMBLE_BURN_IN = 4
const FERMIONS_TASK_E_ENSEMBLE_BLOCKS = 3
const FERMIONS_TASK_E_ENSEMBLE_TRAJECTORIES_PER_BLOCK = 4
const FERMIONS_TASK_E_ENSEMBLE_QCDMEASUREMENTS_VERSION = "0.2.13"
const FERMIONS_TASK_E_ENSEMBLE_QCDMEASUREMENTS_COMMIT =
    "9e04c37bbd68712cf7a749ae5aff10eb6aae4566"

function fermions_task_e_ensemble_summary(values, block_means)
    mean_value = sum(values) / length(values)
    variance = sum((value - mean_value)^2 for value in block_means) /
        (length(block_means) - 1)
    return (
        block_means=block_means,
        mean=mean_value,
        standard_error=sqrt(variance / length(block_means)),
    )
end

function fermions_task_e_ensemble_write_summary(io, name, summary)
    q = Char(34)
    print(io, "    ", q, name, q, ": {\"block_means\": ",
        json_number_array(summary.block_means), ", \"mean\": ", repr(summary.mean),
        ", \"standard_error\": ", repr(summary.standard_error), "}")
end

function fermions_task_e_ensemble_write_measurement(io, record)
    q = Char(34)
    print(io, "    {\"measurement\": ", record.measurement,
        ", \"global_trajectory\": ", record.global_trajectory,
        ", \"block\": ", record.block,
        ", \"trajectory_in_block\": ", record.trajectory_in_block,
        ", \"trajectory_seed\": ", record.trajectory_seed,
        ", \"source_seeds\": ", json_number_array(record.source_seeds),
        ", \"chiral_source_values\": ", json_number_array(record.chiral_source_values),
        ", \"plaquette\": ", repr(record.plaquette),
        ", \"chiral_condensate\": ", repr(record.chiral_condensate),
        ", \"delta_h\": ", repr(record.delta_h),
        ", \"acceptance_probability\": ", repr(record.acceptance_probability),
        ", \"acceptance_uniform\": ", repr(record.acceptance_uniform),
        ", \"accepted\": ", record.accepted, "}")
end

function generate_fermions_task_e_ensemble()
    VERSION == "0.7.2" || error("expected Gaugefields.jl v0.7.2, found $VERSION")
    COMMIT == "9e5719970770f4497405a856315c90bef7f74449" ||
        error("expected Gaugefields.jl commit 9e5719970770f4497405a856315c90bef7f74449")
    LATTICEDIRACOPERATORS_VERSION == "0.6.4" ||
        error("expected LatticeDiracOperators.jl v0.6.4, found $LATTICEDIRACOPERATORS_VERSION")
    LATTICEDIRACOPERATORS_COMMIT == "bdef628184597815ba3e0cddf2536df767e78a02" ||
        error("expected LatticeDiracOperators.jl commit bdef628184597815ba3e0cddf2536df767e78a02")
    WILSONLOOP_VERSION == "0.1.5" ||
        error("expected Wilsonloop.jl v0.1.5, found $WILSONLOOP_VERSION")
    WILSONLOOP_COMMIT == "e1a617fdedb19b785f89bdeb13c30e53b20743a7" ||
        error("expected Wilsonloop.jl commit e1a617fdedb19b785f89bdeb13c30e53b20743a7")

    lattice = (2, 2, 2, 2)
    beta = 5.7
    nf = 2
    boundary = Int8[1, 1, 1, -1]
    parameters = Dict{String,Any}(
        "Dirac_operator" => "Staggered",
        "mass" => FERMIONS_TASK_E_MASS,
        "verbose_level" => 0,
        "boundarycondition" => boundary,
        "eps" => FERMIONS_TASK_E_SOLVER_TOLERANCE,
        "MaxCGstep" => FERMIONS_TASK_E_MAXSTEPS,
        "method_CG" => "cg",
    )
    chiral_parameters = copy(parameters)
    chiral_parameters["method_CG"] = "bicg"

    Random.seed!(2026081901)
    links = Initialize_Gaugefields(NC, 0, lattice...; condition="cold")
    xi = fermions_task_e_field(
        links,
        zeros(ComplexF64, NC, lattice[1], lattice[2], lattice[3], lattice[4], 1),
    )
    phi = similar(xi)
    source = similar(xi)
    solution = similar(xi)
    dirac = LatticeDiracOperators.Dirac_operators.Staggered_Dirac_operator(
        links, xi, parameters)
    fermi_action = FermiAction(dirac, Dict("Nf" => nf))

    gauge_action = GaugeAction(links)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop, plaqloop')
    push!(gauge_action, beta / 2, plaqloop)
    momentum = initialize_TA_Gaugefields(links)
    old_links = similar(links)
    plaq_temp1 = similar(links[1])
    plaq_temp2 = similar(links[1])

    total_measurements = FERMIONS_TASK_E_ENSEMBLE_BLOCKS *
        FERMIONS_TASK_E_ENSEMBLE_TRAJECTORIES_PER_BLOCK
    total_trajectories = FERMIONS_TASK_E_ENSEMBLE_BURN_IN + total_measurements
    trajectory_seeds = [
        FERMIONS_TASK_E_ENSEMBLE_MASTER_SEED + trajectory - 1
        for trajectory in 1:total_trajectories
    ]
    source_seeds = Vector{Vector{Int}}()
    measurements = Any[]
    block_measurements = [Any[] for _ in 1:FERMIONS_TASK_E_ENSEMBLE_BLOCKS]
    burn_in_delta_h = Float64[]
    burn_in_accepted = Bool[]

    measurement_index = 0
    for global_trajectory in 1:total_trajectories
        trajectory_seed = trajectory_seeds[global_trajectory]
        Random.seed!(trajectory_seed)
        gauss_distribution!(momentum)
        gauss_sampling_in_action!(xi, links, fermi_action)
        sample_pseudofermions!(phi, links, fermi_action, xi)

        substitute_U!(old_links, links)
        initial_action = evaluate_FermiAction(fermi_action, links, phi)
        initial_hamiltonian = fermions_task_c_hamiltonian(
            links, gauge_action, momentum, initial_action)
        fermions_task_c_trajectory!(
            links,
            momentum,
            gauge_action,
            fermi_action,
            phi,
            FERMIONS_TASK_E_STEP_SIZE,
            FERMIONS_TASK_E_STEPS,
        )
        proposed_action = evaluate_FermiAction(fermi_action, links, phi)
        proposed_hamiltonian = fermions_task_c_hamiltonian(
            links, gauge_action, momentum, proposed_action)
        delta_h = proposed_hamiltonian - initial_hamiltonian
        acceptance_probability = delta_h <= 0.0 ? 1.0 : exp(-delta_h)
        acceptance_uniform = rand()
        accepted = acceptance_uniform <= acceptance_probability
        accepted || substitute_U!(links, old_links)

        if global_trajectory <= FERMIONS_TASK_E_ENSEMBLE_BURN_IN
            push!(burn_in_delta_h, delta_h)
            push!(burn_in_accepted, accepted)
            continue
        end

        measurement_index += 1
        block = div(measurement_index - 1,
            FERMIONS_TASK_E_ENSEMBLE_TRAJECTORIES_PER_BLOCK) + 1
        trajectory_in_block = mod(measurement_index - 1,
            FERMIONS_TASK_E_ENSEMBLE_TRAJECTORIES_PER_BLOCK) + 1
        chiral_dirac = LatticeDiracOperators.Dirac_operators.Staggered_Dirac_operator(
            links, source, chiral_parameters)
        current_source_seeds = [
            FERMIONS_TASK_E_ENSEMBLE_MASTER_SEED + 100_000 +
            2 * (measurement_index - 1) + source_index
            for source_index in 1:2
        ]
        current_source_values = Float64[]
        for source_seed in current_source_seeds
            Random.seed!(source_seed)
            for index in eachindex(source.f)
                k = rand(0:3)
                source.f[index] = cis(k * pi / 2)
            end
            clear_fermion!(solution)
            solve_DinvX!(solution, chiral_dirac, source)
            push!(current_source_values,
                (nf / 4) / links[1].NV * real(dot(source, solution)))
        end
        chiral_condensate = sum(current_source_values) / length(current_source_values)
        normalized_plaquette = real(calculate_Plaquette(
            links, plaq_temp1, plaq_temp2)) / (6 * links[1].NV * links[1].NC)
        record = (
            measurement=measurement_index,
            global_trajectory=global_trajectory,
            block=block,
            trajectory_in_block=trajectory_in_block,
            trajectory_seed=trajectory_seed,
            source_seeds=current_source_seeds,
            chiral_source_values=current_source_values,
            plaquette=normalized_plaquette,
            chiral_condensate=chiral_condensate,
            delta_h=delta_h,
            acceptance_probability=acceptance_probability,
            acceptance_uniform=acceptance_uniform,
            accepted=accepted,
        )
        push!(source_seeds, current_source_seeds)
        push!(measurements, record)
        push!(block_measurements[block], record)
    end

    plaquette_values = [record.plaquette for record in measurements]
    chiral_values = [record.chiral_condensate for record in measurements]
    delta_h_values = [record.delta_h for record in measurements]
    block_plaquette_means = [
        sum(record.plaquette for record in block) / length(block)
        for block in block_measurements
    ]
    block_chiral_means = [
        sum(record.chiral_condensate for record in block) / length(block)
        for block in block_measurements
    ]
    block_delta_h_means = [
        sum(record.delta_h for record in block) / length(block)
        for block in block_measurements
    ]
    plaquette_summary = fermions_task_e_ensemble_summary(
        plaquette_values, block_plaquette_means)
    chiral_summary = fermions_task_e_ensemble_summary(
        chiral_values, block_chiral_means)
    delta_h_summary = fermions_task_e_ensemble_summary(
        delta_h_values, block_delta_h_means)
    accepted_count = count(record -> record.accepted, measurements)

    out = joinpath(@__DIR__, "fermions_task_e_ensemble")
    isdir(out) && rm(out; recursive=true, force=true)
    mkpath(out)
    q = Char(34)
    open(joinpath(out, "metadata.json"), "w") do io
        print(io, "{\n")
        print(io, "  \"schema\": \"fermions_task_e_ensemble.v1\",\n")
        print(io, "  \"lattice\": [2, 2, 2, 2],\n")
        print(io, "  \"nc\": 3,\n  \"beta\": ", repr(beta), ",\n")
        print(io, "  \"mass\": ", repr(FERMIONS_TASK_E_MASS), ",\n")
        print(io, "  \"nf\": ", nf, ",\n")
        print(io, "  \"boundaries\": [1, 1, 1, -1],\n")
        print(io, "  \"spectral_bounds\": {\"claimed_lower\": ",
            repr(FERMIONS_TASK_E_LAMBDA_LOW),
            ", \"claimed_upper\": ", repr(FERMIONS_TASK_E_LAMBDA_HIGH),
            ", \"coefficient_interval\": [0.0004, 64.0]},\n")
        print(io, "  \"degrees\": {\"refresh\": 15, \"action\": 15, \"md_force\": 10},\n")
        print(io, "  \"solver_parameters\": {\"absolute_squared_tolerance\": ",
            repr(FERMIONS_TASK_E_SOLVER_TOLERANCE),
            ", \"max_iterations\": ", FERMIONS_TASK_E_MAXSTEPS,
            ", \"trajectory_method\": \"cg\", \"chiral_method\": \"bicg\", \"julia_keys\": [\"Dirac_operator\", \"mass\", \"verbose_level\", \"boundarycondition\", \"eps\", \"MaxCGstep\", \"method_CG\"]},\n")
        print(io, "  \"schedule\": {\"initial_condition\": \"cold\", \"burn_in_trajectories\": ",
            FERMIONS_TASK_E_ENSEMBLE_BURN_IN,
            ", \"blocks\": ", FERMIONS_TASK_E_ENSEMBLE_BLOCKS,
            ", \"trajectories_per_block\": ", FERMIONS_TASK_E_ENSEMBLE_TRAJECTORIES_PER_BLOCK,
            ", \"measured_trajectories\": ", total_measurements,
            ", \"dt\": ", repr(FERMIONS_TASK_E_STEP_SIZE),
            ", \"steps\": ", FERMIONS_TASK_E_STEPS,
            ", \"measurement\": \"after each trajectory; rejected links are restored before measurement\",\n")
        print(io, "    \"integrator\": \"U <- exp((dt/2)P)U; P <- P - dt*(gauge_force/NC + fermion_force); U <- exp((dt/2)P)U\",\n")
        print(io, "    \"acceptance\": \"unconditional rand() draw; accept iff rand() <= min(1, exp(-delta_h)); rejected links roll back\"},\n")
        print(io, "  \"seeds\": {\"master\": 2026081901, \"trajectory_seeds\": {\"burn_in\": ",
            json_number_array(trajectory_seeds[1:FERMIONS_TASK_E_ENSEMBLE_BURN_IN]),
            ", \"measured\": ",
            json_number_array(trajectory_seeds[(FERMIONS_TASK_E_ENSEMBLE_BURN_IN + 1):end]),
            "}, \"source_seeds\": [")
        for (index, seeds) in enumerate(source_seeds)
            index > 1 && print(io, ", ")
            print(io, json_number_array(seeds))
        end
        print(io, "], \"stream_policy\": \"Random.seed!(trajectory_seed) before each pinned momentum/pseudofermion/acceptance sequence; Random.seed!(source_seed) before each pinned Z4 source\"},\n")
        print(io, "  \"normalization\": {\"plaquette\": \"real(calculate_Plaquette(U,temp1,temp2)) / (6 * NV * NC)\", \"chiral_condensate\": \"(Nf/4) / NV * Re(dot(r, D^-1*r))\", \"nv\": ", links[1].NV,
            ", \"nf_over_four\": ", repr(nf / 4),
            ", \"standard_error\": \"sample_stddev(block_means) / sqrt(number_of_blocks)\"},\n")
        print(io, "  \"source_generation\": {\"sources_per_configuration\": 2, \"distribution\": \"canonical Z4 implemented in the fixture\", \"seed_call\": \"Random.seed!(source_seed) immediately before each source\", \"source_formula\": \"theta = rand(0:3)*pi/2; r = cos(theta) + im*sin(theta)\"},\n")
        print(io, "  \"measurements\": [\n")
        for (index, record) in enumerate(measurements)
            index > 1 && print(io, ",\n")
            fermions_task_e_ensemble_write_measurement(io, record)
        end
        print(io, "\n  ],\n")
        print(io, "  \"blocks\": [\n")
        for block in 1:length(block_measurements)
            block > 1 && print(io, ",\n")
            records = block_measurements[block]
            print(io, "    {\"block\": ", block, ", \"measurements\": ",
                json_number_array([record.measurement for record in records]),
                ", \"plaquette_mean\": ", repr(block_plaquette_means[block]),
                ", \"chiral_condensate_mean\": ", repr(block_chiral_means[block]),
                ", \"delta_h_mean\": ", repr(block_delta_h_means[block]), "}")
        end
        print(io, "\n  ],\n")
        print(io, "  \"statistics\": {\n")
        fermions_task_e_ensemble_write_summary(io, "plaquette", plaquette_summary)
        print(io, ",\n")
        fermions_task_e_ensemble_write_summary(io, "chiral_condensate", chiral_summary)
        print(io, ",\n")
        fermions_task_e_ensemble_write_summary(io, "delta_h", delta_h_summary)
        print(io, ",\n    \"mean_delta_h\": ", repr(delta_h_summary.mean),
            ",\n    \"acceptance\": {\"accepted\": ", accepted_count,
            ", \"total\": ", total_measurements,
            ", \"rate\": ", repr(accepted_count / total_measurements), "}\n  },\n")
        print(io, "  \"burn_in_summary\": {\"delta_h\": ",
            json_number_array(burn_in_delta_h),
            ", \"accepted\": ", json_number_array(Int.(burn_in_accepted)), "},\n")
        print(io, "  \"provenance\": {\n")
        print(io, "    \"julia\": {\"version\": \"", Base.VERSION,
            "\", \"source_commit\": \"", Base.GIT_VERSION_INFO.commit, "\"},\n")
        print(io, "    \"gaugefields_jl\": {\"package\": \"Gaugefields.jl\", \"version\": \"", VERSION,
            "\", \"commit\": \"", COMMIT, "\", \"clean\": true},\n")
        print(io, "    \"latticediracoperators_jl\": {\"package\": \"LatticeDiracOperators.jl\", \"version\": \"", LATTICEDIRACOPERATORS_VERSION,
            "\", \"commit\": \"", LATTICEDIRACOPERATORS_COMMIT, "\", \"clean\": true},\n")
        print(io, "    \"wilsonloop_jl\": {\"package\": \"Wilsonloop.jl\", \"version\": \"", WILSONLOOP_VERSION,
            "\", \"commit\": \"", WILSONLOOP_COMMIT, "\", \"clean\": true},\n")
        print(io, "    \"qcdmeasurements_jl\": {\"package\": \"QCDMeasurements.jl\", \"version\": \"",
            FERMIONS_TASK_E_ENSEMBLE_QCDMEASUREMENTS_VERSION,
            "\", \"commit\": \"", FERMIONS_TASK_E_ENSEMBLE_QCDMEASUREMENTS_COMMIT, "\"}\n  },\n")
        print(io, "  \"source_functions\": [\"StaggeredFermiAction\", \"FermiAction\", \"gauss_sampling_in_action!\", \"sample_pseudofermions!\", \"evaluate_FermiAction\", \"calc_UdSfdU!\", \"shiftedcg\", \"gauss_distribution!\", \"gauss_distribution_fermion!\", \"solve_DinvX!\", \"GaugeAction\", \"make_loops_fromname\", \"calculate_Plaquette\", \"Traceless_antihermitian_add!\", \"exptU!\", \"Random.seed!\", \"rand\"],\n")
        print(io, "  \"source_urls\": [\n")
        print(io, "    \"https://github.com/shinaoka/LatticeDiracOperators.jl/blob/", LATTICEDIRACOPERATORS_COMMIT, "/src/action/StaggeredFermiAction.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/LatticeDiracOperators.jl/blob/", LATTICEDIRACOPERATORS_COMMIT, "/src/rhmc/rhmc.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/LatticeDiracOperators.jl/blob/", LATTICEDIRACOPERATORS_COMMIT, "/src/Diracoperators.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/LatticeDiracOperators.jl/blob/", LATTICEDIRACOPERATORS_COMMIT, "/src/AbstractFermions_4D.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/LatticeDiracOperators.jl/blob/", LATTICEDIRACOPERATORS_COMMIT, "/test/wilsonhmc.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/Gaugefields.jl/blob/", COMMIT, "/src/action/GaugeActions.jl\",\n")
        print(io, "    \"https://github.com/shinaoka/Gaugefields.jl/blob/", COMMIT, "/src/4D/TA_gaugefields_4D_serial.jl\",\n")
        print(io, "    \"https://github.com/akio-tomiya/Wilsonloop.jl/blob/", WILSONLOOP_COMMIT, "/src/Wilsonloop.jl\",\n")
        print(io, "    \"https://github.com/akio-tomiya/QCDMeasurements.jl/blob/", FERMIONS_TASK_E_ENSEMBLE_QCDMEASUREMENTS_COMMIT, "/src/measurements/measure_chiral_condensate.jl\"\n  ],\n")
        print(io, "  \"upstream_issues\": [{\"package\": \"LatticeDiracOperators.jl\", \"revision\": \"", LATTICEDIRACOPERATORS_COMMIT,
            "\", \"function\": \"Z4_distribution_fermi!\", \"detail\": \"The pinned implementation uses theta=rand(0:3)*pi/4, not the canonical 2*pi/4 phase grid; this fixture avoids that biased source and implements canonical Z4 explicitly.\"}],\n")
        print(io, "  \"generator\": {\"script\": \"fixtures/generate.jl\", \"mode\": \"fermions_task_e_ensemble\", \"files\": [\"metadata.json\"], \"randomness\": \"explicit Julia task-local seeds; no Rust or binary payloads\"}\n")
        print(io, "}\n")
    end
end

if FERMIONS_TASK_E_ENSEMBLE_MODE
    generate_fermions_task_e_ensemble()
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
