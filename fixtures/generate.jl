import Pkg
import Random

if !(isempty(ARGS) || ARGS == ["reproducible_rng"])
    error("usage: julia --startup-file=no fixtures/generate.jl [reproducible_rng]")
end

hex_word(value::UInt64) = "0x" * lpad(string(value, base=16), 16, '0')
json_string_array(values) = "[" * join([string(Char(34), value, Char(34)) for value in values], ", ") * "]"

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
Pkg.activate(REQUESTED_CHECKOUT)
using Gaugefields
using NPZ
using LinearAlgebra

const NC = 3
const BETA = 6.0
const HMC_EPSILON = 0.5
const HMC_DT = 0.125
const VERSION = string(Base.pkgversion(Gaugefields))
const CHECKOUT = dirname(dirname(pathof(Gaugefields)))
const COMMIT = readchomp(`git -C $CHECKOUT rev-parse HEAD`)
const DIRTY = read(`git -C $CHECKOUT status --porcelain --untracked-files=no`, String)
isempty(strip(DIRTY)) || error("refusing fixture provenance from dirty Gaugefields.jl checkout: $CHECKOUT")

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
    if reproducible
        # Gaugefields.jl deliberately resets StableRNG(123) for each direction.
        # Shift each direction along its matching lattice axis so the fixture
        # detects direction swaps while preserving every site-local SU(3) value.
        for mu in 1:4
            shifts = ntuple(axis -> axis == mu + 2 ? 1 : 0, 6)
            links[mu].U .= circshift(links[mu].U, shifts)
        end
    end
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

generate_reproducible_rng()
generate("cold_1x1x1x1", (1, 1, 1, 1), "cold")
generate("random_2x2x2x2", (2, 2, 2, 2), "hot"; reproducible=true, write_observables=true)
generate("random_4x4x4x4", (4, 4, 4, 4), "hot"; reproducible=true, write_observables=true)
generate("shifts_3x2x4x5", (3, 2, 4, 5), "hot"; reproducible=true, write_shifts=true)
generate_exp_ta()
generate_normalize_su3()
