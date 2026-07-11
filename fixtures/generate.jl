import Pkg
const REQUESTED_CHECKOUT = get(ENV, "GAUGEFIELDS_JL_DIR", nothing)
isnothing(REQUESTED_CHECKOUT) && error("set GAUGEFIELDS_JL_DIR to a clean Gaugefields.jl checkout")
Pkg.activate(REQUESTED_CHECKOUT)
using Gaugefields
using NPZ
using LinearAlgebra

const NC = 3
const BETA = 6.0
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

generate("cold_1x1x1x1", (1, 1, 1, 1), "cold")
generate("random_2x2x2x2", (2, 2, 2, 2), "hot"; reproducible=true, write_observables=true)
generate("random_4x4x4x4", (4, 4, 4, 4), "hot"; reproducible=true, write_observables=true)
generate("shifts_3x2x4x5", (3, 2, 4, 5), "hot"; reproducible=true, write_shifts=true)
