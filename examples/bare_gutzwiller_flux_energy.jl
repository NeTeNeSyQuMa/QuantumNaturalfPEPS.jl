# Bare finite-flux Gutzwiller VMC, without QuantumNaturalGradient.
#
# The D=1 PPEPS entries are fixed to 1/sqrt(2), so their product is a
# configuration-independent factor. Sampling and local-energy evaluation are
# performed directly with fixed-Sz determinant-ratio Metropolis updates.
#
# Install the optional plotting environment once from the repository root:
#   julia --project=examples -e 'using Pkg; Pkg.instantiate()'
#
# Run with the package as the primary environment:
#   julia --project=. examples/bare_gutzwiller_flux_energy.jl

examples_environment = @__DIR__
examples_environment in LOAD_PATH || push!(LOAD_PATH, examples_environment)

using CairoMakie
using ITensors
using ITensorMPS
using JLD2
using LaTeXStrings
using LinearAlgebra
using Makie.Colors
using MathTeXEngine
using QuantumNaturalfPEPS
using Random
using Statistics

theme = begin
    fonts = (;
        regular=texfont(:text),
        bold=texfont(:bold),
        italic=texfont(:italic),
        bold_italic=texfont(:bolditalic),
    )
    Theme(
        fonts=fonts,
        fontsize=25,
        Colorbar=(; size=25, labelsize=40),
        Heatmap=(; colormap=:berlin),
        Label=(; lineheight=0.9),
        Scatter=(; markersize=9),
    )
end
set_theme!(theme)

parse_int_list(value) = parse.(Int, strip.(split(value, ',')))
default_flux_grid = join((string(k / 36) for k in 0:18), ',')

linear_sizes = parse_int_list(get(ENV, "PPEPS_LINEAR_SIZES", "6,18"))
number_of_samples = parse(Int, get(ENV, "PPEPS_MEASUREMENT_SAMPLES", "500"))
burnin_sweeps = parse(Int, get(ENV, "PPEPS_BURNIN_SWEEPS", "5"))
block_length = parse(Int, get(ENV, "PPEPS_BLOCK_LENGTH", "25"))
moves_per_sample = parse(Int, get(ENV, "PPEPS_MOVES_PER_SAMPLE", "20"))
base_seed = parse(Int, get(ENV, "PPEPS_SEED", "8128"))

# The QNG run performs three Oks/Eks evaluations before its final measurement.
# Using the same offset gives both final measurements the same initial random
# stream, although their chains diverge if the optimized PPEPS changes an
# acceptance decision.
measurement_seed_offset = parse(Int, get(ENV, "PPEPS_MEASUREMENT_SEED_OFFSET", "3"))

requested_flux_densities = parse.(
    Float64,
    strip.(split(get(ENV, "PPEPS_FLUX_DENSITIES", default_flux_grid), ',')),
)
J1 = parse(Float64, get(ENV, "PPEPS_J1", "1.0"))
J2 = parse(Float64, get(ENV, "PPEPS_J2", string(J1 / 8)))
field = parse(Float64, get(ENV, "PPEPS_FIELD", "0.0"))
magnetizations = (1 // 3, 2 // 3)
closed_shell_tolerance = parse(Float64, get(ENV, "PPEPS_MIN_GAP", "1e-8"))

all(iseven, linear_sizes) || error("all linear sizes must be even for the staggered-pi gauge")
number_of_samples >= 2block_length || error(
    "PPEPS_MEASUREMENT_SAMPLES must be at least twice PPEPS_BLOCK_LENGTH",
)
burnin_sweeps >= 0 || error("PPEPS_BURNIN_SWEEPS must be nonnegative")
moves_per_sample > 0 || error("PPEPS_MOVES_PER_SAMPLE must be positive")
all(density -> 0 <= density <= 0.5, requested_flux_densities) || error(
    "PPEPS_FLUX_DENSITIES entries must lie in [0, 0.5]",
)

function identity_D1_peps(Lx::Int, Ly::Int)
    hilbert = ITensors.siteinds("S=1/2", Lx, Ly)
    peps = PEPS(ComplexF64, hilbert; bond_dim=1, show_warning=false)
    write!(peps, fill(ComplexF64(inv(sqrt(2))), length(peps)))
    return peps
end

function D1_local_weights(peps)
    peps.bond_dim == 1 || throw(ArgumentError("bare sampler requires bond_dim=1"))
    weights = reshape(ComplexF64.(vec(peps)), 2, :)
    all(!iszero, weights) || error("D=1 PPEPS entries must be nonzero")
    return weights
end

@inline function D1_exchange_ratio(weights, configuration, i, j)
    spin_i = configuration[i] + 1
    spin_j = configuration[j] + 1
    return weights[spin_j, i] * weights[spin_i, j] /
           (weights[spin_i, i] * weights[spin_j, j])
end

function random_exchange_cache(rng, projected_state; maximum_attempts=100)
    Nup = something(projected_state.Nup)
    for _ in 1:maximum_attempts
        configuration = ones(Int, projected_state.N)
        configuration[randperm(rng, projected_state.N)[1:Nup]] .= 0
        try
            cache = GutzwillerExchangeCache(projected_state, configuration)
            all(isfinite, cache.inverse_slater) && return cache
        catch exception
            exception isa SingularException || rethrow()
        end
    end
    error("failed to find a nonsingular fixed-Sz Gutzwiller configuration")
end

function metropolis_exchange!(rng, cache, weights, up_sites, down_sites)
    up_position = rand(rng, eachindex(up_sites))
    down_position = rand(rng, eachindex(down_sites))
    i = up_sites[up_position]
    j = down_sites[down_position]
    amplitude_ratio = gutzwiller_exchange_ratio(cache, i, j) *
                      D1_exchange_ratio(weights, cache.configuration, i, j)

    if rand(rng) < min(1.0, abs2(amplitude_ratio))
        accept_gutzwiller_exchange!(cache, i, j)
        up_sites[up_position] = j
        down_sites[down_position] = i
        return true
    end
    return false
end

function exchange_bonds(Lx, Ly)
    bonds = Tuple{Int,Int,Float64}[]
    for (shell, coupling) in ((1, J1), (2, J2))
        iszero(coupling) && continue
        for bond in triangular_torus_bonds(Lx, Ly; shell)
            i = bond.source[1] + (bond.source[2] - 1) * Lx
            j = bond.target[1] + (bond.target[2] - 1) * Lx
            push!(bonds, (i, j, Float64(coupling)))
        end
    end
    return bonds
end

@inline spin_z(spin) = spin == 0 ? 0.5 : -0.5

function bare_local_energy(cache, weights, bonds)
    configuration = cache.configuration
    energy = 0.0 + 0.0im
    for (i, j, coupling) in bonds
        energy += coupling * spin_z(configuration[i]) * spin_z(configuration[j])
        if configuration[i] != configuration[j]
            amplitude_ratio = gutzwiller_exchange_ratio(cache, i, j) *
                              D1_exchange_ratio(weights, configuration, i, j)
            energy += coupling * amplitude_ratio / 2
        end
    end
    energy -= field * sum(spin_z, configuration)
    return energy
end

function blocked_statistics(values, block_length)
    number_of_blocks = fld(length(values), block_length)
    number_of_blocks >= 2 || error("at least two complete blocks are required")
    used_values = view(values, 1:number_of_blocks * block_length)
    block_means = [
        mean(real, view(used_values, (block - 1) * block_length + 1:block * block_length))
        for block in 1:number_of_blocks
    ]
    return (;
        mean=mean(block_means),
        error=std(block_means) / sqrt(number_of_blocks),
        variance=var(real.(used_values)),
        imaginary_mean=mean(imag, used_values),
        number_of_blocks,
    )
end

function bare_projected_energy(rng, projected_state, peps, bonds)
    weights = D1_local_weights(peps)
    cache = random_exchange_cache(rng, projected_state)
    up_sites = findall(==(0), cache.configuration)
    down_sites = findall(==(1), cache.configuration)
    accepted = 0
    attempted = 0

    for _ in 1:(burnin_sweeps * projected_state.N)
        accepted += metropolis_exchange!(rng, cache, weights, up_sites, down_sites)
        attempted += 1
    end

    local_energies = Vector{ComplexF64}(undef, number_of_samples)
    for sample in 1:number_of_samples
        for _ in 1:moves_per_sample
            accepted += metropolis_exchange!(rng, cache, weights, up_sites, down_sites)
            attempted += 1
        end
        local_energies[sample] = bare_local_energy(cache, weights, bonds)
    end
    return merge(
        blocked_statistics(local_energies, block_length),
        (; acceptance=accepted / attempted),
    )
end

function flux_state(L, Q, magnetization)
    N = L^2
    Nup = round(Int, N * (1 + magnetization) / 2)
    Ndown = N - Nup
    Nup - Ndown == N * magnetization || error(
        "magnetization is not commensurate with L=$L",
    )

    hopping, flux = uniform_flux_staggered_pi_hoppings(L, L, Q)
    Haux = Matrix(hamiltonian_aux_triangular_torus(L, L; hopping))
    spectrum = eigen(Hermitian(Haux[1:2:end, 1:2:end]))
    gap_up = spectrum.values[Nup + 1] - spectrum.values[Nup]
    gap_down = spectrum.values[Ndown + 1] - spectrum.values[Ndown]
    closed_shell = min(gap_up, gap_down) > closed_shell_tolerance
    projected_state = closed_shell ? gutzwiller_project(
        spectrum.vectors[:, 1:Nup],
        spectrum.vectors[:, 1:Ndown],
    ) : nothing
    return (; projected_state, Nup, Ndown, flux, gap_up, gap_down, closed_shell)
end

records = NamedTuple[]
for L in linear_sizes
    N = L^2
    peps = identity_D1_peps(L, L)
    bonds = exchange_bonds(L, L)
    flux_quanta = sort!(unique(round.(Int, N .* requested_flux_densities)))

    for magnetization in magnetizations
        println("Scanning bare Gutzwiller state: L=$L, m=$magnetization ...")
        for Q in flux_quanta
            state_data = flux_state(L, Q, magnetization)
            if !state_data.closed_shell
                println("  Q=$Q skipped: open shell")
                continue
            end

            seed = base_seed + 10_000L + 100Q + round(Int, 10magnetization) +
                   measurement_seed_offset
            local statistics
            runtime_seconds = @elapsed statistics = bare_projected_energy(
                MersenneTwister(seed),
                state_data.projected_state,
                peps,
                bonds,
            )
            energy_per_site = statistics.mean / (N * J1)
            error_per_site = statistics.error / (N * J1)
            flux_density = state_data.flux / (2pi)
            push!(records, (;
                L,
                N,
                magnetization=Float64(magnetization),
                Q,
                flux_density,
                energy_per_site,
                error_per_site,
                variance=statistics.variance,
                imaginary_mean=statistics.imaginary_mean,
                acceptance=statistics.acceptance,
                gap_up=state_data.gap_up,
                gap_down=state_data.gap_down,
                number_of_samples,
                burnin_sweeps,
                moves_per_sample,
                runtime_seconds,
            ))
            println(
                "  Q=$(lpad(Q, 3)), phi/(2pi)=$(round(flux_density; digits=5)), " *
                "E/(N J1)=$(round(energy_per_site; digits=7)) +/- " *
                "$(round(error_per_site; digits=7)), acceptance=" *
                "$(round(statistics.acceptance; digits=3)), time=" *
                "$(round(runtime_seconds; digits=2)) s",
            )
        end
    end
end

isempty(records) && error("no closed-shell flux points were found")

output_directory = joinpath(
    @__DIR__,
    "projected_gutzwiller_ppeps_D1_flux_energy_plots",
)
mkpath(output_directory)
data_path = joinpath(output_directory, "flux_energy_data.jld2")
parameters = (;
    method=:bare_fixed_sz_vmc,
    linear_sizes,
    requested_flux_densities,
    number_of_samples,
    burnin_sweeps,
    block_length,
    moves_per_sample,
    measurement_seed_offset,
    J1,
    J2,
    field,
    magnetizations,
    closed_shell_tolerance,
    base_seed,
)
JLD2.jldsave(data_path; records, parameters)

figure = Figure(size=(1200, 520))
axes = [
    Axis(
        figure[1, panel];
        xlabel=L"\phi/2\pi",
        ylabel=panel == 1 ? L"E/(N J_1)" : "",
        title=panel == 1 ? L"m=1/3" : L"m=2/3",
    ) for panel in 1:2
]

colors = Makie.wong_colors()
markers = (:circle, :utriangle, :rect, :diamond, :cross)
for (panel, magnetization) in enumerate(Float64.(magnetizations))
    axis = axes[panel]
    for (size_index, L) in enumerate(linear_sizes)
        selected = filter(
            record -> record.L == L && record.magnetization == magnetization,
            records,
        )
        sort!(selected; by=record -> record.flux_density)
        isempty(selected) && continue
        x = getproperty.(selected, :flux_density)
        y = getproperty.(selected, :energy_per_site)
        yerror = getproperty.(selected, :error_per_site)
        color = colors[mod1(size_index, length(colors))]
        marker = markers[mod1(size_index, length(markers))]
        errorbars!(axis, x, y, yerror; color, whiskerwidth=7, linewidth=1.2)
        scatter!(axis, x, y; color, marker, markersize=10, label=L"N_x=%$L")
    end
    vlines!(axis, [magnetization / 2]; color=:black, linestyle=:dash, linewidth=1.5)
    xlims!(axis, 0, 0.5)
end
axislegend(axes[1]; position=:rb, framevisible=false)
Label(
    figure[0, :],
    "Bare projected staggered-[0, pi] flux state, " *
    "J2/J1=$(round(J2 / J1; digits=4)), H/J1=$(round(field / J1; digits=4))";
    fontsize=22,
)

pdf_path = joinpath(output_directory, "flux_energy_m_1over3_2over3.pdf")
png_path = joinpath(output_directory, "flux_energy_m_1over3_2over3.png")
save(pdf_path, figure)
save(png_path, figure; px_per_unit=2)

println("Saved bare-VMC data to: $data_path")
println("Saved bare flux-energy plot to: $pdf_path")
println("Saved bare flux-energy preview to: $png_path")
