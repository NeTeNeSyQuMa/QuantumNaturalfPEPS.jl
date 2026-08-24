# Flux-energy scan for a Gutzwiller-projected finite-flux state dressed by a
# D=1 PPEPS. The PPEPS parameters are optimized with QuantumNaturalGradient.
#
# Install the optional plotting environment once from the repository root:
#   julia --project=examples -e 'using Pkg; Pkg.instantiate()'
#
# Run with the package as the primary environment:
#   julia --project=. examples/projected_gutzwiller_ppeps_D1_flux_energy.jl
#
# A selected-flux 18x18 and 30x30 run, for example:
#   PPEPS_LINEAR_SIZES=18,30 \
#   PPEPS_FLUX_DENSITIES=0.12,0.1666666667,0.24,0.3333333333,0.4 \
#   PPEPS_QNG_SAMPLES=400 PPEPS_MEASUREMENT_SAMPLES=2000 \
#   PPEPS_MAXITER=20 PPEPS_EIGENCUT=1e-4 julia --project=. \
#       examples/projected_gutzwiller_ppeps_D1_flux_energy.jl

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
using QuantumNaturalGradient
using QuantumNaturalfPEPS
using Statistics

const QNG = QuantumNaturalGradient

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

linear_sizes = parse_int_list(get(ENV, "PPEPS_LINEAR_SIZES", "6"))
qng_samples = parse(Int, get(ENV, "PPEPS_QNG_SAMPLES", "400"))
measurement_samples = parse(Int, get(ENV, "PPEPS_MEASUREMENT_SAMPLES", "2000"))
burnin_sweeps = parse(Int, get(ENV, "PPEPS_BURNIN_SWEEPS", "20"))
block_length = parse(Int, get(ENV, "PPEPS_BLOCK_LENGTH", "20"))
configured_moves_per_sample = parse(Int, get(ENV, "PPEPS_MOVES_PER_SAMPLE", "0"))
maximum_iterations = parse(Int, get(ENV, "PPEPS_MAXITER", "20"))
learning_rate = parse(Float64, get(ENV, "PPEPS_LR", "0.02"))
eigenvalue_cutoff = parse(Float64, get(ENV, "PPEPS_EIGENCUT", "1e-4"))
base_seed = parse(Int, get(ENV, "PPEPS_SEED", "8128"))
output_suffix = get(ENV, "PPEPS_OUTPUT_SUFFIX", "")

flux_density_setting = strip(get(ENV, "PPEPS_FLUX_DENSITIES", ""))
requested_flux_densities = isempty(flux_density_setting) ? nothing :
    parse.(Float64, strip.(split(flux_density_setting, ',')))

J1 = parse(Float64, get(ENV, "PPEPS_J1", "1.0"))
J2 = parse(Float64, get(ENV, "PPEPS_J2", string(J1 / 8)))
field = parse(Float64, get(ENV, "PPEPS_FIELD", "0.0"))
magnetizations = (1 // 3, 2 // 3)
closed_shell_tolerance = parse(Float64, get(ENV, "PPEPS_MIN_GAP", "1e-8"))

all(iseven, linear_sizes) || error("all linear sizes must be even for the staggered-pi gauge")
qng_samples > 1 || error("PPEPS_QNG_SAMPLES must exceed one")
measurement_samples >= 2block_length || error(
    "PPEPS_MEASUREMENT_SAMPLES must be at least twice PPEPS_BLOCK_LENGTH",
)
maximum_iterations > 0 || error("PPEPS_MAXITER must be positive")
configured_moves_per_sample >= 0 || error("PPEPS_MOVES_PER_SAMPLE must be nonnegative")
if !isnothing(requested_flux_densities)
    all(density -> 0 <= density <= 0.5, requested_flux_densities) || error(
        "PPEPS_FLUX_DENSITIES entries must lie in [0, 0.5]",
    )
end

function identity_D1_peps(Lx::Int, Ly::Int)
    hilbert = ITensors.siteinds("S=1/2", Lx, Ly)
    peps = PEPS(ComplexF64, hilbert; bond_dim=1, show_warning=false)
    write!(peps, fill(ComplexF64(inv(sqrt(2))), length(peps)))
    return peps
end

function normalize_D1_parameters!(optimization_state, natural_gradient)
    local_tensors = reshape(optimization_state.θ, 2, :)
    for tensor in eachcol(local_tensors)
        tensor_norm = norm(tensor)
        isfinite(tensor_norm) && tensor_norm > eps(Float64) || error(
            "QNG produced a singular D=1 local PPEPS tensor",
        )
        tensor ./= tensor_norm
    end
    return natural_gradient
end

function blocked_statistics(values, block_length)
    number_of_blocks = fld(length(values), block_length)
    number_of_blocks >= 2 || error("at least two complete error-analysis blocks are required")
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

function flux_state(L, Q, magnetization; gap_tolerance=closed_shell_tolerance)
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
    closed_shell = min(gap_up, gap_down) > gap_tolerance
    projected_state = closed_shell ? gutzwiller_project(
        spectrum.vectors[:, 1:Nup],
        spectrum.vectors[:, 1:Ndown],
    ) : nothing
    return (; projected_state, Nup, Ndown, flux, gap_up, gap_down, closed_shell)
end

physical_hamiltonians = Dict(
    L => hamiltonian_J1J2_H(
        L,
        L;
        J1,
        J2,
        H=field,
        boundary=:periodic,
    ) for L in linear_sizes
)

records = NamedTuple[]
for L in linear_sizes
    N = L^2
    moves_per_sample = iszero(configured_moves_per_sample) ?
        max(1, N ÷ 4) : configured_moves_per_sample
    flux_quanta = if isnothing(requested_flux_densities)
        collect(0:N÷2)
    else
        sort!(unique(round.(Int, N .* requested_flux_densities)))
    end

    for magnetization in magnetizations
        println("Scanning L=$L, m=$magnetization with QNG-optimized D=1 PPEPS ...")
        for Q in flux_quanta
            state_data = flux_state(L, Q, magnetization)
            if !state_data.closed_shell
                println("  Q=$Q skipped: open shell")
                continue
            end

            peps = identity_D1_peps(L, L)
            seed = base_seed + 10_000L + 100Q + round(Int, 10magnetization)
            Oks_and_Eks = generate_Oks_and_Eks(
                peps,
                physical_hamiltonians[L];
                trial_state=state_data.projected_state,
                fixed_sz_metropolis=true,
                burnin_sweeps,
                moves_per_sample,
                seed,
            )

            # QNG's component-wise clipping currently assumes real parameters;
            # these finite-flux PPEPS tensors are complex.
            integrator = QNG.Euler(lr=learning_rate)
            # With qng_samples < 2N, QNG automatically solves the smaller
            # sample-space metric rather than a dense (2N)x(2N) matrix.
            solver = QNG.EigenSolver(eigenvalue_cutoff)
            initial_parameters = vec(peps)

            local trained_parameters, qng_energy, qng_misc, measurement
            runtime_seconds = @elapsed begin
                qng_energy, trained_parameters, qng_misc = QNG.evolve(
                    Oks_and_Eks,
                    initial_parameters;
                    integrator,
                    solver,
                    transform! = normalize_D1_parameters!,
                    sample_nr=qng_samples,
                    maxiter=maximum_iterations,
                    verbosity=1,
                    copy=true,
                )
                measurement = Oks_and_Eks(trained_parameters, measurement_samples)
            end

            statistics = blocked_statistics(measurement[:Eks], block_length)
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
                qng_reported_energy=qng_energy / (N * J1),
                parameter_change_norm=norm(trained_parameters - initial_parameters),
                variance=statistics.variance,
                imaginary_mean=statistics.imaginary_mean,
                acceptance=measurement[:acceptance],
                gap_up=state_data.gap_up,
                gap_down=state_data.gap_down,
                qng_samples,
                measurement_samples,
                maximum_iterations,
                moves_per_sample,
                runtime_seconds,
            ))
            println(
                "  Q=$(lpad(Q, 3)), phi/(2pi)=$(round(flux_density; digits=5)), " *
                "E/(N J1)=$(round(energy_per_site; digits=7)) +/- " *
                "$(round(error_per_site; digits=7)), acceptance=" *
                "$(round(measurement[:acceptance]; digits=3)), time=" *
                "$(round(runtime_seconds; digits=2)) s",
            )
        end
    end
end

isempty(records) && error("no closed-shell flux points were found")

output_directory = joinpath(
    @__DIR__,
    "projected_gutzwiller_ppeps_D1_flux_energy" * output_suffix,
)
mkpath(output_directory)
data_path = joinpath(output_directory, "flux_energy_data.jld2")
parameters = (;
    linear_sizes,
    qng_samples,
    measurement_samples,
    burnin_sweeps,
    block_length,
    configured_moves_per_sample,
    maximum_iterations,
    learning_rate,
    eigenvalue_cutoff,
    requested_flux_densities,
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
        scatter!(
            axis,
            x,
            y;
            color,
            marker,
            markersize=10,
            label=L"N_x=%$L",
        )
    end

    # The C=+/-1 Landau-level state occurs at phi/(2pi)=m/2.
    vlines!(axis, [magnetization / 2]; color=:black, linestyle=:dash, linewidth=1.5)
    xlims!(axis, 0, 0.5)
end
axislegend(axes[1]; position=:rb, framevisible=false)
Label(
    figure[0, :],
    "Projected staggered-[0, pi] flux state + QNG-optimized D=1 PPEPS, " *
    "J2/J1=$(round(J2 / J1; digits=4)), H/J1=$(round(field / J1; digits=4))";
    fontsize=22,
)

pdf_path = joinpath(output_directory, "flux_energy_m_1over3_2over3.pdf")
save(pdf_path, figure)

println("Saved QNG/PPEPS data to: $data_path")
println("Saved flux-energy plot to: $pdf_path")
