# QNG/D=1-PPEPS analogue of the transverse-correlation panels in Fig. 2 of
# "Chirality and quasi-long-range order in finite-flux Gutzwiller states for
# magnetized frustrated magnets" (Wang et al., arXiv:2607.14766).
#
# The publication plots bare Gutzwiller states. Here every state is dressed by
# a D=1 PPEPS and optimized for the triangular J1-J2 Heisenberg Hamiltonian by
# QuantumNaturalGradient.evolve before Cperp is measured. Thus this script is
# a controlled QNG/PPEPS comparison, not digitized publication data.
#
# Install the optional plotting environment once from the repository root:
#   julia --project=examples -e 'using Pkg; Pkg.instantiate()'
#
# Run with the package as the primary environment:
#   julia --project=. examples/projected_gutzwiller_ppeps_D1_figure2_qng.jl

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
    fontsize = 25
    clbr = (size=25, labelsize=40)
    Theme(
        fonts=fonts,
        fontsize=fontsize,
        Colorbar=clbr,
        Heatmap=(; colormap=:berlin),
        Label=(; lineheight=0.9),
        Scatter=(; markersize=9),
    )
end
set_theme!(theme)

qng_samples = parse(Int, get(ENV, "PPEPS_QNG_SAMPLES", "100"))
measurement_samples = parse(Int, get(ENV, "PPEPS_MEASUREMENT_SAMPLES", "500"))
burnin_sweeps = parse(Int, get(ENV, "PPEPS_BURNIN_SWEEPS", "5"))
moves_per_sample = parse(Int, get(ENV, "PPEPS_MOVES_PER_SAMPLE", "20"))
block_length = parse(Int, get(ENV, "PPEPS_BLOCK_LENGTH", "25"))
maximum_iterations = parse(Int, get(ENV, "PPEPS_MAXITER", "3"))
learning_rate = parse(Float64, get(ENV, "PPEPS_LR", "0.02"))
eigenvalue_cutoff = parse(Float64, get(ENV, "PPEPS_EIGENCUT", "1e-4"))
base_seed = parse(Int, get(ENV, "PPEPS_SEED", "220726"))
q_grid_size = parse(Int, get(ENV, "PPEPS_Q_GRID_SIZE", "161"))
monopole_samples = parse(Int, get(ENV, "PPEPS_MONOPOLE_SAMPLES", "5000"))
monopole_burnin_sweeps = parse(Int, get(ENV, "PPEPS_MONOPOLE_BURNIN_SWEEPS", "20"))
monopole_moves_per_sample = parse(Int, get(ENV, "PPEPS_MONOPOLE_MOVES_PER_SAMPLE", "324"))
monopole_block_length = parse(Int, get(ENV, "PPEPS_MONOPOLE_BLOCK_LENGTH", "100"))
monopole_only = lowercase(strip(get(ENV, "PPEPS_MONOPOLE_ONLY", "false"))) in
    ("1", "true", "yes")

J1 = parse(Float64, get(ENV, "PPEPS_J1", "1.0"))
J2 = parse(Float64, get(ENV, "PPEPS_J2", string(J1 / 8)))
field = parse(Float64, get(ENV, "PPEPS_FIELD", "0.0"))
closed_shell_tolerance = parse(Float64, get(ENV, "PPEPS_MIN_GAP", "1e-8"))

available_cases = (
    :fermi_pocket,
    :C1_m1over3,
    :C2_m1over3,
    :C1_m2over3,
)
case_setting = strip(get(ENV, "PPEPS_FIGURE2_CASES", ""))
selected_cases = isempty(case_setting) ? collect(available_cases) :
    Symbol.(strip.(split(case_setting, ',')))
all(case -> case in available_cases, selected_cases) || error(
    "PPEPS_FIGURE2_CASES must be a comma-separated subset of $(available_cases)",
)

qng_samples > 1 || error("PPEPS_QNG_SAMPLES must exceed one")
measurement_samples >= 2block_length || error(
    "PPEPS_MEASUREMENT_SAMPLES must be at least twice PPEPS_BLOCK_LENGTH",
)
maximum_iterations > 0 || error("PPEPS_MAXITER must be positive")
moves_per_sample > 0 || error("PPEPS_MOVES_PER_SAMPLE must be positive")
monopole_samples >= 2monopole_block_length || error(
    "PPEPS_MONOPOLE_SAMPLES must be at least twice PPEPS_MONOPOLE_BLOCK_LENGTH",
)
monopole_burnin_sweeps > 0 || error("PPEPS_MONOPOLE_BURNIN_SWEEPS must be positive")
monopole_moves_per_sample > 0 || error("PPEPS_MONOPOLE_MOVES_PER_SAMPLE must be positive")
isodd(q_grid_size) && q_grid_size >= 51 || error(
    "PPEPS_Q_GRID_SIZE must be an odd integer of at least 51",
)

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

function blocked_energy_statistics(values, block_length)
    number_of_blocks = fld(length(values), block_length)
    number_of_blocks >= 2 || error("at least two complete energy blocks are required")
    block_means = [
        mean(real, view(values, (block - 1) * block_length + 1:block * block_length))
        for block in 1:number_of_blocks
    ]
    return (;
        mean=mean(block_means),
        error=std(block_means) / sqrt(number_of_blocks),
        imaginary_mean=mean(imag, values),
    )
end

function figure2_state_parameters(case, L)
    N = L^2
    if case == :fermi_pocket
        magnetization = if L == 12
            13 // 36
        elseif L == 18
            26 // 81
        else
            error("the paper's Fermi-pocket panels use only L=12 and L=18")
        end
        return (; magnetization, Q=0, chern=0)
    elseif case == :C1_m1over3
        N % 6 == 0 || error("L=$L is incompatible with phi/(2pi)=1/6")
        return (; magnetization=1 // 3, Q=N ÷ 6, chern=1)
    elseif case == :C2_m1over3
        N % 3 == 0 || error("L=$L is incompatible with phi/(2pi)=1/3")
        return (; magnetization=1 // 3, Q=N ÷ 3, chern=2)
    elseif case == :C1_m2over3
        N % 3 == 0 || error("L=$L is incompatible with phi/(2pi)=1/3")
        return (; magnetization=2 // 3, Q=N ÷ 3, chern=1)
    end
    error("unknown Figure-2 case $case")
end

function projected_flux_state(L, case)
    parameters = figure2_state_parameters(case, L)
    N = L^2
    magnetization = parameters.magnetization
    Nup = round(Int, N * (1 + magnetization) / 2)
    Ndown = N - Nup
    Nup - Ndown == N * magnetization || error(
        "magnetization $magnetization is not commensurate with L=$L",
    )

    hopping, flux = uniform_flux_staggered_pi_hoppings(L, L, parameters.Q)
    Haux = Matrix(hamiltonian_aux_triangular_torus(L, L; hopping))
    spectrum = eigen(Hermitian(Haux[1:2:end, 1:2:end]))
    gap_up = spectrum.values[Nup + 1] - spectrum.values[Nup]
    gap_down = spectrum.values[Ndown + 1] - spectrum.values[Ndown]
    min(gap_up, gap_down) > closed_shell_tolerance || error(
        "case=$case, L=$L is not closed-shell: gaps=($gap_up, $gap_down)",
    )
    projected_state = gutzwiller_project(
        spectrum.vectors[:, 1:Nup],
        spectrum.vectors[:, 1:Ndown],
    )
    return merge(parameters, (; projected_state, Nup, Ndown, flux, gap_up, gap_down))
end

function adjacent_monopole_state(L, base_state)
    Q = base_state.Q + 1
    Nup = base_state.Nup + 1
    Ndown = base_state.Ndown - 1
    hopping, flux = uniform_flux_staggered_pi_hoppings(L, L, Q)
    Haux = Matrix(hamiltonian_aux_triangular_torus(L, L; hopping))
    spectrum = eigen(Hermitian(Haux[1:2:end, 1:2:end]))
    gap_up = spectrum.values[Nup + 1] - spectrum.values[Nup]
    gap_down = spectrum.values[Ndown + 1] - spectrum.values[Ndown]
    min(gap_up, gap_down) > closed_shell_tolerance || error(
        "the Q+1 monopole sector is not closed-shell: gaps=($gap_up, $gap_down)",
    )
    projected_state = gutzwiller_project(
        spectrum.vectors[:, 1:Nup],
        spectrum.vectors[:, 1:Ndown],
    )
    return (; projected_state, Q, Nup, Ndown, flux, gap_up, gap_down)
end

function inside_extended_triangular_momentum_zone(qx, qy)
    tolerance = 100eps(Float64)
    return abs(qx) <= 2pi + tolerance &&
           abs(qx + sqrt(3) * qy) <= 4pi + tolerance &&
           abs(qx - sqrt(3) * qy) <= 4pi + tolerance
end

function triangular_structure_factor(Cperp_displacement, qx_axis, qy_axis)
    Lx, Ly = size(Cperp_displacement)
    N = Lx * Ly
    values = fill(NaN, length(qx_axis), length(qy_axis))
    for (qx_index, qx) in enumerate(qx_axis), (qy_index, qy) in enumerate(qy_axis)
        inside_extended_triangular_momentum_zone(qx, qy) || continue
        # Reciprocal phases conjugate to the two integer lattice coordinates.
        kx = qx
        ky = -qx / 2 + sqrt(3) * qy / 2
        Cq = 0.0 + 0.0im
        for dy in 0:Ly-1, dx in 0:Lx-1
            # For momenta between the finite torus grid points, retain the
            # original finite-cluster definition (1/N) sum_ij rather than
            # Fourier transforming a discontinuously wrapped displacement.
            # The two factors below sum exp[i k (r_i-r_j)] over all origins
            # whose target is displaced by (dx,dy) modulo the torus.
            x_phase_sum = (Lx - dx) * exp(-im * kx * dx) +
                          dx * exp(im * kx * (Lx - dx))
            y_phase_sum = (Ly - dy) * exp(-im * ky * dy) +
                          dy * exp(im * ky * (Ly - dy))
            Cq += Cperp_displacement[dx + 1, dy + 1] *
                  x_phase_sum * y_phase_sum / N
        end
        values[qx_index, qy_index] = real(Cq)
    end
    return values
end

case_index(case) = findfirst(==(case), available_cases)
physical_hamiltonians = Dict(
    L => QuantumNaturalfPEPS.hamiltonian_J1J2_H(
        L,
        L;
        J1,
        J2,
        H=field,
        boundary=:periodic,
    ) for L in (12, 18)
)
qx_axis = collect(range(-2pi, 2pi; length=q_grid_size))
# Match the paper's three-BZ hexagonal plotting domain. Choose an odd number
# of qy points with nearly the same Cartesian spacing as the qx grid.
qy_grid_size = 2round(Int, (q_grid_size - 1) / sqrt(3)) + 1
qy_axis = collect(range(-4pi / sqrt(3), 4pi / sqrt(3); length=qy_grid_size))
output_directory = joinpath(@__DIR__, "projected_gutzwiller_ppeps_D1_expectation_value")
mkpath(output_directory)
data_path = joinpath(output_directory, "transverse_correlations_D1.jld2")
replot_only = lowercase(strip(get(ENV, "PPEPS_REPLOT_ONLY", "false"))) in
    ("1", "true", "yes")
replot_only && monopole_only && error(
    "PPEPS_REPLOT_ONLY and PPEPS_MONOPOLE_ONLY cannot both be enabled",
)

if replot_only
    isfile(data_path) || error("cannot replot: data file does not exist at $data_path")
    stored_data = JLD2.load(data_path)
    records = stored_data["records"]
    parameters = stored_data["parameters"]
    monopole_record = stored_data["monopole_record"]
    records = [
        merge(record, (; Cperp_q=triangular_structure_factor(
            record.Cperp_displacement,
            qx_axis,
            qy_axis,
        ))) for record in records
    ]
    println("Loaded saved QNG/PPEPS measurements from: $data_path")
else
existing_data = (monopole_only || length(selected_cases) < length(available_cases)) &&
                isfile(data_path) ?
    JLD2.load(data_path) : nothing
monopole_only && isnothing(existing_data) && error(
    "PPEPS_MONOPOLE_ONLY requires existing Figure-2 data at $data_path",
)
records = NamedTuple[]
if !monopole_only
for case in selected_cases, L in (12, 18)
    state = projected_flux_state(L, case)
    local peps = identity_D1_peps(L, L)
    seed = base_seed + 10_000L + 1_000case_index(case)
    Oks_and_Eks = generate_Oks_and_Eks(
        peps,
        physical_hamiltonians[L];
        trial_state=state.projected_state,
        fixed_sz_metropolis=true,
        burnin_sweeps,
        moves_per_sample,
        seed,
    )

    println(
        "Optimizing case=$case, L=$L, m=$(state.magnetization), " *
        "phi/(2pi)=$(state.flux / (2pi)), |C|=$(state.chern) ...",
    )
    initial_parameters = vec(peps)
    integrator = QNG.Euler(lr=learning_rate)
    solver = QNG.EigenSolver(eigenvalue_cutoff)
    local qng_energy, trained_parameters, qng_misc, measurement
    optimization_seconds = @elapsed begin
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
    end
    measurement_seconds = @elapsed begin
        measurement = Oks_and_Eks(
            trained_parameters,
            measurement_samples;
            measure_transverse=true,
            transverse_reference=1,
            transverse_block_length=block_length,
        )
    end

    Cperp_reference = reshape(copy(measurement[:Cperp_reference]), L, L)
    Cperp_reference_error = reshape(copy(measurement[:Cperp_reference_error]), L, L)
    Cperp_displacement = copy(measurement[:Cperp_displacement])
    Cperp_displacement_error = copy(measurement[:Cperp_displacement_error])
    Cperp_q = triangular_structure_factor(Cperp_displacement, qx_axis, qy_axis)
    energy = blocked_energy_statistics(measurement[:Eks], block_length)
    N = L^2

    push!(records, (;
        case,
        L,
        N,
        magnetization=Float64(state.magnetization),
        Q=state.Q,
        flux_density=state.flux / (2pi),
        chern=state.chern,
        Nup=state.Nup,
        Ndown=state.Ndown,
        gap_up=state.gap_up,
        gap_down=state.gap_down,
        Cperp_reference,
        Cperp_reference_error,
        Cperp_displacement,
        Cperp_displacement_error,
        Cperp_q,
        energy_per_site=energy.mean / (N * J1),
        energy_error_per_site=energy.error / (N * J1),
        energy_imaginary_mean=energy.imaginary_mean,
        qng_reported_energy=qng_energy / (N * J1),
        trained_parameters,
        parameter_change_norm=norm(trained_parameters - initial_parameters),
        acceptance=measurement[:acceptance],
        optimization_seconds,
        measurement_seconds,
    ))
    println(
        "  E/(NJ1)=$(round(energy.mean / (N * J1); digits=7)) +/- " *
        "$(round(energy.error / (N * J1); digits=7)), acceptance=" *
        "$(round(measurement[:acceptance]; digits=3)), optimize=" *
        "$(round(optimization_seconds; digits=2)) s, correlations=" *
        "$(round(measurement_seconds; digits=2)) s",
    )
end
end

# Figure 2(a): inserting one flux quantum into the bare |C|=1, m=2/3 state
# pumps one unit of Sz. Measure <n+1|P S_i^+ P|n>/<n|P|n> with synchronized
# Markov chains in the two flux sectors. Identity D=1 weights reproduce the
# bare Gutzwiller-projected state used in the paper.
monopole_record = isnothing(existing_data) ? nothing : existing_data["monopole_record"]
if monopole_only || :C1_m2over3 in selected_cases
    L = 18
    base_state = projected_flux_state(L, :C1_m2over3)
    target_state = adjacent_monopole_state(L, base_state)
    monopole_peps = identity_D1_peps(L, L)
    identity_parameters = vec(monopole_peps)
    monopole_callback = generate_Oks_and_Eks(
        monopole_peps,
        physical_hamiltonians[L];
        trial_state=base_state.projected_state,
        fixed_sz_metropolis=true,
        burnin_sweeps=monopole_burnin_sweeps,
        moves_per_sample=monopole_moves_per_sample,
        seed=base_seed + 990_018,
    )
    local monopole_measurement
    println(
        "Measuring the bare m=2/3, Q -> Q+1 monopole matrix element on 18x18 " *
        "with $monopole_samples samples ...",
    )
    monopole_seconds = @elapsed begin
        monopole_measurement = monopole_callback(
            identity_parameters,
            monopole_samples;
            monopole_state=target_state.projected_state,
            transverse_block_length=monopole_block_length,
        )
    end
    matrix_element = reshape(
        copy(monopole_measurement[:monopole_matrix_element]),
        L,
        L,
    )
    matrix_element_error = reshape(
        copy(monopole_measurement[:monopole_matrix_element_error]),
        L,
        L,
    )
    monopole_record = (;
        L,
        magnetization=2 // 3,
        chern=1,
        dressing=:identity_D1,
        base_Q=base_state.Q,
        target_Q=target_state.Q,
        base_Nup=base_state.Nup,
        target_Nup=target_state.Nup,
        base_flux_density=base_state.flux / (2pi),
        target_flux_density=target_state.flux / (2pi),
        target_gap_up=target_state.gap_up,
        target_gap_down=target_state.gap_down,
        matrix_element,
        matrix_element_error,
        samples=monopole_samples,
        burnin_sweeps=monopole_burnin_sweeps,
        moves_per_sample=monopole_moves_per_sample,
        block_length=monopole_block_length,
        acceptance=monopole_measurement[:acceptance],
        measurement_seconds=monopole_seconds,
    )
    println(
        "  monopole acceptance=$(round(monopole_measurement[:acceptance]; digits=3)), " *
        "time=$(round(monopole_seconds; digits=2)) s",
    )
end

if !isnothing(existing_data)
    if monopole_only
        records = existing_data["records"]
    else
        retained_records = filter(
            record -> !(record.case in selected_cases),
            existing_data["records"],
        )
        records = vcat(retained_records, records)
    end
end
sort!(records; by=record -> (case_index(record.case), record.L))
if !monopole_only
    records = [
        merge(record, (; Cperp_q=triangular_structure_factor(
            record.Cperp_displacement,
            qx_axis,
            qy_axis,
        ))) for record in records
    ]
end
saved_cases = [case for case in available_cases if any(record -> record.case == case, records)]

monopole_parameters = (;
    monopole_samples,
    monopole_burnin_sweeps,
    monopole_moves_per_sample,
    monopole_block_length,
    monopole_magnetization=2 // 3,
    monopole_dressing=:identity_D1,
)
parameters = if monopole_only
    merge(existing_data["parameters"], monopole_parameters)
else
    (;
        method=:projected_gutzwiller_qng_D1,
        selected_cases=saved_cases,
        real_space_size=12,
        momentum_space_size=18,
        qng_samples,
        measurement_samples,
        burnin_sweeps,
        moves_per_sample,
        block_length,
        maximum_iterations,
        learning_rate,
        eigenvalue_cutoff,
        base_seed,
        q_grid_size,
        qx_axis,
        qy_axis,
        J1,
        J2,
        field,
        closed_shell_tolerance,
        monopole_parameters...,
    )
end
JLD2.jldsave(data_path; records, parameters, monopole_record)
end

saved_cases = [case for case in available_cases if any(record -> record.case == case, records)]
length(saved_cases) == length(available_cases) || begin
    println("Saved partial data to: $data_path")
    println("Skipping the composite plot because the saved data contain only $(saved_cases).")
    exit()
end

monopole_figure = Figure(size=(1150, 760))
monopole_axis = Axis(
    monopole_figure[1, 1];
    xlabel=L"x",
    ylabel=L"y",
    title=L"\langle S_i^+\rangle_{\mathrm{mono}},\quad |C|=1,\ m=2/3\quad (Q\rightarrow Q+1)",
    aspect=DataAspect(),
)
monopole_values = vec(monopole_record.matrix_element)
reference_phase = angle(monopole_values[1])
relative_phases = angle.(monopole_values .* exp(-im * reference_phase))
magnitudes = abs.(monopole_values)
maximum_magnitude = maximum(magnitudes)
x_positions = Float64[]
y_positions = Float64[]
for y in 0:17, x in 0:17
    push!(x_positions, x + y / 2)
    push!(y_positions, y)
end
monopole_plot = scatter!(
    monopole_axis,
    x_positions,
    y_positions;
    color=relative_phases,
    colorrange=(-pi, pi),
    # The cyclic vikO map has the same blue-white-orange-red phase ordering
    # used in the publication, with identical colors at -pi and +pi.
    colormap=:vikO,
    markersize=6 .+ 18 .* sqrt.(magnitudes ./ maximum_magnitude),
    strokecolor=(:black, 0.5),
    strokewidth=0.4,
)
Colorbar(
    monopole_figure[1, 2],
    monopole_plot;
    label="phase relative to (0,0)",
    ticks=([-pi, 0, pi], [L"-\pi", L"0", L"\pi"]),
)
xlims!(monopole_axis, -0.7, 26.2)
ylims!(monopole_axis, -0.7, 17.7)
monopole_pdf_path = joinpath(output_directory, "monopole_matrix_elem_D1.pdf")
save(monopole_pdf_path, monopole_figure)

if monopole_only
    println("Updated monopole data in: $data_path")
    println("Saved monopole plot to: $monopole_pdf_path")
    exit()
end

record_for(case, L) = only(filter(record -> record.case == case && record.L == L, records))
case_titles = Dict(
    :fermi_pocket => L"\phi=0\quad (\mathrm{Fermi\ pocket})",
    :C1_m1over3 => L"|C|=1,\quad m=1/3",
    :C2_m1over3 => L"|C|=2,\quad m=1/3",
    :C1_m2over3 => L"|C|=1,\quad m=2/3",
)
momentum_color_ranges = Dict(
    :fermi_pocket => (0.15, 1.30),
    :C1_m1over3 => (0.15, 4.20),
    :C2_m1over3 => (0.15, 3.20),
    :C1_m2over3 => (0.15, 4.60),
)
momentum_color_ticks = Dict(
    :fermi_pocket => collect(0.25:0.25:1.25),
    :C1_m1over3 => collect(1.0:1.0:4.0),
    :C2_m1over3 => collect(1.0:1.0:3.0),
    :C1_m2over3 => collect(1.0:1.0:4.0),
)

figure = Figure(size=(1650, 1900))
real_heatmaps = Any[]
momentum_heatmaps = Any[]
for (row, case) in enumerate(available_cases)
    real_record = record_for(case, 12)
    momentum_record = record_for(case, 18)

    real_axis = Axis(
        figure[row, 1];
        xlabel=row == length(available_cases) ? L"x" : "",
        ylabel=L"y",
        title=case_titles[case],
        aspect=DataAspect(),
    )
    real_x_positions = Float64[]
    real_y_positions = Float64[]
    colors = Float64[]
    for y in 0:11, x in 0:11
        push!(real_x_positions, x + y / 2)
        push!(real_y_positions, y)
        push!(colors, real(real_record.Cperp_reference[x + 1, y + 1]))
    end
    real_plot = scatter!(
        real_axis,
        real_x_positions,
        real_y_positions;
        color=colors,
        colorrange=(-0.15, 0.15),
        colormap=[:blue, :white, :red],
        markersize=16,
        strokecolor=(:black, 0.35),
        strokewidth=0.35,
    )
    push!(real_heatmaps, real_plot)
    xlims!(real_axis, -0.6, 17.1)
    ylims!(real_axis, -0.6, 11.6)
    Colorbar(figure[row, 2], real_plot; label=L"C^\perp_{0j}")

    momentum_axis = Axis(
        figure[row, 3];
        xlabel=row == length(available_cases) ? L"q_x" : "",
        ylabel=L"q_y",
        aspect=DataAspect(),
        xticks=([-2pi, -pi, 0, pi, 2pi], [L"-2\pi", L"-\pi", L"0", L"\pi", L"2\pi"]),
        yticks=([-2pi, -pi, 0, pi, 2pi], [L"-2\pi", L"-\pi", L"0", L"\pi", L"2\pi"]),
    )
    momentum_plot = heatmap!(
        momentum_axis,
        qx_axis,
        qy_axis,
        momentum_record.Cperp_q;
        colorrange=momentum_color_ranges[case],
        colormap=:viridis,
    )
    push!(momentum_heatmaps, momentum_plot)
    bz_x = [4pi / 3, 2pi / 3, -2pi / 3, -4pi / 3, -2pi / 3, 2pi / 3, 4pi / 3]
    bz_y = [0, 2pi / sqrt(3), 2pi / sqrt(3), 0, -2pi / sqrt(3), -2pi / sqrt(3), 0]
    lines!(momentum_axis, bz_x, bz_y; color=:red, linewidth=2)
    xlims!(momentum_axis, -2pi, 2pi)
    ylims!(momentum_axis, -4pi / sqrt(3), 4pi / sqrt(3))
    Colorbar(
        figure[row, 4],
        momentum_plot;
        label=L"C^\perp(\mathbf q)",
        ticks=momentum_color_ticks[case],
    )
end

Label(
    figure[0, 1:4],
    "QNG-optimized projected Gutzwiller + D=1 PPEPS, " *
    "J2/J1=$(round(J2 / J1; digits=3))";
    fontsize=27,
)
colgap!(figure.layout, 1, 15)
colgap!(figure.layout, 2, 35)
colgap!(figure.layout, 3, 15)

pdf_path = joinpath(output_directory, "transverse_correlations_qng_D1.pdf")
save(pdf_path, figure)

println(
    replot_only ? "Used QNG/PPEPS Figure-2 data from: $data_path" :
    "Saved QNG/PPEPS Figure-2 data to: $data_path",
)
println("Saved monopole plot to: $monopole_pdf_path")
println("Saved Figure-2 plot to: $pdf_path")
