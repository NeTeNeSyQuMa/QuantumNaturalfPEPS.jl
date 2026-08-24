# Set up the independent plotting environment once from the repository root:
#   julia --project=examples -e 'using Pkg; Pkg.instantiate()'
# Run with the package itself as the primary environment:
#   julia --project=. examples/parton_J1_J2_B_field.jl
# Appending the examples environment keeps package dependencies primary while
# making these optional visualization packages available to this script.
examples_environment = @__DIR__
examples_environment in LOAD_PATH || push!(LOAD_PATH, examples_environment)

using LinearAlgebra
using QuantumNaturalfPEPS
using CairoMakie
using Makie.Colors
using JLD2
using LaTeXStrings
using MathTeXEngine

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

t=1.0

# ---------------------------------------------------------------------------
# 18x18 staggered-[0, pi] Hofstadter spectrum and Wannier diagram
# ---------------------------------------------------------------------------

# Figure 1(a,b) of arXiv:2601.14458 scans the complete flux period on an
# 18x18 cluster. It highlights the C=+/-1 spinon gaps for m=1/3 and m=2/3,
# realized at phi/(2pi)=m/2. These targets are separate from the m=1/12
# magnetization used for the 24-site auxiliary states above.
scan_Lx = scan_Ly = 18
scan_N = scan_Lx * scan_Ly
scan_shear = 0
figure1_magnetizations = (1 // 3, 2 // 3)
figure1_markers = (:circle, :rect)
scan_size_label = "$(scan_Lx)x$(scan_Ly)"
fraction_label(x::Rational) = "$(numerator(x))/$(denominator(x))"

# On a finite torus phi/(2pi)=Q/N. Scanning Q=0,...,N-1 covers one complete
# flux period without duplicating the endpoint.
flux_quanta = collect(0:scan_N-1)
flux_densities = flux_quanta ./ scan_N
single_particle_energies = Matrix{Float64}(undef, scan_N, scan_N)
single_particle_gaps = Matrix{Float64}(undef, scan_N - 1, scan_N)

println("Scanning $(length(flux_quanta)) quantized fluxes on the $(scan_size_label) torus...")
for (column, flux_Q) in enumerate(flux_quanta)
    hopping_Q, _ = uniform_flux_staggered_pi_hoppings(scan_Lx, scan_Ly, flux_Q; amplitude=t)
    Haux_Q = hamiltonian_aux_triangular_torus(
        scan_Lx,
        scan_Ly;
        hopping=hopping_Q,
        shear=scan_shear,
    )

    # The two spin blocks are identical, so diagonalize only the up-spin block.
    Haux_Q_matrix = Matrix(Haux_Q)
    energies_Q = eigvals(Hermitian(Haux_Q_matrix[1:2:end, 1:2:end]))
    single_particle_energies[:, column] .= energies_Q
    single_particle_gaps[:, column] .= diff(energies_Q)
end

# Time reversal/complex conjugation maps +phi to -phi without changing the
# spectrum. Check all quantized flux pairs before plotting.
for flux_Q in flux_quanta
    conjugate_Q = mod(-flux_Q, scan_N)
    @assert isapprox(
        single_particle_energies[:, flux_Q + 1],
        single_particle_energies[:, conjugate_Q + 1];
        atol=1e-9,
    )
end

# Locate the two pairs of C=+/-1 filling gaps highlighted in Fig. 1(a,b).
figure1_targets = map(figure1_magnetizations) do magnetization
    flux_Q = round(Int, scan_N * magnetization / 2)
    @assert 2flux_Q // scan_N == magnetization

    filling_up = (1 + magnetization) / 2
    filling_down = (1 - magnetization) / 2
    Nup = round(Int, scan_N * filling_up)
    Ndown = round(Int, scan_N * filling_down)
    @assert Nup == scan_N ÷ 2 + flux_Q
    @assert Ndown == scan_N ÷ 2 - flux_Q

    energies = @view single_particle_energies[:, flux_Q + 1]
    gap_up = single_particle_gaps[Nup, flux_Q + 1]
    gap_down = single_particle_gaps[Ndown, flux_Q + 1]
    midgap_up = (energies[Nup] + energies[Nup + 1]) / 2
    midgap_down = (energies[Ndown] + energies[Ndown + 1]) / 2

    (;
        magnetization,
        flux_Q,
        flux_density=flux_Q // scan_N,
        filling_up,
        filling_down,
        Nup,
        Ndown,
        gap_up,
        gap_down,
        midgap_up,
        midgap_down,
    )
end

for target in figure1_targets
    println(
        "m=$(fraction_label(target.magnetization)): ",
        "phi/(2pi)=$(fraction_label(target.flux_density)), ",
        "Nup=$(target.Nup), Ndown=$(target.Ndown), ",
        "C=+1 gap=$(target.gap_up), C=-1 gap=$(target.gap_down)",
    )
end

plot_directory = joinpath(@__DIR__, "parton_J1_J2_B_field_plots")
mkpath(plot_directory)

# Keep the numerical scan separate from the figures so it can be replotted
# without diagonalizing all scan_N Hamiltonians again.
scan_data_path = joinpath(plot_directory, "hofstadter_scan_$(scan_size_label).jld2")
JLD2.jldsave(
    scan_data_path;
    scan_Lx,
    scan_Ly,
    scan_N,
    flux_quanta,
    flux_densities,
    single_particle_energies,
    single_particle_gaps,
    figure1_magnetizations,
    figure1_targets,
)

# Hofstadter butterfly: one spin block, since the spectrum is spin degenerate.
spectrum_x = repeat(flux_densities; inner=scan_N)
spectrum_figure = Figure(size=(900, 600))
spectrum_axis = Axis(
    spectrum_figure[1, 1];
    xlabel=L"\mathrm{Flux\ density}\quad \phi/2\pi",
    ylabel=L"\mathrm{Single\!\!-\!particle\ energy}\quad \varepsilon/t",
    title="Staggered [0, π] triangular spinons, $(scan_Lx)×$(scan_Ly) torus",
)
scatter!(
    spectrum_axis,
    spectrum_x,
    vec(single_particle_energies);
    markersize=1.3,
    strokewidth=0,
    color=(colorant"black", 0.55),
)
hlines!(spectrum_axis, [0.0]; color=:gray, linewidth=0.8, linestyle=:dash)
for (target, marker) in zip(figure1_targets, figure1_markers)
    scatter!(
        spectrum_axis,
        fill(target.flux_density, 2),
        [target.midgap_down, target.midgap_up];
        marker,
        markersize=13,
        color=:gold,
        strokecolor=:black,
        strokewidth=1.5,
        label="m=$(fraction_label(target.magnetization))",
    )
end
axislegend(spectrum_axis; position=:lb)
xlims!(spectrum_axis, extrema(flux_densities))
spectrum_path = joinpath(plot_directory, "hofstadter_spectrum_$(scan_size_label).pdf")
save(spectrum_path, spectrum_figure; px_per_unit=2)

# Wannier gap diagram. Each point represents a gap at cumulative filling i/N,
# colored by log10 of its magnitude as in Fig. 1(b). Gaps below 1e-2 are
# clipped to the bottom of the color scale so the dominant Chern-gap lines
# remain visible.
gap_fillings = collect(1:scan_N-1) ./ scan_N
gap_x = repeat(flux_densities; inner=scan_N - 1)
gap_y = repeat(gap_fillings; outer=scan_N)
log_gaps = log10.(max.(single_particle_gaps, 1e-2))
gap_colormap = [colorant"#eff3ff", colorant"#08519c"]
gap_figure = Figure(size=(900, 600))
gap_axis = Axis(
    gap_figure[1, 1];
    xlabel=L"\mathrm{Flux\ density}\quad \phi/2\pi",
    ylabel=L"\mathrm{Cumulative\ filling}\quad y",
    title="Staggered [0, π] spinon Wannier diagram, $(scan_Lx)×$(scan_Ly) torus",
)
gap_scatter = scatter!(
    gap_axis,
    gap_x,
    gap_y;
    color=vec(log_gaps),
    colorrange=(-2, 0),
    colormap=gap_colormap,
    markersize=1.8,
    strokewidth=0,
)
Colorbar(gap_figure[1, 2], gap_scatter; label=L"\log_{10}\Delta\varepsilon")
for (target, marker) in zip(figure1_targets, figure1_markers)
    scatter!(
        gap_axis,
        fill(target.flux_density, 2),
        [target.filling_down, target.filling_up];
        marker,
        markersize=13,
        color=:gold,
        strokecolor=:black,
        strokewidth=1.5,
    )
end
xlims!(gap_axis, extrema(flux_densities))
ylims!(gap_axis, extrema(gap_fillings))
gap_path = joinpath(plot_directory, "hofstadter_gaps_$(scan_size_label).pdf")
save(gap_path, gap_figure; px_per_unit=2)

println("Saved numerical flux scan to: $scan_data_path")
println("Saved Hofstadter spectrum to: $spectrum_path")
println("Saved filling-gap diagram to: $gap_path")
