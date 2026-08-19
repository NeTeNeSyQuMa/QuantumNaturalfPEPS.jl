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

# Auxiliary parton states for the triangular J1-J2 model in a magnetic field.
#
# Square-grid coordinates use the triangular nearest-neighbor directions
#   d1 = (1, 0), d2 = (0, 1), d3 = (1, 1).
# In conventional triangular coordinates, d1 = a1, d3 = a2, and
# d2 = a2 - a1.

# ---------------------------------------------------------------------------
# A commensurate 24-site tilted torus
# ---------------------------------------------------------------------------

Lx, Ly = 6, 4
shear = 2
N = Lx * Ly
J1 = 1.0
J2 = J1 / 8
target_m = 1 / 12

# At fixed target_m the physical Zeeman term is a constant, -B*N*target_m/2,
# and therefore does not enter Haux. It is added later when comparing the
# projected variational energies between magnetization sectors.

# The torus translations are T1 = (6, 0) and T2 = (2, 4) in square-grid
# coordinates. They are commensurate with the three-sublattice K order, the
# selected two-sublattice stripe order, and the doubled [0, pi] flux unit cell.
T1 = (Lx, 0)
T2 = (shear, Ly)
@assert N == 24
@assert mod(2 * T1[1] - T1[2], 3) == 0
@assert mod(2 * T2[1] - T2[2], 3) == 0
@assert iseven(T1[1] - T1[2])
@assert iseven(T2[1] - T2[2])
@assert iseven(Ly)

torus_bonds = triangular_torus_bonds(Lx, Ly; shear)
@assert length(torus_bonds) == 3N

# Fix the redundant overall auxiliary hopping scale.
t = 1.0
base_hopping = staggered_pi_flux_hoppings(Lx, Ly; amplitude=t)

"""
Add `Q` quantized units of uniform U(1) flux to the staggered-[0, pi]
background. The returned flux is measured through a primitive rhombus; each
elementary triangle receives half of it. This discrete Landau gauge includes
the boundary phases required on a torus.
"""
function monopole_hoppings(Lx, Ly, Q; amplitude=1.0)
    return uniform_flux_staggered_pi_hoppings(Lx, Ly, Q; amplitude)
end

# K.R = (2pi/3) (2x-y) in the square-grid coordinates used here. Coordinates
# are shifted to start at zero before evaluating phases and sublattices.
k_index(x, y) = mod(2(x - 1) - (y - 1), 3)
sublattice(x, y) = k_index(x, y) + 1 # A=1, B=2, C=3
k_angle(x, y) = 2pi * k_index(x, y) / 3

function filled_auxiliary_state(name::Symbol, Haux::Hermitian, Noccupied::Int)
    spectrum = eigen(Haux)
    occupied_orbitals = spectrum.vectors[:, 1:Noccupied]
    return (;
        name,
        Haux,
        energies=spectrum.values,
        orbitals=spectrum.vectors,
        occupied_orbitals,
        target_m,
        needs_magnetization_projection=true,
    )
end

# ---------------------------------------------------------------------------
# Y state: Eq. (7) and Fig. 2(a) of arXiv:2607.14766
# ---------------------------------------------------------------------------

h1 = 0.35
h2 = 0.25
h3 = 0.15
Delta_Y = 0.80

Y_fields = zeros(Float64, Lx, Ly, 3)
for y in 1:Ly, x in 1:Lx
    sub = sublattice(x, y)
    Y_fields[x, y, :] .= if sub == 1
        (0.0, 0.0, h1)
    elseif sub == 2
        (h2, 0.0, -h3)
    else
        (-h2, 0.0, -h3)
    end
end

# In this gauge the Delta_Y bonds are chosen to connect the B and C
# sublattices. Changing their positive magnitude preserves the [0, pi] flux.
Y_hopping = copy(base_hopping)
for bond in torus_bonds
    source_sub = sublattice(bond.source...)
    target_sub = sublattice(bond.target...)
    if (source_sub == 2 && target_sub == 3) ||
       (source_sub == 3 && target_sub == 2)
        x, y = bond.source
        Y_hopping[x, y, bond.direction] *= Delta_Y / t
    end
end

Haux_Y = hamiltonian_aux_triangular_torus(
    Lx,
    Ly;
    hopping=Y_hopping,
    fields=Y_fields,
    shear,
)
Y_state = filled_auxiliary_state(:Y, Haux_Y, N)

# ---------------------------------------------------------------------------
# Umbrella/cone state: Eq. (8)
# ---------------------------------------------------------------------------

h_umbrella = 0.30
umbrella_fields = zeros(Float64, Lx, Ly, 3)
for y in 1:Ly, x in 1:Lx
    theta = k_angle(x, y)
    umbrella_fields[x, y, :] .= (
        h_umbrella * cos(theta),
        h_umbrella * sin(theta),
        0.0,
    )
end

Haux_umbrella = hamiltonian_aux_triangular_torus(
    Lx,
    Ly;
    hopping=base_hopping,
    fields=umbrella_fields,
    shear,
)
umbrella_state = filled_auxiliary_state(:umbrella, Haux_umbrella, N)

# ---------------------------------------------------------------------------
# Canted-stripe state: Eq. (9) and Fig. 2(b)
# ---------------------------------------------------------------------------

h_stripe = 0.30
Delta_stripe = 0.80

# The selected stripe is invariant along a2=d3 and alternates along a1.
# In square-grid coordinates its phase is (-1)^(x-y).
stripe_fields = zeros(Float64, Lx, Ly, 3)
for y in 1:Ly, x in 1:Lx
    stripe_sign = iseven((x - 1) - (y - 1)) ? 1.0 : -1.0
    stripe_fields[x, y, :] .= (h_stripe * stripe_sign, 0.0, 0.0)
end

stripe_hopping = copy(base_hopping)
stripe_hopping[:, :, 3] .*= Delta_stripe / t

Haux_canted_stripe = hamiltonian_aux_triangular_torus(
    Lx,
    Ly;
    hopping=stripe_hopping,
    fields=stripe_fields,
    shear,
)
canted_stripe_state = filled_auxiliary_state(:canted_stripe, Haux_canted_stripe, N)

# ---------------------------------------------------------------------------
# Monopole/Landau-level state with Q=1, m=2Q/N=1/12
# ---------------------------------------------------------------------------

Q = round(Int, N * target_m / 2)
m = 2Q / N
@assert m == target_m
Nup = N ÷ 2 + Q
Ndown = N ÷ 2 - Q

# The additional flux per rhombus is 2pi*Q/N, or half of this through each
# elementary triangle.
monopole_hopping, flux_per_rhombus = monopole_hoppings(Lx, Ly, Q; amplitude=t)

Haux_monopole = hamiltonian_aux_triangular_torus(
    Lx,
    Ly;
    hopping=monopole_hopping,
    shear,
)

# Haux_monopole is spin diagonal. Fill Nup and Ndown spatial orbitals
# separately so the Slater determinant has S^z=Q before Gutzwiller projection.
Hmonopole_matrix = Matrix(Haux_monopole)
up_spectrum = eigen(Hermitian(Hmonopole_matrix[1:2:end, 1:2:end]))
down_spectrum = eigen(Hermitian(Hmonopole_matrix[2:2:end, 2:2:end]))
zero_mode_tolerance = 1e-10
@assert count(abs.(up_spectrum.values) .< zero_mode_tolerance) == 2Q
@assert count(abs.(down_spectrum.values) .< zero_mode_tolerance) == 2Q
monopole_occupied_orbitals = zeros(ComplexF64, 2N, N)
monopole_occupied_orbitals[1:2:end, 1:Nup] .= up_spectrum.vectors[:, 1:Nup]
monopole_occupied_orbitals[2:2:end, Nup+1:end] .= down_spectrum.vectors[:, 1:Ndown]

monopole_state = (;
    name=:monopole,
    Haux=Haux_monopole,
    energies=(up=up_spectrum.values, down=down_spectrum.values),
    orbitals=(up=up_spectrum.vectors, down=down_spectrum.vectors),
    occupied_orbitals=monopole_occupied_orbitals,
    Q,
    m,
    Nup,
    Ndown,
    target_m,
    needs_magnetization_projection=false,
    flux_per_rhombus,
    flux_per_triangle=flux_per_rhombus / 2,
)

# Gauge-invariant check of the extra monopole flux, including seam-crossing
# triangles of the tilted torus.
function wrap_torus_site(x, y)
    winding_y = fld(y - 1, Ly)
    wrapped_y = y - winding_y * Ly
    shifted_x = x - winding_y * shear
    return (mod1(shifted_x, Lx), wrapped_y)
end

function spin_up_index(site)
    x, y = wrap_torus_site(site...)
    return 2 * ((y - 1) * Lx + x - 1) + 1
end


function monopole_link(source, target)
    return Hmonopole_matrix[spin_up_index(source), spin_up_index(target)]
end


for y in 1:Ly, x in 1:Lx
    r = (x, y)
    rx = (x + 1, y)
    ry = (x, y + 1)
    rxy = (x + 1, y + 1)

    zero_background_triangle =
        monopole_link(r, rx) * monopole_link(rx, rxy) * monopole_link(rxy, r)
    pi_background_triangle =
        monopole_link(r, rxy) * monopole_link(rxy, ry) * monopole_link(ry, r)

    expected_extra_flux = exp(im * flux_per_rhombus / 2)
    @assert isapprox(zero_background_triangle, expected_extra_flux; atol=1e-12)
    @assert isapprox(pi_background_triangle, -expected_extra_flux; atol=1e-12)
end

# All four auxiliary states are collected here for subsequent projection,
# variational-energy evaluation, and PPEPS dressing.
parton_states = (;
    Y=Y_state,
    umbrella=umbrella_state,
    canted_stripe=canted_stripe_state,
    monopole=monopole_state,
)

# Apply single occupancy and, for the spin-mixing ordered ansatze, restrict the
# resulting spin wavefunction to the target fixed-magnetization sector. The
# monopole Hamiltonian conserves Sz, so its separately filled spin blocks
# already fix Nup and Ndown.
gutzwiller_states = (;
    Y=gutzwiller_project(Y_state.occupied_orbitals; Nup),
    umbrella=gutzwiller_project(umbrella_state.occupied_orbitals; Nup),
    canted_stripe=gutzwiller_project(canted_stripe_state.occupied_orbitals; Nup),
    monopole=gutzwiller_project(
        up_spectrum.vectors[:, 1:Nup],
        down_spectrum.vectors[:, 1:Ndown],
    ),
)

println("Constructed $(length(parton_states)) auxiliary states on the $N-site tilted torus.")
for (name, state) in pairs(parton_states)
    println("  ", rpad(string(name), 15), " size(Haux) = ", size(state.Haux))
end
println("Applied the Gutzwiller projector in the Nup=$Nup, Ndown=$Ndown sector.")

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
    hopping_Q, _ = monopole_hoppings(scan_Lx, scan_Ly, flux_Q; amplitude=t)
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
