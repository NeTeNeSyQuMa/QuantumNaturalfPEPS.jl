"""
    PartonMeanFieldResult

Result of [`solve_parton_mean_field`](@ref). The `energy` is the expectation
value of the physical `J1-J2` Hamiltonian in the unprojected Slater
determinant; it is not the sum of the occupied eigenvalues of `hamiltonian`.
The occupied orbitals can be passed directly to [`gutzwiller_project`](@ref).
"""
struct PartonMeanFieldResult
    hamiltonian::Hermitian{ComplexF64,Matrix{ComplexF64}}
    density_matrix::Matrix{ComplexF64}
    occupied_orbitals::Matrix{ComplexF64}
    orbital_energies::Vector{Float64}
    energy::Float64
    exchange_energy::Float64
    zeeman_energy::Float64
    magnetization::Float64
    local_spins::Array{Float64,3}
    local_densities::Matrix{Float64}
    iterations::Int
    converged::Bool
    residual::Float64
    constraint_field::Float64
    chemical_potentials::Matrix{Float64}
end

const _PARTON_PAULI = (
    ComplexF64[0 1; 1 0],
    ComplexF64[0 -im; im 0],
    ComplexF64[1 0; 0 -1],
)

_parton_site_index(x::Integer, y::Integer, Lx::Integer) = (y - 1) * Lx + x
_parton_spin_range(site::Integer) = (2site - 1):(2site)

function _parton_bonds(Lx, Ly, boundary, shear, shell)
    if boundary == :open
        return _triangular_open_bonds(Lx, Ly; shell)
    elseif boundary == :periodic
        return triangular_torus_bonds(Lx, Ly; shear, shell)
    end
    throw(ArgumentError("boundary must be :open or :periodic, got $boundary"))
end

function _validate_parton_lattice(Lx, Ly, boundary, shear)
    boundary in (:open, :periodic) || throw(ArgumentError(
        "boundary must be :open or :periodic, got $boundary",
    ))
    if boundary == :open
        Lx >= 1 || throw(ArgumentError("Lx must be positive"))
        Ly >= 1 || throw(ArgumentError("Ly must be positive"))
        iszero(shear) || throw(ArgumentError(
            "shear is only supported with periodic boundaries",
        ))
    else
        Lx >= 3 || throw(ArgumentError("Lx must be at least 3 for periodic boundaries"))
        Ly >= 3 || throw(ArgumentError("Ly must be at least 3 for periodic boundaries"))
    end
    return nothing
end

"""
    parton_density_matrix(occupied_orbitals)

Return the one-body density matrix `P = V * V'` of a Slater determinant. Its
matrix elements use the convention `P[p,q] = <f_q^dagger f_p>`.
"""
function parton_density_matrix(occupied_orbitals::AbstractMatrix{<:Number})
    size(occupied_orbitals, 1) > 0 || throw(ArgumentError(
        "occupied_orbitals must have at least one row",
    ))
    all(isfinite, occupied_orbitals) || throw(ArgumentError(
        "occupied_orbitals must be finite",
    ))
    orbitals = Matrix{ComplexF64}(occupied_orbitals)
    return orbitals * orbitals'
end

function _parton_local_observables(P, Lx, Ly)
    local_spins = zeros(Float64, Lx, Ly, 3)
    local_densities = zeros(Float64, Lx, Ly)

    for y in 1:Ly, x in 1:Lx
        site = _parton_site_index(x, y, Lx)
        Pii = view(P, _parton_spin_range(site), _parton_spin_range(site))
        local_densities[x, y] = real(tr(Pii))
        for component in 1:3
            local_spins[x, y, component] = real(tr(_PARTON_PAULI[component] * Pii)) / 2
        end
    end
    return local_spins, local_densities
end

function _parton_mean_field_measurements(
    P,
    Lx,
    Ly;
    J1,
    J2,
    H,
    boundary,
    shear,
)
    local_spins, local_densities = _parton_local_observables(P, Lx, Ly)
    exchange_energy = 0.0

    for (shell, coupling) in ((1, J1), (2, J2))
        iszero(coupling) && continue
        for bond in _parton_bonds(Lx, Ly, boundary, shear, shell)
            i = _parton_site_index(bond.source..., Lx)
            j = _parton_site_index(bond.target..., Lx)
            Pij = view(P, _parton_spin_range(i), _parton_spin_range(j))
            Pji = view(P, _parton_spin_range(j), _parton_spin_range(i))
            si = view(local_spins, bond.source..., :)
            sj = view(local_spins, bond.target..., :)

            # Wick's theorem for i != j:
            # <Si.Sj> = si.sj - 1/2 Tr(Pij)Tr(Pji)
            #                       + 1/4 Tr(Pij Pji).
            spin_correlation = dot(si, sj) -
                real(tr(Pij) * tr(Pji)) / 2 +
                real(tr(Pij * Pji)) / 4
            exchange_energy += coupling * spin_correlation
        end
    end

    total_Sz = sum(view(local_spins, :, :, 3))
    zeeman_energy = -H * total_Sz
    magnetization = 2total_Sz / (Lx * Ly)
    return (;
        energy=exchange_energy + zeeman_energy,
        exchange_energy,
        zeeman_energy,
        magnetization,
        local_spins,
        local_densities,
    )
end

"""
    parton_mean_field_energy(
        density_matrix, Lx, Ly;
        J1, J2, H=0, boundary=:periodic, shear=0,
    )

Evaluate the physical triangular-lattice `J1-J2` Heisenberg energy in an
*unprojected* Slater determinant using Wick's theorem. `density_matrix` is the
`2N x 2N` matrix returned by [`parton_density_matrix`](@ref), in the
interleaved basis `(1 up, 1 down, 2 up, 2 down, ...)`.

This is the Hartree-Fock variational energy, including its double-counting
correction. In general it is not equal to the sum of occupied eigenvalues of
an auxiliary Hamiltonian.
"""
function parton_mean_field_energy(
    density_matrix::AbstractMatrix{<:Number},
    Lx::Integer,
    Ly::Integer;
    J1::Real,
    J2::Real,
    H::Real=0.0,
    boundary::Symbol=:periodic,
    shear::Integer=0,
)
    _validate_parton_lattice(Lx, Ly, boundary, shear)
    N = Lx * Ly
    size(density_matrix) == (2N, 2N) || throw(DimensionMismatch(
        "density_matrix must have size $(2N) x $(2N), got $(size(density_matrix))",
    ))
    ishermitian(density_matrix) || throw(ArgumentError("density_matrix must be Hermitian"))

    measurements = _parton_mean_field_measurements(
        density_matrix,
        Lx,
        Ly;
        J1,
        J2,
        H,
        boundary,
        shear,
    )
    return measurements.energy
end

function _parton_exchange_fock(P, Lx, Ly; J1, J2, boundary, shear)
    N = Lx * Ly
    F = zeros(ComplexF64, 2N, 2N)
    local_spins, _ = _parton_local_observables(P, Lx, Ly)
    identity2 = Matrix{ComplexF64}(I, 2, 2)

    for (shell, coupling) in ((1, J1), (2, J2))
        iszero(coupling) && continue
        for bond in _parton_bonds(Lx, Ly, boundary, shear, shell)
            i = _parton_site_index(bond.source..., Lx)
            j = _parton_site_index(bond.target..., Lx)
            irange = _parton_spin_range(i)
            jrange = _parton_spin_range(j)
            Pij = view(P, irange, jrange)

            si = view(local_spins, bond.source..., :)
            sj = view(local_spins, bond.target..., :)
            for component in 1:3
                F[irange, irange] .+=
                    (coupling * sj[component] / 2) .* _PARTON_PAULI[component]
                F[jrange, jrange] .+=
                    (coupling * si[component] / 2) .* _PARTON_PAULI[component]
            end

            Fij = coupling .* (-tr(Pij) / 2 .* identity2 .+ Pij ./ 4)
            F[irange, jrange] .+= Fij
            F[jrange, irange] .+= Fij'
        end
    end
    return Hermitian(F)
end

function _default_parton_initial_hamiltonian(Lx, Ly, boundary, shear, J1, J2, H)
    N = Lx * Ly
    matrix = zeros(ComplexF64, 2N, 2N)
    shell = iszero(J1) && !iszero(J2) ? 2 : 1
    for bond in _parton_bonds(Lx, Ly, boundary, shear, shell)
        i = _parton_site_index(bond.source..., Lx)
        j = _parton_site_index(bond.target..., Lx)
        for spin in 0:1
            matrix[2i-spin, 2j-spin] -= 1
            matrix[2j-spin, 2i-spin] -= 1
        end
    end
    # A tiny deterministic seed avoids an arbitrary basis choice at an exactly
    # degenerate Fermi level while remaining negligible on the exchange scale.
    seed_field = iszero(H) ? 1e-8 : H
    for site in 1:N
        matrix[2site-1, 2site-1] -= seed_field / 2
        matrix[2site, 2site] += seed_field / 2
    end
    return Hermitian(matrix)
end

function _parton_add_constraints!(
    F,
    Lx,
    Ly,
    chemical_potentials,
    effective_field,
)
    for y in 1:Ly, x in 1:Lx
        site = _parton_site_index(x, y, Lx)
        up, down = 2site - 1, 2site
        F[up, up] += chemical_potentials[x, y] - effective_field / 2
        F[down, down] += chemical_potentials[x, y] + effective_field / 2
    end
    return F
end

"""
    solve_parton_mean_field(
        Lx, Ly;
        J1, J2, H=0, boundary=:periodic, shear=0,
        target_m=nothing, initial_hamiltonian=nothing,
        initial_density=nothing, mixing=0.35, constraint_step=0.5,
        initial_constraint_field=0,
        enforce_local_density=true, fock_projection=identity,
        tolerance=1e-8, maxiter=500, verbosity=0,
    )

Solve the unrestricted number-conserving Abrikosov-fermion Hartree-Fock
equations for the triangular `J1-J2` Heisenberg model. Exactly `N=Lx*Ly`
one-particle orbitals are filled. The local Lagrange multipliers enforce
`<n_i>=1` and `target_m`, when supplied, constrains the normalized expected
magnetization `2 sum_i <S_i^z>/N`.

`initial_hamiltonian` is the natural way to seed a Y, umbrella, stripe, or
flux saddle. An initial ansatz does not by itself constrain the iteration to
remain in that symmetry or flux sector. To impose such a restriction, pass a
callable `fock_projection(F)` which returns the Hermitian matrix allowed by
the ansatz. The uniform magnetization and local-density constraint fields are
added after this projection.

When `target_m` is given, `H` affects only the reported physical energy
`E_exchange - H*S^z`; a separately optimized `constraint_field` fixes the
mean magnetization. Its starting value is `initial_constraint_field`. Without
`target_m`, the physical field `H` is included in the self-consistent
Hamiltonian.
"""
function solve_parton_mean_field(
    Lx::Integer,
    Ly::Integer;
    J1::Real,
    J2::Real,
    H::Real=0.0,
    boundary::Symbol=:periodic,
    shear::Integer=0,
    target_m::Union{Nothing,Real}=nothing,
    initial_hamiltonian=nothing,
    initial_density=nothing,
    mixing::Real=0.35,
    constraint_step::Real=0.5,
    initial_constraint_field::Real=0.0,
    enforce_local_density::Bool=true,
    fock_projection=identity,
    tolerance::Real=1e-8,
    maxiter::Integer=500,
    verbosity::Integer=0,
)
    _validate_parton_lattice(Lx, Ly, boundary, shear)
    N = Lx * Ly
    0 < mixing <= 1 || throw(ArgumentError("mixing must lie in (0, 1]"))
    constraint_step > 0 || throw(ArgumentError("constraint_step must be positive"))
    tolerance > 0 || throw(ArgumentError("tolerance must be positive"))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    if !isnothing(target_m)
        -1 <= target_m <= 1 || throw(ArgumentError("target_m must lie in [-1, 1]"))
    end
    isnothing(initial_hamiltonian) || isnothing(initial_density) || throw(ArgumentError(
        "supply at most one of initial_hamiltonian and initial_density",
    ))

    if !isnothing(initial_density)
        size(initial_density) == (2N, 2N) || throw(DimensionMismatch(
            "initial_density must have size $(2N) x $(2N), got $(size(initial_density))",
        ))
        ishermitian(initial_density) || throw(ArgumentError("initial_density must be Hermitian"))
        P = Matrix{ComplexF64}(initial_density)
    else
        initial = if isnothing(initial_hamiltonian)
            _default_parton_initial_hamiltonian(Lx, Ly, boundary, shear, J1, J2, H)
        else
            size(initial_hamiltonian) == (2N, 2N) || throw(DimensionMismatch(
                "initial_hamiltonian must have size $(2N) x $(2N), " *
                "got $(size(initial_hamiltonian))",
            ))
            ishermitian(initial_hamiltonian) || throw(ArgumentError(
                "initial_hamiltonian must be Hermitian",
            ))
            Hermitian(Matrix{ComplexF64}(initial_hamiltonian))
        end
        initial_spectrum = eigen(initial)
        initial_orbitals = initial_spectrum.vectors[:, 1:N]
        P = parton_density_matrix(initial_orbitals)
    end

    chemical_potentials = zeros(Float64, Lx, Ly)
    constraint_field = isnothing(target_m) ? Float64(H) : Float64(initial_constraint_field)
    occupied_orbitals = zeros(ComplexF64, 2N, N)
    orbital_energies = zeros(Float64, 2N)
    final_hamiltonian = Hermitian(zeros(ComplexF64, 2N, 2N))
    converged = false
    residual = Inf
    iterations = 0
    Psolution = copy(P)

    for iteration in 1:maxiter
        iterations = iteration
        exchange_fock = Matrix(_parton_exchange_fock(
            P,
            Lx,
            Ly;
            J1,
            J2,
            boundary,
            shear,
        ))
        projected_fock = fock_projection(Hermitian(exchange_fock))
        size(projected_fock) == (2N, 2N) || throw(DimensionMismatch(
            "fock_projection must return a $(2N) x $(2N) matrix, " *
            "got $(size(projected_fock))",
        ))
        ishermitian(projected_fock) || throw(ArgumentError(
            "fock_projection must return a Hermitian matrix",
        ))

        F = Matrix{ComplexF64}(projected_fock)
        effective_field = isnothing(target_m) ? H : constraint_field
        _parton_add_constraints!(
            F,
            Lx,
            Ly,
            chemical_potentials,
            effective_field,
        )
        final_hamiltonian = Hermitian(F)
        spectrum = eigen(final_hamiltonian)
        orbital_energies = Vector{Float64}(spectrum.values)
        occupied_orbitals = Matrix{ComplexF64}(spectrum.vectors[:, 1:N])
        Pnew = parton_density_matrix(occupied_orbitals)
        Psolution = Pnew
        local_spins, local_densities = _parton_local_observables(Pnew, Lx, Ly)
        current_m = 2sum(view(local_spins, :, :, 3)) / N

        density_residual = maximum(abs, Pnew - P)
        occupancy_residual = enforce_local_density ?
            maximum(abs, local_densities .- 1) : 0.0
        magnetization_residual = isnothing(target_m) ? 0.0 : abs(current_m - target_m)
        residual = max(density_residual, occupancy_residual, magnetization_residual)

        if verbosity > 0 && (iteration == 1 || iteration % verbosity == 0)
            println(
                "parton HF iteration $iteration: residual=$residual, " *
                "m=$current_m, max|n_i-1|=$occupancy_residual",
            )
        end

        if residual < tolerance
            converged = true
            break
        end

        if enforce_local_density
            chemical_potentials .+= constraint_step .* (local_densities .- 1)
            # A uniform chemical-potential shift does not change a fixed-N state.
            chemical_potentials .-= sum(chemical_potentials) / N
        end
        if !isnothing(target_m)
            constraint_field += constraint_step * (target_m - current_m)
        end
        P .= (1 - mixing) .* P .+ mixing .* Pnew
    end

    measurements = _parton_mean_field_measurements(
        Psolution,
        Lx,
        Ly;
        J1,
        J2,
        H,
        boundary,
        shear,
    )
    return PartonMeanFieldResult(
        final_hamiltonian,
        Psolution,
        occupied_orbitals,
        orbital_energies,
        measurements.energy,
        measurements.exchange_energy,
        measurements.zeeman_energy,
        measurements.magnetization,
        measurements.local_spins,
        measurements.local_densities,
        iterations,
        converged,
        residual,
        constraint_field,
        chemical_potentials,
    )
end

"""
    gutzwiller_project(result::PartonMeanFieldResult; Nup=nothing)

Project the converged occupied Hartree-Fock orbitals to one fermion per site.
"""
gutzwiller_project(result::PartonMeanFieldResult; kwargs...) =
    gutzwiller_project(result.occupied_orbitals; kwargs...)


#= Auxiliary functions for constructing triangular-lattice mean-field ansätze =#

function _check_three_sublattice_torus(Lx::Integer, Ly::Integer, shear::Integer)
    mod(2Lx, 3) == 0 || throw(ArgumentError(
        "three-sublattice order requires 2Lx divisible by 3, got Lx=$Lx",
    ))
    mod(2shear - Ly, 3) == 0 || throw(ArgumentError(
        "three-sublattice order requires 2shear-Ly divisible by 3, " *
        "got Ly=$Ly and shear=$shear",
    ))
    return nothing
end

function _check_stripe_torus(Lx::Integer, Ly::Integer, shear::Integer)
    iseven(Lx) || throw(ArgumentError(
        "the selected canted-stripe orientation requires even Lx, got $Lx",
    ))
    iseven(shear - Ly) || throw(ArgumentError(
        "the selected canted-stripe orientation requires even shear-Ly, " *
        "got Ly=$Ly and shear=$shear",
    ))
    return nothing
end

_triangular_k_index(x::Integer, y::Integer) = mod(2(x - 1) - (y - 1), 3)
_triangular_k_angle(x::Integer, y::Integer) = 2pi * _triangular_k_index(x, y) / 3
_triangular_sublattice(x::Integer, y::Integer) = _triangular_k_index(x, y) + 1
_triangular_stripe_sign(x::Integer, y::Integer) =
    iseven((x - 1) - (y - 1)) ? 1.0 : -1.0

function _real_ansatz_parameters(
    η::AbstractVector,
    expected_length::Integer,
    state_name::AbstractString,
)
    length(η) == expected_length || throw(DimensionMismatch(
        "$state_name requires $expected_length mean-field parameters, " *
        "got $(length(η))",
    ))
    all(isreal, η) || throw(ArgumentError(
        "$state_name mean-field parameters must be real",
    ))
    all(isfinite, η) || throw(ArgumentError(
        "$state_name mean-field parameters must be finite",
    ))
    return Float64.(real.(η))
end

"""
    staggered_pi_flux_hoppings(Lx, Ly; amplitude=1.0)

Construct an `Lx × Ly × 3` hopping array for the triangular-lattice Dirac
spin-liquid ansatz with staggered `[0, π]` flux through the two triangles of
every square-grid plaquette. The direction index follows
`1 => (1, 0)`, `2 => (0, 1)`, and `3 => (1, 1)`.

The gauge choice is

```text
t₁(x,y) = (-1)^y amplitude,
t₂(x,y) =        amplitude,
t₃(x,y) = (-1)^y amplitude.
```

It gives zero flux around the oriented triangle
`(x,y) → (x+1,y) → (x+1,y+1) → (x,y)` and `π` flux around
`(x,y) → (x+1,y+1) → (x,y+1) → (x,y)`. Consequently, every
primitive rhombus has `π` flux. This gauge doubles the unit cell along `y`,
so `Ly` must be even on a torus.
"""
function staggered_pi_flux_hoppings(
    Lx::Integer,
    Ly::Integer;
    amplitude::Real=1.0,
)
    Lx >= 3 || throw(ArgumentError("Lx must be at least 3 to avoid duplicate torus bonds"))
    Ly >= 3 || throw(ArgumentError("Ly must be at least 3 to avoid duplicate torus bonds"))
    iseven(Ly) || throw(ArgumentError(
        "Ly must be even for the chosen staggered-[0, π] torus gauge",
    ))
    amplitude > 0 || throw(ArgumentError("amplitude must be positive"))

    hopping = fill(ComplexF64(amplitude), Lx, Ly, 3)
    for y in 1:Ly
        staggered_sign = isodd(y) ? -1.0 : 1.0
        hopping[:, y, 1] .*= staggered_sign
        hopping[:, y, 3] .*= staggered_sign
    end

    return hopping
end

"""
    uniform_flux_staggered_pi_hoppings(Lx, Ly, Q; amplitude=1.0)

Add `Q` quantized units of uniform U(1) flux to the staggered-`[0, pi]`
nearest-neighbor hopping ansatz on an `Lx x Ly` torus. The extra flux through
each primitive rhombus is

```math
phi = 2 pi Q/(L_x L_y),
```

and each elementary triangle carries `phi/2` in addition to its staggered
zero- or `pi`-flux background. Magnetic boundary phases are included in the
returned `Lx x Ly x 3` hopping array. `Q` is understood modulo `Lx*Ly`.

The direction index is the same as for [`staggered_pi_flux_hoppings`](@ref):
`1 => (1,0)`, `2 => (0,1)`, and `3 => (1,1)`.

Returns:
- hopping_Q: `Lx x Ly x 3` array of complex hoppings given the staggered-`[0, pi]`
             background with the uniform flux added.
"""
function uniform_flux_staggered_pi_hoppings(
    Lx::Integer,
    Ly::Integer,
    Q::Integer;
    amplitude::Real=1.0,
)
    N = Lx * Ly
    additional_flux_per_rhombus = 2pi * mod(Q, N) / N
    hopping = staggered_pi_flux_hoppings(Lx, Ly; amplitude)

    # Landau gauge with the compensating y-boundary phase required on a torus.
    for y in 1:Ly, x in 1:Lx
        x0, y0 = x - 1, y - 1
        phases = if y < Ly
            (
                -additional_flux_per_rhombus * y0,
                0.0,
                -additional_flux_per_rhombus * (y0 + 0.5),
            )
        else
            y_boundary_phase = additional_flux_per_rhombus * Ly * x0
            (
                -additional_flux_per_rhombus * y0,
                y_boundary_phase,
                y_boundary_phase + additional_flux_per_rhombus / 2,
            )
        end
        hopping[x, y, :] .*= exp.(im .* phases)
    end

    return hopping, additional_flux_per_rhombus
end

"""
    y_hopping_fields(Lx, Ly, η; hopping_amplitude=1, shear=0)

Construct the hopping and fictitious-field arrays for the triangular-lattice
Y ansatz. The reduced variational vector follows the paper's ordering
`η = [h1, h2, h3, Delta]`. Sublattices are selected by
`mod(2(x-1)-(y-1), 3)` and carry fields

```text
A: (  0, 0,  h1)
B: ( h2, 0, -h3)
C: (-h2, 0, -h3).
```

The `Delta`-magnitude bonds connect the B and C sublattices. All other bonds
have magnitude `hopping_amplitude`; their signs retain the staggered `[0, pi]`
Dirac flux pattern.
"""
function y_hopping_fields(
    Lx::Integer,
    Ly::Integer,
    η::AbstractVector;
    hopping_amplitude::Real=1.0,
    shear::Integer=0,
)
    _check_three_sublattice_torus(Lx, Ly, shear)
    h1, h2, h3, delta = _real_ansatz_parameters(η, 4, "the Y ansatz")
    delta > 0 || throw(ArgumentError("the Y hopping magnitude Delta must be positive"))

    hopping = staggered_pi_flux_hoppings(Lx, Ly; amplitude=hopping_amplitude)
    for bond in triangular_torus_bonds(Lx, Ly; shear)
        source_sublattice = _triangular_sublattice(bond.source...)
        target_sublattice = _triangular_sublattice(bond.target...)
        if (source_sublattice == 2 && target_sublattice == 3) ||
           (source_sublattice == 3 && target_sublattice == 2)
            x, y = bond.source
            hopping[x, y, bond.direction] *= delta / hopping_amplitude
        end
    end

    fields = zeros(Float64, Lx, Ly, 3)
    for y in 1:Ly, x in 1:Lx
        sublattice = _triangular_sublattice(x, y)
        fields[x, y, :] .= if sublattice == 1
            (0.0, 0.0, h1)
        elseif sublattice == 2
            (h2, 0.0, -h3)
        else
            (-h2, 0.0, -h3)
        end
    end
    return hopping, fields
end

"""
    umbrella_hopping_fields(Lx, Ly, η; hopping_amplitude=1, shear=0)

Construct the hopping and fields for the triangular-lattice umbrella ansatz,
whose only optimized mean-field parameter is `η = [h]`. The hopping magnitudes
are uniform and retain the staggered `[0, pi]` flux. The field is
`M_i = h(cos(K*R_i), sin(K*R_i), 0)` with
`K*R_i = 2pi(2(x-1)-(y-1))/3`.
"""
function umbrella_hopping_fields(
    Lx::Integer,
    Ly::Integer,
    η::AbstractVector;
    hopping_amplitude::Real=1.0,
    shear::Integer=0,
)
    _check_three_sublattice_torus(Lx, Ly, shear)
    h = only(_real_ansatz_parameters(η, 1, "the umbrella ansatz"))
    hopping = staggered_pi_flux_hoppings(Lx, Ly; amplitude=hopping_amplitude)
    fields = zeros(Float64, Lx, Ly, 3)
    for y in 1:Ly, x in 1:Lx
        angle = _triangular_k_angle(x, y)
        fields[x, y, :] .= (h * cos(angle), h * sin(angle), 0.0)
    end
    return hopping, fields
end

"""
    cs_hopping_fields(
        Lx, Ly, η; hopping_amplitude=1, stripe_delta=0.8, shear=0
    )

Construct one of the three symmetry-related canted-stripe ansatze. The paper
optimizes only `η = [h]`; `stripe_delta` is therefore a fixed hopping magnitude.
The selected orientation is invariant along `a2 = (1, 1)` and alternates along
`a1 = (1, 0)`, with `M_i = ((-1)^(x-y) h, 0, 0)`. Direction-3 bonds have
magnitude `stripe_delta`, and the remaining bonds have magnitude
`hopping_amplitude`, all with the staggered `[0, pi]` signs.
"""
function cs_hopping_fields(
    Lx::Integer,
    Ly::Integer,
    η::AbstractVector;
    hopping_amplitude::Real=1.0,
    stripe_delta::Real=0.8,
    shear::Integer=0,
)
    _check_stripe_torus(Lx, Ly, shear)
    h = only(_real_ansatz_parameters(η, 1, "the canted-stripe ansatz"))
    stripe_delta > 0 || throw(ArgumentError(
        "the fixed canted-stripe hopping magnitude must be positive",
    ))
    hopping = staggered_pi_flux_hoppings(Lx, Ly; amplitude=hopping_amplitude)
    hopping[:, :, 3] .*= stripe_delta / hopping_amplitude
    fields = zeros(Float64, Lx, Ly, 3)
    for y in 1:Ly, x in 1:Lx
        fields[x, y, :] .= (h * _triangular_stripe_sign(x, y), 0.0, 0.0)
    end
    return hopping, fields
end

#= triangular-lattice mean-field ansätze for hamiltonian_J1J2_H =#

function _triangular_parton_state(
    Lx::Integer,
    Ly::Integer,
    η::AbstractVector,
    hopping::AbstractArray,
    fields::AbstractArray,
    expand_parameters::Function;
    shear::Integer=0,
    particle_number::Integer=Lx * Ly,
    gap_tolerance::Real=1e-8,
    cache_gradients::Bool=false,
)
    number_of_sites = Lx * Ly
    number_of_modes = 2number_of_sites

    0 <= particle_number <= number_of_modes || throw(ArgumentError(
        "particle_number must lie between 0 and $number_of_modes, " *
        "got $particle_number",
    ))
    gap_tolerance >= 0 || throw(ArgumentError(
        "gap_tolerance must be nonnegative, got $gap_tolerance",
    ))

    initial_Haux = hamiltonian_aux_triangular_torus(
        Lx,
        Ly;
        hopping,
        fields,
        shear,
    )
    chemical_potential = _auxiliary_chemical_potential(
        initial_Haux,
        particle_number,
        gap_tolerance,
    )

    hopping_phases = map(hopping) do value
        iszero(value) ? one(value) : value / abs(value)
    end

    H_BdG_func = _triangular_aux_bdg_function(
        Lx,
        Ly,
        triangular_torus_bonds(Lx, Ly; shear),
        hopping_phases,
        chemical_potential,
        length(η),
        expand_parameters,
    )

    return GaussianState(
        H_BdG_func,
        number_of_modes;
        η=copy(η),
        parity_sector=mod(particle_number, 2),
        target_state=0,
        cache_gradients,
    )
end

function monopole_state(
    Lx::Integer,
    Ly::Integer,
    Q::Integer;
    hopping_amplitude::Real=1.0,
    shear::Integer=0,
    particle_number::Integer=Lx * Ly,
    gap_tolerance::Real=1e-8,
    cache_gradients::Bool=false,
)
    hopping, _ = uniform_flux_staggered_pi_hoppings(
        Lx,
        Ly,
        Q;
        amplitude=hopping_amplitude,
    )
    fields = zeros(Float64, Lx, Ly, 3)

    number_of_sites = Lx * Ly
    fixed_parameters = zeros(Float64, 6number_of_sites)

    for y in 1:Ly, x in 1:Lx
        site = (y - 1) * Lx + x

        for direction in 1:3
            fixed_parameters[3(site - 1) + direction] =
                abs(hopping[x, y, direction])
        end

        field_offset = 3number_of_sites + 3(site - 1)
        fixed_parameters[field_offset+1:field_offset+3] .=
            fields[x, y, :]
    end

    expand_parameters = _ -> fixed_parameters

    return _triangular_parton_state(
        Lx,
        Ly,
        Float64[],
        hopping,
        fields,
        expand_parameters;
        shear,
        particle_number,
        gap_tolerance,
        cache_gradients,
    )
end

"""
    y_state(Lx, Ly; η, kwargs...)

Construct the full unprojected Y Gaussian state with the four optimized
parameters `η = [h1, h2, h3, Delta]`.
"""
function y_state(
    Lx::Integer,
    Ly::Integer;
    η::AbstractVector=[0.35, 0.25, 0.15, 0.8],
    hopping_amplitude::Real=1.0,
    shear::Integer=0,
    particle_number::Integer=Lx * Ly,
    gap_tolerance::Real=1e-8,
    cache_gradients::Bool=false,
)
    hopping, fields = y_hopping_fields(
        Lx,
        Ly,
        η;
        hopping_amplitude,
        shear,
    )
    bonds = triangular_torus_bonds(Lx, Ly; shear)
    delta_bond = falses(Lx, Ly, 3)
    for bond in bonds
        source_sublattice = _triangular_sublattice(bond.source...)
        target_sublattice = _triangular_sublattice(bond.target...)
        if (source_sublattice == 2 && target_sublattice == 3) ||
           (source_sublattice == 3 && target_sublattice == 2)
            x, y = bond.source
            delta_bond[x, y, bond.direction] = true
        end
    end
    parameter_offset = zeros(Float64, 6Lx * Ly)
    parameter_map = zeros(Float64, 6Lx * Ly, 4)
    for y in 1:Ly, x in 1:Lx
        site = (y - 1) * Lx + x
        for direction in 1:3
            hopping_index = 3(site - 1) + direction
            if delta_bond[x, y, direction]
                parameter_map[hopping_index, 4] = 1.0
            else
                parameter_offset[hopping_index] = hopping_amplitude
            end
        end
        field_offset = 3Lx * Ly + 3(site - 1)
        sublattice = _triangular_sublattice(x, y)
        if sublattice == 1
            parameter_map[field_offset + 3, 1] = 1.0
        elseif sublattice == 2
            parameter_map[field_offset + 1, 2] = 1.0
            parameter_map[field_offset + 3, 3] = -1.0
        else
            parameter_map[field_offset + 1, 2] = -1.0
            parameter_map[field_offset + 3, 3] = -1.0
        end
    end
    expand_parameters = parameters -> parameter_offset + parameter_map * parameters
    return _triangular_parton_state(
        Lx,
        Ly,
        η,
        hopping,
        fields,
        expand_parameters;
        shear,
        particle_number,
        gap_tolerance,
        cache_gradients,
    )
end


"""
    umbrella_state(Lx, Ly; η, kwargs...)

Construct the full unprojected umbrella Gaussian state with its single
optimized parameter `η = [h]`.
"""
function umbrella_state(
    Lx::Integer,
    Ly::Integer;
    η::AbstractVector=[0.3],
    hopping_amplitude::Real=1.0,
    shear::Integer=0,
    particle_number::Integer=Lx * Ly,
    gap_tolerance::Real=1e-8,
    cache_gradients::Bool=false,
)

    hopping, fields = umbrella_hopping_fields(
        Lx,
        Ly,
        η;
        hopping_amplitude,
        shear,
    )
    parameter_offset = zeros(Float64, 6Lx * Ly)
    parameter_offset[1:(3Lx * Ly)] .= hopping_amplitude
    parameter_map = zeros(Float64, 6Lx * Ly, 1)
    for y in 1:Ly, x in 1:Lx
        site = (y - 1) * Lx + x
        field_offset = 3Lx * Ly + 3(site - 1)
        angle = _triangular_k_angle(x, y)
        parameter_map[field_offset + 1, 1] = cos(angle)
        parameter_map[field_offset + 2, 1] = sin(angle)
    end
    expand_parameters = parameters -> parameter_offset + parameter_map * parameters
    return _triangular_parton_state(
        Lx,
        Ly,
        η,
        hopping,
        fields,
        expand_parameters;
        shear,
        particle_number,
        gap_tolerance,
        cache_gradients,
    )
end

"""
    cs_state(Lx, Ly; η, kwargs...)

Construct the full unprojected canted-stripe Gaussian state with the paper's
single optimized parameter `η = [h]`. The hopping ratio is supplied separately
through the fixed keyword `stripe_delta`.
"""
function cs_state(
    Lx::Integer,
    Ly::Integer;
    η::AbstractVector=[0.3],
    hopping_amplitude::Real=1.0,
    stripe_delta::Real=0.8,
    shear::Integer=0,
    particle_number::Integer=Lx * Ly,
    gap_tolerance::Real=1e-8,
    cache_gradients::Bool=false,
)
    hopping, fields = cs_hopping_fields(
        Lx,
        Ly,
        η;
        hopping_amplitude,
        stripe_delta,
        shear,
    )
    parameter_offset = zeros(Float64, 6Lx * Ly)
    parameter_map = zeros(Float64, 6Lx * Ly, 1)
    for y in 1:Ly, x in 1:Lx
        site = (y - 1) * Lx + x
        for direction in 1:3
            hopping_index = 3(site - 1) + direction
            parameter_offset[hopping_index] =
                direction == 3 ? stripe_delta : hopping_amplitude
        end
        field_offset = 3Lx * Ly + 3(site - 1)
        parameter_map[field_offset + 1, 1] = _triangular_stripe_sign(x, y)
    end
    expand_parameters = parameters -> parameter_offset + parameter_map * parameters
    return _triangular_parton_state(
        Lx,
        Ly,
        η,
        hopping,
        fields,
        expand_parameters;
        shear,
        particle_number,
        gap_tolerance,
        cache_gradients,
    )
end
