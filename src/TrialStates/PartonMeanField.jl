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
