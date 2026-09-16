"""
    FixedGutzwillerProjectedState(occupied_orbitals; Nup=nothing)

Represent the single-occupancy Gutzwiller projection of a half-filled Slater
determinant. `occupied_orbitals` must be a `2N × N` matrix whose columns are the
occupied spin orbitals in the interleaved one-particle basis
`(1↑, 1↓, 2↑, 2↓, ...)`.

If `Nup` is specified, amplitudes outside that fixed-magnetization sector are
set to zero. Here spin configurations use the PEPS convention `0 => ↑` and
`1 => ↓`, so `Nup` is the number of zeros in a configuration.

The projected amplitudes are not normalized.
"""
abstract type AbstractGutzwillerProjectedState <: AbstractTrialState end

struct FixedGutzwillerProjectedState{T<:Number} <: AbstractGutzwillerProjectedState
    occupied_orbitals::Matrix{T}
    N::Int
    Nup::Union{Nothing,Int}
end

function FixedGutzwillerProjectedState(
    occupied_orbitals::AbstractMatrix{<:Number};
    Nup::Union{Nothing,Integer}=nothing,
)
    number_of_modes, number_occupied = size(occupied_orbitals)
    iseven(number_of_modes) || throw(DimensionMismatch(
        "occupied_orbitals must have an even number of rows, got $number_of_modes",
    ))

    N = number_of_modes ÷ 2
    number_occupied == N || throw(DimensionMismatch(
        "single-occupancy projection requires a 2N × N orbital matrix; " *
        "got size $(size(occupied_orbitals))",
    ))
    N > 0 || throw(ArgumentError("the projected state must contain at least one site"))

    fixed_Nup = isnothing(Nup) ? nothing : Int(Nup)
    if !isnothing(fixed_Nup) && !(0 <= fixed_Nup <= N)
        throw(ArgumentError("Nup must lie between 0 and N=$N, got $fixed_Nup"))
    end

    T = float(eltype(occupied_orbitals))
    orbitals = Matrix{T}(occupied_orbitals)
    all(isfinite, orbitals) || throw(ArgumentError("occupied_orbitals must be finite"))
    return FixedGutzwillerProjectedState{T}(orbitals, N, fixed_Nup)
end

"""
    FixedGutzwillerProjectedState(up_orbitals, down_orbitals)

Construct a spin-conserving projected state from occupied spatial orbitals.
`up_orbitals` and `down_orbitals` must have `N` rows and together have `N`
columns. The occupied creation operators are ordered with all up-spin orbitals
before all down-spin orbitals. The resulting state has fixed
`Nup = size(up_orbitals, 2)`.
"""
function FixedGutzwillerProjectedState(
    up_orbitals::AbstractMatrix{<:Number},
    down_orbitals::AbstractMatrix{<:Number},
)
    N = size(up_orbitals, 1)
    size(down_orbitals, 1) == N || throw(DimensionMismatch(
        "up_orbitals and down_orbitals must have the same number of rows",
    ))

    Nup = size(up_orbitals, 2)
    Ndown = size(down_orbitals, 2)
    Nup + Ndown == N || throw(DimensionMismatch(
        "single-occupancy projection requires Nup + Ndown = N; " *
        "got $Nup + $Ndown != $N",
    ))

    T = float(promote_type(eltype(up_orbitals), eltype(down_orbitals)))
    occupied_orbitals = zeros(T, 2N, N)
    occupied_orbitals[1:2:end, 1:Nup] .= up_orbitals
    occupied_orbitals[2:2:end, Nup+1:end] .= down_orbitals
    return FixedGutzwillerProjectedState(occupied_orbitals; Nup)
end

Base.eltype(::FixedGutzwillerProjectedState{T}) where {T} = T
Parameters(::FixedGutzwillerProjectedState) = Float64[]

function _gutzwiller_rows(
    state::AbstractGutzwillerProjectedState,
    spin_configuration::AbstractArray{<:Integer},
)
    length(spin_configuration) == state.N || throw(DimensionMismatch(
        "spin configuration must have length $(state.N), got $(length(spin_configuration))",
    ))

    rows = Vector{Int}(undef, state.N)
    number_up = 0
    for (site, spin) in enumerate(vec(spin_configuration))
        (spin == 0 || spin == 1) || throw(DomainError(
            spin,
            "spin entries must be 0 (up) or 1 (down)",
        ))
        number_up += spin == 0
        rows[site] = 2site - 1 + spin
    end
    return rows, number_up
end

"""
    gutzwiller_amplitude(state, spin_configuration)

Return the unnormalized projected amplitude for a spin configuration. The
configuration may be a vector or an `Lx × Ly` matrix; matrix inputs are read in
Julia column-major order. Entries are `0` for up and `1` for down.

The physical spin basis is defined by site-major fermion ordering,

```text
|σ₁ … σ_N⟩ = f†_{1σ₁} f†_{2σ₂} ⋯ f†_{Nσ_N} |0⟩.
```

With this convention the amplitude is the determinant of the rows selected
from `occupied_orbitals`. This automatically includes the fermionic sign that
appears when a spin-conserving determinant is factorized into up and down
blocks.
"""
function gutzwiller_amplitude(
    state::AbstractGutzwillerProjectedState,
    spin_configuration::AbstractArray{<:Integer},
)
    rows, number_up = _gutzwiller_rows(state, spin_configuration)
    if !isnothing(state.Nup) && number_up != state.Nup
        return zero(eltype(state))
    end
    return det(view(state.occupied_orbitals, rows, :))
end

"""
    gutzwiller_weight(state, spin_configuration)

Return the unnormalized configuration weight `|Ψ_G(σ)|²`.
"""
gutzwiller_weight(state::AbstractGutzwillerProjectedState, spin_configuration) =
    abs2(gutzwiller_amplitude(state, spin_configuration))

get_amplitude(state::AbstractGutzwillerProjectedState, spin_configuration::Vector{Int}) =
    gutzwiller_amplitude(state, spin_configuration)

function get_prob(
    state::AbstractGutzwillerProjectedState,
    spin_configuration::AbstractVector{<:Integer},
)
    return gutzwiller_weight(state, spin_configuration)
end

function get_prob(
    state::AbstractGutzwillerProjectedState,
    spin_configuration::Dict{Int,Int},
)
    if length(spin_configuration) != state.N ||
       any(!haskey(spin_configuration, site) for site in 1:state.N)
        throw(ArgumentError(
            "partial probabilities of a Gutzwiller-projected state are not " *
            "available through this interface",
        ))
    end
    configuration = [spin_configuration[site] for site in 1:state.N]
    return gutzwiller_weight(state, configuration)
end

"""
    gutzwiller_project(occupied_orbitals; Nup=nothing)
    gutzwiller_project(up_orbitals, down_orbitals)
    gutzwiller_project(Haux::Hermitian; Nup=nothing)

Construct a [`FixedGutzwillerProjectedState`](@ref) either from occupied orbitals, from
separate occupied up/down spatial orbitals, or by filling the lowest `N`
one-particle levels of a `2N × 2N` Hermitian auxiliary Hamiltonian.
"""
gutzwiller_project(occupied_orbitals::AbstractMatrix{<:Number}; kwargs...) =
    FixedGutzwillerProjectedState(occupied_orbitals; kwargs...)

gutzwiller_project(
    up_orbitals::AbstractMatrix{<:Number},
    down_orbitals::AbstractMatrix{<:Number},
) = FixedGutzwillerProjectedState(up_orbitals, down_orbitals)

function gutzwiller_project(
    Haux::Hermitian;
    Nup::Union{Nothing,Integer}=nothing,
)
    size(Haux, 1) == size(Haux, 2) || throw(DimensionMismatch(
        "Haux must be square, got size $(size(Haux))",
    ))
    iseven(size(Haux, 1)) || throw(DimensionMismatch(
        "Haux must have size 2N × 2N, got size $(size(Haux))",
    ))

    N = size(Haux, 1) ÷ 2
    spectrum = eigen(Haux)
    return FixedGutzwillerProjectedState(spectrum.vectors[:, 1:N]; Nup)
end

"""
    ParameterizedGutzwillerProjectedState(gaussian_state; Nup=nothing, gap_tolerance=1e-8)

Represent a parameterized single-occupancy Gutzwiller projection of a
number-conserving, half-filled [`GaussianState`](@ref) on `2N` interleaved
spin-orbitals. The projected state copies the Gaussian state's mean-field
parameters and reuses its auxiliary-Hamiltonian function. It caches the
occupied-orbital correlation matrix needed for parent-Gaussian sequential
proposal probabilities, and refreshes that cache when `write!(state, η)` is
called.

The Gaussian state must have no anomalous BdG pairing block and must contain
exactly `N` occupied physical orbitals separated from the unoccupied orbitals
by `gap_tolerance`. Spin configurations use `0 => ↑` and `1 => ↓`. Supplying
`Nup` additionally projects into a fixed-magnetization sector.
"""
mutable struct ParameterizedGutzwillerProjectedState <: AbstractGutzwillerProjectedState
    H_BdG_func::Function
    η::AbstractVector{<:Number}
    number_of_modes::Int
    occupied_orbitals::Matrix{ComplexF64}
    correlation_matrix::Matrix{ComplexF64}
    unoccupied_orbitals::Matrix{ComplexF64}
    occupied_energies::Vector{Float64}
    unoccupied_energies::Vector{Float64}
    N::Int
    Nup::Union{Nothing,Int}
    gap_tolerance::Float64
end

"""
    ProjectedGaussianSchurCache(state; order=1:state.N)

Direct-sampling cache for a parameterized Gutzwiller-projected Slater state.
The parent Slater correlation matrix is conditioned one spin-orbital at a time
and shrunk after every physical spin decision. If `state.Nup` is fixed,
branches that cannot reach that sector are assigned zero proposal weight.
"""
mutable struct ProjectedGaussianSchurCache
    correlation_matrix::Matrix{ComplexF64}
    remaining_sites::Vector{Int}
    target_Nup::Union{Nothing,Int}
    measured_up::Int
end

const _PROJECTED_SCHUR_PROBABILITY_TOLERANCE = 10sqrt(eps(Float64))

function ProjectedGaussianSchurCache(
    state::ParameterizedGutzwillerProjectedState;
    order::AbstractVector{<:Integer}=collect(1:state.N),
)
    length(order) == state.N || throw(DimensionMismatch(
        "sampling order must contain $(state.N) sites, got $(length(order))",
    ))
    sites = collect(Int, order)
    sort(sites) == collect(1:state.N) || throw(ArgumentError(
        "sampling order must be a permutation of 1:$(state.N)",
    ))
    modes = Vector{Int}(undef, 2state.N)
    for (position, site) in enumerate(sites)
        modes[2position - 1] = 2site - 1
        modes[2position] = 2site
    end
    return ProjectedGaussianSchurCache(
        Matrix{ComplexF64}(state.correlation_matrix[modes, modes]),
        sites,
        state.Nup,
        0,
    )
end

"""
    projected_conditional_probabilities(cache; lookahead_depth=0)

Return the two parent-Slater proposal weights for the next physical spin.
Depth 0 uses the original one-site marginal. Depths 1 and 2 additionally sum
over physical spin assignments on the next one or two sites in sampling order,
discarding assignments that cannot reach `target_Nup`. This projects only the
small lookahead window, not all remaining sites. The weights are not normalized;
the sampler combines them with the PEPS weights and records the resulting
normalized proposal in `logpc`. The final target amplitude is unchanged.
"""
function projected_conditional_probabilities(cache::ProjectedGaussianSchurCache; lookahead_depth=0)
    _validate_lookahead_depth(lookahead_depth)
    isempty(cache.remaining_sites) && throw(ArgumentError("all sites have already been measured"))
    correlation = cache.correlation_matrix
    up_density = real(correlation[1, 1])
    down_density = real(correlation[2, 2])
    coherence_squared = abs2(correlation[1, 2])
    # Determinantal two-mode probabilities. Physical convention:
    # 0 => up => (n_up,n_down)=(1,0), and 1 => down => (0,1).
    probabilities = [
        up_density * (1 - down_density) + coherence_squared,
        (1 - up_density) * down_density + coherence_squared,
    ]
    tolerance = _PROJECTED_SCHUR_PROBABILITY_TOLERANCE
    for index in eachindex(probabilities)
        (-tolerance <= probabilities[index] <= 1 + tolerance) || throw(DomainError(
            probabilities[index],
            "the parent Slater correlation matrix gives an invalid spin probability",
        ))
        probabilities[index] = clamp(probabilities[index], 0.0, 1.0)
    end
    if !isnothing(cache.target_Nup) # tracks how many up spins have already been selected
        sites_after = length(cache.remaining_sites) - 1
        for spin in 0:1 # spin=0 => up, spin=1 => down
            up_after = cache.measured_up + (spin == 0)
            if up_after > cache.target_Nup || up_after + sites_after < cache.target_Nup
                probabilities[spin + 1] = 0.0
            end
        end
    end
    if lookahead_depth > 0 && length(cache.remaining_sites) > 1
        return _projected_window_probabilities(cache, lookahead_depth, probabilities)
    end
    return probabilities
end

function _validate_lookahead_depth(depth)
    depth isa Integer && depth in 0:2 || throw(ArgumentError(
        "lookahead_depth must be 0, 1, or 2, got $depth",
    ))
    return depth
end

function _projected_window_probabilities(cache, depth, one_site_probabilities)
    window = min(depth + 1, length(cache.remaining_sites))
    modes = 2window
    # Gaussian occupation marginals only need this principal submatrix.
    # Never copy/condition the full remaining correlation matrix for lookahead.
    correlation = @view cache.correlation_matrix[1:modes, 1:modes]
    measurement = Matrix{ComplexF64}(undef, modes, modes)
    probabilities = zeros(Float64, 2)
    sites_after = length(cache.remaining_sites) - window
    for code in 0:(1 << window)-1
        spin = code & 1
        iszero(one_site_probabilities[spin + 1]) && continue
        up_after = cache.measured_up + window - count_ones(code)
        if !isnothing(cache.target_Nup) &&
           (up_after > cache.target_Nup || up_after + sites_after < cache.target_Nup)
            continue
        end
        copyto!(measurement, correlation)
        for site in 1:window
            local_spin = (code >> (site - 1)) & 1
            # One occupied and one empty orbital per physical site.
            empty_mode = 2site - local_spin
            measurement[empty_mode, empty_mode] -= 1
        end
        # P(n_A) = (-1)^number_empty det(C_A - diag(1 - n_A)).
        probability = (isodd(window) ? -1 : 1) * real(det(Hermitian(measurement)))
        tolerance = _PROJECTED_SCHUR_PROBABILITY_TOLERANCE
        (-tolerance <= probability <= 1 + tolerance) || throw(DomainError(
            probability, "invalid Gaussian lookahead occupation probability",
        ))
        probabilities[spin + 1] += clamp(probability, 0.0, 1.0)
    end
    sum(probabilities) > 0 || throw(ArgumentError(
        "no positive-probability completion of the projected lookahead window",
    ))
    return probabilities
end

"""
    condition_projected_gaussian!(cache, spin)

Commit a physical-spin choice to a projected-state direct-sampling cache.
"""
function condition_projected_gaussian!(
    cache::ProjectedGaussianSchurCache,
    spin::Integer,
)
    spin in (0, 1) || throw(ArgumentError("spin must be 0 or 1, got $spin"))
    probabilities = projected_conditional_probabilities(cache)
    probabilities[spin + 1] > 0 || throw(ArgumentError(
        "cannot condition on a zero-probability physical spin",
    ))

    correlation = cache.correlation_matrix
    size(correlation, 1) >= 2 || error(
        "the projected Gaussian cache must contain two modes per remaining site",
    )
    if size(correlation, 1) == 2
        cache.correlation_matrix = zeros(ComplexF64, 0, 0)
    else
        # Condition on the complete physical-site event in one block. For an
        # occupied mode the measurement block contains C_aa; for an empty mode
        # it contains C_aa - 1. Its Schur complement gives the correlation
        # matrix of the remaining modes. This avoids the potentially tiny and
        # ill-conditioned intermediate probability produced by two sequential
        # one-mode updates.
        measurement_block = Matrix(@view correlation[1:2, 1:2])
        empty_mode = 2 - spin # spin 0 => down empty; spin 1 => up empty
        measurement_block[empty_mode, empty_mode] -= 1
        cross_column = Matrix(@view correlation[3:end, 1:2])
        cross_row = Matrix(@view correlation[1:2, 3:end])
        remaining = Matrix(@view correlation[3:end, 3:end])
        conditioned = remaining - cross_column * (measurement_block \ cross_row)
        conditioned = (conditioned + adjoint(conditioned)) / 2

        # Remove harmless diagonal roundoff immediately so it cannot accumulate
        # over the remaining physical sites. Larger violations still indicate a
        # genuinely unstable update and are rejected.
        tolerance = _PROJECTED_SCHUR_PROBABILITY_TOLERANCE
        for mode in axes(conditioned, 1)
            density = real(conditioned[mode, mode])
            (-tolerance <= density <= 1 + tolerance) || throw(DomainError(
                density,
                "conditioning produced an invalid parent-Slater density",
            ))
            conditioned[mode, mode] = clamp(density, 0.0, 1.0)
        end
        cache.correlation_matrix = conditioned
    end
    cache.measured_up += spin == 0
    popfirst!(cache.remaining_sites)
    return probabilities[spin + 1]
end

function _projected_gaussian_orbital_data(
    H_BdG_func::Function,
    parameters::AbstractVector{<:Number},
    number_of_modes::Integer,
    gap_tolerance::Real,
)
    iseven(number_of_modes) || throw(DimensionMismatch(
        "a spinful projected Gaussian state requires an even number of modes, " *
        "got $number_of_modes",
    ))
    number_of_sites = number_of_modes ÷ 2
    H_BdG = Matrix(H_BdG_func(parameters, number_of_modes))
    size(H_BdG) == (2number_of_modes, 2number_of_modes) || throw(DimensionMismatch(
        "the Gaussian BdG Hamiltonian must have size " *
        "$(2number_of_modes) × $(2number_of_modes), got $(size(H_BdG))",
    ))

    scale = max(maximum(abs, H_BdG), 1.0)
    pairing_block = @view H_BdG[1:number_of_modes, number_of_modes+1:end]
    maximum(abs, pairing_block) <= gap_tolerance * scale || throw(ArgumentError(
        "Gutzwiller projection currently requires a number-conserving Gaussian " *
        "state with a zero anomalous BdG block",
    ))
    particle_hamiltonian = Hermitian(H_BdG[1:number_of_modes, 1:number_of_modes])
    spectrum = eigen(particle_hamiltonian)
    number_occupied = count(<(-gap_tolerance), spectrum.values)
    number_zero = count(energy -> abs(energy) <= gap_tolerance, spectrum.values)
    number_zero == 0 || throw(ArgumentError(
        "the Gaussian Fermi level contains $number_zero modes within " *
        "gap_tolerance=$gap_tolerance",
    ))
    number_occupied == number_of_sites || throw(ArgumentError(
        "single-occupancy projection requires $number_of_sites occupied " *
        "spin-orbitals, but the Gaussian state contains $number_occupied; " *
        "the fixed chemical potential may have crossed a level",
    ))

    occupied = Matrix{ComplexF64}(spectrum.vectors[:, 1:number_of_sites])
    correlation_matrix = occupied * adjoint(occupied)
    unoccupied = Matrix{ComplexF64}(spectrum.vectors[:, number_of_sites+1:end])
    occupied_energies = Float64.(spectrum.values[1:number_of_sites])
    unoccupied_energies = Float64.(spectrum.values[number_of_sites+1:end])
    return (
        occupied,
        correlation_matrix,
        unoccupied,
        occupied_energies,
        unoccupied_energies,
    )
end

function ParameterizedGutzwillerProjectedState(
    gaussian_state::GaussianState;
    Nup::Union{Nothing,Integer}=nothing,
    gap_tolerance::Real=1e-8,
)
    gap_tolerance >= 0 || throw(ArgumentError(
        "gap_tolerance must be nonnegative, got $gap_tolerance",
    ))
    number_of_sites = gaussian_state.N ÷ 2
    fixed_Nup = isnothing(Nup) ? nothing : Int(Nup)
    if !isnothing(fixed_Nup) && !(0 <= fixed_Nup <= number_of_sites)
        throw(ArgumentError(
            "Nup must lie between 0 and N=$number_of_sites, got $fixed_Nup",
        ))
    end
    all(iszero, gaussian_state.occ_ref) || throw(ArgumentError(
        "Gutzwiller projection currently requires the Bogoliubov ground-state " *
        "vacuum (occ_ref == 0)",
    ))
    parameters = copy(Parameters(gaussian_state))
    orbital_data = _projected_gaussian_orbital_data(
        gaussian_state.H_BdG_func,
        parameters,
        gaussian_state.N,
        gap_tolerance,
    )
    return ParameterizedGutzwillerProjectedState(
        gaussian_state.H_BdG_func,
        parameters,
        gaussian_state.N,
        orbital_data...,
        number_of_sites,
        fixed_Nup,
        Float64(gap_tolerance),
    )
end

Base.eltype(::ParameterizedGutzwillerProjectedState) = ComplexF64
Parameters(state::ParameterizedGutzwillerProjectedState) = state.η

function write!(
    state::ParameterizedGutzwillerProjectedState,
    η::AbstractVector{<:Number},
)
    length(η) == length(state.η) || throw(DimensionMismatch(
        "the projected Gaussian state requires $(length(state.η)) parameters, " *
        "got $(length(η))",
    ))
    state.η = η
    occupied, correlation_matrix, unoccupied, occupied_energies, unoccupied_energies =
        _projected_gaussian_orbital_data(
            state.H_BdG_func,
            state.η,
            state.number_of_modes,
            state.gap_tolerance,
        )
    state.occupied_orbitals = occupied
    state.correlation_matrix = correlation_matrix
    state.unoccupied_orbitals = unoccupied
    state.occupied_energies = occupied_energies
    state.unoccupied_energies = unoccupied_energies
    return state
end

"""
    get_prob(state::ParameterizedGutzwillerProjectedState, spin_prefix)

Return the parent Gaussian state's marginal probability for a partial physical
spin configuration. This is the proposal factor used by the ordinary
sequential PEPS sampler; the complete target amplitude is still evaluated from
the Gutzwiller-projected determinant.

For every measured physical site, `0 => up` is mapped to mode occupations
`(n_up, n_down) = (1, 0)` and `1 => down` to `(0, 1)`. Unmeasured modes are
traced out before the single-occupancy Gutzwiller projection. This polynomial
Gaussian marginal is intentionally available only when `Nup === nothing`.
"""
function get_prob(
    state::ParameterizedGutzwillerProjectedState,
    spin_prefix::Dict{Int,Int},
)
    if !isnothing(state.Nup)
        if length(spin_prefix) == state.N &&
           all(haskey(spin_prefix, site) for site in 1:state.N)
            configuration = [spin_prefix[site] for site in 1:state.N]
            return gutzwiller_weight(state, configuration)
        end
        throw(ArgumentError(
            "pre-projection Gaussian marginals are only available when the " *
            "projected state has Nup=nothing",
        ))
    end
    isempty(spin_prefix) && return 1.0

    sites = sort!(collect(keys(spin_prefix)))
    any(site -> !(1 <= site <= state.N), sites) && throw(BoundsError(
        1:state.N,
        first(site for site in sites if !(1 <= site <= state.N)),
    ))
    measured_modes = Vector{Int}(undef, 2length(sites))
    occupations = Vector{Int}(undef, 2length(sites))
    for (position, site) in enumerate(sites)
        spin = spin_prefix[site]
        (spin == 0 || spin == 1) || throw(DomainError(
            spin,
            "spin entries must be 0 (up) or 1 (down)",
        ))
        measured_modes[2position-1] = 2site - 1
        measured_modes[2position] = 2site
        occupations[2position-1] = 1 - spin
        occupations[2position] = spin
    end

    marginal_matrix = copy(state.correlation_matrix[measured_modes, measured_modes])
    number_empty = 0
    for mode in eachindex(occupations)
        if occupations[mode] == 0
            marginal_matrix[mode, mode] -= 1
            number_empty += 1
        end
    end
    log_probability, determinant_phase = logabsdet(Hermitian(marginal_matrix))
    isfinite(log_probability) || return 0.0
    corrected_phase = (isodd(number_empty) ? -1 : 1) * determinant_phase
    phase_tolerance = 1e-8
    abs(imag(corrected_phase)) <= phase_tolerance || throw(ArgumentError(
        "the Gaussian occupation marginal acquired a non-real determinant phase " *
        "$corrected_phase",
    ))
    real(corrected_phase) >= -phase_tolerance || throw(ArgumentError(
        "the Gaussian occupation marginal became negative with determinant phase " *
        "$corrected_phase",
    ))
    return max(real(corrected_phase), 0.0) * exp(log_probability)
end

"""
    gutzwiller_log_gradient(state, spin_configuration)

Return `∂η log(Ψ_G(spin_configuration))` for a projected Gaussian state. The
occupied-orbital response is evaluated in the parallel-transport gauge. This
fixes the otherwise arbitrary common phase of the Slater determinant; physical
log-amplitude differences and VMC estimators are gauge independent.
"""
function gutzwiller_log_gradient(
    state::ParameterizedGutzwillerProjectedState,
    spin_configuration::AbstractArray{<:Integer},
)
    rows, number_up = _gutzwiller_rows(state, spin_configuration)
    if !isnothing(state.Nup) && number_up != state.Nup
        throw(ArgumentError(
            "spin configuration has Nup=$number_up, but the state requires " *
            "Nup=$(state.Nup)",
        ))
    end
    selected_slater = state.occupied_orbitals[rows, :]
    return _gutzwiller_log_gradient(state, rows, inv(selected_slater))
end

function _gutzwiller_log_gradient(
    state::ParameterizedGutzwillerProjectedState,
    rows::AbstractVector{<:Integer},
    inverse_slater::AbstractMatrix{<:Number},
)
    parameters = Parameters(state)
    isempty(parameters) && return ComplexF64[]
    occupied = state.occupied_orbitals
    unoccupied = state.unoccupied_orbitals
    selected_unoccupied = unoccupied[rows, :]
    response_denominators =
        reshape(state.occupied_energies, 1, :) .-
        reshape(state.unoccupied_energies, :, 1)
    minimum(abs, response_denominators) > state.gap_tolerance || throw(ArgumentError(
        "occupied and unoccupied auxiliary levels are not separated by the " *
        "requested gap tolerance",
    ))

    # If dV = Uu * ((Uu' * dH * V) ./ (εocc' - εunocc)), then
    # dlog(det(V[rows,:])) = tr(X*dH), with the X below.
    selected_response = inverse_slater * selected_unoccupied
    response_weights = Matrix(transpose(selected_response)) ./ response_denominators
    X = occupied * transpose(response_weights) * adjoint(unoccupied)
    number_of_modes = state.number_of_modes

    dHs = build_H_BdG_derivatives(
        state.H_BdG_func,
        parameters,
        number_of_modes,
    )
    return ComplexF64[
        sum(transpose(X) .* @view(dH[1:number_of_modes, 1:number_of_modes]))
        for dH in dHs
    ]
end

function get_Ok(
    state::ParameterizedGutzwillerProjectedState,
    spin_configuration::Matrix{Int64},
    Ok,
)
    isempty(Parameters(state)) && return Ok
    gradient = gutzwiller_log_gradient(state, spin_configuration)
    parameter_offset = size(Ok, 1) - length(gradient)
    Ok[parameter_offset+1:end] .= gradient
    return Ok
end

"""
    gutzwiller_project(gaussian_state::GaussianState; Nup=nothing, gap_tolerance=1e-8)

Apply the one-particle-per-site Gutzwiller projection to a trainable,
number-conserving `2N`-mode Gaussian state while retaining its mean-field
parameters.
"""
gutzwiller_project(gaussian_state::GaussianState; kwargs...) =
    ParameterizedGutzwillerProjectedState(gaussian_state; kwargs...)
