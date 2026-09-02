"""
    FixedGutzwillerProjectedState(occupied_orbitals; Nup=nothing)

Represent the single-occupancy Gutzwiller projection of a half-filled Slater
determinant. `occupied_orbitals` must be a `2N × N` matrix whose columns are the
occupied spin orbitals in the interleaved one-particle basis
`(1↑, 1↓, 2↑, 2↓, ...)`.

If `Nup` is specified, amplitudes outside that fixed-magnetization sector are
set to zero. Here spin configurations use the PEPS convention `0 => ↑` and
`1 => ↓`, so `Nup` is the number of zeros in a configuration.

The projected amplitudes are not normalized; their overall normalization is
irrelevant for variational Monte Carlo and amplitude ratios.
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

"""
    GutzwillerExchangeCache(state, spin_configuration)

Cache determinant inverses for fixed-magnetization Monte Carlo updates of a
`FixedGutzwillerProjectedState`. A proposal exchanges two opposite spins and hence
replaces two rows of the full site-ordered Slater matrix. Ratios are evaluated
from a `2 x 2` determinant and accepted moves update the inverse with the
Woodbury identity.
"""
mutable struct GutzwillerExchangeCache
    orbitals::Matrix{ComplexF64}
    configuration::Vector{Int}
    selected_rows::Vector{Int}
    inverse_slater::Matrix{ComplexF64}
    orbital_action::Matrix{ComplexF64}
    log_amplitude::ComplexF64
    accepted_since_rebuild::Int
end

function _gutzwiller_logdet(slater::AbstractMatrix)
    logabs, phase = logabsdet(slater)
    isfinite(logabs) || throw(SingularException(0))
    return ComplexF64(logabs) + log(ComplexF64(phase))
end

function GutzwillerExchangeCache(
    state::AbstractGutzwillerProjectedState,
    spin_configuration::AbstractVector{<:Integer},
)
    rows, number_up = _gutzwiller_rows(state, spin_configuration)
    if !isnothing(state.Nup) && number_up != state.Nup
        throw(ArgumentError(
            "spin configuration has Nup=$number_up, but the state requires Nup=$(state.Nup)",
        ))
    end

    orbitals = Matrix{ComplexF64}(state.occupied_orbitals)
    slater = orbitals[rows, :]
    inverse_slater = inv(slater)
    return GutzwillerExchangeCache(
        orbitals,
        collect(Int, spin_configuration),
        rows,
        inverse_slater,
        orbitals * inverse_slater,
        _gutzwiller_logdet(slater),
        0,
    )
end

function _rebuild_gutzwiller_exchange_cache!(cache::GutzwillerExchangeCache)
    slater = cache.orbitals[cache.selected_rows, :]
    cache.inverse_slater .= inv(slater)
    cache.orbital_action .= cache.orbitals * cache.inverse_slater
    cache.log_amplitude = _gutzwiller_logdet(slater)
    cache.accepted_since_rebuild = 0
    return cache
end

function _gutzwiller_exchange_rows(cache::GutzwillerExchangeCache, i::Int, j::Int)
    N = length(cache.configuration)
    1 <= i <= N || throw(BoundsError(cache.configuration, i))
    1 <= j <= N || throw(BoundsError(cache.configuration, j))
    i != j || throw(ArgumentError("the two exchange sites must be different"))
    cache.configuration[i] != cache.configuration[j] || throw(ArgumentError(
        "a fixed-magnetization exchange requires opposite spins",
    ))
    return (
        2i - 1 + (1 - cache.configuration[i]),
        2j - 1 + (1 - cache.configuration[j]),
    )
end

"""
    gutzwiller_exchange_ratio(cache, i, j)

Return `Psi(swapped)/Psi(current)` when the opposite spins on sites `i` and
`j` are exchanged. Site labels follow Julia column-major lattice ordering.
"""
function gutzwiller_exchange_ratio(cache::GutzwillerExchangeCache, i::Int, j::Int)
    alternative_i, alternative_j = _gutzwiller_exchange_rows(cache, i, j)
    action = cache.orbital_action
    return action[alternative_i, i] * action[alternative_j, j] -
           action[alternative_i, j] * action[alternative_j, i]
end

"""
    accept_gutzwiller_exchange!(cache, i, j; rebuild_after=64)

Update a cache after accepting the exchange of the opposite spins at `i` and
`j`. The inverse is rebuilt periodically to control roundoff accumulation.
"""
function accept_gutzwiller_exchange!(
    cache::GutzwillerExchangeCache,
    i::Int,
    j::Int;
    rebuild_after::Integer=64,
)
    rebuild_after > 0 || throw(ArgumentError("rebuild_after must be positive"))
    amplitude_ratio = gutzwiller_exchange_ratio(cache, i, j)
    iszero(amplitude_ratio) && throw(ArgumentError(
        "cannot accept an exchange with a zero wavefunction-amplitude ratio",
    ))
    alternative_rows = collect(_gutzwiller_exchange_rows(cache, i, j))
    sites = [i, j]
    update_matrix = cache.orbital_action[alternative_rows, sites]
    old_rows = cache.selected_rows[sites]
    row_changes = cache.orbitals[alternative_rows, :] - cache.orbitals[old_rows, :]
    changes_times_inverse = row_changes * cache.inverse_slater
    solved_update = update_matrix \ changes_times_inverse

    cache.inverse_slater .-=
        cache.inverse_slater[:, sites] * solved_update
    cache.orbital_action .-=
        cache.orbital_action[:, sites] * solved_update
    cache.selected_rows[sites] .= alternative_rows
    cache.configuration[i] = 1 - cache.configuration[i]
    cache.configuration[j] = 1 - cache.configuration[j]
    cache.log_amplitude += log(amplitude_ratio)
    cache.accepted_since_rebuild += 1

    if cache.accepted_since_rebuild >= rebuild_after
        _rebuild_gutzwiller_exchange_cache!(cache)
    end
    return cache
end

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
            "available from the sequential PEPS sampler; use fixed-sector " *
            "Metropolis sampling",
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

function gutzwiller_log_gradient(
    state::ParameterizedGutzwillerProjectedState,
    cache::GutzwillerExchangeCache,
)
    length(cache.configuration) == state.N || throw(DimensionMismatch(
        "the exchange cache contains $(length(cache.configuration)) sites, " *
        "but the projected Gaussian state contains $(state.N)",
    ))
    return _gutzwiller_log_gradient(
        state,
        cache.selected_rows,
        cache.inverse_slater,
    )
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

    function contracted_hamiltonian(η)
        H_BdG = Matrix(state.H_BdG_func(η, number_of_modes))
        particle_block = @view H_BdG[1:number_of_modes, 1:number_of_modes]
        return sum(transpose(X) .* particle_block)
    end

    real_gradient = Zygote.gradient(
        η -> real(contracted_hamiltonian(η)),
        parameters,
    )[1]
    imaginary_gradient = Zygote.gradient(
        η -> imag(contracted_hamiltonian(η)),
        parameters,
    )[1]
    return ComplexF64.(real_gradient .+ im .* imaginary_gradient)
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
