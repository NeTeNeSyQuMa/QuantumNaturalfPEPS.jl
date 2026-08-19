"""
    GutzwillerProjectedState(occupied_orbitals; Nup=nothing)

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
struct GutzwillerProjectedState{T<:Number} <: AbstractTrialState
    occupied_orbitals::Matrix{T}
    N::Int
    Nup::Union{Nothing,Int}
end

function GutzwillerProjectedState(
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
    return GutzwillerProjectedState{T}(orbitals, N, fixed_Nup)
end

"""
    GutzwillerProjectedState(up_orbitals, down_orbitals)

Construct a spin-conserving projected state from occupied spatial orbitals.
`up_orbitals` and `down_orbitals` must have `N` rows and together have `N`
columns. The occupied creation operators are ordered with all up-spin orbitals
before all down-spin orbitals. The resulting state has fixed
`Nup = size(up_orbitals, 2)`.
"""
function GutzwillerProjectedState(
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
    return GutzwillerProjectedState(occupied_orbitals; Nup)
end

Base.eltype(::GutzwillerProjectedState{T}) where {T} = T
Parameters(::GutzwillerProjectedState) = Float64[]

function _gutzwiller_rows(
    state::GutzwillerProjectedState,
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
    state::GutzwillerProjectedState,
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
gutzwiller_weight(state::GutzwillerProjectedState, spin_configuration) =
    abs2(gutzwiller_amplitude(state, spin_configuration))

"""
    GutzwillerExchangeCache(state, spin_configuration)

Cache determinant inverses for fixed-magnetization Monte Carlo updates of a
`GutzwillerProjectedState`. A proposal exchanges two opposite spins and hence
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
    state::GutzwillerProjectedState,
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

get_amplitude(state::GutzwillerProjectedState, spin_configuration::Vector{Int}) =
    gutzwiller_amplitude(state, spin_configuration)

function get_prob(
    state::GutzwillerProjectedState,
    spin_configuration::AbstractVector{<:Integer},
)
    return gutzwiller_weight(state, spin_configuration)
end

function get_prob(
    state::GutzwillerProjectedState,
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

Construct a `GutzwillerProjectedState` either from occupied orbitals, from
separate occupied up/down spatial orbitals, or by filling the lowest `N`
one-particle levels of a `2N × 2N` Hermitian auxiliary Hamiltonian.
"""
gutzwiller_project(occupied_orbitals::AbstractMatrix{<:Number}; kwargs...) =
    GutzwillerProjectedState(occupied_orbitals; kwargs...)

gutzwiller_project(
    up_orbitals::AbstractMatrix{<:Number},
    down_orbitals::AbstractMatrix{<:Number},
) = GutzwillerProjectedState(up_orbitals, down_orbitals)

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
    return GutzwillerProjectedState(spectrum.vectors[:, 1:N]; Nup)
end
