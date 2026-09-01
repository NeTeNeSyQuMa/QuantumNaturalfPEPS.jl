"""
    GutzwillerProjectedGaussianState(gaussian_state; Nup=nothing, gap_tolerance=1e-8)

Represent the single-occupancy Gutzwiller projection of a number-conserving,
half-filled [`GaussianState`](@ref) on `2N` interleaved spin-orbitals. The
projected state copies the Gaussian state's mean-field parameters and reuses
its auxiliary-Hamiltonian function. It caches the occupied-orbital correlation
matrix needed for parent-Gaussian sequential proposal probabilities, and
refreshes that cache when `write!(state, η)` is called.

The Gaussian state must have no anomalous BdG pairing block and must contain
exactly `N` occupied physical orbitals separated from the unoccupied orbitals
by `gap_tolerance`. Spin configurations use `0 => ↑` and `1 => ↓`. Supplying
`Nup` additionally projects into a fixed-magnetization sector.
"""
mutable struct GutzwillerProjectedGaussianState <: AbstractGutzwillerProjectedState
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

function GutzwillerProjectedGaussianState(
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
    return GutzwillerProjectedGaussianState(
        gaussian_state.H_BdG_func,
        parameters,
        gaussian_state.N,
        orbital_data...,
        number_of_sites,
        fixed_Nup,
        Float64(gap_tolerance),
    )
end

Base.eltype(::GutzwillerProjectedGaussianState) = ComplexF64
Parameters(state::GutzwillerProjectedGaussianState) = state.η

function write!(
    state::GutzwillerProjectedGaussianState,
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
    get_prob(state::GutzwillerProjectedGaussianState, spin_prefix)

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
    state::GutzwillerProjectedGaussianState,
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
    state::GutzwillerProjectedGaussianState,
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
    state::GutzwillerProjectedGaussianState,
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
    state::GutzwillerProjectedGaussianState,
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
    state::GutzwillerProjectedGaussianState,
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
    GutzwillerProjectedGaussianState(gaussian_state; kwargs...)
