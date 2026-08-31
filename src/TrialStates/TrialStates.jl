abstract type AbstractTrialState end

# Defaults to Bool for IdentityState so it is neutral under `promote_type` (i.e. a real-amplitude trial state never widens the PEPS eltype).
Base.eltype(::AbstractTrialState) = Bool

# Default Trial state is just the identity, which does not change the PEPS optimization
struct IdentityState <: AbstractTrialState
    phys_dim::Int
end
write!(Id::IdentityState, x) = nothing # writing to the identity state does nothing

#= 
    The IdentityState should leave the optimization unchanged so its just a pure PEPS optimization without trial state
=#
function get_prob(Id::IdentityState, occ_dict::Dict{Int, Int})
    return 1
end
function get_amplitude(Id::IdentityState, occ_string::Vector{Int})
    return 1
end

# identity state has no variational parameters
Parameters(Id::IdentityState) = []

"""
    FrozenTrialState

Wraps a trial state so that it still contributes its amplitudes/probabilities to the joint
wavefunction, but exposes no variational parameters. The variational parameter vector Θ then
only contains the PEPS parameters and the trial state stays fixed during optimization.
Used via the `fix_trial_state=true` keyword of `generate_Oks_and_Eks`.
"""
struct FrozenTrialState{T<:AbstractTrialState} <: AbstractTrialState
    state::T
end
write!(f::FrozenTrialState, x) = nothing # frozen: parameters are never updated
get_prob(f::FrozenTrialState, args...) = get_prob(f.state, args...)
get_amplitude(f::FrozenTrialState, occ_string::Vector{Int}) = get_amplitude(f.state, occ_string)
Parameters(f::FrozenTrialState) = []
Base.eltype(f::FrozenTrialState) = eltype(f.state)

include("GaussianState.jl")