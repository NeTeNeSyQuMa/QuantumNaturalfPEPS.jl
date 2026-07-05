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

include("GaussianState.jl")