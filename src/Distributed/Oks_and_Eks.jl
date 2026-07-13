
function compute_importance_weights(logψs, logpcs)
    log_ratios =  2 .* real.(logψs) .- logpcs
    logZ = logsumexp(log_ratios) - log(length(logpcs))

    return exp.(log_ratios .- logZ)
end

function generate_Oks_and_Eks(peps::AbstractPEPS, ham::OpSum; kwargs...)
    hilbert = siteinds(peps)
    ham_op = TensorOperatorSum(ham, hilbert)
    return generate_Oks_and_Eks(peps, ham_op; kwargs...)
end

function generate_Oks_and_Eks(peps::AbstractPEPS, ham_op::TensorOperatorSum; trial_state::AbstractTrialState=IdentityState(dim(siteinds(peps)[1])),
                              threaded=false, multiproc=false, shared_array=true, async_double_layers=false, verbose=false,
                              fix_trial_state=false, fix_peps=false,
                              kwargs...)
    if fix_peps && fix_trial_state
        error("fix_peps=true and fix_trial_state=true leaves nothing to optimize")
    end

    if fix_trial_state
        # freeze the trial state: Θ then only contains the PEPS parameters and only those are updated
        trial_state = FrozenTrialState(trial_state)
    end

    if fix_peps
        # freeze the PEPS: with an all-zero mask it still contributes its amplitudes but exposes no
        # variational parameters (length(peps) == 0, write! and get_Ok skip every tensor), so Θ then
        # only contains the trial-state parameters. Work on a copy so the caller's PEPS keeps its mask.
        peps = deepcopy(peps)
        peps.mask = zeros(size(peps.mask))
    end

    local double_layer_update, stop_thread
    if async_double_layers
        double_layer_update, stop_thread = generate_async_double_layer_envs(peps; verbose)
    else
        double_layer_update = update_double_layer_envs!
    end

    local Oks_and_Eks_func
    
    if multiproc
        if shared_array
            Oks_and_Eks_func = generate_Oks_and_Eks_multiproc_sharedarrays(peps, ham_op; trial_state=trial_state, threaded, double_layer_update, kwargs...)
        else
            Oks_and_Eks_func = generate_Oks_and_Eks_multiproc(peps, ham_op; trial_state=trial_state, threaded, double_layer_update, kwargs...)
        end
    elseif threaded
        Oks_and_Eks_func = generate_Oks_and_Eks_threaded(peps, ham_op; trial_state=trial_state, double_layer_update, kwargs...)
    else
        Oks_and_Eks_func = generate_Oks_and_Eks_singlethread(peps, ham_op; trial_state=trial_state, double_layer_update, kwargs...)
    end

    if async_double_layers
        return Oks_and_Eks_func, stop_thread
    end

    return Oks_and_Eks_func
end

# Trial-state-only optimization (no PEPS): same dispatcher interface as the joint PEPS version,
# routing to the singlethread/threaded/multiproc/sharedarray Slater variants.
function generate_Oks_and_Eks(H_BdG_exact::Hermitian, H_BdG_func::Function, N::Int;
                            trial_state_type::Type{<:AbstractTrialState}=GaussianState,
                            threaded=false, multiproc=false, shared_array=true,
                            kwargs...)
    if multiproc
        if shared_array
            return generate_Oks_and_Eks_multiproc_sharedarrays(H_BdG_exact, H_BdG_func, N; trial_state_type, kwargs...)
        else
            return generate_Oks_and_Eks_multiproc(H_BdG_exact, H_BdG_func, N; trial_state_type, kwargs...)
        end
    elseif threaded
        return generate_Oks_and_Eks_threaded(H_BdG_exact, H_BdG_func, N; trial_state_type, kwargs...)
    end

    if trial_state_type == GaussianState
        return generate_Oks_and_Eks_singlethread(H_BdG_exact, H_BdG_func, N; trial_state_type, kwargs...)
    end
    # add more trial states here

    error("Trial state type $(trial_state_type) not implemented for trial-state-only optimization.")
end


###### Single threaded
# this function returns a Ok_and_Eks function wich can be used to optimise via QNG.evolve
function generate_Oks_and_Eks_singlethread(peps::AbstractPEPS, ham_op::TensorOperatorSum;
                                           trial_state::AbstractTrialState=IdentityState(dim(siteinds(peps)[1])),
                                           timer=TimerOutput(), double_layer_update=update_double_layer_envs!,
                                           kwargs...)
    function Oks_and_Eks_(Θ::Vector{T}, sample_nr::Integer; kwargs2...) where T
        if length(kwargs2) > 0
            kwargs_new = Dict{Symbol,Any}() # Fix of bug in julias merge function
            kwargs = merge(kwargs_new, kwargs, kwargs2)
        end
        write!(peps, Θ[1:(length(Θ)-length(Parameters(trial_state)))])
        write!(trial_state, Θ[(length(Θ)-length(Parameters(trial_state))+1):end])

        @timeit timer "double_layer_envs" double_layer_update(peps) # update the double layer environments once for the peps 
        
        return @timeit timer "Oks_and_Eks" Oks_and_Eks_singlethread(peps, ham_op, sample_nr; trial_state=trial_state, timer=timer, kwargs...)
    end

    # TODO: How to combine with trial state?
    function Oks_and_Eks_(peps_::Parameters{<:AbstractPEPS}, sample_nr::Integer; kwargs2...)
        peps_ = peps_.obj
        if getfield(peps_, :double_layer_envs) === nothing
            @timeit timer "double_layer_envs" double_layer_update(peps_)
        end

        if length(kwargs2) > 0
            kwargs = merge(kwargs, kwargs2)
        end
        return @timeit timer "Oks_and_Eks" Oks_and_Eks_singlethread(peps_, ham_op, sample_nr; trial_state=IdentityState(dim(siteinds(peps)[1])), timer=timer, kwargs...)
    end

    return Oks_and_Eks_
end

# this function returns a Ok_and_Eks function for the trial_state only wich can be used to optimise via QNG.evolve
function generate_Oks_and_Eks_singlethread(H_BdG_exact::Hermitian, H_BdG_func::Function, N::Int;
                                           trial_state_type::Type{<:AbstractTrialState}=GaussianState,
                                           parity_sector::Int=0, target_state::Int=0,
                                           timer=TimerOutput(), kwargs...)
    @assert parity_sector == 0 || parity_sector == 1 "Parity sector must be either 0 (even) or 1 (odd)"

    if trial_state_type == GaussianState
        function Oks_and_Eks_(η::Vector{T}, sample_nr::Integer; kwargs2...) where T
            if length(kwargs2) > 0
                kwargs = merge(kwargs, kwargs2)
            end
            # create the trial state from η
            GS = GaussianState(H_BdG_func, N; η=η, parity_sector=parity_sector, target_state=target_state)
            return @timeit timer "Oks_and_Eks" Oks_and_Eks_singlethread_Slater(GS, H_BdG_exact, sample_nr; timer=timer, kwargs...)
        end
        return Oks_and_Eks_
    end

    # add more trial states here
    error("Trial state type $(trial_state_type) not implemented for trial-state-only optimization.")
end

# The central function is Oks and Eks
function Oks_and_Eks_singlethread(peps::AbstractPEPS, ham_op::TensorOperatorSum, sample_nr::Integer; trial_state::AbstractTrialState=IdentityState(dim(siteinds(peps)[1])), timer=TimerOutput(), kwargs...)
    eltype_ = eltype(peps)
    eltype_real = real(eltype_)
    
    Oks = Matrix{eltype_}(undef, length(peps)+length(Parameters(trial_state)), sample_nr)
    Eks = Vector{eltype_}(undef, sample_nr)
    logψs = Vector{Complex{eltype_real}}(undef, sample_nr)
    samples = Vector{Matrix{Int}}(undef, sample_nr)
    logpc = Vector{eltype_real}(undef, sample_nr)
    contract_dims = Vector{Int}(undef, sample_nr)

    for i in 1:sample_nr
        Ok_view = @view Oks[:, i]
        _, Eks[i], logψs[i], samples[i], logpc[i], contract_dims[i] = Ok_and_Ek(peps, ham_op; trial_state=trial_state, timer, Ok=Ok_view, kwargs...)
    end
    
    #return Ok, E_loc, logψ, samples, compute_importance_weights(logψ, logpc)
    Dict(:Oks => transpose(Oks), :Eks => Eks, :logψs => logψs, :samples => samples, :weights => compute_importance_weights(logψs, logpc), :contract_dims => contract_dims)
    # returns Gradient, local Energy, log(<ψ|S>), samples S, p
end

#= 
    Add more trial states here
=#

# The central function is Oks and Eks
function Oks_and_Eks_singlethread_Slater(GS::GaussianState, H_BdG_exact::Hermitian, sample_nr::Integer; timer=TimerOutput(), kwargs...)
    eltype_ = ComplexF64
    eltype_real = real(eltype_)

    amp_cache = @timeit timer "amp_cache_base" build_amplitude_cache(GS)
    # slater_loggrad_cache = @timeit timer "slater_loggrad_cache" build_slater_loggradient_cache(GS)
    SlaterConnections = @timeit timer "SlaterConnections" get_Slater_Ek_terms(H_BdG_exact)
    
    Oks = Matrix{eltype_}(undef, length(GS.η), sample_nr)
    Eks = Vector{eltype_}(undef, sample_nr)
    logψs = Vector{Complex{eltype_real}}(undef, sample_nr)
    samples = Vector{Matrix{Int}}(undef, sample_nr)
    logpc = Vector{eltype_real}(undef, sample_nr)
    contract_dims = Vector{Int}(undef, sample_nr)

    for i in 1:sample_nr
        Ok_view = @view Oks[:, i]
        _, Eks[i], logψs[i], samples[i], logpc[i], contract_dims[i] = Ok_and_Ek(GS, H_BdG_exact; timer, Ok=Ok_view, amp_cache, SlaterConnections, kwargs...)
    end
    
    #return Ok, E_loc, logψ, samples, compute_importance_weights(logψ, logpc)
    Dict(:Oks => transpose(Oks), :Eks => Eks, :logψs => logψs, :samples => samples, :weights => compute_importance_weights(logψs, logpc), :contract_dims => contract_dims)
    # returns Gradient, local Energy, log(<ψ|S>), samples S, p
end

include("double_layer_async.jl")
include("Oks_and_Eks_threaded.jl")
include("Oks_and_Eks_multiproc.jl")
include("Oks_and_Eks_sharedarray.jl")