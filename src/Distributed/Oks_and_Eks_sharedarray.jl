function generate_Oks_and_Eks_multiproc_sharedarrays(peps::AbstractPEPS, ham_op::TensorOperatorSum; 
                                                     trial_state::AbstractTrialState=IdentityState(dim(siteinds(peps)[1])),
                                                     timer=TimerOutput(), threaded=true, double_layer_update=update_double_layer_envs!,
                                                     reset_double_layer=true,
                                                     kwargs...)
    n_threads = Distributed.remotecall_fetch(()->Threads.nthreads(), workers()[1])
    function Oks_and_Eks_(Θ::Vector{T}, sample_nr::Integer; kwargs2...) where T
        if length(kwargs2) > 0
            kwargs = merge(kwargs, kwargs2)
        end
        # write!(peps, Θ; reset_double_layer)
        write!(peps, Θ[1:(length(Θ)-length(Parameters(trial_state)))]; reset_double_layer)
        write!(trial_state, Θ[(length(Θ)-length(Parameters(trial_state))+1):end])

        if reset_double_layer
            @timeit timer "double_layer_envs" double_layer_update(peps) # update the double layer environments once for the peps
        end

        return @timeit timer "Oks_and_Eks" Oks_and_Eks_multiproc_sharedarrays(peps, ham_op, sample_nr;
                                                               trial_state=trial_state, timer=timer, n_threads=n_threads, kwargs...)
    end

    function Oks_and_Eks_(peps_::Parameters{<:AbstractPEPS}, sample_nr::Integer; kwargs2...)

        peps_ = peps_.obj
        if getfield(peps_, :double_layer_envs) === nothing
            @timeit timer "double_layer_envs" double_layer_update(peps_)
        end

        if length(kwargs2) > 0
            kwargs = merge(kwargs, kwargs2)
        end
        return @timeit timer "Oks_and_Eks" Oks_and_Eks_multiproc_sharedarrays(peps_, ham_op, sample_nr;
                                                               trial_state=trial_state, timer=timer, n_threads=n_threads, kwargs...)
    end
    return Oks_and_Eks_
end

function Oks_and_Eks_multiproc_sharedarrays(peps, ham_op, sample_nr; trial_state = IdentityState(dim(siteinds(peps)[1])), Oks=nothing, importance_weights=true, 
                               n_threads=Distributed.remotecall_fetch(()->Threads.nthreads(), workers()[1]),
                               timer=TimerOutput(),
                               kwargs...)
    nr_procs = length(workers())
    k = ceil(Int, sample_nr / nr_procs)
    k_thread = ceil(Int, k / n_threads)
    k_eff = k_thread * n_threads
    sample_nr_eff = k_eff * nr_procs
    nr_parameters = length(peps)

    seed = rand(UInt)
    eltype_ = eltype(peps)
    eltype_real = real(eltype_)

    # Allocate a shared Oks array if not provided.
    @assert Oks === nothing || Oks isa SharedArray

    if Oks === nothing
        Oks = SharedArray{eltype_}( (nr_parameters + length(Parameters(trial_state)), sample_nr_eff), pids=workers())
    end

    # TODO: Send ham_op only once through the network
    out = []
    @timeit timer "dispatch jobs" for (i, w) in enumerate(workers())
        i1 = k_eff * (i - 1) + 1
        i2 = k_eff * i
        Oks_ = @view Oks[:, i1:i2]
        task = Distributed.remotecall(
            () -> Oks_and_Eks_threaded(peps, ham_op, k;
                                        trial_state=trial_state, importance_weights=false, seed=seed + w,
                                        nr_threads=n_threads, Oks=Oks_, return_Oks=false, kwargs...),
            w)
        push!(out, task)
    end

    # Prepare arrays to gather outputs from the remote calls.
    samples = Vector{Any}(undef, sample_nr_eff)
    Eks = Vector{eltype_}(undef, sample_nr_eff)
    logψs = Vector{Complex{eltype_real}}(undef, sample_nr_eff)
    logpcs = Vector{eltype_real}(undef, sample_nr_eff)
    contract_dims = Vector{Int}(undef, sample_nr_eff)

    @timeit timer "recieve results" for (i, task) in enumerate(out)
        i1 = k_eff * (i - 1) + 1
        i2 = k_eff * i
        out_dict = fetch(task)
        Eks[i1:i2]           = out_dict[:Eks]
        logψs[i1:i2]         = out_dict[:logψs]
        samples[i1:i2]       = out_dict[:samples]
        logpcs[i1:i2]        = out_dict[:weights]
        contract_dims[i1:i2] = out_dict[:contract_dims]
        # No need to copy Oks – it is updated via the SharedArray.
    end
    Oks = sdata(Oks)

    if importance_weights
        weights = compute_importance_weights(logψs, logpcs)
    else
        weights = logpcs
    end
    @everywhere GC.gc() # Force garbage collection of the shared arrays on the remote workers.
    return Dict(:Oks => transpose(Oks), :Eks => Eks, :logψs => logψs,
                :samples => samples, :weights => weights, :contract_dims => contract_dims)
end

###### Trial-state-only variant: optimizes the trial state without a PEPS
function generate_Oks_and_Eks_multiproc_sharedarrays(H_BdG_exact::Hermitian, H_BdG_func::Function, N::Int;
                                                     trial_state_type::Type{<:AbstractTrialState}=GaussianState,
                                                     parity_sector::Int=0, target_state::Int=0,
                                                     timer=TimerOutput(), kwargs...)
    if trial_state_type == GaussianState
        n_threads = Distributed.remotecall_fetch(()->Threads.nthreads(), workers()[1])

        function Oks_and_Eks_(η::Vector{T}, sample_nr::Integer; kwargs2...) where T
            if length(kwargs2) > 0
                kwargs = merge(kwargs, kwargs2)
            end
            # create the trial state from η
            GS = GaussianState(H_BdG_func, N; η=η, parity_sector=parity_sector, target_state=target_state)
            return @timeit timer "Oks_and_Eks" Oks_and_Eks_multiproc_sharedarrays(GS, H_BdG_exact, sample_nr; timer, n_threads, kwargs...)
        end
        return Oks_and_Eks_
    end
    # add more trial states here
    error("Trial state type $(trial_state_type) not implemented for trial-state-only optimization.")
end

function Oks_and_Eks_multiproc_sharedarrays(GS::GaussianState, H_BdG_exact::Hermitian, sample_nr::Integer;
                                            Oks=nothing, importance_weights=true,
                                            n_threads=Distributed.remotecall_fetch(()->Threads.nthreads(), workers()[1]),
                                            timer=TimerOutput(),
                                            kwargs...)
    nr_procs = length(workers())
    k = ceil(Int, sample_nr / nr_procs)
    k_thread = ceil(Int, k / n_threads)
    k_eff = k_thread * n_threads
    sample_nr_eff = k_eff * nr_procs
    nr_parameters = length(GS.η)

    seed = rand(UInt)
    eltype_ = ComplexF64
    eltype_real = real(eltype_)

    # Allocate a shared Oks array if not provided.
    @assert Oks === nothing || Oks isa SharedArray

    if Oks === nothing
        Oks = SharedArray{eltype_}((nr_parameters, sample_nr_eff), pids=workers())
    end

    out = []
    @timeit timer "dispatch jobs" for (i, w) in enumerate(workers())
        i1 = k_eff * (i - 1) + 1
        i2 = k_eff * i
        Oks_ = @view Oks[:, i1:i2]
        task = Distributed.remotecall(
            () -> Oks_and_Eks_threaded(GS, H_BdG_exact, k;
                                       importance_weights=false, seed=seed + w,
                                       nr_threads=n_threads, Oks=Oks_, return_Oks=false, kwargs...),
            w)
        push!(out, task)
    end

    # Prepare arrays to gather outputs from the remote calls.
    samples = Vector{Any}(undef, sample_nr_eff)
    Eks = Vector{eltype_}(undef, sample_nr_eff)
    logψs = Vector{Complex{eltype_real}}(undef, sample_nr_eff)
    logpcs = Vector{eltype_real}(undef, sample_nr_eff)
    contract_dims = Vector{Int}(undef, sample_nr_eff)

    @timeit timer "recieve results" for (i, task) in enumerate(out)
        i1 = k_eff * (i - 1) + 1
        i2 = k_eff * i
        out_dict = fetch(task)
        Eks[i1:i2]           = out_dict[:Eks]
        logψs[i1:i2]         = out_dict[:logψs]
        samples[i1:i2]       = out_dict[:samples]
        logpcs[i1:i2]        = out_dict[:weights]
        contract_dims[i1:i2] = out_dict[:contract_dims]
        # No need to copy Oks – it is updated via the SharedArray.
    end
    Oks = sdata(Oks)

    if importance_weights
        weights = compute_importance_weights(logψs, logpcs)
    else
        weights = logpcs
    end
    @everywhere GC.gc() # Force garbage collection of the shared arrays on the remote workers.
    return Dict(:Oks => transpose(Oks), :Eks => Eks, :logψs => logψs,
                :samples => samples, :weights => weights, :contract_dims => contract_dims)
end