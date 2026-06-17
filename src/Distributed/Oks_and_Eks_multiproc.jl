function generate_Oks_and_Eks_multiproc(peps::AbstractPEPS, ham_op::TensorOperatorSum;
                                        trial_state::AbstractTrialState=IdentityState(dim(siteinds(peps)[1])), 
                                        timer=TimerOutput(), threaded=true, double_layer_update=update_double_layer_envs!,
                                        kwargs...)
    n_threads = Distributed.remotecall_fetch(()->Threads.nthreads(), workers()[1])

    function Oks_and_Eks_(Θ::Vector{T}, sample_nr::Integer; kwargs2...) where T
        if length(kwargs2) > 0
            kwargs = merge(kwargs, kwargs2)
        end
        # write!(peps, Θ)
        write!(peps, Θ[1:(length(Θ)-length(Parameters(trial_state)))])
        write!(trial_state, Θ[(length(Θ)-length(Parameters(trial_state))+1):end])
        @timeit timer "double_layer_envs" double_layer_update(peps) # update the double layer environments once for the peps
        return @timeit timer "Oks_and_Eks" Oks_and_Eks_multiproc(peps, ham_op, sample_nr; trial_state=trial_state, timer, n_threads, kwargs...)
    end

    function Oks_and_Eks_(peps_::Parameters{<:AbstractPEPS}, sample_nr::Integer; kwargs2...)
        peps_ = peps_.obj
        if getfield(peps_, :double_layer_envs) === nothing
            @timeit timer "double_layer_envs" double_layer_update(peps_)
        end

        if length(kwargs2) > 0
            kwargs = merge(kwargs, kwargs2)
        end
        return @timeit timer "Oks_and_Eks" Oks_and_Eks_multiproc(peps_, ham_op, sample_nr; trial_state=trial_state, timer, n_threads, kwargs...)
    end
    return Oks_and_Eks_
end

function Oks_and_Eks_multiproc(peps, ham_op, sample_nr; trial_state = trial_state, Oks=nothing, importance_weights=true, 
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
    # TODO: Send ham_op only once through the network
    out = [Distributed.remotecall(() -> Oks_and_Eks_threaded(peps, ham_op, k; trial_state=trial_state, importance_weights=false, seed=seed + w, kwargs...), w) for w in workers()]
    
    eltype_ = eltype(peps)
    eltype_real = real(eltype_)
    
    samples = Vector{Any}(undef, sample_nr_eff)
    Eks = Vector{eltype_}(undef, sample_nr_eff)
    logψs = Vector{Complex{eltype_real}}(undef, sample_nr_eff)
    logpcs = Vector{eltype_real}(undef, sample_nr_eff)
    contract_dims = Vector{Int}(undef, sample_nr_eff)
    
    if Oks === nothing
        Oks = Matrix{eltype_}(undef, nr_parameters + length(Parameters(trial_state)), sample_nr_eff)
    end

    for (i, out_i) in collect(enumerate(out))
        i1 = k_eff * (i - 1) + 1
        i2 = k_eff * i
        out_dict = fetch(out_i)
        Eks[i1:i2], logψs[i1:i2], samples[i1:i2], logpcs[i1:i2], contract_dims[i1:i2] = out_dict[:Eks], out_dict[:logψs], out_dict[:samples], out_dict[:weights], out_dict[:contract_dims]
        @timeit timer "copy Oks" Oks[:, i1:i2] .= transpose(out_dict[:Oks])
    end
    
    if importance_weights
        weights = compute_importance_weights(logψs, logpcs)
    else
        weights = logpcs
    end
    
    return Dict(:Oks => transpose(Oks), :Eks => Eks, :logψs => logψs, :samples => samples, :weights => weights, :contract_dims => contract_dims)
end