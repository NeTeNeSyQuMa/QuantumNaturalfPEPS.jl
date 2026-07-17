# Generated from the Oks_and_Eks using chatgpt

################################################################################
# Single entry point for generation of local energies
################################################################################

function generate_Eks(peps::AbstractPEPS, ham::OpSum; kwargs...)
    hilbert = siteinds(peps)
    ham_op = TensorOperatorSum(ham, hilbert)
    return generate_Eks(peps, ham_op; kwargs...)
end

function generate_Eks(peps::AbstractPEPS, ham_op::TensorOperatorSum; threaded=false, multiproc=false, kwargs...)
    if multiproc
        return generate_Eks_multiproc(peps, ham_op; threaded, kwargs...)
    elseif threaded
        return generate_Eks_threaded(peps, ham_op; kwargs...)
    else
        return generate_Eks_singlethread(peps, ham_op; kwargs...)
    end
end

################################################################################
# Single-threaded version
################################################################################

function generate_Eks_singlethread(peps::AbstractPEPS, ham_op::TensorOperatorSum; timer=TimerOutput(), kwargs...)
    function Eks_(Θ::Vector{T}, sample_nr::Integer; kwargs2...) where T
        # Merge any new keyword arguments
        if length(kwargs2) > 0
            kwargs_new = Dict{Symbol,Any}()
            kwargs_merged = merge(kwargs_new, kwargs, kwargs2)
            kwargs = kwargs_merged
        end
        # Write the parameters into the PEPS
        no_write = false
        if haskey(kwargs, :no_write)
            no_write = pop!(kwargs, :no_write)
        end

        if !no_write
            write!(peps, Θ)
        end
        # Update the double-layer environment once per call
        @timeit timer "double_layer_envs" update_double_layer_envs!(peps)
        
        return Eks_singlethread(peps, ham_op, sample_nr; timer=timer, kwargs...)
    end
    return Eks_
end

function Eks_singlethread(peps::AbstractPEPS, ham_op::TensorOperatorSum, sample_nr::Integer; 
                          timer=TimerOutput(), kwargs...)
    eltype_ = eltype(peps)
    eltype_real = real(eltype_)

    Eks         = Vector{eltype_}(undef, sample_nr)
    logψs       = Vector{Complex{eltype_real}}(undef, sample_nr)
    samples     = Vector{Matrix{Int}}(undef, sample_nr)
    logpc       = Vector{eltype_real}(undef, sample_nr)
    max_bond    = Vector{Int}(undef, sample_nr)

    for i in 1:sample_nr
        Eks[i], logψs[i], samples[i], logpc[i], max_bond[i] = Ek(peps, ham_op; timer=timer, kwargs...)
    end

    return Dict(
        :Eks => Eks,
        :logψs => logψs,
        :samples => samples,
        :weights => compute_importance_weights(logψs, logpc),
        :max_bond => max_bond
    )
end

################################################################################
# Multi-threaded version
################################################################################

function generate_Eks_threaded(peps::AbstractPEPS, ham_op::TensorOperatorSum; timer=TimerOutput(), kwargs...)
    function Eks_(Θ::Vector{T}, sample_nr::Integer; reset_double_layer=true, kwargs2...) where T
        if length(kwargs2) > 0
            kwargs_merged = merge(kwargs, kwargs2)
            kwargs = kwargs_merged
        end
        kwargs = Dict(kwargs)
        no_write = false
        if haskey(kwargs, :no_write)
            no_write = pop!(kwargs, :no_write)
        end

        if !no_write
            write!(peps, Θ; reset_double_layer)
        end
        
        if reset_double_layer
            @timeit timer "double_layer_envs" update_double_layer_envs!(peps)
        end
        return Eks_threaded(peps, ham_op, sample_nr; timer=timer, kwargs...)
    end
    return Eks_
end

function Eks_threaded(peps, ham_op, sample_nr; importance_weights=true, seed=nothing,
                      timer=TimerOutput(), nr_threads=Threads.nthreads(), kwargs...)
    
    if seed !== nothing
        Random.seed!(seed)
    end

    nr_parameters = length(peps)
    k = ceil(Int, sample_nr / nr_threads)
    
    eltype_ = eltype(peps)
    eltype_real = real(eltype_)

    # Prepare containers
    Eks         = Matrix{eltype_}(undef, k, nr_threads)
    logψs       = Matrix{Complex{eltype_real}}(undef, k, nr_threads)
    samples     = Matrix{Any}(undef, k, nr_threads)
    logpcs      = Matrix{eltype_real}(undef, k, nr_threads)
    max_bonds   = Matrix{Int}(undef, k, nr_threads)

    seed = rand(UInt)
    Threads.@threads for i in 1:nr_threads
        Random.seed!(seed + i)
        for j in 1:k
            Eks[j, i], logψs[j, i], samples[j, i], logpcs[j, i], max_bonds[j, i] = Ek(peps, ham_op; kwargs...)
        end
    end

    # Reshape the results to a single dimension
    Eks       = reshape(Eks, :)
    logψs     = reshape(logψs, :)
    samples   = reshape(samples, :)
    logpcs    = reshape(logpcs, :)
    max_bonds = reshape(max_bonds, :)

    # Compute importance weights if desired
    if importance_weights
        weights = compute_importance_weights(logψs, logpcs)
    else
        weights = logpcs
    end

    return Dict(
        :Eks => Eks,
        :logψs => logψs,
        :samples => samples,
        :weights => weights,
        :max_bond => max_bonds
    )
end

################################################################################
# Multi-processing version
################################################################################

function generate_Eks_multiproc(peps::AbstractPEPS, ham_op::TensorOperatorSum; timer=TimerOutput(), threaded=true, kwargs...)
    function Eks_(Θ::Vector{T}, sample_nr::Integer; kwargs2...) where T
        if length(kwargs2) > 0
            kwargs_merged = merge(kwargs, kwargs2)
            kwargs = kwargs_merged
        end

        @timeit timer "double_layer_envs" update_double_layer_envs!(peps)        
        return @timeit timer "Eks_multiproc" Eks_multiproc(peps, ham_op, sample_nr; timer, kwargs...)
    end
    function Eks_(peps_::Parameters{<:AbstractPEPS}, sample_nr::Integer; kwargs2...)
        peps_ = peps_.obj
        if getfield(peps_, :double_layer_envs) === nothing
            update_double_layer_envs!(peps)
        end
        if length(kwargs2) > 0
            kwargs_merged = merge(kwargs, kwargs2)
            kwargs = kwargs_merged
        end
        return @timeit timer "Eks_multiproc" Eks_multiproc(peps_, ham_op, sample_nr; timer, kwargs...)
    end

    return Eks_
end

function Eks_multiproc(peps, ham_op, sample_nr;
                       importance_weights=true,
                       n_threads=Distributed.remotecall_fetch(() -> Threads.nthreads(), workers()[1]),
                       timer=TimerOutput(), kwargs...)

    nr_procs = length(workers())
    k = ceil(Int, sample_nr / nr_procs)
    k_thread = ceil(Int, k / n_threads)
    k_eff = k_thread * n_threads
    sample_nr_eff = k_eff * nr_procs

    eltype_ = eltype(peps)
    eltype_real = real(eltype_)

    # Distribute work among processes
    # Each remote call returns a Dict with E's, logψs, samples, logpcs, and max_bonds
    seed = rand(UInt)
    outs = [
        @timeit timer "remotecall" Distributed.remotecall(
            () -> Eks_threaded(peps, ham_op, k; importance_weights=false, seed=seed + w, kwargs...),
            w
        ) for w in workers()
    ]

    # Prepare global containers
    Eks         = Vector{eltype_}(undef, sample_nr_eff)
    logψs       = Vector{Complex{eltype_real}}(undef, sample_nr_eff)
    samples     = Vector{Any}(undef, sample_nr_eff)
    logpcs      = Vector{eltype_real}(undef, sample_nr_eff)
    max_bonds   = Vector{Int}(undef, sample_nr_eff)

    # Gather the results
    Threads.@threads for (i, out_i) in collect(enumerate(outs))
        i1 = k_eff * (i - 1) + 1
        i2 = k_eff * i
        out_dict = fetch(out_i)

        # Populate arrays
        Eks[i1:i2]       = out_dict[:Eks]
        logψs[i1:i2]     = out_dict[:logψs]
        samples[i1:i2]   = out_dict[:samples]
        logpcs[i1:i2]    = out_dict[:weights]
        max_bonds[i1:i2] = out_dict[:max_bond]
    end

    # Compute importance weights if needed
    if importance_weights
        weights = compute_importance_weights(logψs, logpcs)
    else
        weights = logpcs
    end

    return Dict(
        :Eks => Eks,
        :logψs => logψs,
        :samples => samples,
        :weights => weights,
        :max_bond => max_bonds
    )
end
