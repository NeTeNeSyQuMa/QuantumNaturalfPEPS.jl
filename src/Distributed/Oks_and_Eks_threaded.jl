###### Multiple threads
function generate_Oks_and_Eks_threaded(peps::AbstractPEPS, ham_op::TensorOperatorSum; timer=TimerOutput(),
                                       double_layer_update=update_double_layer_envs!,
                                       kwargs...)
    function Oks_and_Eks_(Θ::Vector{T}, sample_nr::Integer; reset_double_layer=true, kwargs2...) where T
        if length(kwargs2) > 0
            kwargs = merge(kwargs, kwargs2)
        end
        write!(peps, Θ; reset_double_layer)
        if reset_double_layer
            @timeit timer "double_layer_envs" double_layer_update(peps) # update the double layer environments once for the peps
        end
        return Oks_and_Eks_threaded(peps, ham_op, sample_nr; timer, kwargs...)
    end
    function Oks_and_Eks_(peps_::Parameters{<:AbstractPEPS}, sample_nr::Integer; reset_double_layer=false, kwargs2...)
        peps_ = peps_.obj
        if getfield(peps_, :double_layer_envs) === nothing
            @timeit timer "double_layer_envs" double_layer_update(peps_)
        end
        
        if length(kwargs2) > 0
            kwargs = merge(kwargs, kwargs2)
        end

        return Oks_and_Eks_threaded(peps_, ham_op, sample_nr; timer, kwargs...)
    end
    return Oks_and_Eks_
end

function Oks_and_Eks_threaded(peps, ham_op, sample_nr; Oks=nothing, importance_weights=true,
                              timer=TimerOutput(), nr_threads=Threads.nthreads(), seed=nothing,
                              return_Oks=true,
                              kwargs...)
    
    
    if seed !== nothing
        Random.seed!(seed)
    end

    nr_parameters = length(peps)
    k = ceil(Int, sample_nr / nr_threads)
    
    eltype_ = eltype(peps)
    eltype_real = real(eltype_)

    if Oks === nothing
        Oks = Array{eltype_, 3}(undef, nr_parameters, k, nr_threads)
    elseif Oks isa AbstractArray
        if ndims(Oks) == 2
            Oks = reshape(Oks, nr_parameters, k, nr_threads)
        end
    else
        error("Oks must be either nothing or a Array")
    end
    samples = Matrix{Any}(undef, k, nr_threads)
    Eks = Matrix{eltype_}(undef, k, nr_threads)
    logψs = Matrix{Complex{eltype_real}}(undef, k, nr_threads)
    logpcs = Matrix{eltype_real}(undef, k, nr_threads)
    contract_dims = Matrix{Int}(undef, k, nr_threads)
    
    seed = rand(UInt)
    Threads.@threads for i in 1:nr_threads
        Random.seed!(seed + i)
    
        #   Crude fix for the unexpected inexact error
        for j in 1:k
            Ok = @view Oks[:, j, i]
            _,O1,logψs[j, i],samples[j, i],O4,contract_dims[j, i] = Ok_and_Ek(peps, ham_op; Ok, kwargs...)
            if eltype(O1) != eltype_
                if abs(imag(O1))>10^-6
                    @warn "Large imaginary part detected"
                end    
                Eks[j, i] = real(O1)
            else
                Eks[j, i]=O1
            end 
            logpcs[j, i] = real(O4)
            #_, Eks[j, i], logψs[j, i], samples[j, i], logpcs[j, i], contract_dims[j, i] = Ok_and_Ek(peps, ham_op; Ok, kwargs...)
        end
    end
    Eks = reshape(Eks, :)
    Oks = reshape(Oks, size(Oks, 1), :)
    logψs = reshape(logψs, :)
    samples = reshape(samples, :)
    logpcs = reshape(logpcs, :)
    contract_dims = reshape(contract_dims, :)
    
    if importance_weights
        weights = compute_importance_weights(logψs, logpcs)
    else
        weights = logpcs
    end
    
    data = Dict{Symbol, Any}(:Eks => Eks, :logψs => logψs, :samples => samples, :weights => weights, :contract_dims => contract_dims)

    if return_Oks
        data[:Oks] = transpose(Oks)
    end
    return data
end