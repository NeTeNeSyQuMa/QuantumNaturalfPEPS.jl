function get_ExpectationValue(peps::AbstractPEPS, O; 
        trial_state::AbstractTrialState=IdentityState(dim(siteinds(peps)[1])),
        it=100, threaded=false, multiproc=false)
    hilbert = siteinds(peps)
    if !(O isa Vector)
        O = [O]
    end
    O_op = Array{TensorOperatorSum}(undef, length(O))

    if O[1] isa QuantumNaturalGradient.TensorOperatorSum 
        O_op = O
    else
        for i in 1:length(O)
            O_op[i] = TensorOperatorSum(O[i], hilbert)
        end
    end
    if multiproc
        #get_ExpectationValues_singlethread(peps, [O_op[1]]; it=1)
        return get_ExpectationValues_multiproc(peps, O_op; trial_state=trial_state, it=it)
    elseif threaded
        get_ExpectationValues_singlethread(peps, [O_op[1]]; it=1)
        return get_ExpectationValues_multithread(peps, O_op; trial_state=trial_state, it=it)
    else
        return get_ExpectationValues_singlethread(peps, O_op; trial_state=trial_state, it=it)
    end
end

function get_ExpectationValues_multithread(peps, O_op; 
    trial_state::AbstractTrialState=IdentityState(dim(siteinds(peps)[1])),
        it=100)
    nr_threads = Threads.nthreads()
    k = ceil(Int, it / nr_threads)
    
    Obs = complex.(zeros(k*nr_threads, length(O_op)))
    logψs = Array{Complex}(undef, k*nr_threads)
    logpcs = Array{Complex}(undef, k*nr_threads)
    
    Threads.@threads for i in 1:nr_threads
            slice = (1+(i-1)*k):(i*k)
            Obser = @view Obs[slice, :]
            logψ_thread = @view logψs[slice]
            logpc_thread = @view logpcs[slice]
            get_ExpectationValues!(peps, O_op, Obser, logψ_thread, logpc_thread; trial_state=trial_state, it=k)
    end

    #return Obs, logψs, logpcs
    return Dict(:Obs => Obs, :logψs => logψs, :logpcs => logpcs)
end

function get_ExpectationValues_singlethread(peps, O_op; 
        trial_state::AbstractTrialState=IdentityState(dim(siteinds(peps)[1])),
        it=100)
    O_loc = Array{Complex}(undef, it, length(O_op))
    logψ = Array{Complex}(undef, it)
    logpc = Array{Complex}(undef, it)
    
    get_ExpectationValues!(peps, O_op, O_loc, logψ, logpc; trial_state=trial_state, it=it)
    return O_loc, compute_importance_weights(logψ, logpc)
end

function get_ExpectationValues!(peps, O_op, Observable, logψ, logpc; 
    trial_state::AbstractTrialState=IdentityState(dim(siteinds(peps)[1])),
    it=100)

    for i in 1:it
        S, logpc[i], env_top = get_sample(peps; trial_state=trial_state)

        logψ[i], env_top, env_down, max_bond = get_logψ_and_envs(peps, S, env_top)
        h_envs_r, h_envs_l = get_all_horizontal_envs(peps, env_top, env_down, S)
        fourb_envs_r, fourb_envs_l = get_all_4b_envs(peps, env_top, env_down, S)

        logψ[i] += log(get_amplitude(trial_state, collect(vec(S))))

        logψ_flipped = Dict{Any, Number}()
        for j in 1:length(O_op)
            O_terms = QuantumNaturalGradient.get_precomp_sOψ_elems(O_op[j], S; get_flip_sites=true)
            Observable[i,j] = get_Ek(peps, O_op[j], env_top, env_down, S, logψ[i]; trial_state=trial_state, h_envs_r, h_envs_l, fourb_envs_r, fourb_envs_l, logψ_flipped, Ek_terms=O_terms)
        end
    end

    return Observable, logψ, logpc
end

function get_ExpectationValues_multiproc(peps, O_op; 
    trial_state::AbstractTrialState=IdentityState(dim(siteinds(peps)[1])),
    it=100, 
    n_threads=Distributed.remotecall_fetch(()->Threads.nthreads(), workers()[1]),
    kwargs...)

    nr_procs = length(workers())
    k = ceil(Int, it / nr_procs)
    k_thread = ceil(Int, k / n_threads)
    k_eff = k_thread * n_threads
    sample_nr_eff = k_eff * nr_procs

    out = [Distributed.remotecall(() -> get_ExpectationValues_multithread(peps, O_op; trial_state=trial_state, it=k), w) for w in workers()]

    Obs = Matrix{ComplexF64}(undef, sample_nr_eff, length(O_op))
    logψs = Vector{ComplexF64}(undef, sample_nr_eff)
    logpcs = Vector{Float64}(undef, sample_nr_eff)
    contract_dims = Vector{Int}(undef, sample_nr_eff)

    Threads.@threads for (i, out_i) in collect(enumerate(out))
        i1 = k_eff * (i - 1) + 1
        i2 = k_eff * i

        out_dict = fetch(out_i)
        Obs[i1:i2, :], logψs[i1:i2], logpcs[i1:i2] = out_dict[:Obs], out_dict[:logψs], out_dict[:logpcs]
    end

    return Obs, compute_importance_weights(logψs, logpcs)
end

# Is this function used?
function get_ExpectationValue_sample(peps, O_op, S; trial_state::AbstractTrialState=IdentityState(dim(siteinds(peps)[1])))
    O_loc = Array{Complex}(undef, 1, length(O_op))
    logψ = Array{Complex}(undef, 1)
    logpc = Array{Complex}(undef, 1)
    
    get_ExpectationValues!(peps, O_op, O_loc, logψ, logpc; trial_state=trial_state, it=1)
    return O_loc
end

function weighted_mean_error(O_loc, importance_weights) 
    w = importance_weights

    sumw = sum(w)
    μ = real(sum(O_loc[:,1] .* w) / sumw)
    var = real(sum(w .* abs2.(O_loc[:,1] .- μ)) / sumw)
    neff = 1 / sum( abs2.(w ./ sumw))
    # neff = abs2(sumw) / sum(abs2, w)
    std_err = sqrt(var / max(neff, 1e-12))

    return μ, std_err, neff
end