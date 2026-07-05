
# Calculates the Energy and Gradient of a given peps and hamiltonian
function Ok_and_Ek(peps::AbstractPEPS, ham_op; trial_state::AbstractTrialState=IdentityState(dim(siteinds(peps)[1])), 
                   timer=TimerOutput(), Ok=nothing, sampling_mode=:full,
                   resample=false, correct_sampling_error=true, resample_energy=0, # TODO: remove
                   )
    
    S, logpc, env_top = @timeit timer "sampling" get_sample(peps; trial_state=trial_state, mode=sampling_mode, timer) # draw a sample
    
    if resample
        S = QuantumNaturalGradient.resample_with_H(S, ham_op; resample_energy)
    end
    # If sampling_mode is full, we do not need to overwrite the the top environments as they are already computed accurately
    overwrite = !(sampling_mode == :full)

    logψ, env_top, env_down, max_bond = @timeit timer "vertical_envs" get_logψ_and_envs(peps, S, env_top; overwrite) # compute the environments of the peps according to that sample
    h_envs_r, h_envs_l = @timeit timer "horizontal_envs" get_all_horizontal_envs(peps, env_top, env_down, S) # computes the horizontal environments of the already sampled peps
    
    # initialize the flipped logψ dictionary, will be used to compute other observables or for the resampling
    logψ_flipped = Dict{Any, Number}()
    Ek_terms = @timeit timer "precomp_sHψ_elems"  QuantumNaturalGradient.get_precomp_sOψ_elems(ham_op, S; get_flip_sites=true)
    grad = @timeit timer "log_gradients" get_Ok(peps, env_top, env_down, S, logψ; trial_state=trial_state, h_envs_r, h_envs_l, Ok) # compute the gradient

    # add logψ from trial state such that we get the correct logψ_combined
    logψ += log(get_amplitude(trial_state, collect(vec(S))))
    E_loc = @timeit timer "energy" get_Ek(peps, ham_op, env_top, env_down, S, logψ; trial_state=trial_state, h_envs_r, h_envs_l, logψ_flipped, Ek_terms, timer) # compute the local energy

    if resample # adjust logpc, this will introduce errors as this is only an approximation of the true logpc
        @assert !correct_sampling_error "Correcting the sampling error with resampling is not implemented"
        logpc = QuantumNaturalGradient.get_logprob_resample(S, Ek_terms, logψ_flipped, ham_op; resample_energy)
    end

    if !correct_sampling_error
        logpc = 2* real(logψ)
    end

    return grad, E_loc, logψ, S, logpc, max_bond
end

"""
Calculates logψ and the environments of a given peps and sample
"""
function get_logψ_function(peps; kwargs...)
    function logψ_func(sample)
        logψ, = get_logψ_and_envs(peps, sample; kwargs...)
        return logψ
    end
    return logψ_func
end
"""
Calculates the Energy of a given a peps and hamiltonian
"""
function Ek(peps, ham_op; timer=TimerOutput(),
            trial_state::AbstractTrialState=IdentityState(dim(siteinds(peps)[1])),
            sampling_mode=:full,
            slow_energy=false, slow_energy_pos=(size(peps, 1)-1) ÷ 2)

    S, logpc, env_top = @timeit timer "sampling" get_sample(peps; trial_state=trial_state, mode=sampling_mode, timer) # draw a sample

    # If sampling_mode is full, we do not need to overwrite the the top environments as they are already computed accurately
    overwrite = !(sampling_mode == :full)

    local E_loc, logψ, max_bond
    if slow_energy
        logψ, env_top, env_down, max_bond = @timeit timer "vertical_envs" get_logψ_and_envs(peps, S, env_top; pos=slow_energy_pos, overwrite) # compute the environments of the peps according to that sample
        func = get_logψ_function(peps; pos=slow_energy_pos)
        E_loc = convert_if_real(QuantumNaturalGradient.get_Ek(S, ham_op, func))
    else
        logψ, env_top, env_down, max_bond = @timeit timer "vertical_envs" get_logψ_and_envs(peps, S, env_top; overwrite) # compute the environments of the peps according to that sample
        # add logψ from trial state such that we get the correct logψ_combined
        logψ += log(get_amplitude(trial_state, collect(vec(S))))

        # initialize the flipped logψ dictionary, will be used to compute other observables or for the resampling
        logψ_flipped = Dict{Any, Number}() 
        Ek_terms = @timeit timer "precomp_sHψ_elems"  QuantumNaturalGradient.get_precomp_sOψ_elems(ham_op, S; get_flip_sites=true)
        E_loc = @timeit timer "energy" get_Ek(peps, ham_op, env_top, env_down, S, logψ; trial_state=trial_state, logψ_flipped, Ek_terms, timer) # compute the local energy
    end
    return E_loc, logψ, S, logpc, max_bond
end