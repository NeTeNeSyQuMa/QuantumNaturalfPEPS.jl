function get_Ok(peps::AbstractPEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, logψ::Number, h_envs_r, h_envs_l, i::Int, j::Int)
    ok_tensor = 1
    f = 0
    if j != size(peps, 2)
        ok_tensor *= h_envs_r[i,j]
    end
    if i != 1
        ok_tensor *= env_top[i-1].env[j]
        f += env_top[i-1].f
    end
    if i != size(peps, 1)
        ok_tensor *= env_down[end-i+1].env[j]
        f += env_down[end-i+1].f
    end
    if j != 1
        ok_tensor *= h_envs_l[i,j-1]
    end
    g = exp(f - logψ)
    # if the tensor is real, we only want the real part of the gradient
    if isreal(ok_tensor) 
        g = real(g)
    end
    
    ok_tensor *= g
    @assert eltype(ok_tensor) === eltype(peps) "The gradient is $(eltype(ok_tensor)) but the PEPS is $(eltype(peps))"
    return ok_tensor
end


# calculates the gradient: d(<ψ|S>)/d(Θ) / <ψ|S>
function get_Ok(peps::AbstractPEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, logψ::Number;
                trial_state::AbstractTrialState=IdentityState(dim(siteinds(peps)[1])), h_envs_r=nothing, h_envs_l=nothing, Ok=nothing, mask=peps.mask)
    if Ok === nothing
        Ok = Vector{eltype(peps)}(undef, length(peps) + length(Parameters(trial_state)))
    end
    if h_envs_r === nothing || h_envs_l === nothing
        h_envs_r, h_envs_l = get_all_horizontal_envs(peps, env_top, env_down, S) # computes the horizontal environments of the already sampled peps
    end

    pos = 1
    shift = 0
    
    for j in 1:size(peps, 2), i in 1:size(peps, 1) # use column major ordering of PEPS here
        if mask[i,j] != 0
            ok_tensor = get_Ok(peps, env_top, env_down, logψ, h_envs_r, h_envs_l, i, j)
            # lastly we reshape the tensor to a vector to obtain the gradient
            shift = prod(dim.(inds(ok_tensor)))
            loc_dim = dim(siteind(peps, i,j))
            
            # loop through every possible sample
            for spin in 0:loc_dim-1
                # if we get to the actually sampled value write the tensor in the gradient, else fill with zeros
                if S[i,j] == spin
                    # Write in Gradient
                    x = @view Ok[pos+spin:loc_dim:pos+loc_dim*shift-1]
                    permute_reshape_and_copy!(x, ok_tensor, linkinds(peps, i, j))
                else
                    # Fill with zeros instead
                    Ok[pos+spin:loc_dim:pos+loc_dim*shift-1] .= 0
                end
            end
            pos = pos + loc_dim * shift
        end
    end

    # gradient part of the trial state
    Ok = get_Ok(trial_state, S, Ok)

    return Ok
end

#= 
    Here Ok funcitons for trial states
=#
function get_Ok(trial_state::IdentityState, S::Matrix{Int64}, Ok)
    return Ok # identity state has no variational parameters, so return the Ok object unchanged
end

function get_Ok(trial_state::FrozenTrialState, S::Matrix{Int64}, Ok)
    return Ok # frozen trial state exposes no variational parameters, so return the Ok object unchanged
end

"""
    get_Ok(trial_state::GaussianState, S::Matrix{Int64}, Ok)

Calculates the gradient of the log-amplitude Ok_j = ∂η ln(S_j) for a given GaussianState with variational parameters `trial_state.η` and sample `S`.

It is calculated according to the formula:
```
Oⱼ,ᵢ(η) = ∂η ln(Sⱼ(η)) = 0.25 * Tr[Fⱼ⁻¹(η) Γ⁻¹(η) ∂ηᵢ(Γ(η)) Γ⁻¹(η)]
```

For pure Gaussian states: `Γ⁻¹ = -Γ` and the formula can be simplified to:
```
Oⱼ,ᵢ(η) = ∂η ln(Sⱼ(η)) = 0.25 * Tr[Fⱼ⁻¹(η) ∂ηᵢ Γ(η)]
```

where `Fⱼ = Mⱼ - Γ⁻¹` and `Mⱼ` is the matrix for the occupation projector to the configuration `j`.

"""
function get_Ok(trial_state::GaussianState, S::Matrix{Int64}, Ok)
    η_idx = size(Ok, 1) - length(trial_state.η) # start index for the variational parameters of the trial state
    occ_string = collect(vec(S))
    N = length(occ_string)

    slater_loggrad_cache = trial_state.slater_loggrad_cache
    Γ = trial_state.Γ

    is_pure = Γ * Γ ≈ -I/4

    # Build matrix M_j to occupation projector to configuration S_j
    M_j = build_occ_projector_matrix(to_occ_dict(occ_string), N)

    F_j = begin
        tmp = similar(Γ)
        # factor of 2 is needed as in our convention we have Γ² = -I/4
        if is_pure # Fⱼ = Mⱼ + Γ
            tmp .= 2*Γ
            tmp .+= M_j
        else # Fⱼ = Mⱼ - Γ⁻¹
            tmp .= M_j
            tmp .-= inv(2*Γ)
        end
        tmp
    end

    # check=false for speedup as it doesnt check for singularities then
    F_fac = lu!(F_j; check=false)
    invΓ = is_pure ? nothing : inv(lu!(copy(Γ); check=false))

    X = similar(slater_loggrad_cache.dΓs[1])
    for i in eachindex(trial_state.η)
        dΓ = slater_loggrad_cache.dΓs[i] # ∂ηᵢ Γ

        if is_pure
            # =========================================================================
            # O_k = 0.25 * Tr(F_j⁻¹ dΓ_k )
            # =========================================================================

            # Solve: F_j X = dΓ, so X = F_j⁻¹ dΓ
            ldiv!(X, F_fac, dΓ)
            Ok[η_idx + i] = real(0.25 * tr(X))
        else
            # =========================================================================
            # O_k = 0.25 * Tr(F_j⁻¹ Γ⁻¹ dΓ_k Γ⁻¹)
            # =========================================================================

            # solve F_j X = Y, with Y = Γ⁻¹ * dΓ * Γ⁻¹
            ldiv!(X, F_fac, invΓ * dΓ * invΓ)
            Ok[η_idx + i] = real(0.25 * tr(X))
        end
    end

    return Ok
end

function numerical_Ok_exact(peps::AbstractPEPS, S::Matrix{Int64}, direction; dt=0.001)
    p2 = deepcopy(peps)
    θ = vec(peps)
    f = 0
    for i in [-1/2, 1/2]
        write!(p2, θ .+ i .* dt .* direction)
        x = get_projected(p2, S)
        logψ = log(contract_peps_exact(x)+0im)
        f += sign(i) * logψ
    end
    return f / dt
end


function numerical_Ok(peps::AbstractPEPS, S::Matrix{Int64}, direction; dt=0.001)
    p2 = deepcopy(peps)
    θ = vec(peps)
    f = 0
    for i in [-1/2, 1/2]
        write!(p2, θ .+ i .* dt .* direction)
        x = get_projected(p2, S)
        logψ, = QuantumNaturalfPEPS.get_logψ_and_envs(p2, S);
        f += sign(i) * logψ
    end
    return f / dt
end