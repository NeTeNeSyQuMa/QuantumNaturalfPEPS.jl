function generate_double_layer_env_row(peps_row, sites, maxdim; cutoff=1e-13)
    bra = noprime.(prime.(conj(peps_row)), sites') # TODO: Improve this
    bra = MPO(bra)
    ket = MPO(peps_row)

    E_mpo = contract(bra, ket; maxdim, cutoff)
    E_mps = MPS(E_mpo[:])
    
    return Environment(E_mps; normalize=true)
end

function generate_double_layer_env_row(peps_row, sites, peps_double_env, maxdim; cutoff=1e-13)
    bra = noprime.(prime.(conj(peps_row)), sites') # TODO: Improve this

    E_mpo = MPO(peps_row .* bra)
    E_mps = contract(E_mpo, peps_double_env.env; maxdim, cutoff) # This costs (D^2 * maxdim) ^ 3, expensive!

    return Environment(E_mps, peps_double_env.f; normalize=true)
end

function generate_double_layer_envs(peps::AbstractPEPS)
    Lx = size(peps, 1)
    
    maxdim = peps.double_contract_dim
    cutoff = peps.double_contract_cutoff
    sites = siteinds(peps)

    # for every row we calculate the double layer environment
    double_layer_envs = Vector{Environment}(undef, Lx - 1)
    double_layer_envs[end] = generate_double_layer_env_row(peps[Lx, :], sites[Lx, :], maxdim; cutoff)

    for i in Lx-1:-1:2
        double_layer_envs[i-1] = generate_double_layer_env_row(peps[i, :], sites[i, :], double_layer_envs[i], maxdim; cutoff)
    end
    return double_layer_envs
end

# adds the double layer environments to the PEPS
function update_double_layer_envs!(peps::AbstractPEPS)
    peps.double_layer_envs = generate_double_layer_envs(peps) 
end

###########################################
# Sampling
###########################################


# calculates the the ket layer for the smaplings and applies (if available) already sampled rows (from above)
function get_ket(peps, i, env_top=nothing)
    ket = MPO(peps[i, :])
    if i == 1
        return ket
    end
    @assert env_top != nothing "env_top is not defined"
    return contract(ket, env_top[i-1].env; maxdim=peps.sample_dim, cutoff=peps.sample_cutoff)
end

# calculates the unsampled contractions along a row (from right to left the sites are contracted along the physical Index)
function calculate_unsampled_Env_row(ket, bra, peps, i, sites)
    Ly = size(peps, 2) 
    E = Vector{ITensor}(undef, Ly-1)
    bra = noprime.(bra, sites')

    if i == size(peps, 1)
        E[end] = bra[end] * ket[end]
        for j in Ly-1:-1:2
            E[j-1] = E[j] * ket[j] * bra[j]
        end
        return E
    end

    E[end] = bra[end] * peps.double_layer_envs[i].env[end] * ket[end]
    for j in Ly-1:-1:2
        E[j-1] = E[j] * ket[j] * peps.double_layer_envs[i].env[j] * bra[j]
    end
    return E
end

# returns the phys_dimxphys_sim matrix ρ_r which is needed to sample from. Also updates sigma (used to store the contraction of already sampled sites from the left edge to the current site)

#= 
LR:

This function computes the local reduced density matrix for a strictly single site (i,j) out of a 2D PEPS. 
It assumes that all spins to the "left" of column j in the current row i have already been sampled, 
and all spins in the rows completely above row i have also been sampled.

E: An array of "right environments". It contains the contraction of all the columns strictly to the right of j (pre-computed bounding boxes).
=#
function get_reduced_ρ(ket_j, bra_j, peps, i, j, E, sigma)
   
    if i != size(peps, 1)
        uncombined_double_layer = peps.double_layer_envs[i].env[j]
        sigma = sigma * ket_j * peps.double_layer_envs[i].env[j] * bra_j
    else
        sigma = sigma * ket_j * bra_j
    end

    if j == size(peps, 2)
        ρ_r = sigma
        return ρ_r, sigma
    end

    ρ_r = sigma * E[j]
    return ρ_r, sigma
end

row_major_site(i::Int, j::Int, Ly::Int) = (i - 1) * Ly + j
col_major_site(i::Int, j::Int, Lx::Int) = i + (j - 1) * Lx

# samples from ρ_r and updates pc
function sample_ρr(
    ρ_r,
    S,
    r,
    c;
    trial_state::AbstractTrialState=IdentityState(size(ρ_r, 1)),
    trial_sampling_cache=nothing,
    lookahead_depth=0,
)
    # prepare prob vector for PEPS
    k = size(ρ_r, 1) 
    T = real(eltype(ρ_r))
    p = Vector{T}(undef, k)
    diagonal = [ρ_r[i, i] for i in 1:k]
    diagonal_scale = maximum(abs, diagonal)
    imaginary_tolerance =
        100 * eps(T) * max(diagonal_scale, one(T)) + 1e-6 * diagonal_scale
    for i in 1:k
        z = diagonal[i]
        abs(imag(z)) <= imaginary_tolerance || throw(ArgumentError(
            "ρ_r diagonal is not real: z=$z, tolerance=$imaginary_tolerance",
        ))
        p[i] = abs(real(z))
    end

    # prepare prob vector for trial state
    current_site_key = col_major_site(r, c, size(S,1)) # use column major ordering here
    p_trial = Vector{T}(undef, k)
    if trial_sampling_cache isa ProjectedGaussianSchurCache
        k == 2 || throw(DimensionMismatch(
            "Gutzwiller-projected spin sampling requires local dimension 2, got $k",
        ))
        first(trial_sampling_cache.remaining_sites) == current_site_key ||
            throw(ArgumentError(
                "projected Gaussian Schur cache expects site " *
                "$(first(trial_sampling_cache.remaining_sites)), got $current_site_key",
            ))
        p_trial .= projected_conditional_probabilities(trial_sampling_cache; lookahead_depth)
    else
        occ_dict = Dict{Int, Int}()
        for i in 1:size(S, 1), j in 1:size(S, 2)
            if i < r || (i == r && j < c)
                occ_dict[col_major_site(i, j, size(S, 1))] = S[i, j]
            end
        end
        for i in 1:k
            occ_dict[current_site_key] = i-1
            p_trial[i] = get_prob(trial_state, occ_dict) # joint probability
        end
    end
    
    p_final = p .* p_trial

    i = sample_p(p_final, normalize=true)
    if trial_sampling_cache isa ProjectedGaussianSchurCache
        condition_projected_gaussian!(trial_sampling_cache, i - 1)
    end
    return i-1, p_final[i]
end

_direct_sampling_cache(::AbstractTrialState, _) = nothing
_direct_sampling_cache(state::ParameterizedGutzwillerProjectedState, order) =
    ProjectedGaussianSchurCache(state; order)
_direct_sampling_cache(state::FrozenTrialState, order) =
    _direct_sampling_cache(state.state, order)

_supports_projected_lookahead(::AbstractTrialState) = false
_supports_projected_lookahead(::ParameterizedGutzwillerProjectedState) = true
_supports_projected_lookahead(state::FrozenTrialState) = _supports_projected_lookahead(state.state)

function _validate_direct_lookahead(trial_state, depth)
    _validate_lookahead_depth(depth)
    depth == 0 || _supports_projected_lookahead(trial_state) || throw(ArgumentError(
        "lookahead_depth > 0 requires a ParameterizedGutzwillerProjectedState " *
        "(optionally wrapped in FrozenTrialState)",
    ))
    return depth
end

function sample_p(probs::Vector{T}; normalize=true) where T<:Real
    if normalize
        probs ./= sum(probs)
    end
    r = rand()
    psum = 0
    for (i, p) in enumerate(probs)
        psum += p
        if psum > r
            return i
        end
    end
    error("probs is not normalized sum(probs)=$(sum(probs))")
end

# generates a sample of a given peps along with pc and the top environments
#= 
    The sample has column-major ordering: S = [s1 s3; s2 s4] 
    vec(S) = [s1, s2, s3, s4] where s1 is the occupation of site (1,1), s2 of site (2,1), s3 of site (1,2) and s4 of site (2,2)
=#
function get_sample(peps::AbstractPEPS; mode::Symbol=:full, alg="densitymatrix", timer=TimerOutput(), trial_state::AbstractTrialState=IdentityState(dim(siteinds(peps)[1,1])), lookahead_depth=0)
    _validate_direct_lookahead(trial_state, lookahead_depth)
    S = Array{Int64}(undef, size(peps)) # uses row major ordering
    
    env_top = Array{Environment}(undef, size(peps, 1)-1)
    sites = siteinds(peps)
    ρ_r = ITensor()
    sampling_order = [
        col_major_site(i, j, size(peps, 1))
        for i in 1:size(peps, 1) for j in 1:size(peps, 2)
    ]
    trial_sampling_cache = _direct_sampling_cache(trial_state, sampling_order)
    
    logpc = 0
    # we loop through every row (This uses row-major ordering)
    for i in 1:size(peps, 1)
        sigma = 1
        ket = @timeit timer "env_sample" get_ket(peps, i, env_top)
        bra = prime.(conj(ket[:]))

        # we then calculate the unsampled environment (in one row)
        E = @timeit timer "env_row" calculate_unsampled_Env_row(ket, bra, peps, i, sites[i, :])

        # then we loop through the different sites in one row
        for j in 1:size(peps, 2)
            
            # calculate the phys_dimxphys_dim matrix from which we sample
            ρ_r, sigma = get_reduced_ρ(ket[j], bra[j], peps, i, j, E, sigma)
            
            # sample from ρ_r
            S[i, j], pc = sample_ρr(
                ρ_r,
                S,
                i,
                j;
                trial_state,
                trial_sampling_cache,
                lookahead_depth,
            )
            logpc += log(pc)
            
            # after the sampling of the current site, it is fixed and its contraction with the aleady sampled sites is stored in sigma
            site = siteind(peps, i, j)
            sigma = sigma * get_projector(S[i, j], sites[i, j]) * get_projector(S[i, j], sites[i, j]') 
            sigma ./= pc # we divide by pc to avoid numerical issues
        end
        
        if mode === :fast
            # the sampled bra is used to generate the top environments
            ket = ket .* [get_projector(S[i, j], siteind(peps, i, j)) for j in 1:size(peps, 2)]
            if i == 1
                env_top[i] = Environment(MPS(ket.data); normalize=true)
            elseif i != size(peps, 1) 
                env_top[i] = Environment(MPS(ket.data), env_top[i-1].f; normalize=true)
            end

        elseif mode === :full
            if i == 1
                peps_projected_1 = get_projected(peps, S, 1, :)
                @timeit timer "env_top" env_top[1] = generate_env_row(peps_projected_1, peps.contract_dim; alg, cutoff=peps.contract_cutoff)
            elseif i != size(peps, 1) 
                peps_projected_row = get_projected(peps, S, i, :)
                @timeit timer "env_top" env_top[i] = generate_env_row(peps_projected_row, peps.contract_dim; env_row_above=env_top[i-1], alg, cutoff=peps.contract_cutoff)
            end  
        end
    end
    
    return S, logpc, env_top
end

#= 
    Trial state variants
=#


# samples from ρ_r and updates pc
function sample_ρr(GS::GaussianState, S, r, c, M_cache::OccupationProjectorCache)
    n_measured = (r - 1) * size(S, 2) + c
    site_idx = (c - 1) * size(S, 1) + r # true column-major linear index

    # flip the occupation of the current site to compute the probabilities for both configurations
    set_occ_projector_block!(M_cache.M_j, site_idx, 0)
    p0 = get_prob!(GS, M_cache, n_measured)
    set_occ_projector_block!(M_cache.M_j, site_idx, 1)
    p1 = get_prob!(GS, M_cache, n_measured)

    p_final = [p0, p1]

    i = sample_p(p_final, normalize=true)

    # update occ projector for drawn occupation
    set_occ_projector_block!(M_cache.M_j, site_idx, i-1)

    return i-1, p_final[i]
end

# generates a sample of a given peps along with pc and the top environments
function get_sample(GS::GaussianState; timer=TimerOutput())
    L = Int(sqrt(GS.N)) # TODO: this only works for square lattices, we should make this more general

    S = Array{Int64}(undef, L, L)
    M_cache = OccupationProjectorCache(GS.N)
    
    logpc = 0
    # we loop through every row
    for i in 1:L
        # then we loop through the different sites in one row
        for j in 1:L
            # sample from Slater wave function
            S[i, j], pc = sample_ρr(GS, S, i, j, M_cache)
            logpc += log(pc)
        end
    end
    
    return S, logpc
end
