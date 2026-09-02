"""
    generate_Oks_and_Eks_fixed_sz(peps, ham_op; trial_state, kwargs...)

Build the `Oks_and_Eks` callback used by `QuantumNaturalGradient.evolve` for a
Gutzwiller-projected state in a fixed-`S^z` sector. Samples are drawn from the
combined `D=1` PPEPS--Gutzwiller probability with opposite-spin Metropolis
exchanges. For `D=1`, both the PPEPS amplitude ratios and its logarithmic
derivatives are exact scalar contractions.

When the projected state is trainable, the callback parameter vector and the
columns of `Oks` are ordered as `vcat(vec(peps), Parameters(trial_state))`.

This backend is selected through
`generate_Oks_and_Eks(...; fixed_sz_metropolis=true)`. It intentionally only
supports `bond_dim == 1`; nontrivial PPEPS bond dimensions must use the usual
PEPS contraction and sampling backend.
"""
function generate_Oks_and_Eks_fixed_sz(
    peps::AbstractPEPS,
    ham_op::TensorOperatorSum;
    trial_state::AbstractGutzwillerProjectedState,
    burnin_sweeps::Integer=10,
    moves_per_sample::Union{Nothing,Integer}=nothing,
    rebuild_after::Integer=64,
    maximum_initialization_attempts::Integer=100,
    seed::Union{Nothing,Integer}=nothing,
    kwargs...,
)
    _check_fixed_sz_d1_inputs(peps, ham_op, trial_state)
    compiled_hamiltonian = _compile_fixed_sz_hamiltonian(ham_op)
    burnin_sweeps >= 0 || throw(ArgumentError("burnin_sweeps must be nonnegative"))
    if !isnothing(moves_per_sample)
        moves_per_sample > 0 || throw(ArgumentError("moves_per_sample must be positive"))
    end
    rebuild_after > 0 || throw(ArgumentError("rebuild_after must be positive"))
    maximum_initialization_attempts > 0 || throw(ArgumentError(
        "maximum_initialization_attempts must be positive",
    ))

    evaluation = Ref(0)
    function Oks_and_Eks_(θ::Vector{T}, sample_nr::Integer; kwargs2...) where {T}
        number_of_peps_parameters = length(peps)
        number_of_trial_parameters = length(Parameters(trial_state))
        expected_parameters = number_of_peps_parameters + number_of_trial_parameters
        length(θ) == expected_parameters || throw(DimensionMismatch(
            "the fixed-Sz D=1 backend expects $expected_parameters parameters, " *
            "consisting of $number_of_peps_parameters PPEPS and " *
            "$number_of_trial_parameters trial-state parameters; " *
            "got $(length(θ))",
        ))
        write!(peps, θ[1:number_of_peps_parameters]; reset_double_layer=false)
        if number_of_trial_parameters > 0
            write!(trial_state, θ[number_of_peps_parameters+1:end])
        end
        evaluation[] += 1
        evaluation_seed = isnothing(seed) ? nothing : seed + evaluation[] - 1
        return Oks_and_Eks_fixed_sz(
            peps,
            ham_op,
            sample_nr;
            trial_state,
            burnin_sweeps,
            moves_per_sample,
            rebuild_after,
            maximum_initialization_attempts,
            compiled_hamiltonian,
            seed=evaluation_seed,
            kwargs...,
            kwargs2...,
        )
    end

    function Oks_and_Eks_(parameters::Parameters{<:AbstractPEPS}, sample_nr::Integer; kwargs2...)
        parameters.obj === peps || throw(ArgumentError(
            "the Parameters object must wrap the PPEPS used to construct this callback",
        ))
        θ = vcat(vec(peps), Parameters(trial_state))
        return Oks_and_Eks_(θ, sample_nr; kwargs2...)
    end
    return Oks_and_Eks_
end

struct _FixedSzDiagonalTerm
    sites::Vector{Int}
    dimensions::Vector{Int}
    values::Vector{ComplexF64}
end

struct _FixedSzExchangeTransition
    i::Int
    j::Int
    old_i::Int
    old_j::Int
    coefficient::ComplexF64
end

struct _FixedSzCompiledHamiltonian
    diagonal_terms::Vector{_FixedSzDiagonalTerm}
    exchange_transitions::Vector{_FixedSzExchangeTransition}
end

function _compile_fixed_sz_hamiltonian(ham_op::TensorOperatorSum)
    diagonal_terms = [
        _FixedSzDiagonalTerm(
            collect(Int, sites),
            collect(Int, dimensions),
            ComplexF64.(values),
        ) for (sites, dimensions, values) in zip(
            ham_op.diag_sites,
            ham_op.diag_dims,
            ham_op.diag_tensors,
        )
    ]

    # Combining equal transitions removes, for example, the same-spin pair
    # flips that cancel between Sx*Sx and Sy*Sy.
    transition_coefficients = Dict{NTuple{6,Int},ComplexF64}()
    hilbert = vec(ham_op.hilbert)
    for (tensor, sites_untyped) in zip(ham_op.tensors, ham_op.sites)
        sites = collect(Int, sites_untyped)
        length(sites) == 2 || throw(ArgumentError(
            "fixed_sz_metropolis only supports two-site off-diagonal operators",
        ))
        local_hilbert = hilbert[sites]
        for input_code in 0:3, output_code in 0:3
            input = digits(input_code; base=2, pad=2)
            output = digits(output_code; base=2, pad=2)
            indices = Pair{Index{Int64},Int}[]
            append!(indices, [index' => spin + 1 for (index, spin) in zip(local_hilbert, input)])
            append!(indices, [index => spin + 1 for (index, spin) in zip(local_hilbert, output)])
            coefficient = ComplexF64(tensor[indices...])
            iszero(coefficient) && continue

            changed = findall(input .!= output)
            isempty(changed) && throw(ArgumentError(
                "an off-diagonal TensorOperatorSum entry contains a diagonal matrix element",
            ))
            length(changed) == 2 || throw(ArgumentError(
                "fixed_sz_metropolis encountered a term that changes $(length(changed)) spins",
            ))

            first_local, second_local = changed
            i, j = sites[first_local], sites[second_local]
            old_i, old_j = input[first_local], input[second_local]
            new_i, new_j = output[first_local], output[second_local]
            if j < i
                i, j = j, i
                old_i, old_j = old_j, old_i
                new_i, new_j = new_j, new_i
            end
            key = (i, j, old_i, new_i, old_j, new_j)
            transition_coefficients[key] = get(transition_coefficients, key, 0.0im) + coefficient
        end
    end

    exchange_transitions = _FixedSzExchangeTransition[]
    for (key, coefficient) in transition_coefficients
        abs(coefficient) <= 100eps(Float64) && continue
        i, j, old_i, new_i, old_j, new_j = key
        old_i != old_j && new_i == old_j && new_j == old_i || throw(ArgumentError(
            "fixed_sz_metropolis Hamiltonian contains a non-Sz-conserving transition $key",
        ))
        push!(exchange_transitions, _FixedSzExchangeTransition(
            i,
            j,
            old_i,
            old_j,
            coefficient,
        ))
    end
    sort!(exchange_transitions; by=transition -> (
        transition.i,
        transition.j,
        transition.old_i,
        transition.old_j,
    ))
    return _FixedSzCompiledHamiltonian(diagonal_terms, exchange_transitions)
end

function _check_fixed_sz_d1_inputs(peps, ham_op, trial_state)
    peps.bond_dim == 1 || throw(ArgumentError(
        "fixed_sz_metropolis currently requires a D=1 PPEPS, got bond_dim=$(peps.bond_dim)",
    ))
    all(!iszero, peps.mask) || throw(ArgumentError(
        "fixed_sz_metropolis currently requires every PPEPS tensor to be trainable",
    ))
    isnothing(trial_state.Nup) && throw(ArgumentError(
        "fixed_sz_metropolis requires a projected state with a fixed Nup",
    ))
    size(peps) == size(ham_op) || throw(DimensionMismatch(
        "PPEPS size $(size(peps)) does not match Hamiltonian size $(size(ham_op))",
    ))
    length(peps) == 2trial_state.N || throw(DimensionMismatch(
        "a spin-1/2 D=1 PPEPS on $(trial_state.N) sites must have " *
        "$(2trial_state.N) parameters, got $(length(peps))",
    ))
    return nothing
end

function _fixed_sz_d1_weights(peps)
    weights = reshape(ComplexF64.(vec(peps)), 2, :)
    all(!iszero, weights) || throw(ArgumentError(
        "D=1 PPEPS entries must be nonzero for fixed-sector amplitude ratios",
    ))
    all(isfinite, weights) || throw(ArgumentError("D=1 PPEPS entries must be finite"))
    return weights
end

function _random_gutzwiller_exchange_cache(
    rng,
    state::AbstractGutzwillerProjectedState;
    maximum_attempts::Integer,
)
    Nup = something(state.Nup)
    for _ in 1:maximum_attempts
        configuration = ones(Int, state.N)
        configuration[randperm(rng, state.N)[1:Nup]] .= 0
        try
            cache = GutzwillerExchangeCache(state, configuration)
            all(isfinite, cache.inverse_slater) && return cache
        catch exception
            exception isa SingularException || rethrow()
        end
    end
    error("failed to find a nonsingular fixed-Sz Gutzwiller configuration")
end

@inline function _d1_exchange_ratio(weights, configuration, i::Int, j::Int)
    spin_i = configuration[i] + 1
    spin_j = configuration[j] + 1
    return weights[spin_j, i] * weights[spin_i, j] /
           (weights[spin_i, i] * weights[spin_j, j])
end

function _fixed_sz_metropolis_move!(
    rng,
    cache,
    weights,
    up_sites,
    down_sites;
    rebuild_after,
    adjacent_sector_cache=nothing,
    adjacent_sector_anchor=nothing,
)
    isnothing(adjacent_sector_cache) == isnothing(adjacent_sector_anchor) || throw(
        ArgumentError("adjacent-sector cache and anchor must be supplied together"),
    )
    up_position = rand(rng, eachindex(up_sites))
    down_position = rand(rng, eachindex(down_sites))
    i = up_sites[up_position]
    j = down_sites[down_position]
    amplitude_ratio = gutzwiller_exchange_ratio(cache, i, j) *
                      _d1_exchange_ratio(weights, cache.configuration, i, j)

    if rand(rng) < min(1.0, abs2(amplitude_ratio))
        accept_gutzwiller_exchange!(cache, i, j; rebuild_after)
        if !isnothing(adjacent_sector_cache)
            if j == adjacent_sector_anchor[]
                # The base-sector exchange moves the flipped (anchor) spin to
                # i, while the adjacent-sector configuration itself is unchanged.
                adjacent_sector_anchor[] = i
            else
                accept_gutzwiller_exchange!(adjacent_sector_cache, i, j; rebuild_after)
            end
        end
        up_sites[up_position] = j
        down_sites[down_position] = i
        return true
    end
    return false
end

function _initialize_adjacent_sector_cache(base_cache, adjacent_state)
    base_Nup = count(==(0), base_cache.configuration)
    adjacent_state.N == length(base_cache.configuration) || throw(DimensionMismatch(
        "the adjacent flux-sector state must have the same number of sites",
    ))
    adjacent_state.Nup == base_Nup + 1 || throw(ArgumentError(
        "the monopole state must have Nup=$(base_Nup + 1), got $(adjacent_state.Nup)",
    ))

    for anchor in findall(==(1), base_cache.configuration)
        adjacent_configuration = copy(base_cache.configuration)
        adjacent_configuration[anchor] = 0
        try
            adjacent_cache = GutzwillerExchangeCache(
                adjacent_state,
                adjacent_configuration,
            )
            all(isfinite, adjacent_cache.inverse_slater) || continue
            return adjacent_cache, Ref(anchor)
        catch exception
            exception isa SingularException || rethrow()
        end
    end
    error("failed to initialize a nonsingular adjacent-sector monopole cache")
end

@inline function _fixed_sz_linear_site(site::Integer, ::Tuple{Int,Int})
    return Int(site)
end

@inline function _fixed_sz_linear_site(site::CartesianIndex{2}, lattice_size::Tuple{Int,Int})
    return LinearIndices(lattice_size)[site]
end

@inline function _fixed_sz_linear_site(site::Tuple{<:Integer,<:Integer}, lattice_size::Tuple{Int,Int})
    return LinearIndices(lattice_size)[site...]
end

function _fixed_sz_local_energy(cache, weights, compiled_hamiltonian)
    configuration = cache.configuration
    energy = 0.0 + 0.0im
    for term in compiled_hamiltonian.diagonal_terms
        table_index = 1
        stride = 1
        for (site, dimension) in zip(term.sites, term.dimensions)
            table_index += stride * configuration[site]
            stride *= dimension
        end
        energy += term.values[table_index]
    end
    for transition in compiled_hamiltonian.exchange_transitions
        i, j = transition.i, transition.j
        if configuration[i] == transition.old_i && configuration[j] == transition.old_j
            amplitude_ratio = gutzwiller_exchange_ratio(cache, i, j) *
                              _d1_exchange_ratio(weights, configuration, i, j)
            energy += transition.coefficient * amplitude_ratio
        end
    end
    return energy
end

function _fixed_sz_log_amplitude(cache, weights)
    result = cache.log_amplitude
    for (site, spin) in enumerate(cache.configuration)
        result += log(weights[spin + 1, site])
    end
    return result
end

function _fill_fixed_sz_d1_Ok!(Ok, weights, configuration)
    fill!(Ok, zero(eltype(Ok)))
    for (site, spin) in enumerate(configuration)
        parameter = 2site - 1 + spin
        Ok[parameter] = inv(weights[spin + 1, site])
    end
    return Ok
end

function _fill_fixed_sz_transverse_correlations!(
    reference_correlator,
    displacement_correlator,
    cache,
    weights,
    lattice_size::Tuple{Int,Int},
    reference_site::Int,
)
    Lx, Ly = lattice_size
    N = Lx * Ly
    configuration = cache.configuration
    fill!(reference_correlator, 0)
    fill!(displacement_correlator, 0)

    # Sx_i*Sx_i + Sy_i*Sy_i = 1/2 for spin 1/2.
    reference_correlator[reference_site] = 0.5
    displacement_correlator[1, 1] = 0.5

    for i in 1:N
        spin_i = configuration[i]
        xi = mod(i - 1, Lx)
        yi = fld(i - 1, Lx)
        for j in 1:N
            i == j && continue
            spin_i == configuration[j] && continue

            # Exactly one of S_i^+ S_j^- and S_i^- S_j^+ acts on an
            # opposite-spin configuration, hence the factor 1/2 in Cperp.
            estimator = gutzwiller_exchange_ratio(cache, i, j) *
                        _d1_exchange_ratio(weights, configuration, i, j) / 2
            if i == reference_site
                reference_correlator[j] = estimator
            end

            xj = mod(j - 1, Lx)
            yj = fld(j - 1, Lx)
            dx = mod(xj - xi, Lx) + 1
            dy = mod(yj - yi, Ly) + 1
            displacement_correlator[dx, dy] += estimator / N
        end
    end
    return nothing
end

function _fill_fixed_sz_monopole_matrix_element!(
    estimator,
    base_cache,
    adjacent_cache,
    adjacent_anchor::Int,
    weights,
)
    fill!(estimator, 0)
    base_configuration = base_cache.configuration
    base_configuration[adjacent_anchor] == 1 || error(
        "the adjacent-sector anchor must be down in the base configuration",
    )
    adjacent_cache.configuration[adjacent_anchor] == 0 || error(
        "the adjacent-sector anchor must be up in the adjacent configuration",
    )

    anchor_ratio = exp(adjacent_cache.log_amplitude - base_cache.log_amplitude) *
                   weights[1, adjacent_anchor] / weights[2, adjacent_anchor]
    for site in eachindex(base_configuration)
        base_configuration[site] == 1 || continue
        ratio = if site == adjacent_anchor
            anchor_ratio
        else
            anchor_ratio * gutzwiller_exchange_ratio(
                adjacent_cache,
                adjacent_anchor,
                site,
            ) * _d1_exchange_ratio(
                weights,
                adjacent_cache.configuration,
                adjacent_anchor,
                site,
            )
        end
        # Sampling uses |psi_n|^2, so the local cross-sector estimator is
        # conj(psi_{n+1}(sigma with i flipped) / psi_n(sigma)).
        estimator[site] = conj(ratio)
    end
    return estimator
end

function _fixed_sz_permutation_ratio(cache, weights, sites, target_spins)
    configuration = cache.configuration
    changed_sites = Int[]
    for (site, target_spin) in zip(sites, target_spins)
        configuration[site] == target_spin || push!(changed_sites, site)
    end
    isempty(changed_sites) && return 1.0 + 0.0im
    length(changed_sites) == 2 || error(
        "a spin permutation must leave zero or two sites changed",
    )
    first_site, second_site = changed_sites
    return gutzwiller_exchange_ratio(cache, first_site, second_site) *
           _d1_exchange_ratio(weights, configuration, first_site, second_site)
end

function _fixed_sz_scalar_chirality(cache, weights, i::Int, j::Int, k::Int)
    configuration = cache.configuration
    spins = (configuration[i], configuration[j], configuration[k])

    # For chi_ijk = S_i dot (S_j x S_k),
    # chi = (i/4) (P_ij P_jk - P_jk P_ij).  In a VMC local estimator the
    # permutations act on the wavefunction to the right, hence the two target
    # configurations below appear in inverse order.
    inverse_cycle = (spins[2], spins[3], spins[1])
    forward_cycle = (spins[3], spins[1], spins[2])
    sites = (i, j, k)
    inverse_ratio = _fixed_sz_permutation_ratio(
        cache,
        weights,
        sites,
        inverse_cycle,
    )
    forward_ratio = _fixed_sz_permutation_ratio(
        cache,
        weights,
        sites,
        forward_cycle,
    )
    return im * (inverse_ratio - forward_ratio) / 4
end

function _fill_fixed_sz_scalar_chirality!(up, down, cache, weights, lattice_size)
    Lx, Ly = lattice_size
    linear_site(x, y) = mod(x, Lx) + 1 + mod(y, Ly) * Lx
    for y in 0:Ly-1, x in 0:Lx-1
        r = linear_site(x, y)
        rx = linear_site(x + 1, y)
        ry = linear_site(x, y + 1)
        rxy = linear_site(x + 1, y + 1)

        # Both vertex lists are counterclockwise for primitive vectors
        # a1=(1,0), a2=(-1/2,sqrt(3)/2).
        up[x + 1, y + 1] = _fixed_sz_scalar_chirality(cache, weights, r, rx, rxy)
        down[x + 1, y + 1] = _fixed_sz_scalar_chirality(cache, weights, r, rxy, ry)
    end
    return nothing
end

function _fixed_sz_block_error(samples, block_length::Integer)
    number_of_samples = size(samples, 1)
    number_of_blocks = fld(number_of_samples, block_length)
    number_of_blocks >= 2 || throw(ArgumentError(
        "transverse measurements require at least two complete blocks; " *
        "got sample_nr=$number_of_samples and transverse_block_length=$block_length",
    ))

    result_shape = size(samples)[2:end]
    block_means = Array{ComplexF64}(undef, number_of_blocks, result_shape...)
    for block in 1:number_of_blocks
        sample_range = (block - 1) * block_length + 1:block * block_length
        selectdim(block_means, 1, block) .= dropdims(
            mean(view(samples, sample_range, ntuple(_ -> Colon(), ndims(samples) - 1)...); dims=1);
            dims=1,
        )
    end
    return dropdims(std(real.(block_means); dims=1); dims=1) / sqrt(number_of_blocks)
end

function Oks_and_Eks_fixed_sz(
    peps::AbstractPEPS,
    ham_op::TensorOperatorSum,
    sample_nr::Integer;
    trial_state::AbstractGutzwillerProjectedState,
    burnin_sweeps::Integer=10,
    moves_per_sample::Union{Nothing,Integer}=nothing,
    rebuild_after::Integer=64,
    maximum_initialization_attempts::Integer=100,
    compiled_hamiltonian::Union{Nothing,_FixedSzCompiledHamiltonian}=nothing,
    seed::Union{Nothing,Integer}=nothing,
    measure_transverse::Bool=false,
    transverse_reference::Integer=1,
    transverse_block_length::Integer=20,
    monopole_state::Union{Nothing,AbstractGutzwillerProjectedState}=nothing,
    measure_chirality::Bool=false,
    chirality_block_length::Integer=20,
    kwargs...,
)
    sample_nr > 1 || throw(ArgumentError("sample_nr must be at least 2"))
    _check_fixed_sz_d1_inputs(peps, ham_op, trial_state)
    N = trial_state.N
    1 <= transverse_reference <= N || throw(ArgumentError(
        "transverse_reference must lie between 1 and N=$N",
    ))
    transverse_block_length > 0 || throw(ArgumentError(
        "transverse_block_length must be positive",
    ))
    chirality_block_length > 0 || throw(ArgumentError(
        "chirality_block_length must be positive",
    ))
    moves = isnothing(moves_per_sample) ? N : moves_per_sample
    moves > 0 || throw(ArgumentError("moves_per_sample must be positive"))
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    weights = _fixed_sz_d1_weights(peps)
    compiled_hamiltonian = isnothing(compiled_hamiltonian) ?
        _compile_fixed_sz_hamiltonian(ham_op) : compiled_hamiltonian
    cache = _random_gutzwiller_exchange_cache(
        rng,
        trial_state;
        maximum_attempts=maximum_initialization_attempts,
    )
    adjacent_cache, adjacent_anchor = isnothing(monopole_state) ?
        (nothing, nothing) : _initialize_adjacent_sector_cache(cache, monopole_state)
    up_sites = findall(==(0), cache.configuration)
    down_sites = findall(==(1), cache.configuration)
    isempty(up_sites) && throw(ArgumentError("the fully down-polarized sector has no exchange moves"))
    isempty(down_sites) && throw(ArgumentError("the fully up-polarized sector has no exchange moves"))

    accepted = 0
    attempted = 0
    for _ in 1:(burnin_sweeps * N)
        accepted += _fixed_sz_metropolis_move!(
            rng,
            cache,
            weights,
            up_sites,
            down_sites;
            rebuild_after,
            adjacent_sector_cache=adjacent_cache,
            adjacent_sector_anchor=adjacent_anchor,
        )
        attempted += 1
    end

    number_of_peps_parameters = length(peps)
    number_of_trial_parameters = length(Parameters(trial_state))
    Oks = zeros(
        ComplexF64,
        sample_nr,
        number_of_peps_parameters + number_of_trial_parameters,
    )
    Eks = Vector{ComplexF64}(undef, sample_nr)
    logψs = Vector{ComplexF64}(undef, sample_nr)
    samples = Vector{Matrix{Int}}(undef, sample_nr)
    reference_samples = measure_transverse ? zeros(ComplexF64, sample_nr, N) : nothing
    displacement_samples = measure_transverse ?
        zeros(ComplexF64, sample_nr, size(peps)...) : nothing
    monopole_samples = isnothing(monopole_state) ? nothing :
        zeros(ComplexF64, sample_nr, N)
    chirality_up_samples = measure_chirality ?
        zeros(ComplexF64, sample_nr, size(peps)...) : nothing
    chirality_down_samples = measure_chirality ?
        zeros(ComplexF64, sample_nr, size(peps)...) : nothing
    for sample_index in 1:sample_nr
        for _ in 1:moves
            accepted += _fixed_sz_metropolis_move!(
                rng,
                cache,
                weights,
                up_sites,
                down_sites;
                rebuild_after,
                adjacent_sector_cache=adjacent_cache,
                adjacent_sector_anchor=adjacent_anchor,
            )
            attempted += 1
        end
        _fill_fixed_sz_d1_Ok!(
            view(Oks, sample_index, 1:number_of_peps_parameters),
            weights,
            cache.configuration,
        )
        if number_of_trial_parameters > 0
            view(Oks, sample_index, number_of_peps_parameters+1:size(Oks, 2)) .=
                gutzwiller_log_gradient(trial_state, cache)
        end
        Eks[sample_index] = _fixed_sz_local_energy(cache, weights, compiled_hamiltonian)
        logψs[sample_index] = _fixed_sz_log_amplitude(cache, weights)
        samples[sample_index] = reshape(copy(cache.configuration), size(peps))
        if measure_transverse
            _fill_fixed_sz_transverse_correlations!(
                view(reference_samples, sample_index, :),
                view(displacement_samples, sample_index, :, :),
                cache,
                weights,
                size(peps),
                Int(transverse_reference),
            )
        end
        if !isnothing(monopole_state)
            _fill_fixed_sz_monopole_matrix_element!(
                view(monopole_samples, sample_index, :),
                cache,
                adjacent_cache,
                adjacent_anchor[],
                weights,
            )
        end
        if measure_chirality
            _fill_fixed_sz_scalar_chirality!(
                view(chirality_up_samples, sample_index, :, :),
                view(chirality_down_samples, sample_index, :, :),
                cache,
                weights,
                size(peps),
            )
        end
    end

    result = Dict{Symbol,Any}(
        :Oks => Oks,
        :Eks => Eks,
        :logψs => logψs,
        :samples => samples,
        :weights => ones(Float64, sample_nr),
        :contract_dims => ones(Int, sample_nr),
        :acceptance => accepted / attempted,
    )
    if measure_transverse
        result[:Cperp_reference] = dropdims(mean(reference_samples; dims=1); dims=1)
        result[:Cperp_reference_error] = _fixed_sz_block_error(
            reference_samples,
            transverse_block_length,
        )
        result[:Cperp_displacement] = dropdims(mean(displacement_samples; dims=1); dims=1)
        result[:Cperp_displacement_error] = _fixed_sz_block_error(
            displacement_samples,
            transverse_block_length,
        )
    end
    if !isnothing(monopole_state)
        result[:monopole_matrix_element] = dropdims(mean(monopole_samples; dims=1); dims=1)
        real_error = _fixed_sz_block_error(monopole_samples, transverse_block_length)
        imaginary_error = _fixed_sz_block_error(im .* monopole_samples, transverse_block_length)
        result[:monopole_matrix_element_error] = hypot.(real_error, imaginary_error)
    end
    if measure_chirality
        result[:chirality_up] = dropdims(mean(chirality_up_samples; dims=1); dims=1)
        result[:chirality_down] = dropdims(mean(chirality_down_samples; dims=1); dims=1)
        result[:chirality_up_error] = _fixed_sz_block_error(
            chirality_up_samples,
            chirality_block_length,
        )
        result[:chirality_down_error] = _fixed_sz_block_error(
            chirality_down_samples,
            chirality_block_length,
        )
        result[:chirality_up_imaginary_mean] = dropdims(
            mean(imag.(chirality_up_samples); dims=1);
            dims=1,
        )
        result[:chirality_down_imaginary_mean] = dropdims(
            mean(imag.(chirality_down_samples); dims=1);
            dims=1,
        )
    end
    return result
end
