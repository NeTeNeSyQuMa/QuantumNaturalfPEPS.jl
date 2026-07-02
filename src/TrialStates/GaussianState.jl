"""
    GaussianAmplitudeCache

A struct to cache intermediate results for efficient amplitude calculations in Gaussian states. 

# Fields
- `R_mat_full::Matrix{ComplexF64}`: The full R matrix used in the amplitude calculation, with dimensions M_A x N where M_A is the number of occupied sites in the reference state.
- `Q_mat::Matrix{ComplexF64}`: The Q matrix used in the amplitude calculation, with dimensions N x N.
- `parity::Int`: The parity of the reference state, defined as mod(M_A, 2) where M_A is the number of occupied sites in the reference state. This is used for the parity selection rule in amplitude calculations.
- `inv_v_prod::Float64`: The inverse of the product of the absolute values of the Vbar[i-1, i] elements from the Bloch-Messiah decomposition, which is a normalization factor in the amplitude calculation.

"""
struct GaussianAmplitudeCache
    R_mat_full::Matrix{ComplexF64}
    Q_mat::Matrix{ComplexF64}
    parity::Int
    inv_v_prod::Float64
end

"""
    SlaterLogGradientCache

A struct to cache intermediate results for efficient gradient calculations in Slater determinant states.

# Fields
- `dΓs::Vector{Matrix{ComplexF64}}`: A vector of matrices containing the derivatives of the covariance matrix Γ with respect to each variational parameter ηᵢ, defined as dΓᵢ = ∂ηᵢ Γ.

"""
struct SlaterLogGradientCache
    dΓs::Vector{Matrix{ComplexF64}}
end

"""
    SlaterConnection

A struct to store the connection information for the nonzero matrix elements of a quadratic Hamiltonian in the occupation number basis. 

# Fields
- `a::Int`: the first index of the connected pair
- `b::Int`: the second index of the connected pair
- `coeff::Union{ComplexF64, Float64}`: the coefficient `coeff_{a->b}` of the connection, i.e. the matrix element of the Hamiltonian corresponding to the connection.

"""
struct SlaterConnection
    a::Int
    b::Int
    t::Union{ComplexF64, Float64}
    Δ::Union{ComplexF64, Float64}
end

"""
    OccupationProjectorCache

A struct to cache the occupation projector matrix M for a given occupation configuration. 
This is used to save allocations
"""
mutable struct OccupationProjectorCache
    M_j::Matrix{Float64}
end
OccupationProjectorCache(N::Integer) = OccupationProjectorCache(zeros(Float64, 2N, 2N))

"""
    GaussianState

A struct representing a fermionic Gaussian state, defined by its covariance matrix Γ in the Majorana basis and the corresponding Bogoliubov-de Gennes Hamiltonian H_BdG in the Dirac basis (qp-ordered). 
The struct also includes the variational parameters η used to construct H_BdG.

# Fields
- `Γ::Matrix{ComplexF64}`: The covariance matrix of the Gaussian state in the Majorana basis (qq-ordered).
- `H_BdG_func::Function`: The function to construct the Bogoliubov-de Gennes Hamiltonian matrix in the Dirac basis (qp-ordered) that defines the Gaussian state. **Important: This function must follow the column-major order as ITensors uses column-major order.**
- `η::AbstractVector{<:Number}`: The vector of variational parameters used to construct H_BdG, which can be optimized during the training process.
- `N::Int`: The number of sites.
- `parity_sector::Int`: The parity sector of the state, which can be either 0 (even) or 1 (odd).
- `occ_ref::Vector{Int}`: The quasiparticle occupation reference, which is important for selecting the correct Bogoliubov vacuum in the Bloch-Messiah decomposition. 
                        It is constructed from the combination of the `parity_sector` and the `target_state`.
- `slater_loggrad_cache::SlaterLogGradientCache`: A cache for efficient gradient calculations in Slater determinant states, which stores the A matrix and its derivatives with respect to the variational parameters.

"""
mutable struct GaussianState <: AbstractTrialState
    Γ::Matrix{ComplexF64} # covariance matrix in the Majorana basis
    H_BdG_func::Function # function to construct the Bogoliubov-de Gennes Hamiltonian matrix in the Dirac basis (qp-ordered)
    η::AbstractVector{<:Number} # MF parameters
    N::Int # number of sites
    parity_sector::Int # parity sector of the state: either 0 (even) or 1 (odd)
    target_state::Int # ground state (0), first excited state (1) and so on up to the Nth mode
    occ_ref::Vector{Int} # quasiparticle occupation reference. Important for selecting the correct Bogoliubov vacuum in Bloch-Messiah decomposition.
    slater_loggrad_cache::SlaterLogGradientCache # Cache for efficient gradient calculations in Slater determinant states
    amplitude_cache::GaussianAmplitudeCache # Cache for efficient amplitude calculations

    function GaussianState(H_BdG_func::Function, N::Int; η=Float64[], parity_sector::Int=0, target_state::Int=0)
        @assert parity_sector == 0 || parity_sector == 1 "Parity must be either 0 (even) or 1 (odd)"
        Γ, occ_ref = get_Γ_from_H_BdG(H_BdG_func(η, N), parity_sector; target_state=target_state)
        slater_loggrad_cache = build_slater_loggradient_cache(H_BdG_func, η, N; parity_sector=parity_sector, target_state=target_state)
        amplitude_cache = build_amplitude_cache(H_BdG_func(η, N), parity_sector, occ_ref)
        new(Γ, H_BdG_func, η, N, parity_sector, target_state, occ_ref, slater_loggrad_cache, amplitude_cache)
    end
end
getParity(GS::GaussianState) = Int(sign(real(pfaffian(2 * GS.Γ)))) == 1 ? 0 : 1
getParity(Γ::AbstractMatrix) = Int(sign(real(pfaffian(2 * Γ)))) == 1 ? 0 : 1

Parameters(GS::GaussianState) = GS.η

"""
    write!(GS::GaussianState, η::AbstractVector{<:Number})

Updates the variational parameters `η` of the Gaussian state `GS` and recompute the covariance matrix `Γ` and the Slater log-gradient cache accordingly.

"""
function write!(GS::GaussianState, η::AbstractVector{<:Number})
    GS.η = η
    GS.Γ, GS.occ_ref = get_Γ_from_H_BdG(GS.H_BdG_func(η, GS.N), GS.parity_sector; target_state=GS.target_state)
    GS.slater_loggrad_cache = build_slater_loggradient_cache(GS.H_BdG_func, η, GS.N; parity_sector=GS.parity_sector, target_state=GS.target_state)
    GS.amplitude_cache = build_amplitude_cache(GS.H_BdG_func(η, GS.N), GS.parity_sector, GS.occ_ref)
end

###########################################################################################################

"""
    build_general_H_BdG_2D_NN(η::AbstractVector{<:Number}, Lx::Int, Ly::Int)

Function that builds a general mean-field Bogoliubov-de Gennes Hamiltonian matrix with nearest-neighbor (NN) hopping and pairing terms.
The overall number of mean-field parameters is maximal, so we have:
- `Lx * Ly` parameters for the chemical potential μ at each site
- `Lx * (Ly - 1)` hopping / pairing in x direction
- `Ly * (Lx - 1)` hopping / pairing in y direction

In total this gives us `5 * Lx * Ly - 2 * (Lx + Ly)` parameters, which is the maximal number of parameters for a quadratic Hamiltonian with only NN terms on a 2D lattice.

The ordering of the vector of variational parameters `η` is as follows:
- `η[1 : Lx*Ly]`: chemical potential μ for each site
- `η[Lx*Ly + 1 : Lx*Ly + Ly*(Lx-1)]`: hopping in x-direction
- `η[Lx*Ly + Ly*(Lx-1) + 1 : Lx*Ly + Ly*(Lx-1) + Lx*(Ly-1)]`: hopping in y-direction
- `η[Lx*Ly + Ly*(Lx-1) + Lx*(Ly-1) + 1 : Lx*Ly + Ly*(Lx-1) + Lx*(Ly-1) + Ly*(Lx-1)]`: pairing in x-direction
- `η[Lx*Ly + Ly*(Lx-1) + Lx*(Ly-1) + Ly*(Lx-1) + 1 : end]`: pairing in y-direction

The ordering of the sites is as follows and follows the column-major ordering of ITensors:
```
    --→ y       1 -- 4 -- 7
    |           |    |    |
    ↓           2 -- 5 -- 8
    x           |    |    |
                3 -- 6 -- 9
```

# Returns
- `H_BdG::Hermitian`: The Bogoliubov-de Gennes Hamiltonian matrix `H_BdG = [T D; D' -T']`

"""
function build_general_H_BdG_2D_NN(η::AbstractVector{<:Number}, Lx::Int, Ly::Int)
    N = Lx * Ly

    @assert length(η) == get_max_num_MF_params_NN(Lx, Ly) "Length of η ($(length(η))) must be equal to the maximal number of mean-field parameters ($(get_max_num_MF_params(Lx, Ly))) for the given lattice size for this constructor"

    μs = η[1:N]
    hopping_x = η[N + 1 : N + Ly * (Lx - 1)]
    hopping_y = η[N + Ly * (Lx - 1) + 1 : N + Ly * (Lx - 1) + Lx * (Ly - 1)]
    pairing_x = η[N + Ly * (Lx - 1) + Lx * (Ly - 1) + 1 : N + Ly * (Lx - 1) + Lx * (Ly - 1) + Ly * (Lx - 1)]
    pairing_y = η[N + Ly * (Lx - 1) + Lx * (Ly - 1) + Ly * (Lx - 1) + 1 : end]

    # Remove the boundary-crossing terms (every L-th bond)
    hopping_x = [((i % Lx == 0) ? 0.0 : hopping_x[i - div(i, Lx)]) for i in 1:(N-1)]
    pairing_x = [((i % Lx == 0) ? 0.0 : pairing_x[i - div(i, Lx)]) for i in 1:(N-1)]

    # Constructing T and D blocks
    T = diagm( 
        0 => μs,
        1 => hopping_x,
        Lx => hopping_y,
        -1 => conj.(hopping_x),
        -Lx => conj.(hopping_y)
    )

    D = diagm(
        1 => -pairing_x,
        Lx => -pairing_y,
        -1 => pairing_x,
        -Lx => pairing_y
    )

    H_BdG = Matrix{eltype(η)}([T D; D' -transpose(T)])
    return Hermitian(H_BdG)
end
build_general_H_BdG_2D_NN(η::AbstractVector{<:Number}, N::Int) = build_general_H_BdG_2D_NN(η::AbstractVector{<:Number}, Int(sqrt(N)), Int(sqrt(N)))

# helper functions for maximal number of parameters for 2D NN Hamiltonian
get_max_num_MF_params_NN(Lx::Int, Ly::Int) = 5 * Lx * Ly - 2 * (Lx + Ly)
get_max_num_hopping_x_NN(Lx::Int, Ly::Int) = Ly * (Lx - 1)
get_max_num_hopping_y_NN(Lx::Int, Ly::Int) = Lx * (Ly - 1)

"""
    build_H_BdG_derivatives(H_BdG_func::Function, η::AbstractVector{<:Number}, N::Int)

Builds the derivative matrices dH/dη for the BdG Hamiltonian using automatic differentiation. 
Returns a vector of matrices corresponding to the derivatives with respect to each variational parameter in `η`

"""
function build_H_BdG_derivatives(H_BdG_func::Function, η::AbstractVector{<:Number}, N::Int)
    dimH = 2 * N
    n_entries = dimH * dimH
    
    T = eltype(H_BdG_func(η, N))

    dHs = Vector{Matrix{T}}(undef, length(η))
    if T <: AbstractFloat
        f = θ -> vec(Matrix(H_BdG_func(θ, N)))
        J = Zygote.jacobian(f, η)[1]  # (2*dimH^2) x length(η)

        for a in eachindex(η)
            dH_real = reshape(@view(J[1:n_entries, a]), dimH, dimH)
            dHs[a] = dH_real
        end
    elseif T <: Complex
        η_reim = vcat(real.(η), imag.(η))

        function f_reim(x)
            n = length(x) ÷ 2
            θ = ComplexF64.(x[1:n] .+ im .* x[n+1:end])

            H_vec = vec(Matrix(H_BdG_func(θ, N)))

            return vcat(real.(H_vec), imag.(H_vec))
        end

        J = Zygote.jacobian(f_reim, η_reim)[1]

        nη = length(η)

        for a in 1:nη
            # derivative wrt Re η_a
            dH_re = reshape(@view(J[1:n_entries, a]), dimH, dimH) .+
                    im .* reshape(@view(J[n_entries+1:2n_entries, a]), dimH, dimH)

            # derivative wrt Im η_a
            dH_im = reshape(@view(J[1:n_entries, nη+a]), dimH, dimH) .+
                    im .* reshape(@view(J[n_entries+1:2n_entries, nη+a]), dimH, dimH)

            # Correct Wirtinger derivative
            dH = 0.5 .* (dH_re .- im .* dH_im)

            # enforce Hermiticity only numerically
            # dH = 0.5 .* (dH + dH')

            dHs[a] = dH
        end
    end
    
    return dHs
end
function build_H_BdG_derivatives(GS::GaussianState)
    return build_H_BdG_derivatives(GS.H_BdG_func, GS.η, GS.N)
end

"""
    to_occ_dict(occ_string::Vector{Int})

Convert a vector of occupation numbers (0 or 1) into a dictionary mapping site indices to occupation numbers.

For example, `occ_string = [0, 1, 0]` would be converted to `Dict(1 => 0, 2 => 1, 3 => 0)`.

"""
function to_occ_dict(occ_string::Vector{Int})
    occ_dict = Dict{Int, Int}()
    for i in 1:length(occ_string)
        occ_dict[i] = occ_string[i]
    end
    return occ_dict
end

"""
    get_prob(GS::GaussianState, occ_dict::Dict{Int, Int})

Calculate the probability of observing a specific occupation configuration `occ_dict` in the Gaussian state `GS`. 

The probability is computed using the covariance matrix `Γ` and the occupation projector matrix `M` corresponding to the specified occupation configuration. 

The formula uses Eq. (20) from Bravyi (https://arxiv.org/abs/quant-ph/0404180):

(TODO: Add our future paper maybe)

```
P(s) = |<Ψ(s)|Ψ(s)>|² = <Ψ| P̂ |Ψ> = Tr(ρ * P̂) 
     = (0.5^N_measured) * sqrt(abs(det(Γ * 2 * M - I)))
```
where `P̂` is the occupation projector defined by `occ_dict`.

# Keyword Arguments
- `GS::GaussianState`: The Gaussian state for which to compute the probability.
- `occ_dict::Dict{Int, Int}`: A dictionary mapping site indices to occupation numbers (e.g. Dict(2 => 0) means site 2 is unoccupied) for the sites being measured. 
                              This must not contain all sites, as one can also construct joint probabilities with only specifying a subset of sites.

# Returns
- `Float64`: The joint probability amplitude `P(s)`, where `s ∈ (s1,s2,...,sN)` is a subset of occupation configurations of all `N` sites.

"""
function get_prob(GS::GaussianState, occ_dict::Dict{Int, Int})
    N_sites = size(GS.Γ, 1) ÷ 2
    N_measured = length(occ_dict) # Number of sites actually projected
    M = build_occ_projector_matrix(occ_dict, N_sites)
    return _get_prob_from_projector_matrix!(GS, M, N_measured)
end

"""
    get_prob(GS::GaussianState, occ_string::Vector{Int})

# Keyword Arguments
- `GS::GaussianState`: The Gaussian state for which to compute the probability.
- `occ_string::Vector{Int}`: A vector of occupation numbers (0 or 1) for each site, of length `N`.

# Returns
- `Float64`: The full probability amplitude `P(s1,s2,...,sN)`, where `(s1,s2,...,sN)` is the occupation configuration of all `N` sites.

"""
function get_prob(GS::GaussianState, occ_string::Vector{Int})
    return get_prob(GS, to_occ_dict(occ_string))
end

"""
    Returns: (0.5^N_measured) * sqrt(abs(det(Γ * 2 * M - I)))
"""
function _get_prob_from_projector_matrix!(GS::GaussianState, M::AbstractMatrix, N_measured::Int)
    A = similar(GS.Γ)

    # This computes A = Γ * 2 * M - I efficiently
    mul!(A, GS.Γ, M)
    @inbounds for idx in eachindex(A)
        A[idx] *= 2
    end
    @inbounds for i in axes(A, 1)
        A[i, i] -= 1
    end
    return (0.5^N_measured) * sqrt(abs(det(A)))
end

"""
    get_prob(GS::GaussianState, occ_string::Vector{Int}, n_measured::Int)

Returns the probability of observing the occupation configuration specified by `occ_string` for the first `n_measured` sites in the Gaussian state `GS`.
This is used for sequential sampling, where one measures sites one by one and conditions on the previously measured sites.

"""
function get_prob(GS::GaussianState, occ_string::Vector{Int}, n_measured::Int)
    N_sites = size(GS.Γ, 1) ÷ 2
    @assert 0 <= n_measured <= N_sites

    M_j = zeros(Float64, 2N_sites, 2N_sites)
    build_occ_projector_matrix!(M_j, occ_string, n_measured)
    return _get_prob_from_projector_matrix!(GS, M_j, n_measured)
end
function get_prob!(GS::GaussianState, M_cache::OccupationProjectorCache, n_measured::Int)
    return _get_prob_from_projector_matrix!(GS, M_cache.M_j, n_measured)
end

"""
    build_occ_projector_matrix(occ_dict::Dict{Int, Int}, N::Int)

Builds the matrix M for the occupation projector `P_j(θ) ∼ exp(i/2 θᵀ M θ)`, where `M` is a block-diagonal matrix with 2x2 blocks corresponding to the sites in `occ_dict`. Each block is defined as:
```
[ 0  s_j]
[-s_j  0]
```
where `s_j = 1 - 2*n_j` and `n_j` is the occupation number for site `j`. The resulting matrix `M` has dimensions `2N x 2N`, where `N` is the total number of sites. 

Note: This is already in the Majorana basis and uses the convention:
```
c_j  = 1/2 (γ_2j-1 + iγ_2j)
c†_j = 1/2 (γ_2j-1 - iγ_2j)
```
where `γ_2j-1` and `γ_2j` are the Majorana operators corresponding to site `j`.

"""
function build_occ_projector_matrix(occ_dict::Dict{Int, Int}, N::Int)
    M_j = zeros(Int8, 2N, 2N)

    for (site_colmajor, n_j) in occ_dict
        s_j = 1 - 2*n_j
        row = 2*site_colmajor - 1
        M_j[row, row + 1] = s_j
        M_j[row + 1, row] = -s_j
    end

    return M_j
end

"""
    build_occ_projector_matrix!(M_j::AbstractMatrix, occ_string::AbstractVector{Int}, n_measured::Int)

Builds the occupation projector matrix M in-place for the first `n_measured` sites based on the occupation configuration specified in `occ_string`.
This is used for sequential sampling, where one measures sites one by one and conditions on the previously measured sites.
"""
function build_occ_projector_matrix!(M_j::AbstractMatrix, occ_string::AbstractVector{Int}, n_measured::Int)
    fill!(M_j, 0)
    @inbounds for site_colmajor in 1:n_measured
        set_occ_projector_block!(M_j, site_colmajor, occ_string[site_colmajor])
    end
    return M_j
end

function set_occ_projector_block!(M_j::AbstractMatrix, site_colmajor::Int, n_j::Int)
    s_j = 1 - 2*n_j
    row = 2*site_colmajor - 1
    M_j[row, row + 1] = s_j
    M_j[row + 1, row] = -s_j
    return M_j
end

"""
    get_amplitude(cache::GaussianAmplitudeCache, occ_string::Vector{Int})

Returns the amplitude `⟨s|ψ⟩` for a fermionic Gaussian state |Ψ>.
It uses the overlap formula from: http://arxiv.org/abs/2111.09101 and https://link.aps.org/doi/10.1103/PhysRevB.107.125128

# Keyword Arguments
- `cache::GaussianAmplitudeCache`: The preconstructed cache of intermediate matrices.
- `occ_string::Vector{Int}`: The configuration `s`. 
                            A vector of occupation numbers (0 or 1) for each site, of length `N`. 
                            The ordering of the occupation string is: `[n_1, n_2, ..., n_N]` where `n_j` is the occupation number for site `j`.

# Returns
- `ComplexF64`: The amplitude `⟨s|ψ⟩`, where `s=(s1,s2,...,sN)` is the occupation configuration of all `N` sites.

"""
function get_amplitude(cache::GaussianAmplitudeCache, occ_string::Vector{Int})
    # Boolean occupation vector to select rows from R_mat_full (true if occupied)
    occ_bool = occ_string .== 1
    M_prime = sum(occ_bool)

    # parity selection rule: if M and M' have different parity, overlap vanishes
    if mod(M_prime, 2) != cache.parity
        return zero(eltype(cache.Q_mat))
    end

    if M_prime != 0
        R_mat = cache.R_mat_full[occ_bool, :]

        fsign = isodd((M_prime * (M_prime - 1)) ÷ 2) ? -1 : 1 # fermionic sign from reordering
        pf = pfaffian([zeros(ComplexF64, M_prime, M_prime) R_mat; -transpose(R_mat) cache.Q_mat])

        return fsign * pf * cache.inv_v_prod
    end

    return pfaffian(cache.Q_mat) * cache.inv_v_prod
end
function get_amplitude(GS::GaussianState, occ_string::Vector{Int})
    return get_amplitude(GS.amplitude_cache, occ_string)
end
# function get_amplitude(H_BdG::Hermitian, occ_string::Vector{Int})
#     return get_amplitude(build_amplitude_cache(H_BdG), occ_string)
# end

"""
    build_amplitude_cache(H_BdG::Hermitian)

Preconstructs the matrices and factors needed for efficient amplitude calculations in Gaussian states via `get_amplitude(cache::GaussianAmplitudeCache, occ_string::Vector{Int})`.

"""
function build_amplitude_cache(H_BdG::Hermitian, parity::Int, occ_ref::Vector{Int})
    _, M = bogoliubov(H_BdG)

    #  select the occupied modes from M based on the reference (Gaussian) state
    N = size(H_BdG, 1) ÷ 2
    M = M[:, vcat([occ_ref[k] == 1 ? N + k : k for k in 1:N], [occ_ref[k] == 1 ? k : N + k for k in 1:N])]

    # Bloch-Messiah decomposition
    Dmat, UVmat, Cmat = bloch_messiah_decomposition(M)

    Dmat_prime, UVmat_prime, Cmat_prime = truncated_bloch_messiah(Dmat, UVmat, Cmat)

    D, Ubar, Vbar, _ = get_mats_from_bloch_messiah(Dmat_prime, UVmat_prime, Cmat_prime)

    # Vbar_trunc has the structure [ I 0; 0 ⨁_p (i v_p σ_y)] so we need to skip the identity block
    vp_prod_start_ind = findlast(x -> abs(x) ≈ 1.0, diag(Vbar))
    vp_prod_start_ind = vp_prod_start_ind === nothing ? 2 : vp_prod_start_ind + 2
    v_prod = prod([Vbar[i-1, i] for i in vp_prod_start_ind:2:size(Vbar, 2)])

    # compute full matrices for overlap
    R_mat_full = D * Vbar # has the same ordering as H
    Q_mat = Ubar * Vbar   # has the same ordering as H
    Q_mat = (Q_mat - transpose(Q_mat)) / 2 # enforce exact skew-symmetry

    return GaussianAmplitudeCache(R_mat_full, Q_mat, parity, 1 / v_prod)
end
build_amplitude_cache(GS::GaussianState) = build_amplitude_cache(GS.H_BdG_func(GS.η, GS.N), GS.parity_sector, GS.occ_ref)

"""
    get_Slater_Ek_terms(H_BdG::Hermitian)

Given a BdG Hamiltonian, returns the terms that contribute to the local energy for a given sample. 
For quadratic Hamiltonians, only configurations differing by at most two occupations can contribute as one only as terms:
- c_i^† c_j     (hopping j -> i)
- c_i c_j^†     (hopping i -> j)
- c_i c_j       (pairing i,j)
- c_i^† c_j^†   (pairing i,j)

# Returns
- A Dictionary of `SlaterConnection` objects representing the nonzero matrix elements of the Hamiltonian.

"""
function get_Slater_Ek_terms(H_BdG::Hermitian)
    N = size(H_BdG, 1) ÷ 2
    T = H_BdG[1:N, 1:N]
    D = H_BdG[1:N, N+1:end]

    connections = Dict{Tuple{Int, Int}, SlaterConnection}()

    for i in 1:N, j in i:N
        t = T[i, j]
        Δ = D[i, j]

        # Recall t_ij = t_ji* and Δ_ij = -Δ_ji
        if !iszero(t) || !iszero(Δ)
            connections[(i, j)] = SlaterConnection(i, j, t, Δ)
        end
    end

    return connections
end

"""
    build_slater_loggradient_cache(
        GS::GaussianState;
        dH_dη=nothing, 
        N::Int;
        parity_sector::Int=0,
        target_state::Int=0
    )


Computes and stores dΓ which is used for the analytic gradient of the log-amplitude:

```
    Oⱼ,ᵢ(η) = ∂η ln(Sⱼ(η)) = 0.25 * Tr[Fⱼ⁻¹ Γ⁻¹ (∂ηᵢ Γ) Γ⁻¹]
```
where `Fⱼ = Mⱼ - Γ⁻¹` and `Mⱼ` is the matrix for the occupation projector to the configuration `j`.

# Steps:

1. Get `dΓ`:
    - As for fermionic Gaussian states ( ground and thermal states of fermionic quadratic Hamiltonians ), the covariance matrix `Γ` commutes with the Hamiltonian -> `[H, Γ] = 0`, we can use this property to derive an equation for `dΓ`:
    ```
        dη [H, Γ] = [dH, Γ] + [H, dΓ] = 0
        => [H, dΓ] = -[dH, Γ]
    ```
    - This is a Sylvester equation of the form `A X + X B + C = 0` with `A = H`, `B = -H`, `C = -[dH, Γ]` and `X = dΓ`, which can be solved efficiently with `LinearAlgebra.sylvester`.
"""
function build_slater_loggradient_cache(
    H_BdG_func::Function, 
    η::AbstractVector{<:Number}, 
    N::Int;
    parity_sector::Int=0,
    target_state::Int=0
)
    H = Matrix(H_BdG_func(η, N))
    # as Γ is in the Majorana basis (qq), we need to transform H to the same basis
    H_maj = transform_H_to_majorana_qq(H)
    dim = size(H_maj, 1)
    Γ, _ = get_Γ_from_H_BdG(Hermitian(H), parity_sector; target_state=target_state)

    dHs = build_H_BdG_derivatives(H_BdG_func, η, N)
    dΓs = Vector{Matrix{ComplexF64}}(undef, length(dHs))
    # =========================================================================
    # Solve commutator equation: [H, dΓ] = -[dH, Γ] to get dΓ
    #  => H dΓ - dΓ H = -[dH, Γ]
    # 
    # We can use the Sylvester equation for that:
    # X = sylvester(A, B, C), solves AX + XB + C = 0
    # with A = H, B = -H, C = -[dH, Γ], such that X = dΓ
    # =========================================================================
    I_dim = Matrix{ComplexF64}(I, dim, dim)

    # Optional regularization near gap closings: H dΓ - dΓ (H + reg) + C = 0 to make it more robust
    reg = 1e-10
    B = -H_maj + reg .* I_dim
    for a in eachindex(dHs)
        dH = Matrix(dHs[a])
        dH_maj = transform_H_to_majorana_qq(dH)

        C = (dH_maj * Γ - Γ * dH_maj) # +[dH, Γ]

        dΓ = LinearAlgebra.sylvester(H_maj, B, C) # dΓ given by the solution to the Sylvester equation
        dΓ = 0.5 .* (dΓ .- transpose(dΓ)) # Enforce exact skew-symmetry: Γᵀ = -Γ

        # Remove tiny numerical noise
        dΓ[abs.(dΓ) .< 1e-14] .= 0.0
        dΓs[a] = dΓ
    end

    return SlaterLogGradientCache(dΓs)
end

function build_slater_loggradient_cache(GS::GaussianState)
    return build_slater_loggradient_cache(GS.H_BdG_func, GS.η, GS.N; parity_sector=GS.parity_sector, target_state=GS.target_state)
end

"""
    get_Γ_from_H_BdG(H_BdG::Hermitian, occ_string::Vector{Int})

Given a Bogoliubov-de Gennes Hamiltonian matrix `H_BdG` and an occupation string `occ_string`, this function computes the correlation matrix Γ in the Majorana basis (qq-ordered). 

# Keyword Arguments
- `H_BdG::Hermitian`: The Bogoliubov-de Gennes Hamiltonian matrix `H_BdG = [T D; D† -Tᵀ]` (qp-ordered) of size `2N x 2N`.
- `occ_string::Vector{Int}`: A vector of occupation numbers (0 or 1) for each site, of length `N`.

"""
function get_Γ_from_H_BdG(H_BdG::Hermitian, parity_sector::Int; target_state::Int=0)
    N = size(H_BdG, 1) ÷ 2
    @assert target_state >= 0 && target_state <= N "target_state must be between 0 (ground state) and N=$(N) (fully excited state)"

    # Diagonalize the BdG Hamiltonian with the Bogoliubov transformation M
    _, M = bogoliubov(H_BdG)

    # Construct the Correlation matrix in the Dirac basis (diagonal, quasiparticles) (qp-ordered)
    parity_vac = getParity(get_Γ0_from_H_BdG(H_BdG))
    nfill = ((parity_sector + parity_vac) % 2) + 2 * target_state # fill correct number of modes depending on the parity_sector and the parity of the ground state for the current M
    @assert nfill <= N "The parity sector and target state are incompatible for the given system size N=$(N). Please choose a different target_state or parity_sector."
    hole_occ = ones(Int, N)
    if nfill > 0
        @views hole_occ[(N - nfill + 1):N] .= 0
    end
    particle_occ = 1 .- hole_occ
    G_diag_dirac = Diagonal(vcat(particle_occ, hole_occ))

    # Transform G to the original basis using the Bogoliubov transformation M (qp-ordered)
    G_dirac = M * G_diag_dirac * M' 

    # bring to qq-ordering
    perm = begin
        p = zeros(Int, 2N)
        p[1:2:2N] = 1:N
        p[2:2:2N] = N+1:2N
        p
    end
    G_dirac = G_dirac[perm, perm]

    # Transform G from the Dirac basis to the Majorana basis (qq-ordered) using the transformation matrix Ω
    Ω0 = [1 1; im -im] ./ sqrt(2)
    Ω = kron(I(N), Ω0) # Extend to all sites 

    G_majorana = Ω * G_dirac * Ω'

    # the covariance matrix is then obtained by 
    Γ_majorana = ( -im .* (2G_majorana - I)) ./ 2

    @assert Γ_majorana ≈ -transpose(Γ_majorana) "Γ is not skew-symmetric!"

    return (Γ_majorana - transpose(Γ_majorana)) / 2, particle_occ # we symmetrize to avoid numerical issues
end

function transform_H_to_majorana_qq(H_BdG::AbstractMatrix)
    N = size(H_BdG, 1) ÷ 2

    # bring to qq-ordering
    perm = begin
        p = zeros(Int, 2N)
        p[1:2:2N] = 1:N
        p[2:2:2N] = N+1:2N
        p
    end
    H_qq = H_BdG[perm, perm]

    # Transform H from the Dirac basis to the Majorana basis (qq-ordered) using the transformation matrix Ω
    Ω0 = [1 1; im -im] ./ sqrt(2)
    Ω = kron(I(N), Ω0) # Extend to all sites 

    H_majorana = Ω * H_qq * Ω'

    # Majorana generator
    h = 2 .* imag(H_majorana)

    return real.(h)
end

function get_Γ0_from_H_BdG(H_BdG::Hermitian)
    N = size(H_BdG, 1) ÷ 2

    # Diagonalize the BdG Hamiltonian with the Bogoliubov transformation M
    _, M = bogoliubov(H_BdG)

    G_diag_dirac = build_G_diag_dirac(zeros(Int, N)) # vacuum state

    # Transform G to the original basis using the Bogoliubov transformation M (qp-ordered)
    G_dirac = M * G_diag_dirac * M' 

    # bring to qq-ordering
    perm = begin
        p = zeros(Int, 2N)
        p[1:2:2N] = 1:N
        p[2:2:2N] = N+1:2N
        p
    end
    G_dirac = G_dirac[perm, perm]

    # Transform G from the Dirac basis to the Majorana basis (qq-ordered) using the transformation matrix Ω
    Ω0 = [1 1; im -im] ./ sqrt(2)
    Ω = kron(I(N), Ω0) # Extend to all sites 

    G_majorana = Ω * G_dirac * Ω'

    # the covariance matrix is then obtained by 
    Γ_majorana = ( -im .* (2G_majorana - I)) ./ 2

    @assert Γ_majorana ≈ -transpose(Γ_majorana) "Γ is not skew-symmetric!"

    return (Γ_majorana - transpose(Γ_majorana)) / 2 # we symmetrize to avoid numerical issues
end

"""
    build_G_diag_dirac(occ_string::Vector{Int})

Builds the diagonal correlation matrix G matrix in the Dirac basis (qp-ordered) for a given occupation string.

"""
function build_G_diag_dirac(occ_string::Vector{Int})
    particle_occ = occ_string
    hole_occ = 1 .- particle_occ

    return Diagonal([particle_occ; hole_occ])
end

"""
    get_bogoliubov_blocks(M::AbstractMatrix)

Extract the `U` and `V` blocks from the Bogoliubov transformation matrix `M = [U conj(V); V conj(U)]`.
"""
function get_bogoliubov_blocks(M::AbstractMatrix)
    N = div(size(M, 1), 2)
    U = M[1:N, 1:N]
    V = M[N+1:end, 1:N]
    return U, V
end

"""
    bogoliubov(H::Hermitian)

Return the spectrum and canonical transform that diagonalize the fermionic quadratic Hamiltonian `H`.

The Bogoliubov matrix `M = [U conj(V); V conj(U)]` satisfies `M' * H * M == diagm(vcat(E, -E))`, and the
returned vector `E` contains the positive eigenvalues in descending order.

Degenerate spectra are handled robustly by splitting the modes into two groups:

- **Nonzero modes** (`|E| > zero_tol`). Their `+E` / `-E` eigenspaces are distinct, but a near-degenerate
  `±E` pair at very small `|E|` is mixed by the eigensolver and would break the particle-hole pairing.
  We therefore orthogonalize `[X  C(X)]` through its SVD (polar) factor, which absorbs that mixing as a
  subspace rotation and restores the canonical structure.
- **Exact zero modes** (`|E| <= zero_tol`). Here `+E` and `-E` coincide, so `[X  C(X)]` becomes rank
  deficient and the SVD step is ill-defined. The zero-mode subspace is invariant under particle-hole
  conjugation, so we instead rebuild a particle-hole symmetric (Majorana) basis of it and pair the
  Majoranas into fermions.

This guarantees the canonical (anti)commutation relations by construction, so the resulting `M` is always a
valid Bogoliubov transformation and is well-conditioned for the subsequent Bloch-Messiah decomposition.
"""
function bogoliubov(H::Hermitian; tol=1e-8, zero_tol=1e-9)
    N = div(size(H, 1), 2)

    # Particle-hole conjugation C: [X_u; X_v] -> [conj(X_v); conj(X_u)]. C is the antiunitary
    # symmetry of the BdG Hamiltonian (C H C⁻¹ = -H): it maps an eigenvector at energy +E to
    # one at -E. Hence M = [X  C(X)], with X the N quasiparticle (creation) vectors built from
    # the non-negative part of the spectrum.
    _ph_conj(X) = vcat(conj.(X[N+1:end, :]), conj.(X[1:N, :]))

    E0, M0 = eigen(H)

    # Split the spectrum into strictly positive modes and exact zero modes (|E| <= zero_tol).
    pos_idx = findall(>(zero_tol), E0)
    pos_idx = pos_idx[sortperm(E0[pos_idx]; rev=true)] # descending energy
    zero_idx = findall(x -> abs(x) <= zero_tol, E0)

    n_zero_pairs = N - length(pos_idx)
    @assert 2 * n_zero_pairs == length(zero_idx) "Zero-mode subspace has odd dimension ($(length(zero_idx))); adjust zero_tol=$zero_tol."

    # Nonzero modes: orthogonalize [X  C(X)] via its SVD (polar) factor. This is full rank away
    # from exact zero modes and robustly repairs near-degenerate ±E pairs that the eigensolver mixed.
    X = M0[:, pos_idx]
    if !isempty(pos_idx)
        F = svd(hcat(X, _ph_conj(X)))
        X = (F.U * F.V')[:, 1:length(pos_idx)]
    end

    # Exact zero modes: the E = 0 eigenspace is mapped onto itself by C and makes [X  C(X)] rank
    # deficient, so we rebuild a particle-hole symmetric (Majorana) basis and pair them into fermions.
    if n_zero_pairs > 0
        X = hcat(X, _zero_mode_fermions(M0[:, zero_idx], _ph_conj, n_zero_pairs; tol=tol))
    end

    M = hcat(X, _ph_conj(X))

    U = M[1:N, 1:N]
    V = M[N+1:end, 1:N]

    # E = diag(M' H M) so that E[k] = -E[k+N] exactly.
    E = real.(diag(M' * H * M))

    @assert isapprox(M' * M, I, atol=tol) "Bogoliubov M is not unitary."
    @assert isapprox(U'U + V'V, I, atol=tol) "Bogoliubov blocks violate U'U + V'V = I."
    @assert isapprox(transpose(U) * V + transpose(V) * U, zeros(N, N), atol=tol) "Bogoliubov blocks violate UᵀV + VᵀU = 0."

    return E, M
end

"""
    _zero_mode_fermions(Z, _ph_conj, n_pairs; tol=1e-7)

Build `n_pairs` fermionic zero-mode creation vectors from the `2*n_pairs` orthonormal zero-energy
eigenvectors stored as columns of `Z`. The zero-mode subspace is invariant under particle-hole
conjugation `C`, so we first construct a particle-hole symmetric (Majorana) basis `ω` satisfying
`C(ω) = ω`. The Majorana condition `A·conj(w) = w` is the `+1` eigenspace of the antiunitary involution
`T(w) = A·conj(w)`; representing `T` as a real symmetric involution on `(Re w, Im w)` lets us obtain that
basis exactly (to machine precision) and robustly via a single real symmetric eigendecomposition. The
Majoranas are then paired into complex fermions `c† = (γ₁ + i γ₂)/√2`, whose columns, together with their
`C`-images, satisfy the CAR exactly.
"""
function _zero_mode_fermions(Z::AbstractMatrix, _ph_conj, n_pairs::Int; tol=1e-7)
    # Re-orthonormalize the zero-mode eigenvectors. For a degenerate eigenvalue cluster (all the
    # zero modes share E = 0), LAPACK's MRRR driver (syevr, used by `eigen` for real-symmetric
    # matrices) can return eigenvectors that span the correct subspace but are not mutually
    # orthonormal — by as much as ~1e-4 on some BLAS builds (e.g. OpenBLAS on Linux), while being
    # orthonormal to machine precision on others (Windows). Everything below assumes Z'Z = I (the
    # null space is exactly particle-hole invariant, so A is unitary only for orthonormal Z). A QR
    # orthonormalization preserves the span (Z stays in the null space) and makes the construction
    # platform independent. Without it the PH-unitarity assertion below fails only on Linux.
    Z = Matrix(qr(Z).Q)
    # Particle-hole operator restricted to the zero-mode subspace (in the Z basis): a vector
    # ω with coordinates w is Majorana (C-real) iff A * conj(w) = w. A is symmetric unitary,
    # so T(w) = A * conj(w) is an antiunitary involution (T² = I).
    A = Z' * _ph_conj(Z)
    A = (A + transpose(A)) / 2 # enforce the exact symmetry expected of a PH involution
    @assert isapprox(A' * A, I, atol=tol) "Particle-hole operator is not unitary on the zero-mode subspace."

    dim = size(A, 1) # = 2 * n_pairs
    # Real representation of T on (Re w, Im w): writing w = wr + i·wi and A = Ar + i·Ai,
    # T(w) = (Ar·wr + Ai·wi) + i·(Ai·wr − Ar·wi), i.e. the real symmetric involution below.
    # Its +1 eigenspace (dimension `dim`) is exactly the orthonormal Majorana (C-real) basis.
    Ar, Ai = real.(A), imag.(A)
    Tmat = Symmetric([Ar Ai; Ai -Ar])
    F = eigen(Tmat)
    plus = findall(>(0), F.values) # eigenvalues are ±1; keep the C-real (+1) subspace
    @assert length(plus) == dim "Zero-mode real structure has wrong +1 multiplicity ($(length(plus)) vs $dim)."
    R = F.vectors[:, plus] # (2·dim) × dim, orthonormal real columns
    B = R[1:dim, :] .+ im .* R[dim+1:end, :] # orthonormal Majorana modes (columns, in Z basis)

    Zm = Z * B # Majorana zero modes in the original 2N-dimensional space

    # Pair consecutive Majoranas into fermionic creation operators c† = (γ₁ + i γ₂)/√2.
    X0 = Matrix{ComplexF64}(undef, size(Z, 1), n_pairs)
    for j in 1:n_pairs
        @views X0[:, j] .= (Zm[:, 2j-1] .+ im .* Zm[:, 2j]) ./ sqrt(2)
    end
    return X0
end

"""
    skew_canonical_form(P::AbstractMatrix)

Return a pair `(S, X)` where `X = transpose(S)*P*S` is the canonical form for `P` (See: https://doi.org/10.1007/BF02906230).
"""
function skew_canonical_form(P::AbstractMatrix)
    # Check skew-symmetry
    @assert isapprox(transpose(P), -P; atol=1e-10) "P should be skew-symmetric"

    W = P'P
    @assert ishermitian(W)

    E, Φ = eigen(Hermitian(W); sortby = (x -> -real(x)))
    alphas = sqrt.(abs.(E))
    tol = 1e-7

    # sort indices by magnitude descending to make pairing stable
    idx_sorted = sortperm(alphas, rev = true)
    nonzero_idx = [i for i in idx_sorted if !isapprox(alphas[i], 0.0; atol=tol)]
    zero_idx = [i for i in idx_sorted if isapprox(alphas[i], 0.0; atol=tol)]

    # ensure we have an even number of nonzero modes (otherwise pairing impossible)
    if isodd(length(nonzero_idx))
        error("skew_canonical_form: odd number of nonzero canonical values (check tolerance).")
    end

    # Initialize with zeros to safely allow projections
    S = zeros(eltype(P), size(P))
    pos = 1

    # build paired columns using Gram-Schmidt to safely handle degeneracies
    for i in nonzero_idx
        v = copy(Φ[:, i])
        
        # Project out all previously established basis vectors in S
        for prev in 1:(pos-1)
            v -= S[:, prev] * (S[:, prev]' * v)
        end
        
        # If the vector is fully spanned by previous pairs, skip it
        if norm(v) < tol
            continue
        end
        
        v1 = v / norm(v)
        v2 = (P' * conj(v1)) / alphas[i]
        
        S[:, pos]   = v1
        S[:, pos+1] = v2
        pos += 2
    end

    # append nullspace vectors safely
    for idx in zero_idx
        v = copy(Φ[:, idx])
        for prev in 1:(pos-1)
            v -= S[:, prev] * (S[:, prev]' * v)
        end
        if norm(v) < tol
            continue
        end
        S[:, pos] = v / norm(v)
        pos += 1
    end

    # enforce orthonormality (more stable with qr)
    Q = qr(S).Q
    S = Matrix(Q)

    # create canonical transformation X and zero small entries
    X = S' * P * conj(S)

    # permutation to have positive elements in the upper-right of each 2x2 block
    perm_mat = canonical_skew_permutation(X)
    X = perm_mat' * X * perm_mat
    S = S * perm_mat

    X[abs.(X) .< tol] .= 0.0

    return S, X
end

"""
    absorb_phases(S::AbstractMatrix, X::AbstractMatrix)

Adjust phases of the paired columns in `S` so that the corresponding canonical matrix `X` becomes real with
positive entries in its upper-right elements. Returns the modified `(S2, X2)` pair.
"""
function absorb_phases(S::AbstractMatrix, X::AbstractMatrix)
    S2 = copy(S)
    X2 = copy(X)
    tol_absorb = 1e-10

    n = size(X2,1)
    i = 1
    while i <= n-1
        x = X2[i, i+1]
        y = X2[i+1, i]
        # If already real with nonnegative value, skip
        if !(abs(imag(x)) ≈ 0 && real(x) >= 0)
            φ = angle(x)
            d = exp(1im * φ/2)          # uniform phase for the pair
            @views S2[:, i  ] .*= d
            @views S2[:, i+1] .*= d
            # Block transforms by conj(d)^2
            # After this, value becomes real ≈ |x|
            X2[i, i+1] = abs(x)
            X2[i+1, i] = -abs(x)
        end
        i += 2
    end
    max_imag = maximum(abs, imag(X2))
    if max_imag > tol_absorb
        @warn "absorb_phases: residual imaginary part in X2 exceeds tolerance" max_imag=max_imag tol=tol_absorb
    end
    # @assert isapprox(imag(X2), zeros(ComplexF64, n,n), atol=1e-10) "X2 should be real after phase absorption"

    return S2, real(X2)
end

"""
    canonical_skew_permutation(P::AbstractMatrix)

Return a permutation matrix that reorders 2×2 skew blocks so their upper-right elements have a nonnegative real part.
"""
function canonical_skew_permutation(P::AbstractMatrix)
    n = size(P,1)
    perm = collect(1:n)
    i = 1
    while i < n
        a = P[perm[i], perm[i+1]]
        b = P[perm[i+1], perm[i]]
        # Detect a 2×2 skew block (nonzero pair with b ≈ -a)
        if abs(a) > 0 && isapprox(b, -a; atol=1e-14, rtol=1e-10)
            # If real part of upper-right entry is < 0, swap the two indices
            if real(a) < 0
                perm[i], perm[i+1] = perm[i+1], perm[i]
            end
            i += 2
        else
            i += 1
        end
    end
    S = Matrix{eltype(P)}(I, n, n)
    return S[:, perm]
end

"""bloch_messiah_decomposition(M::AbstractMatrix)

Compute the Bloch–Messiah decomposition of the Bogoliubov transformation `M` and return the
left (`Dmat`), middle (`UV_mat`), and right (`Cmat`) blocks such that
`M ≈ Dmat * UV_mat * Cmat`.
"""
function bloch_messiah_decomposition(M::AbstractMatrix)
    N = div(size(M, 1), 2)

    U,V = get_bogoliubov_blocks(M)

    Q = conj.(V) * transpose(V)
    @assert Q' ≈ Q "Q should be Hermitian"
    Q = Hermitian(Q) # enforce exact Hermiticity
    P = conj.(V) * transpose(U)

    @assert isapprox(transpose(P), -P; atol=1e-10) "P should be skew-symmetric"
    P = (P - transpose(P)) / 2 # enforce exact skew-symmetry
    @assert isapprox(Q*P, P*conj.(Q); atol=1e-10) "Q*P != P*conj.(Q)"

    E_Q, B = eigen(Q; sortby = (x -> -real(x)))
    # Q_bar = real(B'*Q*B)
    P_bar = B'*P*conj.(B)
    @assert isapprox(P_bar, -transpose(P_bar); atol=1e-10) "P_bar should be skew-symmetric"
    P_bar = (P_bar - transpose(P_bar)) / 2 # enforce exact skew-symmetry
    
    # Bring P_bar to canonical form by block-diagonalizing within degenerate subspaces of Q to avoid mixing
    # As P_bar is block diagonal
    #
    # `degeneracy_atol` groups the eigenvalues of Q into degenerate subspaces. Different LAPACK/BLAS
    # builds (OpenBLAS on Linux vs. Windows) resolve degenerate eigenvalues only to ~1e-10, so a tight
    # 1e-10 tolerance splits a structured degenerate block into 1×1 pieces on some platforms; those then
    # take the `S_sub = I` fallback and leave the real O(1) pairing un-canonicalized, giving a complex
    # Ubar/Vbar and a broken reconstruction (CI passed on Windows but failed on Ubuntu). The eigenvalues
    # of Q ∈ [0, 1] are either degenerate to ≲1e-10 or separated by ≳1e-5, so 1e-8 sits safely in between.
    degeneracy_atol = 1e-8
    S = zeros(ComplexF64, size(P_bar))
    visited = falses(length(E_Q)) # Track which indices have been processed
    for i in 1:length(E_Q)
        if !visited[i]
            idx = findall(x -> isapprox(x, E_Q[i]; atol=degeneracy_atol), E_Q)
            visited[idx] .= true # Mark all indices in this block as visited
            P_sub = P_bar[idx, idx] # Extract the sub-block corresponding to the eigenvalue
            if norm(P_sub, Inf) < 1e-10
                # P carries no useful pairing information in this degenerate block
                # (empty or fully occupied Slater block). The gauge is then completely
                # underdetermined, so we bypass the skew-canonical form routine and choose
                # a stable, default gauge S_sub = I to avoid numerical instability.
                S_sub = Matrix{ComplexF64}(I, length(idx), length(idx))
            else
                # P has structure, use it to fix the gauge.
                S_sub, X_sub = skew_canonical_form(P_sub) # Canonical form for this block
                S_sub, _ = absorb_phases(S_sub, X_sub)  # makes canonical blocks real
                @assert (norm(S_sub' * S_sub - I(length(idx)), Inf) < 1e-10) "Gauge fixing failed in a degenerate block."
            end
            S[idx, idx] = S_sub # Place the canonical transformation in the correct block of S
        end
    end
    P_canonical = S' * P_bar * conj.(S)

    A = permute_zero_cols_to_end(P_canonical)

    D = B * S * A
    @assert D' * D ≈ I "D should be unitary"

    @assert isapprox(D'*P*conj(D), D'*conj(V)*transpose(U)*conj(D); atol=1e-10)
    
    F = MatrixFactorizations.rq(D' * U)
    R = Matrix(F.R)
    Q = Matrix(F.Q)
    # Ubar = R
    # C = Q

    # Fix phases so diagonal of R becomes positive real.
    d = diag(R)
    ph = similar(d)
    for i in eachindex(d)
        ph[i] = (abs(d[i]) > 0) ? d[i]/abs(d[i]) : one(d[i])   # unit-modulus (or 1 if zero)
    end
    Φ  = Diagonal(conj.(ph))            # multiply R on right by Φ to remove phases
    Rpos = R * Φ                        # now diagonal(Rpos) = abs.(d) ≥ 0 (real)
    Qnew = Φ' * Q                       # keep A invariant: (R Φ)(Φ' Q) = R Q
    Ubar = Rpos                         # Ū with positive diagonal
    C    = Qnew

    Vbar = transpose(D) * V * C'
    # @assert D'*conj(V)*transpose(U)*conj(D) ≈ D'*conj(V)*transpose(C)*conj(C)*transpose(U)*conj(D)
    # @assert D'*conj(V)*transpose(Q)*conj(Q)*transpose(U)*conj(D) ≈ D'*conj(V)*transpose(Q)*R

    # Canonicalize fully occupied Slater blocks (|v| = 1, hence u = 0). The RQ factorization
    # above is driven by U, which vanishes on these columns, so it leaves their gauge
    # undetermined and Vbar generically complex there. We identify the occupied columns (zero
    # Ubar column) and bring their Vbar sub-block to a real form via an SVD, absorbing the left
    # and right unitaries into D and C so that M = Dmat * UV_mat * Cmat stays invariant.
    occ = findall(j -> norm(@view Ubar[:, j]) < 1e-9, axes(Ubar, 2))
    if !isempty(occ)
        row_support = findall(i -> norm(@view Vbar[i, occ]) > 1e-9, axes(Vbar, 1))
        @assert length(row_support) == length(occ) "Occupied Slater block is not square; cannot canonicalize."
        Vblk = Vbar[row_support, occ]
        @assert isapprox(Vblk' * Vblk, I, atol=1e-8) "Occupied Slater block is not unitary."

        Fo = svd(Vblk) # Vblk = Fo.U * Diagonal(Fo.S) * Fo.V'
        D[:, row_support] = D[:, row_support] * conj(Fo.U) # rotate Vbar rows by Fo.U'
        C[occ, :] = Fo.V' * C[occ, :]                      # rotate Vbar columns by Fo.V
        Ubar = D' * U * C'
        Vbar = transpose(D) * V * C'
    end

    # Fix phases on identity block of Vbar
    diagV = diag(Vbar)
    id_cols = findall(x -> isapprox(abs(x), 1.0; atol=1e-10), diagV)
    if !isempty(id_cols)
        phase = ones(eltype(Vbar), size(Vbar, 2))
        for j in id_cols
            phase[j] = exp(-1im * angle(diagV[j]))
        end
        Phi = Diagonal(phase)

        #= 
            Absorb these phases into Ubar and C to keep the overall transformation invariant: 
                Ubar*C = (Ubar*Phi)*(Phi'*C)
                Vbar*C = (Vbar*Phi)*(Phi'*C)
        =#
        Ubar = Ubar * Phi
        Vbar = Vbar * Phi
        C = Phi' * C
    end

    @assert C'C ≈ I
    @assert Q'Q ≈ I

    @assert U ≈ D*Ubar*C "Something went wrong with Bloch-Messiah decomposition for U"
    @assert V ≈ conj.(D)*Vbar*C "Something went wrong with Bloch-Messiah decomposition for V"

    # remove numerical noise
    @assert isapprox(Ubar, real.(Ubar); atol=1e-10) "Ubar should be real"
    Ubar = real(Ubar)
    Ubar[abs.(Ubar) .< 1e-12] .= 0.0
    @assert isapprox(Vbar, real.(Vbar); atol=1e-10) "Vbar should be real"
    Vbar = real(Vbar)
    Vbar[abs.(Vbar) .< 1e-12] .= 0.0

    Dmat = [D zeros(N,N); zeros(N,N) conj.(D)]
    UV_mat = [Ubar Vbar; Vbar Ubar]
    Cmat = [C zeros(N,N); zeros(N,N) conj.(C)]

    @assert isapprox(M, Dmat * UV_mat * Cmat; atol=1e-10) "Bloch-Messiah decomposition failed to reconstruct Bogoliubov transformation M"
    return Dmat, UV_mat, Cmat
end

"""permute_zero_cols_to_end(P::AbstractMatrix)

Return a permutation matrix that shifts zero-valued columns of `P` to the end while preserving the order of the others.
"""
function permute_zero_cols_to_end(P::AbstractMatrix)
    n = size(P,1)
    perm = collect(1:n)
    i = 1
    j = n
    while i < j
        if all(iszero, P[:, perm[i]])
            perm[i], perm[j] = perm[j], perm[i]
            j -= 1
        else
            i += 1
        end
    end
    A = Matrix{eltype(P)}(I, n, n)
    return A[:, perm]
end

"""get_mats_from_bloch_messiah(Dmat, UVmat, Cmat)

Extract the `D`, `Ubar`, `Vbar`, and `C` matrices from the doubled Bloch–Messiah blocks.
"""
function get_mats_from_bloch_messiah(Dmat, UVmat, Cmat)
    N = div(size(UVmat, 1), 2)

    Ubar = UVmat[1:div(size(UVmat, 1), 2), 1:div(size(UVmat, 2), 2)]
    Vbar = UVmat[div(size(UVmat, 1), 2)+1:end, 1:div(size(UVmat, 2), 2)]
    C = Cmat[1:div(size(Cmat, 1), 2), 1:div(size(Cmat, 2), 2)]
    D = Dmat[1:div(size(Dmat, 1), 2), 1:div(size(Dmat, 2), 2)]

    return D,Ubar,Vbar,C
end

"""truncated_bloch_messiah(Dmat,UVmat,Cmat)

Return a truncated decomposition that removes zero columns from the `Vbar` block, keeping compatible block structure.
"""
function truncated_bloch_messiah(Dmat,UVmat,Cmat)
    D,Ubar,Vbar,C = get_mats_from_bloch_messiah(Dmat, UVmat, Cmat)

    # discard numerically zero columns
    tol = 1e-10
    zero_ind = findfirst(col -> maximum(abs.(col)) < tol, eachcol(Vbar))

    if zero_ind === nothing
        return Dmat, UVmat, Cmat
    end
    
    D_prime = D[:, 1:zero_ind-1]
    Vbar_prime = Vbar[1:zero_ind-1, 1:zero_ind-1]
    Ubar_prime = Ubar[1:zero_ind-1, 1:zero_ind-1]
    C_prime = C[1:zero_ind-1, :]

    Dmat_prime = [D_prime zeros(size(D_prime)); zeros(size(D_prime)) conj.(D_prime)]
    UVmat_prime = [Ubar_prime Vbar_prime; Vbar_prime Ubar_prime]
    Cmat_prime = [C_prime zeros(size(C_prime)); zeros(size(C_prime)) conj.(C_prime)]

    return Dmat_prime, UVmat_prime, Cmat_prime
end

function get_matrix_element(H_BdG::Hermitian, j_prime::Vector{Int}, j::Vector{Int})
    N = size(H_BdG, 1) ÷ 2

    @assert length(j_prime) == length(j) "j and j' should have the same length"
    @assert length(j) == N "j and j' should have length N"
    @assert all(x -> x == 0 || x == 1, j) "j should be a binary vector"
    @assert all(x -> x == 0 || x == 1, j_prime) "j' should be a binary vector"

    T = H_BdG[1:N, 1:N]
    D = H_BdG[1:N, N+1:end]

    # Find the sites where the occupation differs
    diff_sites = findall(j_prime .!= j)
    num_diff = length(diff_sites)

    if num_diff == 0
        # Diagonal element: sum over single-particle energies
        # H_diag = sum_a T_{aa}
        val = 0.0 + 0.0im
        for a in 1:N
            if j[a] == 1
                val += T[a, a]
            end
        end
        return val

    elseif num_diff == 2
        # Differs by 2 sites: could be a hopping or a pairing term
        a, b = diff_sites[1], diff_sites[2]

        # Determine the sign from the fermionic string (parity of particles between the two sites)
        particles_between = sum(j[min(a,b)+1 : max(a,b)-1])
        fermion_sign = (-1)^particles_between

        if j_prime[a] == 1 && j_prime[b] == 0 && j[a] == 0 && j[b] == 1
            # Hopping: c^†_a c_b
            return fermion_sign * T[a, b]
        elseif j_prime[a] == 0 && j_prime[b] == 1 && j[a] == 1 && j[b] == 0
            # Hopping: c^†_b c_a
            # Notice the fermion string sign implicitly handles the commutation
            return fermion_sign * T[b, a]
        elseif j_prime[a] == 1 && j_prime[b] == 1 && j[a] == 0 && j[b] == 0
            # Pair creation: c^†_a c^†_b
            return fermion_sign * D[a, b]  # D is usually antisymmetric
        elseif j_prime[a] == 0 && j_prime[b] == 0 && j[a] == 1 && j[b] == 1
            # Pair annihilation: c_a c_b
            # Matrix element is D^†_{ab} or conj(D_{ba})
            return fermion_sign * conj(D[a, b])
        else
            return 0.0 + 0.0im
        end
    else
        # Operators strictly act on at most 2 sites. 
        # If configurations differ by 1, 3, 4, or more sites, the matrix element is 0.
        return 0.0 + 0.0im
    end
end


###########################################################################################################################
# The following functions are only for testing the Slater trial state optimization WITHOUT PEPS joint sampling.
#
# TODO: Remove in future versions or move to dedicated test files
###########################################################################################################################

function generate_Oks_and_Eks_Slater(H_BdG_exact::Hermitian, H_BdG_func::Function, N::Int; parity_sector::Int = 0, target_state::Int=0)
    @assert parity_sector == 0 || parity_sector == 1 "Parity sector must be either 0 (even) or 1 (odd)"

    function Oks_and_Eks_(η::Vector{T}, sample_nr::Integer; timer=TimerOutput(), kwargs...) where T
        # create GS from η
        GS = QuantumNaturalfPEPS.GaussianState(H_BdG_func, N; η=η, parity_sector=parity_sector, target_state=target_state)
        return @timeit timer "Oks_and_Eks" Oks_and_Eks_singlethread_Slater(GS, H_BdG_exact, sample_nr; timer=timer, kwargs...)     
    end

    return Oks_and_Eks_
end

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
        _, Eks[i], logψs[i], samples[i], logpc[i], contract_dims[i] = Ok_and_Ek_Slater(GS, H_BdG_exact; timer, Ok=Ok_view, amp_cache, SlaterConnections, kwargs...)
    end
    
    #return Ok, E_loc, logψ, samples, compute_importance_weights(logψ, logpc)
    Dict(:Oks => transpose(Oks), :Eks => Eks, :logψs => logψs, :samples => samples, :weights => compute_importance_weights(logψs, logpc), :contract_dims => contract_dims)
    # returns Gradient, local Energy, log(<ψ|S>), samples S, p
end

# |psi> = sum_j S_j |j>
# <j'|psi> = sum_j S_j <j'|j> = S_j_prime
# E_loc = <j'|H|psi> / <j'|psi> =  sum_j <j_prime|H|j> * S_j / S_j_prime, where j_prime is the sampled configuration and j runs through all possible configurations.
function get_Ek_Slater(GS::GaussianState, H_BdG_exact::Hermitian, S::Matrix{Int64}; 
    timer=TimerOutput(), 
    parity_sector::Int = 0,
    amp_cache=nothing, 
    SlaterConnections::Union{Nothing,Dict{Tuple{Int, Int}, SlaterConnection}}=nothing)

    Ek = zero(ComplexF64)
    j_prime = collect(vec(S))

    amp_cache = isnothing(amp_cache) ? build_amplitude_cache(GS) : amp_cache
    SlaterConnections = isnothing(SlaterConnections) ? get_Slater_Ek_terms(H_BdG_exact) : SlaterConnections

    S_jprime = get_amplitude(amp_cache, j_prime)

    # Only configurations differing by at most two occupations can contribute
    # for quadratic BdG Hamiltonians.

    # diagonal contribution j = j_prime
    for a in eachindex(j_prime)
        if j_prime[a] == 1 && haskey(SlaterConnections, (a, a))
            Ek += SlaterConnections[(a, a)].t
        end
    end

    # off-diagonal contributions for nonzero connectivity only
    # only loop over i<j pairs to avoid double counting and also note that we can use: t_ij = t_ji* and Δ_ij = -Δ_ji
    j = copy(j_prime)
    @timeit timer "off-diagonal H_BdG_exact elements" for (a, b) in keys(SlaterConnections)
        a==b && continue # skip diagonal terms, already included above

        ja = j_prime[a] 
        jb = j_prime[b]
        particles_between_ab = sum(@view j_prime[(a+1):(b-1)])
        fsign = isodd(particles_between_ab) ? -1 : 1

        coeff = zero(ComplexF64)
        if ja == 0 && jb == 1 # hopping from b to a such that we have nonzero overlap: <j'| t_ab c_a^† c_b |j> = sign * t_ab * S_j
            coeff = SlaterConnections[(a, b)].t
        elseif ja == 1 && jb == 0 # hopping from a to b such that we have nonzero overlap: <j'| t_ba c_b^† c_a |j> = sign * t_ba * S_j
            coeff = conj(SlaterConnections[(a, b)].t)
        elseif ja == 0 && jb == 0 # pairing of a and b such that we have nonzero overlap: <j'| Δ_ab c_a c_b |j> = sign * Δ_ab * S_j
            coeff = SlaterConnections[(a, b)].Δ
        else # pairing of a and b such that we have nonzero overlap: <j'| Δ_ab* c†_a c†_b |j> = sign * conj(Δ_ab) * S_j
            coeff = conj(SlaterConnections[(a, b)].Δ)
        end

        h_elem = fsign * coeff
        iszero(h_elem) && continue

        # create allowed configuration j by flipping the occupations at a and b compared to j_prime
        j[a] = 1 - ja
        j[b] = 1 - jb
        S_j = get_amplitude(amp_cache, j)

        Ek += h_elem * exp(log(S_j) - log(S_jprime))

        # restore j to j_prime for the next iteration
        j[a] = ja
        j[b] = jb
    end

    return real(Ek)
end

# Calculates the Energy and Gradient of a given peps and hamiltonian
function Ok_and_Ek_Slater(GS::GaussianState, H_BdG_exact::Hermitian; timer=TimerOutput(), Ok=nothing, sampling_mode=:full,
                   resample=false, correct_sampling_error=true, resample_energy=0, # TODO: remove
                   amp_cache=nothing, SlaterConnections=nothing,
                   )
    
    S, logpc = @timeit timer "sampling" get_sample(GS; timer) # draw a sample
    occ_string = collect(vec(S')) # julia vec(matrix) is column major, whereas we use row major in the sampling logic

    if amp_cache === nothing
        amp_cache = @timeit timer "amp_cache" build_amplitude_cache(GS)
    end
    
    # initialize the flipped logψ dictionary, will be used to compute other observables or for the resampling
    # Ek_terms = @timeit timer "precomp_sHψ_elems"  QuantumNaturalGradient.get_precomp_sOψ_elems(ham_op, S; get_flip_sites=true)
    E_loc = @timeit timer "energy" get_Ek_Slater(GS, H_BdG_exact, S; timer, amp_cache=amp_cache, SlaterConnections=SlaterConnections) # compute the local energy
    grad = @timeit timer "log_gradients" get_Ok(GS, S, Ok) # compute the gradient

    logψ = log(get_amplitude(amp_cache, occ_string))
    max_bond = 2 # dummy
    return grad, E_loc, logψ, S, logpc, max_bond
end

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