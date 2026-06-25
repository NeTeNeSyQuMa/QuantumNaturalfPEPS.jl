using Test
using QuantumNaturalfPEPS
using QuantumNaturalGradient
using ITensors, ITensorMPS
using LinearAlgebra
using Random

Random.seed!(1234)

#= 
    Custom sampling function to also return the uncombined probabilities from the PEPS and the Gaussian state
=#
function _sample_ρr(ρ_r, S, r, c; trial_state::QuantumNaturalfPEPS.AbstractTrialState=IdentityState(size(ρ_r, 1)))
    occ_dict = Dict{Int, Int}()
    for i in 1:size(S,1), j in 1:size(S,2)
        if i < r || (i == r && j < c)
            # use linear indexing (assume square lattice here)
            occ_dict[(j-1)*size(S,1) + i] = S[i,j] # -> use Column Major Order here to be consistent with the PEPS site ordering and the sampling order in get_sample()
        end
    end

    # prepare prob vector for PEPS
    k = size(ρ_r, 1) 
    T = real(eltype(ρ_r))
    r_prob = Vector{T}(undef, k)
    for i in 1:k
        r_prob[i] = abs(ρ_r[i, i])
        @assert imag(ρ_r[i, i]) / (r_prob[i] + 1e-10) < 1e-8 "ρ_r is not real $(ρ_r[i, i])"
    end
    r_prob ./= sum(r_prob) # normalize the PEPS probabilities

    # prepare prob vector for trial state
    current_site_key = (c-1)*size(S,1) + r # use column major ordering here
    q_prob = Vector{T}(undef, k)
    for i in 1:k
        occ_dict[current_site_key] = i-1
        q_prob[i] = QuantumNaturalfPEPS.get_prob(trial_state, occ_dict) # joint probability
    end
    p_final = r_prob .* q_prob

    i = QuantumNaturalfPEPS.sample_p(p_final, normalize=true)
    return i-1, p_final[i], p_final, q_prob, r_prob
end

function _build_H_BdG_mat(η, N)
    t = η[1]
    Δ = η[2]
    μ = η[3]

    T = diagm(0 => fill(-μ, N), 1 => fill(-t, N-1), -1 => fill(-conj(t), N-1))
    D = diagm(1 => fill(Δ, N-1), -1 => fill(-Δ, N-1))

    H = [T D; D' -transpose(T)]
    return Hermitian(H)
end

@testset "Test: Sampling" begin
    N_samples = 10000
    tol = 1 / sqrt(N_samples) # set tolerance based on standard error
    L = 2
    hilbert = siteinds("S=1/2", L, L)
    exact_dim = 2^(L^2)

    # create tight binding Hamiltonian and corresponding Gaussian state
    N = L*L
    # simple Tight-binding Hamiltonian as test
    η = [1.0, 1.0, 0.0]
    GS = QuantumNaturalfPEPS.GaussianState(_build_H_BdG_mat, N; η=η)

    # exact prob distribution from Gaussian state
    p_GS = zeros(Float64, 2^N)
    for idx in 0:(2^N - 1)
        occ_string = digits(idx, base=2, pad=N)
        p_GS[idx + 1] = QuantumNaturalfPEPS.get_prob(GS, occ_string)
    end

    @testset "|Ψ> = |Id> ⊗ |Slater>" begin
        # create a uniform state for the PEPS part |Id>
        peps_uniform = PEPS(hilbert; bond_dim=2, contract_dim=exact_dim, double_contract_dim=exact_dim)
        tensors = map(peps_uniform.tensors) do t
            uniform_tensor = ITensor(1.0, inds(t))
            return uniform_tensor ./ norm(uniform_tensor)
        end
        QuantumNaturalfPEPS.set_params!(peps_uniform, tensors)

        # draw samples
        S_arr_string = Array{String}(undef, N_samples)
        S_idx = Vector{Int64}(undef, N_samples)
        # Threads.@threads for j in 1:N_samples
        for j in 1:N_samples
            S, _, _ = QuantumNaturalfPEPS.get_sample(peps_uniform; trial_state=GS)
            S_arr_string[j] = join(vec(S))
            S_idx[j] = evalpoly(2, vec(S)) + 1 
        end

        # get prob distribution from samples
        sample_counts = zeros(Int, exact_dim)
        for idx in S_idx
            sample_counts[idx] += 1
        end
        p_samples = sample_counts / N_samples

        # test if sampled distribution matches exact distribution
        @test isapprox(p_samples, p_GS; atol=tol)
    end

    @testset "|Ψ> = |PEPS> ⊗ |Id>" begin
        peps_rnd = PEPS(hilbert; bond_dim=2, contract_dim=exact_dim, double_contract_dim=exact_dim) # Create random peps
        # get exact prob distribution from PEPS part
        p_PEPS = zeros(Float64, exact_dim)
        for idx in 0:(exact_dim - 1)
            # Generate bit string with (MSB) most-significant bit corresponding to site (1,1) and least-significant bit to site (L,L)
            occ_string = digits(idx, base=2, pad=N)
            
            # Map cleanly to matrix using your strict column-major arrangement
            S = Array{Int64}(undef, L, L)
            for r in 1:L, c in 1:L
                S[r, c] = occ_string[(c - 1) * L + r]
            end
            
            # Project and contract the exact scalar amplitude
            amp = QuantumNaturalfPEPS.contract_peps_exact(QuantumNaturalfPEPS.get_projected(peps_rnd, S))
            p_PEPS[idx + 1] = abs2(amp)
        end
        p_PEPS ./= sum(p_PEPS)

        # draw samples
        S_arr_string = Array{String}(undef, N_samples)
        S_idx = Vector{Int64}(undef, N_samples)
        Threads.@threads for j in 1:N_samples
            S, _, _ = QuantumNaturalfPEPS.get_sample(peps_rnd; trial_state=QuantumNaturalfPEPS.IdentityState(2))
            S_arr_string[j] = join(vec(S))
            S_idx[j] = evalpoly(2, vec(S)) + 1 
        end

        # get prob distribution from samples
        sample_counts = zeros(Int, exact_dim)
        for idx in S_idx
            sample_counts[idx] += 1
        end
        p_samples = sample_counts / N_samples

        # test if sampled distribution matches exact distribution
        @test isapprox(p_samples, p_PEPS; atol=tol)
    end

    @testset "|Ψ> = |PEPS> ⊗ |Slater>" begin
        peps = PEPS(hilbert; bond_dim=2, contract_dim=exact_dim, double_contract_dim=exact_dim)

        @testset "Joint probability sampling: Theory" begin
            # draw one explicit sample 
            S = Array{Int64}(undef, size(peps))
            env_top = Array{QuantumNaturalfPEPS.Environment}(undef, size(peps, 1)-1)
            sites = siteinds(peps)
            ρ_r = ITensor()

            r_prob_vec = Vector{Vector{Float64}}(undef, N)
            q_prob_vec = Vector{Vector{Float64}}(undef, N)
            p_final_vec = Vector{Vector{Float64}}(undef, N)

            # we loop through every row
            for i in 1:size(peps, 1)
                sigma = 1
                ket = QuantumNaturalfPEPS.get_ket(peps, i, env_top)
                bra = prime.(conj(ket[:]))
                # we then calculate the unsampled environment (in one row)
                E = QuantumNaturalfPEPS.calculate_unsampled_Env_row(ket, bra, peps, i, sites[i, :])

                # then we loop through every column
                for j in 1:size(peps, 2)
                    # calculate the phys_dimxphys_dim matrix from which we sample
                    ρ_r, sigma = QuantumNaturalfPEPS.get_reduced_ρ(ket[j], bra[j], peps, i, j, E, sigma)
                    
                    S[i,j], pc, p_final, q_prob, r_prob = _sample_ρr(ρ_r, S, i, j; trial_state=GS)
                    lin_index = (j-1)*size(peps,1) + i # linear index for column-major ordering

                    r_prob_vec[lin_index] = r_prob
                    q_prob_vec[lin_index] = q_prob
                    p_final_vec[lin_index] = p_final

                    # after the sampling of the current site, it is fixed and its contraction with the aleady sampled sites is stored in sigma
                    site = siteind(peps, i, j)
                    sigma = sigma * QuantumNaturalfPEPS.get_projector(S[i, j], sites[i, j]) * QuantumNaturalfPEPS.get_projector(S[i, j], sites[i, j]') 
                    sigma ./= pc # we divide by pc to avoid numerical issues
                end

                # Should we be recalculating the top environment here? Is it slower?
                # The answer is yes, it is slower, but not by match. But it is also more accurate.
                if i == 1
                    peps_projected_1 = QuantumNaturalfPEPS.get_projected(peps, S, 1, :)
                    env_top[1] = QuantumNaturalfPEPS.generate_env_row(peps_projected_1, peps.contract_dim; cutoff=peps.contract_cutoff)
                elseif i != size(peps, 1) 
                    peps_projected_row = QuantumNaturalfPEPS.get_projected(peps, S, i, :)
                    env_top[i] = QuantumNaturalfPEPS.generate_env_row(peps_projected_row, peps.contract_dim; env_row_above=env_top[i-1], cutoff=peps.contract_cutoff)
                end  
            end

            #= 
                Recall: We measure in column-major order, so for L=2 we measure the sites in that order:
                1 -> 3 -> 2 -> 4
            =#

            # cond probs for PEPS part
            r_s1_0 = r_prob_vec[1][1]
            r_s1_1 = r_prob_vec[1][2]
            r_s3_0_s1 = r_prob_vec[3][1]
            r_s3_1_s1 = r_prob_vec[3][2]
            r_s2_0_s1s3 = r_prob_vec[2][1]
            r_s2_1_s1s3 = r_prob_vec[2][2]
            r_s4_0_s1s2s3 = r_prob_vec[4][1]
            r_s4_1_s1s2s3 = r_prob_vec[4][2]

            @test sum(r_prob_vec[1]) ≈ 1.0
            @test sum(r_prob_vec[2]) ≈ 1.0
            @test sum(r_prob_vec[3]) ≈ 1.0
            @test sum(r_prob_vec[4]) ≈ 1.0

            # test for Gaussian state part
            # s1
            q_s1_0 = QuantumNaturalfPEPS.get_prob(GS, [0])
            q_s1_1 = QuantumNaturalfPEPS.get_prob(GS, [1])
            @test q_s1_0 + q_s1_1 ≈ 1.0

            # s3 (site index 3) given s1 (site index 1)
            q_s3_0_Λ_s1_0 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 0, 3 => 0))
            q_s3_1_Λ_s1_0 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 0, 3 => 1))
            q_s3_0_Λ_s1_1 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 1, 3 => 0))
            q_s3_1_Λ_s1_1 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 1, 3 => 1))
            @test q_s3_0_Λ_s1_0 + q_s3_1_Λ_s1_0 + q_s3_0_Λ_s1_1 + q_s3_1_Λ_s1_1 ≈ 1.0

            q_s3_given_s1_0 = [q_s3_0_Λ_s1_0 / q_s1_0, q_s3_1_Λ_s1_0 / q_s1_0]
            q_s3_given_s1_1 = [q_s3_0_Λ_s1_1 / q_s1_1, q_s3_1_Λ_s1_1 / q_s1_1]
            @test sum(q_s3_given_s1_0) ≈ 1.0
            @test sum(q_s3_given_s1_1) ≈ 1.0

            # s2
            q_s2_0_Λ_s1s3_00 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 0, 3 => 0, 2 => 0))
            q_s2_1_Λ_s1s3_00 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 0, 3 => 0, 2 => 1))
            q_s2_0_Λ_s1s3_01 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 0, 3 => 1, 2 => 0))
            q_s2_1_Λ_s1s3_01 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 0, 3 => 1, 2 => 1))
            q_s2_0_Λ_s1s3_10 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 1, 3 => 0, 2 => 0))
            q_s2_1_Λ_s1s3_10 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 1, 3 => 0, 2 => 1))
            q_s2_0_Λ_s1s3_11 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 1, 3 => 1, 2 => 0))
            q_s2_1_Λ_s1s3_11 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 1, 3 => 1, 2 => 1))
            @test q_s2_0_Λ_s1s3_00 + q_s2_1_Λ_s1s3_00 + q_s2_0_Λ_s1s3_01 + q_s2_1_Λ_s1s3_01 + q_s2_0_Λ_s1s3_10 + q_s2_1_Λ_s1s3_10 + q_s2_0_Λ_s1s3_11 + q_s2_1_Λ_s1s3_11 ≈ 1.0

            q_s2_given_s1s3_00 = [q_s2_0_Λ_s1s3_00 / q_s3_0_Λ_s1_0, q_s2_1_Λ_s1s3_00 / q_s3_0_Λ_s1_0]
            q_s2_given_s1s3_01 = [q_s2_0_Λ_s1s3_01 / q_s3_1_Λ_s1_0, q_s2_1_Λ_s1s3_01 / q_s3_1_Λ_s1_0]
            q_s2_given_s1s3_10 = [q_s2_0_Λ_s1s3_10 / q_s3_0_Λ_s1_1, q_s2_1_Λ_s1s3_10 / q_s3_0_Λ_s1_1]
            q_s2_given_s1s3_11 = [q_s2_0_Λ_s1s3_11 / q_s3_1_Λ_s1_1, q_s2_1_Λ_s1s3_11 / q_s3_1_Λ_s1_1]
            @test sum(q_s2_given_s1s3_00) ≈ 1.0
            @test sum(q_s2_given_s1s3_01) ≈ 1.0
            @test sum(q_s2_given_s1s3_10) ≈ 1.0
            @test sum(q_s2_given_s1s3_11) ≈ 1.0

            # s4
            q_s4_0_Λ_s1s2s3_000 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 0, 3 => 0, 2 => 0, 4 => 0))
            q_s4_1_Λ_s1s2s3_000 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 0, 3 => 0, 2 => 0, 4 => 1))
            q_s4_0_Λ_s1s2s3_001 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 0, 3 => 0, 2 => 1, 4 => 0))
            q_s4_1_Λ_s1s2s3_001 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 0, 3 => 0, 2 => 1, 4 => 1))
            q_s4_0_Λ_s1s2s3_010 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 0, 3 => 1, 2 => 0, 4 => 0))
            q_s4_1_Λ_s1s2s3_010 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 0, 3 => 1, 2 => 0, 4 => 1))
            q_s4_0_Λ_s1s2s3_011 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 0, 3 => 1, 2 => 1, 4 => 0))
            q_s4_1_Λ_s1s2s3_011 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 0, 3 => 1, 2 => 1, 4 => 1))
            q_s4_0_Λ_s1s2s3_110 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 1, 3 => 1, 2 => 0, 4 => 0))
            q_s4_1_Λ_s1s2s3_110 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 1, 3 => 1, 2 => 0, 4 => 1))
            q_s4_0_Λ_s1s2s3_100 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 1, 3 => 0, 2 => 0, 4 => 0))
            q_s4_1_Λ_s1s2s3_100 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 1, 3 => 0, 2 => 0, 4 => 1))
            q_s4_0_Λ_s1s2s3_101 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 1, 3 => 0, 2 => 1, 4 => 0))
            q_s4_1_Λ_s1s2s3_101 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 1, 3 => 0, 2 => 1, 4 => 1))
            q_s4_0_Λ_s1s2s3_111 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 1, 3 => 1, 2 => 1, 4 => 0))
            q_s4_1_Λ_s1s2s3_111 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 1, 3 => 1, 2 => 1, 4 => 1))
            @test q_s4_0_Λ_s1s2s3_000 + q_s4_1_Λ_s1s2s3_000 + q_s4_0_Λ_s1s2s3_001 + q_s4_1_Λ_s1s2s3_001 + q_s4_0_Λ_s1s2s3_010 + q_s4_1_Λ_s1s2s3_010 + q_s4_0_Λ_s1s2s3_011 + q_s4_1_Λ_s1s2s3_011 + q_s4_0_Λ_s1s2s3_100 + q_s4_1_Λ_s1s2s3_100 + q_s4_0_Λ_s1s2s3_101 + q_s4_1_Λ_s1s2s3_101 + q_s4_0_Λ_s1s2s3_110 + q_s4_1_Λ_s1s2s3_110 + q_s4_0_Λ_s1s2s3_111 + q_s4_1_Λ_s1s2s3_111 ≈ 1.0

            q_s4_given_s1s2s3_000 = [q_s4_0_Λ_s1s2s3_000 / q_s2_0_Λ_s1s3_00, q_s4_1_Λ_s1s2s3_000 / q_s2_0_Λ_s1s3_00]
            q_s4_given_s1s2s3_001 = [q_s4_0_Λ_s1s2s3_001 / q_s2_1_Λ_s1s3_00, q_s4_1_Λ_s1s2s3_001 / q_s2_1_Λ_s1s3_00]
            q_s4_given_s1s2s3_010 = [q_s4_0_Λ_s1s2s3_010 / q_s2_0_Λ_s1s3_01, q_s4_1_Λ_s1s2s3_010 / q_s2_0_Λ_s1s3_01]
            q_s4_given_s1s2s3_011 = [q_s4_0_Λ_s1s2s3_011 / q_s2_1_Λ_s1s3_01, q_s4_1_Λ_s1s2s3_011 / q_s2_1_Λ_s1s3_01]
            q_s4_given_s1s2s3_100 = [q_s4_0_Λ_s1s2s3_100 / q_s2_0_Λ_s1s3_10, q_s4_1_Λ_s1s2s3_100 / q_s2_0_Λ_s1s3_10]
            q_s4_given_s1s2s3_101 = [q_s4_0_Λ_s1s2s3_101 / q_s2_1_Λ_s1s3_10, q_s4_1_Λ_s1s2s3_101 / q_s2_1_Λ_s1s3_10]
            q_s4_given_s1s2s3_110 = [q_s4_0_Λ_s1s2s3_110 / q_s2_0_Λ_s1s3_11, q_s4_1_Λ_s1s2s3_110 / q_s2_0_Λ_s1s3_11]
            q_s4_given_s1s2s3_111 = [q_s4_0_Λ_s1s2s3_111 / q_s2_1_Λ_s1s3_11, q_s4_1_Λ_s1s2s3_111 / q_s2_1_Λ_s1s3_11]
            @test sum(q_s4_given_s1s2s3_000) ≈ 1.0
            @test sum(q_s4_given_s1s2s3_001) ≈ 1.0
            @test sum(q_s4_given_s1s2s3_010) ≈ 1.0
            @test sum(q_s4_given_s1s2s3_011) ≈ 1.0
            @test sum(q_s4_given_s1s2s3_100) ≈ 1.0
            @test sum(q_s4_given_s1s2s3_101) ≈ 1.0
            @test sum(q_s4_given_s1s2s3_110) ≈ 1.0
            @test sum(q_s4_given_s1s2s3_111) ≈ 1.0

            # Now test for the drawn config S
            p_s1_1 = p_final_vec[1][S[1,1] + 1]
            r_drawn = (S[1,1] == 0) ? r_s1_0 : r_s1_1
            q_drawn = (S[1,1] == 0) ? q_s1_0 : q_s1_1
            @test p_s1_1 ≈ r_drawn * q_drawn / (r_s1_0 * q_s1_0 + r_s1_1 * q_s1_1)

            # s3 = (1,2), lin_index = 3
            p_s3_0_given_s1_1 = p_final_vec[3][S[1,2] + 1]
            r_drawn = (S[1,2] == 0) ? r_s3_0_s1 : r_s3_1_s1
            q_drawn = (S[1,1] == 0) ? q_s3_given_s1_0 : q_s3_given_s1_1
            @test p_s3_0_given_s1_1 ≈ r_drawn * q_drawn[S[1,2] + 1] / (q_drawn[1] * r_s3_0_s1 + q_drawn[2] * r_s3_1_s1)

            # s2 = (2,1), lin_index = 2
            p_s2_0_given_s1s3_10 = p_final_vec[2][S[2,1] + 1]
            r_drawn = (S[2,1] == 0) ? r_s2_0_s1s3 : r_s2_1_s1s3
            q_drawn = begin
                if S[1,1] == 0 && S[1,2] == 0
                    q_s2_given_s1s3_00
                elseif S[1,1] == 0 && S[1,2] == 1
                    q_s2_given_s1s3_01
                elseif S[1,1] == 1 && S[1,2] == 0
                    q_s2_given_s1s3_10
                else
                    q_s2_given_s1s3_11
                end
            end
            @test p_s2_0_given_s1s3_10 ≈ r_drawn * q_drawn[S[2,1] + 1] / (q_drawn[1] * r_s2_0_s1s3 + q_drawn[2] * r_s2_1_s1s3)

            # s4 = (2,2), lin_index = 4
            p_s4_1_given_s1s2s3_100 = p_final_vec[4][S[2,2] + 1]
            r_drawn = (S[2,2] == 0) ? r_s4_0_s1s2s3 : r_s4_1_s1s2s3
            q_drawn = begin
                if S[1,1] == 0 && S[1,2] == 0 && S[2,1] == 0
                    q_s4_given_s1s2s3_000
                elseif S[1,1] == 0 && S[1,2] == 0 && S[2,1] == 1
                    q_s4_given_s1s2s3_001
                elseif S[1,1] == 0 && S[1,2] == 1 && S[2,1] == 0
                    q_s4_given_s1s2s3_010
                elseif S[1,1] == 0 && S[1,2] == 1 && S[2,1] == 1
                    q_s4_given_s1s2s3_011
                elseif S[1,1] == 1 && S[1,2] == 0 && S[2,1] == 0
                    q_s4_given_s1s2s3_100
                elseif S[1,1] == 1 && S[1,2] == 0 && S[2,1] == 1
                    q_s4_given_s1s2s3_101
                elseif S[1,1] == 1 && S[1,2] == 1 && S[2,1] == 0
                    q_s4_given_s1s2s3_110
                else
                    q_s4_given_s1s2s3_111
                end
            end
            @test p_s4_1_given_s1s2s3_100 ≈ r_drawn * q_drawn[S[2,2] + 1] / (q_drawn[1] * r_s4_0_s1s2s3 + q_drawn[2] * r_s4_1_s1s2s3)

        end

        @testset "Test sampled distribution" begin
            # build exact joint distribution by enumerating configurations
            N = L^2
            p_exact = ones(Float64, 2^N)
            for idx in 0:(2^N-1)
                bit = digits(idx, base=2, pad=N)
                # @show bit

                S = Array{Int64}(undef, size(peps))
                env_top = Array{QuantumNaturalfPEPS.Environment}(undef, size(peps, 1)-1)
                sites = siteinds(peps)
                ρ_r = ITensor()

                # we loop through every row
                for i in 1:size(peps, 1)
                    sigma = 1
                    ket = QuantumNaturalfPEPS.get_ket(peps, i, env_top)
                    bra = prime.(conj(ket[:]))
                    # we then calculate the unsampled environment (in one row)
                    E = QuantumNaturalfPEPS.calculate_unsampled_Env_row(ket, bra, peps, i, sites[i, :])

                    # then we loop through the different sites in one row
                    for j in 1:size(peps, 2)
                        lin_indx = (j-1)*size(peps, 1) + i # linear index for column-major ordering to match site ordering
                        s_lin_indx_occ = bit[lin_indx]
                        S[i, j] = s_lin_indx_occ

                        # calculate the phys_dimxphys_dim matrix from which we sample
                        ρ_r, sigma = QuantumNaturalfPEPS.get_reduced_ρ(ket[j], bra[j], peps, i, j, E, sigma)

                        occ_dict = Dict{Int, Int}()
                        for m in 1:size(S,1), n in 1:size(S,2)
                            if m < i || (m == i && n < j)
                                occ_dict[(n-1)*size(S,1) + m] = S[m,n] # use column-major site ordering (site index)
                            end
                        end

                        # prepare prob vector for PEPS
                        k = size(ρ_r, 1) 
                        T = real(eltype(ρ_r))
                        r_prob = Vector{T}(undef, k)
                        for l in 1:k
                            r_prob[l] = abs(ρ_r[l, l])
                            @assert imag(ρ_r[l, l]) / (r_prob[l] + 1e-10) < 1e-8 "ρ_r is not real $(ρ_r[l, l])"
                        end
                        r_prob ./= sum(r_prob) # normalize the PEPS probabilities

                        # prepare prob vector for trial state
                        q_prob = Vector{T}(undef, k)
                        for l in 1:k
                            occ_dict[(j-1)*size(S,1) + i] = l-1
                            q_prob[l] = QuantumNaturalfPEPS.get_prob(GS, occ_dict) # joint probability
                        end
                        p_final = r_prob .* q_prob
                        p_final ./= sum(p_final) # normalize the final probabilities

                        pc = p_final[s_lin_indx_occ + 1] # probability of the sampled configuration for the current site
                        p_exact[idx+1] = pc * p_exact[idx+1] # we multiply the probabilities of the sampled configuration for each site to get the joint probability of the full configuration

                        # after the sampling of the current site, it is fixed and its contraction with the aleady sampled sites is stored in sigma
                        site = siteind(peps, i, j)
                        sigma = sigma * QuantumNaturalfPEPS.get_projector(S[i, j], sites[i, j]) * QuantumNaturalfPEPS.get_projector(S[i, j], sites[i, j]') 
                        sigma ./= pc # we divide by pc to avoid numerical issues
                    end

                    # Should we be recalculating the top environment here? Is it slower?
                    # The answer is yes, it is slower, but not by match. But it is also more accurate.
                    if i == 1
                        peps_projected_1 = QuantumNaturalfPEPS.get_projected(peps, S, 1, :)
                        env_top[1] = QuantumNaturalfPEPS.generate_env_row(peps_projected_1, peps.contract_dim; cutoff=peps.contract_cutoff)
                    elseif i != size(peps, 1) 
                        peps_projected_row = QuantumNaturalfPEPS.get_projected(peps, S, i, :)
                        env_top[i] = QuantumNaturalfPEPS.generate_env_row(peps_projected_row, peps.contract_dim; env_row_above=env_top[i-1], cutoff=peps.contract_cutoff)
                    end  
                end
            end
            p_exact ./= sum(p_exact) # normalize the exact probabilities
            # p_exact = joint_distribution(peps, GS)

            # draw samples
            S_arr_string = Array{String}(undef, N_samples)
            S_idx = Vector{Int64}(undef, N_samples)
            Threads.@threads for j in 1:N_samples
                S, _, _ = QuantumNaturalfPEPS.get_sample(peps; trial_state=GS)
                S_arr_string[j] = join(vec(S))
                S_idx[j] = evalpoly(2, vec(S)) + 1 
            end

            # get prob distribution from samples
            sample_counts = zeros(Int, exact_dim)
            for idx in S_idx
                sample_counts[idx] += 1
            end
            p_samples = sample_counts / N_samples
            
            # test if sampled distribution matches exact distribution
            @test isapprox(p_samples, p_exact; atol=tol)
        end
    end
end;