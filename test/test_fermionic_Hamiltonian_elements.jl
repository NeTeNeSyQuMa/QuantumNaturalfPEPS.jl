using QuantumNaturalfPEPS
using LinearAlgebra
using Random
using Test
using ITensors, ITensorMPS

@testset "Fermionic Hamiltonian matrix elements" begin
    # Define the Lattice
    L = 2
    hilbert = siteinds("Fermion", L, L)
    N = L^2

    # from Eks.jl
    function find_flip_site(sample_orig, sample_shifted, sites)
        diffs = []
        for (i, s_orig, s_shift) in zip(sites, sample_orig, sample_shifted)
            if s_orig != s_shift
                push!(diffs, (i, s_shift))
            end
        end
        return (diffs...,)
    end
    function apply_flip_site(sample, patch)
        sample_c = copy(sample)
        for (index, value) in patch
            sample_c[index...] = value
        end
        return sample_c
    end

    #= MF parameters =#
    μ = 1.0
    t = 2.0
    Δ = 0.5
    η = [t, Δ, μ]

    # build BdG Hamiltonian matrix
    function build_H_BdG_mat(η, L)
        N = L^2

        t = η[1]
        Δ = η[2]
        μ = η[3]

        # Remove the boundary-crossing terms (every L-th bond)
        hopping_x = [((i % L == 0) ? 0.0 : t) for i in 1:(N-1)]
        hopping_y = fill(t, N-L)

        pairing_x = [((i % L == 0) ? 0.0 : Δ) for i in 1:(N-1)]
        pairing_y = fill(Δ, N-L)

        T = diagm(0 => fill(-μ, N), 1 => -hopping_x, -1 => -hopping_x, L => -hopping_y, -L => -hopping_y)
        D = diagm(1 => pairing_x, -1 => -pairing_x, L => pairing_y, -L => -pairing_y)

        H = [T D; D' -transpose(T)]
        return Hermitian(H)
    end

    # helper functions
    sample_index(S) = evalpoly(2, vec(S)) + 1 
    fermion_parity_between(sample, i, j) = isodd(sum(sample[min(i,j)+1 : max(i,j)-1])) ? -1 : 1

    function occ_string_to_matrix(occ_string, Lx, Ly)
        S = Array{Int64}(undef, Lx, Ly)
        for i in 1:Lx, j in 1:Ly
            lin = (j - 1) * Lx + i
            S[i, j] = occ_string[lin]
        end
        return S
    end

    # build full many-body Hamiltonian from the BdG matrix
    function build_H_manybody_analytic(η, L)
        N = L^2
        H_BdG = build_H_BdG_mat(η, L)

        T = H_BdG[1:N, 1:N]
        D = H_BdG[1:N, N+1:2N]

        dim = 2^N
        H = zeros(ComplexF64, dim, dim)

        # samples like in sampling
        samples = [occ_string_to_matrix(digits(i, base=2, pad=N), L, L) for i in 0:(2^N-1)]

        # test linear indexing
        for i in 1:N
            @test sample_index(samples[i]) == i
        end

        # loop over all samples <s|
        for sample_ in samples
            sample = vec(sample_) # convert to correct vector
            row = sample_index(sample) # <s|

            # Hopping + chemical potential (T block)
            for i in 1:N, j in 1:N
                tij = T[i, j]
                if tij != 0
                    if i == j # chem pot
                        # number operator term (already in T)
                        if sample[i] == 1
                            H[row, row] += tij
                        end
                    else
                        # cdag_i c_j
                        if sample[j] == 1 && sample[i] == 0
                            s2 = copy(sample)
                            s2[j] = 0
                            s2[i] = 1
                            col = sample_index(s2) # |s'>
                            sign = fermion_parity_between(sample, i, j)
                            H[row, col] += tij * sign
                        end
                    end
                end
            end

            # Pairing (D block)
            for i in 1:N-1, j in i+1:N
                dij = D[i, j]
                if dij != 0
                    # cdag_i cdag_j
                    if sample[i] == 0 && sample[j] == 0
                        s2 = copy(sample)
                        s2[j] = 1
                        s2[i] = 1
                        col = sample_index(s2) # |s'>
                        sign = fermion_parity_between(sample, i, j)
                        H[row, col] += dij * sign
                    end

                    # c_j c_i = -(c_i c_j)
                    if sample[i] == 1 && sample[j] == 1
                        s2 = copy(sample)
                        s2[j] = 0
                        s2[i] = 0
                        col = sample_index(s2) # |s'>
                        sign = fermion_parity_between(sample, i, j)
                        H[row, col] += dij * sign
                    end
                end
            end
        end

        return H
    end

    # build full many-body Hamiltonian from ITensors
    function get_ITensors_H_mat(L)
        N = L^2
        dim = 2^N
        ham_mat_manybody = zeros(ComplexF64, dim, dim)

        # Flatten the Hilbert space for indexing
        hilbert_flat = ham_op.hilbert[:]

        # Generate all 2^N possible basis configurations (0 = empty, 1 = occupied)
        samples = [occ_string_to_matrix(digits(i, base=2, pad=N), L, L) for i in 0:(2^N-1)]
        offset = 1

        #= basically the function from get_precomp_sOψ_elems! but it stores the matrix elements rather than computing the sum =#
        for sample in samples
            sample = vec(sample) # convert to correct vector

            # Offset by 1 because ITensor state indices for Fermions are 1-based (1 and 2)
            sample_ = sample .+ offset
            row = sample_index(sample)

            # 1) Process standard non-diagonal tensors
            for (tensor, sites) in zip(ham_op.tensors, ham_op.sites)
                sample_r = sample_[sites]
                hilbert_r = hilbert_flat[sites]
                
                indices_sample = collect(hi' => s for (hi, s) in zip(hilbert_r, sample_r))
                tensor_proj = onehot(eltype(tensor), indices_sample) * tensor

                perm = NDTensors.getperm(ITensors.inds(tensor_proj), hilbert_r)
                tensor_proj = ITensor(permutedims(tensor_proj.tensor, perm))

                inds = findall(x -> x != 0, tensor_proj.tensor)
                for ind in inds
                    sample_r2 = length(ind) == 1 ? ind : ind.I
                    key = find_flip_site(sample_r .- offset, sample_r2 .- offset, sites)
                    col_sample = apply_flip_site(sample, key)

                    vi = tensor_proj[ind]
                    col = sample_index(col_sample)
                    ham_mat_manybody[row, col] += vi
                end
            end

            # 2) Process diagonal tensors
            for (v, sites) in zip(ham_op.diag_tensors, ham_op.diag_sites)
                # Since v is a flat diagonal representation, we index it using 
                # the linear configuration index of the subsystem active sites
                sub_sample = sample[sites]
                # Convert binary configuration to a 1-based linear index
                diag_idx = evalpoly(2, sub_sample) + 1
                
                # Diagonal elements map the state back to itself (row == col)
                ham_mat_manybody[row, row] += v[diag_idx]
            end
        end

        return ham_mat_manybody
    end

    #= 
        Test analytic Hamiltonian
    =#
    H_BdG = build_H_BdG_mat(η, L)
    T, D = H_BdG[1:N, 1:N], H_BdG[1:N, N+1:2N]
    H_analytic = build_H_manybody_analytic(η, L)

    @testset "Analytic Hamiltonian" begin
        # <1100|H|1100> = <1100|μ (c_1† c_1 + c_2† c_2) |1100> = -2μ
        S = [1,1,0,0]
        @test H_analytic[sample_index(S), sample_index(S)] == T[1,1] + T[2,2]

        # <1100|H|1010> = <1100|-t (c_2† c_3) |1010> = <1100|-t (c_2† c_3) c_1† c_3† |0000> = -t
        S_prime = [1,0,1,0]
        S = [1,1,0,0]
        @test H_analytic[sample_index(S), sample_index(S_prime)] == fermion_parity_between(S_prime, 2, 3) * T[2,3]

        # <1110|H|1011> = <1110|-t (c_2† c_4) |1011> = <1110|-t (c_2† c_4) c_1† c_3† c_4† |0000> = -(-t)
        S_prime = [1,0,1,1]
        S = [1,1,1,0]
        @test H_analytic[sample_index(S), sample_index(S_prime)] == fermion_parity_between(S_prime, 2, 4) * T[2,4]

        # <1111|H|1010> = <1111|Δ (c_2† c_4†) |1010> = <1111|Δ (c_2† c_4†) c_1† c_3† |0000> = -Δ
        S_prime = [1,0,1,0]
        S = [1,1,1,1]
        @test H_analytic[sample_index(S), sample_index(S_prime)] == fermion_parity_between(S_prime, 2, 4) * D[2,4]

        # <1111|H|1010> = <1111|Δ (c_2† c_4†) |1010> = <1111|Δ (c_2† c_4†) c_1† c_3† |0000> = -Δ
        S_prime = [1,0,1,0]
        S = [1,1,1,1]
        @test H_analytic[sample_index(S), sample_index(S_prime)] == fermion_parity_between(S_prime, 2, 4) * D[2,4]
    end

    #= 
        Test ITensors Hamiltonian
    =#
    H_tight_binding_os = OpSum()
    for i in 1:L, j in 1:L
        if j < L
            H_tight_binding_os += (-t,"Cdag", (i, j), "C", (i, j+1))
            H_tight_binding_os += (-t,"Cdag", (i, j+1), "C", (i, j))
            H_tight_binding_os += (Δ,"Cdag", (i, j), "Cdag", (i, j+1))
            H_tight_binding_os += (Δ,"C", (i, j+1), "C", (i, j))
        end
        if i < L
            H_tight_binding_os += (-t,"Cdag", (i, j), "C", (i+1, j))
            H_tight_binding_os += (-t,"Cdag", (i+1, j), "C", (i, j))
            H_tight_binding_os += (Δ,"Cdag", (i, j), "Cdag", (i+1, j))
            H_tight_binding_os += (Δ,"C", (i+1, j), "C", (i, j))
        end
        H_tight_binding_os += (-μ, "N", (i, j))
    end
    ham_op = QuantumNaturalfPEPS.TensorOperatorSum(H_tight_binding_os, hilbert)

    #= 
        Test ITensors functionality
    =#
    @testset "Analytic Hamiltonian (ITensors)" begin
        # <1110|H|1011> = <1110|-t (c_2† c_4) |1011> = <1110|-t (c_2† c_4) c_1† c_3† c_4† |0000> = -(-t)
        hilbert_flat = ham_op.hilbert[:]
        sample = [1,0,1,1] # corresponds to |1011>
        sites = [4,2] # select [0,1] from sample
        sample_ = sample .+ 1
        sample_r = sample_[sites] # [1,2] with offset
        hilbert_r = hilbert_flat[sites]
        isfermionic = hastags(hilbert_flat[1], "Fermion")

        for (tensor, sites) in zip(ham_op.tensors, ham_op.sites)
            if sites == [4,2]
                indices_sample = collect(hi' => s for (hi, s) in zip(hilbert_r, sample_r)) # Selects the indices that act on the tensor from the left O|s>
                tensor_proj = onehot(eltype(tensor), indices_sample) * tensor # <s'|T

                inds = findall(x -> x != 0, tensor_proj.tensor)
                for ind in inds
                    if length(ind) == 1
                        sample_r2 = ind
                    else
                        sample_r2 = ind.I
                    end
                    key = find_flip_site(sample_r .- 1, sample_r2 .- 1, sites)
                    sample_flipped = apply_flip_site(sample, key)

                    vi = tensor_proj[ind]
                    @test isapprox(vi, t; atol=1e-12)
                end
            end
        end

        ham_mat_manybody_analytic = build_H_manybody_analytic([t, Δ, μ], L)
        ham_mat_manybody = get_ITensors_H_mat(L)

        @test isapprox(ham_mat_manybody, ham_mat_manybody_analytic; atol=1e-10)
    end
end;