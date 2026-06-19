using Test
using LinearAlgebra
using QuantumNaturalfPEPS

@testset "Gaussian States" begin

    @testset "Gaussian states (real MF parameters)" begin

        function build_H_BdG_mat(η, N)
            t = η[1]
            Δ = η[2]
            μ = η[3]

            T = diagm(0 => fill(-μ, N), 1 => fill(-t, N-1), -1 => fill(-conj(t), N-1))
            D = diagm(1 => fill(Δ, N-1), -1 => fill(-Δ, N-1))

            H = [T D; D' -transpose(T)]
            return Hermitian(H)
        end

        @testset "Bogoliubov transformation" begin
            N = 4
            η_vec = [[1.0, 2.0, 3.0], [1.0, 1.0, 0.0]]

            for η in η_vec
                H_BdG = build_H_BdG_mat(η, N)

                E, M = QuantumNaturalfPEPS.bogoliubov(H_BdG)

                @test isapprox(M' * M, I, atol=1e-8)
                @test isapprox((M' * H_BdG * M)[1:N, 1:N], Diagonal(E)[1:N, 1:N], atol=1e-8)
            end
        end

        @testset "Covariance matrix construction" begin
            N = 3

            # simple Tight-binding Hamiltonian as test
            t = 1.0
            Δ = 0.0
            μ = 0.0

            H_BdG = build_H_BdG_mat([t, Δ, μ], N)

            @testset "Even parity GS" begin 
                Γ, _ = QuantumNaturalfPEPS.get_Γ_from_H_BdG(H_BdG, 0; target_state=0)

                @test size(Γ) == (2N, 2N)
                @test Γ*Γ' ≈ I ./ 4
                @test QuantumNaturalfPEPS.getParity(Γ) == 0
            end
            @testset "Odd parity GS" begin 
                Γ, _ = QuantumNaturalfPEPS.get_Γ_from_H_BdG(H_BdG, 1; target_state=0)

                @test size(Γ) == (2N, 2N)
                @test Γ*Γ' ≈ I ./ 4
                @test QuantumNaturalfPEPS.getParity(Γ) == 1
            end
        end

        @testset "Covariance matrix sampling" begin
            @testset "Even parity GS" begin 
                # create tight binding Hamiltonian and corresponding Gaussian state
                L = 2
                N = L*L
                # simple Tight-binding Hamiltonian as test
                t1 = 1.0
                Δ1 = 2.0
                μ = 0.0
                GS = QuantumNaturalfPEPS.GaussianState(build_H_BdG_mat, N; η=[t1, Δ1, μ], parity_sector=0, target_state=0)

                p_1 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 0))
                p_2 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 1))

                @test p_1 + p_2 ≈ 1.0

                p_vec = zeros(Float64, 2^N)
                for idx in 0:(2^N - 1)
                    occ_dict = Dict(j => ((idx >> (j-1)) & 1) for j in 1:N)
                    p_vec[idx+1] = QuantumNaturalfPEPS.get_prob(GS, occ_dict)
                end
                @test sum(p_vec) ≈ 1.0
                @test QuantumNaturalfPEPS.getParity(GS) == 0
            end

            @testset "Odd parity GS" begin 
                # create tight binding Hamiltonian and corresponding Gaussian state
                L = 2
                N = L*L
                # simple Tight-binding Hamiltonian as test
                t1 = 1.0
                Δ1 = 2.0
                μ = 0.0
                GS = QuantumNaturalfPEPS.GaussianState(build_H_BdG_mat, N; η=[t1, Δ1, μ], parity_sector=1, target_state=0)

                p_1 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 0))
                p_2 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 1))

                @test p_1 + p_2 ≈ 1.0

                p_vec = zeros(Float64, 2^N)
                for idx in 0:(2^N - 1)
                    occ_dict = Dict(j => ((idx >> (j-1)) & 1) for j in 1:N)
                    p_vec[idx+1] = QuantumNaturalfPEPS.get_prob(GS, occ_dict)
                end
                @test sum(p_vec) ≈ 1.0
                @test QuantumNaturalfPEPS.getParity(GS) == 1
            end
        end

        @testset "Prob amplitude" begin
            # Also test decompositions for spectrum with zero modes e.g. (1.0, 1.0, 0.0)
            param_set = [(1.0, 2.0, 3.0), (1.0, 1.0, 1.0), (1.0, 1.0, 0.0), (0.7109298471140131, 0.7035138780787269, 0.12769593636363138)]
            # param_set = [(1.0, 2.0, 3.0), (1.0, 1.0, 1.0), (1.0, 1.0, 0.0)]

            @testset "Even parity GS" begin 
                for (t, Δ, μ) in param_set
                    # create tight binding Hamiltonian and corresponding Gaussian state
                    L = 2
                    N = L*L
                    GS = QuantumNaturalfPEPS.GaussianState(build_H_BdG_mat, N; η=[t, Δ, μ], parity_sector=0, target_state=0)
                    @test QuantumNaturalfPEPS.getParity(GS) == 0

                    for idx in 0:(2^N - 1)
                        occ_string = digits(idx, base=2, pad=N)

                        S_j_square = QuantumNaturalfPEPS.get_prob(GS, occ_string)
                        S_j = QuantumNaturalfPEPS.get_amplitude(GS, occ_string)

                        if !isapprox(S_j_square, abs2(S_j), atol=1e-10)
                            @show occ_string
                        end

                        @test isapprox(S_j_square, abs2(S_j), atol=1e-10)
                    end
                end
            end
            @testset "Odd parity GS" begin 
                for (t, Δ, μ) in param_set
                    # create tight binding Hamiltonian and corresponding Gaussian state
                    L = 2
                    N = L*L
                    GS = QuantumNaturalfPEPS.GaussianState(build_H_BdG_mat, N; η=[t, Δ, μ], parity_sector=1, target_state=0)
                    @test QuantumNaturalfPEPS.getParity(GS) == 1

                    for idx in 0:(2^N - 1)
                        occ_string = digits(idx, base=2, pad=N)

                        S_j_square = QuantumNaturalfPEPS.get_prob(GS, occ_string)
                        S_j = QuantumNaturalfPEPS.get_amplitude(GS, occ_string)

                        if !isapprox(S_j_square, abs2(S_j), atol=1e-10)
                            @show occ_string
                        end

                        @test isapprox(S_j_square, abs2(S_j), atol=1e-10)
                    end
                end
            end
        end
    end

    @testset "Gaussian states (complex MF parameters)" begin

        function build_H_BdG_mat(η, N)
            t_real = η[1]
            t_imag = η[2]
            Δ_real = η[3]
            Δ_imag = η[4]
            μ = η[5]

            t = t_real + 1im * t_imag
            Δ = Δ_real + 1im * Δ_imag

            T = diagm(0 => fill(-μ, N), 1 => fill(-t, N-1), -1 => fill(-conj(t), N-1))
            D = diagm(1 => fill(Δ, N-1), -1 => fill(-Δ, N-1))

            H = [T D; D' -transpose(T)]
            return Hermitian(H)
        end

        @testset "Bogoliubov transformation" begin
            N = 4
            η_vec = [[1.0, 0.3, 2.0, 0.2, 3.0], [1.0, 0.1, 1.0, 0.1, 0.0]]

            for η in η_vec
                H_BdG = build_H_BdG_mat(η, N)

                E, M = QuantumNaturalfPEPS.bogoliubov(H_BdG)

                @test isapprox(M' * M, I, atol=1e-8)
                @test isapprox((M' * H_BdG * M)[1:N, 1:N], Diagonal(E)[1:N, 1:N], atol=1e-8)
            end
        end

        @testset "Covariance matrix construction" begin
            N = 3

            # simple Tight-binding Hamiltonian as test
            t = 1.0
            Δ = 0.0
            μ = 0.0

            H_BdG = build_H_BdG_mat([t, 0.1, Δ, 0.2, μ], N)

            @testset "Even parity GS" begin 
                Γ, _ = QuantumNaturalfPEPS.get_Γ_from_H_BdG(H_BdG, 0; target_state=0)

                @test size(Γ) == (2N, 2N)
                @test Γ*Γ' ≈ I ./ 4
                @test QuantumNaturalfPEPS.getParity(Γ) == 0
            end
            @testset "Odd parity GS" begin 
                Γ, _ = QuantumNaturalfPEPS.get_Γ_from_H_BdG(H_BdG, 1; target_state=0)

                @test size(Γ) == (2N, 2N)
                @test Γ*Γ' ≈ I ./ 4
                @test QuantumNaturalfPEPS.getParity(Γ) == 1
            end
        end

        @testset "Covariance matrix sampling" begin
            @testset "Even parity GS" begin 
                # create tight binding Hamiltonian and corresponding Gaussian state
                L = 2
                N = L*L
                # simple Tight-binding Hamiltonian as test
                t1 = 1.0
                Δ1 = 2.0
                μ = 0.0
                GS = QuantumNaturalfPEPS.GaussianState(build_H_BdG_mat, N; η=[t1, 0.3, Δ1, 0.4, μ], parity_sector=0, target_state=0)

                p_1 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 0))
                p_2 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 1))

                @test p_1 + p_2 ≈ 1.0

                p_vec = zeros(Float64, 2^N)
                for idx in 0:(2^N - 1)
                    occ_dict = Dict(j => ((idx >> (j-1)) & 1) for j in 1:N)
                    p_vec[idx+1] = QuantumNaturalfPEPS.get_prob(GS, occ_dict)
                end
                @test sum(p_vec) ≈ 1.0
                @test QuantumNaturalfPEPS.getParity(GS) == 0
            end

            @testset "Odd parity GS" begin 
                # create tight binding Hamiltonian and corresponding Gaussian state
                L = 2
                N = L*L
                # simple Tight-binding Hamiltonian as test
                t1 = 1.0
                Δ1 = 2.0
                μ = 0.0
                GS = QuantumNaturalfPEPS.GaussianState(build_H_BdG_mat, N; η=[t1, 0.7, Δ1, 0.6, μ], parity_sector=1, target_state=0)

                p_1 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 0))
                p_2 = QuantumNaturalfPEPS.get_prob(GS, Dict(1 => 1))

                @test p_1 + p_2 ≈ 1.0

                p_vec = zeros(Float64, 2^N)
                for idx in 0:(2^N - 1)
                    occ_dict = Dict(j => ((idx >> (j-1)) & 1) for j in 1:N)
                    p_vec[idx+1] = QuantumNaturalfPEPS.get_prob(GS, occ_dict)
                end
                @test sum(p_vec) ≈ 1.0
                @test QuantumNaturalfPEPS.getParity(GS) == 1
            end
        end

        @testset "Prob amplitude" begin
            # Also test decompositions for spectrum with zero modes e.g. (1.0, 1.0, 0.0)
            param_set = [(1.0, 2.0, 3.0), (1.0, 1.0, 1.0), (1.0, 1.0, 0.0), (0.7109298471140131, 0.7035138780787269, 0.12769593636363138)]
            # param_set = [(1.0, 2.0, 3.0), (1.0, 1.0, 1.0), (1.0, 1.0, 0.0)]

            @testset "Even parity GS" begin 
                for (t, Δ, μ) in param_set
                    # create tight binding Hamiltonian and corresponding Gaussian state
                    L = 2
                    N = L*L
                    GS = QuantumNaturalfPEPS.GaussianState(build_H_BdG_mat, N; η=[t, rand(), Δ, rand(), μ], parity_sector=0, target_state=0)
                    @test QuantumNaturalfPEPS.getParity(GS) == 0

                    for idx in 0:(2^N - 1)
                        occ_string = digits(idx, base=2, pad=N)

                        S_j_square = QuantumNaturalfPEPS.get_prob(GS, occ_string)
                        S_j = QuantumNaturalfPEPS.get_amplitude(GS, occ_string)

                        if !isapprox(S_j_square, abs2(S_j), atol=1e-10)
                            @show occ_string
                        end

                        @test isapprox(S_j_square, abs2(S_j), atol=1e-10)
                    end
                end
            end
            @testset "Odd parity GS" begin 
                for (t, Δ, μ) in param_set
                    # create tight binding Hamiltonian and corresponding Gaussian state
                    L = 2
                    N = L*L
                    GS = QuantumNaturalfPEPS.GaussianState(build_H_BdG_mat, N; η=[t, rand(), Δ, rand(), μ], parity_sector=1, target_state=0)
                    @test QuantumNaturalfPEPS.getParity(GS) == 1

                    for idx in 0:(2^N - 1)
                        occ_string = digits(idx, base=2, pad=N)

                        S_j_square = QuantumNaturalfPEPS.get_prob(GS, occ_string)
                        S_j = QuantumNaturalfPEPS.get_amplitude(GS, occ_string)

                        if !isapprox(S_j_square, abs2(S_j), atol=1e-10)
                            @show occ_string
                        end

                        @test isapprox(S_j_square, abs2(S_j), atol=1e-10)
                    end
                end
            end
        end
    end
end;