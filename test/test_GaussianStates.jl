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

    @testset "Bloch-Messiah decomposition robustness" begin
        # When the pairing matrix P carries no useful information within a degenerate
        # subspace of Q (‖P_sub‖_∞ < 1e-10, i.e. an empty / fully occupied Slater block),
        # the gauge is underdetermined. The decomposition must fall back to a stable
        # default gauge (S_sub = I) instead of failing. We scan distinct phase-diagram
        # points on a deterministic meshgrid of the mean-field parameters
        # (μ, t, Δ ∈ [-2, 2]) for both 1D and 2D systems.

        # Build a general 1D nearest-neighbour BdG Hamiltonian from the combined mean-field
        # vector η = [μ_1..μ_L, t_1..t_{L-1}, Δ_1..Δ_{L-1}] on an open chain of L sites.
        function build_H_BdG_1D(η::AbstractVector{<:Number}, L::Int)
            @assert length(η) == 3L - 2 "η must have length 3L-2 = $(3L-2) for a 1D chain of L=$L sites"
            μs = η[1:L]
            ts = η[L+1 : 2L-1]
            Δs = η[2L : 3L-2]
            T = diagm(0 => -μs, 1 => -ts, -1 => -conj.(ts))
            D = diagm(1 => Δs, -1 => -Δs)
            H = [T D; D' -transpose(T)]
            return Hermitian(Matrix(H))
        end

        # Build the combined mean-field vectors (scalar μ, t, Δ on every site / bond)
        combined_η_1D(μ, t, Δ, L) = vcat(fill(float(μ), L), fill(float(t), L - 1), fill(float(Δ), L - 1))
        function combined_η_2D(μ, t, Δ, Lx, Ly)
            N = Lx * Ly
            nhx = QuantumNaturalfPEPS.get_max_num_hopping_x_NN(Lx, Ly)
            nhy = QuantumNaturalfPEPS.get_max_num_hopping_y_NN(Lx, Ly)
            return vcat(fill(float(μ), N), fill(float(t), nhx), fill(float(t), nhy),
                        fill(float(Δ), nhx), fill(float(Δ), nhy))
        end

        # Diagonalize H with the Bogoliubov transformation and feed the resulting M straight
        # into the Bloch-Messiah decomposition, then verify M is faithfully reconstructed
        # (the decomposition additionally asserts unitarity of its factors internally).
        function check_bloch_messiah(H_BdG)
            _, M = QuantumNaturalfPEPS.bogoliubov(H_BdG)
            Dmat, UVmat, Cmat = QuantumNaturalfPEPS.bloch_messiah_decomposition(M)
            return isapprox(M, Dmat * UVmat * Cmat; atol=1e-8)
        end

        # Sweep the (μ, t, Δ) meshgrid and report the number of failing points together
        # with the first failing point for diagnostics.
        function sweep_grid(build_H, μ_vals, t_vals, Δ_vals)
            n_fail = 0
            first_fail = nothing
            for μ in μ_vals, t in t_vals, Δ in Δ_vals
                ok = true
                try
                    ok = check_bloch_messiah(build_H(μ, t, Δ))
                catch err
                    ok = false
                end
                if !ok
                    n_fail += 1
                    first_fail === nothing && (first_fail = (μ = μ, t = t, Δ = Δ))
                end
            end
            return n_fail, first_fail
        end

        #= 
            TODO: Is the grid fine enough?
        =#
        μ_vals = range(-2.0, 2.0; length=9) # Coarse μ grid
        t_vals = range(-2.0, 2.0; length=100)
        Δ_vals = range(-2.0, 2.0; length=100)

        @testset "1D systems ($(length(μ_vals)) × $(length(t_vals)) × $(length(Δ_vals)) meshgrid)" begin
            L = 6
            n_fail, first_fail = sweep_grid((μ, t, Δ) -> build_H_BdG_1D(combined_η_1D(μ, t, Δ, L), L), μ_vals, t_vals, Δ_vals)
            if n_fail > 0
                @show first_fail
            end
            @test n_fail == 0
        end

        @testset "2D systems ($(length(μ_vals)) × $(length(t_vals)) × $(length(Δ_vals)) meshgrid)" begin
            Lx, Ly = 3, 2
            n_fail, first_fail = sweep_grid((μ, t, Δ) -> QuantumNaturalfPEPS.build_general_H_BdG_2D_NN(combined_η_2D(μ, t, Δ, Lx, Ly), Lx, Ly), μ_vals, t_vals, Δ_vals)
            if n_fail > 0
                @show first_fail
            end
            @test n_fail == 0
        end

        #= NOTE: Whenever we encouter problematic points, we can add them here to cover these errors properly =#
        @testset "manual edge cases (empty / fully occupied Slater blocks)" begin
            L = 6
            # Δ = 0 gives a pure hopping model: no pairing, so all degenerate Q blocks are
            # empty / fully occupied and exercise the S_sub = I fallback branch.
            @test check_bloch_messiah(build_H_BdG_1D(combined_η_1D(0.5, 1.0, 0.0, L), L))

            # Uniform chain at half filling with a zero mode (t = Δ = 1, μ = 0).
            @test check_bloch_messiah(build_H_BdG_1D(combined_η_1D(0.0, 1.0, 1.0, L), L))

            # 2D square lattice (t =- 1, μ = 0, Δ = 0).
            Lx, Ly = 4, 4
            N = Lx * Ly
            n_max_MF_params = QuantumNaturalfPEPS.get_max_num_MF_params_NN(Lx, Ly)
            η = zeros(Float64, n_max_MF_params)
            nx = QuantumNaturalfPEPS.get_max_num_hopping_x_NN(Lx, Ly)
            ny = QuantumNaturalfPEPS.get_max_num_hopping_y_NN(Lx, Ly)
            # hopping
            hx_range = N+1 : N+nx
            hy_range = N+nx+1 : N+nx+ny
            t_mf = 1.0 # small mean-field hopping
            η[hx_range] .= -t_mf
            η[hy_range] .= -t_mf
            @test check_bloch_messiah(QuantumNaturalfPEPS.build_general_H_BdG_2D_NN(η, Lx, Ly))
        end
    end
end;