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

        @testset "Marginal probabilities at zero modes (sampling)" begin
            # The joint sampler draws site-by-site from the Gaussian marginal `get_prob(GS, occ_dict)`, so
            # its robustness rests on that marginal being (i) normalized and (ii) consistent with summing
            # the full distribution over the unmeasured sites. We check both at parameter points with zero
            # modes / degenerate spectra (t=Δ, μ=0) AND at a particle-number-conserving Slater point (Δ=0,
            # all Bloch-Messiah blocks fully occupied/empty). Where `get_amplitude` is itself normalized
            # (Δ≠0) we additionally cross-check it as an independent ground truth.
            param_set = [(1.0, 1.0, 0.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.5), (2.0, 2.0, 0.0)]
            N = 4
            for ps in (0, 1), (t, Δ, μ) in param_set
                GS = QuantumNaturalfPEPS.GaussianState(build_H_BdG_mat, N; η=[t, Δ, μ], parity_sector=ps, target_state=0)
                # (i) the full Gaussian distribution normalizes to 1
                p_full(occ) = QuantumNaturalfPEPS.get_prob(GS, occ)
                @test isapprox(sum(p_full(digits(idx, base=2, pad=N)) for idx in 0:(2^N - 1)), 1.0; atol=1e-8)
                for k in 1:(N - 1), pidx in 0:(2^k - 1)
                    prefix = digits(pidx, base=2, pad=k)
                    # (ii) the marginal of the first k sites equals summing the full distribution over the rest
                    summed = sum(p_full(vcat(prefix, digits(s, base=2, pad=N - k))) for s in 0:(2^(N - k) - 1))
                    marginal = QuantumNaturalfPEPS.get_prob(GS, Dict(i => prefix[i] for i in 1:k))
                    @test isapprox(marginal, summed; atol=1e-8)
                end
                # independent ground truth where the amplitude is normalized (no fully occupied/empty blocks)
                if Δ != 0
                    for idx in 0:(2^N - 1)
                        occ = digits(idx, base=2, pad=N)
                        @test isapprox(p_full(occ), abs2(QuantumNaturalfPEPS.get_amplitude(GS, occ)); atol=1e-10)
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

        @testset "truncation of null Vbar columns at the noise floor" begin
            # Regression: for a Slater determinant (no pairing, e.g. a π-flux hopping state)
            # half the columns of Vbar vanish and their numerical noise floor sits at ~1e-10 —
            # exactly where an *absolute* 1e-10 cutoff stops recognising them as zero. A null
            # column then survives, Q_mat comes out one dimension too large, and pfaffian()
            # returns 0 for *every* configuration: logψ = -Inf everywhere, so the importance
            # weights come out NaN and the optimisation dies with
            # "ArgumentError: weights cannot contain Inf or NaN values".
            # truncated_bloch_messiah only slices blocks, so synthetic input suffices here:
            # n_pair paired modes (v_p ~ 0.5) followed by n_null columns of pure noise.
            function synthetic_bloch_messiah(n_pair, n_null, noise)
                n = n_pair + n_null
                Vbar = zeros(ComplexF64, n, n)
                for p in 1:2:n_pair
                    v = 0.3 + 0.05p # deterministic stand-in for a Bloch-Messiah singular value
                    Vbar[p, p+1] = im * v
                    Vbar[p+1, p] = -im * v
                end
                for c in n_pair+1:n, r in 1:n
                    Vbar[r, c] = noise * (1 + 0.1r) # deterministic stand-in for eigensolver noise
                end
                Ubar = Matrix{ComplexF64}(I, n, n)
                D = Matrix{ComplexF64}(I, n, n)
                C = Matrix{ComplexF64}(I, n, n)
                return ([D zeros(size(D)); zeros(size(D)) conj.(D)],
                        [Ubar Vbar; Vbar Ubar],
                        [C zeros(size(C)); zeros(size(C)) conj.(C)])
            end

            # The null columns must be truncated at every noise level, not just far below 1e-10.
            n_pair = 8
            n_null = 8
            for noise in (1e-16, 1e-13, 1e-11, 1e-10, 5e-10)
                Dmat, UVmat, Cmat = synthetic_bloch_messiah(n_pair, n_null, noise)
                _, UVmat_prime, _ = QuantumNaturalfPEPS.truncated_bloch_messiah(Dmat, UVmat, Cmat)
                @test size(UVmat_prime, 1) ÷ 2 == n_pair # the null columns are truncated away
            end

            # A Vbar that is nothing but noise carries no paired modes at all and must still be
            # truncated away completely.
            Dmat, UVmat, Cmat = synthetic_bloch_messiah(0, n_null, 1e-13)
            _, UVmat_prime, _ = QuantumNaturalfPEPS.truncated_bloch_messiah(Dmat, UVmat, Cmat)
            @test size(UVmat_prime, 1) ÷ 2 == 0 # no paired modes in this test
        end
    end

    @testset "truncate null modes of a physical pi-flux Slater state" begin
        Lx = Ly = 4
        N = Lx * Ly

        # pi-flux hopping matrix on an Lx x Ly open lattice, site (i, j) -> i + (j-1) * Lx
        function build_T_pi_flux(t, Lx, Ly)
            N = Lx * Ly
            hopping_x = [(a % Lx == 0) ? 0.0 : -t for a in 1:N-1]   # every Lx-th link crosses the boundary
            hopping_y = [-t * (-1)^mod1(a, Lx) for a in 1:N-Lx]     # staggered sign -> pi flux per plaquette

            T = diagm(1 => hopping_x, -1 => conj.(hopping_x), Lx => hopping_y, -Lx => conj.(hopping_y))
            return T
        end

        T = build_T_pi_flux(1.0, Lx, Ly)

        # This is the BdG representation of H = sum_ab T_ab c†_a c_b.
        pairing = zeros(ComplexF64, N, N)
        H_BdG = Hermitian([T pairing; pairing -transpose(T)])

        @test T ≈ T'
        @test iszero(H_BdG[1:N, N+1:2N])

        _, M = QuantumNaturalfPEPS.bogoliubov(H_BdG) # M diagonalizes H_BdG 
        Dmat, UVmat, Cmat =
            QuantumNaturalfPEPS.bloch_messiah_decomposition(M)
        _, _, Vbar, _ =
            QuantumNaturalfPEPS.get_mats_from_bloch_messiah(Dmat, UVmat, Cmat)

        # At half filling Vbar has rank N/2: eight active occupied modes followed
        # by eight inactive modes. The truncation must remove the inactive half.
        @test rank(Vbar) == N ÷ 2
        @test all(maximum.(abs, eachcol(Vbar)[1:N÷2]) .> 1e-2) # the active half is well above the noise floor
        @test all(maximum.(abs, eachcol(Vbar)[N÷2+1:end]) .≈ 0.0) # the inactive half is zero as we dont have pairing

        Dmat_trunc, UVmat_trunc, Cmat_trunc =
            QuantumNaturalfPEPS.truncated_bloch_messiah(Dmat, UVmat, Cmat)

        @test size(Dmat_trunc) == (2N, N)
        @test size(Cmat_trunc) == (N, 2N)
        @test rank(UVmat_trunc[1:(N ÷ 2), 1:(N ÷ 2)]) == 0 # the truncated Vbar is empty

        Vbar_trunc = UVmat_trunc[(N ÷ 2 + 1):end, 1:(N ÷ 2)]
        @test Vbar_trunc' * Vbar_trunc ≈ I
    end

    @testset "H_BdG derivatives" begin
        # NN hopping chain with one parameter per bond: H = [T 0; 0 -Tᵀ] with
        # T[i, i+1] = -η[i], T[i+1, i] = -conj(η[i]). Each Jacobian column is then
        # known exactly — a handful of ±1 entries on bond `a` and zeros everywhere else.
        N = 4
        function hopping_H_BdG(η, N)
            Z = zeros(eltype(η), N, N)
            T = diagm(1 => -collect(η), -1 => -conj.(collect(η)))
            return Hermitian([T Z; Z -transpose(T)])
        end

        @testset "real MF parameters" begin
            η = [0.7, -1.3, 0.2]
            dHs = QuantumNaturalfPEPS.build_H_BdG_derivatives(hopping_H_BdG, η, N)

            @test length(dHs) == length(η)
            for a in eachindex(η)
                expected = zeros(Float64, 2N, 2N)
                expected[a, a+1] = expected[a+1, a] = -1.0
                expected[N+a, N+a+1] = expected[N+a+1, N+a] = 1.0
                @test dHs[a] ≈ expected
            end
        end

        @testset "complex MF parameters" begin
            # Wirtinger ∂/∂ηₐ: the conjugated entries are antiholomorphic (f(z) = i+iv -> conj(f) = i-iv) and vanish.
            # so only T[a, a+1] and (-Tᵀ)[N+a+1, N+a] survive.
            η = ComplexF64[0.7 + 0.4im, -1.3 - 0.9im, 0.2im]
            dHs = QuantumNaturalfPEPS.build_H_BdG_derivatives(hopping_H_BdG, η, N)

            @test length(dHs) == length(η)
            for a in eachindex(η)
                expected = zeros(ComplexF64, 2N, 2N)
                expected[a, a+1] = -1.0
                expected[N+a+1, N+a] = 1.0
                @test dHs[a] ≈ expected
            end
        end
    end
end;