using Test
using LinearAlgebra
using QuantumNaturalfPEPS

const QNF = QuantumNaturalfPEPS

# Sz read straight off the covariance matrix: n_m = 1/2 - Γ[2m-1, 2m] in the Majorana (qq) convention.
# With pairing switched on the individual n_m are fractional, but this combination is exactly a
# half-integer whenever the mean field conserves Sz -- which is what makes it a usable assertion.
function Sz_from_Γ(Γ, nf)
    N = size(Γ, 1) ÷ 2
    s = [(nf + 1) / 2 - QNF.parton_flavour(m, nf) for m in 1:N]
    n = [0.5 - real(Γ[2m-1, 2m]) for m in 1:N]
    return sum(s .* n)
end

@testset "Fixed Sz sector" begin

    @testset "flavour-conserving mean field" begin
        for (Lx, Ly, nf) in ((2, 2, 2), (3, 2, 2), (2, 2, 3), (2, 2, 4))
            L = Lx * Ly
            N = nf * L
            nη = QNF.get_max_num_MF_params_NN_parton(Lx, Ly, nf)
            η = randn(nη)
            H = QNF.build_general_H_BdG_2D_NN_fixed_parton_flavour(η, Lx, Ly, nf)

            T = H[1:N, 1:N]
            D = H[1:N, N+1:end]

            @test ishermitian(H)
            @test norm(D + transpose(D), Inf) < 1e-12

            # [H_BdG, Ŝz] = 0 for arbitrary η, because the violating terms have no parameter at all
            s = [(nf + 1) / 2 - QNF.parton_flavour(m, nf) for m in 1:N]
            Sz = Diagonal(vcat(s, -s))
            @test norm(Matrix(H) * Sz - Sz * Matrix(H), Inf) < 1e-12

            for m in 1:N, m2 in 1:N
                f1 = QNF.parton_flavour(m, nf)
                f2 = QNF.parton_flavour(m2, nf)
                f1 != f2          && @test T[m, m2] == 0   # no flavour-changing hopping
                f1 + f2 != nf + 1 && @test D[m, m2] == 0   # no Sz-charged pairing
            end
        end

        # n_flavours = 1 must reproduce the existing spinless parameter count
        for (Lx, Ly) in ((2, 2), (3, 3), (3, 4))
            @test QNF.get_max_num_MF_params_NN_parton(Lx, Ly, 1) == QNF.get_max_num_MF_params_NN(Lx, Ly)
        end
    end

    @testset "occ_ref selection lands in the requested sector" begin
        for (Lx, Ly, nf) in ((2, 2, 2), (3, 2, 2), (2, 2, 3))
            L = Lx * Ly
            N = nf * L
            nη = QNF.get_max_num_MF_params_NN_parton(Lx, Ly, nf)
            H = QNF.build_general_H_BdG_2D_NN_fixed_parton_flavour(randn(nη), Lx, Ly, nf)

            for parity in 0:1
                reached = 0
                for twoSz in -N:N
                    target = twoSz / 2
                    local Γ, occ
                    try
                        Γ, occ = QNF.get_Γ_from_H_BdG(H, parity; target_Sz=target, n_flavours=nf)
                    catch e
                        e isa AssertionError || rethrow()
                        continue   # not reachable in this parity sector
                    end
                    reached += 1

                    @test isapprox(Sz_from_Γ(Γ, nf), target; atol=1e-9)   # the requested sector
                    @test isapprox(Γ * Γ, -I(2N) / 4; atol=1e-9)          # still a pure Gaussian state
                    @test isapprox(Γ, -transpose(Γ); atol=1e-10)
                    @test QNF.getParity(Γ) == parity                      # parity stays an independent knob
                    @test length(occ) == N && all(x -> x in (0, 1), occ)
                end
                @test reached > 0
            end
        end
    end

    @testset "reference is energy-minimal and deterministic" begin
        Lx, Ly, nf = 2, 2, 2
        N = nf * Lx * Ly
        nη = QNF.get_max_num_MF_params_NN_parton(Lx, Ly, nf)
        H = QNF.build_general_H_BdG_2D_NN_fixed_parton_flavour(randn(nη), Lx, Ly, nf)

        _, o1 = QNF.get_Γ_from_H_BdG(H, 0; target_Sz=0.0, n_flavours=nf)
        _, o2 = QNF.get_Γ_from_H_BdG(H, 0; target_Sz=0.0, n_flavours=nf)
        @test o1 == o2

        # brute force every reference and confirm the DP found the cheapest one in the sector
        E, M = QNF.bogoliubov(H)
        parity_vac = QNF.getParity(QNF.get_Γ0_from_H_BdG(H))
        occ = QNF.select_occ_ref_by_Sz(M, E, 0, parity_vac, 0.0, nf)
        cost = sum(E[k] for k in 1:N if occ[k] == 1; init=0.0)

        U, V = QNF.get_bogoliubov_blocks(M)
        s = [(nf + 1) / 2 - QNF.parton_flavour(m, nf) for m in 1:N]
        Sz_vac = sum(s .* vec(sum(abs2.(V), dims=2)))
        q = vec(transpose(s) * (abs2.(U) .- abs2.(V)))

        best = Inf
        for mask in 0:(2^N - 1)
            o = [(mask >> (k - 1)) & 1 for k in 1:N]
            isapprox(Sz_vac + sum(o .* q), 0.0; atol=1e-9) || continue
            (sum(o) % 2) == parity_vac % 2 || continue
            best = min(best, sum(E[k] for k in 1:N if o[k] == 1; init=0.0))
        end
        @test isapprox(cost, best; atol=1e-9)
    end

    @testset "Zeeman field shifts the vacuum but not the selection" begin
        Lx, Ly, nf = 2, 2, 2
        N = nf * Lx * Ly
        nη = QNF.get_max_num_MF_params_NN_parton(Lx, Ly, nf)
        η = randn(nη)
        η[1:N] .= [QNF.parton_flavour(m, nf) == 1 ? -0.8 : 0.8 for m in 1:N]
        H = QNF.build_general_H_BdG_2D_NN_fixed_parton_flavour(η, Lx, Ly, nf)

        for target in (-1.0, 0.0, 1.0)
            Γ, _ = QNF.get_Γ_from_H_BdG(H, 0; target_Sz=target, n_flavours=nf)
            @test isapprox(Sz_from_Γ(Γ, nf), target; atol=1e-9)
        end
    end

    @testset "an Sz-breaking mean field is rejected" begin
        Lx, Ly = 2, 2
        H = QNF.build_general_H_BdG_2D_NN(randn(QNF.get_max_num_MF_params_NN(Lx, Ly)), Lx, Ly)
        @test_throws AssertionError QNF.get_Γ_from_H_BdG(H, 0; target_Sz=0.0, n_flavours=2)
    end

    @testset "default (index-and-parity) fill is unchanged" begin
        Lx, Ly = 2, 2
        H = QNF.build_general_H_BdG_2D_NN(randn(QNF.get_max_num_MF_params_NN(Lx, Ly)), Lx, Ly)
        Γa, oa = QNF.get_Γ_from_H_BdG(H, 0)
        Γb, ob = QNF.get_Γ_from_H_BdG(H, 0; target_state=1)
        @test QNF.getParity(Γa) == 0
        @test QNF.getParity(Γb) == 0
        @test sum(oa) + 2 == sum(ob)
    end

    @testset "write! holds the sector across parameter updates" begin
        Lx, Ly, nf = 2, 2, 2
        N = nf * Lx * Ly
        nη = QNF.get_max_num_MF_params_NN_parton(Lx, Ly, nf)
        H_func = (η, n) -> QNF.build_general_H_BdG_2D_NN_fixed_parton_flavour(η, Lx, Ly, nf)

        GS = QNF.GaussianState(H_func, N; η=randn(nη), parity_sector=0, target_Sz=1.0, n_flavours=nf)
        @test isapprox(Sz_from_Γ(GS.Γ, nf), 1.0; atol=1e-9)

        for _ in 1:5
            QNF.write!(GS, randn(nη))
            @test isapprox(Sz_from_Γ(GS.Γ, nf), 1.0; atol=1e-9)
            @test QNF.getParity(GS.Γ) == 0
        end
    end
end
