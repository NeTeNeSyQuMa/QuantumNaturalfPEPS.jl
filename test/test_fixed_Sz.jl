using Test
using LinearAlgebra
using QuantumNaturalfPEPS

@testset "Fixed Sz sector" begin
    @testset "Parton indexing" begin
        Lx, Ly = 2,2
        L = Lx * Ly

        @testset "S=1/2" begin
            nflavours = 2

            N = nflavours * L

            @test QuantumNaturalfPEPS.parton_flavour(1, nflavours) == 1
            @test QuantumNaturalfPEPS.parton_flavour(2, nflavours) == 2
            @test QuantumNaturalfPEPS.parton_flavour(3, nflavours) == 1
            @test QuantumNaturalfPEPS.parton_flavour(4, nflavours) == 2
            @test QuantumNaturalfPEPS.parton_flavour(5, nflavours) == 1
            @test QuantumNaturalfPEPS.parton_flavour(6, nflavours) == 2
            @test QuantumNaturalfPEPS.parton_flavour(7, nflavours) == 1
            @test QuantumNaturalfPEPS.parton_flavour(8, nflavours) == 2

            @test QuantumNaturalfPEPS.parton_site(1, nflavours) == 1
            @test QuantumNaturalfPEPS.parton_site(2, nflavours) == 1
            @test QuantumNaturalfPEPS.parton_site(3, nflavours) == 2
            @test QuantumNaturalfPEPS.parton_site(4, nflavours) == 2
            @test QuantumNaturalfPEPS.parton_site(5, nflavours) == 3
            @test QuantumNaturalfPEPS.parton_site(6, nflavours) == 3
            @test QuantumNaturalfPEPS.parton_site(7, nflavours) == 4
            @test QuantumNaturalfPEPS.parton_site(8, nflavours) == 4
        end
        @testset "S=1" begin
            nflavours = 3
            N = nflavours * L

            @test QuantumNaturalfPEPS.parton_flavour(1, nflavours) == 1
            @test QuantumNaturalfPEPS.parton_flavour(2, nflavours) == 2
            @test QuantumNaturalfPEPS.parton_flavour(3, nflavours) == 3
            @test QuantumNaturalfPEPS.parton_flavour(4, nflavours) == 1
            @test QuantumNaturalfPEPS.parton_flavour(5, nflavours) == 2
            @test QuantumNaturalfPEPS.parton_flavour(6, nflavours) == 3
            @test QuantumNaturalfPEPS.parton_flavour(7, nflavours) == 1
            @test QuantumNaturalfPEPS.parton_flavour(8, nflavours) == 2
            @test QuantumNaturalfPEPS.parton_flavour(9, nflavours) == 3
            @test QuantumNaturalfPEPS.parton_flavour(10, nflavours) == 1
            @test QuantumNaturalfPEPS.parton_flavour(11, nflavours) == 2
            @test QuantumNaturalfPEPS.parton_flavour(12, nflavours) == 3

            @test QuantumNaturalfPEPS.parton_site(1, nflavours) == 1
            @test QuantumNaturalfPEPS.parton_site(2, nflavours) == 1
            @test QuantumNaturalfPEPS.parton_site(3, nflavours) == 1
            @test QuantumNaturalfPEPS.parton_site(4, nflavours) == 2
            @test QuantumNaturalfPEPS.parton_site(5, nflavours) == 2
            @test QuantumNaturalfPEPS.parton_site(6, nflavours) == 2
            @test QuantumNaturalfPEPS.parton_site(7, nflavours) == 3
            @test QuantumNaturalfPEPS.parton_site(8, nflavours) == 3
            @test QuantumNaturalfPEPS.parton_site(9, nflavours) == 3
            @test QuantumNaturalfPEPS.parton_site(10, nflavours) == 4
            @test QuantumNaturalfPEPS.parton_site(11, nflavours) == 4
            @test QuantumNaturalfPEPS.parton_site(12, nflavours) == 4
        end

    end
end

# checks if the BdG Hamiltonian preserves Sz
@testset "BdG matrix (NN) with fixed Sz sector" begin
    function check_fixed_Sz_BdG(Lx,Ly, nflavours)
        parton_site = QuantumNaturalfPEPS.parton_site
        parton_flavour = QuantumNaturalfPEPS.parton_flavour

        # sites of the original lattice that may be connected: equal, x-neighbours (same row) or y-neighbours
        is_onsite_or_NN(i, j) = (i == j) ||
            (abs(i - j) == 1 && div(i - 1, Lx) == div(j - 1, Lx)) ||
            (abs(i - j) == Lx)

        N = nflavours * Lx * Ly

        η = ones(QuantumNaturalfPEPS.get_max_num_MF_params_NN_parton(Lx, Ly, nflavours))
        H_bdG = QuantumNaturalfPEPS.build_general_H_BdG_2D_NN_fixed_Sz(η, Lx, Ly, nflavours)

        @test size(H_bdG) == (2N, 2N)

        T = H_bdG[1:N, 1:N]
        D = H_bdG[1:N, N+1:end]

        @test diag(T) == ones(N)   # on-site potentials are flavour preserving
        @test T ≈ T'               # hopping block is hermitian
        @test D ≈ -transpose(D)    # pairing block is antisymmetric

        for i in 1:N, j in 1:N
            si, sj = parton_site(i, nflavours), parton_site(j, nflavours)
            f1, f2 = parton_flavour(i, nflavours), parton_flavour(j, nflavours)

            if !is_onsite_or_NN(si,sj) continue end

            if f1==f2
                # c†_{i,f1} c_{j,f2} changes the total flavour unless f1 == f2
                @test T[i, j] == 1

                # c_{i,f1} c_{j,f2} changes the total flavour unless the flavours are partners (flavour_f1 + flavour_f2 = nflavours + 1)
                if f1 + f2 == nflavours + 1 && i!=j

                    @test abs(D[i, j]) == 1
                else
                    @test D[i, j] == 0
                end
            else
                @test T[i, j] == 0
                
                # c_{i,f1} c_{j,f2} changes the total flavour unless the flavours are partners (flavour_f1 + flavour_f2 = nflavours + 1)
                f1 + f2 == nflavours + 1 || @test abs(D[i, j]) == 0
            end
        end
    end

    @testset "S=1/2" begin
        check_fixed_Sz_BdG(2,2,2)
        check_fixed_Sz_BdG(3,2,2)
    end

    @testset "S=1" begin
        check_fixed_Sz_BdG(3,3,3)
        check_fixed_Sz_BdG(2,3,3)
    end
end

#=
    Physical test case 1: hopping only, no pairing.

    A flavour-split on-site term (μ = -h on one flavour, +h on all others, with h well above the
    hopping bandwidth) separates the flavour bands, so the ground state fills every mode of that one
    flavour and nothing else. With no pairing the state is a Slater determinant, so every occupation is
    exactly 0 or 1 and the whole state is known in closed form -- which makes this the case that pins
    the sector selection independently of the Bloch-Messiah / pfaffian machinery.
=#
@testset "hopping only: fully polarized ground state (analytic)" begin
    QNF = QuantumNaturalfPEPS

    # μ = -h on the filled flavour, +h on all others, uniform hopping, pairing left at exactly zero.
    function polarized_η(Lx, Ly, nf, f_fill; h=8.0, t=0.2)
        N = nf * Lx * Ly
        η = zeros(QNF.get_max_num_MF_params_NN_parton(Lx, Ly, nf))
        η[1:N] .= [QNF.parton_flavour(m, nf) == f_fill ? -h : h for m in 1:N]
        nhop = nf * (QNF.get_max_num_hopping_x_NN(Lx, Ly) + QNF.get_max_num_hopping_y_NN(Lx, Ly))
        η[N+1 : N+nhop] .= -t
        return η
    end

    # Lx*Ly is odd in two of these, so S*Lx*Ly is a genuine half-integer target
    for (Lx, Ly, nf) in ((2, 2, 2), (3, 1, 2), (3, 3, 2), (2, 2, 3))
        L = Lx * Ly
        N = nf * L
        # the polarized state holds exactly L partons, so its fermion parity is L mod 2
        ps = L % 2
        H_func(θ, n) = QNF.build_general_H_BdG_2D_NN_fixed_Sz(θ, Lx, Ly, nf)

        @testset "$(Lx)x$(Ly), n_flavours=$nf" begin
            for f_fill in 1:nf
                η = polarized_η(Lx, Ly, nf, f_fill)
                s_fill = QNF.sz_per_mode(nf, nf)[f_fill]   # Sz carried by the filled flavour
                Sz_expected = s_fill * L

                # the pairing block really is zero, so this is a pure Slater determinant
                @test norm(Matrix(H_func(η, N))[1:N, N+1:end], Inf) == 0.0

                # --- the plain ground state already is the polarized state ------------------------
                GS = QNF.GaussianState(H_func, N; η=η, parity_sector=ps, target_state=0, n_flavours=nf)
                n_occ = [0.5 - real(GS.Γ[2m-1, 2m]) for m in 1:N]
                n_exact = [QNF.parton_flavour(m, nf) == f_fill ? 1.0 : 0.0 for m in 1:N]

                @test maximum(abs, n_occ .- n_exact) < 1e-12          # occupations are exactly 0 / 1
                @test isapprox(QNF.get_Sz_from_Γ(GS), Sz_expected; atol=1e-12)
                @test QNF.getParity(GS.Γ) == ps

                # --- requesting that Sz must reproduce the very same state ------------------------
                GS_t = QNF.GaussianState(H_func, N; η=η, parity_sector=ps, target_Sz=Sz_expected, n_flavours=nf)
                @test isapprox(QNF.get_Sz_from_Γ(GS_t), Sz_expected; atol=1e-12)
                @test GS_t.Γ ≈ GS.Γ
                @test GS_t.occ_ref == zeros(Int, N)   # the Bogoliubov vacuum already is the sector
            end

            # --- the opposite polarization is the mirror state ------------------------------------
            # flavour 1 carries +S, flavour nf carries -S
            GS_up = QNF.GaussianState(H_func, N; η=polarized_η(Lx, Ly, nf, 1),
                                      parity_sector=ps, target_Sz=+(nf - 1) / 2 * L, n_flavours=nf)
            GS_dn = QNF.GaussianState(H_func, N; η=polarized_η(Lx, Ly, nf, nf),
                                      parity_sector=ps, target_Sz=-(nf - 1) / 2 * L, n_flavours=nf)
            @test isapprox(QNF.get_Sz_from_Γ(GS_up), -QNF.get_Sz_from_Γ(GS_dn); atol=1e-12)
            @test isapprox(QNF.get_Sz_from_Γ(GS_up), (nf - 1) / 2 * L; atol=1e-12)

            if nf == 2
                # For S=1/2 the maximally polarized state is unique: it needs all L modes of one
                # flavour and nothing else, so its parton count -- and hence its parity -- is fixed.
                # The same Sz in the other parity sector therefore does not exist, and must be
                # rejected rather than silently answered with some other state.
                # (Not true for nf >= 3, where Sz = S*L is also reachable at a different parton count.)
                @test_throws AssertionError QNF.GaussianState(H_func, N; η=polarized_η(Lx, Ly, nf, 1),
                                                              parity_sector=1 - ps, target_Sz=L / 2, n_flavours=nf)
            end
        end
    end
end

#=
    Physical test case 2: hopping AND pairing, both switched on.

    With pairing the occupations are fractional and the amplitudes go through the Bloch-Messiah /
    pfaffian path, so this exercises the machinery end to end. Uniform η is included deliberately: it
    is the fully spin-degenerate mean field, i.e. the one that fails without the Sz alignment of the
    degenerate Bogoliubov multiplets, and it is where an optimization starts.
=#
@testset "hopping + pairing: ground state lands in the requested Sz sector" begin
    using Random
    QNF = QuantumNaturalfPEPS

    for (Lx, Ly, nf) in ((2, 2, 2), (2, 2, 3))
        L = Lx * Ly
        N = nf * L
        S = (nf - 1) / 2
        H_func(θ, n) = QNF.build_general_H_BdG_2D_NN_fixed_Sz(θ, Lx, Ly, nf)
        nη = QNF.get_max_num_MF_params_NN_parton(Lx, Ly, nf)
        Random.seed!(2026 + nf)

        @testset "$(Lx)x$(Ly), n_flavours=$nf" begin
            for (η_label, η) in (("uniform", ones(nη)), ("random", randn(nη)))
                @testset "$η_label eta" begin
                    # the mean field conserves Sz for whatever η it is handed
                    Hm = Matrix(H_func(η, N))
                    Sz_op = QNF.sz_operator_BdG(N, nf)
                    @test norm(Hm * Sz_op - Sz_op * Hm, Inf) < 1e-12 # commutes with Sz

                    n_reachable = 0
                    for ps in (0, 1), tSz in (-S*L):0.5:(S*L)
                        local GS
                        try
                            GS = QNF.GaussianState(H_func, N; η=η, parity_sector=ps,
                                                   target_Sz=tSz, n_flavours=nf)
                        catch err
                            # not every (Sz, parity) pair exists; it has to say so rather than
                            # silently hand back a state from a different sector
                            err isa AssertionError || rethrow()
                            continue
                        end
                        n_reachable += 1
                        configs() = (digits(i, base=2, pad=N) for i in 0:(2^N - 1))

                        @test isapprox(QNF.get_Sz_from_Γ(GS), tSz; atol=1e-9)   # requested sector
                        @test QNF.getParity(GS.Γ) == ps                         # requested parity
                        @test isapprox(sum(QNF.get_prob(GS, c) for c in configs()), 1.0; atol=1e-9) # normalization correct
                        @test isapprox(sum(abs2(QNF.get_amplitude(GS, c)) for c in configs()), 1.0; atol=1e-9) # normalization correct
                        @test maximum(abs(QNF.get_prob(GS, c) - abs2(QNF.get_amplitude(GS, c))) for c in configs()) < 1e-9 # probabilities and amplitudes consistent
                    end
                    @test n_reachable > 0 # at least one Sz sector must be reachable for any η

                    # a target far outside the reachable range must raise, not clamp
                    @test_throws AssertionError QNF.GaussianState(H_func, N; η=η, parity_sector=0,
                                                                  target_Sz=S * L + 5, n_flavours=nf)
                end
            end
        end
    end
end;
