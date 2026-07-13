using Test
using ITensors
using QuantumNaturalfPEPS
using QuantumNaturalGradient
using LinearAlgebra
using Random

Random.seed!(1234)

@testset "4x4 Hubbard model half-filling" begin
    # set up model
    Lx, Ly = 4, 4
    N = Lx * Ly
    parity_sector = 0
    target_state = 0

    # the solutions in these tests are all D=1 PEPS. A constant identity PEPS helps for convergence to make the test run faster
    # NOTE: In the \example folder, we should use a random PEPS for the these parameters just to see that it also works with non constant PEPS
    function constant_peps_tensor(::Type{S}, incoming, outgoing) where {S<:Number}
        inds = (incoming..., outgoing...)
        data = ones(S, map(dim, inds)...)
        return ITensor(data, inds...)
    end

    # set up simulation parameters
    Nsamples = 200
    maxiters = 10
    Nmeasure = 1000

    @testset "Test no-hopping limit with t=0.0 and U=2.0" begin
        t = 0.0
        U = 2.0
        Hubbard_ham = QuantumNaturalfPEPS.hamiltonian_hubbard(t, U, Lx, Ly)

        # set up Hilbert space and PEPS parameters
        bond_dim = 1 # ground state is a CDW and should be completely described by the Gaussian state, so a trivial PEPS is sufficient
        hilbert = ITensors.siteinds("Fermion", Lx, Ly)
        peps = PEPS(hilbert; bond_dim=bond_dim, tensor_init=constant_peps_tensor)
        # peps = PEPS(hilbert; bond_dim=bond_dim)
        # QuantumNaturalfPEPS.multiply_algebraic_spectrum!(peps, 3.) # Multiply the spectrum of the PEPS by a power-law factor as described in arXiv/2503.12557

        # set up mean-field parameters
        n_max_MF_params = QuantumNaturalfPEPS.get_max_num_MF_params_NN(Lx, Ly)
        η = zeros(Float64, n_max_MF_params)

        nx = QuantumNaturalfPEPS.get_max_num_hopping_x_NN(Lx, Ly)
        ny = QuantumNaturalfPEPS.get_max_num_hopping_y_NN(Lx, Ly)

        # hopping
        hx_range = N+1 : N+nx
        hy_range = N+nx+1 : N+nx+ny

        # pairing
        px_range = N+nx+ny+1 : N+nx+ny+nx
        py_range = N+nx+ny+nx+1 : N+nx+ny+nx+ny

        t_mf = 0.05 # small mean-field hopping
        Δ = 0.0 # no Cooper pairing for a pure CDW reference
        m_cdw = 1.0 # staggered onsite potential strength

        # staggered onsite potential
        for y in 1:Ly, x in 1:Lx
            idx = QuantumNaturalfPEPS.col_major_site(x, y, Lx)
            η[idx] = -m_cdw * (-1)^(x + y)
        end

        η[hx_range] .= -t_mf
        η[hy_range] .= -t_mf
        η[px_range] .= Δ
        η[py_range] .= Δ
        # η = 1e-8 .+ (9e-8 - 1e-8) * rand(n_max_MF_params)

        # create trial state as Gaussian state
        trial_state = QuantumNaturalfPEPS.GaussianState(QuantumNaturalfPEPS.build_general_H_BdG_2D_NN, N; η=η, parity_sector=parity_sector, target_state=target_state)

        θ_PEPS = vec(QuantumNaturalGradient.Parameters(peps).obj)
        θ = Vector{eltype(θ_PEPS)}(vcat(θ_PEPS, η))

        # Setup the Integrator and Solver
        integrator = QuantumNaturalGradient.Euler(lr=0.05)
        solver = QuantumNaturalGradient.EigenSolver()
        Oks_and_Eks = QuantumNaturalfPEPS.generate_Oks_and_Eks(peps, Hubbard_ham; trial_state=trial_state)

        @time loss_value, trained_θ, misc = QuantumNaturalGradient.evolve(Oks_and_Eks, θ; 
        integrator, 
        verbosity=0,
        sample_nr=Nsamples,
        maxiter=maxiters
        )

        # CDW (t=0): n_i n_j=0 (no doubly-occupied NN bonds), each of the N_b=24 NN bonds (4×4 OBC) has one occupied site.
        # E = -U/2 * N_b = -2/2 * 24 = -24.
        E_exact = -24.0
        @test isapprox(loss_value, E_exact; atol=1e-10)

        energy , energy_err, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, Hubbard_ham; trial_state=trial_state, it=Nmeasure)...)
        Ntot_mean , Ntot_err, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, QuantumNaturalfPEPS.build_Ntot_op(Lx, Ly); trial_state=trial_state, it=Nmeasure)...)
        M2_mean , M2_error, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, QuantumNaturalfPEPS.build_M_cdw2_op(Lx, Ly); trial_state=trial_state, it=Nmeasure)...)
        nn_avg_mean , nn_avg_error, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, QuantumNaturalfPEPS.build_nn_dd_corr_op(Lx, Ly); trial_state=trial_state, it=Nmeasure)...)

        # check if the error is within the expected sampling error
        atol = 1 / sqrt(Nmeasure)
        @test Ntot_err <= atol
        @test energy_err <= atol
        @test M2_error <= 3*atol
        @test nn_avg_error <= atol

        # check for the accuracy of sampled results
        @test isapprox(Ntot_mean, 8.0; atol=atol)
        @test isapprox(energy, E_exact; atol=atol)
        @test isapprox(M2_mean/N, 4.0; atol=atol)
        @test isapprox(nn_avg_mean, 0.0; atol=atol)
    end

    @testset "Test no onsite-potential limit with t=1.0 and U=0.0" begin
        t = 1.0
        U = 0.0
        Hubbard_ham = QuantumNaturalfPEPS.hamiltonian_hubbard(t, U, Lx, Ly)
        
        # set up Hilbert space and PEPS parameters
        bond_dim = 1 # free fermions should be completely described by the Gaussian state, so a trivial PEPS is sufficient
        hilbert = ITensors.siteinds("Fermion", Lx, Ly)
        peps = PEPS(hilbert; bond_dim=bond_dim, tensor_init=constant_peps_tensor)

        # peps = PEPS(hilbert; bond_dim=bond_dim)
        # QuantumNaturalfPEPS.multiply_algebraic_spectrum!(peps, 3.) # Multiply the spectrum of the PEPS by a power-law factor as described in arXiv/2503.12557

        # set up mean-field parameters
        n_max_MF_params = QuantumNaturalfPEPS.get_max_num_MF_params_NN(Lx, Ly)
        η = zeros(Float64, n_max_MF_params)

        nx = QuantumNaturalfPEPS.get_max_num_hopping_x_NN(Lx, Ly)
        ny = QuantumNaturalfPEPS.get_max_num_hopping_y_NN(Lx, Ly)

        # hopping
        hx_range = N+1 : N+nx
        hy_range = N+nx+1 : N+nx+ny

        # pairing
        px_range = N+nx+ny+1 : N+nx+ny+nx
        py_range = N+nx+ny+nx+1 : N+nx+ny+nx+ny

        t_mf = -1.0 # small mean-field hopping
        η[hx_range] .= t_mf
        η[hy_range] .= t_mf

        # create trial state as Gaussian state
        trial_state = QuantumNaturalfPEPS.GaussianState(QuantumNaturalfPEPS.build_general_H_BdG_2D_NN, N; η=η, parity_sector=parity_sector, target_state=target_state)

        θ_PEPS = vec(QuantumNaturalGradient.Parameters(peps).obj)
        θ = Vector{eltype(θ_PEPS)}(vcat(θ_PEPS, η))

        # Setup the Integrator and Solver
        integrator = QuantumNaturalGradient.Euler(lr=0.05)
        solver = QuantumNaturalGradient.EigenSolver()
        Oks_and_Eks = QuantumNaturalfPEPS.generate_Oks_and_Eks(peps, Hubbard_ham; trial_state=trial_state)

        @time loss_value, trained_θ, misc = QuantumNaturalGradient.evolve(Oks_and_Eks, θ; 
        integrator, 
        verbosity=0,
        sample_nr=Nsamples,
        maxiter=maxiters
        )

        # Free fermion (U=0) 4×4 OBC: ε_{m,n} = -2t[cos(mπ/5)+cos(nπ/5)]; 6 negative levels -(1+√5), -√5(×2), -(√5-1), -1(×2).
        # At half-filling (N=8), filling 6 negative + 2 zero-energy levels: E = -(1+√5) - 2√5 - (√5-1) - 2·1 = -2 - 4√5.
        E_exact = -2 - 4*sqrt(5)
        @test isapprox(loss_value, E_exact; atol=1e-10) 

        energy , energy_err, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, Hubbard_ham; trial_state=trial_state, it=Nmeasure)...)
        Ntot_mean , Ntot_err, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, QuantumNaturalfPEPS.build_Ntot_op(Lx, Ly); trial_state=trial_state, it=Nmeasure)...)
        M2_mean , M2_error, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, QuantumNaturalfPEPS.build_M_cdw2_op(Lx, Ly); trial_state=trial_state, it=Nmeasure)...)
        nn_avg_mean , nn_avg_error, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, QuantumNaturalfPEPS.build_nn_dd_corr_op(Lx, Ly); trial_state=trial_state, it=Nmeasure)...)

        # check if the error is within the expected sampling error
        atol = 1 / sqrt(Nmeasure)
        @test Ntot_err <= atol
        @test energy_err <= atol
        @test M2_error / N <= atol
        @test nn_avg_error <= atol

        # check for the accuracy of sampled results
        @test isapprox(Ntot_mean, 8.0; atol=atol)
        @test isapprox(energy, -2-4*sqrt(5); atol=atol)
        @test 0.375 - M2_error <= M2_mean / N <= 0.625 + M2_error
        @test 0.178 - nn_avg_error < nn_avg_mean < 0.218 + nn_avg_error
    end

    # =====================================================================================
    # Fixed trial state tests: first optimize the trial state alone until convergence
    # (trial-state-only Oks_and_Eks, no PEPS), then freeze it via fix_trial_state=true and
    # optimize only a D=1 PEPS on the full Hubbard Hamiltonian.
    # =====================================================================================

    function optimize_trial_state(H_BdG_exact, η0; maxiter=500, lr=0.05, sample_nr=Nsamples)
        Oks_and_Eks_slater = QuantumNaturalfPEPS.generate_Oks_and_Eks(H_BdG_exact, QuantumNaturalfPEPS.build_general_H_BdG_2D_NN, N;
                                                                      parity_sector=parity_sector, target_state=target_state)
        integrator = QuantumNaturalGradient.Euler(lr=lr)
        solver = QuantumNaturalGradient.EigenSolver()
        loss_value, trained_η, misc = QuantumNaturalGradient.evolve(Oks_and_Eks_slater, η0;
                integrator,
                verbosity=0,
                solver,
                sample_nr,
                maxiter,
        )
        return loss_value, trained_η
    end

    # exact ground state energy of the quadratic surrogate Hamiltonian
    function exact_quadratic_energy(H_BdG_exact)
        eigenvalues = eigvals(H_BdG_exact)
        return real(sum(eigenvalues[eigenvalues .< 0]) / 2 + sum(diag(H_BdG_exact[1:N, 1:N])) / 2)
    end

    function optimize_fixed_trial_state_peps(Hubbard_ham, trained_η; maxiter=maxiters)
        trial_state = QuantumNaturalfPEPS.GaussianState(QuantumNaturalfPEPS.build_general_H_BdG_2D_NN, N;
                                                        η=trained_η, parity_sector=parity_sector, target_state=target_state)
        hilbert = ITensors.siteinds("Fermion", Lx, Ly)
        peps = PEPS(hilbert; bond_dim=1, tensor_init=constant_peps_tensor)

        θ = vec(QuantumNaturalGradient.Parameters(peps).obj) # only PEPS parameters, the trial state is fixed

        integrator = QuantumNaturalGradient.Euler(lr=0.05)
        Oks_and_Eks = QuantumNaturalfPEPS.generate_Oks_and_Eks(peps, Hubbard_ham; trial_state=trial_state, fix_trial_state=true)

        loss_value, trained_θ, misc = QuantumNaturalGradient.evolve(Oks_and_Eks, θ;
                integrator,
                verbosity=0,
                sample_nr=Nsamples,
                maxiter,
        )
        return loss_value, trained_θ, length(θ)
    end

    # @testset "Fixed trial state: no-hopping limit with t=0.0 and U=2.0" begin
    #     Hubbard_ham = QuantumNaturalfPEPS.hamiltonian_hubbard(0.0, 2.0, Lx, Ly)

    #     n_max_MF_params = QuantumNaturalfPEPS.get_max_num_MF_params_NN(Lx, Ly)
    #     nx = QuantumNaturalfPEPS.get_max_num_hopping_x_NN(Lx, Ly)
    #     ny = QuantumNaturalfPEPS.get_max_num_hopping_y_NN(Lx, Ly)

    #     # quadratic surrogate with the same (CDW) ground state: staggered onsite potential only
    #     η_cdw = zeros(Float64, n_max_MF_params)
    #     for y in 1:Ly, x in 1:Lx
    #         idx = QuantumNaturalfPEPS.col_major_site(x, y, Lx)
    #         η_cdw[idx] = -1.0 * (-1)^(x + y)
    #     end
    #     H_BdG_exact = QuantumNaturalfPEPS.build_general_H_BdG_2D_NN(η_cdw, N)

    #     # start away from the ground state with a small hopping admixture
    #     η0 = copy(η_cdw)
    #     η0[N+1 : N+nx+ny] .= -0.05

    #     loss_slater, trained_η = optimize_trial_state(H_BdG_exact, η0)
    #     @test isapprox(loss_slater, exact_quadratic_energy(H_BdG_exact); atol=1e-8)

    #     # freeze the converged trial state and optimize only the D=1 PEPS on the Hubbard Hamiltonian
    #     loss_value, trained_θ, n_θ = optimize_fixed_trial_state_peps(Hubbard_ham, trained_η)
    #     @test length(trained_θ) == n_θ # variational vector contains only the PEPS parameters
    #     @test isapprox(loss_value, -24.0; atol=1e-8)
    # end

    # @testset "Fixed trial state: no onsite-potential limit with t=1.0 and U=0.0" begin
    #     Hubbard_ham = QuantumNaturalfPEPS.hamiltonian_hubbard(1.0, 0.0, Lx, Ly)

    #     n_max_MF_params = QuantumNaturalfPEPS.get_max_num_MF_params_NN(Lx, Ly)
    #     nx = QuantumNaturalfPEPS.get_max_num_hopping_x_NN(Lx, Ly)
    #     ny = QuantumNaturalfPEPS.get_max_num_hopping_y_NN(Lx, Ly)

    #     # the free hopping Hamiltonian is itself quadratic and exactly representable
    #     η_ff = zeros(Float64, n_max_MF_params)
    #     η_ff[N+1 : N+nx+ny] .= -1.0
    #     H_BdG_exact = QuantumNaturalfPEPS.build_general_H_BdG_2D_NN(η_ff, N)

    #     # ponytail: start at the mean-field solution (exact for U=0). A perturbed start converges only
    #     # very slowly here because of the degenerate zero modes at half filling; genuine Slater
    #     # optimization is already covered by test_mean_field_optimization.jl and
    #     # test_fixed_parity_optimization.jl — this test validates the fixed-trial-state pipeline.
    #     loss_slater, trained_η = optimize_trial_state(H_BdG_exact, copy(η_ff); maxiter=20)
    #     @test isapprox(loss_slater, exact_quadratic_energy(H_BdG_exact); atol=1e-8)

    #     # freeze the converged trial state and optimize only the D=1 PEPS on the Hubbard Hamiltonian
    #     loss_value, trained_θ, n_θ = optimize_fixed_trial_state_peps(Hubbard_ham, trained_η)
    #     @test length(trained_θ) == n_θ # variational vector contains only the PEPS parameters
    #     @test isapprox(loss_value, -2 - 4*sqrt(5); atol=1e-8)
    # end
end;
