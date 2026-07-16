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

    function create_CDW_trial_state(Lx, Ly; m_cdw=1.0, t_mf=0.05, Δ=0.0)
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

        # staggered onsite potential
        for y in 1:Ly, x in 1:Lx
            idx = QuantumNaturalfPEPS.col_major_site(x, y, Lx)
            η[idx] = -m_cdw * (-1)^(x + y)
        end

        η[hx_range] .= -t_mf
        η[hy_range] .= -t_mf
        η[px_range] .= Δ
        η[py_range] .= Δ

        # create trial state as Gaussian state
        return η, QuantumNaturalfPEPS.GaussianState(QuantumNaturalfPEPS.build_general_H_BdG_2D_NN, N; η=η, parity_sector=parity_sector, target_state=target_state)
    end

    function create_free_fermion_trial_state(Lx, Ly; t_mf=-1.0)
        n_max_MF_params = QuantumNaturalfPEPS.get_max_num_MF_params_NN(Lx, Ly)
        η = zeros(Float64, n_max_MF_params)

        nx = QuantumNaturalfPEPS.get_max_num_hopping_x_NN(Lx, Ly)
        ny = QuantumNaturalfPEPS.get_max_num_hopping_y_NN(Lx, Ly)

        # hopping
        hx_range = N+1 : N+nx
        hy_range = N+nx+1 : N+nx+ny

        η[hx_range] .= t_mf
        η[hy_range] .= t_mf

        # create trial state as Gaussian state
        return η, QuantumNaturalfPEPS.GaussianState(QuantumNaturalfPEPS.build_general_H_BdG_2D_NN, N; η=η, parity_sector=parity_sector, target_state=target_state)
    end

    # helper function to optimize the trial state only, without any PEPS parameters
    function optimize_trial_state_full_ham(Hubbard_ham, η0; maxiter=20, lr=0.05, verbosity=0)
        trial_state = QuantumNaturalfPEPS.GaussianState(QuantumNaturalfPEPS.build_general_H_BdG_2D_NN, N;
                                                        η=η0, parity_sector=parity_sector, target_state=target_state)
        hilbert = ITensors.siteinds("Fermion", Lx, Ly)
        peps = PEPS(hilbert; bond_dim=1, tensor_init=constant_peps_tensor)

        integrator = QuantumNaturalGradient.Euler(lr=lr)
        Oks_and_Eks = QuantumNaturalfPEPS.generate_Oks_and_Eks(peps, Hubbard_ham; trial_state=trial_state, fix_peps=true)

        # H_BdG is linear in η, so the state only depends on the ray η/‖η‖ (exact gauge freedom).
        # The solver keeps θdot ⊥ η, hence ‖η‖ grows strictly monotonically every Euler step
        # (‖θ‖ → √(‖θ‖² + lr²‖θdot‖²)); Fix the gauge by renormalizing after each step.
        η_norm = norm(η0)
        gauge_fix! = (o, ng) -> (o.θ .*= η_norm / norm(o.θ); ng)

        # Far from convergence var(E) is O(1), so metric eigendirections below
        # λ/λmax ≈ var(E)/(Nₛ·λmax) carry pure sampling noise; inverting them (default revcut 1e-6)
        # amplifies that noise into huge θdot. Cut at the noise floor instead.
        solver = QuantumNaturalGradient.EigenSolver(1e-3)

        loss_value, trained_η, misc = QuantumNaturalGradient.evolve(Oks_and_Eks, copy(η0);
                integrator,
                solver,
                verbosity=verbosity,
                sample_nr=Nsamples,
                maxiter,
                transform! = gauge_fix!,
        )
        return loss_value, trained_η
    end

    # helper function to optimize the PEPS only, with a fixed trial state
    function optimize_fixed_trial_state_peps(peps, Hubbard_ham, trained_η; maxiter=maxiters, lr=0.05, verbosity=0)
        trial_state = QuantumNaturalfPEPS.GaussianState(QuantumNaturalfPEPS.build_general_H_BdG_2D_NN, N;
                                                        η=trained_η, parity_sector=parity_sector, target_state=target_state)
        θ = vec(QuantumNaturalGradient.Parameters(peps).obj) # only PEPS parameters, the trial state is fixed

        integrator = QuantumNaturalGradient.Euler(lr=lr)
        Oks_and_Eks = QuantumNaturalfPEPS.generate_Oks_and_Eks(peps, Hubbard_ham; trial_state=trial_state, fix_trial_state=true, threaded=true)

        loss_value, trained_θ, misc = QuantumNaturalGradient.evolve(Oks_and_Eks, θ;
                integrator,
                verbosity=verbosity,
                sample_nr=Nsamples,
                maxiter,
        )
        return peps, loss_value, trained_θ
    end

    Nsamples = 200
    maxiters = 10
    Nmeasure = 1000

    @testset "Test no-hopping limit with t=0.0 and U=2.0" begin
        t = 0.0
        U = 2.0
        Hubbard_ham = QuantumNaturalfPEPS.hamiltonian_hubbard(t, U, Lx, Ly)

        # CDW (t=0): n_i n_j=0 (no doubly-occupied NN bonds), each of the N_b=24 NN bonds (4×4 OBC) has one occupied site.
        # E = -U/2 * N_b = -2/2 * 24 = -24.
        E_exact = -24.0

        # set up Hilbert space and PEPS parameters
        bond_dim = 1 # ground state is a CDW and should be completely described by the Gaussian state, so a trivial PEPS is sufficient
        hilbert = ITensors.siteinds("Fermion", Lx, Ly)

        #= Create different initial PEPS =#
        peps_const = PEPS(hilbert; bond_dim=bond_dim, tensor_init=constant_peps_tensor)
        peps_rnd = PEPS(hilbert; bond_dim=bond_dim) # random initial PEPS for frozen trial state optimization
        QuantumNaturalfPEPS.multiply_algebraic_spectrum!(peps_rnd, 3.0) # Multiply the spectrum of the PEPS by a power-law factor as described in arXiv/2503.12557

        # CDW trial state
        η, trial_state = create_CDW_trial_state(Lx, Ly)

        # Setup the Integrator and Solver
        integrator = QuantumNaturalGradient.Euler(lr=0.05)
        solver = QuantumNaturalGradient.EigenSolver()

        # optimize trial state
        loss_value, trained_η = optimize_trial_state_full_ham(Hubbard_ham, rand(length(η)); maxiter=50, lr=0.05, verbosity=0)
        @test isapprox(loss_value, E_exact; atol=1e-8)
        optimal_trial_state = QuantumNaturalfPEPS.GaussianState(QuantumNaturalfPEPS.build_general_H_BdG_2D_NN, N; η=trained_η, parity_sector=parity_sector, target_state=target_state)

        @testset "Joint optimization" begin
            @testset "Optimize trial state first (fixed) then PEPS" begin

                # @testset "Guessed trial state" begin # CDW ansatz
                #     peps = deepcopy(peps_rnd)
                #     peps, loss_value, trained_η = optimize_fixed_trial_state_peps(peps, Hubbard_ham, trial_state.η; maxiter=50, lr=0.05, verbosity=0)
                #     @test isapprox(loss_value, E_exact; atol=1e-8)

                #     energy , energy_err, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, Hubbard_ham; trial_state=trial_state, it=Nmeasure)...)
                #     Ntot_mean , Ntot_err, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, QuantumNaturalfPEPS.build_Ntot_op(Lx, Ly); trial_state=trial_state, it=Nmeasure)...)
                #     M2_mean , M2_error, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, QuantumNaturalfPEPS.build_M_cdw2_op(Lx, Ly); trial_state=trial_state, it=Nmeasure)...)
                #     nn_avg_mean , nn_avg_error, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, QuantumNaturalfPEPS.build_nn_dd_corr_op(Lx, Ly); trial_state=trial_state, it=Nmeasure)...)

                #     # check if the error is within the expected sampling error
                #     atol = 3 / sqrt(Nmeasure)
                #     @test Ntot_err <= atol
                #     @test energy_err <= atol
                #     @test M2_error <= atol
                #     @test nn_avg_error <= atol

                #     # check for the accuracy of sampled results
                #     @test isapprox(Ntot_mean, 8.0; atol=atol)
                #     @test isapprox(energy, E_exact; atol=atol)
                #     @test isapprox(M2_mean/N, 4.0; atol=atol)
                #     @test isapprox(nn_avg_mean, 0.0; atol=atol)
                # end

                @testset "Optimized trial state" begin
                    Random.seed!(1234) # set seed here explicitly so the run converges with the poor number of samples and iterations

                    peps = deepcopy(peps_rnd)
                    peps, loss_value, trained_η = optimize_fixed_trial_state_peps(peps, Hubbard_ham, optimal_trial_state.η; maxiter=50, lr=0.05, verbosity=2)
                    @test isapprox(loss_value, E_exact; atol=1e-8)

                    energy , energy_err, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, Hubbard_ham; trial_state=optimal_trial_state, it=Nmeasure)...)
                    Ntot_mean , Ntot_err, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, QuantumNaturalfPEPS.build_Ntot_op(Lx, Ly); trial_state=optimal_trial_state, it=Nmeasure)...)
                    M2_mean , M2_error, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, QuantumNaturalfPEPS.build_M_cdw2_op(Lx, Ly); trial_state=optimal_trial_state, it=Nmeasure)...)
                    nn_avg_mean , nn_avg_error, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, QuantumNaturalfPEPS.build_nn_dd_corr_op(Lx, Ly); trial_state=optimal_trial_state, it=Nmeasure)...)

                    # check if the error is within the expected sampling error
                    atol = 3 / sqrt(Nmeasure)
                    @test Ntot_err <= atol
                    @test energy_err <= atol
                    @test M2_error <= atol
                    @test nn_avg_error <= atol

                    # check for the accuracy of sampled results
                    @test isapprox(Ntot_mean, 8.0; atol=atol)
                    @test isapprox(energy, E_exact; atol=atol)
                    @test isapprox(M2_mean/N, 4.0; atol=atol)
                    @test isapprox(nn_avg_mean, 0.0; atol=atol)
                end
            end

            # @testset "Optimize trial state and PEPS simultaneously" begin
            #     peps = deepcopy(peps_const)
            #     trial_state_CDW = deepcopy(trial_state) # use the guessed trial state as initial guess for the optimization
            #     θ = vcat(vec(QuantumNaturalGradient.Parameters(peps).obj), trial_state_CDW.η) # only PEPS parameters, the trial state is fixed
                
            #     Oks_and_Eks = QuantumNaturalfPEPS.generate_Oks_and_Eks(peps, Hubbard_ham; trial_state=trial_state_CDW, threaded=true)

            #     loss_value, trained_θ, misc = QuantumNaturalGradient.evolve(Oks_and_Eks, θ;
            #             integrator,
            #             verbosity=0,
            #             sample_nr=Nsamples,
            #             maxiter=maxiters,
            #     )
            #     @test isapprox(loss_value, E_exact; atol=1e-8)

            #     energy , energy_err, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, Hubbard_ham; trial_state=trial_state_CDW, it=Nmeasure)...)
            #     Ntot_mean , Ntot_err, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, QuantumNaturalfPEPS.build_Ntot_op(Lx, Ly); trial_state=trial_state_CDW, it=Nmeasure)...)
            #     M2_mean , M2_error, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, QuantumNaturalfPEPS.build_M_cdw2_op(Lx, Ly); trial_state=trial_state_CDW, it=Nmeasure)...)
            #     nn_avg_mean , nn_avg_error, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, QuantumNaturalfPEPS.build_nn_dd_corr_op(Lx, Ly); trial_state=trial_state_CDW, it=Nmeasure)...)

            #     # check if the error is within the expected sampling error
            #     atol = 3 / sqrt(Nmeasure)
            #     @test Ntot_err <= atol
            #     @test energy_err <= atol
            #     @test M2_error <= atol
            #     @test nn_avg_error <= atol

            #     # check for the accuracy of sampled results
            #     @test isapprox(Ntot_mean, 8.0; atol=atol)
            #     @test isapprox(energy, E_exact; atol=atol)
            #     @test isapprox(M2_mean/N, 4.0; atol=atol)
            #     @test isapprox(nn_avg_mean, 0.0; atol=atol)
            # end
        end
    end

    # This is a short sanity check and designed in such a way, that the time needed to run this test file is not too long
    # @testset "Test no onsite-potential limit with t=1.0 and U=0.0" begin
    #     t = 1.0
    #     U = 0.0
    #     Hubbard_ham = QuantumNaturalfPEPS.hamiltonian_hubbard(t, U, Lx, Ly)
        
    #     # Free fermion (U=0) 4×4 OBC: ε_{m,n} = -2t[cos(mπ/5)+cos(nπ/5)]; 6 negative levels -(1+√5), -√5(×2), -(√5-1), -1(×2).
    #     # At half-filling (N=8), filling 6 negative + 2 zero-energy levels: E = -(1+√5) - 2√5 - (√5-1) - 2·1 = -2 - 4√5.
    #     E_exact = -2 - 4*sqrt(5)

    #     # set up Hilbert space and PEPS parameters
    #     bond_dim = 1 # free fermions should be completely described by the Gaussian state, so a trivial PEPS is sufficient
    #     hilbert = ITensors.siteinds("Fermion", Lx, Ly)
        
    #     #= Create different initial PEPS =#
    #     peps_const = PEPS(hilbert; bond_dim=bond_dim, tensor_init=constant_peps_tensor)

    #     # Free fermion trial state
    #     η, trial_state = create_free_fermion_trial_state(Lx, Ly)

    #     # Setup the Integrator and Solver
    #     integrator = QuantumNaturalGradient.Euler(lr=0.05)
    #     solver = QuantumNaturalGradient.EigenSolver()

    #     # optimize trial state (TODO: this is not converged with the sample size and number of iterations used here, but it is sufficient for a sanity check)
    #     loss_value, trained_η = optimize_trial_state_full_ham(Hubbard_ham, rand(length(η)); maxiter=50, lr=0.05, verbosity=0)
    #     @test isapprox(loss_value, E_exact; atol=1e-2)

    #     # do sanity check with optimal trial state from theory and id peps -> should converge directly
    #     Oks_and_Eks = QuantumNaturalfPEPS.generate_Oks_and_Eks(peps_const, Hubbard_ham; trial_state=trial_state, fix_trial_state=true, threaded=true)
    #     θ = vec(QuantumNaturalGradient.Parameters(peps_const).obj) # only PEPS parameters, the trial state is fixed

    #     @time loss_value, trained_θ, misc = QuantumNaturalGradient.evolve(Oks_and_Eks, θ; 
    #     integrator, 
    #     verbosity=0,
    #     sample_nr=Nsamples,
    #     maxiter=maxiters
    #     )

    #     @test loss_value ≈ E_exact

    #     energy , energy_err, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps_const, Hubbard_ham; trial_state=trial_state, it=Nmeasure)...)
    #     Ntot_mean , Ntot_err, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps_const, QuantumNaturalfPEPS.build_Ntot_op(Lx, Ly); trial_state=trial_state, it=Nmeasure)...)
    #     M2_mean , M2_error, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps_const, QuantumNaturalfPEPS.build_M_cdw2_op(Lx, Ly); trial_state=trial_state, it=Nmeasure)...)
    #     nn_avg_mean , nn_avg_error, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps_const, QuantumNaturalfPEPS.build_nn_dd_corr_op(Lx, Ly); trial_state=trial_state, it=Nmeasure)...)

    #     # check if the error is within the expected sampling error
    #     atol = 1 / sqrt(Nmeasure)
    #     @test Ntot_err <= atol
    #     @test energy_err <= atol
    #     @test M2_error / N <= atol
    #     @test nn_avg_error <= atol

    #     # check for the accuracy of sampled results
    #     @test isapprox(Ntot_mean, 8.0; atol=atol)
    #     @test isapprox(energy, -2-4*sqrt(5); atol=atol)
    #     @test 0.375 - M2_error <= M2_mean / N <= 0.625 + M2_error
    #     @test 0.178 - nn_avg_error < nn_avg_mean < 0.218 + nn_avg_error
    # end
end