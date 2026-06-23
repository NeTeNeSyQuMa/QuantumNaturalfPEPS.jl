using Test
using ITensors
using QuantumNaturalfPEPS
using QuantumNaturalGradient
using Random

Random.seed!(1234)

@testset "4x4 Hubbard model half-filling" begin
    # set up model
    Lx, Ly = 4, 4
    N = Lx * Ly
    parity_sector = 0
    target_state = 0

    # set up simulation parameters
    Nsamples = 200
    maxiters = 10
    Nmeasure = 1000

    @testset "Test no-hopping limit with t=0.0 and U=2.0" begin
        t = 0.0
        U = 2.0
        Hubbard_ham = QuantumNaturalfPEPS.hamiltonian_hubbard(t, U, Lx, Ly)

        # set up Hilbert space and PEPS parameters
        bond_dim = 2
        hilbert = ITensors.siteinds("Fermion", Lx, Ly)
        peps = PEPS(hilbert; bond_dim=bond_dim)
        QuantumNaturalfPEPS.multiply_algebraic_spectrum!(peps, 3.) # Multiply the spectrum of the PEPS by a power-law factor as described in arXiv/2503.12557

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

        E_exact = -24.0 # exact energy for t=0.0, U=2.0 at half-filling
        @test isapprox(loss_value, E_exact; atol=1e-10)

        energy , energy_err, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, Hubbard_ham; trial_state=trial_state, it=Nmeasure)...)
        Ntot_mean , Ntot_err, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, QuantumNaturalfPEPS.build_Ntot_op(Lx); trial_state=trial_state, it=Nmeasure)...)
        M2_mean , M2_error, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, QuantumNaturalfPEPS.build_M_cdw2_op(Lx); trial_state=trial_state, it=Nmeasure)...)
        nn_avg_mean , nn_avg_error, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, QuantumNaturalfPEPS.build_nn_dd_corr_op(Lx); trial_state=trial_state, it=Nmeasure)...)

        # check if the error is within the expected sampling error
        atol = 1 / sqrt(Nmeasure)
        @test Ntot_err <= atol
        @test energy_err <= atol
        @test M2_error <= 3*atol
        @test nn_avg_error <= atol

        # check for the accuracy of sampled results
        @test isapprox(Ntot_mean, 8.0; atol=atol)
        @test isapprox(energy, -24.0; atol=atol)
        @test isapprox(M2_mean/N, 4.0; atol=atol)
        @test isapprox(nn_avg_mean, 0.0; atol=atol)
    end
end
