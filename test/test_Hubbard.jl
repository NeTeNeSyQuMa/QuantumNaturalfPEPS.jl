using Test
# import Pkg
# Pkg.activate("/home/psireal42/Work/phd-projects/qnfp_env"; shared=false)
using ITensors
using QuantumNaturalfPEPS
using QuantumNaturalGradient
using Random
include("../test/helpers/hubbard.jl")


Random.seed!(1234)


@testset "4x4 Hubbard model half-filling" begin
    # set up model
    parity_sector = 0
    target_state = 0

    # set up simulation parameters
    Nsamples = 200
    maxiters = 50
    Nmeasure = 1000

    Lx, Ly = 4, 4
    N = Lx * Ly

    function constant_peps_tensor(::Type{S}, incoming, outgoing) where {S<:Number}
        inds = (incoming..., outgoing...)
        data = ones(S, map(dim, inds)...)
        return ITensor(data, inds...)
    end

    @testset "Test no-hopping limit with t=0.0 and U=2.0" begin
        
        t = 0.0
        U = 2.0
        Hubbard_ham = QuantumNaturalfPEPS.hamiltonian_hubbard(t, U, Lx, Ly)

        # set up Hilbert space and PEPS parameters
        bond_dim = 2
        hilbert = ITensors.siteinds("Fermion", Lx, Ly)
        peps = PEPS(
            Float64,
            hilbert;
            bond_dim=2,
            tensor_init=constant_peps_tensor,
        )


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
        verbosity=2,
        sample_nr=Nsamples,
        maxiter=maxiters
        )

        E_exact = -24.0 # exact energy for t=0.0, U=2.0 at half-filling
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
        @test isapprox(energy, -24.0; atol=atol)
        @test isapprox(M2_mean/N, 4.0; atol=atol)
        @test isapprox(nn_avg_mean, 0.0; atol=atol)
    end

    @testset "Test no onsite-potential limit with t=1.0 and U=0.0" begin

        t = 1.0
        U = 0.0
        Hubbard_ham = QuantumNaturalfPEPS.hamiltonian_hubbard(t, U, Lx, Ly)

        # set up Hilbert space and PEPS parameters
        hilbert = ITensors.siteinds("Fermion", Lx, Ly)
        peps = PEPS(
            Float64,
            hilbert;
            bond_dim=2,
            tensor_init=constant_peps_tensor,
        )

        # set up mean-field parameters
        n_max_MF_params = QuantumNaturalfPEPS.get_max_num_MF_params_NN(Lx, Ly)
        η = zeros(Float64, n_max_MF_params)

        nx = QuantumNaturalfPEPS.get_max_num_hopping_x_NN(Lx, Ly)
        ny = QuantumNaturalfPEPS.get_max_num_hopping_y_NN(Lx, Ly)

        # chemical potential
        μ_range  = 1:N

        # hopping
        hx_range = N+1 : N+nx
        hy_range = N+nx+1 : N+nx+ny

        # pairing
        px_range = N+nx+ny+1 : N+nx+ny+nx
        py_range = N+nx+ny+nx+1 : N+nx+ny+nx+ny

        t_mf = 1.0 # small mean-field hopping

        # η[μ_range] .= -1e-15
        η[hx_range] .= -t_mf
        η[hy_range] .= -t_mf
        

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
        verbosity=2,
        sample_nr=Nsamples,
        maxiter=maxiters
        )

        E_exact = -2 - 4sqrt(5)
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
        @test isapprox(M2_mean/N, 11/24; atol=atol)
        @test isapprox(nn_avg_mean, (721/3600)-(sqrt(5)/240); atol=atol)
    end

    # @testset "Test interaction case with t=1.0 and U=1.0" begin

    #     # set up model
    #     parity_sector = 0
    #     target_state = 0

    #     # set up simulation parameters
    #     Nsamples = 200
    #     maxiters = 150
    #     Nmeasure = 1000

    #     Lx, Ly = 4, 4
    #     N = Lx * Ly

    #     function constant_peps_tensor(::Type{S}, incoming, outgoing) where {S<:Number}
    #         inds = (incoming..., outgoing...)
    #         data = ones(S, map(dim, inds)...)
    #         return ITensor(data, inds...)
    #     end

    #     t = 1.0
    #     U = 1.0
    #     Hubbard_ham = QuantumNaturalfPEPS.hamiltonian_hubbard(t, U, Lx, Ly)

    #     # set up Hilbert space and PEPS parameters
    #     hilbert = ITensors.siteinds("Fermion", Lx, Ly)
    #     peps = PEPS(
    #         Float64,
    #         hilbert;
    #         bond_dim=3,
    #         tensor_init=constant_peps_tensor,
    #     )

    #     # set up mean-field parameters
    #     n_max_MF_params = QuantumNaturalfPEPS.get_max_num_MF_params_NN(Lx, Ly)
    #     η = zeros(Float64, n_max_MF_params)

    #     nx = QuantumNaturalfPEPS.get_max_num_hopping_x_NN(Lx, Ly)
    #     ny = QuantumNaturalfPEPS.get_max_num_hopping_y_NN(Lx, Ly)

    #     # hopping
    #     hx_range = N+1 : N+nx
    #     hy_range = N+nx+1 : N+nx+ny

    #     # pairing
    #     px_range = N+nx+ny+1 : N+nx+ny+nx
    #     py_range = N+nx+ny+nx+1 : N+nx+ny+nx+ny

    #     # hopping
    #     hx_range = N+1 : N+nx
    #     hy_range = N+nx+1 : N+nx+ny

    #     t_mf = 1.0 # small mean-field hopping
    #     Δ = 0.0 # no Cooper pairing for a pure CDW reference
    #     m_cdw = 1e-3 # staggered onsite potential strength

    #     # staggered onsite potential
    #     for y in 1:Ly, x in 1:Lx
    #         idx = QuantumNaturalfPEPS.col_major_site(x, y, Lx)
    #         η[idx] = m_cdw * (-1)^(x + y)
    #     end

    #     η[hx_range] .= -t_mf
    #     η[hy_range] .= -t_mf
    #     η[px_range] .= Δ
    #     η[py_range] .= Δ

    #     # create trial state as Gaussian state
    #     trial_state = QuantumNaturalfPEPS.GaussianState(QuantumNaturalfPEPS.build_general_H_BdG_2D_NN, N; η=η, parity_sector=parity_sector, target_state=target_state)

    #     saved_mask = copy(peps.mask)
    #     peps.mask .= 0

    #     Oks_mf = QuantumNaturalfPEPS.generate_Oks_and_Eks(
    #         peps, Hubbard_ham; trial_state
    #     )

    #     η0 = copy(QuantumNaturalfPEPS.Parameters(trial_state))

    #     integrator_mf = QuantumNaturalGradient.Euler(
    #         lr=0.005
    #     )
    #     solver_mf = QuantumNaturalGradient.EigenSolver(1e-3)

    #     E_mf, η_opt, misc_mf = QuantumNaturalGradient.evolve(
    #         Oks_mf, η0;
    #         integrator=integrator_mf,
    #         solver=solver_mf,
    #         sample_nr=Nsamples,
    #         maxiter=25,
    #         verbosity=2,
    #     )

    #     write!(trial_state, η_opt)
    #     peps.mask .= saved_mask

    #     η_fixed = copy(QuantumNaturalfPEPS.Parameters(trial_state))
    #     θp0 = vec(QuantumNaturalGradient.Parameters(peps).obj)
    #     npeps = length(θp0)

    #     Oks_joint = QuantumNaturalfPEPS.generate_Oks_and_Eks(
    #         peps, Hubbard_ham; trial_state
    #     )

    #     function Oks_peps_only(θp, sample_nr; kwargs...)
    #         out = Oks_joint(vcat(θp, η_fixed), sample_nr; kwargs...)
    #         out[:Oks] = out[:Oks][:, 1:npeps]
    #         return out
    #     end

    #     integrator_peps = QuantumNaturalGradient.Euler(
    #         lr=0.005
    #     )
    #     solver_peps = QuantumNaturalGradient.EigenSolver(1e-3)

    #     E_peps, θp_opt, misc_peps = QuantumNaturalGradient.evolve(
    #         Oks_peps_only, θp0;
    #         integrator=integrator_peps,
    #         solver=solver_peps,
    #         sample_nr=500,
    #         maxiter=25,
    #         verbosity=2,
    #     )

    #     write!(peps, θp_opt)

    #     # θ_PEPS = vec(QuantumNaturalGradient.Parameters(peps).obj)
    #     θ = Vector{eltype(θp_opt)}(vcat(θp_opt, η_opt))

    #     # Setup the Integrator and Solver
    #     integrator = QuantumNaturalGradient.Euler(
    #         lr=0.002,
    #         use_clipping=true,
    #         clip_norm=5.0,
    #         clip_val=0.5,
    #     )
    #     solver = QuantumNaturalGradient.EigenSolver()
    #     Oks_and_Eks = QuantumNaturalfPEPS.generate_Oks_and_Eks(peps, Hubbard_ham; trial_state=trial_state)

    #     @time loss_value, trained_θ, misc = QuantumNaturalGradient.evolve(Oks_and_Eks, θ; 
    #     integrator, 
    #     verbosity=2,
    #     sample_nr=500,
    #     maxiter=maxiters
    #     )

    #     E_ED, ψ0, H, basis = ground_energy_spinless_tV_ED(
    #             Lx,
    #             Ly,
    #             t,
    #             U;
    #             shifted=true)
    #     exact_obs = diagonal_observables_ED(ψ0, basis, Lx, Ly)
    #     # @test isapprox(loss_value, E_ED; atol=1e-2)

    #     energy , energy_err, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, Hubbard_ham; trial_state=trial_state, it=Nmeasure)...)
    #     Ntot_mean , Ntot_err, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, QuantumNaturalfPEPS.build_Ntot_op(Lx, Ly); trial_state=trial_state, it=Nmeasure)...)
    #     M2_mean , M2_error, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, QuantumNaturalfPEPS.build_M_cdw2_op(Lx, Ly); trial_state=trial_state, it=Nmeasure)...)
    #     nn_avg_mean , nn_avg_error, _ = QuantumNaturalfPEPS.weighted_mean_error(QuantumNaturalfPEPS.get_ExpectationValue(peps, QuantumNaturalfPEPS.build_nn_dd_corr_op(Lx, Ly); trial_state=trial_state, it=Nmeasure)...)

    #     # check if the error is within the expected sampling error
    #     atol = 1 / sqrt(Nmeasure)
    #     @test Ntot_err <= atol
    #     @test energy_err <= atol
    #     @test M2_error / N <= atol
    #     @test nn_avg_error <= atol

    #     # check for the accuracy of sampled results
    #     @test isapprox(exact_obs.Ntot, 8.0; atol=atol)
    #     @test isapprox(energy, E_ED; atol=atol)
    #     @test isapprox(M2_mean/N, exact_obs.S_pi_pi; atol=atol)
    #     @test isapprox(nn_avg_mean, exact_obs.nn_avg; atol=atol)
    # end
end
