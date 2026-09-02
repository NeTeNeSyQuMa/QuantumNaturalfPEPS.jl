using Test
using LinearAlgebra
using ITensors
using QuantumNaturalfPEPS

@testset "Triangular monopole Gaussian state" begin
    Lx, Ly = 4, 6
    N = Lx * Ly

    @testset "constructs the frozen zero-flux-sector state" begin
        state = monopole_state(Lx, Ly, 0; particle_number=N)
        projected = gutzwiller_project(state)

        @test state isa QuantumNaturalfPEPS.GaussianState
        @test state.N == 2N
        @test isempty(QuantumNaturalfPEPS.Parameters(state))
        @test QuantumNaturalfPEPS.getParity(state) == mod(N, 2)
        @test projected isa ParameterizedGutzwillerProjectedState
        @test projected.N == N
        @test isempty(QuantumNaturalfPEPS.Parameters(projected))
    end

    @testset "preserves the uniform-flux hopping Hamiltonian" begin
        Q = 8
        hopping, _ = uniform_flux_staggered_pi_hoppings(Lx, Ly, Q)
        Haux = Matrix(hamiltonian_aux_triangular_torus(
            Lx,
            Ly;
            hopping,
            fields=zeros(Float64, Lx, Ly, 3),
        ))
        state = monopole_state(Lx, Ly, Q; particle_number=N)
        H_BdG = Matrix(state.H_BdG_func(
            QuantumNaturalfPEPS.Parameters(state),
            state.N,
        ))
        particle_block = H_BdG[1:state.N, 1:state.N]
        chemical_potential_shift = particle_block[1, 1] - Haux[1, 1]

        @test particle_block ≈ Haux + chemical_potential_shift * I atol=1e-12
    end

    @testset "validates filling and the Fermi gap" begin
        @test_throws ArgumentError monopole_state(
            Lx,
            Ly,
            0;
            particle_number=-1,
        )
        @test_throws ArgumentError monopole_state(
            Lx,
            Ly,
            0;
            gap_tolerance=-1.0,
        )
        @test_throws ArgumentError monopole_state(
            Lx,
            Ly,
            1;
            particle_number=N,
        )
    end
end

@testset "Reduced ordered-state Gaussian ansatze" begin
    Lx = Ly = 6
    N = Lx * Ly
    eta_Y = [0.35, 0.25, 0.15, 0.8]
    eta_umbrella = [0.3]
    eta_stripe = [0.3]

    Y_hopping, Y_fields = y_hopping_fields(Lx, Ly, eta_Y)
    umbrella_hopping, umbrella_fields =
        umbrella_hopping_fields(Lx, Ly, eta_umbrella)
    stripe_hopping, stripe_fields =
        cs_hopping_fields(Lx, Ly, eta_stripe)

    @test size(Y_hopping) == (Lx, Ly, 3)
    @test size(Y_fields) == (Lx, Ly, 3)
    @test sort(unique(abs.(Y_hopping))) ≈ [0.8, 1.0]
    @test all(iszero, Y_fields[:, :, 2])
    @test all(isapprox.(sqrt.(
        umbrella_fields[:, :, 1].^2 .+ umbrella_fields[:, :, 2].^2,
    ), only(eta_umbrella); atol=1e-12))
    @test all(iszero, umbrella_fields[:, :, 3])
    @test all(abs.(umbrella_hopping) .≈ 1.0)
    @test all(abs.(stripe_hopping[:, :, 1:2]) .≈ 1.0)
    @test all(abs.(stripe_hopping[:, :, 3]) .≈ 0.8)
    @test all(iszero, stripe_fields[:, :, 2:3])
    @test all(abs.(stripe_fields[:, :, 1]) .≈ only(eta_stripe))

    states = (
        Y=y_state(Lx, Ly; η=eta_Y),
        umbrella=umbrella_state(Lx, Ly; η=eta_umbrella),
        stripe=cs_state(Lx, Ly; η=eta_stripe),
    )
    arrays = (
        Y=(Y_hopping, Y_fields),
        umbrella=(umbrella_hopping, umbrella_fields),
        stripe=(stripe_hopping, stripe_fields),
    )
    expected_parameter_counts = (Y=4, umbrella=1, stripe=1)
    for name in keys(states)
        state = getproperty(states, name)
        hopping, fields = getproperty(arrays, name)
        parameters = QuantumNaturalfPEPS.Parameters(state)
        @test length(parameters) == getproperty(expected_parameter_counts, name)
        @test state.N == 2N

        Haux = Matrix(hamiltonian_aux_triangular_torus(
            Lx,
            Ly;
            hopping,
            fields,
        ))
        H_BdG = Matrix(state.H_BdG_func(parameters, state.N))
        particle_block = H_BdG[1:state.N, 1:state.N]
        shift = particle_block[1, 1] - Haux[1, 1]
        @test particle_block ≈ Haux + shift * I atol=1e-12

        gradient_result = QuantumNaturalfPEPS.Zygote.gradient(parameters) do trial_parameters
            H = Matrix(state.H_BdG_func(trial_parameters, state.N))
            return real(sum(abs2, H))
        end
        gradient = gradient_result[1]
        @test length(gradient) == length(parameters)
        @test all(isfinite, gradient)

        projected = gutzwiller_project(state; Nup=N ÷ 2)
        @test projected isa ParameterizedGutzwillerProjectedState
        @test length(QuantumNaturalfPEPS.Parameters(projected)) == length(parameters)
    end

    @test_throws ArgumentError y_hopping_fields(4, 6, eta_Y)
    @test_throws ArgumentError cs_hopping_fields(5, 6, eta_stripe)
    @test_throws DimensionMismatch y_hopping_fields(Lx, Ly, eta_Y[1:3])
    @test_throws DimensionMismatch umbrella_hopping_fields(Lx, Ly, Float64[])
    @test_throws DimensionMismatch cs_hopping_fields(Lx, Ly, [0.2, 0.3])
end

@testset "Trainable triangular auxiliary Gaussian state" begin
    Lx = Ly = 3
    N = Lx * Ly
    hopping = fill(1.0 + 0.0im, Lx, Ly, 3)
    hopping[1, 1, 1] = 2cis(0.3)
    hopping[2, 1, 2] = 0.7cis(-0.2)
    fields = zeros(Float64, Lx, Ly, 3)
    fields[:, :, 3] .= 12.0

    state = triangular_aux_gaussian_state(
        Lx,
        Ly;
        hopping,
        fields,
        particle_number=N,
        cache_gradients=true,
    )
    η = QuantumNaturalfPEPS.Parameters(state)

    @test state.N == 2N
    @test length(η) == 6N
    @test η[1:3] == [2.0, 1.0, 1.0]
    @test η[3N+1:3N+3] == [0.0, 0.0, 12.0]
    @test length(state.slater_loggrad_cache.dΓs) == 6N

    H0 = Matrix(state.H_BdG_func(η, state.N))
    number_of_modes = state.N
    particle_block0 = H0[1:number_of_modes, 1:number_of_modes]

    # The first parameter changes only direction 1 out of site (1, 1), for
    # both spin species, while retaining its initial Peierls phase.
    η_hopping = copy(η)
    η_hopping[1] += 0.25
    H_hopping = Matrix(state.H_BdG_func(η_hopping, state.N))
    hopping_difference =
        H_hopping[1:number_of_modes, 1:number_of_modes] - particle_block0
    first_bond = only(filter(
        bond -> bond.source == (1, 1) && bond.direction == 1,
        triangular_torus_bonds(Lx, Ly),
    ))
    expected_change = 0.25cis(0.3)
    for spin in 1:2
        i = 2 * ((first_bond.source[2] - 1) * Lx + first_bond.source[1] - 1) + spin
        j = 2 * ((first_bond.target[2] - 1) * Lx + first_bond.target[1] - 1) + spin
        @test hopping_difference[i, j] ≈ expected_change atol=1e-12
        @test hopping_difference[j, i] ≈ conj(expected_change) atol=1e-12
    end
    @test count(!iszero, hopping_difference) == 4

    # The first site's Mx parameter changes only the local spin-mixing entry.
    η_field = copy(η)
    η_field[3N+1] += 0.4
    H_field = Matrix(state.H_BdG_func(η_field, state.N))
    field_difference = H_field[1:number_of_modes, 1:number_of_modes] - particle_block0
    @test field_difference[1, 2] ≈ 0.2 atol=1e-12
    @test field_difference[2, 1] ≈ 0.2 atol=1e-12
    @test count(!iszero, field_difference) == 2

    # The cached covariance derivative agrees with a centered finite
    # difference when the first bond magnitude is varied.
    epsilon = 1e-6
    η_plus = copy(η)
    η_minus = copy(η)
    η_plus[1] += epsilon
    η_minus[1] -= epsilon
    Γ_plus, _ = QuantumNaturalfPEPS.get_Γ_from_H_BdG(
        state.H_BdG_func(η_plus, state.N),
        state.parity_sector,
    )
    Γ_minus, _ = QuantumNaturalfPEPS.get_Γ_from_H_BdG(
        state.H_BdG_func(η_minus, state.N),
        state.parity_sector,
    )
    numerical_dΓ = (Γ_plus - Γ_minus) / (2epsilon)
    @test state.slater_loggrad_cache.dΓs[1] ≈ numerical_dΓ rtol=2e-5 atol=2e-7
end

@testset "Gutzwiller projection of a trainable Gaussian state" begin
    Lx = Ly = 3
    N = Lx * Ly
    hopping = fill(0.2 + 0.0im, Lx, Ly, 3)
    hopping[1, 1, 1] *= cis(0.25)
    fields = zeros(Float64, Lx, Ly, 3)
    fields[:, :, 1] .= 10.0

    gaussian_state = triangular_aux_gaussian_state(
        Lx,
        Ly;
        hopping,
        fields,
        particle_number=N,
    )
    projected_state = gutzwiller_project(gaussian_state; Nup=5)

    @test projected_state isa ParameterizedGutzwillerProjectedState
    @test projected_state isa AbstractGutzwillerProjectedState
    @test projected_state.N == N
    @test projected_state.Nup == 5
    @test QuantumNaturalfPEPS.Parameters(projected_state) ==
          QuantumNaturalfPEPS.Parameters(gaussian_state)
    @test QuantumNaturalfPEPS.Parameters(projected_state) !==
          QuantumNaturalfPEPS.Parameters(gaussian_state)
    @test length(QuantumNaturalfPEPS.Parameters(projected_state)) == 6N

    Haux = hamiltonian_aux_triangular_torus(Lx, Ly; hopping, fields)
    direct_spectrum = eigen(Haux)
    direct_projector =
        direct_spectrum.vectors[:, 1:N] * direct_spectrum.vectors[:, 1:N]'
    projected_projector =
        projected_state.occupied_orbitals * projected_state.occupied_orbitals'
    @test projected_projector ≈ direct_projector atol=1e-10

    configuration_a = [0, 0, 0, 0, 0, 1, 1, 1, 1]
    configuration_b = [0, 0, 0, 0, 1, 0, 1, 1, 1]
    @test !iszero(gutzwiller_amplitude(projected_state, configuration_a))
    @test !iszero(gutzwiller_amplitude(projected_state, configuration_b))

    gradient_a = gutzwiller_log_gradient(projected_state, configuration_a)
    gradient_b = gutzwiller_log_gradient(projected_state, configuration_b)
    exchange_cache = GutzwillerExchangeCache(projected_state, configuration_a)
    @test gutzwiller_log_gradient(projected_state, exchange_cache) ≈ gradient_a
    analytic_ratio_gradient = gradient_a - gradient_b
    η0 = copy(QuantumNaturalfPEPS.Parameters(projected_state))
    epsilon = 1e-6
    for parameter in (1, 3N + 3)
        ηplus = copy(η0)
        ηminus = copy(η0)
        ηplus[parameter] += epsilon
        ηminus[parameter] -= epsilon

        QuantumNaturalfPEPS.write!(projected_state, ηplus)
        ratio_plus = gutzwiller_amplitude(projected_state, configuration_a) /
                     gutzwiller_amplitude(projected_state, configuration_b)
        QuantumNaturalfPEPS.write!(projected_state, ηminus)
        ratio_minus = gutzwiller_amplitude(projected_state, configuration_a) /
                      gutzwiller_amplitude(projected_state, configuration_b)
        numerical_gradient = (log(ratio_plus) - log(ratio_minus)) / (2epsilon)
        @test analytic_ratio_gradient[parameter] ≈ numerical_gradient rtol=2e-4 atol=2e-6
    end
    QuantumNaturalfPEPS.write!(projected_state, η0)

    hilbert = ITensors.siteinds("S=1/2", Lx, Ly)
    peps = PEPS(ComplexF64, hilbert; bond_dim=1, show_warning=false)
    physical_hamiltonian = hamiltonian_J1J2_H(
        Lx,
        Ly;
        J1=1.0,
        J2=1 / 8,
        H=0.0,
        boundary=:periodic,
    )
    Oks_and_Eks = generate_Oks_and_Eks(
        peps,
        physical_hamiltonian;
        trial_state=projected_state,
        fixed_sz_metropolis=true,
        burnin_sweeps=0,
        moves_per_sample=1,
        seed=29,
    )
    θ = vcat(vec(peps), QuantumNaturalfPEPS.Parameters(projected_state))
    batch = Oks_and_Eks(θ, 2)
    @test size(batch[:Oks]) == (2, length(peps) + 6N)
    @test all(sample -> count(==(0), sample) == 5, batch[:samples])
    @test all(isfinite, batch[:Oks])
    @test all(isfinite, batch[:Eks])
end
