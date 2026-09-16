using Test
using LinearAlgebra
using ITensors
using QuantumNaturalfPEPS

@testset "Triangular monopole Gaussian state" begin
    Lx, Ly = 4, 6
    N = Lx * Ly

    @testset "preserves the uniform-flux hopping Hamiltonian" begin
        Q = 8
        hopping, _ = uniform_flux_staggered_pi_hoppings(Lx, Ly, Q)
        Haux = Matrix(hamiltonian_aux_triangular_torus(
            Lx,
            Ly;
            hopping,
            fields=zeros(Float64, Lx, Ly, 3),
        )) # 2LxLy spinful open-particle modes
        state = monopole_state(Lx, Ly, Q; particle_number=N)
        H_BdG = Matrix(state.H_BdG_func(
            QuantumNaturalfPEPS.Parameters(state),
            state.N,
        )) # shift the particle Hamiltonian s.t. the Fermi level is at zero energy
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
        #= An N-particle state is closed-shell when ε_N < μ < ε_{N+1}
        with non-0 Fermi gap ε_{N+1} - ε_N. The N-single-particle states 
        lie below the chemical potential μ and are occupied.
        =#
        @test_throws ArgumentError monopole_state(
            Lx,
            Ly,
            1;
            particle_number=N,
        )
    end
end

@testset "Triangular ordered-state Gaussian ansatze" begin
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

        dHs = QuantumNaturalfPEPS.build_H_BdG_derivatives(state)

        @test length(dHs) == length(parameters)

        T = real(float(eltype(parameters)))
        relative_step = cbrt(eps(T))

        for a in eachindex(parameters)
            step = relative_step * max(one(T), abs(parameters[a]))

            parameters_plus = copy(parameters)
            parameters_minus = copy(parameters)
            parameters_plus[a] += step
            parameters_minus[a] -= step

            H_plus = Matrix(state.H_BdG_func(parameters_plus, state.N))
            H_minus = Matrix(state.H_BdG_func(parameters_minus, state.N))
            dH_finite_difference = (H_plus - H_minus) / (2step)

            @test all(isfinite, dHs[a])
            @test dHs[a] ≈ dH_finite_difference rtol=1e-8 atol=1e-10
        end

        projected = gutzwiller_project(state; Nup=N ÷ 2)
        @test projected isa ParameterizedGutzwillerProjectedState
        @test length(QuantumNaturalfPEPS.Parameters(projected)) == length(parameters)

        if name == :stripe
            cache = QuantumNaturalfPEPS.ProjectedGaussianSchurCache(projected)
            spin_configuration = Int[]
            while !isempty(cache.remaining_sites)
                probabilities = projected_conditional_probabilities(cache)
                spin = argmax(probabilities) - 1
                push!(spin_configuration, spin)
                QuantumNaturalfPEPS.condition_projected_gaussian!(cache, spin)
            end
            gradient = gutzwiller_log_gradient(projected, spin_configuration)
            @test length(gradient) == length(parameters)
            @test all(isfinite, gradient)
        end
    end

    @test_throws ArgumentError y_hopping_fields(4, 6, eta_Y)
    @test_throws ArgumentError cs_hopping_fields(5, 6, eta_stripe)
    @test_throws DimensionMismatch y_hopping_fields(Lx, Ly, eta_Y[1:3])
    @test_throws DimensionMismatch umbrella_hopping_fields(Lx, Ly, Float64[])
    @test_throws DimensionMismatch cs_hopping_fields(Lx, Ly, [0.2, 0.3])
end
