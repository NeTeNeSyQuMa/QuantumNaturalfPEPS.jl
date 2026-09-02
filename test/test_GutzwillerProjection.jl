using Test
using LinearAlgebra
using Random
using ITensors
using ITensorMPS
using QuantumNaturalGradient
using QuantumNaturalfPEPS

@testset "Gutzwiller-projected Slater states" begin
    rng = MersenneTwister(8128)
    N = 4

    raw_orbitals = randn(rng, ComplexF64, 2N, N)
    occupied_orbitals = Matrix(qr(raw_orbitals).Q[:, 1:N])
    state = FixedGutzwillerProjectedState(occupied_orbitals)

    @test state.N == N
    @test state.Nup === nothing
    @test eltype(state) == ComplexF64
    @test isempty(QuantumNaturalfPEPS.Parameters(state))

    @testset "full spin-orbital determinant" begin
        for bits in 0:2^N-1
            configuration = digits(bits; base=2, pad=N)
            selected_rows = [2site - 1 + configuration[site] for site in 1:N]
            expected = det(occupied_orbitals[selected_rows, :])
            @test gutzwiller_amplitude(state, configuration) ≈ expected
            @test gutzwiller_weight(state, configuration) ≈ abs2(expected)
        end

        configuration = [0, 1, 1, 0]
        @test gutzwiller_amplitude(state, reshape(configuration, 2, 2)) ≈
              gutzwiller_amplitude(state, configuration)
        @test QuantumNaturalfPEPS.get_amplitude(state, configuration) ≈
              gutzwiller_amplitude(state, configuration)
    end

    @testset "fixed magnetization" begin
        fixed_state = gutzwiller_project(occupied_orbitals; Nup=2)
        @test fixed_state.Nup == 2
        @test gutzwiller_amplitude(fixed_state, [0, 1, 1, 0]) ≈
              gutzwiller_amplitude(state, [0, 1, 1, 0])
        @test iszero(gutzwiller_amplitude(fixed_state, [0, 0, 0, 1]))

        full_configuration = Dict(site => [0, 1, 1, 0][site] for site in 1:N)
        @test QuantumNaturalfPEPS.get_prob(fixed_state, full_configuration) ≈
              gutzwiller_weight(fixed_state, [0, 1, 1, 0])
        @test_throws ArgumentError QuantumNaturalfPEPS.get_prob(
            fixed_state,
            Dict(1 => 0),
        )
    end

    @testset "spin-conserving factorization and sign" begin
        Nup = 2
        up_orbitals = Matrix(qr(randn(rng, ComplexF64, N, Nup)).Q[:, 1:Nup])
        down_orbitals = Matrix(qr(randn(rng, ComplexF64, N, N - Nup)).Q[:, 1:N-Nup])
        block_state = gutzwiller_project(up_orbitals, down_orbitals)

        @test block_state.Nup == Nup
        for bits in 0:2^N-1
            configuration = digits(bits; base=2, pad=N)
            count(==(0), configuration) == Nup || continue

            up_sites = findall(==(0), configuration)
            down_sites = findall(==(1), configuration)
            inversions = count(down < up for down in down_sites for up in up_sites)
            sign = isodd(inversions) ? -1 : 1
            factorized_amplitude = sign *
                det(up_orbitals[up_sites, :]) *
                det(down_orbitals[down_sites, :])

            @test gutzwiller_amplitude(block_state, configuration) ≈ factorized_amplitude
        end
        @test iszero(gutzwiller_amplitude(block_state, [0, 0, 0, 1]))
    end

    @testset "occupied-orbital gauge covariance" begin
        orbital_rotation = Matrix(qr(randn(rng, ComplexF64, N, N)).Q)
        rotated_state = gutzwiller_project(occupied_orbitals * orbital_rotation)
        gauge_phase = det(orbital_rotation)

        for bits in 0:2^N-1
            configuration = digits(bits; base=2, pad=N)
            @test gutzwiller_amplitude(rotated_state, configuration) ≈
                  gauge_phase * gutzwiller_amplitude(state, configuration)
            @test gutzwiller_weight(rotated_state, configuration) ≈
                  gutzwiller_weight(state, configuration)
        end
    end

    @testset "fixed-sector exchange cache" begin
        exchange_N = 6
        exchange_Nup = 4
        exchange_orbitals = Matrix(qr(
            randn(rng, ComplexF64, 2exchange_N, exchange_N),
        ).Q[:, 1:exchange_N])
        exchange_state = gutzwiller_project(exchange_orbitals; Nup=exchange_Nup)
        configuration = [0, 1, 0, 0, 1, 0]
        cache = GutzwillerExchangeCache(exchange_state, configuration)
        amplitude = gutzwiller_amplitude(exchange_state, configuration)

        for i in findall(==(0), configuration), j in findall(==(1), configuration)
            swapped = copy(configuration)
            swapped[i], swapped[j] = swapped[j], swapped[i]
            @test gutzwiller_exchange_ratio(cache, i, j) ≈
                  gutzwiller_amplitude(exchange_state, swapped) / amplitude
        end

        ratio = gutzwiller_exchange_ratio(cache, 1, 2)
        accept_gutzwiller_exchange!(cache, 1, 2; rebuild_after=100)
        configuration[1], configuration[2] = configuration[2], configuration[1]
        amplitude *= ratio
        @test cache.configuration == configuration
        @test cache.inverse_slater ≈ inv(exchange_state.occupied_orbitals[
            [2site - 1 + configuration[site] for site in 1:exchange_N],
            :,
        ])
        @test amplitude ≈ gutzwiller_amplitude(exchange_state, configuration)
        @test exp(cache.log_amplitude) ≈ amplitude
        @test_throws ArgumentError gutzwiller_exchange_ratio(cache, 2, 3)
        @test_throws BoundsError gutzwiller_exchange_ratio(cache, 0, 1)
    end

    @testset "fixed-Sz D=1 PPEPS backend and QNG.evolve" begin
        L = 4
        number_of_sites = L^2
        number_up = 10
        up_orbitals = Matrix(qr(
            randn(rng, ComplexF64, number_of_sites, number_up),
        ).Q[:, 1:number_up])
        down_orbitals = Matrix(qr(
            randn(rng, ComplexF64, number_of_sites, number_of_sites - number_up),
        ).Q[:, 1:number_of_sites-number_up])
        projected_state = gutzwiller_project(up_orbitals, down_orbitals)

        hilbert = ITensors.siteinds("S=1/2", L, L)
        peps = PEPS(ComplexF64, hilbert; bond_dim=1, show_warning=false)
        θ = ComplexF64.(1 .+ 0.05randn(rng, length(peps)))
        write!(peps, θ)
        hamiltonian = hamiltonian_J1J2_H(
            L,
            L;
            J1=1.0,
            J2=1 / 8,
            H=0.2,
            boundary=:periodic,
        )
        ham_op = QuantumNaturalGradient.TensorOperatorSum(hamiltonian, hilbert)
        Oks_and_Eks = generate_Oks_and_Eks(
            peps,
            ham_op;
            trial_state=projected_state,
            fixed_sz_metropolis=true,
            burnin_sweeps=1,
            moves_per_sample=4,
            seed=91,
        )
        batch = Oks_and_Eks(θ, 8)

        @test size(batch[:Oks]) == (8, 2number_of_sites)
        @test length(batch[:Eks]) == 8
        @test all(sample -> count(==(0), sample) == number_up, batch[:samples])
        @test all(==(1.0), batch[:weights])
        @test 0 <= batch[:acceptance] <= 1

        transverse_batch = Oks_and_Eks(
            θ,
            8;
            measure_transverse=true,
            transverse_block_length=4,
        )
        @test size(transverse_batch[:Cperp_reference]) == (number_of_sites,)
        @test size(transverse_batch[:Cperp_displacement]) == (L, L)
        @test transverse_batch[:Cperp_reference][1] == 0.5
        @test transverse_batch[:Cperp_displacement][1, 1] == 0.5
        @test all(isfinite, transverse_batch[:Cperp_reference_error])
        @test all(isfinite, transverse_batch[:Cperp_displacement_error])

        chirality_batch = Oks_and_Eks(
            θ,
            8;
            measure_chirality=true,
            chirality_block_length=4,
        )
        @test size(chirality_batch[:chirality_up]) == (L, L)
        @test size(chirality_batch[:chirality_down]) == (L, L)
        @test size(chirality_batch[:chirality_up_error]) == (L, L)
        @test size(chirality_batch[:chirality_down_error]) == (L, L)
        @test all(isfinite, chirality_batch[:chirality_up])
        @test all(isfinite, chirality_batch[:chirality_down])
        @test all(isfinite, chirality_batch[:chirality_up_error])
        @test all(isfinite, chirality_batch[:chirality_down_error])

        adjacent_up_orbitals = Matrix(qr(
            randn(rng, ComplexF64, number_of_sites, number_up + 1),
        ).Q[:, 1:number_up+1])
        adjacent_down_orbitals = Matrix(qr(
            randn(rng, ComplexF64, number_of_sites, number_of_sites - number_up - 1),
        ).Q[:, 1:number_of_sites-number_up-1])
        adjacent_state = gutzwiller_project(adjacent_up_orbitals, adjacent_down_orbitals)
        monopole_batch = Oks_and_Eks(
            θ,
            8;
            monopole_state=adjacent_state,
            transverse_block_length=4,
        )
        @test size(monopole_batch[:monopole_matrix_element]) == (number_of_sites,)
        @test size(monopole_batch[:monopole_matrix_element_error]) == (number_of_sites,)
        @test all(isfinite, monopole_batch[:monopole_matrix_element])
        @test all(isfinite, monopole_batch[:monopole_matrix_element_error])

        local_weights = reshape(θ, 2, :)
        combined_amplitude(sample) = gutzwiller_amplitude(projected_state, sample) *
            prod(local_weights[sample[site] + 1, site] for site in eachindex(sample))

        test_configuration = vec(first(batch[:samples]))
        test_cache = GutzwillerExchangeCache(projected_state, test_configuration)
        reference_correlator = zeros(ComplexF64, number_of_sites)
        displacement_correlator = zeros(ComplexF64, L, L)
        QuantumNaturalfPEPS._fill_fixed_sz_transverse_correlations!(
            reference_correlator,
            displacement_correlator,
            test_cache,
            local_weights,
            (L, L),
            1,
        )
        test_amplitude = combined_amplitude(test_configuration)
        for site in 1:number_of_sites
            expected = if site == 1
                0.5 + 0.0im
            elseif test_configuration[site] == test_configuration[1]
                0.0 + 0.0im
            else
                exchanged = copy(test_configuration)
                exchanged[1], exchanged[site] = exchanged[site], exchanged[1]
                combined_amplitude(exchanged) / (2test_amplitude)
            end
            @test reference_correlator[site] ≈ expected
        end
        @test sum(displacement_correlator) ≈ begin
            total = 0.5number_of_sites + 0.0im
            for i in 1:number_of_sites, j in 1:number_of_sites
                i == j && continue
                test_configuration[i] == test_configuration[j] && continue
                exchanged = copy(test_configuration)
                exchanged[i], exchanged[j] = exchanged[j], exchanged[i]
                total += combined_amplitude(exchanged) / (2test_amplitude)
            end
            total / number_of_sites
        end

        chirality_up = zeros(ComplexF64, L, L)
        chirality_down = zeros(ComplexF64, L, L)
        QuantumNaturalfPEPS._fill_fixed_sz_scalar_chirality!(
            chirality_up,
            chirality_down,
            test_cache,
            local_weights,
            (L, L),
        )
        sites = (1, 2, L + 2)
        spins = Tuple(test_configuration[site] for site in sites)
        inverse_configuration = copy(test_configuration)
        inverse_configuration[collect(sites)] .= (spins[2], spins[3], spins[1])
        forward_configuration = copy(test_configuration)
        forward_configuration[collect(sites)] .= (spins[3], spins[1], spins[2])
        expected_chirality = im * (
            combined_amplitude(inverse_configuration) -
            combined_amplitude(forward_configuration)
        ) / (4test_amplitude)
        @test chirality_up[1, 1] ≈ expected_chirality


        anchor = findfirst(==(1), test_configuration)
        adjacent_configuration = copy(test_configuration)
        adjacent_configuration[anchor] = 0
        adjacent_cache = GutzwillerExchangeCache(adjacent_state, adjacent_configuration)
        monopole_estimator = zeros(ComplexF64, number_of_sites)
        QuantumNaturalfPEPS._fill_fixed_sz_monopole_matrix_element!(
            monopole_estimator,
            test_cache,
            adjacent_cache,
            anchor,
            local_weights,
        )
        adjacent_amplitude(sample) = gutzwiller_amplitude(adjacent_state, sample) *
            prod(local_weights[sample[site] + 1, site] for site in eachindex(sample))
        for site in 1:number_of_sites
            expected = if test_configuration[site] == 0
                0.0 + 0.0im
            else
                flipped = copy(test_configuration)
                flipped[site] = 0
                conj(adjacent_amplitude(flipped) / test_amplitude)
            end
            @test monopole_estimator[site] ≈ expected
        end
        for (sample_index, sample) in enumerate(batch[:samples])
            for site in 1:number_of_sites, spin in 0:1
                parameter = 2site - 1 + spin
                expected_Ok = sample[site] == spin ? inv(local_weights[spin + 1, site]) : 0
                @test batch[:Oks][sample_index, parameter] ≈ expected_Ok
            end

            amplitude = combined_amplitude(sample)
            expected_energy = 0.0 + 0.0im
            terms = QuantumNaturalGradient.get_precomp_sOψ_elems(
                ham_op,
                sample;
                get_flip_sites=true,
            )
            for (patch, coefficient) in terms
                flipped = copy(sample)
                for (site, new_spin) in patch
                    flipped[site...] = new_spin
                end
                expected_energy += coefficient * combined_amplitude(flipped) / amplitude
            end
            @test batch[:Eks][sample_index] ≈ expected_energy
        end

        qng_energy, trained_θ, _ = QuantumNaturalGradient.evolve(
            Oks_and_Eks,
            θ;
            integrator=QuantumNaturalGradient.Euler(lr=0.001),
            solver=QuantumNaturalGradient.EigenSolver(1e-4),
            sample_nr=12,
            maxiter=1,
            copy=true,
        )
        @test isfinite(qng_energy)
        @test length(trained_θ) == length(θ)

        peps_D2 = PEPS(ComplexF64, hilbert; bond_dim=2, show_warning=false)
        @test_throws ArgumentError generate_Oks_and_Eks(
            peps_D2,
            ham_op;
            trial_state=projected_state,
            fixed_sz_metropolis=true,
        )
    end

    @testset "auxiliary-Hamiltonian constructor" begin
        onsite_energies = [-4.0, 4.0, -3.0, 3.0, -2.0, 2.0, -1.0, 1.0]
        Haux = Hermitian(Matrix(Diagonal(onsite_energies)))
        polarized_state = gutzwiller_project(Haux; Nup=N)

        @test gutzwiller_weight(polarized_state, zeros(Int, N)) ≈ 1.0
        @test iszero(gutzwiller_amplitude(polarized_state, [0, 0, 0, 1]))
    end

    @testset "input validation" begin
        @test_throws DimensionMismatch FixedGutzwillerProjectedState(zeros(5, 2))
        @test_throws DimensionMismatch FixedGutzwillerProjectedState(zeros(2N, N - 1))
        @test_throws DimensionMismatch FixedGutzwillerProjectedState(zeros(N, 1), zeros(N + 1, 3))
        @test_throws DimensionMismatch FixedGutzwillerProjectedState(zeros(N, 1), zeros(N, 1))
        @test_throws ArgumentError FixedGutzwillerProjectedState(occupied_orbitals; Nup=N + 1)
        @test_throws DimensionMismatch gutzwiller_amplitude(state, [0, 1])
        @test_throws DomainError gutzwiller_amplitude(state, [0, 1, 2, 0])
    end
end
