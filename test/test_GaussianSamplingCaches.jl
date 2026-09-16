using Test
using LinearAlgebra
using Random
using QuantumNaturalfPEPS
using ITensors
using ITensorMPS

function _sampling_cache_hamiltonian(parameters, number_of_sites)
    hopping, pairing, chemical_potential = parameters
    particle_block = diagm(
        0 => fill(-chemical_potential, number_of_sites),
        1 => fill(-hopping, number_of_sites - 1),
        -1 => fill(-conj(hopping), number_of_sites - 1),
    )
    pairing_block = diagm(
        1 => fill(pairing, number_of_sites - 1),
        -1 => fill(-pairing, number_of_sites - 1),
    )
    return Hermitian([
        particle_block pairing_block
        pairing_block' -transpose(particle_block)
    ])
end

@testset "Projected parent-Slater Schur cache" begin
    Lx = Ly = 4
    number_of_sites = Lx * Ly
    gaussian_state = cs_state(
        Lx,
        Ly;
        η=[10.0],
        hopping_amplitude=0.2,
        stripe_delta=0.2,
        particle_number=number_of_sites,
    )
    projected_state = gutzwiller_project(gaussian_state)
    order = [x + (y - 1) * Lx for x in 1:Lx for y in 1:Ly]
    cache = QuantumNaturalfPEPS.ProjectedGaussianSchurCache(projected_state; order)
    prefix = Dict{Int,Int}()
    prefix_probability = 1.0

    for site in order
        probabilities = projected_conditional_probabilities(cache)
        exact = map(0:1) do spin
            candidate = copy(prefix)
            candidate[site] = spin
            QuantumNaturalfPEPS.get_prob(projected_state, candidate) /
                prefix_probability
        end
        @test probabilities ≈ exact atol=1e-10
        spin = argmax(probabilities) - 1
        prefix[site] = spin
        prefix_probability = QuantumNaturalfPEPS.get_prob(projected_state, prefix)
        QuantumNaturalfPEPS.condition_projected_gaussian!(cache, spin)
    end

    target_Sz = 1.0
    target_Nup = Int(number_of_sites / 2 + target_Sz)
    fixed_state = gutzwiller_project(gaussian_state; Nup=target_Nup)
    fixed_cache = QuantumNaturalfPEPS.ProjectedGaussianSchurCache(fixed_state; order)
    for position in eachindex(order)
        probabilities = projected_conditional_probabilities(fixed_cache)
        spin = if fixed_cache.measured_up < fixed_cache.target_Nup
            0
        else
            1
        end
        QuantumNaturalfPEPS.condition_projected_gaussian!(fixed_cache, spin)
        if position == length(order)
            @test fixed_cache.measured_up == fixed_cache.target_Nup
        end
        @test all(>=(0), probabilities)
    end

    hilbert = siteinds("S=1/2", Lx, Ly)
    peps = PEPS(ComplexF64, hilbert; bond_dim=1, show_warning=false)
    write!(peps, fill(ComplexF64(inv(sqrt(2))), length(peps)))
    sample, _, _ = QuantumNaturalfPEPS.get_sample(peps; trial_state=fixed_state)
    sampled_Sz = sum(spin == 0 ? 0.5 : -0.5 for spin in sample)
    @test sampled_Sz == target_Sz
end

@testset "Gaussian low-rank sampling caches" begin
    number_of_sites = 4
    state = QuantumNaturalfPEPS.GaussianState(
        _sampling_cache_hamiltonian,
        number_of_sites;
        η=[1.0 + 0.2im, 0.7 - 0.1im, 0.2],
        parity_sector=0,
    )

    @testset "direct sampler defaults to Schur" begin
        Random.seed!(8723)
        sample, log_probability = QuantumNaturalfPEPS.get_sample(state)
        configuration = collect(vec(sample))
        @test exp(log_probability) ≈
            QuantumNaturalfPEPS.get_prob(state, configuration) atol=1e-10
        @test mod(sum(configuration), 2) == state.parity_sector
    end
end
