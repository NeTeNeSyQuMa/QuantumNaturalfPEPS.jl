using Test
using ITensors
using QuantumNaturalGradient
using QuantumNaturalfPEPS

@testset "Frozen PEPS/trial-state optimization" begin
    Lx = Ly = 2
    N = Lx * Ly
    sites = ITensors.siteinds("Fermion", Lx, Ly)

    function constant_tensor(::Type{T}, incoming, outgoing) where {T<:Number}
        indices = (incoming..., outgoing...)
        return ITensor(ones(T, ITensors.dim.(indices)), indices...)
    end

    peps = PEPS(sites; bond_dim=1, tensor_init=constant_tensor)
    original_mask = copy(peps.mask)
    ham = QuantumNaturalfPEPS.hamiltonian_hubbard(0.0, 1.0, Lx, Ly)

    η = zeros(QuantumNaturalfPEPS.get_max_num_MF_params_NN(Lx, Ly))
    η[1:N] .= [-2.0, 2.0, 2.0, -2.0]
    trial_state = QuantumNaturalfPEPS.GaussianState(
        QuantumNaturalfPEPS.build_general_H_BdG_2D_NN,
        N;
        η,
    )

    peps_parameters = vec(QuantumNaturalGradient.Parameters(peps).obj)

    fixed_trial = QuantumNaturalfPEPS.generate_Oks_and_Eks(
        peps,
        ham;
        trial_state,
        fix_trial_state=true,
    )
    fixed_trial_batch = fixed_trial(peps_parameters, 2)
    @test size(fixed_trial_batch[:Oks]) == (2, length(peps_parameters))

    fixed_peps = QuantumNaturalfPEPS.generate_Oks_and_Eks(
        peps,
        ham;
        trial_state,
        fix_peps=true,
    )
    fixed_peps_batch = fixed_peps(copy(η), 2)
    @test size(fixed_peps_batch[:Oks]) == (2, length(η))
    @test peps.mask == original_mask

    @test_throws ArgumentError QuantumNaturalfPEPS.generate_Oks_and_Eks(
        peps,
        ham;
        trial_state,
        fix_peps=true,
        fix_trial_state=true,
    )
end
