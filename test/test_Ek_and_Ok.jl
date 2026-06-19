using Test
using QuantumNaturalfPEPS
using QuantumNaturalGradient
using Random

import QuantumNaturalfPEPS: AbstractPEPS
import ITensorMPS: siteinds, OpSum

Random.seed!(1234)

function Test_Ok(peps::AbstractPEPS, S::Matrix{Int64}, direction; env_top=nothing, env_down=nothing, h_envs_r=nothing, h_envs_l=nothing, logψ=nothing, dt=0.0001)
    if logψ === nothing || env_top === nothing || env_down === nothing
        logψ, env_top, env_down = QuantumNaturalfPEPS.get_logψ_and_envs(peps, S)
    end
    if h_envs_r === nothing || h_envs_l === nothing
        h_envs_r, h_envs_l = QuantumNaturalfPEPS.get_all_horizontal_envs(peps, env_top, env_down, S) 
    end

    Ok = transpose(QuantumNaturalfPEPS.get_Ok(peps, env_top, env_down, S, logψ; h_envs_r, h_envs_l)) * direction
    Oknum = QuantumNaturalfPEPS.numerical_Ok(peps, S, direction; dt=dt)

    @test Ok ≈ Oknum rtol=1e-5
end

function Test_logψ(peps::AbstractPEPS, S)
    logψ, et, ed = QuantumNaturalfPEPS.get_logψ_and_envs(peps, S)
    logψex = QuantumNaturalfPEPS.logψ_exact(peps, S)

    @test logψ ≈ logψex rtol=1e-10
end

function Test_Ek(peps::AbstractPEPS, ham::OpSum; it=1)
    hilbert = siteinds(peps)
    ham_op = QuantumNaturalfPEPS.TensorOperatorSum(ham, hilbert)

    return Test_Ek(peps::AbstractPEPS, ham_op; it)
end

function Test_Ek(peps::AbstractPEPS, ham_op; it=1)
    E = Array{eltype(peps)}(undef, it)
    Enum = Array{eltype(peps)}(undef, it)

    func = get_logψ_func(peps)
    for i in 1:it
        S = rand([0,1], size(peps)) 
        logψ, env_top, env_down = QuantumNaturalfPEPS.get_logψ_and_envs(peps, S)
        h_envs_r, h_envs_l = QuantumNaturalfPEPS.get_all_horizontal_envs(peps, env_top, env_down, S) 
        fourb_envs_r, fourb_envs_l = QuantumNaturalfPEPS.get_all_4b_envs(peps, env_top, env_down, S)

        E[i] = QuantumNaturalfPEPS.get_Ek(peps, ham_op, env_top, env_down, S, logψ; h_envs_r, h_envs_l, fourb_envs_r, fourb_envs_l)
        Enum[i] = QuantumNaturalfPEPS.convert_if_real(QuantumNaturalGradient.get_Ek(S, ham_op, func))
    end

    @test E ≈ Enum rtol=1e-5
end

function get_logψ_func(peps)
    function logψ_func(sample)
        return (QuantumNaturalfPEPS.logψ_exact(peps, sample))
    end
    return logψ_func
end

function get_Ek(peps::AbstractPEPS, ham::OpSum; it=100)
    hilbert = siteinds(peps)
    ham_op = TensorOperatorSum(ham, hilbert)
    return get_Ek(peps::AbstractPEPS, ham_op; it)
end

function get_Ek(peps, ham_op; it=100)
    E = Array{eltype(peps)}(undef, it)

    QuantumNaturalfPEPS.update_double_layer_envs!(peps)

    for i in 1:it
        S, _ = QuantumNaturalfPEPS.get_sample(peps)
        logψ, env_top, env_down = QuantumNaturalfPEPS.get_logψ_and_envs(peps, S)
        h_envs_r, h_envs_l = QuantumNaturalfPEPS.get_all_horizontal_envs(peps, env_top, env_down, S) 
        fourb_envs_r, fourb_envs_l = QuantumNaturalfPEPS.get_all_4b_envs(peps, env_top, env_down, S)

        E[i] = QuantumNaturalfPEPS.get_Ek(peps, ham_op, env_top, env_down, S, logψ; h_envs_r, h_envs_l, fourb_envs_r, fourb_envs_l)
    end

    return E
end

@testset "Ok and Ek tests" begin
    L = 3
    hilbert = siteinds("S=1/2", L, L)
    peps = PEPS(hilbert; bond_dim=2)
    S = rand([0,1], size(peps))
    
    @testset "Test Ok" begin
        Test_Ok(peps, S, randn(length(peps)); dt=1e-5)
    end
    @testset "Test logψ" begin
        Test_logψ(peps, S)
    end
    @testset "Test Ek" begin
        ham_J1J2 = QuantumNaturalfPEPS.hamiltonain_J1J2(1.0, 0.5, L, L)
        Test_Ek(peps, ham_J1J2; it=10)
    end
end;