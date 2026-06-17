using Test
using ITensors, ITensorMPS
using QuantumNaturalfPEPS

@testset "ITensors site ordering is column-major" begin
    L = 3
    hilbert = siteinds("S=1/2", L, L)

    linear_index_col_major(i, j, L) = i + (j - 1) * L

    for i in 1:L, j in 1:L
        os = OpSum()
        os += (1.0, "Z", (i, j))
        ham_op = QuantumNaturalfPEPS.TensorOperatorSum(os, hilbert)

        @test length(ham_op.sites) == 1
        sites = ham_op.sites[1]
        @test length(sites) == 1
        @test sites[1] == linear_index_col_major(i, j, L)
    end
end;