using Test
using ITensors, ITensorMPS
using QuantumNaturalfPEPS

@testset "PEPS ordering is column-major" begin
    Lx = 2
    Ly = 3
    hilbert = siteinds("S=1/2", Lx, Ly)
    peps = PEPS(hilbert; bond_dim=1)

    theta_vec = Float64[i for i in 1:2*Lx*Ly]
    write!(peps, theta_vec)

    @test vec(peps) == theta_vec
end;