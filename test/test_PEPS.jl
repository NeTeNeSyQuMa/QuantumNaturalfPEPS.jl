using Test
using ITensors, ITensorMPS
using QuantumNaturalfPEPS

@testset "PEPS functionalities" begin

    @testset "PEPS ordering is column-major" begin
        Lx = 2
        Ly = 3
        hilbert = ITensors.siteinds("S=1/2", Lx, Ly)
        peps = PEPS(hilbert; bond_dim=1)

        theta_vec = Float64[i for i in 1:2*Lx*Ly]
        write!(peps, theta_vec)

        @test vec(peps) == theta_vec
    end

    @testset "Write a raw array into a local PEPS tensor with write_Tensor!" begin
        hilbert = ITensors.siteinds("S=1/2", 2, 1)
        peps = PEPS(hilbert; bond_dim=2) # 2-site peps

        # write data to the tensor at (2, 1) w/ total dimension 4 (2 physical x 2 virtual)
        data = [1.0 2.0; 3.0 4.0]
        QuantumNaturalfPEPS.write_Tensor!(peps, data, 2, 1)

        link = only(linkinds(peps, 2, 1))
        site = siteind(peps, 2, 1)

        @test collect(inds(peps[2, 1])) == [site, link]
        @test peps[2, 1][site => 1, link => 1] == data[1, 1]
        @test peps[2, 1][site => 1, link => 2] == data[2, 1]
        @test peps[2, 1][site => 2, link => 1] == data[1, 2]
        @test peps[2, 1][site => 2, link => 2] == data[2, 2]
    end

    @testset "Write an ITensor tensor into a local PEPS tensor with write_Tensor!" begin
        hilbert = siteinds("S=1/2", 2, 1)
        peps = PEPS(hilbert; bond_dim=2)

        data = [1.0 2.0; 3.0 4.0]
        input_link = Index(2, "input_link")
        input_site = Index(2, "input_site")
        tensor = ITensor(data, input_link, input_site)
        QuantumNaturalfPEPS.write_Tensor!(peps, tensor, 2, 1)

        link = only(linkinds(peps, 2, 1))
        site = siteind(peps, 2, 1)

        @test collect(inds(peps[2, 1])) == [site, link]
        @test peps[2, 1][site => 1, link => 1] == data[1, 1]
        @test peps[2, 1][site => 1, link => 2] == data[2, 1]
        @test peps[2, 1][site => 2, link => 1] == data[1, 2]
        @test peps[2, 1][site => 2, link => 2] == data[2, 2]
    end
    
end;