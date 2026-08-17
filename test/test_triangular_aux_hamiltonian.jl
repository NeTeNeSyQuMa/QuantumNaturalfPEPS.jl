using Test
using LinearAlgebra
using QuantumNaturalfPEPS

@testset "Triangular-torus auxiliary Hamiltonian" begin
    Lx, Ly = 4, 5
    N = Lx * Ly
    bonds = triangular_torus_bonds(Lx, Ly)

    @test length(bonds) == 3N
    @test length(unique((bond.source, bond.target) for bond in bonds)) == 3N

    @test any(
        bond.source == (Lx, 2) &&
        bond.target == (1, 2) &&
        bond.winding == (1, 0)
        for bond in bonds
    )
    @test any(
        bond.source == (Lx, Ly) &&
        bond.target == (1, 1) &&
        bond.winding == (1, 1)
        for bond in bonds
    )

    tilted_bonds = triangular_torus_bonds(6, 4; shear=2)
    @test any(
        bond.source == (1, 4) &&
        bond.displacement == (0, 1) &&
        bond.target == (5, 1) &&
        bond.winding == (-1, 1)
        for bond in tilted_bonds
    )

    H_uniform = hamiltonian_aux_triangular_torus(Lx, Ly)
    @test size(H_uniform) == (2N, 2N)
    @test ishermitian(H_uniform)

    # Every spin-orbital has six triangular-lattice nearest neighbors.
    for row in axes(H_uniform, 1)
        @test count(x -> !iszero(x), H_uniform[row, :]) == 6
    end

    # A hopping callback sees the unwrapped target and can identify boundary
    # bonds when assigning Peierls phases.
    hopping = function (source, unwrapped_target, direction)
        crosses_x = unwrapped_target[1] > Lx
        crosses_y = unwrapped_target[2] > Ly
        return exp(im * (crosses_x * 0.2 + crosses_y * 0.3))
    end
    H_flux = hamiltonian_aux_triangular_torus(Lx, Ly; hopping)
    @test ishermitian(H_flux)

    @testset "staggered [0, π] plaquette flux" begin
        flux_Lx, flux_Ly = 4, 6
        flux_shear = 2
        staggered_hopping = staggered_pi_flux_hoppings(flux_Lx, flux_Ly)
        H_staggered = Matrix(hamiltonian_aux_triangular_torus(
            flux_Lx,
            flux_Ly;
            hopping=staggered_hopping,
            shear=flux_shear,
        ))

        function wrap_site(x, y)
            winding_y = fld(y - 1, flux_Ly)
            wrapped_y = y - winding_y * flux_Ly
            shifted_x = x - winding_y * flux_shear
            return (mod1(shifted_x, flux_Lx), wrapped_y)
        end
        function spin_up_index(x, y)
            wrapped_x, wrapped_y = wrap_site(x, y)
            return 2 * ((wrapped_y - 1) * flux_Lx + wrapped_x - 1) + 1
        end
        link(source, target) = H_staggered[
            spin_up_index(source...),
            spin_up_index(target...),
        ]

        # Compute the gauge-invariant hopping product around both oriented
        # triangles in every plaquette. This includes loops crossing the x
        # seam, y seam, and both seams of the torus.
        for y in 1:flux_Ly, x in 1:flux_Lx
            r = (x, y)
            rx = (x + 1, y)
            ry = (x, y + 1)
            rxy = (x + 1, y + 1)

            zero_flux_loop = link(r, rx) * link(rx, rxy) * link(rxy, r)
            pi_flux_loop = link(r, rxy) * link(rxy, ry) * link(ry, r)

            @test isapprox(zero_flux_loop, 1.0 + 0.0im; atol=1e-12)
            @test isapprox(pi_flux_loop, -1.0 + 0.0im; atol=1e-12)
        end

        # The doubled gauge unit cell cannot close around an odd-Ly torus.
        @test_throws ArgumentError staggered_pi_flux_hoppings(flux_Lx, 5)
    end

    # Check M · S on one site, including the complex Sy matrix elements.
    fields = zeros(Float64, Lx, Ly, 3)
    fields[2, 3, :] .= (2.0, 4.0, 6.0)
    H_field = Matrix(hamiltonian_aux_triangular_torus(
        Lx,
        Ly;
        hopping=0.0,
        fields,
    ))
    site = (3 - 1) * Lx + 2
    up, down = 2site - 1, 2site
    @test H_field[up, up] == 3.0
    @test H_field[down, down] == -3.0
    @test H_field[up, down] == 1.0 - 2.0im
    @test H_field[down, up] == 1.0 + 2.0im
    @test count(x -> !iszero(x), H_field) == 4

    @test_throws DimensionMismatch hamiltonian_aux_triangular_torus(
        Lx,
        Ly;
        hopping=zeros(Lx, Ly, 2),
    )
    @test_throws ArgumentError triangular_torus_bonds(2, Ly)
end
