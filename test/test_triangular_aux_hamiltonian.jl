using Test
using LinearAlgebra
using ITensors
using QuantumNaturalfPEPS

@testset "Triangular-lattice Hamiltonians" begin
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

    @testset "next-nearest-neighbor torus bonds" begin
        J2_bonds = triangular_torus_bonds(4, 4; shell=2)
        @test length(J2_bonds) == 3 * 4 * 4
        @test Set(bond.displacement for bond in J2_bonds) ==
              Set(((1, -1), (1, 2), (2, 1)))

        # On an ordinary 4x4 torus, every wrapped J2 pair is distinct.
        J2_pairs = [
            minmax(
                (bond.source[2] - 1) * 4 + bond.source[1],
                (bond.target[2] - 1) * 4 + bond.target[1],
            )
            for bond in J2_bonds
        ]
        @test length(unique(J2_pairs)) == length(J2_pairs)

        # For the tilted 24-site torus, T2=(2,4)=2(1,2), so the two
        # periodic images along that direction connect the same wrapped pair.
        # Both bonds must remain in the quotient-lattice Hamiltonian.
        tilted_J2_bonds = triangular_torus_bonds(6, 4; shear=2, shell=2)
        tilted_J2_pairs = [
            minmax(
                (bond.source[2] - 1) * 6 + bond.source[1],
                (bond.target[2] - 1) * 6 + bond.target[1],
            )
            for bond in tilted_J2_bonds
        ]
        @test length(tilted_J2_bonds) == 3 * 24
        @test length(unique(tilted_J2_pairs)) == 60
        @test maximum(count(==(pair), tilted_J2_pairs) for pair in unique(tilted_J2_pairs)) == 2

        @test_throws ArgumentError triangular_torus_bonds(4, 4; shell=3)
    end

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

    @testset "uniform flux on the staggered [0, π] background" begin
        flux_Lx, flux_Ly = 4, 6
        Q = 5
        hopping, uniform_flux = uniform_flux_staggered_pi_hoppings(
            flux_Lx,
            flux_Ly,
            Q,
        )
        H_uniform_flux = Matrix(hamiltonian_aux_triangular_torus(
            flux_Lx,
            flux_Ly;
            hopping,
        ))

        spin_up_index(x, y) = 2 * ((mod1(y, flux_Ly) - 1) * flux_Lx + mod1(x, flux_Lx) - 1) + 1
        link(source, target) = H_uniform_flux[
            spin_up_index(source...),
            spin_up_index(target...),
        ]

        expected_triangle_phase = exp(im * uniform_flux / 2)
        for y in 1:flux_Ly, x in 1:flux_Lx
            r = (x, y)
            rx = (x + 1, y)
            ry = (x, y + 1)
            rxy = (x + 1, y + 1)
            zero_background = link(r, rx) * link(rx, rxy) * link(rxy, r)
            pi_background = link(r, rxy) * link(rxy, ry) * link(ry, r)

            @test zero_background ≈ expected_triangle_phase atol=1e-12
            @test pi_background ≈ -expected_triangle_phase atol=1e-12
        end
        @test uniform_flux ≈ 2pi * Q / (flux_Lx * flux_Ly)
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

    @testset "physical triangular J1-J2 Hamiltonian" begin
        physical_N = 4 * 4
        physical_hamiltonian = hamiltonian_J1J2_H(
            4,
            4;
            J1=1.0,
            J2=1 / 8,
            H=0.3,
        )

        # The open 4x4 patch has 33 J1 bonds and 21 J2 bonds. Each bond
        # contributes three spin components, and the field contributes N terms.
        @test length(physical_hamiltonian) == 3 * (33 + 21) + physical_N

        physical_terms = ITensors.Ops.terms(physical_hamiltonian)
        first_exchange = first(physical_terms)
        first_exchange_ops = ITensors.Ops.terms(ITensors.Ops.argument(first_exchange))
        @test ITensors.Ops.coefficient(first_exchange) == 1.0
        @test ITensors.Ops.which_op.(first_exchange_ops) == ["Sx", "Sx"]
        @test ITensors.Ops.site.(first_exchange_ops) == [(1, 1), (2, 1)]

        last_field = last(physical_terms)
        last_field_ops = ITensors.Ops.terms(ITensors.Ops.argument(last_field))
        @test ITensors.Ops.coefficient(last_field) == -0.3
        @test only(last_field_ops) == ITensors.Ops.Op("Sz", (4, 4))

        periodic_hamiltonian = hamiltonian_J1J2_H(
            4,
            4;
            J1=1.0,
            J2=1 / 8,
            H=0.3,
            boundary=:periodic,
        )
        @test length(periodic_hamiltonian) ==
              3 * (3physical_N + 3physical_N) + physical_N

        @test length(hamiltonian_J1J2_H(4, 4; J1=1, J2=0, H=0)) == 3 * 33
        @test length(hamiltonian_J1J2_H(4, 4; J1=0, J2=0, H=1)) == physical_N
        @test isempty(hamiltonian_J1J2_H(4, 4; J1=0, J2=0, H=0))
        @test_throws ArgumentError hamiltonian_J1J2_H(
            4,
            4;
            J1=1,
            J2=0,
            H=0,
            boundary=:cylindrical,
        )
        @test_throws ArgumentError hamiltonian_J1J2_H(
            4,
            4;
            J1=1,
            J2=0,
            H=0,
            boundary=:open,
            shear=1,
        )
    end
end
