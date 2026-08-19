using Test
using LinearAlgebra
using QuantumNaturalfPEPS

@testset "Parton Hartree-Fock self-consistency" begin
    @testset "Wick energy of product and hopping states" begin
        # The 2x1 open triangular patch contains one nearest-neighbor bond.
        Lx, Ly = 2, 1
        N = Lx * Ly

        ferromagnet = zeros(ComplexF64, 2N, N)
        ferromagnet[1, 1] = 1
        ferromagnet[3, 2] = 1
        Pferro = parton_density_matrix(ferromagnet)
        @test parton_mean_field_energy(
            Pferro,
            Lx,
            Ly;
            J1=1,
            J2=0,
            H=0.4,
            boundary=:open,
        ) ≈ 1 / 4 - 0.4

        neel_product = zeros(ComplexF64, 2N, N)
        neel_product[1, 1] = 1
        neel_product[4, 2] = 1
        Pneel = parton_density_matrix(neel_product)
        @test parton_mean_field_energy(
            Pneel,
            Lx,
            Ly;
            J1=1,
            J2=0,
            boundary=:open,
        ) ≈ -1 / 4

        # Fill the up- and down-spin bonding orbitals. Wick's theorem gives
        # <S1.S2> = -3/8 in this unprojected, number-fluctuating state.
        bonding = ComplexF64[
            inv(sqrt(2)) 0
            0 inv(sqrt(2))
            inv(sqrt(2)) 0
            0 inv(sqrt(2))
        ]
        @test parton_mean_field_energy(
            parton_density_matrix(bonding),
            Lx,
            Ly;
            J1=1,
            J2=0,
            boundary=:open,
        ) ≈ -3 / 8
    end

    @testset "Fock matrix differentiates the Wick functional" begin
        Lx = Ly = 2
        number_of_modes = 2Lx * Ly
        A = reshape(
            ComplexF64.(1:number_of_modes^2),
            number_of_modes,
            number_of_modes,
        )
        A .+= im .* reverse(A; dims=1)
        P = (A + A') / (20number_of_modes^2)
        direction = Hermitian(Matrix{ComplexF64}(I, number_of_modes, number_of_modes))
        direction.data[1, 3] = 0.2 + 0.3im
        direction.data[3, 1] = 0.2 - 0.3im

        energy(X) = parton_mean_field_energy(
            X,
            Lx,
            Ly;
            J1=1,
            J2=1 / 8,
            boundary=:open,
        )
        epsilon = 1e-6
        numerical_derivative = (
            energy(P + epsilon * direction) - energy(P - epsilon * direction)
        ) / (2epsilon)
        F = QuantumNaturalfPEPS._parton_exchange_fock(
            P,
            Lx,
            Ly;
            J1=1,
            J2=1 / 8,
            boundary=:open,
            shear=0,
        )
        @test numerical_derivative ≈ real(tr(F * direction)) atol=1e-8
    end

    @testset "self-consistent polarized saddle" begin
        Lx = Ly = 3
        result = solve_parton_mean_field(
            Lx,
            Ly;
            J1=0,
            J2=0,
            H=1,
            boundary=:periodic,
            target_m=1,
            initial_constraint_field=1,
            mixing=1,
            tolerance=1e-11,
            maxiter=20,
        )

        @test result.converged
        @test result.magnetization ≈ 1 atol=1e-12
        @test result.local_densities ≈ ones(Lx, Ly) atol=1e-12
        @test result.exchange_energy ≈ 0 atol=1e-12
        @test result.zeeman_energy ≈ -(Lx * Ly) / 2 atol=1e-12
        @test result.energy ≈ -(Lx * Ly) / 2 atol=1e-12
        @test size(result.occupied_orbitals) == (2Lx * Ly, Lx * Ly)

        projected = gutzwiller_project(result; Nup=Lx * Ly)
        @test projected.Nup == Lx * Ly
        @test gutzwiller_weight(projected, zeros(Int, Lx * Ly)) ≈ 1
    end

    @testset "validation" begin
        @test_throws DimensionMismatch parton_mean_field_energy(
            zeros(3, 3),
            2,
            1;
            J1=1,
            J2=0,
            boundary=:open,
        )
        @test_throws ArgumentError solve_parton_mean_field(
            3,
            3;
            J1=1,
            J2=0,
            target_m=2,
        )
    end
end
