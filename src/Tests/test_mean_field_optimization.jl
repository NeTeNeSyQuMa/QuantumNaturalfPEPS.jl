using QuantumNaturalfPEPS
using QuantumNaturalGradient
using LinearAlgebra
using Test
using Random
using SkewLinearAlgebra

Random.seed!(1234)

N = 4

η_exact_arr = [[1.0, 2.0, -0.2], [1.0, -2.0, 0.05], [-5.0, 4.0, 0.01]]
η0_arr = [rand(3) for _ in 1:length(η_exact_arr)]

function build_H_BdG_mat(η, N)
    t = η[1]
    Δ = η[2]
    μ = η[3]

    T = diagm(0 => fill(-μ, N), 1 => fill(-t, N-1), -1 => fill(-t, N-1))
    D = diagm(1 => fill(Δ, N-1), -1 => fill(-Δ, N-1))

    H = [T D; D' -transpose(T)]
    return Hermitian(H)
end

for (i,η_exact) in enumerate(η_exact_arr)
    @show η_exact

    η_exact ./= η_exact[1]
    H_BdG = build_H_BdG_mat(η_exact, N)

    η0 = η0_arr[i]
    η_start = copy(η0)

    @show η0

    # get exact ground state energy for comparison
    eigenvalues = eigvals(H_BdG)
    E_exact = real(sum(eigenvalues[eigenvalues .< 0]) / 2 + sum(diag(H_BdG[1:N, 1:N])) / 2)
    @show E_exact

    # Generate Operators for QNG
    Oks_and_Eks = QuantumNaturalfPEPS.generate_Oks_and_Eks_Slater(H_BdG, build_H_BdG_mat, N)

    # Setup the Integrator and Solver
    integrator = QuantumNaturalGradient.Euler(lr=0.08)
    solver = QuantumNaturalGradient.EigenSolver()

    # Evolve for a fixed (small) number of iterations as a demo
    @time loss_value, trained_η, misc = QuantumNaturalGradient.evolve(Oks_and_Eks, η0; 
            integrator, 
            verbosity=2,
            solver,
            sample_nr=100,
            maxiter=500,
    )

    @show η_start
    @show η_exact
    @show trained_η ./ trained_η[1]
    @show E_exact
    @show loss_value

    @test isapprox(loss_value, E_exact; atol=1e-10)
    @test all(isfinite, trained_η)

    @test isapprox((trained_η ./ trained_η[1]), η_exact ./ η_exact[1]; atol=1e-10)
end;