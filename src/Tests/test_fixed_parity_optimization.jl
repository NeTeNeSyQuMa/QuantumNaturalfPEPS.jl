using QuantumNaturalfPEPS
using QuantumNaturalGradient
using LinearAlgebra
using Random
using Test

Random.seed!(1234)

L = 3
N = L*L
Nsamples = 100

tol = 1e-10

# η_exact = [3.0, 1.0, 2.5]
η_exact = [1.0, 1.0, 0.0]
η_exact ./= η_exact[1]

η0 = rand(length(η_exact))

function build_history_callback()
    η_history = Vector{Vector{Float64}}()

    callback = function (; state, misc, niter)
        # @show state.θ
        push!(η_history, copy(Float64.(state.θ)))
        return true
    end

    return callback, η_history
end

function build_H_BdG_mat(η, N)
    t = η[1]
    Δ = η[2]
    μ = η[3]

    T = diagm(0 => fill(-μ, N), 1 => fill(-t, N-1), -1 => fill(-t, N-1))
    D = diagm(1 => fill(Δ, N-1), -1 => fill(-Δ, N-1))

    H = [T D; D' -transpose(T)]
    return Hermitian(H)
end

H_BdG = build_H_BdG_mat(η_exact, N)

# get exact ground state energy for comparison
eigenvalues = eigvals(H_BdG)
E_exact = real(sum(eigenvalues[1:N]) / 2 + sum(diag(H_BdG[1:N, 1:N])) / 2)
# @show E_exact

# get energy of first excited state
E_excited = E_exact + minimum(abs.(eigenvalues))
# @show E_excited

# determine parity of ground state
GS = QuantumNaturalfPEPS.GaussianState(build_H_BdG_mat, N; η=η_exact)

parity_sector = begin
    if GS.occ_ref == zeros(Int, N)
        # println("Ground state parity: " * (QuantumNaturalfPEPS.getParity(GS) == 0 ? "Even" : "Odd"))
        QuantumNaturalfPEPS.getParity(GS)
    else # if the reference state is not the quasiparticle vacuum, the quasiparticle vacuum has flipped parity compared to the state we construct
        # println("Ground state parity: " * (QuantumNaturalfPEPS.getParity(GS) == 0 ? "Odd" : "Even"))
        1 - QuantumNaturalfPEPS.getParity(GS)
    end
end

@testset "Correct parity sector for ground state" begin
    η_init = deepcopy(η0)

    # Generate Operators for QNG
    Oks_and_Eks = QuantumNaturalfPEPS.generate_Oks_and_Eks_Slater(H_BdG, build_H_BdG_mat, N; parity_sector=parity_sector, target_state=0)

    # Setup the Integrator and Solver
    integrator = QuantumNaturalGradient.Euler(lr=0.09)
    solver = QuantumNaturalGradient.EigenSolver()
    callback, η_history = build_history_callback()

    # Evolve for a fixed (small) number of iterations as a demo
    @time loss_value, trained_η, misc = QuantumNaturalGradient.evolve(Oks_and_Eks, η_init; 
            integrator, 
            verbosity=0,
            callback,
            solver,
            sample_nr=Nsamples,
            maxiter=100,
    )   

    η_history_mat = isempty(η_history) ? zeros(length(η0), 0) : hcat(η_history...)
    parity_hist = Vector{Int}(undef, size(η_history_mat, 2))
    for i in 1:size(η_history_mat, 2)
        η = vec(η_history_mat[:, i])
        parity_hist[i] = QuantumNaturalfPEPS.getParity(QuantumNaturalfPEPS.GaussianState(build_H_BdG_mat, N; η=η, parity_sector=parity_sector, target_state=0))
    end

    # @show η0
    # @show η_exact
    # @show trained_η ./ trained_η[1]
    # @show loss_value
    # @show E_exact
    # @show E_excited

    @test isapprox(trained_η ./ trained_η[1], η_exact ./ η_exact[1]; atol=tol)
    @test isapprox(loss_value, E_exact; atol=tol)
    @test all(parity_hist .== parity_sector)
end

@testset "Wrong parity sector for ground state matches 1st excited state test" begin
    parity_sector_1st_excited = 1 - parity_sector # flip parity sector compared to ground state

    η_init = deepcopy(η0)

    # Generate Operators for QNG
    Oks_and_Eks = QuantumNaturalfPEPS.generate_Oks_and_Eks_Slater(H_BdG, build_H_BdG_mat, N; parity_sector=parity_sector_1st_excited, target_state=0)

    # Setup the Integrator and Solver
    integrator = QuantumNaturalGradient.Euler(lr=0.05)
    solver = QuantumNaturalGradient.EigenSolver()
    callback, η_history = build_history_callback()

    # Evolve for a fixed (small) number of iterations as a demo
    @time loss_value, trained_η, misc = QuantumNaturalGradient.evolve(Oks_and_Eks, η_init; 
            integrator, 
            verbosity=0,
            callback,
            solver,
            sample_nr=Nsamples,
            maxiter=100,
    )   

    η_history_mat = isempty(η_history) ? zeros(length(η0), 0) : hcat(η_history...)
    parity_hist = Vector{Int}(undef, size(η_history_mat, 2))
    for i in 1:size(η_history_mat, 2)
        η = vec(η_history_mat[:, i])
        parity_hist[i] = QuantumNaturalfPEPS.getParity(QuantumNaturalfPEPS.GaussianState(build_H_BdG_mat, N; η=η, parity_sector=parity_sector_1st_excited, target_state=0))
    end

    # @show η_exact
    # @show trained_η ./ trained_η[1]
    # @show loss_value
    # @show E_exact
    # @show E_excited

    @test isapprox(trained_η ./ trained_η[1], η_exact ./ η_exact[1]; atol=tol)
    @test isapprox(loss_value, E_excited; atol=tol)
    @test all(parity_hist .== parity_sector_1st_excited)
end;