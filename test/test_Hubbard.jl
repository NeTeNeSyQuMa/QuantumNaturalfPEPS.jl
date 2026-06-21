using Test
using ITensors
using QuantumNaturalfPEPS
using QuantumNaturalGradient
using Random

Random.seed!(1234)

@testset "Hubbard model half-filling" begin
    
    function build_Hubbard_hamiltonian(Lx, Ly, t, U)
        Hubbard_ham = ITensors.OpSum()

        for i in 1:Lx, j in 1:Ly
            if j < Ly
                
                if t != 0
                    Hubbard_ham .+= (-t, "Cdag", (i, j),   "C", (i, j+1))
                    Hubbard_ham .+= (-t, "Cdag", (i, j+1), "C", (i, j))
                end

                # V (n_i - 1/2)(n_j - 1/2), constant dropped
                Hubbard_ham .+= ( U,    "N", (i, j),   "N", (i, j+1))
                Hubbard_ham .+= (-U/2, "N", (i, j))
                Hubbard_ham .+= (-U/2, "N", (i, j+1))
            end

            if i < Lx
                
                if t != 0
                    Hubbard_ham .+= (-t, "Cdag", (i, j),   "C", (i+1, j))
                    Hubbard_ham .+= (-t, "Cdag", (i+1, j), "C", (i, j))
                end

                # V (n_i - 1/2)(n_j - 1/2), constant dropped
                Hubbard_ham .+= ( U,    "N", (i, j),   "N", (i+1, j))
                Hubbard_ham .+= (-U/2, "N", (i, j))
                Hubbard_ham .+= (-U/2, "N", (i+1, j))
            end

        end
        return Hubbard_ham
    end

    # Callback function to save data at each iteration
    function build_history_callback()
        θ_history = Vector{Vector{Float64}}()
        θ_PEPS_history = Vector{Vector{Float64}}()
        η_history = Vector{Vector{Float64}}()

        callback = function (; state, misc, niter)
            θ_now = copy(Float64.(state.θ))

            θ_PEPS_now = θ_now[1:length(θ_PEPS)]
            η_now = θ_now[length(θ_PEPS)+1:end]

            push!(θ_history, copy(θ_now))
            push!(θ_PEPS_history, copy(θ_PEPS_now))
            push!(η_history, copy(η_now))
            
            # Save current parameter vectors for restart
            params = Dict(
                "θ_PEPS" => copy(θ_PEPS_now),
                "η" => copy(η_now),
            )

        end

        return callback, θ_history, θ_PEPS_history, η_history
    end

    # functions to compute observables
    function build_Ntot_op(L::Int)
        Ntot_op = ITensors.OpSum()

        for i in 1:L, j in 1:L
            Ntot_op .+= (1.0, "N", (i, j))
        end

        return Ntot_op
    end

    function build_M_cdw2_op(L::Int)
        M2_op = ITensors.OpSum()

        sites = Tuple{Int,Int,Float64}[]

        for i in 1:L, j in 1:L
            push!(sites, (i, j, float((-1)^(i + j))))
        end

        # M^2 = sum_a n_a + 2 sum_{a<b} s_a s_b n_a n_b
        # because n_a^2 = n_a for fermion number operators.
        for a in eachindex(sites)
            i, j, s = sites[a]
            M2_op .+= (1.0, "N", (i, j))

            for b in a+1:length(sites)
                k, l, sp = sites[b]
                M2_op .+= (2.0 * s * sp, "N", (i, j), "N", (k, l))
            end
        end

        return M2_op
    end

    function build_nn_avg_op(L::Int)
        nn_op = ITensors.OpSum()
        Nb = 2L * (L - 1)
        coeff = 1.0 / Nb

        for i in 1:L, j in 1:L
            if j < L
                nn_op .+= (coeff, "N", (i, j), "N", (i, j+1))
            end

            if i < L
                nn_op .+= (coeff, "N", (i, j), "N", (i+1, j))
            end
        end

        return nn_op
    end

    # set up model
    Lx, Ly = 4, 4
    N = Lx * Ly
    t = 0.0
    V = 2.0
    Hubbard_ham = build_Hubbard_hamiltonian(Lx, Ly, t, V)
    parity_sector = 0
    target_state = 0

    # set up simulation parameters
    Nsamples = 1000
    maxiters = 50
    Nmeasure = 100
    multiproc=false
    shared_array=false

    # set up Hilbert space and PEPS parameters
    bond_dim = 2
    hilbert = ITensors.siteinds("Fermion", Lx, Ly)
    peps = PEPS(hilbert; bond_dim=bond_dim)
    QuantumNaturalfPEPS.multiply_algebraic_spectrum!(peps, 3.) # Multiply the spectrum of the PEPS by a power-law factor as described in arXiv/2503.12557

    # set up mean-field parameters
    n_max_MF_params = QuantumNaturalfPEPS.get_max_num_MF_params_NN(Lx, Ly)
    η = zeros(Float64, n_max_MF_params)

    nx = QuantumNaturalfPEPS.get_max_num_hopping_x_NN(Lx, Ly)
    ny = QuantumNaturalfPEPS.get_max_num_hopping_y_NN(Lx, Ly)
    hx_range = N+1 : N+nx
    hy_range = N+nx+1 : N+nx+ny
    px_range = N+nx+ny+1 : N+nx+ny+nx
    py_range = N+nx+ny+nx+1 : N+nx+ny+nx+ny

    t_mf = 0.05      # small hopping: keeps the reference soft
    Δ = 0.0         # no Cooper pairing for a pure CDW reference
    m_cdw = 1.0     # staggered onsite potential strength

    # staggered onsite potential
    for y in 1:Ly, x in 1:Lx
        idx = x + (y - 1) * Lx   # same linearization as build_general_H_BdG_2D_NN
        η[idx] = -m_cdw * (-1)^(x + y)
    end

    # small hopping to avoid a hard delta distribution
    η[hx_range] .= -t_mf
    η[hy_range] .= -t_mf

    # no pairing for CDW
    η[px_range] .= 0.0
    η[py_range] .= 0.0
    # η = 1e-8 .+ (9e-8 - 1e-8) * rand(n_max_MF_params)


    # set up PEPS and mean-field states
    trial_state = QuantumNaturalfPEPS.GaussianState(QuantumNaturalfPEPS.build_general_H_BdG_2D_NN, N; η=η, parity_sector=parity_sector, target_state=target_state)
    θ_PEPS, η = vec(QuantumNaturalGradient.Parameters(peps).obj), QuantumNaturalfPEPS.Parameters(trial_state)
    θ = Vector{eltype(θ_PEPS)}(vcat(θ_PEPS, η))
    write!(peps, θ_PEPS)
    write!(trial_state, η)

    # Setup the Integrator and Solver
    integrator = QuantumNaturalGradient.Euler(lr=0.05)
    solver = QuantumNaturalGradient.EigenSolver()
    Oks_and_Eks = QuantumNaturalfPEPS.generate_Oks_and_Eks(peps, Hubbard_ham; trial_state=trial_state, multiproc=multiproc, shared_array=shared_array)
    callback, θ_history, θ_PEPS_history, η_history = build_history_callback()

    @time loss_value, trained_θ, misc = QuantumNaturalGradient.evolve(Oks_and_Eks, θ; 
    integrator, 
    verbosity=2,
    callback,
    solver,
    sample_nr=Nsamples,
    maxiter=maxiters)

    η_history_mat = hcat(η_history...);
    θ_history_mat = hcat(θ_history...);

    energy , energy_err, _ = get_observable(peps, trial_state, Hubbard_ham, trained_θ; sample_nr=Nmeasure, multiproc=multiproc)
    Ntot_mean , Ntot_err, _ = get_observable(peps, trial_state, build_Ntot_op(Lx), trained_θ; sample_nr=Nmeasure, multiproc=multiproc)
    M2_mean , M2_error, _ = get_observable(peps, trial_state, build_M_cdw2_op(Lx), trained_θ; sample_nr=Nmeasure, multiproc=multiproc)
    nn_avg_mean , nn_avg_error, _ = get_observable(peps, trial_state, build_nn_avg_op(Lx), trained_θ; sample_nr=Nmeasure, multiproc=multiproc)

    atol = 1e-8
    @test abs(Ntot_mean - 8.0) <= 5Ntot_err + atol
    @test abs(loss_value - (-24.0)) <= 5energy_err + atol
    @test abs(M2_mean / N - 4.0) <= M2_error + atol
    @test abs(nn_avg_mean - 0.0) <= nn_avg_error + atol

end