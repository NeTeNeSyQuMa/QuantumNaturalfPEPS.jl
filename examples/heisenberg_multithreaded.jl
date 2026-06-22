using LinearAlgebra
using Statistics
using JLD2
using ITensors, ITensorMPS
using QuantumNaturalGradient
using QuantumNaturalfPEPS
using Distributed
using TimerOutputs

# Initialize 2 processes with 8 threads each, assuming you have a 16 core cpu
nr_procs, nr_threads = 2, 8 
addprocs(nr_procs, exeflags="-t $nr_threads")

@everywhere begin
    using QuantumNaturalfPEPS
    using LinearAlgebra
    BLAS.set_num_threads(1) # Each thread should only use one Blas thread
end
BLAS.set_num_threads(16) # For computing the double layers

# Simulation Parameters
params = Dict(
    :T              => Float64,
    :Lx             => 4,
    :Ly             => 4,
    :bdim           => 2,
    :lr             => 0.05,
    :J1             => 1,
    :eigencut       => 1e-4,
    :contract_cutoff=> 1e-4,
    :sample_cutoff  => 1e-3,
    :contract_dim   => 200,
    :maxiter        => 10,
    :α_init         => 3.0,
    :sample_nr      => 1000,
)

# Define the Lattice and PEPS
hilbert = siteinds("S=1/2", params[:Lx], params[:Ly])
peps = PEPS(params[:T], hilbert; bond_dim=params[:bdim], show_warning=true,
    contract_cutoff=params[:contract_cutoff], 
    contract_dim=params[:contract_dim], 
    sample_cutoff=params[:sample_cutoff],
)

# Multiply the spectrum of the PEPS by a power-law factor to make it contractible
QuantumNaturalfPEPS.multiply_algebraic_spectrum!(peps, params[:α_init])

# Construct the Hamiltonian
ham_J1 = OpSum()
for i in 1:params[:Lx], j in 1:params[:Ly], t in ["X", "Y", "Z"]
    if j < params[:Ly]
        ham_J1 .+= (params[:J1], t, (i, j), t, (i, j+1))
    end
    if i < params[:Lx]
        ham_J1 .+= (params[:J1], t, (i, j), t, (i+1, j))
    end
end

# Define a Callback Function, that is called after each iteration
function callback(; niter=1, kwargs...)
    kwargs = Dict(string(key) => v for (key, v) in kwargs)
    kwargs["params"] = params
    kwargs["solver"] = solver
    kwargs["integrator"] = integrator
    kwargs["peps"] = peps
    kwargs["timer"] = timer

    save("save.jld2", kwargs)
end

# Timer to measure the time taken for each step and subtask
timer = TimerOutput()

# Generate Operators for QNG
Oks_and_Eks = QuantumNaturalfPEPS.generate_Oks_and_Eks(peps, ham_J1; threaded=true, multiproc=true, timer) 
#Oks_and_Eks, stop_threads = QuantumNaturalfPEPS.generate_Oks_and_Eks(peps, ham_J1J2; threaded=true, multiproc=true, 
#                                                                     async_double_layers=true, verbose=true, timer) # will compute the double layers in a seperate thread

# Setup the Integrator and Solver
integrator = QuantumNaturalGradient.Euler(lr=params[:lr])
solver = QuantumNaturalGradient.EigenSolver(params[:eigencut], verbose=true)

# Initialize Parameters and Evolve
θ = QuantumNaturalGradient.Parameters(peps)

# Logger functions, will be called after each iteration and results will be saved in the misc["history"] dictionary
contract_dim(; contract_dims) = mean(contract_dims)
logger_funcs = [contract_dim]

# Evolve for a fixed (small) number of iterations as a demo
@time loss_value, trained_θ, misc = QuantumNaturalGradient.evolve(Oks_and_Eks, θ; 
        integrator, 
        verbosity=2,
        solver,
        sample_nr=params[:sample_nr],
        maxiter=params[:maxiter],
        logger_funcs,
        #misc_restart, #to restart from a previous run
        callback, 
        timer,
        )
rm("save.jld2") # remove the save file to clean up the examples directory
# stop_threads() # stop the threads, if you used async_double_layers=true