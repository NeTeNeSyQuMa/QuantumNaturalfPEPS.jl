# QuantumNaturalfPEPS.jl
`QuantumNaturalfPEPS` is a Julia package for simulating quantum many-body systems using the PEPS (Projected Entangled Pair States) framework via sampling. 
# Feature Overview
- **PEPS Framework**: Provides a flexible and efficient way to represent quantum many-body states.
- **Natural Gradient Descent**: Implements the natural gradient descent algorithm for optimizing PEPS parameters using sampling techniques as described in [arXiv/2503.12557](https://arxiv.org/abs/2503.12557).
- **Parallelization Support**: Designed to efficiently parallelize sampling and optimization across multiple cores or nodes.

# Minimal Example

```julia
using ITensors
using ITensorMPS
using QuantumNaturalGradient
using QuantumNaturalfPEPS

# Define the Lattice and PEPS
L = 4
hilbert = siteinds("S=1/2", L, L)
peps = PEPS(hilbert; bond_dim=2)

# Multiply the spectrum of the PEPS by a power-law factor as described in arXiv/2503.12557
QuantumNaturalfPEPS.multiply_algebraic_spectrum!(peps, 3.)

# Construct the Heisenberg Hamiltonian
ham_J1 = OpSum()
for i in 1:L, j in 1:L, t in ["X", "Y", "Z"]
    if j < L
        ham_J1 .+= (t, (i, j), t, (i, j+1))
    end
    if i < L
        ham_J1 .+= (t, (i, j), t, (i+1, j))
    end
end

# Generate Operators for QNG
Oks_and_Eks = QuantumNaturalfPEPS.generate_Oks_and_Eks(peps, ham_J1)

# Setup the Integrator and Solver
integrator = QuantumNaturalGradient.Euler(lr=0.05)
solver = QuantumNaturalGradient.EigenSolver()

# Define a Parameters object to be evolved
θ = QuantumNaturalGradient.Parameters(peps)

# Evolve for a fixed (small) number of iterations as a demo
@time loss_value, trained_θ, misc = QuantumNaturalGradient.evolve(Oks_and_Eks, θ; 
        integrator, 
        verbosity=2,
        solver,
        sample_nr=1000,
        maxiter=10,)

```

Other features include calculation of Observables (Operator in the form of an OpSum) via sampling
```julia
observables, importance_weights = QuantumNaturalfPEPS.get_ExpectationValue(peps, Operator; it=10, multiproc=true)
```

# Citation
When citing our code, please use the following reference

Puente, D. A., Weerda, E. L., Schröder, K., & Rizzi, M. (2025). Efficient optimization and conceptual barriers in variational finite Projected Entangled-Pair States. arXiv preprint arXiv:2503.12557.
