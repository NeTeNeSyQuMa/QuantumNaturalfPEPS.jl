# - For J2/J1=0.125, H=0, 18x18 lattice with PBC, compare energies with VMC data from [Budaraju et al. ('26)​] 
# At m=0, the monopole state reduces to a Dirac state with half-filled staggered-[0,\pi] background flux

# H/J1        CS         Mono         Umb         Y
# 0.        -0.493694    -0.501894    -0.5019        -0.499973 
using Revise
import Pkg
Pkg.activate("/home/psireal42/Work/phd-projects/qnfp_env"; shared=false)
using QuantumNaturalGradient
using QuantumNaturalfPEPS
using ITensors
using LinearAlgebra

const QNG = QuantumNaturalGradient


J1 = 1.0
J2 = J1 / 8
B_field = 0.0
Lx, Ly = 6, 6
N = Lx*Ly

hilbert = ITensors.siteinds("S=1/2", Lx, Ly)
peps = PEPS(ComplexF64, hilbert; bond_dim=1, show_warning=false)
physical_hamiltonians = hamiltonian_J1J2_H(
        Lx,
        Ly;
        J1,
        J2,
        H=B_field,
        boundary=:periodic,
)

### mean-field ansätze
# monopole ansatz - no free parameters => no optimization
# Haux = Matrix(hamiltonian_aux_triangular_torus(Lx, Ly; hopping))
# spectrum = eigen(Hermitian(Haux[1:2:end, 1:2:end])) # diagonalization in 1 spin sector
# Nspin = N ÷ 2
# gap = spectrum.values[Nspin + 1] - spectrum.values[Nspin]
# Vspin = spectrum.vectors[:, 1:Nspin]

monopole_gs = monopole_state(
    Lx,
    Ly,
    0;
    particle_number=N,
)
monopole_projected_state = gutzwiller_project(monopole_gs)

# Y state: η_Y = [h1, h2, h3, Δ]
η_Y = [0.35, 0.25, 0.15, 0.8]
Y_gs = y_state(
    Lx,
    Ly;
    η=η_Y,
    particle_number=N,
)
Y_projected_state = gutzwiller_project(Y_gs)


# Canted-stripe state: η_CS = [h]. The paper optimizes h only;
# Delta/t in Fig. 2(b) is retained as the fixed stripe_delta input.
η_CS = [0.3]
stripe_delta = 0.8
CS_gs = cs_state(
    Lx,
    Ly;
    η=η_CS,
    stripe_delta,
    particle_number=N,
)
CS_projected_state = gutzwiller_project(CS_gs)


# Umbrella state: eta_umbrella = [h]
η_umbrella = [0.3]
umbrella_hopping, umbrella_fields = umbrella_hopping_fields(
    Lx,
    Ly,
    η_umbrella,
)
umbrella_gs = umbrella_state(
    Lx,
    Ly;
    η=η_umbrella,
    particle_number=N,
)
umbrella_projected_state = gutzwiller_project(umbrella_gs)


# Select which ansatz to optimize. Y, canted-stripe, and umbrella contain
# transverse fields and therefore need not have a fixed total Sz. The monopole
# state is spin-conserving and remains in a fixed sector intrinsically.
projected_state = Y_projected_state
Oks_and_Eks = QuantumNaturalfPEPS.generate_Oks_and_Eks(
    peps,
    physical_hamiltonians;
    trial_state=projected_state,
)

integrator = QuantumNaturalGradient.Euler(lr=0.2)
θ = vcat(vec(peps), QuantumNaturalGradient.Parameters(projected_state))
loss_value, trained_θ, misc = QuantumNaturalGradient.evolve(Oks_and_Eks, θ;
                integrator,
                verbosity=true,
                sample_nr=5,
                maxiter=5,
        )
