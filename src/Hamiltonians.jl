function hamiltonain_J1J2(J1, J2, Lx, Ly; operators=["X", "Y", "Z"])
    ham_J1J2 = OpSum()
    for i in 1:Lx, j in 1:Ly, t in operators
        # J1
        if j != Ly
            ham_J1J2 += (J1, t, (i,j), t, (i,j+1))
        end
        if i != Lx
            ham_J1J2 += (J1, t, (i,j), t, (i+1,j))
        end

        # J2
        if i != Lx && j != Ly
            ham_J1J2 += (J2, t, (i,j), t, (i+1,j+1))
            ham_J1J2 += (J2, t, (i+1,j), t, (i,j+1))
        end
    end
    return ham_J1J2
end


# Define Helper Functions for P Operators
function P_matrix(factor)
    # Create an 8-index tensor and fill specific indices with the factor.
    P = zeros(ComplexF64, 2, 2, 2, 2, 2, 2, 2, 2)
    for i1 in 1:2, i2 in 1:2, i3 in 1:2, i4 in 1:2
        P[i1, i2, i2, i3, i3, i4, i4, i1] = factor
    end
    return P
end

function P_operator(hilbert, spins; P=nothing, factor=1)
    # Generate the ITensor operator for a given set of spins.
    if P === nothing
        P = P_matrix(factor)
    end
    inds = [hilbert[s]' for s in spins]  # use prime on physical indices
    append!(inds, [hilbert[s] for s in spins])
    return ITensor(P, inds)
end

function add_P_operators!(ham_op, hilbert, Lx, Ly, factor)
    sites = Int[]
    for i in 1:(Lx-1)
        for j in 1:(Ly-1)
            # Define the sites of the plaquette
            push!(sites, i   + (j-1)*Lx)
            push!(sites, i+1 + (j-1)*Lx)
            push!(sites, i+1 + (j)*Lx)
            push!(sites, i   + (j)*Lx)
            # Create and add the operator
            P_op = P_operator(hilbert, sites; factor=factor)
            push!(ham_op.tensors, P_op)
            push!(ham_op.sites, copy(sites))
            empty!(sites)
        end
    end
end

function hamiltonain_CSL(hilbert, J1, J2, lambda; kwargs...)
    ham_J1J2 = hamiltonain_J1J2(J1/4, J2/4, size(hilbert)...; kwargs...)

    # Create the tensor operator for the Hamiltonian
    tn_sum = QuantumNaturalGradient.TensorOperatorSum(ham_J1J2, hilbert)
    
    # Add P operators with both positive and negative imaginary factors
    add_P_operators!(tn_sum, hilbert, size(hilbert)..., im * lambda)
    add_P_operators!(tn_sum, hilbert, size(hilbert)..., -im * lambda)
    return tn_sum
end

function hamiltonian_hubbard(t, U, Lx, Ly)
    ham_hubbard = ITensors.OpSum()

    for i in 1:Lx, j in 1:Ly
        if j < Ly
            if t != 0
                ham_hubbard .+= (-t, "Cdag", (i, j),   "C", (i, j+1))
                ham_hubbard .+= (-t, "Cdag", (i, j+1), "C", (i, j))
            end
            # V (n_i - 1/2)(n_j - 1/2), constant dropped
            ham_hubbard .+= ( U, "N", (i, j), "N", (i, j+1))
            ham_hubbard .+= (-U/2, "N", (i, j))
            ham_hubbard .+= (-U/2, "N", (i, j+1))
        end

        if i < Lx
            if t != 0
                ham_hubbard .+= (-t, "Cdag", (i, j),   "C", (i+1, j))
                ham_hubbard .+= (-t, "Cdag", (i+1, j), "C", (i, j))
            end
            # V (n_i - 1/2)(n_j - 1/2), constant dropped
            ham_hubbard .+= ( U, "N", (i, j), "N", (i+1, j))
            ham_hubbard .+= (-U/2, "N", (i, j))
            ham_hubbard .+= (-U/2, "N", (i+1, j))
        end

    end
    
    return ham_hubbard
end

# total number operator
function build_Ntot_op(Lx::Int, Ly::Int)
    Ntot_op = ITensors.OpSum()

    for i in 1:Lx, j in 1:Ly
        Ntot_op .+= (1.0, "N", (i, j))
    end

    return Ntot_op
end

# CDW order parameter M^2 = (1/N^2) sum_{a,b} (-1)^(x_a+y_a+x_b+y_b) n_a n_b
function build_M_cdw2_op(Lx::Int, Ly::Int)
    M2_op = ITensors.OpSum()

    sites = Tuple{Int,Int,Float64}[]

    for i in 1:Lx, j in 1:Ly
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

# average nearest-neighbor density-density correlation (1/Nb) sum_{<a,b>} n_a n_b
function build_nn_dd_corr_op(Lx::Int, Ly::Int)
    nn_op = ITensors.OpSum()
    Nb = Lx * (Ly - 1) + Ly * (Lx - 1)
    coeff = 1.0 / Nb

    for i in 1:Lx, j in 1:Ly
        if j < Ly
            nn_op .+= (coeff, "N", (i, j), "N", (i, j+1))
        end

        if i < Lx
            nn_op .+= (coeff, "N", (i, j), "N", (i+1, j))
        end
    end

    return nn_op
end