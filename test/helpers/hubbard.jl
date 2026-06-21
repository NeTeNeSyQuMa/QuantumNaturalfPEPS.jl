using ITensors

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