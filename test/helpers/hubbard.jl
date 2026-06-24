using ITensors
# using SparseArrays
# using KrylovKit


col_major_site(i::Int, j::Int, Lx::Int) = i + (j - 1) * Lx

# functions to compute observables
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

### functions for ED
# function nn_bonds_linear(Lx::Int, Ly::Int)
#     bonds = Tuple{Int,Int}[]

#     for i in 1:Lx, j in 1:Ly
#         a = col_major_site(i, j, Lx)

#         if j < Ly
#             push!(bonds, (a, col_major_site(i, j+1, Lx)))
#         end

#         if i < Lx
#             push!(bonds, (a, col_major_site(i+1, j, Lx)))
#         end
#     end

#     return bonds
# end

# function fixed_N_basis(N::Int, Nf::Int)
#     basis = UInt64[]

#     for s in UInt64(0):(UInt64(1) << N) - UInt64(1)
#         if count_ones(s) == Nf
#             push!(basis, s)
#         end
#     end

#     return basis
# end

# function bitcount_between(state::UInt64, a::Int, b::Int)
#     lo, hi = min(a, b), max(a, b)

#     n_between = 0
#     for k in lo+1:hi-1
#         n_between += Int((state >> (k - 1)) & UInt64(1))
#     end

#     return n_between
# end

# function apply_cdag_c(state::UInt64, to::Int, from::Int)
#     occ_to = (state >> (to - 1)) & UInt64(1)
#     occ_from = (state >> (from - 1)) & UInt64(1)

#     if occ_from == 0 || occ_to == 1
#         return nothing
#     end

#     new_state = state
#     new_state &= ~(UInt64(1) << (from - 1))
#     new_state |=  (UInt64(1) << (to - 1))

#     n_between = bitcount_between(state, to, from)
#     sign = iseven(n_between) ? 1.0 : -1.0

#     return new_state, sign
# end

# function coordination_numbers_linear(Lx::Int, Ly::Int)
#     N = Lx*Ly
#     z = zeros(Int, N)

#     for i in 1:Lx, j in 1:Ly
#         a = col_major_site(i, j, Lx)

#         z[a] += (j > 1)
#         z[a] += (j < Ly)
#         z[a] += (i > 1)
#         z[a] += (i < Lx)
#     end

#     return z
# end

# function build_spinless_tV_ED(
#     Lx::Int,
#     Ly::Int,
#     t::Real,
#     V::Real;
#     Nf::Int = (Lx*Ly) ÷ 2,
#     shifted::Bool = false,
# )
#     N = Lx*Ly
#     bonds = nn_bonds_linear(Lx,Ly)
#     z = coordination_numbers_linear(Lx, Ly)

#     basis = fixed_N_basis(N, Nf)
#     dim = length(basis)

#     state_to_idx = Dict{UInt64, Int}()
#     for (idx, s) in enumerate(basis)
#         state_to_idx[s] = idx
#     end

#     rows = Int[]
#     cols = Int[]
#     vals = Float64[]

#     for (col, s) in enumerate(basis)
#         # diagonal part
#         Ediag = 0.0

#         for (a, b) in bonds
#             na = Int((s >> (a - 1)) & UInt64(1))
#             nb = Int((s >> (b - 1)) & UInt64(1))
#             Ediag += V * na * nb
#         end

#         if shifted
#             zN = 0.0
#             for a in 1:N
#                 na = Int((s >> (a - 1)) & UInt64(1))
#                 zN += z[a] * na
#             end
#             Ediag -= (V / 2) * zN
#         end

#         push!(rows, col)
#         push!(cols, col)
#         push!(vals, Ediag)

#         # off-diagonal hopping
#         for (a, b) in bonds
#             res_ab = apply_cdag_c(s, a, b)
#             if res_ab !== nothing
#                 snew, sign = res_ab
#                 row = state_to_idx[snew]

#                 push!(rows, row)
#                 push!(cols, col)
#                 push!(vals, -t * sign)
#             end

#             res_ba = apply_cdag_c(s, b, a)
#             if res_ba !== nothing
#                 snew, sign = res_ba
#                 row = state_to_idx[snew]

#                 push!(rows, row)
#                 push!(cols, col)
#                 push!(vals, -t * sign)
#             end
#         end
#     end

#     H = sparse(rows, cols, vals, dim, dim)

#     return H, basis
# end

# function ground_energy_spinless_tV_ED(
#     Lx::Int,
#     Ly::Int,
#     t::Real,
#     V::Real;
#     Nf::Int = (Lx * Ly) ÷ 2,
#     shifted::Bool = false,
# )
#     H, basis = build_spinless_tV_ED(Lx, Ly, t, V; Nf=Nf, shifted=shifted)

#     vals, vecs, info = eigsolve(H, 1, :SR; ishermitian=true)
#     ψ0 = ComplexF64.(vecs[1])
#     ψ0 ./= norm(ψ0)

#     return real(vals[1]), ψ0, H, basis
# end

# function diagonal_observables_ED(ψ, basis, Lx::Int, Ly::Int)
#     N = Lx*Ly
#     bonds = nn_bonds_linear(Lx,Ly)
#     Nb = length(bonds)

#     probs = abs2.(ψ)
#     probs ./= sum(probs)

#     Ntot_mean = 0.0
#     Ntot2_mean = 0.0

#     M_mean = 0.0
#     absM_mean = 0.0
#     M2_mean = 0.0

#     nn_avg_mean = 0.0

#     for (p, s) in zip(probs, basis)
#         Ntot = 0
#         M = 0.0

#         for i in 1:Lx, j in 1:Ly
#             a = col_major_site(i, j, Lx)
#             n = Int((s >> (a - 1)) & UInt64(1))

#             Ntot += n
#             M += (-1)^(i + j) * n
#         end

#         nn_count = 0.0
#         for (a, b) in bonds
#             na = Int((s >> (a - 1)) & UInt64(1))
#             nb = Int((s >> (b - 1)) & UInt64(1))
#             nn_count += na * nb
#         end

#         nn_avg = nn_count / Nb

#         Ntot_mean += p * Ntot
#         Ntot2_mean += p * Ntot^2

#         M_mean += p * M
#         absM_mean += p * abs(M)
#         M2_mean += p * M^2

#         nn_avg_mean += p * nn_avg
#     end

#     Ntot_var = Ntot2_mean - Ntot_mean^2

#     return (
#         Ntot = Ntot_mean,
#         Ntot_var = Ntot_var,

#         M_cdw = M_mean,
#         abs_M_cdw = absM_mean,
#         M_cdw2 = M2_mean,

#         m_cdw = M_mean / N,
#         abs_m_cdw = absM_mean / N,
#         m_cdw2 = M2_mean / N^2,

#         S_pi_pi = M2_mean / N,

#         nn_avg = nn_avg_mean,
#     )
# end