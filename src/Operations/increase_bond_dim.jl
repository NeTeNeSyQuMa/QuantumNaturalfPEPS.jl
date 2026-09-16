function increase_bond_dim(o1::ITensor, l, ln; value=0)
    ol = noncommoninds(o1, l)
    perm = NDTensors.getperm((ol..., l), inds(o1))
    o1t = permutedims(o1.tensor, perm)
    o1t = reshape(o1t.storage, :, dim(l))

    o1n = randomITensor(eltype(o1), ol..., ln) .* (std(o1.tensor) * value)
    o1nt = reshape(o1n.tensor.storage, :, dim(ln))
    o1nt[:, 1:dim(l)] .= o1t
    return o1n
end

function increase_bond_dim(o1::ITensor, o2::ITensor, new_bond_dim::Int; value=0, svalue=0)
    l = commonind(o1, o2)
    @assert dim(l) <= new_bond_dim
    ln = Index(new_bond_dim; tags=tags(l))
    ElT = eltype(o1)
    
    local U
    if value == 0 && svalue == 0
        t = NDTensors.random_unitary(ElT, dim(l), dim(ln))
        U = ITensor(t, l, ln)
        # Use conjugate isometries on the two sides of the bond so that
        # contracting the enlarged bond inserts U * U† = I. 
        return o1 * U, o2 * dag(U)
    end
    if value == 0
        value = 1.
    end
    o1n = increase_bond_dim(o1, l, ln; value=value)
    o2n = increase_bond_dim(o2, l, ln; value=svalue/value)

    #@show norm(o1n*o2n - o1*o2)
    return o1n, o2n
end

"""
increase_bond_dim!(peps::AbstractPEPS, new_bond_dim::Int; value=0)

Increase the bond dimension of a PEPS by increasing the bond dimension of each tensor in the PEPS without changing the contracted peps.
value = 0: will lead to an unchanged spectrum of the PEPS
value != 0: will lead to a modified spectrum of the PEPS, the larger the value, the larger the change in the spectrum and the less contractable the PEPS will be.
"""
function increase_bond_dim!(peps::AbstractPEPS, new_bond_dim::Int; value=0, svalue=0)

    for i in 1:size(peps, 1), j in 1:size(peps, 2)
        if i < size(peps, 1)
            peps[i, j], peps[i+1, j] = increase_bond_dim(peps[i, j], peps[i+1,j], new_bond_dim; value, svalue)
        end
        if j < size(peps, 2)
            peps[i, j], peps[i, j+1] = increase_bond_dim(peps[i, j], peps[i,j+1], new_bond_dim; value, svalue)
        end
    end
    peps.bond_dim = new_bond_dim
    return peps
end

increase_bond_dim(peps::AbstractPEPS, new_bond_dim::Int; kwargs...) = increase_bond_dim!(deepcopy(peps), new_bond_dim; kwargs...)
