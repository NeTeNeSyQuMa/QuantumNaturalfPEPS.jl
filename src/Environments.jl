# the entries of the environments are all normalized by the absolute value of the biggest entrie of the first ITensor
# to get the true environtment: contract with the MPS and afterwards multiply by exp(f)
mutable struct Environment
    env::MPS
    f::Real
    function Environment(env::MPS, f::Real; normalize=true)
        env = new(env, f)
        if normalize
            normalize!(env)
        end
        return env
    end
    Environment(env; kwargs...) = Environment(env, 0.; kwargs...)
end
convert_eltype(::Type{S}, t::ITensor) where S = itensor(S.(t.tensor))
Environment(::Type{S}, env::Environment) where S = Environment(convert_eltype.(S, env.env), S(env.f); normalize=false)


Base.getindex(env::Environment, i::Int) = env.env[i]
Base.reverse(env::Environment) = ReverseEnvironment(env)
ITensorMPS.maxlinkdim(env::Environment) = maxlinkdim(env.env)
ITensorMPS.maxlinkdim(envs::Vector{Environment}) = maximum(maxlinkdim.(envs))

struct ReverseEnvironment
    env::Environment
end
Base.getindex(env::ReverseEnvironment, i::Int) = reverse(env).env[end-i+1]
Base.getproperty(x::ReverseEnvironment, y::Symbol) = getproperty(reverse(x), y)
Base.reverse(env::ReverseEnvironment) = getfield(env, :env)

function normalize!(env::Environment)
    #If it is orthogonal, normalizing the MPS is cheap
    if env.env.llim == 0 && env.env.rlim == 2
        norm_ = norm(env.env[1])
        env.env[1] ./= norm_
        env.f += log(norm_)
    else
        for env_i in env.env
            norm_ = max_norm(env_i)
            env_i ./= norm_
            env.f += log(norm_)
        end
        env.env.llim = 0
        env.env.rlim = length(env.env) + 1
    end
    return env
end

# Computes the environments and log(<ψ|S>)
function get_logψ_and_envs(peps::AbstractPEPS, S::Array{Int64,2}, env_top=Array{Environment}(undef, size(S,1)-1);
                           alg="densitymatrix", overwrite=nothing, kwargs...)
    
    Lx = size(peps, 1)
    if overwrite === nothing
        # if env_top is given and the bond dimension is sufficient, we do not need to calculate it again
        overwrite = !(isassigned(env_top, Lx-1) && maxlinkdim(env_top[Lx-1].env) >= peps.contract_dim)   
    end
    
    env_down = Array{Environment}(undef, size(peps, 1) - 1)

    peps_projected = get_projected(peps, S)
    
    if overwrite
        env_top[1] = generate_env_row(peps_projected[1, :], peps.contract_dim; alg, cutoff=peps.contract_cutoff)
    end
    env_down[1] = generate_env_row(peps_projected[Lx, :], peps.contract_dim; alg, cutoff=peps.contract_cutoff)
    
    # for every row we calculate the environments once from the top down and once from the bottom up
    for i in 2:Lx-1
        i_prime = Lx+1-i 
        if overwrite
            env_top[i] = generate_env_row(peps_projected[i, :], peps.contract_dim; env_row_above=env_top[i-1], alg, cutoff=peps.contract_cutoff)
        end
        env_down[i] = generate_env_row(peps_projected[i_prime, :], peps.contract_dim; env_row_above=env_down[i-1], alg, cutoff=peps.contract_cutoff)
    end

    # Check if maximal bond dimension is reached
    max_bond = max(maxlinkdim(env_top), maxlinkdim(env_down))
    if max_bond == peps.contract_dim && peps.show_warning
        @warn "horizontal environments at maximal bond dimension $max_bond"
    end
    
    # once we calculated all environments we calculate <ψ|S> using the environments
    return get_logψ(env_top, env_down; kwargs...), env_top, env_down, max_bond
end

# calculates the environments for a given row and contracts that with env_row_above
function generate_env_row(peps_projected, contract_dim; env_row_above=nothing, alg="densitymatrix", cutoff=1e-13)
    norm_shift = 0
    if env_row_above === nothing
        peps_projected = MPS(peps_projected)
    else
        peps_projected = contract(MPO(peps_projected), env_row_above.env; maxdim=contract_dim, alg, cutoff)
        norm_shift = env_row_above.f
    end

    return Environment(peps_projected, norm_shift; normalize=true)
end

function get_logψ(env_top::Vector{Environment}, env_down::Vector{Environment}; pos=max(length(env_top)÷2, 1)) # max: on a two-row lattice length(env_top)÷2 would be 0
    #ψS = contract(env_top[pos].env.*env_down[end-pos+1].env)[1]
    #logψS = log(Complex(ψS))
    logψS = _log_or_not_dot(env_top[pos].env, env_down[end-pos+1].env, true; dag=false)
    return logψS + env_top[pos].f + env_down[end-pos+1].f
end

function logψ_exact_D1(peps::AbstractPEPS, sample)
    peps.bond_dim == 1 || throw(ArgumentError(
        "logψ_exact_D1 requires bond dimension one",
    ))
    size(sample) == size(peps) || throw(DimensionMismatch(
        "sample size $(size(sample)) does not match PEPS size $(size(peps))",
    ))

    # Every virtual index has dimension one, so the projected PEPS contraction
    # is just a product of one selected tensor entry per site. Accumulating the
    # logarithms avoids both the high-order intermediates created by the generic
    # exact contraction and underflow on large lattices.
    log_amplitude = 0.0 + 0.0im
    for column in 1:size(peps, 2), row in 1:size(peps, 1)
        tensor = peps[row, column]
        physical_index = siteind(peps, row, column)
        coordinates = map(inds(tensor)) do index
            index => (index == physical_index ? sample[row, column] + 1 : 1)
        end
        local_amplitude = tensor[coordinates...]
        iszero(local_amplitude) && return ComplexF64(-Inf, 0.0)
        log_amplitude += log(complex(local_amplitude))
    end
    return log_amplitude
end

function logψ_exact(peps, sample)
    peps.bond_dim == 1 && return logψ_exact_D1(peps, sample)
    proj = get_projected(peps, sample)
    con = contract_peps_exact(proj)
    return log(Complex(con))
end

function get_all_horizontal_envs(peps::AbstractPEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64},
                                 all_horizontal_envs_r::Array{ITensor}=Array{ITensor}(undef, size(peps, 1), size(peps, 2)-1),
                                 all_horizontal_envs_l::Array{ITensor}=Array{ITensor}(undef, size(peps, 1), size(peps, 2)-1))
    for i in 1:size(peps, 1)
        view_r = @view all_horizontal_envs_r[i, :]
        view_l = @view all_horizontal_envs_l[i, :]
        get_horizontal_envs!(peps, env_top, env_down, S, i, view_r, view_l)
    end
    return all_horizontal_envs_r, all_horizontal_envs_l
end

function get_horizontal_envs(peps::AbstractPEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, i::Int64, horizontal_envs_r=Matrix{ITensor}(undef, size(peps, 2)-1), horizontal_envs_l=Matrix{ITensor}(undef, size(peps, 2)-1))
    get_horizontal_envs!(peps, env_top, env_down, S,i ,horizontal_envs_r, horizontal_envs_l)
    return horizontal_envs_r, horizontal_envs_l
end

# this function computes the horizontal environments for a given row
function get_horizontal_envs!(peps::AbstractPEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, i::Int64, horizontal_envs_r, horizontal_envs_l)
    peps_i = get_projected(peps, S, i, :)    #contract the row with S
    
    # now we loop through every site and compute the environments (once from the right and once from the left) by MPO-MPS contraction.
    if i == 1
        contract_recursiv!(horizontal_envs_r, peps_i, env_down[end].env)
        contract_recursiv!(horizontal_envs_l, peps_i, env_down[end].env, right_to_left=false)
    elseif i == size(peps, 1)
        contract_recursiv!(horizontal_envs_r, peps_i, env_top[end].env)
        contract_recursiv!(horizontal_envs_l, peps_i, env_top[end].env, right_to_left=false)
    else
        contract_recursiv!(horizontal_envs_r, env_top[i-1].env, peps_i; c=env_down[end-i+1].env)
        contract_recursiv!(horizontal_envs_l, env_top[i-1].env, peps_i; c=env_down[end-i+1].env, right_to_left=false)
    end
end

function contract_recursiv!(h_envs, a, b; c=ones(eltype(a[1]), length(a)), d=ones(eltype(a[1]), length(a)), right_to_left=true)
    if right_to_left
        h_envs[end] = a[end]*b[end]*c[end]*d[end]
        for j in length(a)-1:-1:2
            h_envs[j-1] = h_envs[j]*a[j]*b[j]*c[j]*d[j]
        end
    else
        h_envs[1] = a[1]*b[1]*c[1]*d[1]
        for j in 2:length(a)-1
            h_envs[j] = h_envs[j-1]*a[j]*b[j]*c[j]*d[j]
        end
    end
end

function _contract_strip_column(layers, column)
    contracted = layers[1][column]
    for layer in @view layers[2:end]
        contracted = contracted * layer[column]
    end
    return contracted
end

function _contract_strip_recursiv!(horizontal_envs, layers; right_to_left=true)
    number_of_columns = length(layers[1])
    all(length(layer) == number_of_columns for layer in layers) || throw(
        DimensionMismatch("all strip layers must have the same number of columns"),
    )
    length(horizontal_envs) == number_of_columns - 1 || throw(
        DimensionMismatch(
            "a $number_of_columns-column strip requires " *
            "$(number_of_columns - 1) horizontal environments",
        ),
    )

    if right_to_left
        horizontal_envs[end] = _contract_strip_column(layers, number_of_columns)
        for column in number_of_columns-1:-1:2
            horizontal_envs[column-1] =
                horizontal_envs[column] * _contract_strip_column(layers, column)
        end
    else
        horizontal_envs[1] = _contract_strip_column(layers, 1)
        for column in 2:number_of_columns-1
            horizontal_envs[column] =
                horizontal_envs[column-1] * _contract_strip_column(layers, column)
        end
    end
    return horizontal_envs
end

"""
    get_strip_envs(peps, env_top, env_down, sample, first_row, row_count)

Construct left and right contraction environments for a contiguous strip of
projected PEPS rows. Rows outside the strip are represented by the precomputed
top and bottom boundary MPS environments. The resulting environments permit
local flipped-amplitude contractions spanning more than two rows without
contracting the complete PEPS.
"""
function get_strip_envs(
    peps::AbstractPEPS,
    env_top::Vector{Environment},
    env_down::Vector{Environment},
    sample::Matrix{Int64},
    first_row::Integer,
    row_count::Integer,
)
    number_of_rows, number_of_columns = size(peps)
    row_count > 0 || throw(ArgumentError("row_count must be positive"))
    last_row = first_row + row_count - 1
    1 <= first_row <= last_row <= number_of_rows || throw(BoundsError(
        peps,
        (first_row:last_row, :),
    ))
    number_of_columns >= 2 || throw(ArgumentError(
        "strip environments require at least two PEPS columns",
    ))

    layers = Any[]
    if first_row > 1
        push!(layers, env_top[first_row-1].env)
    end
    for row in first_row:last_row
        push!(layers, get_projected(peps, sample, row, :))
    end
    if last_row < number_of_rows
        push!(layers, env_down[number_of_rows-last_row].env)
    end

    right = Vector{ITensor}(undef, number_of_columns - 1)
    left = similar(right)
    _contract_strip_recursiv!(right, layers)
    _contract_strip_recursiv!(left, layers; right_to_left=false)
    return right, left
end

function get_all_4b_envs(peps::AbstractPEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, all_4b_envs_r::Array{ITensor}=Array{ITensor}(undef, size(peps, 1)-1, size(peps, 2)-1), all_4b_envs_l::Array{ITensor}=Array{ITensor}(undef, size(peps, 1)-1, size(peps, 2)-1))
    for i in 1:size(peps, 1)-1
        view_r = @view all_4b_envs_r[i, :]
        view_l = @view all_4b_envs_l[i, :]
        get_4b_envs!(peps, env_top, env_down, S, i, view_r, view_l)
    end
    return all_4b_envs_r, all_4b_envs_l
end

function get_4b_envs(peps::AbstractPEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, i::Int64, fourb_envs_r=Matrix{ITensor}(undef, size(peps, 2)-1), fourb_envs_l=Matrix{ITensor}(undef, size(peps, 2)-1))
    get_4b_envs!(peps, env_top, env_down,S,i,fourb_envs_r, fourb_envs_l)
    return fourb_envs
end

function get_4b_envs!(peps::AbstractPEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, i::Int64, fourb_envs_r, fourb_envs_l)
    peps_i = get_projected(peps, S, i, :)  
    peps_j = get_projected(peps, S, i+1, :)

    if size(peps, 1) == 2
        # two-row lattice: the row pair covers the whole lattice, no environment above or below
        contract_recursiv!(fourb_envs_r, peps_i, peps_j)
        contract_recursiv!(fourb_envs_l, peps_i, peps_j, right_to_left=false)
    elseif i == 1
        contract_recursiv!(fourb_envs_r, peps_i, peps_j, c=env_down[end-1].env)
        contract_recursiv!(fourb_envs_l, peps_i, peps_j, c=env_down[end-1].env, right_to_left=false)
    elseif i == size(peps, 1)-1
        contract_recursiv!(fourb_envs_r, env_top[end-1].env, peps_i, c=peps_j)
        contract_recursiv!(fourb_envs_l, env_top[end-1].env, peps_i, c=peps_j, right_to_left=false)
    else
        contract_recursiv!(fourb_envs_r, env_top[i-1].env, peps_i, c=peps_j, d=env_down[end-i].env)
        contract_recursiv!(fourb_envs_l, env_top[i-1].env, peps_i, c=peps_j, d=env_down[end-i].env, right_to_left=false)
    end
end
