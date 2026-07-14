
# function that sorts the dictionary of flipped spin Ek_terms into the categories horizontal,vertical and 4body Ek_terms. Also sorts so that the row-number gets bigger
# vertical=false sorts all vertical Ek_terms into the 4body array
function sort_dict(Ek_terms; vertical=true)
    hor = Vector{Any}()
    vert = Vector{Any}()
    four = Vector{Any}()
    longerHor = Vector{Any}()
    other = Vector{Any}()
    # loop through every term
    for flip_term in keys(Ek_terms)
        if flip_term == () # if no spin flip
            # do nothing
        else 
            xs = []
            ys = []
            for flip_term_i in flip_term
                (x, y), Sij = flip_term_i
                push!(xs, x)
                push!(ys, y)
            end
            if length(xs) == 1
                insert(hor, flip_term)
            elseif maximum(xs)-minimum(xs) == 0 && maximum(ys)-minimum(ys) <= 1
                insert(hor, flip_term)
            elseif maximum(xs)-minimum(xs) <= 1 && maximum(ys)-minimum(ys) == 0 && vertical
                insert(vert, flip_term)
            elseif maximum(xs)-minimum(xs) <= 1 && maximum(ys)-minimum(ys) <=1
                insert(four, flip_term)
            elseif maximum(xs)-minimum(xs) == 0
                insert(longerHor, flip_term)
            else
                insert(other, flip_term)
            end
        end
    end
    return hor, vert, four, longerHor, other
end
   
# inserts the element x into the array arr at the desired position
function insert(arr, x)
    if isempty(arr)
        push!(arr,x)
        return
    end
    x_min = x[1][1][1]
    if length(x) == 2
        xmin = minimum([x_min, x[2][1][1]])
    end
    
    for i in 1:length(arr)
        if arr[i][1][1][1] >= x_min
            insert!(arr,i,x)
            return
        end
    end
    push!(arr,x)
end

# a function that computes the contraction of the PEPS with flipped spins at a position specified in flip_term
function get_4body_term(peps::AbstractPEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, flip_term, fourb_envs_r, fourb_envs_l)
    con_left = 1
    con_right = 1
    f = 0
    
    xs = []
    ys = []
    Sijs = Dict{Any, Number}()
    for flip_term_i in flip_term
        (x, y), Sij = flip_term_i
        push!(xs, x)
        push!(ys, y)
        Sijs[x,y] = Sij
    end
    
    maxx = maximum(xs)
    minx = minimum(xs)
    maxy = maximum(ys)
    miny = minimum(ys)

    @assert maxx-minx <= 1 "Only adjacent rows/coloumns allowed in 4body_term"
    @assert maxy-miny <= 1 "Only adjacent rows/coloumns allowed in 4body_term"

    unflipped_spins = setdiff([(xi, yi) for xi in unique(xs), yi in unique(ys)], zip(xs, ys))
    for us in unflipped_spins
        x,y = us
        push!(xs, x)
        push!(ys, y)
        Sijs[x,y] = S[x,y]
    end

    if miny != 1
        con_left = con_left*fourb_envs_l[miny-1]
    end
    if maxy != size(peps, 2)
        con_right = con_right*fourb_envs_r[maxy]
    end

    if minx != 1
        con_left = con_left*env_top[minx-1].env[miny]
        if miny != maxy
            con_right = con_right*env_top[minx-1].env[maxy]
        end
        f += env_top[minx-1].f
    end

    pairs = unique([(x, y) for x in (minx, maxx) for y in (miny, maxy)])
    for p in pairs
        if p[2] == miny
            con_left = con_left*get_projected(peps, Sijs[p], p[1], p[2])
        else
            con_right = con_right*get_projected(peps, Sijs[p], p[1], p[2])
        end
    end

    if maxx != size(peps, 1)
        con_left = con_left*env_down[end-maxx+1].env[miny]
        if miny != maxy
            con_right = con_right*env_down[end-maxx+1].env[maxy]
        end
        f += env_down[end-maxx+1].f
    end

    c = (con_right*con_left)[]
    if c isa Real && c < 0
        c = complex(c)
    end
    logψ_flipped = log(c) + f
    
    return logψ_flipped
end

# same as get_4body_term but for horizontal Ek_terms
function get_term(peps::AbstractPEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, flip_term, h_envs_r, h_envs_l)
    f = 0
    @assert length(flip_term) <= 2 " Only nearest and next nearest neighbour interactions are efficiently supported. Note that if the opertor is in he computational basis, any interaction length is possible."
    ys = Int[]
    Sijs = Int[]
    x = flip_term[1][1][1] # x cordinate of the first spin that was flipped
    for flip_term_i in flip_term
        (x_, y), Sij = flip_term_i
        @assert x == x_ "Only one x coordinate is allowed in a horizontal term"
        push!(ys, y)
        push!(Sijs, Sij)
    end

    # Sort the y values
    if length(ys) == 2 && ys[1] > ys[2]
        ys = reverse(ys)
        Sijs = reverse(Sijs)
    end
    
    flip = get_projected(peps, Sijs[1], x, ys[1])
    if ys[1] != 1
        flip = flip*h_envs_l[ys[1]-1]
    end
    
    if x != size(peps, 1)
        flip = flip*env_down[end-x+1].env[ys[1]]
        f += env_down[end - x + 1].f
    end
    if x != 1
        flip = env_top[x - 1].env[ys[1]]*flip
        f += env_top[x - 1].f
    end
    
    if length(flip_term) == 2
        flip = flip * get_projected(peps, Sijs[2], x, ys[2])
        if x != size(peps, 1)
            flip = flip * env_down[end - x + 1].env[ys[2]]
        end
        if x != 1
            flip = env_top[x-1].env[ys[2]]*flip
        end
    end
    maxy = maximum(ys)
    if maxy != size(peps, 2)
        flip = flip * h_envs_r[maxy]
    end
    c = contract(flip)[]
    if c isa Real && c < 0
        c = complex(c)
    end
    logψ_flipped = log(c) + f
       
    return logψ_flipped
end

# same as get_term but for longer horizontal Ek_terms
function get_longerHor_term(peps::AbstractPEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, flip_term, h_envs_r, h_envs_l)
    f = 0
    ys = Int[]
    Sijs = Int[]
    x = flip_term[1][1][1] # x cordinate of the first spin that was flipped
    for flip_term_i in flip_term
        (x_, y), Sij = flip_term_i
        @assert x == x_ "Only one x coordinate is allowed in a horizontal term"
        push!(ys, y)
        push!(Sijs, Sij)
    end

    # Sort the y values
    perm = sortperm(ys)
    ys = ys[perm]
    Sijs = Sijs[perm]
    
    flip = get_projected(peps, Sijs[1], x, ys[1])
    if ys[1] != 1
        flip = flip*h_envs_l[ys[1]-1]
    end
    
    if x != size(peps, 1)
        flip = flip*env_down[end-x+1].env[ys[1]]
        f += env_down[end - x + 1].f
    end
    if x != 1
        flip = env_top[x - 1].env[ys[1]]*flip
        f += env_top[x - 1].f
    end
    
    for i in ys[1]+1:ys[end]
        if i in ys
            flip = flip * get_projected(peps, Sijs[findall(x->x==i, ys)[1]], x, i)
        else
            flip = flip * get_projected(peps, S[x,i], x, i)
        end
        if x != size(peps, 1)
            flip = flip * env_down[end - x + 1].env[i]
        end
        if x != 1
            flip = env_top[x-1].env[i]*flip
        end
    end

    maxy = maximum(ys)
    if maxy != size(peps, 2)
        flip = flip * h_envs_r[maxy]
    end
    c = contract(flip)[]
    if c isa Real && c < 0
        c = complex(c)
    end
    logψ_flipped = log(c) + f
       
    return logψ_flipped
end

# computes the local energy <sample|H|ψ>/<sample|ψ>
function get_logψ_flipped(peps::AbstractPEPS, Ek_terms, env_top::Vector{Environment}, env_down::Vector{Environment},
                          sample::Matrix{Int64}, logψ::Number;
                          h_envs_r=nothing, h_envs_l=nothing, fourb_envs_r=nothing, fourb_envs_l=nothing, logψ_flipped=nothing,
                          timer=TimerOutput())
    
    if logψ_flipped === nothing
        logψ_flipped = Dict{Any, Number}()
    end
      
    # deals with the term with no flipped spin
    if haskey(Ek_terms, ())
        logψ_flipped[()] = logψ
    end
    
    # sorts the dictionary into the different categories
    horizontal, vertical, fourBody, longerHor, other = sort_dict(Ek_terms, vertical=false)

    # loop through every horizontal components
    @timeit timer "horizontal" for flip_term in horizontal 
        # calculate the Energy contribution of the specific term and add it to the total Ek
        if h_envs_r === nothing || h_envs_l === nothing
            @timeit timer "horizontal_envs" h_envs_r, h_envs_l = get_all_horizontal_envs(peps, env_top, env_down, sample)
        end
        if !haskey(logψ_flipped, flip_term)
            logψ_flipped[flip_term] = get_term(peps, env_top, env_down, sample, flip_term, h_envs_r[flip_term[1][1][1], :], h_envs_l[flip_term[1][1][1], :])
        end
    end
    
    # same for non-horizontal Ek_terms
    if !isempty(fourBody)
        if fourb_envs_r === nothing || fourb_envs_l === nothing 
            @timeit timer "fourbody_envs" fourb_envs_r, fourb_envs_l = get_all_4b_envs(peps, env_top, env_down, sample)
        end
        @timeit timer "fourbody" for flip_term in fourBody
            if !haskey(logψ_flipped, flip_term)
                row_values = map(t -> t[1][1], flip_term)
                upperrow = minimum(row_values)
                logψ_flipped[flip_term] = get_4body_term(peps, env_top, env_down, sample, flip_term, fourb_envs_r[upperrow, :], fourb_envs_l[upperrow, :])
            end
        end
    end

    if !isempty(longerHor)
        if h_envs_r === nothing || h_envs_l === nothing
            @timeit timer "horizontal_envs" h_envs_r, h_envs_l = get_all_horizontal_envs(peps, env_top, env_down, sample)
        end
        for flip_term in longerHor 
            if !haskey(logψ_flipped, flip_term)
                logψ_flipped[flip_term] = get_longerHor_term(peps, env_top, env_down, sample, flip_term, h_envs_r[flip_term[1][1][1], :], h_envs_l[flip_term[1][1][1], :])
            end
        end
    end

    if !isempty(other)
        @warn "Only nearest and next nearest neighbour interactions are efficiently supported. Note that if the opertor is in the computational basis, any interaction length is possible."
        for flip_term in other
            if !haskey(logψ_flipped, flip_term)
                sample_flipped = copy(sample)
                for flip_term_i in flip_term
                    (x, y), Sij = flip_term_i
                    sample_flipped[x,y] = Sij
                end
                logψ_flipped[flip_term] = logψ_exact(peps, sample_flipped)
            end
        end
    end

    return logψ_flipped
end

function get_Ek(peps::AbstractPEPS, ham::OpSum, sample; kwargs...)
    hilbert = siteinds(peps)
    ham_op = TensorOperatorSum(ham, hilbert)
    return get_Ek(peps, ham_op, sample; kwargs...)
end

function get_Ek(peps::AbstractPEPS, ham_op::TensorOperatorSum, sample; env_top=Array{Environment}(undef, size(sample,1)-1), kwargs...)
    # get the environment tensors
    logψ, env_top, env_down = get_logψ_and_envs(peps, sample, env_top) # compute the environments of the peps according to that sample
    return get_Ek(peps, ham_op, env_top, env_down, sample, logψ; kwargs...)
end


function get_Ek(peps::AbstractPEPS, ham_op::TensorOperatorSum, env_top::Vector{Environment}, env_down::Vector{Environment}, sample::Matrix{Int64}, logψ::Number; h_envs_r=nothing, h_envs_l=nothing, fourb_envs_r=nothing, fourb_envs_l=nothing, logψ_flipped=nothing, Ek_terms=nothing, kwargs...)
    if Ek_terms === nothing
        Ek_terms = QuantumNaturalGradient.get_precomp_sOψ_elems(ham_op, sample; get_flip_sites=true)
    end
    logψ_flipped = get_logψ_flipped(peps, Ek_terms, env_top, env_down, sample, logψ; h_envs_r, h_envs_l, fourb_envs_r, fourb_envs_l, logψ_flipped, kwargs...)
    
    Ek = 0
    for flip_term in keys(Ek_terms)
        if flip_term == ()
            Ek += Ek_terms[flip_term]
        else
            Ek += Ek_terms[flip_term] * exp(logψ_flipped[flip_term] - logψ)
        end
    end

    return convert_if_real(Ek)
end