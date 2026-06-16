using ITensorMPS: sim!, deprecate_make_inds_match!, check_hascommoninds
# From abstractmps.jl
function _log_or_not_dot(
   M1::MPST, M2::MPST, loginner::Bool; dag=true, make_inds_match::Bool=true
 )::Number where {MPST<:ITensorMPS.AbstractMPS}
   N = length(M1)
   if length(M2) != N
     throw(DimensionMismatch("inner: mismatched lengths $N and $(length(M2))"))
   end
   M1dag = M1
   if dag # modified code
       M1dag = dag(M1)
   end
   sim!(linkinds, M1dag)
    M1dag, M2 = deprecate_make_inds_match!(
      ITensorMPS._log_or_not_dot, M1dag, M2, loginner; make_inds_match
    )
    check_hascommoninds(siteinds, M1dag, M2)
    O = M1dag[1] * M2[1]
  
    if loginner
      normO = norm(O)
      log_inner_tot = log(normO)
      O ./= normO
    end
  
    for j in eachindex(M1)[2:end]
      O = (O * M1dag[j]) * M2[j]
  
      if loginner
        normO = norm(O)
        log_inner_tot += log(normO)
        O ./= normO
      end
    end
  
    if loginner
      if !isreal(O[]) || real(O[]) < 0
        log_inner_tot += log(complex(O[]))
      end
      return log_inner_tot
    end
  
    dot_M1_M2 = O[]
  
    if !isfinite(dot_M1_M2)
      @warn "The inner product (or norm²) you are computing is very large " *
        "($dot_M1_M2). You should consider using `lognorm` or `loginner` instead, " *
        "which will help avoid floating point errors. For example if you are trying " *
        "to normalize your MPS/MPO `A`, the normalized MPS/MPO `B` would be given by " *
        "`B = A ./ z` where `z = exp(lognorm(A) / length(A))`."
    end
  
    return dot_M1_M2
  end