function _check_three_sublattice_torus(Lx::Integer, Ly::Integer, shear::Integer)
    mod(2Lx, 3) == 0 || throw(ArgumentError(
        "three-sublattice order requires 2Lx divisible by 3, got Lx=$Lx",
    ))
    mod(2shear - Ly, 3) == 0 || throw(ArgumentError(
        "three-sublattice order requires 2shear-Ly divisible by 3, " *
        "got Ly=$Ly and shear=$shear",
    ))
    return nothing
end

function _check_stripe_torus(Lx::Integer, Ly::Integer, shear::Integer)
    iseven(Lx) || throw(ArgumentError(
        "the selected canted-stripe orientation requires even Lx, got $Lx",
    ))
    iseven(shear - Ly) || throw(ArgumentError(
        "the selected canted-stripe orientation requires even shear-Ly, " *
        "got Ly=$Ly and shear=$shear",
    ))
    return nothing
end

_triangular_k_index(x::Integer, y::Integer) = mod(2(x - 1) - (y - 1), 3)
_triangular_k_angle(x::Integer, y::Integer) = 2pi * _triangular_k_index(x, y) / 3
_triangular_sublattice(x::Integer, y::Integer) = _triangular_k_index(x, y) + 1
_triangular_stripe_sign(x::Integer, y::Integer) =
    iseven((x - 1) - (y - 1)) ? 1.0 : -1.0

function _real_ansatz_parameters(η, expected_length::Integer, state_name::AbstractString)
    length(η) == expected_length || throw(DimensionMismatch(
        "$state_name requires $expected_length mean-field parameters, " *
        "got $(length(η))",
    ))
    all(isreal, η) || throw(ArgumentError(
        "$state_name mean-field parameters must be real",
    ))
    all(isfinite, η) || throw(ArgumentError(
        "$state_name mean-field parameters must be finite",
    ))
    return Float64.(real.(η))
end

"""
    triangular_y_hopping_fields(Lx, Ly, η; hopping_amplitude=1, shear=0)

Construct the hopping and fictitious-field arrays for the triangular-lattice
Y ansatz. The reduced variational vector follows the paper's ordering
`η = [h1, h2, h3, Delta]`. Sublattices are selected by
`mod(2(x-1)-(y-1), 3)` and carry fields

```text
A: (  0, 0,  h1)
B: ( h2, 0, -h3)
C: (-h2, 0, -h3).
```

The `Delta`-magnitude bonds connect the B and C sublattices. All other bonds
have magnitude `hopping_amplitude`; their signs retain the staggered `[0, pi]`
Dirac flux pattern.
"""
function triangular_y_hopping_fields(
    Lx::Integer,
    Ly::Integer,
    η::AbstractVector;
    hopping_amplitude::Real=1.0,
    shear::Integer=0,
)
    _check_three_sublattice_torus(Lx, Ly, shear)
    h1, h2, h3, delta = _real_ansatz_parameters(η, 4, "the Y ansatz")
    delta > 0 || throw(ArgumentError("the Y hopping magnitude Delta must be positive"))
    hopping = staggered_pi_flux_hoppings(Lx, Ly; amplitude=hopping_amplitude)
    for bond in triangular_torus_bonds(Lx, Ly; shear)
        source_sublattice = _triangular_sublattice(bond.source...)
        target_sublattice = _triangular_sublattice(bond.target...)
        if (source_sublattice == 2 && target_sublattice == 3) ||
           (source_sublattice == 3 && target_sublattice == 2)
            x, y = bond.source
            hopping[x, y, bond.direction] *= delta / hopping_amplitude
        end
    end

    fields = zeros(Float64, Lx, Ly, 3)
    for y in 1:Ly, x in 1:Lx
        sublattice = _triangular_sublattice(x, y)
        fields[x, y, :] .= if sublattice == 1
            (0.0, 0.0, h1)
        elseif sublattice == 2
            (h2, 0.0, -h3)
        else
            (-h2, 0.0, -h3)
        end
    end
    return hopping, fields
end

"""
    umbrella_hopping_fields(Lx, Ly, η; hopping_amplitude=1, shear=0)

Construct the hopping and fields for the triangular-lattice umbrella ansatz,
whose only optimized mean-field parameter is `η = [h]`. The hopping magnitudes
are uniform and retain the staggered `[0, pi]` flux. The field is
`M_i = h(cos(K*R_i), sin(K*R_i), 0)` with
`K*R_i = 2pi(2(x-1)-(y-1))/3`.
"""
function umbrella_hopping_fields(
    Lx::Integer,
    Ly::Integer,
    η::AbstractVector;
    hopping_amplitude::Real=1.0,
    shear::Integer=0,
)
    _check_three_sublattice_torus(Lx, Ly, shear)
    h = only(_real_ansatz_parameters(η, 1, "the umbrella ansatz"))
    hopping = staggered_pi_flux_hoppings(Lx, Ly; amplitude=hopping_amplitude)
    fields = zeros(Float64, Lx, Ly, 3)
    for y in 1:Ly, x in 1:Lx
        angle = _triangular_k_angle(x, y)
        fields[x, y, :] .= (h * cos(angle), h * sin(angle), 0.0)
    end
    return hopping, fields
end

"""
    cs_hopping_fields(
        Lx, Ly, η; hopping_amplitude=1, stripe_delta=0.8, shear=0
    )

Construct one of the three symmetry-related canted-stripe ansatze. The paper
optimizes only `η = [h]`; `stripe_delta` is therefore a fixed hopping magnitude.
The selected orientation is invariant along `a2 = (1, 1)` and alternates along
`a1 = (1, 0)`, with `M_i = ((-1)^(x-y) h, 0, 0)`. Direction-3 bonds have
magnitude `stripe_delta`, and the remaining bonds have magnitude
`hopping_amplitude`, all with the staggered `[0, pi]` signs.
"""
function cs_hopping_fields(
    Lx::Integer,
    Ly::Integer,
    η::AbstractVector;
    hopping_amplitude::Real=1.0,
    stripe_delta::Real=0.8,
    shear::Integer=0,
)
    _check_stripe_torus(Lx, Ly, shear)
    h = only(_real_ansatz_parameters(η, 1, "the canted-stripe ansatz"))
    stripe_delta > 0 || throw(ArgumentError(
        "the fixed canted-stripe hopping magnitude must be positive",
    ))
    hopping = staggered_pi_flux_hoppings(Lx, Ly; amplitude=hopping_amplitude)
    hopping[:, :, 3] .*= stripe_delta / hopping_amplitude
    fields = zeros(Float64, Lx, Ly, 3)
    for y in 1:Ly, x in 1:Lx
        fields[x, y, :] .= (h * _triangular_stripe_sign(x, y), 0.0, 0.0)
    end
    return hopping, fields
end

function _reduced_triangular_gaussian_state(
    Lx::Integer,
    Ly::Integer,
    η::AbstractVector,
    hopping,
    fields,
    expand_parameters::Function;
    shear::Integer,
    particle_number::Integer,
    gap_tolerance::Real,
    cache_gradients::Bool,
)
    number_of_sites = Lx * Ly
    number_of_modes = 2number_of_sites
    0 <= particle_number <= number_of_modes || throw(ArgumentError(
        "particle_number must lie between 0 and $number_of_modes, got $particle_number",
    ))
    gap_tolerance >= 0 || throw(ArgumentError(
        "gap_tolerance must be nonnegative, got $gap_tolerance",
    ))
    initial_Haux = hamiltonian_aux_triangular_torus(
        Lx,
        Ly;
        hopping,
        fields,
        shear,
    )
    chemical_potential = _auxiliary_chemical_potential(
        initial_Haux,
        particle_number,
        gap_tolerance,
    )
    hopping_phases = map(hopping) do value
        iszero(value) ? one(value) : value / abs(value)
    end
    H_BdG_func = _triangular_aux_bdg_function(
        Lx,
        Ly,
        triangular_torus_bonds(Lx, Ly; shear),
        hopping_phases,
        chemical_potential,
        length(η),
        expand_parameters,
    )
    return GaussianState(
        H_BdG_func,
        number_of_modes;
        η=copy(η),
        parity_sector=mod(particle_number, 2),
        target_state=0,
        cache_gradients,
    )
end

"""
    y_state(Lx, Ly; η, kwargs...)

Construct the full unprojected Y Gaussian state with the four optimized
parameters `η = [h1, h2, h3, Delta]`.
"""
function y_state(
    Lx::Integer,
    Ly::Integer;
    η::AbstractVector=[0.35, 0.25, 0.15, 0.8],
    hopping_amplitude::Real=1.0,
    shear::Integer=0,
    particle_number::Integer=Lx * Ly,
    gap_tolerance::Real=1e-8,
    cache_gradients::Bool=false,
)
    η0 = _real_ansatz_parameters(η, 4, "the Y ansatz")
    hopping, fields = triangular_y_hopping_fields(
        Lx,
        Ly,
        η0;
        hopping_amplitude,
        shear,
    )
    bonds = triangular_torus_bonds(Lx, Ly; shear)
    delta_bond = falses(Lx, Ly, 3)
    for bond in bonds
        source_sublattice = _triangular_sublattice(bond.source...)
        target_sublattice = _triangular_sublattice(bond.target...)
        if (source_sublattice == 2 && target_sublattice == 3) ||
           (source_sublattice == 3 && target_sublattice == 2)
            x, y = bond.source
            delta_bond[x, y, bond.direction] = true
        end
    end
    parameter_offset = zeros(Float64, 6Lx * Ly)
    parameter_map = zeros(Float64, 6Lx * Ly, 4)
    for y in 1:Ly, x in 1:Lx
        site = (y - 1) * Lx + x
        for direction in 1:3
            hopping_index = 3(site - 1) + direction
            if delta_bond[x, y, direction]
                parameter_map[hopping_index, 4] = 1.0
            else
                parameter_offset[hopping_index] = hopping_amplitude
            end
        end
        field_offset = 3Lx * Ly + 3(site - 1)
        sublattice = _triangular_sublattice(x, y)
        if sublattice == 1
            parameter_map[field_offset + 3, 1] = 1.0
        elseif sublattice == 2
            parameter_map[field_offset + 1, 2] = 1.0
            parameter_map[field_offset + 3, 3] = -1.0
        else
            parameter_map[field_offset + 1, 2] = -1.0
            parameter_map[field_offset + 3, 3] = -1.0
        end
    end
    expand_parameters = parameters -> parameter_offset + parameter_map * parameters
    return _reduced_triangular_gaussian_state(
        Lx,
        Ly,
        η0,
        hopping,
        fields,
        expand_parameters;
        shear,
        particle_number,
        gap_tolerance,
        cache_gradients,
    )
end

"""
    umbrella_state(Lx, Ly; η, kwargs...)

Construct the full unprojected umbrella Gaussian state with its single
optimized parameter `η = [h]`.
"""
function umbrella_state(
    Lx::Integer,
    Ly::Integer;
    η::AbstractVector=[0.3],
    hopping_amplitude::Real=1.0,
    shear::Integer=0,
    particle_number::Integer=Lx * Ly,
    gap_tolerance::Real=1e-8,
    cache_gradients::Bool=false,
)
    η0 = _real_ansatz_parameters(η, 1, "the umbrella ansatz")
    hopping, fields = umbrella_hopping_fields(
        Lx,
        Ly,
        η0;
        hopping_amplitude,
        shear,
    )
    parameter_offset = zeros(Float64, 6Lx * Ly)
    parameter_offset[1:(3Lx * Ly)] .= hopping_amplitude
    parameter_map = zeros(Float64, 6Lx * Ly, 1)
    for y in 1:Ly, x in 1:Lx
        site = (y - 1) * Lx + x
        field_offset = 3Lx * Ly + 3(site - 1)
        angle = _triangular_k_angle(x, y)
        parameter_map[field_offset + 1, 1] = cos(angle)
        parameter_map[field_offset + 2, 1] = sin(angle)
    end
    expand_parameters = parameters -> parameter_offset + parameter_map * parameters
    return _reduced_triangular_gaussian_state(
        Lx,
        Ly,
        η0,
        hopping,
        fields,
        expand_parameters;
        shear,
        particle_number,
        gap_tolerance,
        cache_gradients,
    )
end

"""
    cs_state(Lx, Ly; η, kwargs...)

Construct the full unprojected canted-stripe Gaussian state with the paper's
single optimized parameter `η = [h]`. The hopping ratio is supplied separately
through the fixed keyword `stripe_delta`.
"""
function cs_state(
    Lx::Integer,
    Ly::Integer;
    η::AbstractVector=[0.3],
    hopping_amplitude::Real=1.0,
    stripe_delta::Real=0.8,
    shear::Integer=0,
    particle_number::Integer=Lx * Ly,
    gap_tolerance::Real=1e-8,
    cache_gradients::Bool=false,
)
    η0 = _real_ansatz_parameters(η, 1, "the canted-stripe ansatz")
    hopping, fields = cs_hopping_fields(
        Lx,
        Ly,
        η0;
        hopping_amplitude,
        stripe_delta,
        shear,
    )
    parameter_offset = zeros(Float64, 6Lx * Ly)
    parameter_map = zeros(Float64, 6Lx * Ly, 1)
    for y in 1:Ly, x in 1:Lx
        site = (y - 1) * Lx + x
        for direction in 1:3
            hopping_index = 3(site - 1) + direction
            parameter_offset[hopping_index] =
                direction == 3 ? stripe_delta : hopping_amplitude
        end
        field_offset = 3Lx * Ly + 3(site - 1)
        parameter_map[field_offset + 1, 1] = _triangular_stripe_sign(x, y)
    end
    expand_parameters = parameters -> parameter_offset + parameter_map * parameters
    return _reduced_triangular_gaussian_state(
        Lx,
        Ly,
        η0,
        hopping,
        fields,
        expand_parameters;
        shear,
        particle_number,
        gap_tolerance,
        cache_gradients,
    )
end
