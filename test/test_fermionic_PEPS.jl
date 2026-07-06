using Test
using ITensors, ITensorMPS
using QuantumNaturalfPEPS
using QuantumNaturalGradient
using Random

#=
    Fermionic PEPS with siteinds("Fermion", Lx, Ly) and trial_state = IdentityState.

    Convention: the PEPS stores the Jordan-Wigner image of the fermionic state in the
    column-major mode ordering n = i + (j-1)*Lx (same ordering as vec(S)). All fermionic
    signs therefore live in the Hamiltonian matrix elements: TensorOperatorSum inserts the
    JW string "F" between the fermionic operators (see QuantumNaturalGradient.insert_JW_string).
    The dense tensor permutes of the package consequently must NOT introduce any signs, while
    ITensors' permute/getindex on fermionic QN indices (auto_fermion) must apply the JW signs.
=#

@testset "Permute functions" begin
    #=
        Purpose: permute_and_copy! and permute_reshape_and_copy! are the package's raw-storage
        permutation helpers. They are used in every place where PEPS tensors are flattened into
        or read from the parameter vector θ (vec(peps), write!(peps, θ), get_Ok), so a wrong
        permutation (or a spurious sign) here would corrupt every gradient and every parameter
        update. We check them on a tensor that looks exactly like an fPEPS site tensor — a
        dense "Fermion" site index from siteinds("Fermion", Lx, Ly) plus link indices — against
        the analytic reference (Julia's permutedims) and against ITensors' own permute, for all
        3! = 6 possible index orders. Note that these helpers dispatch on dense storage
        (NDTensors.DenseTensor) only; that is exactly what the fPEPS uses — QN/block-sparse
        tensors never enter this pipeline.
    =#
    @testset "permute_and_copy! / permute_reshape_and_copy! (dense Fermion site index)" begin
        hilbert = ITensors.siteinds("Fermion", 2, 2)
        s = hilbert[1, 1]                            # dense "Fermion,Site" index, dim 2
        l1, l2 = Index(3, "h_link"), Index(4, "v_link") # link-like indices (dims distinct on purpose)
        A = reshape(collect(1.0:24.0), 2, 3, 4)
        T = ITensor(A, s, l1, l2)

        for target in [(s, l1, l2), (l2, s, l1), (l1, s, l2), (l2, l1, s), (s, l2, l1), (l1, l2, s)]
            perm = map(ind -> findfirst(==(ind), (s, l1, l2)), target)
            expected = permutedims(A, collect(perm))

            # permute_and_copy! must reproduce the analytic permutation of the raw data —
            # no fermionic sign may appear (dense indices carry no parity information)
            dest = zeros(size(expected))
            QuantumNaturalfPEPS.permute_and_copy!(dest, T, target)
            @test dest == expected

            # permute_reshape_and_copy! is the flattened variant (used to fill segments of θ):
            # same permutation, but written as a vector in column-major order
            destv = zeros(length(A))
            QuantumNaturalfPEPS.permute_reshape_and_copy!(destv, T, target)
            @test destv == vec(expected)

            # cross-check against ITensors.permute: reading the ITensors-permuted tensor in the
            # target index order must give exactly the same array as the package helper
            Tp = ITensors.permute(T, target...)
            @test dest == Array(Tp, target...)
        end

        # vec(peps) writes each tensor through a view into one segment of the parameter vector θ.
        # Check that writing through a view fills exactly the addressed slice (first 24 entries)
        # and does not touch the rest of the buffer.
        buf = zeros(2 * length(A))
        QuantumNaturalfPEPS.permute_reshape_and_copy!((@view buf[1:24]), T, (l2, s, l1))
        @test buf[1:24] == vec(permutedims(A, (3, 1, 2)))
        @test all(buf[25:end] .== 0)
    end

    #=
        Purpose: the PEPS in this package is built from dense (non-QN) ITensors, even for the
        "Fermion" sitetype. Without quantum numbers there is no parity information on the
        indices, so permuting must NOT introduce any fermionic signs — by convention all JW
        signs are produced by the Hamiltonian (TensorOperatorSum inserts the "F" strings).
        If ITensors or the package helpers ever started signing dense permutes, every sampled
        amplitude would silently change sign structure; this testset pins the convention down.
    =#
    @testset "dense Fermion-tagged permutes are sign-free" begin
        s = siteinds("Fermion", 2)
        F = ITensor(s[1], s[2])
        F[s[1] => 2, s[2] => 2] = 1.0 # |11>-amplitude (both modes occupied)

        # ITensors.permute on dense indices: the |11> amplitude keeps its sign,
        # no matter in which index order the permuted tensor is read
        Fp = ITensors.permute(F, s[2], s[1])
        @test Fp[s[1] => 2, s[2] => 2] == 1.0
        @test Fp[s[2] => 2, s[1] => 2] == 1.0

        # the package helper must behave identically: plain transpose of the data, no sign
        dest = zeros(2, 2)
        QuantumNaturalfPEPS.permute_and_copy!(dest, F, (s[2], s[1]))
        @test dest == [0.0 0.0; 0.0 1.0]
    end

    #=
        Purpose: with conserve_qns=true the "Fermion" site indices carry fermionic parity and
        ITensors' updated permute/getindex apply the JW signs automatically (the ITensors
        feature this issue is based on). We verify the signs against the analytic fermionic
        exchange rule: reordering fermionic modes picks up (-1)^(number of exchanges of two
        occupied modes); empty modes commute with everything.

        NOTE: the sign machinery is only active while ITensors' global auto_fermion flag is
        enabled — without it, QN fermionic indices permute like bosonic ones (no signs).
        We enable it explicitly here (and disable it again in `finally`, so that no state
        leaks into the other testsets, which rely on the sign-free dense behavior).
    =#
    @testset "ITensors permute applies fermionic signs (auto_fermion, QN)" begin
        ITensors.enable_auto_fermion()
        try
        s = siteinds("Fermion", 2; conserve_qns=true)

        # |11>: reading the two occupied (odd-parity) modes in swapped order exchanges
        # two fermions once -> amplitude changes sign
        A = ITensor(s[1], s[2])
        A[s[1] => 2, s[2] => 2] = 1.0
        @test A[s[2] => 2, s[1] => 2] == -1.0
        Ap = ITensors.permute(A, s[2], s[1])
        # permute preserves the abstract (basis-independent) state: reading the permuted tensor
        # in the ORIGINAL index order must reproduce the original amplitude ...
        @test Ap[s[1] => 2, s[2] => 2] == 1.0
        # ... while its storage in the new index order carries the JW exchange sign
        @test Ap[s[2] => 2, s[1] => 2] == -1.0

        # |01>: only one mode occupied, swapping an occupied with an empty mode is sign-free
        B = ITensor(s[1], s[2])
        B[s[1] => 1, s[2] => 2] = 3.0
        @test B[s[2] => 2, s[1] => 1] == 3.0
        Bp = ITensors.permute(B, s[2], s[1])
        @test Bp[s[1] => 1, s[2] => 2] == 3.0

        s3 = siteinds("Fermion", 3; conserve_qns=true)

        # |111>: a cyclic permutation of three occupied modes is two exchanges -> even -> sign +1
        C = ITensor(s3[1], s3[2], s3[3])
        C[s3[1] => 2, s3[2] => 2, s3[3] => 2] = 1.0
        @test C[s3[3] => 2, s3[1] => 2, s3[2] => 2] == 1.0
        Cp = ITensors.permute(C, s3[3], s3[1], s3[2])
        @test Cp[s3[1] => 2, s3[2] => 2, s3[3] => 2] == 1.0

        # |101>: modes 1 and 3 occupied. Full reversal (1,2,3)->(3,2,1) exchanges the two
        # occupied modes exactly once (the empty mode 2 contributes nothing) -> sign -1
        D = ITensor(s3[1], s3[2], s3[3])
        D[s3[1] => 2, s3[2] => 1, s3[3] => 2] = 7.0
        @test D[s3[3] => 2, s3[2] => 1, s3[1] => 2] == -7.0
        Dp = ITensors.permute(D, s3[3], s3[2], s3[1])
        @test Dp[s3[1] => 2, s3[2] => 1, s3[3] => 2] == 7.0  # original order: unchanged
        @test Dp[s3[3] => 2, s3[2] => 1, s3[1] => 2] == -7.0 # reversed order: exchange sign

        # swapping the first two indices (occupied, empty) leaves the amplitude unchanged
        Ep = ITensors.permute(D, s3[2], s3[1], s3[3])
        @test Ep[s3[2] => 1, s3[1] => 2, s3[3] => 2] == 7.0
        finally
            ITensors.disable_auto_fermion()
        end
    end

    #=
        Purpose: vec(peps) flattens all PEPS tensors into θ via permute_reshape_and_copy! and
        write!(peps, θ) writes θ back, re-permuting into each tensor's index order. On a
        fermionic PEPS this circle must be exactly lossless — any inconsistency between the
        two permutation directions would make every QNG parameter update corrupt the state.
    =#
    @testset "vec/write! circle on a fermionic PEPS" begin
        Random.seed!(7)
        hilbert = ITensors.siteinds("Fermion", 3, 3)
        peps = PEPS(hilbert; bond_dim=2)
        θ = vec(peps)
        write!(peps, θ)
        # after one vec -> write! -> vec circle the parameter vector must be bit-identical
        @test vec(peps) == θ
    end
end

#=
    Purpose: verify that ALL 16 amplitudes <S|ψ>, S = 0000 ... 1111, of a fermionic state on
    the 2x2 lattice agree between three independent representations:
      [1] ITensors with QN "Fermion" sites and auto_fermion: the state is built by applying
          Cdag operators to the vacuum, so ITensors produces the fermionic signs,
      [2] the analytic amplitudes with hand-computed Jordan-Wigner signs,
      [3] the package fPEPS constructed with siteinds("Fermion", 2, 2): amplitudes computed by
          projecting and contracting the PEPS (exactly and through the sampling machinery),
          and again after a vec(peps) -> write! round trip.

    Reference state (JW modes col-major: 1=(1,1), 2=(2,1), 3=(1,2), 4=(2,2)):
        |ψ> = (u1 + v1 c†_1 c†_3)(u2 + v2 c†_2 c†_4)|0>
    In the canonical basis convention |S> = (c†_1)^n1 (c†_2)^n2 (c†_3)^n3 (c†_4)^n4 |0>:
        <0000|ψ> =  u1 u2
        <1010|ψ> =  v1 u2                    (c†_1 c†_3 already in canonical order)
        <0101|ψ> =  u1 v2                    (c†_2 c†_4 already in canonical order)
        <1111|ψ> = -v1 v2                    (c†_1 c†_3 c†_2 c†_4 = -c†_1 c†_2 c†_3 c†_4,
                                              one exchange of c†_3 and c†_2)
    and all other amplitudes vanish. The pairs (1,3) and (2,4) are the two horizontal bonds
    of the 2x2 lattice, so this state has a genuine JW crossing sign that a "bosonic" tensor
    network treatment would miss.
=#
@testset "Fermionic amplitudes 0000...1111 on the 2x2 lattice" begin
    u1, v1, u2, v2 = 2.0, 3.0, 5.0, 7.0

    # [2] analytic amplitudes, indexed [n1+1, n2+1, n3+1, n4+1]
    amps_analytic = zeros(2, 2, 2, 2)
    amps_analytic[1, 1, 1, 1] = u1 * u2
    amps_analytic[2, 1, 2, 1] = v1 * u2
    amps_analytic[1, 2, 1, 2] = u1 * v2
    amps_analytic[2, 2, 2, 2] = -v1 * v2

    #=
        [1] ITensors reference. Note: unlike permute/getindex (previous testset), fermionic
        OPERATOR PRODUCTS via apply() require the auto_fermion machinery to be enabled,
        otherwise the Cdag applications commute like bosonic operators and no sign appears.
    =#
    ITensors.enable_auto_fermion()
    try
        sQN = siteinds("Fermion", 4; conserve_qns=true)
        vac = onehot(sQN[1] => 1) * onehot(sQN[2] => 1) * onehot(sQN[3] => 1) * onehot(sQN[4] => 1)

        # apply c† operators, rightmost first: order = [1,3] builds c†_1 c†_3 |0>
        function apply_cdags(order)
            ψ = vac
            for n in reverse(order)
                ψ = ITensors.apply(op("Cdag", sQN[n]), ψ)
            end
            return ψ
        end
        # elementwise read in the canonical site order = amplitude in the canonical convention
        amp_IT(ψ, occ) = ψ[sQN[1] => occ[1]+1, sQN[2] => occ[2]+1, sQN[3] => occ[3]+1, sQN[4] => occ[4]+1]

        # each component of |ψ> in the operator order of the state definition:
        ψ_1010 = apply_cdags([1, 3])       # c†_1 c†_3 |0>
        ψ_0101 = apply_cdags([2, 4])       # c†_2 c†_4 |0>
        ψ_1111 = apply_cdags([1, 3, 2, 4]) # c†_1 c†_3 c†_2 c†_4 |0>

        # vacuum and canonically ordered pairs: ITensors returns +1 (no reordering necessary)
        @test amp_IT(vac, [0, 0, 0, 0]) == 1.0
        @test amp_IT(ψ_1010, [1, 0, 1, 0]) == 1.0
        @test amp_IT(ψ_0101, [0, 1, 0, 1]) == 1.0
        # the 4-fermion component: ITensors must reproduce the analytic reordering sign -1
        @test amp_IT(ψ_1111, [1, 1, 1, 1]) == -1.0

        # anticommutation check: applying the pair in swapped order c†_3 c†_1 |0> = -c†_1 c†_3 |0>
        @test amp_IT(apply_cdags([3, 1]), [1, 0, 1, 0]) == -1.0

        # tie the ITensors signs to the analytic amplitude table
        @test amps_analytic[1, 1, 1, 1] == u1 * u2 * amp_IT(vac, [0, 0, 0, 0])
        @test amps_analytic[2, 1, 2, 1] == v1 * u2 * amp_IT(ψ_1010, [1, 0, 1, 0])
        @test amps_analytic[1, 2, 1, 2] == u1 * v2 * amp_IT(ψ_0101, [0, 1, 0, 1])
        @test amps_analytic[2, 2, 2, 2] == v1 * v2 * amp_IT(ψ_1111, [1, 1, 1, 1])
    finally
        ITensors.disable_auto_fermion()
    end

    #=
        [3] package fPEPS with dense siteinds("Fermion", 2, 2), bond_dim = 2, built by hand:
        the occupation of pair (1,3) travels on the top horizontal link (α = 2 <-> pair
        occupied), the occupation of pair (2,4) on the bottom horizontal link (β), and the
        right vertical link δ copies β upwards so that site (1,2) can apply the JW crossing
        sign (-1)^(n2*n3) — exactly the sign a fermionic PEPS has to encode in its tensors.
        The left vertical link is a dim-2 dummy of which only the first component is used.
    =#
    hilbert = ITensors.siteinds("Fermion", 2, 2)
    peps = PEPS(hilbert; bond_dim=2)

    si11 = ITensors.siteind(peps, 1, 1); si21 = ITensors.siteind(peps, 2, 1)
    si12 = ITensors.siteind(peps, 1, 2); si22 = ITensors.siteind(peps, 2, 2)
    h_top = commonind(peps[1, 1], peps[1, 2]); h_bot = commonind(peps[2, 1], peps[2, 2])
    v_l = commonind(peps[1, 1], peps[2, 1]);   v_r = commonind(peps[1, 2], peps[2, 2])

    T11 = ITensor(Float64, si11, h_top, v_l)
    T11[si11 => 1, h_top => 1, v_l => 1] = 1.0 # site (1,1) empty    <-> pair (1,3) empty
    T11[si11 => 2, h_top => 2, v_l => 1] = 1.0 # site (1,1) occupied <-> pair (1,3) occupied

    T21 = ITensor(Float64, si21, h_bot, v_l)
    T21[si21 => 1, h_bot => 1, v_l => 1] = 1.0 # same for pair (2,4) on the bottom link
    T21[si21 => 2, h_bot => 2, v_l => 1] = 1.0

    T22 = ITensor(Float64, si22, h_bot, v_r)
    T22[si22 => 1, h_bot => 1, v_r => 1] = u2  # pair (2,4) empty:    coefficient u2, δ = 1
    T22[si22 => 2, h_bot => 2, v_r => 2] = v2  # pair (2,4) occupied: coefficient v2, δ = 2

    T12 = ITensor(Float64, si12, h_top, v_r)
    T12[si12 => 1, h_top => 1, v_r => 1] = u1  # pair (1,3) empty
    T12[si12 => 1, h_top => 1, v_r => 2] = u1
    T12[si12 => 2, h_top => 2, v_r => 1] = v1  # pair (1,3) occupied, pair (2,4) empty
    T12[si12 => 2, h_top => 2, v_r => 2] = -v1 # both pairs occupied: JW crossing sign

    peps[1, 1] = T11; peps[2, 1] = T21; peps[1, 2] = T12; peps[2, 2] = T22
    peps.double_layer_envs = nothing

    # all 16 amplitudes from the exact contraction of the projected PEPS must reproduce the
    # analytic fermionic amplitudes (including the 12 vanishing ones and the -v1*v2 sign)
    for n1 in 0:1, n2 in 0:1, n3 in 0:1, n4 in 0:1
        S = [n1 n3; n2 n4] # col-major: vec(S) = [n1, n2, n3, n4] matches the JW mode ordering
        amp = QuantumNaturalfPEPS.contract_peps_exact(QuantumNaturalfPEPS.get_projected(peps, S))
        @test isapprox(amp, amps_analytic[n1+1, n2+1, n3+1, n4+1]; atol=1e-12)
    end

    # the sampling machinery (get_logψ_and_envs = boundary-MPS contraction used during
    # Ok/Ek evaluation) must give the same amplitudes; only the four nonzero ones can be
    # checked since log(0) is singular. exp(logψ) is complex for the negative amplitude.
    for occ in ([0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1])
        S = reshape(occ, 2, 2)
        logψ, _, _, _ = QuantumNaturalfPEPS.get_logψ_and_envs(peps, S; pos=1)
        @test isapprox(exp(logψ), amps_analytic[(occ .+ 1)...]; atol=1e-10)
    end

    # "the amplitude we get from vec(peps)": flatten the PEPS into θ, write θ into a second
    # PEPS with the same indices (this passes through permute_reshape_and_copy!/write!) and
    # verify that all 16 amplitudes survive the round trip through the parameter vector
    θ = vec(peps)
    peps2 = deepcopy(peps)
    for i in 1:2, j in 1:2 # scramble the tensor data so write! has to do the real work
        peps2[i, j] = ITensors.random_itensor(Float64, inds(peps[i, j])...)
    end
    peps2.double_layer_envs = nothing
    write!(peps2, θ)
    for n1 in 0:1, n2 in 0:1, n3 in 0:1, n4 in 0:1
        S = [n1 n3; n2 n4]
        amp = QuantumNaturalfPEPS.contract_peps_exact(QuantumNaturalfPEPS.get_projected(peps2, S))
        @test isapprox(amp, amps_analytic[n1+1, n2+1, n3+1, n4+1]; atol=1e-12)
    end
end

#=
    Purpose: end-to-end validation of the full fermionic pipeline — sampling, Ok/Ek evaluation
    and QuantumNaturalGradient.evolve — for a PEPS with siteinds("Fermion", Lx, Ly) and
    trial_state = IdentityState(dim(siteinds(peps)[1])). We use the two exactly solvable limits
    of the (spinless) Hubbard model on the 4x4 open lattice.
=#
@testset "4x4 Hubbard limits, fermionic PEPS" begin
    Lx, Ly = 4, 4

    # D=1 product PEPS with single-site state a|0> + b|1> on every site
    function product_peps(hilbert, a, b)
        function product_tensor(::Type{S}, incoming, outgoing) where {S<:Number}
            inds_ = (incoming..., outgoing...)
            data = zeros(S, map(dim, inds_)...)
            data[1, ntuple(_ -> 1, length(inds_) - 1)...] = a
            data[2, ntuple(_ -> 1, length(inds_) - 1)...] = b
            return ITensor(data, inds_...)
        end
        return PEPS(hilbert; bond_dim=1, tensor_init=product_tensor)
    end

    # set a D=1 peps to the product state with occupation pattern occ(i,j)::Bool
    # ε > 0 admixes the opposite occupation, i.e. the site state becomes |n⟩ + ε|1-n⟩
    function set_occupation_peps!(peps, occ; ε=0.0)
        for i in 1:size(peps, 1), j in 1:size(peps, 2)
            si = ITensors.siteind(peps, i, j)
            inds_ = inds(peps[i, j])
            pos = findfirst(==(si), collect(inds_))
            data = zeros(Float64, dim.(inds_)...)
            data[ntuple(k -> k == pos ? 1 : 1, length(inds_))...] = occ(i, j) ? ε : 1.0
            data[ntuple(k -> k == pos ? 2 : 1, length(inds_))...] = occ(i, j) ? 1.0 : ε
            peps[i, j] = ITensor(data, inds_...)
        end
        peps.double_layer_envs = nothing
        return peps
    end

    is_cdw_occupied(i, j) = iseven(i + j)

    #=
        Purpose: free-fermion limit (U=0), part 1 — a FIXED D=1 product state. For a product
        state the energy of the hopping Hamiltonian is known in closed form INCLUDING the JW
        string factors, so the sampled energy tests the complete chain
        OpSum -> TensorOperatorSum (JW insertion) -> sampling -> Ek against an analytic value.
        A second reference value E_wrong_sign (strings ignored) makes sure the test actually
        resolves the fermionic sign and not just the overall magnitude.
    =#
    @testset "U=0 limit (t=1): sampled energy vs analytic value incl. JW strings" begin
        Random.seed!(42)
        t, U = 1.0, 0.0
        Hubbard_ham = QuantumNaturalfPEPS.hamiltonian_hubbard(t, U, Lx, Ly)

        # product state with a^2 - b^2 = -1/2 so the JW string factor on horizontal bonds is
        # (a^2-b^2)^3 = -1/8 and a wrong sign treatment is clearly distinguishable
        a, b = 0.5, sqrt(3) / 2
        hilbert = ITensors.siteinds("Fermion", Lx, Ly)
        peps = product_peps(hilbert, a, b)
        trial_state = QuantumNaturalfPEPS.IdentityState(dim(siteinds(peps)[1]))

        #=
            Analytic expectation value for the product state ⊗_k (a|0⟩ + b|1⟩):

            Under the JW transformation with column-major mode ordering n = i + (j-1)*Lx the
            hopping term for modes n < m becomes
                c†_n c_m  ->  σ⁺_n [ Π_{k=n+1}^{m-1} F_k ] σ⁻_m,   F = 1 - 2n̂ = diag(1, -1),
            i.e. a string of parity operators F acts on every mode STRICTLY BETWEEN n and m.
            For the product state each factor takes its single-site expectation value:
                ⟨σ⁺⟩-type endpoint factors: ⟨a,b|σ⁺|a,b⟩ = a*b   (one per bond endpoint)
                string sites:               ⟨a,b|F|a,b⟩ = a² - b²

            - 12 vertical bonds (i,j)-(i+1,j): the JW modes n and n+1 are ADJACENT, the string
              is empty:                       ⟨c†c⟩ = (ab)·(ab), plus h.c. -> 2(ab)²
            - 12 horizontal bonds (i,j)-(i,j+1): the JW modes n and n+Lx are 4 apart, so the
              string covers the 3 modes n+1, n+2, n+3 (the sites below (i,j) in column j and
              above (i,j+1) in column j+1). Each contributes one factor (a² - b²):
                                              ⟨c†c + h.c.⟩ = 2(ab)² · (a² - b²)³
        =#
        ab2 = (a * b)^2
        string_factor = (a^2 - b^2)^3
        E_exact = -t * (12 * 2 * ab2 + 12 * 2 * ab2 * string_factor)
        E_wrong_sign = -t * (12 * 2 * ab2 + 12 * 2 * ab2 * abs(string_factor)) # if JW strings were ignored

        Nmeasure = 2000
        E, E_err, _ = QuantumNaturalfPEPS.weighted_mean_error(
            QuantumNaturalfPEPS.get_ExpectationValue(peps, Hubbard_ham; trial_state=trial_state, it=Nmeasure)...)

        # the Monte-Carlo error bar is σ(Ek)/sqrt(Nmeasure) with σ(Ek) ≈ 3.3 for this product
        # state (the local energy fluctuates over ±24 hopping bonds), so it can never drop
        # below σ/sqrt(N) no matter how many samples are taken. The factor 5 makes the bound a
        # sanity check of the error bar itself: a blow-up would signal broken sampling
        # probabilities / degenerate importance weights
        @test E_err <= 5 / sqrt(Nmeasure)
        # sampled energy agrees with the analytic JW value within the error bar
        @test abs(E - E_exact) < E_err
        # ... and is statistically distinguishable from the value with ignored JW strings,
        # i.e. the test actually resolves the sign of the string factor
        @test abs(E - E_wrong_sign) > E_err
    end

    #=
        Purpose: free-fermion limit (U=0), part 2 — QuantumNaturalGradient.evolve towards the
        exact ground state
            ε_{m,n} = -2t[cos(mπ/5) + cos(nπ/5)]  (4x4 OBC single-particle levels)
            at half-filling: E_exact = -(1+√5) - 2√5 - (√5-1) - 2·1 = -2 - 4√5 ≈ -10.944.

        A D=1 PEPS with IdentityState is a bare product state and CANNOT represent this
        correlated state: the best product state has E = -6t (all sites in (|0⟩+|1⟩)/√2, where
        the horizontal JW string factor ⟨F⟩³ vanishes and only the 12 vertical bonds
        contribute -t/2 each). To capture the fermionic correlations — including the internal
        JW crossing signs tested amplitude-by-amplitude in the 2x2 testset above — the PEPS
        needs bond dimension D >= 2, so this testset evolves a D=2 PEPS.

        Init: from a completely random D=2 PEPS the optimization converges too slowly for a
        test budget (E ≈ -8.8 after 150 iterations, still descending). As in the CDW testset
        below we therefore use the ε-logic: start from the best D=1 product state (the uniform
        superposition, written into the D=1 block) plus ε times the random D=2 tensors, which
        reaches the same trajectory ~50 iterations earlier.

        D=2 is still a truncated variational ansatz, so the energy cannot reach E_exact
        exactly; the test asserts that evolve gets within ~15% of E_exact (empirically the
        run passes -9.5) AND does not overshoot below it. The lower bound is a real physics
        check: without the JW strings the model would effectively be hardcore BOSONS, whose
        ground state is not sign-frustrated and lies BELOW the fermionic one — overshooting
        E_exact would therefore indicate broken fermionic statistics.
    =#
    @testset "U=0 limit (t=1): D=2 evolve towards the free-fermion ground state" begin
        Random.seed!(42)
        t, U = 1.0, 0.0
        Hubbard_ham = QuantumNaturalfPEPS.hamiltonian_hubbard(t, U, Lx, Ly)
        E_exact = -2 - 4 * sqrt(5)

        hilbert = ITensors.siteinds("Fermion", Lx, Ly)
        peps = PEPS(hilbert; bond_dim=2)
        trial_state = QuantumNaturalfPEPS.IdentityState(dim(siteinds(peps)[1]))

        # ε-biased init: uniform-superposition product state on the D=1 block + ε * random rest
        ε = 0.3
        for i in 1:Lx, j in 1:Ly
            si = ITensors.siteind(peps, i, j)
            inds_ = inds(peps[i, j])
            pos = findfirst(==(si), collect(inds_))
            data = ε .* Array(peps[i, j], inds_...)
            data[ntuple(k -> k == pos ? 1 : 1, length(inds_))...] += 1.0
            data[ntuple(k -> k == pos ? 2 : 1, length(inds_))...] += 1.0
            peps[i, j] = ITensor(data, inds_...)
        end
        peps.double_layer_envs = nothing

        θ = vec(QuantumNaturalGradient.Parameters(peps).obj)
        integrator = QuantumNaturalGradient.Euler(lr=0.1)
        Oks_and_Eks = QuantumNaturalfPEPS.generate_Oks_and_Eks(peps, Hubbard_ham; trial_state=trial_state)

        @time loss_value, trained_θ, misc = QuantumNaturalGradient.evolve(Oks_and_Eks, θ;
            integrator, verbosity=2, sample_nr=200, maxiter=150)

        # the optimization must approach the exact ground-state energy from above:
        # close (D=2 truncation + stochastic noise floor) ...
        @test loss_value < -9.0
        # ... but never significantly below it (fermionic statistics intact, see comment above;
        # the 0.5 slack covers the noise of the stochastic loss estimate)
        @test loss_value > E_exact - 0.5

        # independent re-measurement of the evolved PEPS
        Nmeasure = 1000
        E2, E2_err, _ = QuantumNaturalfPEPS.weighted_mean_error(
            QuantumNaturalfPEPS.get_ExpectationValue(peps, Hubbard_ham; trial_state=trial_state, it=Nmeasure)...)
        @test E2 < -9.0
        @test E2 > E_exact - 0.5
    end

    #=
        Purpose: atomic limit (t=0). The Hamiltonian is diagonal in the occupation basis and
        the ground state is the checkerboard CDW product state with exactly one occupied site
        per bond: E = -U/2 per bond, i.e. E = -24 for U=2 on the 24 bonds of the 4x4 lattice.
        A D=1 PEPS represents this state exactly, so both the measured energy and the evolve
        loss must reproduce -24 to machine/sampling-free precision. This testset also serves
        as the end-to-end check that QuantumNaturalGradient.evolve works with
        generate_Oks_and_Eks(peps, ham; trial_state=IdentityState(...)).
    =#
    @testset "t=0 limit (U=2): exact CDW energy and evolve convergence" begin
        Random.seed!(1234)
        t, U = 0.0, 2.0
        Hubbard_ham = QuantumNaturalfPEPS.hamiltonian_hubbard(t, U, Lx, Ly)
        # CDW (t=0): each of the 24 NN bonds has exactly one occupied site: E = -U/2 * 24 = -24
        E_exact = -24.0

        hilbert = ITensors.siteinds("Fermion", Lx, Ly)
        peps = PEPS(hilbert; bond_dim=1)
        trial_state = QuantumNaturalfPEPS.IdentityState(dim(siteinds(peps)[1]))

        # exact CDW product PEPS: the CDW is an eigenstate of the diagonal Hamiltonian, so
        # EVERY sample returns exactly Ek = -24 — the mean is exact and the variance is zero
        set_occupation_peps!(peps, is_cdw_occupied)
        E, E_err, _ = QuantumNaturalfPEPS.weighted_mean_error(
            QuantumNaturalfPEPS.get_ExpectationValue(peps, Hubbard_ham; trial_state=trial_state, it=20)...)
        # sampled energy is exact (not just within sampling error)
        @test isapprox(E, E_exact; atol=1e-10)
        # zero-variance check: all local energies identical
        @test E_err <= 1e-10

        # QuantumNaturalGradient.evolve runs end-to-end and converges to the exact value.
        # Biased init: from a completely random init the diagonal t=0 Hamiltonian has
        # domain-wall local minima (the optimization gets stuck at E = -20, one broken bond
        # row), so we start from the CDW pattern with an ε admixture on every site.
        set_occupation_peps!(peps, is_cdw_occupied; ε=0.3)
        θ = vec(QuantumNaturalGradient.Parameters(peps).obj)

        integrator = QuantumNaturalGradient.Euler(lr=0.1)
        Oks_and_Eks = QuantumNaturalfPEPS.generate_Oks_and_Eks(peps, Hubbard_ham; trial_state=trial_state)

        @time loss_value, trained_θ, misc = QuantumNaturalGradient.evolve(Oks_and_Eks, θ;
            integrator, verbosity=2, sample_nr=200, maxiter=50)

        # the optimizer must reach the exact eigenstate: once the state is the pure CDW all
        # samples give Ek = -24 exactly, so the loss is exact as well (no sampling tolerance)
        @test isapprox(loss_value, E_exact; atol=1e-10)

        Nmeasure = 500
        E2, E2_err, _ = QuantumNaturalfPEPS.weighted_mean_error(
            QuantumNaturalfPEPS.get_ExpectationValue(peps, Hubbard_ham; trial_state=trial_state, it=Nmeasure)...)
        # independent re-measurement of the evolved PEPS confirms the converged energy
        @test isapprox(E2, E_exact; atol=1/sqrt(Nmeasure))
    end
end;
