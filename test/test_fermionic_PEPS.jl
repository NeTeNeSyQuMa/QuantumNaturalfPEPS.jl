using Test
using ITensors, ITensorMPS
using QuantumNaturalfPEPS
using QuantumNaturalGradient
using LinearAlgebra
using Random

@testset "Fermionic PEPS functionalities" begin

    #=
        Fermionic PEPS with siteinds("Fermion", Lx, Ly).

        Convention: the PEPS stores the Jordan-Wigner image of the fermionic state in the
        column-major mode ordering n = i + (j-1)*Lx (same ordering as vec(S)). All fermionic
        signs therefore live in the Hamiltonian matrix elements: TensorOperatorSum inserts the
        JW string "F" between the fermionic operators (see QuantumNaturalGradient.insert_JW_string).
        The dense tensor permutes of the package consequently must NOT introduce any signs, while
        ITensors' permute/getindex on fermionic QN indices (auto_fermion) must apply the JW signs.

        Small example why we dont get signs for amplitude calculations <S|ψ>:

        Let |ψ> = (c₁†)ⁿ¹ (c₂†)ⁿ² ... |0> and |S> = (c₁†)ˢ¹ (c₂†)ˢ² ... |0> be the canonical basis state. Then

            <S|ψ> = <0| ... (c₂)ˢ² (c₁)ˢ¹ (c₁†)ⁿ¹ (c₂†)ⁿ² ... |0>

            Using the anticommutation relations {cᵢ, cⱼ†} = δᵢⱼ and {cᵢ, cⱼ} = 0 
            <S|ψ> ≠ 0 only if nᵢ = sᵢ for all i. 
            In that case, the annihilation operators in <S| exactly cancel the creation operators in |ψ> using {cᵢ, cⱼ†} = δᵢⱼ
            and we dont get any signs
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
                expected = permutedims(A, collect(perm)) # analytic reference: permutedims of the raw data array

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
        Purpose: the sampler fixes sites row by row — (1,1),(1,2),...,(2,1),... — i.e. in the JW
        mode order 1→3→2→4 on the 2x2 lattice, while amplitudes ψ(S) are defined in the global
        column-major mode order 1→2→3→4. This mismatch CANNOT introduce fermionic signs: the
        sampler only evaluates marginals p(s_A) = ⟨ψ| Π_{k∈A} P̂_k(s_k) |ψ⟩ of occupation
        projectors P̂_k(1) = n̂_k = c†_k c_k, P̂_k(0) = 1 - n̂_k, which are parity-even (0 or 2
        fermion operators each) and therefore commute pairwise — reordering the conditioning
        P̂_1 P̂_3 P̂_2 P̂_4 = P̂_1 P̂_2 P̂_3 P̂_4 costs (-1)^even = +1. The chain rule of the
        classical measure π(S) = |ψ(S)|²/Z holds for ANY fixing order, so with exact (untruncated)
        contractions the sampler's joint probability exp(logpc) must equal |ψ(S)|²/Z, with ψ(S)
        evaluated in the column-major convention. The test checks this identity deterministically
        for every drawn sample (no statistical tolerance).

        Note what this can and cannot catch: probabilities are sign-blind, so it cannot detect
        DROPPED fermionic signs (that is the job of the Ek-vs-ED testset below). Its role is the
        converse: it fails if anyone "fixes" the sampler by inserting JW signs into the sampling
        environments or conditionals, where none belong.
    =#
    @testset "Sequential sampling (row-major fixing) reproduces |ψ(S)|² (column-major ψ)" begin
        Random.seed!(11)
        for (Lx, Ly) in ((3, 2), (3, 3))
            hilbert = ITensors.siteinds("Fermion", Lx, Ly)
            # contraction dims chosen large enough that nothing is truncated -> exact conditionals
            peps = PEPS(hilbert; bond_dim=2, sample_dim=32, contract_dim=32, double_contract_dim=32)

            # exact amplitudes of all 2^N configurations; vec(S) is the column-major mode string
            N = Lx * Ly
            amps = Dict{Vector{Int}, Float64}()
            for idx in 0:(2^N - 1)
                occ = digits(idx, base=2, pad=N)
                S = reshape(occ, Lx, Ly)
                amps[occ] = QuantumNaturalfPEPS.contract_peps_exact(QuantumNaturalfPEPS.get_projected(peps, S))
            end
            Z = sum(abs2, values(amps))

            for _ in 1:20
                S, logpc, _ = QuantumNaturalfPEPS.get_sample(peps)
                @test isapprox(exp(logpc), amps[collect(vec(S))]^2 / Z; rtol=1e-8, atol=1e-12)
            end
        end
    end

    #=
        Purpose: the decisive fermionic-sign test — deterministic, no Monte Carlo. For EVERY basis
        configuration S of a generic (random, dense) fermionic PEPS on the 2x2 lattice the local
        energy computed by the package,
            Ek(S) = Σ_{S'} ⟨S|H|S'⟩ ψ(S')/ψ(S)   (get_precomp_sOψ_elems + get_logψ_flipped),
        must equal (Hψ)_S / ψ_S with the exact 16x16 Hamiltonian built from full-string
        Jordan-Wigner operators in the column-major mode convention (mode n = i + (j-1)Lx):
            c†_n = F_1 ⊗ ... ⊗ F_{n-1} ⊗ cdag_n ⊗ 1 ⊗ ...,   F = diag(1, -1).
        These matrices satisfy the CAR exactly, so all anticommutation signs of the reference are
        guaranteed by construction, independently of the package code under test.

        Sign discrimination: on the 2x2 lattice the horizontal bonds are the mode pairs (1,3) and
        (2,4); their JW strings cover exactly one site each — site (2,1) (mode 2) and site (1,2)
        (mode 3) — so e.g. ⟨1 s 0 ·|c†_1 c_3|0 s 1 ·⟩ = (-1)^s flips sign with the occupation of
        the string site. We also build the WRONG-SIGN Hamiltonian (strings dropped, i.e. hardcore
        bosons) and require the package's Ek to be distinguishable from it: a pipeline that
        silently ignores the JW strings would pass a "runs without error" bar but fails this
        rejection.
    =#
    @testset "Local energy Ek(S) vs exact JW diagonalization (2x2)" begin
        Random.seed!(1234)
        Lx, Ly = 2, 2
        N = Lx * Ly
        t, U = 1.0, 0.7

        hilbert = ITensors.siteinds("Fermion", Lx, Ly)
        peps = PEPS(hilbert; bond_dim=2, sample_dim=32, contract_dim=32, double_contract_dim=32)

        ham = QuantumNaturalfPEPS.hamiltonian_hubbard(t, U, Lx, Ly)
        ham_op = QuantumNaturalGradient.TensorOperatorSum(ham, hilbert)

        # full-string JW operators; mode 1 is the FASTEST index (Julia kron: last factor = fastest),
        # matching the basis enumeration idx = 1 + Σ_n s_n 2^(n-1) below
        Id2 = [1.0 0.0; 0.0 1.0]
        Fm = [1.0 0.0; 0.0 -1.0]
        cdag_loc = [0.0 0.0; 1.0 0.0] # |0⟩ -> |1⟩
        n_loc = [0.0 0.0; 0.0 1.0]
        site_op(op, n; strings=true) = reduce(kron, reverse([k == n ? op : (k < n && strings ? Fm : Id2) for k in 1:N]))

        function H_hubbard_matrix(; strings=true)
            Cd = [site_op(cdag_loc, n; strings) for n in 1:N]
            C = [Matrix(transpose(Cdn)) for Cdn in Cd]
            n_ops = [site_op(n_loc, n; strings=false) for n in 1:N]
            # bonds in column-major modes: vertical (adjacent modes, no string), horizontal (string)
            bonds = [(1, 2), (3, 4), (1, 3), (2, 4)]
            H = zeros(2^N, 2^N)
            for (a, b) in bonds
                H .+= -t .* (Cd[a] * C[b] .+ Cd[b] * C[a])
                H .+= U .* (n_ops[a] * n_ops[b]) .- (U / 2) .* n_ops[a] .- (U / 2) .* n_ops[b]
            end
            return H
        end
        H_jw = H_hubbard_matrix(strings=true)
        H_ns = H_hubbard_matrix(strings=false)

        # exact PEPS amplitudes in the column-major basis: vec(S) = (s_1, ..., s_4), s_1 fastest
        ψvec = [QuantumNaturalfPEPS.contract_peps_exact(
                    QuantumNaturalfPEPS.get_projected(peps, reshape(digits(idx - 1, base=2, pad=N), Lx, Ly)))
                for idx in 1:2^N] # amplitudes of the PEPS
        @test minimum(abs, ψvec) > 1e-6 # generic state: every configuration contributes

        Hψ = H_jw * ψvec
        Hnsψ = H_ns * ψvec
        Ek_pkg = zeros(ComplexF64, 2^N)
        for idx in 1:2^N
            S = reshape(digits(idx - 1, base=2, pad=N), Lx, Ly)
            logψ, env_top, env_down, _ = QuantumNaturalfPEPS.get_logψ_and_envs(peps, S)
            Ek_pkg[idx] = QuantumNaturalfPEPS.get_Ek(peps, ham_op, env_top, env_down, S, logψ)
        end

        # the package's local energy must reproduce the fermionic reference for every S ...
        for idx in 1:2^N
            @test isapprox(Ek_pkg[idx], Hψ[idx] / ψvec[idx]; atol=1e-8, rtol=1e-8)
        end
        # ... and must NOT be explainable by the string-less (hardcore-boson) Hamiltonian
        Ek_jw = Hψ ./ ψvec
        Ek_ns = Hnsψ ./ ψvec
        @test maximum(abs.(Ek_jw .- Ek_ns)) > 0.5
        @test any(idx -> !isapprox(Ek_pkg[idx], Ek_ns[idx]; atol=1e-3), 1:2^N)

        #=
            Trial-state convention: with trial_state = GS the variational amplitudes are
            Ψ(S) = ψ_PEPS(S) · φ_GS(S), and every flipped term in Ek picks up the SIGNED Gaussian
            amplitude ratio φ(S')/φ(S) (get_amplitude, Pfaffian formula). φ carries genuine
            fermionic reordering signs, and its mode ordering must be the same column-major
            convention as the PEPS and the JW Hamiltonian — nothing else in the test suite pins
            this alignment. The reference is again exact: Ek(S) = (H_JW χ)_S / χ_S with χ = ψ .* φ.
            (Hopping conserves particle number, so all flips stay inside the even-parity sector
            where φ is supported.)
        =#
        @testset "GaussianState trial state: joint local energy vs ED" begin
            η = [0.3, 0.45, 0.35, 0.55, 0.7, 0.6, 0.65, 0.75, 0.5, 0.8, 0.55, 0.4]
            GS = QuantumNaturalfPEPS.GaussianState(QuantumNaturalfPEPS.build_general_H_BdG_2D_NN, N; η=η, parity_sector=0)

            φvec = [QuantumNaturalfPEPS.get_amplitude(GS, digits(idx - 1, base=2, pad=N)) for idx in 1:2^N] # amplitude of the Gaussian state
            even = [idx for idx in 1:2^N if iseven(sum(digits(idx - 1, base=2, pad=N)))]
            odd = setdiff(1:2^N, even)
            @test all(abs.(φvec[odd]) .< 1e-12)   # parity selection rule
            @test minimum(abs, φvec[even]) > 1e-3 # generic inside the even sector

            χ = ψvec .* φvec # joint amplitude of the PEPS and the Gaussian trial state
            Hχ = H_jw * χ
            for idx in even
                S = reshape(digits(idx - 1, base=2, pad=N), Lx, Ly)
                logψ, env_top, env_down, _ = QuantumNaturalfPEPS.get_logψ_and_envs(peps, S)
                # combined amplitude of the sample, exactly as Ok_and_Ek does before calling get_Ek
                logψ += log(QuantumNaturalfPEPS.get_amplitude(GS, collect(vec(S))))
                Ek = QuantumNaturalfPEPS.get_Ek(peps, ham_op, env_top, env_down, S, logψ; trial_state=GS)
                @test isapprox(Ek, Hχ[idx] / χ[idx]; atol=1e-6, rtol=1e-6)
            end
        end
    end

    #=
        Purpose: convention guards.
        (1) The dense-only design is enforced by the type system: PEPS cannot be constructed from
            QN-conserving (block-sparse, parity-graded) site indices. This is deliberate — the
            package's sign convention (signs live in the tensor entries + JW strings in the
            Hamiltonian) would double-count signs if combined with ITensors' auto_fermion
            machinery, and all raw-storage helpers assume dense layout.
        (2) The operator side of the convention at unit level: TensorOperatorSum must insert the
            JW string for fermionic bilinears. The matrix element of c†_(1,1) c_(1,2) (modes 1 and
            3) flips sign with the occupation s of the string site (2,1) (mode 2):
                ⟨1 s 0 ·| c†_1 c_3 |0 s 1 ·⟩ = (-1)^s.
            Without insert_JW_string the element would be +1 for both s.
    =#
    @testset "Dense-only guard and JW string insertion" begin
        hilbertQN = [ITensors.siteind("Fermion"; conserve_qns=true, addtags="nx=$i,ny=$j") for i in 1:2, j in 1:2]
        @test_throws MethodError PEPS(hilbertQN; bond_dim=2)

        hilbert = ITensors.siteinds("Fermion", 2, 2)
        hop = OpSum()
        hop += (1.0, "Cdag", (1, 1), "C", (1, 2))
        hop_op = QuantumNaturalGradient.TensorOperatorSum(hop, hilbert)

        for (s2, expected) in ((0, 1.0), (1, -1.0))
            S = [1 0; s2 0] # sample: mode 1 = site (1,1) occupied; string site (2,1) carries s2
            terms = QuantumNaturalGradient.get_precomp_sOψ_elems(hop_op, S; get_flip_sites=true)
            flips = [k for k in keys(terms) if k != ()]
            @test length(flips) == 1
            @test terms[only(flips)] ≈ expected
        end
    end

    #=
        Purpose: end-to-end validation of the full fermionic pipeline — sampling, Ok/Ek evaluation
        and QuantumNaturalGradient.evolve — for a PEPS with siteinds("Fermion", Lx, Ly)
        (trial_state = IdentityState, We use the t → 0 limit of the
        (spinless) Hubbard model on the 4x4 open lattice where the ground state is a CDW.
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
            set_occupation_peps!(peps, is_cdw_occupied; ε=0.2)
            θ = vec(QuantumNaturalGradient.Parameters(peps).obj)

            integrator = QuantumNaturalGradient.Euler(lr=0.1)
            Oks_and_Eks = QuantumNaturalfPEPS.generate_Oks_and_Eks(peps, Hubbard_ham; trial_state=trial_state)

            @time loss_value, trained_θ, misc = QuantumNaturalGradient.evolve(Oks_and_Eks, θ;
                integrator, verbosity=0, sample_nr=200, maxiter=50)

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
end;