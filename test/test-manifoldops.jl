using DDF

using DifferentialForms: bitsign
using LinearAlgebra
using StaticArrays
using Test

@testset "Manifold ops D=$D P=$P R=$R" for D in 0:Dmax, P in (Pr, Dl), R in 0:D
    S = Float64

    mfds = [empty_manifold(Val(D), S), simplex_manifold(Val(D), S),
            hypercube_manifold(Val(D), S),
            delaunay_hypercube_manifold(Val(D), S),
            large_delaunay_hypercube_manifold(Val(D), S),
            refined_simplex_manifold(Val(D), S)]

    for mfd in mfds
        funs = Fun{D,P,R,D,S}[]
        push!(funs, zero(Fun{D,P,R,D,S,S}, mfd))
        e = id(Fun{D,P,0,D,S,SVector{D,S}}, mfd)
        if R == 0
            push!(funs, e)
        end

        s = P == Pr ? +1 : -1

        # Boundary
        R2 = R - 2s
        if 0 <= R2 <= D
            b1 = boundary(Val(P), Val(R), mfd)
            b2 = boundary(Val(P), Val(R - s), mfd)
            b21 = b2 * b1
            b21::Op{D,P,R2,P,R,Int8}
            @test iszero(b21)
        end

        R1 = R - s
        if 0 <= R1 <= D
            b = boundary(Val(P), Val(R), mfd)
            for f in funs
                bf = b * f
                T = eltype(f)
                bf::Fun{D,P,R1,D,S,T}
                bf′ = boundary(f)
                @test bf′ == bf
            end
        end

        # Derivative
        R2 = R + 2s
        if 0 <= R2 <= D
            d1 = deriv(Val(P), Val(R), mfd)
            d2 = deriv(Val(P), Val(R + s), mfd)
            d21 = d2 * d1
            d21::Op{D,P,R2,P,R,Int8}
            @test iszero(d21)
        end

        R1 = R + s
        if 0 <= R1 <= D
            d = deriv(Val(P), Val(R), mfd)
            for f in funs
                df = d * f
                T = eltype(f)
                df::Fun{D,P,R1,D,S,T}
                df′ = deriv(f)
                @test df′ == df
            end
        end

        # Only if completely well-centred [arXiv:0802.2108 [cs.CG]]
        if Manifolds.dualkind == BarycentricDuals ||
           mfd.name ∉ ["hypercube manifold", "delaunay hypercube manifold",
            "large delaunay hypercube manifold", "refined simplex manifold"]

            # Hodge
            h = hodge(Val(P), Val(R), mfd)
            h′ = hodge(Val(!P), Val(R), mfd)
            @test isempty(h) || all(x -> x != 0 && isfinite(x), h.values.diag)
            @test isempty(h′) || all(x -> x != 0 && isfinite(x), h′.values.diag)
            @test invhodge(Val(P), Val(R), mfd) == h′
            @test invhodge(Val(!P), Val(R), mfd) == h
            h21 = h′ * h
            h21::Op{D,P,R,P,R,S}
            @test isempty(h21) ||
                  all(x -> x != 0 && isfinite(x), h21.values.diag)
            if !isempty(h21)
                @test norm((h21 - one(h21)).values, Inf) <= 10eps(S)
            end

            for f in funs
                hf = h * f
                T = eltype(f)
                hf::Fun{D,!P,R,D,S,T}
                hf′ = ⋆f
                @test hf′ == hf
            end

            # Coderivative
            R2 = R - 2s
            if 0 <= R2 <= D
                δ1 = coderiv(Val(P), Val(R), mfd)
                δ2 = coderiv(Val(P), Val(R - s), mfd)
                δ21 = δ2 * δ1
                δ21::Op{D,P,R2,P,R,S}
                nδ1 = norm(δ1, Inf)
                nδ2 = norm(δ2, Inf)
                maxerr = nδ1 * nδ2 * 10eps(S)
                @test norm(δ21, Inf) <= maxerr
            end

            R1 = R - s
            if 0 <= R1 <= D
                δ = coderiv(Val(P), Val(R), mfd)
                @test isempty(δ) ||
                      all(x -> x != 0 && isfinite(x), nonzeros(δ.values))
                for f in funs
                    δf = δ * f
                    T = eltype(f)
                    δf::Fun{D,P,R1,D,S,T}
                    δf′ = coderiv(f)
                    @test δf′ == δf
                end
            end

            # Laplace
            Δ = laplace(Val(P), Val(R), mfd)
            @test isempty(Δ) ||
                  all(x -> x != 0 && isfinite(x), nonzeros(Δ.values))
            for f in funs
                Δf = Δ * f
                T = eltype(f)
                Δf::Fun{D,P,R,D,S,T}
                Δf′ = laplace(f)
                @test Δf′ == Δf
            end

        end

    end
end
